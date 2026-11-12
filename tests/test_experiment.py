import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from rl.experiment import (
    EvalRecord, environment_config, save_replay_metadata, validate_args,
    validate_model_config, validate_replay,
)
from rl.train_vae_sac import CappedDynamicGradientStepsCallback, ConfigCheckpointCallback, DonkeyVaeSACEnv
from rl.eval_paired_randomized import run_episode


def arguments(tmp_path):
    vae = tmp_path / "vae.pt"
    vae.write_bytes(b"encoder")
    return argparse.Namespace(
        env_id="test", host="127.0.0.1", port=9091, encoder="vae", vae_model=vae, encoder_crop_top=0,
        n_command_history=20, min_throttle=.2, max_throttle=.7,
        max_steering=1., max_steering_diff=.2, max_cte_error=2., cte_target=0.,
        cte_speed_penalty_weight=.25, encoder_projection=None, latent_lags=(),
        output_dir=tmp_path / "eval", save_telemetry=True, episodes=1,
    )


def test_configuration_catches_same_shape_wrong_encoder_or_action(tmp_path):
    args = arguments(tmp_path)
    config = environment_config(args)
    model = SimpleNamespace(donkey_config=config)
    args.vae_model.write_bytes(b"different encoder, same dimension")
    with pytest.raises(ValueError, match="observation mismatch"):
        validate_model_config(model, environment_config(args))
    changed = dict(config, action=dict(config["action"], max_throttle=.8))
    with pytest.raises(ValueError, match="action mismatch"):
        validate_model_config(model, changed)


def test_replay_rejects_changed_rewards_and_mismatched_file(tmp_path):
    config = environment_config(arguments(tmp_path))
    path = tmp_path / "buffer.pkl"
    path.write_bytes(b"stored rewards")
    with pytest.raises(ValueError, match="metadata missing"):
        validate_replay(path, config)
    with pytest.warns(UserWarning, match="legacy replay"):
        validate_replay(path, config, allow_legacy=True)
    save_replay_metadata(path, SimpleNamespace(donkey_config=config, num_timesteps=12))
    validate_replay(path, config)
    changed = dict(config, reward=dict(config["reward"], alive_reward=2))
    with pytest.raises(ValueError, match="rewards differ"):
        validate_replay(path, changed)
    path.write_bytes(b"some other buffer")
    with pytest.raises(ValueError, match="hash differs"):
        validate_replay(path, config)


def test_records_partial_results_on_failure_and_writes_strict_json(tmp_path):
    args = arguments(tmp_path)
    with pytest.raises(RuntimeError, match="reload"):
        with EvalRecord(args, {"vae": args.vae_model}) as record:
            record.step({"episode": 1}, 1, np.array([0., 1.]), 1., False, True, {"cte": 0.})
            record.episode({"steps": 1, "period": float("inf")})
            raise RuntimeError("reload failed")
    meta = json.loads((args.output_dir / "metadata.json").read_text())
    assert meta["status"] == "failed"
    result = json.loads((args.output_dir / "results.json").read_text())
    assert result["episodes"][0]["period"] is None
    assert json.loads((args.output_dir / "telemetry.jsonl").read_text())["truncated"] is True
    assert (args.output_dir / "episodes.csv").is_file()
    with pytest.raises(FileExistsError):
        EvalRecord(args, {"vae": args.vae_model})


class TinyEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    observation_space = gym.spaces.Box(-1, 1, (3,), np.float32)

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get("seed"))
        self.steps = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(3, np.float32), 1., False, self.steps == 4, {
            "lap_count": self.steps // 2, "last_lap_time": 9., "cte": 0., "speed": 1.,
        }


def test_equal_lap_times_are_both_counted():
    model = SimpleNamespace(predict=lambda obs, **kwargs: (np.zeros(2), None))
    result = run_episode(TinyEnv(), model, 4)
    assert result["lap_times"] == [9., 9.]
    assert result["trunc"]


def test_real_sb3_save_preserves_contract_and_replay_metadata(tmp_path):
    config = environment_config(arguments(tmp_path))
    model = SAC("MlpPolicy", TinyEnv(), buffer_size=16, learning_starts=100,
                policy_kwargs={"net_arch": [8]}, device="cpu")
    model.donkey_config = config
    callback = ConfigCheckpointCallback(save_freq=4, save_path=str(tmp_path), save_replay_buffer=True)
    model.learn(4, callback=callback)
    saved = SAC.load(tmp_path / "rl_model_4_steps.zip", device="cpu")
    assert saved.donkey_config == config
    validate_replay(tmp_path / "rl_model_replay_buffer_4_steps.pkl", config)


def test_dynamic_updates_do_not_override_step_training():
    callback = CappedDynamicGradientStepsCallback(cap=1000, floor=50)
    callback.model = SimpleNamespace(
        train_freq=SimpleNamespace(unit=SimpleNamespace(value="step")), gradient_steps=7,
    )
    callback._on_rollout_end()
    assert callback.model.gradient_steps == 7


def test_invalid_arguments_rejected_before_connecting(tmp_path):
    args = arguments(tmp_path)
    args.episodes = 0
    with pytest.raises(ValueError, match="episodes"):
        validate_args(args)
    args.episodes = 1
    args.min_throttle = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_args(args)
    args.min_throttle = .2
    args.gradient_steps_cap = 1000
    args.gradient_steps_min = 50
    args.gradient_steps_scale = 1.
    args.gradient_steps = 7
    args.train_freq_unit = "episode"
    with pytest.raises(ValueError, match="would override"):
        validate_args(args)


def test_wrapper_rejects_missing_cte_and_failed_reload(monkeypatch):
    encoder = SimpleNamespace(z_size=3, encode=lambda obs: np.zeros(3, np.float32))
    base = TinyEnv()
    base.viewer = SimpleNamespace(handler=SimpleNamespace(SceneToLoad="test"))
    env = DonkeyVaeSACEnv(base, vae=encoder, scene_reload_every=1)
    monkeypatch.setattr("rl.train_vae_sac.reload_scene", lambda *args: False)
    with pytest.raises(RuntimeError, match="layout not counted"):
        env.reset()
    env.scene_reload_every = 0
    env.reset()
    base.step = lambda action: (np.zeros(3), 0., False, False, {})
    with pytest.raises(ValueError, match="CTE"):
        env.step(np.zeros(2))


def test_vae_resume_does_not_replace_historical_best(tmp_path, monkeypatch):
    from rl import train_vae
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": 2, "metrics": [{"epoch": 1, "val": {"loss": 1.}}, {"epoch": 2, "val": {"loss": 2.}}]}
    resume = tmp_path / "last.pt"
    torch.save(checkpoint, resume)
    best = tmp_path / "best.pt"
    checkpoint["metrics"] = [{"epoch": 1, "val": {"loss": .5}}]
    torch.save(checkpoint, best)
    previous = best.read_bytes()
    args = SimpleNamespace(output_dir=tmp_path, cache_dir=tmp_path, seed=0, device="cpu",
                           max_train_samples=None, max_val_samples=None, batch_size=2, num_workers=0,
                           z_size=2, learning_rate=.001, resume=resume, epochs=3, kl_tolerance=.5,
                           beta=1., log_interval=0, save_samples=1)
    monkeypatch.setattr(train_vae, "parse_args", lambda: args)
    monkeypatch.setattr(train_vae, "VaeMemmapDataset", lambda *args: torch.zeros(4, 2))
    monkeypatch.setattr(train_vae, "ConvVAE", lambda **kwargs: torch.nn.Linear(2, 2))
    monkeypatch.setattr(train_vae, "run_epoch", lambda *args, **kwargs: {"loss": 3., "r_loss": 2., "kl_loss": 1.})
    monkeypatch.setattr(train_vae, "save_reconstructions", lambda *args: None)
    train_vae.main()
    assert best.read_bytes() == previous


def test_paired_reload_failure_is_recorded_and_closes_env(tmp_path, monkeypatch):
    from rl import eval_paired_randomized as paired
    args = arguments(tmp_path)
    args.max_episode_steps = 4
    args.order_seed = 42
    env = TinyEnv()
    env.viewer = SimpleNamespace(handler=SimpleNamespace(SceneToLoad="test"))
    env.close = Mock()
    monkeypatch.setattr(paired, "make_encoder", lambda *args, **kwargs: SimpleNamespace(z_size=3, model=torch.nn.Linear(3, 3)))
    monkeypatch.setattr(paired.gym, "make", lambda *args, **kwargs: env)
    model = SimpleNamespace(donkey_config=environment_config(args))
    monkeypatch.setattr(paired.SAC, "load", lambda *args, **kwargs: model)
    monkeypatch.setattr(paired, "reload_scene", lambda *args: False)
    episode = Mock(side_effect=AssertionError("Failed reload must not run an episode"))
    monkeypatch.setattr(paired, "run_episode", episode)
    with pytest.raises(RuntimeError, match="layout not counted"):
        with EvalRecord(args, {"policy": args.vae_model}) as record:
            args.layouts = 1
            paired._evaluate(args, [("test", args.vae_model, 0)], record)
    episode.assert_not_called()
    env.close.assert_called_once()
    assert json.loads((args.output_dir / "metadata.json").read_text())["status"] == "failed"


def resumed_training(tmp_path, monkeypatch, flags):
    import sys
    from rl import train_loop_vae_sac as training
    monkeypatch.setattr(sys, "argv", ["train", *flags])
    args = training.parse_args()
    args.output_dir = tmp_path
    args.vae_model = tmp_path / "vae.pt"
    args.vae_model.write_bytes(b"encoder")
    args.resume_model = tmp_path / "resume.zip"
    config = environment_config(args)
    env = TinyEnv()
    model = SAC("MlpPolicy", env, batch_size=4, buffer_size=16, device="cpu",
                policy_kwargs={"net_arch": {"pi": [8, 8], "qf": [8, 8]}})
    model.donkey_config = config
    monkeypatch.setattr(training.SAC, "load", lambda *args, **kwargs: model)
    return training, args, config, env, model


def test_resume_rejects_explicit_ignored_batch_size(tmp_path, monkeypatch):
    training, args, config, env, model = resumed_training(tmp_path, monkeypatch, ["--batch-size", "8"])
    with pytest.raises(ValueError, match="batch-size is inherited as 4"):
        training._train(args, env, torch.device("cpu"), config)
    assert not (tmp_path / "final_model.zip").exists()


def test_learning_rate_override_survives_save_and_reload(tmp_path, monkeypatch):
    training, args, config, env, model = resumed_training(
        tmp_path, monkeypatch, ["--override-learning-rate", "--learning-rate", "0.0002"])
    monkeypatch.setattr(SAC, "learn", lambda self, **kwargs: self)
    training._train(args, env, torch.device("cpu"), config)
    # Use the original loader, since resumed_training patches SAC.load.
    from stable_baselines3.common.base_class import BaseAlgorithm
    reloaded = BaseAlgorithm.load.__func__(SAC, tmp_path / "final_model.zip", device="cpu")
    assert reloaded.learning_rate == .0002
    assert reloaded.lr_schedule(1.) == .0002


def test_fresh_step_training_keeps_updates_and_saves_effective_config(tmp_path, monkeypatch):
    import sys
    from rl import train_loop_vae_sac as training
    monkeypatch.setattr(sys, "argv", ["train", "--timesteps", "8", "--train-freq-unit", "step",
        "--gradient-steps", "7", "--save-final-replay-buffer", "--save-replay-buffer",
        "--checkpoint-freq", "4", "--hidden-size", "8", "--buffer-size", "16",
        "--learning-starts", "100", "--batch-size", "4"])
    args = training.parse_args()
    args.output_dir = tmp_path
    args.vae_model = tmp_path / "vae.pt"
    args.vae_model.write_bytes(b"encoder")
    config = environment_config(args)
    training._train(args, TinyEnv(), torch.device("cpu"), config)
    saved = SAC.load(tmp_path / "final_model.zip", device="cpu")
    assert saved.num_timesteps == 8 and saved.gradient_steps == 7
    assert saved.donkey_config == config
    effective = json.loads((tmp_path / "run_config.json").read_text())["effective_algorithm"]
    assert effective["batch_size"] == 4
    assert effective["dynamic_gradient_steps"]["enabled"] is False
    validate_replay(tmp_path / "final_replay_buffer.pkl", config)


def test_wrapper_preserves_action_mapping_and_history():
    encoder = SimpleNamespace(z_size=3, encode=lambda obs: np.zeros(3, np.float32))
    env = DonkeyVaeSACEnv(TinyEnv(), vae=encoder, min_throttle=.2, max_throttle=.7)
    obs, _ = env.reset()
    assert obs.shape == (43,)
    obs, reward, terminated, truncated, info = env.step(np.array([1., 1.]))
    assert info["rl_steer"] == pytest.approx(.39998)
    assert info["rl_throttle"] == pytest.approx(.7)
    assert obs[-2:] == pytest.approx([.39998, .7])
    assert reward == 1. and not terminated and not truncated


def test_steering_rate_limit_does_not_depend_on_history_length():
    encoder = SimpleNamespace(z_size=3, encode=lambda obs: np.zeros(3, np.float32))
    env = DonkeyVaeSACEnv(TinyEnv(), vae=encoder, n_command_history=0)
    obs, _ = env.reset()
    assert obs.shape == (3,)
    steers = [env.step(np.array([1., 0.]))[4]["rl_steer"] for _ in range(2)]
    assert steers == pytest.approx([.39998, .79996])
    env.reset()
    assert env.step(np.array([-1., 0.]))[4]["rl_steer"] == pytest.approx(-.39998)


def test_paired_and_loop_evaluations_use_the_requested_rewards(monkeypatch):
    import sys
    from rl import eval_paired_randomized as paired
    from rl.train_vae_sac import reward_config_from_args
    monkeypatch.setattr(sys, "argv", ["eval", "--steps", "1", "--alive-reward", "1.7", "--cte-target", "3.5"])
    config = reward_config_from_args(paired.parse_args())
    assert (config.alive_reward, config.speed_reward_weight, config.cte_target) == (1.7, .15, 3.5)
    assert reward_config_from_args(SimpleNamespace(max_cte_error=2.5)).max_cte_error == 2.5


class EpisodeEnv(TinyEnv):
    """Episodes of `length` steps, so a checkpoint can fall mid-episode."""
    length = 6

    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        return obs, reward, terminated, self.steps == self.length, info


def test_replay_buffer_is_saved_at_episode_end_not_mid_episode(tmp_path):
    config = environment_config(arguments(tmp_path))
    model = SAC("MlpPolicy", EpisodeEnv(), buffer_size=64, learning_starts=100,
                policy_kwargs={"net_arch": [8]}, device="cpu")
    model.donkey_config = config
    seen = []

    class Watch(ConfigCheckpointCallback):
        def _save_pending_replay(self):
            if self._pending_replay_path is not None:
                seen.append((self.num_timesteps, self._pending_replay_path.name))
            super()._save_pending_replay()

    model.learn(9, callback=Watch(save_freq=4, save_path=str(tmp_path), save_replay_buffer=True))
    # Checkpoint at step 4 waits for the episode end at 6; the one at 8 is
    # flushed when training stops at 9, before its episode ends.
    assert seen == [(6, "rl_model_replay_buffer_4_steps.pkl"), (9, "rl_model_replay_buffer_8_steps.pkl")]
    assert (tmp_path / "rl_model_4_steps.zip").is_file()
    validate_replay(tmp_path / "rl_model_replay_buffer_4_steps.pkl", config)


def test_resume_without_buffer_waits_for_data_and_keeps_configuration(tmp_path, monkeypatch):
    training, args, config, env, _ = resumed_training(
        tmp_path, monkeypatch, ["--timesteps", "24", "--resume-warmup-steps", "10",
                                "--gradient-steps-min", "5", "--checkpoint-freq", "4"])
    # Episode training with dynamic updates, as in the real loop recipes.
    model = SAC("MlpPolicy", env, batch_size=4, buffer_size=16, device="cpu", train_freq=(1, "episode"),
                gradient_steps=-1, policy_kwargs={"net_arch": {"pi": [8, 8], "qf": [8, 8]}})
    model.donkey_config = config
    model.num_timesteps = 1000  # far past learning_starts, as on a real resume
    monkeypatch.setattr(training.SAC, "load", lambda *args, **kwargs: model)
    updates = []
    monkeypatch.setattr(SAC, "train", lambda self, gradient_steps, batch_size:
                        updates.append((self.replay_buffer.size(), gradient_steps)))
    (tmp_path / "run_config.json").write_text("{}")
    training._train(args, env, torch.device("cpu"), config)
    # TinyEnv episodes are 4 steps: nothing may be updated on 4 or 8
    # transitions; afterwards the dynamic floor of 5 applies again.
    assert updates == [(12, 5), (16, 5), (16, 5), (16, 5)]
    from stable_baselines3.common.base_class import BaseAlgorithm
    load = BaseAlgorithm.load.__func__
    assert load(SAC, tmp_path / "final_model.zip", device="cpu").gradient_steps == 5
    # Saved during the warm-up: must not carry the temporary zero.
    assert load(SAC, tmp_path / "sac_loop_vae_1008_steps.zip", device="cpu").gradient_steps != 0
    assert (tmp_path / "run_config.json").read_text() == "{}"
    assert (tmp_path / "run_config_from_1000_steps.json").is_file()


def test_paired_statistics_report_intervals_and_exact_paired_test():
    from rl.experiment import paired_statistics, wilson_interval

    low, high = wilson_interval(8, 12)
    assert (round(low, 3), round(high, 3)) == (0.391, 0.862)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == pytest.approx(1.0)

    # a completes all 8 layouts; b fails 6 of them and never wins one.
    episodes = [{"model": "a", "layout": i, "trunc": True, "steps": 2000, "laps": 7} for i in range(8)]
    episodes += [{"model": "b", "layout": i, "trunc": i < 2, "steps": 2000 if i < 2 else 500,
                  "laps": 7 if i < 2 else 1} for i in range(8)]
    episodes.append({"model": "a", "layout": 99, "trunc": False, "steps": 100, "laps": 0})  # b never ran it
    stats = paired_statistics(episodes)
    assert stats["models"]["a"]["completions"] == 8 and stats["models"]["a"]["layouts"] == 9
    assert stats["models"]["b"]["completion_rate"] == 0.25
    pair, = stats["pairs"]
    assert (pair["layouts"], pair["only_a_completed"], pair["only_b_completed"]) == (8, 6, 0)
    assert pair["sign_test_p"] == 2 / 2 ** 6
    assert pair["mean_steps_difference"] == 1125.0
    assert pair["mean_laps_difference"] == 4.5 and pair["mean_laps_difference_ci95"][0] > 0
    assert stats["models"]["b"]["mean_laps"] == 2.5
    assert stats["models"]["b"]["mean_laps_ci95"][0] < 2.5 < stats["models"]["b"]["mean_laps_ci95"][1]
    low, high = pair["mean_steps_difference_ci95"]
    assert 0 < low < 1125 < high <= 1500

    same = paired_statistics([dict(e, model=m) for e in episodes[:8] for m in ("x", "y")])
    assert same["pairs"][0]["sign_test_p"] == 1.0


def test_observation_carries_earlier_latents_in_order_and_refills_on_reset():
    frames = iter(range(1, 100))
    encoder = SimpleNamespace(z_size=1, encode=lambda obs: np.array([next(frames)], np.float32))
    env = DonkeyVaeSACEnv(TinyEnv(), vae=encoder, n_command_history=1, latent_lags=(1, 3))
    assert env.observation_space.shape == (5,)  # latent now, 1 step ago, 3 steps ago, one command
    obs, _ = env.reset()
    assert obs[:3].tolist() == [1, 1, 1]  # no earlier frame yet: the first one stands in
    seen = [env.step(np.array([0., 0.]))[0][:3].tolist() for _ in range(4)]
    assert seen == [[2, 1, 1], [3, 2, 1], [4, 3, 1], [5, 4, 2]]
    obs, _ = env.reset()
    assert obs[:3].tolist() == [6, 6, 6]  # nothing leaks across episodes

    plain = DonkeyVaeSACEnv(TinyEnv(), vae=encoder, n_command_history=1)
    assert plain.reset()[0].shape == (3,)


def test_new_observation_options_are_part_of_the_checkpoint_contract(tmp_path):
    args = arguments(tmp_path)
    plain = environment_config(args)
    assert "latent_lags" not in plain["observation"] and "projection_sha256" not in plain["observation"]

    args.latent_lags = (2, 4)
    lagged = environment_config(args)
    with pytest.raises(ValueError, match="observation mismatch"):
        validate_model_config(SimpleNamespace(donkey_config=plain), lagged)

    args.encoder, args.encoder_projection = "dinov2_vits14_grid", tmp_path / "projection.pt"
    with pytest.raises(ValueError, match="encoder-projection"):
        validate_args(argparse.Namespace(**dict(vars(args), encoder_projection=None)))
    args.encoder_projection.write_bytes(b"one projection")
    saved = environment_config(args)
    args.encoder_projection.write_bytes(b"another projection, same shape")
    with pytest.raises(ValueError, match="observation mismatch"):
        validate_model_config(SimpleNamespace(donkey_config=saved), environment_config(args))


def test_paired_evaluation_gives_each_model_its_own_observation(tmp_path, monkeypatch):
    from rl import eval_paired_randomized as paired
    args = arguments(tmp_path)
    args.max_episode_steps, args.order_seed, args.layouts, args.encoder = 4, 42, 1, "dinov2_vits14"
    env = TinyEnv()
    env.viewer = SimpleNamespace(handler=SimpleNamespace(SceneToLoad="test"))
    made = []

    def make_encoder(name, **kwargs):
        made.append((name, kwargs["crop_top"]))
        return SimpleNamespace(z_size=3, model=torch.nn.Linear(3, 3), encode=lambda obs: np.zeros(3, np.float32))

    lagged_args = argparse.Namespace(**dict(vars(args), latent_lags=(2, 4)))
    models = {"old.zip": SimpleNamespace(), "new.zip": SimpleNamespace(donkey_config=environment_config(lagged_args))}
    shapes = {}

    def episode(env, model, max_steps, record, context):
        shapes[context["model"]] = env.reset()[0].shape
        return {"trunc": True, "steps": 4, "mean_cte": 0., "max_cte": 0., "mean_speed": 1., "laps": 1,
                "best_lap": 1., "lap_times": [1.], "steer_delta": 0., "cte_std": 0., "steer_period": 1., "cte_period": 1.}

    monkeypatch.setattr(paired, "make_encoder", make_encoder)
    monkeypatch.setattr(paired.gym, "make", lambda *a, **k: env)
    monkeypatch.setattr(paired.SAC, "load", lambda path, **k: models[path])
    monkeypatch.setattr(paired, "reload_scene", lambda *a: True)
    monkeypatch.setattr(paired, "run_episode", episode)
    with pytest.warns(UserWarning, match="Legacy checkpoint"):
        with EvalRecord(args, {"policy": args.vae_model}) as record:
            paired._evaluate(args, [("old", "old.zip", 0), ("new", "new.zip", 0)], record)
    assert shapes == {"old": (3 + 40,), "new": (9 + 40,)}
    assert made == [("dinov2_vits14", 0)]  # same encoder and crop: built once, shared
