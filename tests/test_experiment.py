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
        cte_speed_penalty_weight=.25,
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
