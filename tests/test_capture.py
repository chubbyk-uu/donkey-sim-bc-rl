import csv
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from rl.experiment import EvalRecord
from rl.train_vae_sac import DonkeyVaeSACEnv
from test_experiment import TinyEnv


def record_args(tmp_path):
    return SimpleNamespace(output_dir=tmp_path / "run", save_telemetry=False, capture_tail_steps=3)


def test_capture_is_bounded_copies_images_and_separates_episodes(tmp_path):
    args = record_args(tmp_path)
    with EvalRecord(args, {}) as record:
        for episode, count in [(1, 5), (2, 2)]:
            image = np.zeros((4, 5, 3), np.uint8)
            for step in range(1, count + 1):
                image.fill(step * 30)
                record.step({"episode": episode}, step, [0., 0.], 1., episode == 1 and step == count,
                            episode == 2 and step == count, {"cte": 0.}, image=image)
            image.fill(255)
            assert len(record.capture) <= 3
            record.episode({"episode": episode, "steps": count})
        record.finish({"episodes": 2})
    data = json.loads((args.output_dir / "results.json").read_text())
    first, second = [row["capture"] for row in data["episodes"]]
    assert (first["first_step"], first["last_step"], first["frames"]) == (3, 5, 3)
    assert (second["first_step"], second["last_step"], second["frames"]) == (1, 2, 2)
    assert first["directory"] != second["directory"]
    pixels = np.array(Image.open(args.output_dir / first["directory"] / "step_000003.png"))
    assert np.all(pixels == 90)
    with Image.open(args.output_dir / first["animation"]) as animation:
        assert animation.n_frames == 3
    with (args.output_dir / "episodes.csv").open() as stream:
        assert json.loads(next(csv.DictReader(stream))["capture"]) == first
    assert len((args.output_dir / "telemetry.jsonl").read_text().splitlines()) == 7


def test_exception_preserves_partial_capture(tmp_path):
    args = record_args(tmp_path)
    with pytest.raises(RuntimeError, match="simulator stopped"):
        with EvalRecord(args, {}) as record:
            record.step({"episode": 1}, 1, [0., 0.], 1., False, False, {},
                        image=np.zeros((4, 5, 3), np.uint8))
            raise RuntimeError("simulator stopped")
    meta = json.loads((args.output_dir / "metadata.json").read_text())
    assert meta["status"] == "failed" and meta["partial_capture"]["frames"] == 1
    assert (args.output_dir / meta["partial_capture"]["animation"]).exists()


def test_capture_uses_copied_encoder_input_instead_of_live_camera():
    image = np.full((120, 160, 3), 30, np.uint8)
    base = TinyEnv()
    base.reset = lambda **kwargs: (image, {})
    base.step = lambda action: (image, 1., False, False, {"cte": 0., "speed": 1.})
    encoder = SimpleNamespace(z_size=1, encode=lambda obs: np.array([obs.mean()], np.float32))
    env = DonkeyVaeSACEnv(base, vae=encoder)
    env.reset()
    obs, _, _, _, info = env.step(np.zeros(2))
    image.fill(200)  # next asynchronous telemetry arrives
    assert obs[0] == 30
    assert np.all(env.last_raw_obs == 30)
    assert info["encoder_seconds"] >= 0
