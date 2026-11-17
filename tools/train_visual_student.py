"""One bounded offline behavior-cloning run, camera/self-command inputs only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.visual_student import (GeometryProjection, OBSERVATION_CONTRACT, VisualStudentActor,
                               map_actions, prepare_visual_inputs)  # noqa: E402
from tools.compare_geometry_control import action_metrics, persistent_bias, sequence_segments  # noqa: E402


@torch.no_grad()
def teacher_labels(teacher, true_state, issued_history):
    # Privileged state exists only on this label side; never passed to the builder/actor.
    return np.concatenate([teacher.predict(np.concatenate([true_state[i:i+2048], issued_history[i:i+2048]], axis=1),
                                           deterministic=True)[0] for i in range(0, len(true_state), 2048)])


@torch.no_grad()
def evaluate(actor, observations, labels, state, validation, groups, run, layout, condition, step):
    raw = actor.predict(observations)[0]
    history = observations[:, 45:]
    mapped = map_actions(torch.from_numpy(raw), torch.from_numpy(history)).numpy()
    target = map_actions(torch.from_numpy(labels), torch.from_numpy(history)).numpy()
    subsets = {"all": validation, "startup": validation & (step <= 250),
               "large_offset": validation & (np.abs(state[:, 0]*2) >= 1),
               "negative_large": validation & (state[:, 0]*2 <= -1),
               "positive_large": validation & (state[:, 0]*2 >= 1)}
    metrics = {key: action_metrics(raw, mapped, labels, target, history, mask, state) for key, mask in subsets.items()}
    metrics["per_layout_steer_mae"] = {str(g): float(np.mean(np.abs(mapped[validation & (groups == g), 0] - target[validation & (groups == g), 0])))
                                         for g in sorted(set(groups[validation]))}
    segments = sequence_segments(run[validation], layout[validation], condition[validation], step[validation])
    metrics["persistent_bias"] = persistent_bias((mapped[:, 0]-target[:, 0])[validation], segments)[0]
    return metrics, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("logs/round_b_negative_geometry_train_v1"))
    parser.add_argument("--geometry-head", type=Path, default=Path("models/dinov2_geometry_15d/proj15_lags_ridge.pt"))
    parser.add_argument("--teacher", type=Path, default=Path("models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/visual_student_no_motion_v1"))
    parser.add_argument("--export", type=Path, default=Path("models/visual_student_no_motion_v1/student.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--training-seconds", type=float, default=180)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.export.exists() or (args.output_dir / "training_manifest.json").exists():
        parser.error("Use new artifact paths; existing student/training run is preserved")
    if args.epochs < 1 or not 0 < args.training_seconds <= 180:
        parser.error("Need positive epochs and a training budget <=180 seconds")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    manifest = json.loads((args.cache_dir / "training_manifest.json").read_text())
    canonical = {k:v for k,v in manifest.items() if k != "fingerprint_sha256"}
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest() == manifest["fingerprint_sha256"]
    captures = manifest["captures"]
    # First run uses complete, reset-aligned trajectories. Tail-only legacy rows
    # remain untouched and are not silently treated as episode starts.
    with np.load(args.cache_dir / "feature_cache.npz", allow_pickle=False) as cache:
        use = np.isin(cache["run"], sorted({e["run"] for e in captures}))
        history = cache["history"][use]
        run, layout, condition, step = (cache[k][use] for k in ["run", "layout", "model", "step"])
        validation = cache["internal_validation"][use]
        sequence = np.array([f"{r}/{l}/{c}" for r,l,c in zip(run,layout,condition)])
        projection = GeometryProjection(args.geometry_head, args.device)
        x, eligible, full_context, geometry = prepare_visual_inputs(cache["patch_grid"][use], history, sequence, step, projection)
        # Read true state only AFTER constructing the separate camera-only input.
        true_state = cache["state"][use]
    group = np.array([f"{r}/{l}" for r,l in zip(run,layout)])
    assert not set(group[validation]) & set(group[~validation])
    train = ~validation & eligible
    failed_capture = np.zeros(len(x), bool)
    provenance = []
    for entry in captures:
        mask = (run == entry["run"]) & (layout == entry["layout"]) & (condition == entry["model"])
        assert np.all(validation[mask] == entry["internal_validation"])
        with np.load(entry["path"], allow_pickle=False) as cap:
            if str(cap["split"]) != "dagger_train_pool":
                raise ValueError("Evaluation holdout cannot enter the training dataset")
            if not bool(cap["completed"]):
                failed_capture |= mask
            provenance.append(dict(entry, episode_completed=bool(cap["completed"])))
    # Complete successful demonstrations only for this first control fit.
    # Failed validation trajectories are retained, not filtered to inflate scores.
    train &= ~failed_capture
    validation &= eligible
    spec = {"observation_contract": OBSERVATION_CONTRACT, "geometry_head": str(args.geometry_head),
            "geometry_head_sha256": projection.sha256, "teacher_labels_only": str(args.teacher),
            "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
            "source_cache_manifest_sha256": hashlib.sha256((args.cache_dir/"training_manifest.json").read_bytes()).hexdigest(),
            "captures": provenance, "training_frames": int(train.sum()), "validation_frames": int(validation.sum()),
            "excluded_failed_training_frames": int((~validation & failed_capture).sum()),
            "validation_groups": sorted(set(group[validation])), "legacy_tail_rows_used": False,
            "initialization": "Teacher geometry/history columns transferred; three motion columns discarded; past geometry columns initialized zero.",
            "gate": {"mapped_steer_mae_ratio_max":0.8,"large_offset_steer_mae_ratio_max":1.1,"missed_unwind_rate_max_increase":0.02},
            "training": {"seed":42,"lr":1e-4,"weight_decay":1e-4,"batch_size":512,"epochs_cap":args.epochs,
                         "loss":"mapped steer MSE + .1 raw steer MSE + .5 mapped throttle MSE","early_stop_patience":3}}
    (args.output_dir/"training_manifest.json").write_text(json.dumps(spec,indent=2)+'\n')
    np.savez(args.output_dir/"student_inputs.npz", observations=x, train=train, validation=validation,
             full_context=full_context, groups=group, step=step)
    teacher = SAC.load(str(args.teacher), device="cpu")
    y = teacher_labels(teacher, true_state, history)
    np.savez(args.output_dir/"teacher_labels.npz", raw_actions=y)
    actor = VisualStudentActor()
    actor.initialize_from_teacher(teacher)
    initial, initial_raw = evaluate(actor,x,y,true_state,validation,group,run,layout,condition,step)
    actor.to(args.device)
    inputs = torch.from_numpy(x).to(args.device)
    targets = torch.from_numpy(y).to(args.device)
    indices = np.flatnonzero(train)
    optimizer = torch.optim.AdamW(actor.parameters(),lr=1e-4,weight_decay=1e-4)
    best_weights = copy.deepcopy(actor.state_dict())
    best = initial["all"]["steer_mae"]
    epochs, patience = [], 0
    fit_started = time.monotonic()
    for epoch in range(1,args.epochs+1):
        actor.train()
        losses=[]
        for rows in np.array_split(rng.permutation(indices),max(1,int(np.ceil(len(indices)/512)))):
            if time.monotonic()-fit_started > args.training_seconds:
                break
            batch = inputs[rows]; labels=targets[rows]; raw=actor(batch)
            command=map_actions(raw,batch[:,45:]); target=map_actions(labels,batch[:,45:])
            loss=(command[:,0]-target[:,0]).square().mean()+.1*(raw[:,0]-labels[:,0]).square().mean()+.5*(command[:,1]-target[:,1]).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        actor.eval()
        metrics,_=evaluate(actor,x,y,true_state,validation,group,run,layout,condition,step)
        score=metrics["all"]["steer_mae"]
        epochs.append({"epoch":epoch,"loss":float(np.mean(losses)) if losses else None,"validation_steer_mae":score})
        print(json.dumps(epochs[-1]),flush=True)
        if score<best:
            best=score;best_weights=copy.deepcopy(actor.state_dict());patience=0
        else:patience+=1
        if patience>=3 or time.monotonic()-fit_started>args.training_seconds:break
    actor.load_state_dict(best_weights);actor.eval()
    final, final_raw=evaluate(actor,x,y,true_state,validation,group,run,layout,condition,step)
    mae_ratio=final['all']['steer_mae']/initial['all']['steer_mae']
    large_ratio=final['large_offset']['steer_mae']/initial['large_offset']['steer_mae']
    unwind_change=final['all']['missed_unwind_rate']-initial['all']['missed_unwind_rate']
    gate=mae_ratio<=.8 and large_ratio<=1.1 and unwind_change<=.02
    report={"status":"offline_only","initialization":initial,"student":final,"epochs":epochs,
            "mapped_steer_mae_ratio":mae_ratio,"large_offset_steer_mae_ratio":large_ratio,
            "missed_unwind_rate_change":unwind_change,"offline_gate_passed":gate,
            "fit_seconds":time.monotonic()-fit_started,"total_seconds":time.monotonic()-started,
            "simulator_driven":False,"baseline_overwritten":False,
            "next":"eligible for a separately authorized closed-loop screen" if gate else "stop; no automatic refit or simulation"}
    actor.save(args.export,geometry_head_sha256=projection.sha256,metadata={"teacher_sha256":spec['teacher_sha256'],"offline_gate_passed":gate})
    loaded=VisualStudentActor.load(args.export,geometry_head_sha256=projection.sha256,device=args.device)
    np.testing.assert_allclose(loaded.predict(x[:256])[0],final_raw[:256],atol=1e-6,rtol=0)
    np.savez(args.output_dir/'validation_predictions.npz',initial_raw=initial_raw[validation],student_raw=final_raw[validation],teacher_raw=y[validation])
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    source=args.output_dir/'source';source.mkdir()
    for file in [Path(__file__),ROOT/'rl/visual_student.py',ROOT/'tools/compare_geometry_control.py']:
        shutil.copy2(file,source/file.name)
    print('OFFLINE GATE',gate,'ratio',mae_ratio,'large',large_ratio,'unwind change',unwind_change,flush=True)


if __name__=='__main__':
    main()
