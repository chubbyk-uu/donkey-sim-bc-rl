"""Read-only remote-desktop/GPU sampling alongside a paced perception benchmark.

Does not stop services, change remote settings, drive the simulator, or train.
Repeat with the same arguments/source frames after a manual remote disconnect.
This isolates perception latency; it does not measure Unity camera/TCP latency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rl.geometry_encoder import DinoGridLinearLag15Encoder, sha256_file  # noqa: E402
from rl.train_vae_sac import FrozenDinoGridEncoder  # noqa: E402


WINDOWS_SAMPLER = r'''
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ErrorActionPreference = "Stop"
$deadline = [DateTime]::UtcNow.AddSeconds(DURATION_SECONDS)
while ([DateTime]::UtcNow -lt $deadline) {
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $utc = [DateTime]::UtcNow.ToString("o")
    try {
        $procs = @(Get-Process | Where-Object {
            $_.ProcessName -match "AweSun|awesun_guard|Sunlogin|Oray|donkey_sim|^dwm$|NVIDIA Overlay"
        } | Select-Object ProcessName,Id,CPU,WorkingSet64)
        $names = @{}
        foreach ($p in $procs) { $names[[string]$p.Id] = $p.ProcessName }
        $engines = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine |
            Where-Object { $_.UtilizationPercentage -gt 0 } | ForEach-Object {
                $processId = ""
                if ($_.Name -match "pid_(\d+)_") { $processId = $Matches[1] }
                [pscustomobject]@{name=$_.Name;process_id=$processId;process_name=$names[$processId];utilization=$_.UtilizationPercentage}
            })
        [pscustomobject]@{utc=$utc;processes=$procs;gpu_engines=$engines;sample_ms=$watch.Elapsed.TotalMilliseconds} |
            ConvertTo-Json -Depth 5 -Compress
    } catch {
        [pscustomobject]@{utc=$utc;error=$_.Exception.Message} | ConvertTo-Json -Compress
    }
    $remaining = [Math]::Max(0, 1000-$watch.Elapsed.TotalMilliseconds)
    Start-Sleep -Milliseconds ([int]$remaining)
}
'''


def summarize_windows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    valid = [r for r in rows if "error" not in r]
    encode_samples = 0
    names = set()
    max_engine = 0
    for row in valid:
        remote_encoding = False
        for engine in row["gpu_engines"]:
            name = engine.get("process_name") or ""
            if any(x in name.lower() for x in ("awesun", "sunlogin", "oray")):
                names.add(name)
                if "VideoEncode" in engine["name"]:
                    remote_encoding = True
                    max_engine = max(max_engine, engine["utilization"])
        encode_samples += remote_encoding
    return {"samples": len(rows), "errors": len(rows)-len(valid),
            "remote_processes_with_gpu_activity": sorted(names),
            "samples_with_remote_video_encoding": encode_samples,
            "max_remote_individual_encode_engine_percent": max_engine,
            "sampler_ms_p50_p95": np.percentile([r["sample_ms"] for r in valid], [50, 95]).tolist() if valid else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True, help="Operator label; actual encoder activity is recorded independently")
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--period-ms", type=float, default=25)
    parser.add_argument("--capture", type=Path, default=Path("logs/round_b_recovery_collect_v1/captures/layout_01_teacher_recovery_dagger_train_pool.npz"))
    parser.add_argument("--head", type=Path, default=Path("models/dinov2_geometry_15d/proj15_lags_ridge.pt"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    args.output_dir.mkdir(parents=True)
    with np.load(args.capture) as cap:
        images = cap["images"]
    device = torch.device("cuda")
    torch.set_num_threads(2)
    dino = FrozenDinoGridEncoder(device=device, crop_top=0, projection=None)
    encoder = DinoGridLinearLag15Encoder(args.head, track=None, device=device, dino_encoder=dino)
    for i in range(20):
        encoder.predict_geometry_from_grid_tensor(dino.grid_features(images[i % len(images)][None]))
    torch.cuda.synchronize(device)
    encoder.reset_episode()
    metadata = {"phase": args.phase, "duration_seconds": args.duration, "period_ms": args.period_ms,
                "capture": str(args.capture), "capture_sha256": sha256_file(args.capture),
                "head": str(args.head), "head_sha256": sha256_file(args.head),
                "source_sha256": sha256_file(Path(__file__)), "started_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Recorded-frame perception only; Unity camera, TCP and driving are not measured. Services remain untouched."}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    host_path = args.output_dir / "windows_samples.jsonl"
    script = WINDOWS_SAMPLER.replace("DURATION_SECONDS", str(args.duration + 3))
    samples = []
    with host_path.open("w", encoding="utf-8") as host, (args.output_dir / "windows_sampler.stderr").open("w") as err:
        sampler = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], stdout=host, stderr=err)
        period = args.period_ms / 1000
        start = time.perf_counter()
        deadline = start
        index = 0
        while time.perf_counter() - start < args.duration:
            time.sleep(max(0, deadline-time.perf_counter()))
            t0 = time.perf_counter()
            utc = datetime.now(timezone.utc).isoformat()
            encoder.predict_geometry_from_grid_tensor(dino.grid_features(images[index % len(images)][None]))
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            next_deadline = deadline + period
            skipped = max(0, math.ceil((t1-next_deadline)/period))
            samples.append({"utc": utc, "index": index, "perception_ms": (t1-t0)*1000,
                            "wake_lateness_ms": (t0-deadline)*1000, "skipped_slots": skipped})
            deadline = next_deadline + skipped*period
            index += 1
        sampler.wait(timeout=15)
    with (args.output_dir / "perception_samples.jsonl").open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample)+"\n")
    durations = np.array([r["perception_ms"] for r in samples])
    summary = {"phase": args.phase, "samples": len(samples),
               "perception_p50_p95_p99_max_ms": np.percentile(durations, [50,95,99,100]).tolist(),
               "perception_over_25ms": int(np.sum(durations>25)), "skipped_slots": sum(r["skipped_slots"] for r in samples),
               "wake_lateness_p99_max_ms": np.percentile([r["wake_lateness_ms"] for r in samples], [99,100]).tolist(),
               "windows_sampler_exit_code": sampler.returncode, "windows": summarize_windows(host_path)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
