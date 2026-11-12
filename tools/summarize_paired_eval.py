"""Print completion-rate intervals and paired tests for a finished or aborted
paired evaluation (rl/eval_paired_randomized.py output directory). No simulator.

Usage:
    python tools/summarize_paired_eval.py logs/randomtree_baseline_12layouts_v2_50k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rl.experiment import format_paired_statistics, paired_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    for run in parser.parse_args().runs:
        episodes = json.loads((run / "results.json").read_text())["episodes"]
        print(f"{run}\n{format_paired_statistics(paired_statistics(episodes))}\n")


if __name__ == "__main__":
    main()
