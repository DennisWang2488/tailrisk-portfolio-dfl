from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tailrisk_dfl.config import load_config
from tailrisk_dfl.experiment import run_experiment
from tailrisk_dfl.plotting import write_summary_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic DFL vs two-stage CVaR portfolio experiments.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument("--output", required=True, help="Directory for CSV outputs.")
    parser.add_argument("--plots", action="store_true", help="Write summary plots.")
    args = parser.parse_args()

    config = load_config(args.config)
    results, summary = run_experiment(config, args.output)
    if args.plots:
        write_summary_plots(summary, Path(args.output) / "plots")
    print(f"Wrote {len(results)} method/seed rows to {args.output}")
    print(summary[["regime", "method", "test_cvar", "oracle_regret", "tc_adjusted_return"]].to_string(index=False))


if __name__ == "__main__":
    main()

