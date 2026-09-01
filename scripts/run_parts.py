"""Run one regime of a config (for cheap parallelism across processes), or merge parts.

    python scripts/run_parts.py --config C --output OUT --regime-index i   # writes OUT/part_i/
    python scripts/run_parts.py --merge OUT                                 # OUT/part_*/ -> OUT/
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tailrisk_dfl.config import config_to_dict, load_config  # noqa: E402
from tailrisk_dfl.evaluation import summarize_results  # noqa: E402
from tailrisk_dfl.experiment import run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--output", required=True)
    ap.add_argument("--regime-index", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    out = Path(a.output)
    if a.merge:
        parts = sorted(out.glob("part_*/results_by_seed.csv"))
        results = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        results.to_csv(out / "results_by_seed.csv", index=False)
        summarize_results(results).to_csv(out / "summary.csv", index=False)
        cfg = json.loads((parts[0].parent / "config_resolved.json").read_text())
        (out / "config_resolved.json").write_text(json.dumps(cfg, indent=2))
        print(f"merged {len(parts)} parts, {len(results)} rows")
        return
    config = load_config(a.config)
    config = dataclasses.replace(config, regimes=[config.regimes[a.regime_index]])
    run_experiment(config, out / f"part_{a.regime_index}")


if __name__ == "__main__":
    main()
