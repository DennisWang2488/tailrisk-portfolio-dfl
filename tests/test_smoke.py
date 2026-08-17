from __future__ import annotations

from pathlib import Path

from tailrisk_dfl.config import load_config
from tailrisk_dfl.experiment import run_experiment


def test_smoke_experiment_runs() -> None:
    config = load_config(Path("configs") / "smoke.json")
    output = Path("outputs") / "test_smoke"
    results, summary = run_experiment(config, output)
    assert not results.empty
    assert not summary.empty
    assert {"equal_weight", "two_stage", "dfl"}.issubset(set(results["method"]))
    assert results["test_cvar"].notna().all()
