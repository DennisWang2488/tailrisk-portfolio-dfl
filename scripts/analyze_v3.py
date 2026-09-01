"""Paired-by-seed tables for the v3 ablation grid and the sample-size curve.

Usage:
    python scripts/analyze_v3.py outputs/research_grid_v3 [outputs/research_grid_v2]
    python scripts/analyze_v3.py --curve outputs/sample_size_curve [outputs/sample_size_curve_lp]
    python scripts/analyze_v3.py --lp outputs/research_grid_v4 outputs/research_grid_v3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired(df: pd.DataFrame, a: str, b: str, metric: str = "test_cvar") -> pd.DataFrame:
    """Per-regime paired comparison of method a vs b (a - b; negative = a better, lower is better)."""
    rows = []
    for regime, frame in df.groupby("regime", sort=False):
        piv = frame.pivot(index="seed", columns="method", values=metric)
        if a not in piv or b not in piv:
            continue
        d = piv[a] - piv[b]
        t = stats.ttest_rel(piv[a], piv[b]) if len(d) > 1 else None
        rows.append(
            {
                "regime": regime,
                a: piv[a].mean(),
                b: piv[b].mean(),
                "ratio": piv[a].mean() / piv[b].mean() if piv[b].mean() != 0 else np.nan,
                "paired_t": t.statistic if t else np.nan,
                "wins": f"{int((d < 0).sum())}/{len(d)}",
            }
        )
    return pd.DataFrame(rows)


def rank_table(df: pd.DataFrame, metric: str = "test_cvar") -> pd.DataFrame:
    mean = df.groupby(["regime", "method"])[metric].mean().unstack("method")
    return mean.rank(axis=1).astype(int)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--lp" in sys.argv:
        v4 = pd.read_csv(Path(args[0]) / "results_by_seed.csv")
        v3 = pd.read_csv(Path(args[1]) / "results_by_seed.csv")
        df = pd.concat([v3, v4], ignore_index=True)
        keep = ["two_stage", "mlp_two_stage", "linear_dfl", "dfl", "dfl_lp", "mlp_dfl_lp"]
        df = df[df.method.isin(keep)]
        for metric in ["test_cvar", "oracle_regret"]:
            print(f"\n== v3+v4 means: {metric} ==")
            print(df.pivot_table(index="regime", columns="method", values=metric, aggfunc="mean")[keep].to_string(float_format="%.5f"))
            print(f"\n== ranks (1 = best): {metric} ==")
            print(rank_table(df, metric)[keep].to_string())
        for a, b, why in [
            ("dfl_lp", "two_stage", "linear forecaster: LP-finetuned vs MSE"),
            ("mlp_dfl_lp", "mlp_two_stage", "MLP forecaster: LP-finetuned vs MSE"),
            ("dfl_lp", "linear_dfl", "linear: through-the-LP vs softmax policy"),
            ("mlp_dfl_lp", "dfl", "MLP: through-the-LP vs softmax policy"),
            ("mlp_dfl_lp", "two_stage", "best true-DFL vs ridge two-stage"),
        ]:
            print(f"\n== {a} - {b} ({why}) ==")
            print(paired(df, a, b).to_string(index=False, float_format="%.5f"))
        print("\n== selected (v4) ==")
        print(v4[["regime", "seed", "method", "selected"]].to_string(index=False))
        return

    if "--curve" in sys.argv:
        df = pd.concat([pd.read_csv(Path(a) / "results_by_seed.csv") for a in args], ignore_index=True)
        df["n"] = df["n_periods"]
        for metric in ["test_cvar", "oracle_regret"]:
            print(f"\n== sample-size curve: {metric} (mean over seeds) ==")
            print(df.pivot_table(index="n", columns="method", values=metric, aggfunc="mean").to_string(float_format="%.5f"))
        print("\n== paired: dfl - two_stage, by n ==")
        print(paired(df.assign(regime=df["n"]), "dfl", "two_stage").to_string(index=False, float_format="%.5f"))
        print("\n== paired: linear_dfl - two_stage, by n ==")
        print(paired(df.assign(regime=df["n"]), "linear_dfl", "two_stage").to_string(index=False, float_format="%.5f"))
        print("\n== paired: mlp_two_stage - two_stage, by n ==")
        print(paired(df.assign(regime=df["n"]), "mlp_two_stage", "two_stage").to_string(index=False, float_format="%.5f"))
        for a, b in [("dfl_lp", "two_stage"), ("mlp_dfl_lp", "mlp_two_stage"), ("mlp_dfl_lp", "dfl")]:
            if a in set(df.method):
                print(f"\n== paired: {a} - {b}, by n ==")
                print(paired(df.assign(regime=df["n"]), a, b).to_string(index=False, float_format="%.5f"))
        return

    v3 = pd.read_csv(Path(args[0]) / "results_by_seed.csv")
    for metric in ["test_cvar", "oracle_regret"]:
        print(f"\n== v3 means: {metric} ==")
        print(v3.pivot_table(index="regime", columns="method", values=metric, aggfunc="mean").to_string(float_format="%.5f"))
        print(f"\n== v3 ranks (1 = best): {metric} ==")
        print(rank_table(v3, metric).to_string())

    print("\n== 2x2: objective axis. linear_dfl - two_stage (same model class, CVaR vs MSE) ==")
    print(paired(v3, "linear_dfl", "two_stage").to_string(index=False, float_format="%.5f"))
    print("\n== 2x2: model-class axis. mlp_two_stage - two_stage (same objective, MLP vs ridge) ==")
    print(paired(v3, "mlp_two_stage", "two_stage").to_string(index=False, float_format="%.5f"))
    print("\n== 2x2: both. dfl - two_stage ==")
    print(paired(v3, "dfl", "two_stage").to_string(index=False, float_format="%.5f"))
    print("\n== dfl - mlp_two_stage (same model class, CVaR vs MSE) ==")
    print(paired(v3, "dfl", "mlp_two_stage").to_string(index=False, float_format="%.5f"))
    print("\n== dfl_stateful - dfl (prev-weights input) ==")
    print(paired(v3, "dfl_stateful", "dfl").to_string(index=False, float_format="%.5f"))

    if len(args) > 1:
        v2 = pd.read_csv(Path(args[1]) / "results_by_seed.csv")
        both = pd.concat([v2[v2.method == "dfl"].assign(method="dfl_v2"), v3[v3.method == "dfl"].assign(method="dfl_v3")])
        print("\n== early stopping effect: dfl_v3 - dfl_v2 (same data, val-selected vs fixed 250 epochs) ==")
        print(paired(both, "dfl_v3", "dfl_v2").to_string(index=False, float_format="%.5f"))

    print("\n== selected DFL hyperparameters (v3) ==")
    sel = v3[v3.method.isin(["dfl", "linear_dfl", "dfl_stateful", "mlp_two_stage"])][["regime", "seed", "method", "selected"]]
    print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
