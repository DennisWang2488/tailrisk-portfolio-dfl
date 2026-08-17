from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def write_summary_plots(summary: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for metric, title in [
        ("test_cvar", "Out-of-sample CVaR"),
        ("oracle_regret", "Oracle Regret"),
        ("tc_adjusted_return", "Transaction-cost-adjusted Return"),
        ("turnover", "Turnover"),
    ]:
        if metric not in summary:
            continue
        plt.figure(figsize=(11, 5))
        sns.barplot(data=summary, x="regime", y=metric, hue="method")
        plt.title(title)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output / f"{metric}.png", dpi=180)
        plt.close()

