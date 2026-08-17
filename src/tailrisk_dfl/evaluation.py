from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .optimizer import CVaROptimizerParams, empirical_cvar, equal_weight, solve_cvar_lp
from .synthetic import SyntheticMarket


@dataclass
class BacktestResult:
    method: str
    returns: np.ndarray
    net_returns: np.ndarray
    weights: np.ndarray
    oracle_net_returns: np.ndarray | None = None


def backtest_method(method, features: np.ndarray, returns: np.ndarray, transaction_cost: float) -> BacktestResult:
    n_assets = returns.shape[1]
    prev = equal_weight(n_assets)
    weights = []
    gross = []
    net = []
    for x_t, r_t in zip(features, returns):
        w = method.decide(x_t, prev)
        turnover = np.abs(w - prev).sum()
        ret = float(w @ r_t)
        weights.append(w)
        gross.append(ret)
        net.append(ret - transaction_cost * turnover)
        prev = w
    return BacktestResult(method.name, np.asarray(gross), np.asarray(net), np.vstack(weights))


def oracle_backtest(
    market: SyntheticMarket,
    features: np.ndarray,
    returns: np.ndarray,
    params: CVaROptimizerParams,
    transaction_cost: float,
    n_scenarios: int,
    seed: int,
) -> BacktestResult:
    rng = np.random.default_rng(seed)
    prev = equal_weight(returns.shape[1])
    weights = []
    gross = []
    net = []
    for x_t, r_t in zip(features, returns):
        scenarios = market.sample_conditional_returns(x_t, n_scenarios, rng)
        w = solve_cvar_lp(scenarios, params, prev)
        turnover = np.abs(w - prev).sum()
        ret = float(w @ r_t)
        weights.append(w)
        gross.append(ret)
        net.append(ret - transaction_cost * turnover)
        prev = w
    return BacktestResult("oracle", np.asarray(gross), np.asarray(net), np.vstack(weights))


def compute_metrics(
    result: BacktestResult,
    alpha: float,
    gamma: float,
    oracle: BacktestResult | None = None,
) -> dict[str, float | str]:
    net = result.net_returns
    gross = result.returns
    weights = result.weights
    losses = -net
    downside = net[net < 0]
    wealth = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / np.maximum(peak, 1e-12) - 1.0
    turnover = np.abs(weights[1:] - weights[:-1]).sum(axis=1).mean() if len(weights) > 1 else 0.0
    vol = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    # Sortino downside deviation: sqrt(mean(min(r,0)^2)) over the FULL sample
    # (the previous version took np.std of only the negative returns, which is
    # not Sortino, and divided by 1e-12 when there were <2 losing periods).
    downside_dev = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)))
    cvar = empirical_cvar(losses, alpha)
    objective = cvar - gamma * float(net.mean())

    metrics: dict[str, float | str] = {
        "method": result.method,
        "test_cvar": cvar,
        "objective": objective,
        "mean_return": float(gross.mean()),
        "tc_adjusted_return": float(net.mean()),
        "volatility": vol,
        "sharpe": float(net.mean() / (vol + 1e-12)),
        "sortino": float(net.mean() / downside_dev) if downside_dev > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover),
        # Exceedance of a FIXED ex-ante threshold (the oracle portfolio's VaR),
        # shared across methods. The previous version measured each series
        # against its own quantile, which equals ceil(0.05n)/n for any series.
        "tail_loss_frequency": (
            float(np.mean(losses >= np.quantile(-oracle.net_returns, alpha)))
            if oracle is not None
            else float("nan")
        ),
    }
    if oracle is not None:
        oracle_losses = -oracle.net_returns
        oracle_obj = empirical_cvar(oracle_losses, alpha) - gamma * float(oracle.net_returns.mean())
        metrics["oracle_objective"] = oracle_obj
        metrics["oracle_regret"] = objective - oracle_obj  # lower is better; can be negative if we beat the oracle sample
    return metrics


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in results.columns
        if c not in {"experiment", "regime", "method", "seed"} and pd.api.types.is_numeric_dtype(results[c])
    ]
    grouped = results.groupby(["regime", "method"], as_index=False)
    mean = grouped[metric_cols].mean()
    sem = grouped[metric_cols].sem().fillna(0.0)
    sem = sem.rename(columns={c: f"{c}_sem" for c in metric_cols})
    summary = mean.merge(sem, on=["regime", "method"])

    win_rows = []
    for regime, frame in results.groupby("regime"):
        for metric, ascending in [("test_cvar", True), ("oracle_regret", True), ("tc_adjusted_return", False)]:
            if metric not in frame:
                continue
            winners = frame.loc[
                frame.groupby("seed")[metric].idxmin() if ascending else frame.groupby("seed")[metric].idxmax()
            ]
            rates = winners["method"].value_counts(normalize=True)
            for method, rate in rates.items():
                win_rows.append({"regime": regime, "method": method, f"{metric}_win_rate": rate})
    if win_rows:
        win_df = pd.DataFrame(win_rows)
        win_df = win_df.groupby(["regime", "method"], as_index=False).max()
        summary = summary.merge(win_df, on=["regime", "method"], how="left")
    return summary.fillna(0.0)

