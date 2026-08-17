from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class CVaROptimizerParams:
    alpha: float = 0.95
    gamma: float = 0.2
    w_max: float | None = 0.2
    turnover_penalty: float = 0.001
    turnover_limit: float | None = None


def equal_weight(n_assets: int) -> np.ndarray:
    return np.ones(n_assets, dtype=float) / n_assets


def project_to_capped_simplex(weights: np.ndarray, w_max: float | None = None) -> np.ndarray:
    """Project nonnegative weights to the simplex with an optional upper cap."""
    weights = np.asarray(weights, dtype=float)
    n_assets = weights.size
    cap = 1.0 if w_max is None else float(w_max)
    if cap * n_assets < 1.0 - 1e-12:
        raise ValueError("w_max is infeasible for the number of assets")

    # bisection on the water-filling offset. 100 steps is way more than we need.
    lower = weights.min() - cap
    upper = weights.max()
    for _ in range(100):
        mid = 0.5 * (lower + upper)
        projected = np.clip(weights - mid, 0.0, cap)
        total = projected.sum()
        if abs(total - 1.0) < 1e-12:
            return projected
        if total > 1.0:
            lower = mid
        else:
            upper = mid
    projected = np.clip(weights - 0.5 * (lower + upper), 0.0, cap)
    return projected / projected.sum()


def solve_cvar_lp(
    scenarios: np.ndarray,
    params: CVaROptimizerParams,
    prev_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve long-only scenario CVaR optimization using a linear program."""
    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 2:
        raise ValueError("scenarios must be a 2D array")
    n_scenarios, n_assets = scenarios.shape
    if n_scenarios < 2:
        raise ValueError("at least two scenarios are required")
    if prev_weights is None:
        prev_weights = equal_weight(n_assets)
    prev_weights = np.asarray(prev_weights, dtype=float)

    # vars: [w | eta | xi_s | u_i]
    # eta = VaR, xi = tail excess, u = |w - w_prev| (ℓ1 turnover)
    n_vars = n_assets + 1 + n_scenarios + n_assets
    eta_idx = n_assets
    xi_start = n_assets + 1
    u_start = xi_start + n_scenarios

    c = np.zeros(n_vars)
    c[:n_assets] = -params.gamma * scenarios.mean(axis=0)
    c[eta_idx] = 1.0
    c[xi_start:u_start] = 1.0 / ((1.0 - params.alpha) * n_scenarios)
    c[u_start:] = params.turnover_penalty

    a_eq = np.zeros((1, n_vars))
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    a_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []

    for s in range(n_scenarios):
        row = np.zeros(n_vars)
        row[:n_assets] = -scenarios[s]
        row[eta_idx] = -1.0
        row[xi_start + s] = -1.0
        a_ub_rows.append(row)
        b_ub.append(0.0)

    for i in range(n_assets):
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[u_start + i] = -1.0
        a_ub_rows.append(row)
        b_ub.append(float(prev_weights[i]))

        row = np.zeros(n_vars)
        row[i] = -1.0
        row[u_start + i] = -1.0
        a_ub_rows.append(row)
        b_ub.append(float(-prev_weights[i]))

    if params.turnover_limit is not None:
        row = np.zeros(n_vars)
        row[u_start:] = 1.0
        a_ub_rows.append(row)
        b_ub.append(float(params.turnover_limit))

    bounds = []
    cap = 1.0 if params.w_max is None else float(params.w_max)
    for _ in range(n_assets):
        bounds.append((0.0, cap))
    bounds.append((None, None))  # eta can be anything
    for _ in range(n_scenarios):
        bounds.append((0.0, None))
    for _ in range(n_assets):
        bounds.append((0.0, None))

    result = linprog(
        c,
        A_ub=np.vstack(a_ub_rows),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return equal_weight(n_assets)  # rare; don't kill a 5-seed grid over one LP
    return project_to_capped_simplex(result.x[:n_assets], params.w_max)


def solve_min_variance(
    covariance: np.ndarray,
    w_max: float | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_limit: float | None = None,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    n_assets = covariance.shape[0]
    start = equal_weight(n_assets) if prev_weights is None else np.asarray(prev_weights, dtype=float)
    cap = 1.0 if w_max is None else float(w_max)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if turnover_limit is not None and prev_weights is not None:
        constraints.append({"type": "ineq", "fun": lambda w: turnover_limit - np.abs(w - prev_weights).sum()})

    result = minimize(
        lambda w: float(w @ covariance @ w),
        start,
        method="SLSQP",
        bounds=[(0.0, cap)] * n_assets,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-10, "disp": False},
    )
    if not result.success:
        return project_to_capped_simplex(start, w_max)
    return project_to_capped_simplex(result.x, w_max)


def empirical_cvar(losses: np.ndarray, alpha: float) -> float:
    losses = np.asarray(losses, dtype=float)
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    if tail.size == 0:
        return float(var)
    return float(tail.mean())

