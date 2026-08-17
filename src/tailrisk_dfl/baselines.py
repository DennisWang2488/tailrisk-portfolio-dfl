from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .optimizer import CVaROptimizerParams, equal_weight, solve_cvar_lp, solve_min_variance


class PortfolioMethod:
    name: str

    def fit(self, features: np.ndarray, returns: np.ndarray) -> "PortfolioMethod":
        return self

    def decide(self, feature: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class EqualWeightMethod(PortfolioMethod):
    n_assets: int
    name: str = "equal_weight"

    def decide(self, feature: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        return equal_weight(self.n_assets)


@dataclass
class MinVarianceMethod(PortfolioMethod):
    n_assets: int
    w_max: float | None
    turnover_limit: float | None
    name: str = "min_variance"

    def fit(self, features: np.ndarray, returns: np.ndarray) -> "MinVarianceMethod":
        self.covariance_ = LedoitWolf().fit(returns).covariance_
        return self

    def decide(self, feature: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        return solve_min_variance(self.covariance_, self.w_max, prev_weights, self.turnover_limit)


class TwoStageCVaRMethod(PortfolioMethod):
    def __init__(
        self,
        name: str,
        params: CVaROptimizerParams,
        ridge_alpha: float,
        n_scenarios: int,
        seed: int,
        robust: bool = False,
    ):
        self.name = name
        self.params = params
        self.ridge_alpha = ridge_alpha
        self.n_scenarios = n_scenarios
        self.rng = np.random.default_rng(seed)
        self.robust = robust

    def fit(self, features: np.ndarray, returns: np.ndarray) -> "TwoStageCVaRMethod":
        self.model_ = make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(Ridge(alpha=self.ridge_alpha)),
        )
        self.model_.fit(features, returns)
        fitted = self.model_.predict(features)
        self.residuals_ = returns - fitted
        # residual bootstrap: predicted mean + a resampled residual row.
        # robust version overweight the bad residuals so the LP sees more tail.
        self.tail_probs_ = self._tail_sampling_probs(self.residuals_) if self.robust else None
        return self

    def decide(self, feature: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        pred = self.model_.predict(np.asarray(feature)[None, :])[0]
        if self.robust:
            pred = 0.6 * pred  # shrink the mean; ridge already shrinks, this is extra
            indices = self.rng.choice(len(self.residuals_), size=self.n_scenarios, replace=True, p=self.tail_probs_)
        else:
            indices = self.rng.choice(len(self.residuals_), size=self.n_scenarios, replace=True)
        scenarios = pred[None, :] + self.residuals_[indices]
        return solve_cvar_lp(scenarios, self.params, prev_weights)

    @staticmethod
    def _tail_sampling_probs(residuals: np.ndarray) -> np.ndarray:
        tail_score = -residuals.mean(axis=1)
        ranks = np.argsort(np.argsort(tail_score)).astype(float)
        # 1 + 3*(rank/max) → worst residual is 4× as likely as the best. arbitrary.
        probs = 1.0 + 3.0 * ranks / max(1.0, ranks.max())
        return probs / probs.sum()

