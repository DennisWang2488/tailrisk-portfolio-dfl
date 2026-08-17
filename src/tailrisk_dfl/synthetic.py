from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RegimeConfig


@dataclass
class MarketPath:
    features: np.ndarray
    returns: np.ndarray
    conditional_means: np.ndarray
    crash_flags: np.ndarray


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


class SyntheticMarket:
    """Synthetic conditional return generator with controllable tail-risk regimes."""

    def __init__(self, config: RegimeConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_assets = config.n_assets
        self.n_features = config.n_features
        self.n_factors = min(5, max(2, config.n_assets // 8))

        self.a_matrix = self.rng.normal(0.0, 1.0, size=(self.n_assets, self.n_features))
        self.a_matrix /= np.linalg.norm(self.a_matrix, axis=1, keepdims=True) + 1e-8

        self.b_matrix = self.rng.normal(0.0, 0.5, size=(self.n_assets, self.n_factors))
        self.factor_cov = self._make_factor_cov()
        self.idio_scale = self.rng.uniform(0.006, 0.018, size=self.n_assets)
        self.hidden_loading = self.rng.normal(0.0, 1.0, size=self.n_assets)
        self.hidden_loading /= np.linalg.norm(self.hidden_loading) + 1e-8
        self.crash_direction = -np.abs(self.rng.normal(0.0, 1.0, size=self.n_assets))
        self.crash_direction /= np.abs(self.crash_direction).mean() + 1e-8
        self.crash_beta = self.rng.normal(0.0, 1.0, size=self.n_features)
        self.crash_beta /= np.linalg.norm(self.crash_beta) + 1e-8

    def _make_factor_cov(self) -> np.ndarray:
        k = self.n_factors
        corr = self.config.correlation
        if corr == "independent":
            mat = np.eye(k)
        elif corr == "equicorrelation":
            mat = np.full((k, k), 0.35)
            np.fill_diagonal(mat, 1.0)
        elif corr == "block":
            mat = np.eye(k)
            split = max(1, k // 2)
            mat[:split, :split] = 0.55
            mat[split:, split:] = 0.35
            np.fill_diagonal(mat, 1.0)
        elif corr == "crisis":
            mat = np.full((k, k), 0.65)
            np.fill_diagonal(mat, 1.0)
        else:
            load = self.rng.normal(0.0, 0.35, size=(k, k))
            mat = load @ load.T
            d = np.sqrt(np.diag(mat))
            mat = mat / np.outer(d, d)
            mat = 0.65 * mat + 0.35 * np.eye(k)
        scale = np.linspace(0.008, 0.018, k)
        return mat * np.outer(scale, scale)

    def simulate(self) -> MarketPath:
        cfg = self.config
        features = np.zeros((cfg.n_periods, cfg.n_features))
        for t in range(1, cfg.n_periods):
            features[t] = 0.65 * features[t - 1] + self.rng.normal(0.0, 1.0, size=cfg.n_features)
        features = _standardize_columns(features)

        conditional_means = np.vstack([self.conditional_mean(x) for x in features])
        shocks, crash_flags = self.sample_shocks(features, self.rng)
        returns = conditional_means + shocks
        return MarketPath(features, returns, conditional_means, crash_flags)

    def conditional_mean(self, x: np.ndarray) -> np.ndarray:
        linear = self.a_matrix @ x
        nonlinear = np.zeros(self.n_assets)
        strength = self.config.misspecification_strength
        if self.config.misspecification == "nonlinear":
            basis = np.sin(x[: self.n_features // 2].sum()) + 0.5 * (x[-1] ** 2 - 1.0)
            nonlinear = strength * basis * self.hidden_loading
        elif self.config.misspecification == "heteroskedastic":
            nonlinear = strength * np.tanh(x[0]) * self.hidden_loading
        elif self.config.misspecification == "hidden_crash":
            nonlinear = strength * np.tanh(x @ self.crash_beta) * self.hidden_loading

        raw = linear + nonlinear
        raw = raw - raw.mean()
        shock_scale = np.sqrt(np.mean(np.diag(self.b_matrix @ self.factor_cov @ self.b_matrix.T)) + np.mean(self.idio_scale**2))
        target_mean_scale = self.config.snr * shock_scale
        raw_std = raw.std() + 1e-8
        return target_mean_scale * raw / raw_std

    def sample_conditional_returns(
        self,
        x: np.ndarray,
        n_scenarios: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = self.rng if rng is None else rng
        mean = self.conditional_mean(x)
        features = np.repeat(np.asarray(x)[None, :], n_scenarios, axis=0)
        shocks, _ = self.sample_shocks(features, rng)
        return mean[None, :] + shocks

    def sample_shocks(
        self,
        features: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_count = features.shape[0]
        factor = self._multivariate_skew_t(t_count, self.factor_cov, self.config.tail_df, self.config.skew, rng)
        idio_raw = rng.standard_t(df=max(2.2, self.config.tail_df), size=(t_count, self.n_assets))
        skew_component = self.config.skew * np.abs(rng.normal(size=(t_count, self.n_assets)))
        idio = (idio_raw + skew_component) * self.idio_scale
        shocks = factor @ self.b_matrix.T + idio

        crash_flags = np.zeros(t_count, dtype=bool)
        if self.config.crash_prob > 0:
            logits = features @ self.crash_beta
            logits = logits - logits.mean()
            probs = self.config.crash_prob * (1.0 + 1.5 / (1.0 + np.exp(-logits)))
            probs = np.clip(probs, 0.0, 0.45)
            crash_flags = rng.uniform(size=t_count) < probs
            if crash_flags.any():
                crisis_factor = rng.normal(1.0, 0.25, size=crash_flags.sum())[:, None]
                shocks[crash_flags] += crisis_factor * 0.035 * self.crash_direction[None, :]
        return shocks, crash_flags

    def _multivariate_skew_t(
        self,
        n: int,
        cov: np.ndarray,
        df: float,
        skew: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        dim = cov.shape[0]
        chol = np.linalg.cholesky(cov + 1e-10 * np.eye(dim))
        normal = rng.normal(size=(n, dim)) @ chol.T
        scale = np.sqrt(df / rng.chisquare(df, size=n))[:, None]
        skew_term = skew * np.abs(rng.normal(size=(n, dim))) @ chol.T
        return normal * scale + skew_term

