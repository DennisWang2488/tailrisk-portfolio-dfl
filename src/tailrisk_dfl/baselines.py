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

    def fit(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        val_features: np.ndarray | None = None,
        val_returns: np.ndarray | None = None,
    ) -> "PortfolioMethod":
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

    def fit(self, features: np.ndarray, returns: np.ndarray, val_features=None, val_returns=None) -> "MinVarianceMethod":
        if val_features is not None and len(val_features) > 0:
            returns = np.vstack([returns, val_returns])
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
        regressor: str = "ridge",
        mlp_kwargs: dict | None = None,
    ):
        self.name = name
        self.params = params
        self.ridge_alpha = ridge_alpha
        self.n_scenarios = n_scenarios
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.robust = robust
        self.regressor = regressor
        self.mlp_kwargs = mlp_kwargs or {}

    def fit(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        val_features: np.ndarray | None = None,
        val_returns: np.ndarray | None = None,
    ) -> "TwoStageCVaRMethod":
        if self.regressor == "ridge":
            # ridge doesn't need a val block; give it the same rows everyone else refits on
            if val_features is not None and len(val_features) > 0:
                features = np.vstack([features, val_features])
                returns = np.vstack([returns, val_returns])
            self.model_ = make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(Ridge(alpha=self.ridge_alpha)),
            )
            self.model_.fit(features, returns)
        elif self.regressor == "mlp":
            # same architecture as the DFL net, trained on MSE. this is the
            # "model class" arm of the 2x2: MLP + predict-then-optimize.
            self.model_ = TorchMLPRegressor(seed=self.seed, **self.mlp_kwargs)
            self.model_.fit(features, returns, val_features, val_returns)
            if val_features is not None and len(val_features) > 0:
                features = np.vstack([features, val_features])
                returns = np.vstack([returns, val_returns])
        else:
            raise ValueError(f"unknown regressor: {self.regressor}")
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



class TorchMLPRegressor:
    """Multi-output MLP fit by MSE, with the same early-stopping protocol as the DFL net:
    pick (lr, hidden, epoch) on the val block, then refit on train+val for that many epochs."""

    def __init__(
        self,
        seed: int,
        hidden: int = 64,
        epochs: int = 250,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        patience: int = 30,
        lr_grid: list[float] | None = None,
        hidden_grid: list[int] | None = None,
    ):
        self.seed = seed
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.lr_grid = lr_grid or [lr]
        self.hidden_grid = hidden_grid or [hidden]
        self.selected_: dict = {}

    def _build(self, n_in: int, n_out: int, hidden: int):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        if hidden <= 0:
            return nn.Linear(n_in, n_out)
        return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def _train(self, x, y, hidden, lr, epochs, x_val=None, y_val=None):
        import copy

        import torch

        model = self._build(x.shape[1], y.shape[1], hidden)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        best_val, best_epoch, best_state, since_best = float("inf"), epochs, None, 0
        for epoch in range(1, epochs + 1):
            loss = torch.mean((model(x) - y) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if x_val is not None:
                with torch.no_grad():
                    val = float(torch.mean((model(x_val) - y_val) ** 2))
                if val < best_val - 1e-12:
                    best_val, best_epoch, since_best = val, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, best_val

    def fit(self, features, returns, val_features=None, val_returns=None):
        import torch

        use_val = val_features is not None and len(val_features) > 0
        all_x = features if not use_val else np.vstack([features, val_features])
        all_y = returns if not use_val else np.vstack([returns, val_returns])
        self.x_mean_ = all_x.mean(axis=0, keepdims=True)
        self.x_std_ = all_x.std(axis=0, keepdims=True) + 1e-8
        # returns are ~1e-2; scale the target so MSE isn't numerically tiny
        self.y_scale_ = float(all_y.std()) + 1e-12
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        prep = lambda a: t((a - self.x_mean_) / self.x_std_)
        if not use_val:
            self.model_, _, _ = self._train(prep(features), t(returns / self.y_scale_), self.hidden, self.lr, self.epochs)
            self.selected_ = {"hidden": self.hidden, "lr": self.lr, "epochs": self.epochs}
            return self
        best = None
        for hidden in self.hidden_grid:
            for lr in self.lr_grid:
                _, best_epoch, val = self._train(
                    prep(features), t(returns / self.y_scale_), hidden, lr, self.epochs, prep(val_features), t(val_returns / self.y_scale_)
                )
                if best is None or val < best[0]:
                    best = (val, hidden, lr, best_epoch)
        val, hidden, lr, best_epoch = best
        self.model_, _, _ = self._train(prep(all_x), t(all_y / self.y_scale_), hidden, lr, max(1, best_epoch))
        self.selected_ = {"hidden": hidden, "lr": lr, "epochs": best_epoch, "val_mse": val}
        return self

    def predict(self, features):
        import torch

        x = torch.tensor((np.asarray(features) - self.x_mean_) / self.x_std_, dtype=torch.float32)
        with torch.no_grad():
            return self.model_(x).numpy() * self.y_scale_
