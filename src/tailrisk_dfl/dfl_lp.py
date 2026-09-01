from __future__ import annotations

import copy

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

from .baselines import PortfolioMethod, TorchMLPRegressor
from .dfl import _smooth_cvar
from .optimizer import CVaROptimizerParams, solve_cvar_lp

# "Real" decision-focused learning: the forecaster is trained by
# differentiating through the CVaR LP itself (cvxpylayers), not by a softmax
# policy that bypasses the optimizer.
#
#   x --f_θ--> μ(x) --+ bootstrap residuals--> scenarios --CVaR LP--> w
#   loss = smoothed CVaR of realized w·r over the minibatch  (same as dfl.py)
#
# f_θ is warm-started from the MSE fit (i.e. from two_stage / mlp_two_stage),
# so this is "estimate-then-optimize init, integrated fine-tune". Residuals are
# taken from that MSE fit and held fixed. At decision time we run the exact
# same HiGHS LP as two_stage (with prev weights + turnover), so the only thing
# that differs from two_stage is what the forecaster was trained on.


def _build_layer(
    n_scenarios: int, n_assets: int, alpha: float, gamma: float, w_max: float | None, quad_reg: float
) -> CvxpyLayer:
    R = cp.Parameter((n_scenarios, n_assets))
    w = cp.Variable(n_assets)
    eta = cp.Variable()
    xi = cp.Variable(n_scenarios, nonneg=True)
    objective = eta + cp.sum(xi) / ((1.0 - alpha) * n_scenarios) - gamma * (cp.sum(R, axis=0) @ w) / n_scenarios
    if quad_reg > 0:
        # QPTL trick (Wilder et al. 2019): an LP argmin is piecewise constant
        # in its parameters, so its gradient is ~0 a.e. A small quadratic term
        # makes the solution map smooth. Only in the training layer.
        objective = objective + quad_reg * cp.sum_squares(w)
    constraints = [xi >= -R @ w - eta, cp.sum(w) == 1, w >= 0]
    if w_max is not None:
        constraints.append(w <= w_max)
    # no turnover term in the training layer: the policy is stateless in
    # training (prev = equal weight for every row), so it would be a constant.
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return CvxpyLayer(problem, parameters=[R], variables=[w])


class DifferentiableLPMethod(PortfolioMethod):
    def __init__(
        self,
        name: str,
        params: CVaROptimizerParams,
        n_scenarios: int,
        seed: int,
        hidden: int,
        init_kwargs: dict,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        train_scenarios: int = 50,
        smooth_tau: float = 1e-4,
        patience: int = 6,
        early_stopping: bool = True,
        quad_reg: float = 1e-3,
        lr_grid: list[float] | None = None,
    ):
        self.quad_reg = quad_reg
        self.lr_grid = lr_grid or [lr]
        self.name = name
        self.params = params
        self.n_scenarios = n_scenarios
        self.seed = seed
        self.hidden = hidden
        self.init_kwargs = init_kwargs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.train_scenarios = train_scenarios
        self.smooth_tau = smooth_tau
        self.patience = patience
        self.early_stopping = early_stopping
        self.rng = np.random.default_rng(seed)
        self.selected_: dict = {}

    # --- helpers -----------------------------------------------------------
    def _scenarios(self, mu: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # mu: (B, N). residuals_: (T, N). idx: (B, S) fixed per row (common
        # random numbers, so the training objective is deterministic in θ).
        return mu[:, None, :] + self.residuals_t_[idx]

    def _decision_loss(self, model, x: torch.Tensor, r: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        mu = model(x) * self.y_scale_
        scen = self._scenarios(mu, idx)
        (w,) = self.layer_(scen, solver_args={"solve_method": "Clarabel"})
        realized = (w * r).sum(dim=1)
        return _smooth_cvar(-realized, self.params.alpha, self.smooth_tau) - self.params.gamma * realized.mean()

    def _prep(self, features: np.ndarray) -> torch.Tensor:
        return torch.tensor((features - self.x_mean_) / self.x_std_, dtype=torch.float32)

    def _finetune(self, model, x, r, epochs, x_val=None, r_val=None, lr=None):
        model = copy.deepcopy(model)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr if lr is None else lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)
        n = x.shape[0]
        T = self.residuals_t_.shape[0]
        scen_idx = torch.randint(0, T, (n, self.train_scenarios), generator=gen)
        val_idx = torch.randint(0, T, (x_val.shape[0], self.train_scenarios), generator=gen) if x_val is not None else None
        best_val, best_epoch, best_state, since_best = float("inf"), 0, copy.deepcopy(model.state_dict()), 0
        if x_val is not None:
            with torch.no_grad():
                best_val = float(self._decision_loss(model, x_val, r_val, val_idx))
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                if idx.numel() < 32:
                    continue
                loss = self._decision_loss(model, x[idx], r[idx], scen_idx[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if x_val is not None:
                with torch.no_grad():
                    val = float(self._decision_loss(model, x_val, r_val, val_idx))
                if val < best_val - 1e-9:
                    best_val, best_epoch, since_best = val, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break
        if x_val is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, best_val

    # --- API ---------------------------------------------------------------
    def fit(self, features, returns, val_features=None, val_returns=None):
        use_val = self.early_stopping and val_features is not None and len(val_features) > 0
        # 1. warm start: the MSE forecaster, exactly what (mlp_)two_stage uses.
        init = TorchMLPRegressor(seed=self.seed, hidden=self.hidden, **self.init_kwargs)
        init.fit(features, returns, val_features, val_returns)
        self.x_mean_, self.x_std_, self.y_scale_ = init.x_mean_, init.x_std_, init.y_scale_
        all_x = features if not use_val else np.vstack([features, val_features])
        all_r = returns if not use_val else np.vstack([returns, val_returns])
        self.residuals_ = all_r - init.predict(all_x)
        self.residuals_t_ = torch.tensor(self.residuals_, dtype=torch.float32)
        self.layer_ = _build_layer(
            self.train_scenarios, returns.shape[1], self.params.alpha, self.params.gamma, self.params.w_max, self.quad_reg
        )

        t = lambda a: torch.tensor(a, dtype=torch.float32)
        if not use_val:
            self.model_, _, _ = self._finetune(init.model_, self._prep(features), t(returns), self.epochs)
            self.selected_ = {"init": init.selected_, "epochs": self.epochs}
            return self
        # 2. fine-tune through the LP on train, pick (lr, epoch) on val, refit on train+val.
        # best_epoch == 0 means fine-tuning never beat the MSE init on val → stays two_stage.
        best = None
        for lr in self.lr_grid:
            _, best_epoch, best_val = self._finetune(
                init.model_, self._prep(features), t(returns), self.epochs, self._prep(val_features), t(val_returns), lr=lr
            )
            if best is None or best_val < best[0]:
                best = (best_val, lr, best_epoch)
        best_val, lr, best_epoch = best
        self.model_, _, _ = self._finetune(init.model_, self._prep(all_x), t(all_r), best_epoch, lr=lr)
        self.selected_ = {"init": init.selected_, "lr": lr, "epochs": best_epoch, "val_objective": best_val}
        return self

    def decide(self, feature, prev_weights):
        with torch.no_grad():
            pred = (self.model_(self._prep(np.asarray(feature)[None, :])) * self.y_scale_).numpy()[0]
        indices = self.rng.choice(len(self.residuals_), size=self.n_scenarios, replace=True)
        scenarios = pred[None, :] + self.residuals_[indices]
        return solve_cvar_lp(scenarios, self.params, prev_weights)
