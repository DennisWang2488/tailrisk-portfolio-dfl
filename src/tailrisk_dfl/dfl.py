from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .baselines import PortfolioMethod
from .optimizer import equal_weight, project_to_capped_simplex

# Softmax policy + a smoothed CVaR objective. I didn't unroll the LP — the
# gradient through HiGHS was garbage at these sample sizes.


class PortfolioNet(nn.Module):
    def __init__(self, n_features: int, n_assets: int, hidden: int):
        super().__init__()
        if hidden <= 0:
            # linear policy: same objective as the MLP, ridge-sized model class.
            self.net = nn.Linear(n_features, n_assets)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_assets),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # already on the simplex; we still project later for the 20% cap
        return torch.softmax(self.net(x), dim=-1)


def _smooth_cvar(losses: torch.Tensor, alpha: float, tau: float) -> torch.Tensor:
    # detach the quantile — torch.quantile isn't a useful backward pass.
    # tau has to be << the loss sd or this is just a smeared mean. v1 had 0.01
    # and only ~7–40% of the gradient landed on the actual tail. use 1e-4.
    eta = torch.quantile(losses.detach(), alpha)
    excess = torch.nn.functional.softplus((losses - eta) / tau) * tau
    return eta + excess.mean() / (1.0 - alpha)


@dataclass
class DecisionFocusedCVaRMethod(PortfolioMethod):
    n_features: int
    n_assets: int
    alpha: float
    gamma: float
    w_max: float | None
    transaction_cost: float
    hidden: int
    epochs: int
    lr: float
    weight_decay: float
    smooth_tau: float
    seed: int
    robust: bool = False
    name: str = "dfl"
    # feed w_{t-1} to the policy. training rolls the policy through time with a
    # detached previous weight, so the turnover term is the real path, not a proxy.
    use_prev_weights: bool = False
    # early stopping / model selection on a held-out block. when val data is
    # passed to fit(), every (lr, hidden) in the grids is trained on train,
    # scored on val, and the winner is refit on train+val for its best epoch.
    early_stopping: bool = False
    patience: int = 30
    lr_grid: list[float] = field(default_factory=list)
    hidden_grid: list[int] = field(default_factory=list)
    selected_: dict = field(default_factory=dict)

    @property
    def _input_dim(self) -> int:
        return self.n_features + (self.n_assets if self.use_prev_weights else 0)

    def _prep(self, features: np.ndarray) -> torch.Tensor:
        return torch.tensor((features - self.x_mean_) / self.x_std_, dtype=torch.float32)

    def _rollout(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.use_prev_weights:
            return model(x)
        prev = torch.full((self.n_assets,), 1.0 / self.n_assets)
        out = []
        for t in range(x.shape[0]):
            # prev weights are scaled to O(1) so they don't vanish next to the standardized features
            inp = torch.cat([x[t], (prev - 1.0 / self.n_assets) * self.n_assets])
            w = model(inp[None, :])[0]
            out.append(w)
            prev = w.detach()
        return torch.stack(out)

    def _objective(self, weights: torch.Tensor, r: torch.Tensor, alpha: float) -> torch.Tensor:
        gross_returns = (weights * r).sum(dim=1)
        losses = -gross_returns
        cvar = _smooth_cvar(losses, alpha, self.smooth_tau)
        mean_term = -self.gamma * gross_returns.mean()
        # consecutive training rows. without use_prev_weights the policy is
        # stateless so this is only a turnover *proxy*.
        turnover = torch.abs(weights[1:] - weights[:-1]).sum(dim=1).mean()
        cap_penalty = torch.tensor(0.0)
        if self.w_max is not None:
            cap_penalty = 10.0 * torch.relu(weights - self.w_max).pow(2).mean()  # 10 is a hack, just has to be loud
        return cvar + mean_term + self.transaction_cost * turnover + cap_penalty

    def _train(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        hidden: int,
        lr: float,
        epochs: int,
        x_val: torch.Tensor | None = None,
        r_val: torch.Tensor | None = None,
    ) -> tuple[nn.Module, int, float]:
        torch.manual_seed(self.seed)
        model = PortfolioNet(self._input_dim, self.n_assets, hidden)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        alpha = min(0.99, self.alpha + 0.025) if self.robust else self.alpha
        best_val = float("inf")
        best_epoch = epochs
        best_state = None
        since_best = 0
        for epoch in range(1, epochs + 1):
            weights = self._rollout(model, x)
            objective = self._objective(weights, r, alpha)
            opt.zero_grad()
            objective.backward()
            opt.step()
            if x_val is not None:
                with torch.no_grad():
                    val_obj = float(self._objective(self._rollout(model, x_val), r_val, alpha))
                if val_obj < best_val - 1e-9:
                    best_val, best_epoch, since_best = val_obj, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, best_val

    def fit(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        val_features: np.ndarray | None = None,
        val_returns: np.ndarray | None = None,
    ) -> "DecisionFocusedCVaRMethod":
        use_val = self.early_stopping and val_features is not None and len(val_features) > 0
        all_x = features if not use_val else np.vstack([features, val_features])
        all_r = returns if not use_val else np.vstack([returns, val_returns])
        self.x_mean_ = all_x.mean(axis=0, keepdims=True)
        self.x_std_ = all_x.std(axis=0, keepdims=True) + 1e-8

        if not use_val:
            self.model_, _, _ = self._train(self._prep(features), torch.tensor(returns, dtype=torch.float32), self.hidden, self.lr, self.epochs)
            self.selected_ = {"hidden": self.hidden, "lr": self.lr, "epochs": self.epochs}
            return self

        x_tr, r_tr = self._prep(features), torch.tensor(returns, dtype=torch.float32)
        x_va, r_va = self._prep(val_features), torch.tensor(val_returns, dtype=torch.float32)
        lrs = self.lr_grid or [self.lr]
        hiddens = self.hidden_grid or [self.hidden]
        best = None
        for hidden in hiddens:
            for lr in lrs:
                _, best_epoch, val_obj = self._train(x_tr, r_tr, hidden, lr, self.epochs, x_va, r_va)
                if best is None or val_obj < best[0]:
                    best = (val_obj, hidden, lr, best_epoch)
        val_obj, hidden, lr, best_epoch = best
        # refit on train+val for the selected epoch count so every method sees the same rows
        self.model_, _, _ = self._train(self._prep(all_x), torch.tensor(all_r, dtype=torch.float32), hidden, lr, max(1, best_epoch))
        self.selected_ = {"hidden": hidden, "lr": lr, "epochs": best_epoch, "val_objective": val_obj}
        return self

    def decide(self, feature: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        x = self._prep(np.asarray(feature)[None, :])[0]
        if self.use_prev_weights:
            prev = torch.tensor(prev_weights, dtype=torch.float32)
            x = torch.cat([x, (prev - 1.0 / self.n_assets) * self.n_assets])
        with torch.no_grad():
            weights = self.model_(x[None, :]).cpu().numpy()[0]
        return project_to_capped_simplex(weights, self.w_max)
