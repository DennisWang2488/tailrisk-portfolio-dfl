from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .baselines import PortfolioMethod
from .optimizer import project_to_capped_simplex


class PortfolioNet(nn.Module):
    def __init__(self, n_features: int, n_assets: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


def _smooth_cvar(losses: torch.Tensor, alpha: float, tau: float) -> torch.Tensor:
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

    def fit(self, features: np.ndarray, returns: np.ndarray) -> "DecisionFocusedCVaRMethod":
        torch.manual_seed(self.seed)
        self.x_mean_ = features.mean(axis=0, keepdims=True)
        self.x_std_ = features.std(axis=0, keepdims=True) + 1e-8
        x = torch.tensor((features - self.x_mean_) / self.x_std_, dtype=torch.float32)
        r = torch.tensor(returns, dtype=torch.float32)
        self.model_ = PortfolioNet(self.n_features, self.n_assets, self.hidden)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        alpha = min(0.99, self.alpha + 0.025) if self.robust else self.alpha
        for _ in range(self.epochs):
            weights = self.model_(x)
            gross_returns = (weights * r).sum(dim=1)
            losses = -gross_returns
            cvar = _smooth_cvar(losses, alpha, self.smooth_tau)
            mean_term = -self.gamma * gross_returns.mean()
            turnover = torch.abs(weights[1:] - weights[:-1]).sum(dim=1).mean()
            cap_penalty = torch.tensor(0.0)
            if self.w_max is not None:
                cap_penalty = 10.0 * torch.relu(weights - self.w_max).pow(2).mean()
            objective = cvar + mean_term + self.transaction_cost * turnover + cap_penalty
            opt.zero_grad()
            objective.backward()
            opt.step()
        return self

    def decide(self, feature: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        x = torch.tensor((np.asarray(feature)[None, :] - self.x_mean_) / self.x_std_, dtype=torch.float32)
        with torch.no_grad():
            weights = self.model_(x).cpu().numpy()[0]
        return project_to_capped_simplex(weights, self.w_max)

