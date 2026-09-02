from __future__ import annotations

import numpy as np
import torch

from tailrisk_dfl.dfl_lp import DifferentiableLPMethod, _build_layer
from tailrisk_dfl.optimizer import CVaROptimizerParams


def test_layer_gradient_is_nonzero_with_quad_reg() -> None:
    torch.manual_seed(0)
    layer = _build_layer(20, 5, 0.95, 0.2, 0.5, quad_reg=1e-3)
    scen = (0.01 * torch.randn(4, 20, 5)).requires_grad_(True)
    (w,) = layer(scen, solver_args={"solve_method": "Clarabel"})
    assert torch.allclose(w.sum(dim=1).float(), torch.ones(4), atol=1e-4)
    (w * scen[:, 0, :]).sum().backward()
    assert scen.grad.abs().sum() > 0


def test_dfl_lp_fits_and_decides() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 4))
    r = 0.005 * x @ rng.normal(size=(4, 6)) + 0.01 * rng.standard_t(5, size=(120, 6))
    params = CVaROptimizerParams(alpha=0.9, gamma=0.2, w_max=0.5)
    m = DifferentiableLPMethod(
        "dfl_lp", params, n_scenarios=40, seed=0, hidden=0,
        init_kwargs=dict(epochs=20, lr=0.01, patience=5), epochs=2, batch_size=64, train_scenarios=10, patience=2,
    )
    m.fit(x[:90], r[:90], x[90:], r[90:])
    w = m.decide(x[0], np.ones(6) / 6)
    assert w.shape == (6,) and abs(w.sum() - 1) < 1e-6 and (w >= -1e-9).all()
