from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .baselines import EqualWeightMethod, MinVarianceMethod, TwoStageCVaRMethod
from .config import ExperimentConfig, RegimeConfig, config_to_dict
from .dfl import DecisionFocusedCVaRMethod
from .dfl_lp import DifferentiableLPMethod
from .evaluation import backtest_method, compute_metrics, oracle_backtest, summarize_results
from .optimizer import CVaROptimizerParams
from .synthetic import SyntheticMarket


def _split_indices(config: RegimeConfig) -> tuple[slice, slice, slice]:
    # config says 60/15/25. we don't actually use the val block — see fit_slice
    # below. left it in so a later run can pick DFL hyperparameters on it.
    train_end = int(config.n_periods * config.train_fraction)
    val_end = train_end + int(config.n_periods * config.validation_fraction)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, config.n_periods)


def _make_methods(config: ExperimentConfig, regime: RegimeConfig, seed: int):
    opt = config.optimizer
    train = config.training
    params = CVaROptimizerParams(
        alpha=opt.alpha,
        gamma=opt.gamma,
        w_max=opt.w_max,
        turnover_penalty=opt.turnover_penalty,
        turnover_limit=regime.turnover_limit,
    )
    dfl_common = dict(
        n_features=regime.n_features,
        n_assets=regime.n_assets,
        alpha=opt.alpha,
        gamma=opt.gamma,
        w_max=opt.w_max,
        transaction_cost=regime.transaction_cost,
        hidden=train.dfl_hidden,
        epochs=train.dfl_epochs,
        lr=train.dfl_lr,
        weight_decay=train.dfl_weight_decay,
        smooth_tau=train.dfl_smooth_tau,
        early_stopping=train.use_validation,
        patience=train.dfl_patience,
        lr_grid=list(train.dfl_lr_grid),
        hidden_grid=list(train.dfl_hidden_grid),
    )
    mlp_kwargs = dict(
        hidden=train.dfl_hidden,
        epochs=train.dfl_epochs,
        lr=train.dfl_lr,
        weight_decay=train.dfl_weight_decay,
        patience=train.dfl_patience,
        lr_grid=list(train.dfl_lr_grid) or None,
        hidden_grid=list(train.dfl_hidden_grid) or None,
    )
    methods = []
    for name in config.methods:
        if name == "equal_weight":
            methods.append(EqualWeightMethod(regime.n_assets))
        elif name == "min_variance":
            methods.append(MinVarianceMethod(regime.n_assets, opt.w_max, regime.turnover_limit))
        elif name == "two_stage":
            methods.append(TwoStageCVaRMethod(name, params, train.ridge_alpha, opt.n_scenarios, seed + 101))
        elif name == "robust_two_stage":
            # +101 / +202 / +303 so the methods don't share an rng stream
            methods.append(TwoStageCVaRMethod(name, params, train.ridge_alpha, opt.n_scenarios, seed + 202, robust=True))
        elif name == "mlp_two_stage":
            # 2x2 ablation, "model class" arm: MLP forecaster + the same LP
            methods.append(
                TwoStageCVaRMethod(name, params, train.ridge_alpha, opt.n_scenarios, seed + 505, regressor="mlp", mlp_kwargs=mlp_kwargs)
            )
        elif name == "dfl":
            methods.append(DecisionFocusedCVaRMethod(seed=seed + 303, robust=False, name="dfl", **dfl_common))
        elif name == "robust_dfl":
            methods.append(DecisionFocusedCVaRMethod(seed=seed + 404, robust=True, name="robust_dfl", **dfl_common))
        elif name == "linear_dfl":
            # 2x2 ablation, "objective" arm: ridge-sized model, CVaR objective
            kw = {**dfl_common, "hidden": 0, "hidden_grid": []}
            methods.append(DecisionFocusedCVaRMethod(seed=seed + 606, robust=False, name="linear_dfl", **kw))
        elif name in ("dfl_lp", "mlp_dfl_lp"):
            # true DFL: forecaster fine-tuned through the CVaR LP (cvxpylayers),
            # warm-started from the MSE fit. linear (dfl_lp) or MLP (mlp_dfl_lp).
            lp_kwargs = dict(
                epochs=train.dfl_lp_epochs,
                lr=train.dfl_lp_lr,
                weight_decay=train.dfl_weight_decay,
                batch_size=train.dfl_lp_batch_size,
                train_scenarios=train.dfl_lp_train_scenarios,
                smooth_tau=train.dfl_smooth_tau,
                patience=train.dfl_lp_patience,
                early_stopping=train.use_validation,
                quad_reg=train.dfl_lp_quad_reg,
                lr_grid=list(train.dfl_lp_lr_grid) or None,
            )
            hidden = 0 if name == "dfl_lp" else train.dfl_hidden
            init_kwargs = {**mlp_kwargs}
            init_kwargs.pop("hidden")
            if name == "dfl_lp":
                init_kwargs["hidden_grid"] = None
            methods.append(
                DifferentiableLPMethod(name, params, opt.n_scenarios, seed + (808 if name == "dfl_lp" else 909), hidden, init_kwargs, **lp_kwargs)
            )
        elif name == "dfl_stateful":
            # sees w_{t-1} like the LP does; removes the turnover confound
            methods.append(
                DecisionFocusedCVaRMethod(seed=seed + 707, robust=False, name="dfl_stateful", use_prev_weights=True, **dfl_common)
            )
        else:
            raise ValueError(f"unknown method: {name}")
    return methods, params


def _selected(method) -> dict:
    sel = getattr(method, "selected_", None)
    if sel is None:
        inner = getattr(method, "model_", None)
        sel = getattr(inner, "selected_", None)
    return sel or {}


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for regime in config.regimes:
        for seed in config.seeds:
            market = SyntheticMarket(regime, seed)
            path = market.simulate()
            train_slice, val_slice, test_slice = _split_indices(regime)
            if config.training.use_validation:
                x_fit, r_fit = path.features[train_slice], path.returns[train_slice]
                x_val, r_val = path.features[val_slice], path.returns[val_slice]
            else:
                fit_slice = slice(train_slice.start, val_slice.stop)  # train+val. val is unused (v1/v2 behaviour).
                x_fit, r_fit = path.features[fit_slice], path.returns[fit_slice]
                x_val, r_val = None, None
            x_test = path.features[test_slice]
            r_test = path.returns[test_slice]
            methods, params = _make_methods(config, regime, seed)
            oracle = oracle_backtest(
                market,
                x_test,
                r_test,
                params,
                regime.transaction_cost,
                config.optimizer.oracle_scenarios,
                seed + 999,
            )
            for method in methods:
                method.fit(x_fit, r_fit, x_val, r_val)
                result = backtest_method(method, x_test, r_test, regime.transaction_cost)
                metrics = compute_metrics(result, config.optimizer.alpha, config.optimizer.gamma, oracle)
                metrics.update(
                    {
                        "experiment": config.experiment_name,
                        "regime": regime.name,
                        "seed": seed,
                        "n_periods": regime.n_periods,
                        "selected": json.dumps(_selected(method)),
                    }
                )
                rows.append(metrics)

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    results.to_csv(output / "results_by_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "config_resolved.json").write_text(json.dumps(config_to_dict(config), indent=2), encoding="utf-8")
    return results, summary

