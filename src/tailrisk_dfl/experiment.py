from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .baselines import EqualWeightMethod, MinVarianceMethod, TwoStageCVaRMethod
from .config import ExperimentConfig, RegimeConfig, config_to_dict
from .dfl import DecisionFocusedCVaRMethod
from .evaluation import backtest_method, compute_metrics, oracle_backtest, summarize_results
from .optimizer import CVaROptimizerParams
from .synthetic import SyntheticMarket


def _split_indices(config: RegimeConfig) -> tuple[slice, slice, slice]:
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
    methods = []
    for name in config.methods:
        if name == "equal_weight":
            methods.append(EqualWeightMethod(regime.n_assets))
        elif name == "min_variance":
            methods.append(MinVarianceMethod(regime.n_assets, opt.w_max, regime.turnover_limit))
        elif name == "two_stage":
            methods.append(TwoStageCVaRMethod(name, params, train.ridge_alpha, opt.n_scenarios, seed + 101))
        elif name == "robust_two_stage":
            methods.append(TwoStageCVaRMethod(name, params, train.ridge_alpha, opt.n_scenarios, seed + 202, robust=True))
        elif name == "dfl":
            methods.append(
                DecisionFocusedCVaRMethod(
                    regime.n_features,
                    regime.n_assets,
                    opt.alpha,
                    opt.gamma,
                    opt.w_max,
                    regime.transaction_cost,
                    train.dfl_hidden,
                    train.dfl_epochs,
                    train.dfl_lr,
                    train.dfl_weight_decay,
                    train.dfl_smooth_tau,
                    seed + 303,
                    robust=False,
                    name="dfl",
                )
            )
        elif name == "robust_dfl":
            methods.append(
                DecisionFocusedCVaRMethod(
                    regime.n_features,
                    regime.n_assets,
                    opt.alpha,
                    opt.gamma,
                    opt.w_max,
                    regime.transaction_cost,
                    train.dfl_hidden,
                    train.dfl_epochs,
                    train.dfl_lr,
                    train.dfl_weight_decay,
                    train.dfl_smooth_tau,
                    seed + 404,
                    robust=True,
                    name="robust_dfl",
                )
            )
        else:
            raise ValueError(f"unknown method: {name}")
    return methods, params


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for regime in config.regimes:
        for seed in config.seeds:
            market = SyntheticMarket(regime, seed)
            path = market.simulate()
            train_slice, val_slice, test_slice = _split_indices(regime)
            fit_slice = slice(train_slice.start, val_slice.stop)
            x_fit = path.features[fit_slice]
            r_fit = path.returns[fit_slice]
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
                method.fit(x_fit, r_fit)
                result = backtest_method(method, x_test, r_test, regime.transaction_cost)
                metrics = compute_metrics(result, config.optimizer.alpha, config.optimizer.gamma, oracle)
                metrics.update(
                    {
                        "experiment": config.experiment_name,
                        "regime": regime.name,
                        "seed": seed,
                    }
                )
                rows.append(metrics)

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    results.to_csv(output / "results_by_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "config_resolved.json").write_text(json.dumps(config_to_dict(config), indent=2), encoding="utf-8")
    return results, summary

