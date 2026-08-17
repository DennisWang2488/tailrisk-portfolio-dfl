from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    n_assets: int = 25
    n_features: int = 8
    n_periods: int = 500
    train_fraction: float = 0.6
    validation_fraction: float = 0.15  # advertised, not used. see experiment.py
    snr: float = 0.5
    tail_df: float = 6.0
    skew: float = 0.0
    correlation: str = "factor"
    misspecification: str = "none"
    misspecification_strength: float = 0.0
    crash_prob: float = 0.0
    transaction_cost: float = 0.001
    turnover_limit: float | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    alpha: float = 0.95
    gamma: float = 0.2
    w_max: float | None = 0.2
    turnover_penalty: float = 0.001
    n_scenarios: int = 300
    oracle_scenarios: int = 1200  # v1 default. v2 json sets this to 300. don't use this as-is.


@dataclass(frozen=True)
class TrainingConfig:
    ridge_alpha: float = 1.0
    dfl_epochs: int = 250
    dfl_hidden: int = 64
    dfl_lr: float = 0.005
    dfl_weight_decay: float = 1e-4
    dfl_smooth_tau: float = 0.01  # also a v1 leftover. research_grid_v2.json has 1e-4.


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    seeds: list[int] = field(default_factory=lambda: [0])
    regimes: list[RegimeConfig] = field(default_factory=list)
    methods: list[str] = field(
        default_factory=lambda: [
            "equal_weight",
            "min_variance",
            "two_stage",
            "robust_two_stage",
            "dfl",
            "robust_dfl",
        ]
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    regimes = [RegimeConfig(**item) for item in payload.get("regimes", [])]
    optimizer = OptimizerConfig(**payload.get("optimizer", {}))
    training = TrainingConfig(**payload.get("training", {}))
    return ExperimentConfig(
        experiment_name=payload["experiment_name"],
        seeds=list(payload.get("seeds", [0])),
        regimes=regimes,
        methods=list(payload.get("methods", [])),
        optimizer=optimizer,
        training=training,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    def convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return convert(config)

