# When Does End-to-End Learning Help Tail-Risk Portfolio Construction?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

Simulation: decision-focused learning (train the forecast through the optimizer) vs predict-then-optimize (fit returns, then solve a CVaR LP). 95% CVaR portfolios.

I used synthetic returns so I can hand the same optimizer the true conditional distribution and score everything on regret against that oracle. A Sharpe that depends on which decade you sampled is not that useful here.

## Result

Two-stage has the lowest out-of-sample 95% CVaR and the lowest oracle regret in all five regimes. That includes the misspecified ones, which is where DFL is supposed to help. The two two-stage variants are 1st and 2nd of six methods in every cell.

Best two-stage vs best DFL, paired by seed (lower is better):

| Regime | true-mean linear R² | best two-stage | best DFL | ratio | paired *t* | seeds won |
|---|---:|---:|---:|---:|---:|:---:|
| `well_specified_high_sample` | 0.93 | **−0.00299** | 0.00086 | — | 8.42 | 5/5 |
| `heavy_tail_medium_sample` | 0.93 | **0.00796** | 0.01158 | 1.45× | 2.29 | 5/5 |
| `nonlinear_misspecified` | 0.45 | **0.00692** | 0.01108 | 1.60× | 4.94 | 5/5 |
| `predictable_tail_crash` | 0.80 | **0.01725** | 0.02167 | 1.26× | 5.19 | 5/5 |
| `high_cost_turnover_constrained` | 0.60 | **0.01817** | 0.02066 | 1.14× | 2.12 | 4/5 |

Means over 5 seeds. Per-seed rows are in [`outputs/research_grid_v2/results_by_seed.csv`](outputs/research_grid_v2/results_by_seed.csv). Oracle-regret ranking is the same everywhere (paired *t* = 2.02–9.46).

<p align="center">
  <img src="outputs/research_grid_v2/plots/test_cvar.png" width="49%" alt="Out-of-sample 95% CVaR by regime and method">
  <img src="outputs/research_grid_v2/plots/oracle_regret.png" width="49%" alt="Oracle regret by regime and method">
</p>

n = 500–900 periods, 25 assets. At that size the estimation error is just bigger than the “wrong training loss” problem. I wouldn’t read this as DFL being a bad idea in general.

## Things I had to fix

v1 said the same thing with much bigger gaps (“2–3× worse”). I went back through the config. Three bugs, two of which were helping that conclusion:

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | `dfl_smooth_tau = 0.01` was about the same size as the per-period loss sd | Only 6.8–41% of the DFL gradient landed on the true worst 5%. In four regimes the “CVaR” objective was basically a smeared mean — about 5.7× biased. | τ → `1e-4` (92.9–98.0% of the weight on the tail) |
| 2 | `misspecification_strength` scaled an L2-normalized loading vector | Nonlinear term ~5× weaker than I thought. The “misspecified” regimes had linear R² of 0.916 and 0.931. | strength → 8 / 15 / 5 (R² 0.45 / 0.80 / 0.60) |
| 3 | Oracle got 1200 scenarios, everyone else 300 | 60 tail scenarios vs 15, so “oracle regret” mixed estimation error with Monte Carlo error. | oracle → 300 |

Sortino was also exploding (~1e10) from a `/1e-12`, and `tail_loss_frequency` compared each series to its own quantile, so it was the same for every method.

Fixing τ improved DFL’s own test CVaR by 26–77%. Gap is now 1.14–1.60×. Ranking didn’t move.

I also reran the old v1 code as-is, 10 seeds, six (τ × misspec) settings. Two-stage still wins every regime. Smaller τ helps DFL (regret ratio 7.19 → 3.69 as τ goes 1e-2 → 1e-5 in the well-specified regime) but doesn’t flip anyone. Making the misspecification actually hard mostly just makes two-stage worse.

Full dump: [`RESULTS_v2.md`](RESULTS_v2.md).

## Methods

95% CVaR as a Rockafellar–Uryasev LP ([`optimizer.py`](src/tailrisk_dfl/optimizer.py)). Long-only, fully invested, 20% cap, mean-return tilt, ℓ₁ turnover penalty in the objective and again in the backtest, optional hard turnover constraint. HiGHS.

Everyone uses that same decision layer:

| method | |
|---|---|
| `equal_weight` | 1/N |
| `min_variance` | Ledoit–Wolf + SLSQP |
| `two_stage` | multi-output ridge → residual bootstrap → CVaR LP |
| `robust_two_stage` | same, mean shrinkage 0.6×, tail-tilted residual resampling |
| `dfl` | MLP → softmax, trained on softplus-smoothed CVaR |
| `robust_dfl` | same, trained at α + 0.025 |
| `oracle` | CVaR LP given the true conditional distribution |

[`synthetic.py`](src/tailrisk_dfl/synthetic.py) has the knobs: SNR, Student-t tails, skew, factor/block/crisis correlation, nonlinear or hidden-crash misspecification, crash probability, 5–50 bps costs.

## Run

```bash
pip install -e .
OMP_NUM_THREADS=1 python scripts/run_experiment.py --config configs/smoke.json --output outputs/smoke
```

Smoke is a few minutes, same code path. Full grid:

```bash
OMP_NUM_THREADS=1 python scripts/run_experiment.py --config configs/research_grid_v2.json --output outputs/research_grid_v2 --plots
pytest
```

Set `OMP_NUM_THREADS=1`. If you don’t, torch and numpy fight over OpenMP and it segfaults (exit 139). Old scikit-learn 1.0.x with scipy ≥ 1.11 will also blow up on `scipy.linalg.solve(sym_pos=)`.

## Caveats

- Synthetic only. You need the true conditional law for oracle regret, so this says nothing about real markets.
- One hyperparameter setting per method. I only corrected τ; no LR / width sweep for DFL.
- Two-stage gets `prev_weights` and the turnover constraint at decision time. DFL is stateless and only sees a turnover proxy in training, so don’t lean on `high_cost_turnover_constrained`.
- Two cells are weak: `high_cost` on CVaR (*t* = 2.12, 4/5 seeds) and `heavy_tail` on regret (*t* = 2.02, 3/5 seeds).
- The split is 75/25, not the 60/15/25 in the config. Validation gets concatenated into training ([`experiment.py`](src/tailrisk_dfl/experiment.py)) and never used. I left it; should spend it on DFL hyperparameters later.
- `predictable_tail_crash` can’t get much more nonlinear than R² ≈ 0.71–0.80. The tanh basis saturates.

```
src/tailrisk_dfl/   optimizer, DFL, generator, walk-forward, metrics
configs/            smoke.json, research_grid.json (v1), research_grid_v2.json
outputs/            committed results + plots
RESULTS_v2.md       current numbers
RESULTS_v1_superseded.md
research_notes.md
```

MIT, [LICENSE](LICENSE).
