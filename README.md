# When Does End-to-End Learning Help Tail-Risk Portfolio Construction?

**Short answer, on this evidence: it doesn't — and the interesting part is what it took to be sure.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green.svg)

A controlled simulation study comparing **decision-focused learning** (DFL — train the forecaster
through the optimizer, directly on decision loss) against **predict-then-optimize** (fit a return
model, then solve a CVaR program) for 95% CVaR portfolio construction.

The design goal was deliberately *not* to make either side win. It was to build a setting where an
**oracle** — an optimizer handed the true conditional return distribution — is computable, so that
every method can be scored on *regret against a perfect forecast* rather than on a Sharpe ratio
that depends on which decade you sampled.

---

## The result

Estimate-then-optimize attains **both** the lowest out-of-sample 95% CVaR **and** the lowest oracle
regret in **5 of 5 regimes** — including the two regimes constructed to favour end-to-end training.
Across both metrics and all five regimes, the two two-stage variants take the **top two slots** of
six methods, every time.

Best two-stage variant vs. best DFL variant, paired by seed (lower is better):

| Regime | true-mean linear R² | best two-stage | best DFL | ratio | paired *t* | seeds won |
|---|---:|---:|---:|---:|---:|:---:|
| `well_specified_high_sample` | 0.93 | **−0.00299** | 0.00086 | — | 8.42 | 5/5 |
| `heavy_tail_medium_sample` | 0.93 | **0.00796** | 0.01158 | 1.45× | 2.29 | 5/5 |
| `nonlinear_misspecified` | **0.45** | **0.00692** | 0.01108 | 1.60× | 4.94 | 5/5 |
| `predictable_tail_crash` | 0.80 | **0.01725** | 0.02167 | 1.26× | 5.19 | 5/5 |
| `high_cost_turnover_constrained` | 0.60 | **0.01817** | 0.02066 | 1.14× | 2.12 | 4/5 |

<sub>Out-of-sample 95% CVaR, mean over 5 seeds. Per-seed rows: [`outputs/research_grid_v2/results_by_seed.csv`](outputs/research_grid_v2/results_by_seed.csv).
On oracle regret the ranking is identical in all five regimes, paired *t* = 2.02–9.46.</sub>

<p align="center">
  <img src="outputs/research_grid_v2/plots/test_cvar.png" width="49%" alt="Out-of-sample 95% CVaR by regime and method">
  <img src="outputs/research_grid_v2/plots/oracle_regret.png" width="49%" alt="Oracle regret by regime and method">
</p>

**Read it as:** at n = 500–900 periods with 25 assets, **estimation error dominates objective
misalignment**. The end-to-end gradient signal through a smoothed CVaR argmin is too noisy to pay
for itself at this sample size. That is a claim about a sample-size regime, not a claim that DFL is
a bad idea.

---

## The audit

The first version of this study reported the same *direction* with much larger margins — "2–3×
worse". Before writing any of it up, I attacked my own configuration. Three defects turned up, and
**two of them were rigging the comparison in favour of my own conclusion**:

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | `dfl_smooth_tau = 0.01` was the same order as the per-period loss sd | The softplus transition was so wide that only **6.8–41% of the DFL gradient landed on the true worst 5%**. The "CVaR" objective was, in four of five regimes, a smeared mean loss — a 5.7× biased estimate of CVaR. DFL was losing to a mis-scaled hyperparameter. | τ → `1e-4`, putting 92.9–98.0% of gradient weight on the tail |
| 2 | `misspecification_strength` scaled an L2-normalized loading vector | The nonlinear term came out ~5× weaker than nominal. The regimes labelled "misspecified" had **true-mean linear R² of 0.916 and 0.931** — linear regimes with a misleading name, so the headline "including under misspecification" was unsupported. | strength → 8 / 15 / 5, giving genuine R² of 0.45 / 0.80 / 0.60 |
| 3 | Oracle solved with 1200 scenarios against competitors' 300 | The oracle saw 60 tail scenarios to everyone else's 15, so "oracle regret" conflated estimation error with Monte Carlo error. | oracle → 300, matched |

Plus two metric repairs: Sortino had a `/1e-12` divide-by-zero emitting ~1e10 values, and
`tail_loss_frequency` measured each series against its *own* quantile — `ceil(0.05n)/n` for any
series whatsoever, hence identical across methods by construction and entirely uninformative.

**What changed.** Fixing τ improved DFL's own test CVaR by **26–77%** across four regimes (76.8% in
the well-specified one). The margin collapsed from "2–3×" to **1.14–1.60×**. The ranking did not
move in a single regime.

**Why this strengthens the finding rather than weakening it.** The comparison now runs against a
*competent* end-to-end learner, in regimes that are *actually* misspecified, against a fairly
resourced oracle. At linear R² = 0.45 — precisely the setting end-to-end training is supposed to
own — two-stage still wins 5/5 seeds at *t* = 4.94.

<details>
<summary><b>Robustness: an independent re-run from the original artifact — click to expand</b></summary>

<br>

To check that the conclusion was not an artifact of the fixes themselves, the study was re-run from
the **unmodified v1 code**, driven externally, at **10 seeds** (not 5), over a 6-arm grid crossing
the smoothing parameter with misspecification strength:

| arm | τ | misspec. | regimes won by two-stage (regret) | (test CVaR) | seed-level |
|---|---|---|:---:|:---:|:---:|
| `A0_base` | 1e-2 | shipped | 5/5 | 5/5 | 50/50 |
| `T3` | 1e-3 | shipped | 5/5 | 5/5 | 50/50 |
| `T4` | 1e-4 | shipped | 5/5 | 5/5 | 48/50 |
| `T5` | 1e-5 | shipped | 5/5 | 5/5 | 47/50 |
| `M_hard` | 1e-2 | hardened | 5/5 | 5/5 | 50/50 |
| `MT_both` | 1e-4 | hardened | 5/5 | 5/5 | 48/50 |

Paired *t* on oracle regret ranges 4.06–12.65 with both defects fixed, all *p* ≤ 0.0028.

The nuance that matters: **sharpening τ improves DFL monotonically** — the oracle-regret ratio goes
7.19 → 5.45 → 4.14 → 3.69 as τ goes 1e-2 → 1e-5 in the well-specified regime — and never flips the
ranking in the range tested. So τ = 1e-4 is not a cherry-pick, but the trend is real, and it is the
first thing a sceptical reader should push on.

Hardening the misspecification narrows the gap mostly by *degrading two-stage* (nonlinear-regime
regret 0.00208 → 0.00357): the regime genuinely got harder, DFL just failed to exploit it.

</details>

---

## What's in the box

**Optimizer** — 95% CVaR as a Rockafellar–Uryasev linear program
([`optimizer.py`](src/tailrisk_dfl/optimizer.py)): long-only, fully invested, 20% position cap,
mean-return tilt γ, ℓ₁ turnover penalty charged in the objective *and* re-charged in the backtest,
optional hard turnover constraint. Solved with HiGHS; weights projected onto the capped simplex.

**Six methods on one decision layer**, so differences are attributable to the training loss and
nothing else:

| method | what it is |
|---|---|
| `equal_weight` | 1/N — the null that beats most things |
| `min_variance` | Ledoit–Wolf shrinkage covariance + SLSQP |
| `two_stage` | multi-output ridge → residual bootstrap scenarios → CVaR LP |
| `robust_two_stage` | same, with mean shrinkage (0.6×) and tail-tilted residual resampling |
| `dfl` | MLP → softmax weights, trained on a **softplus-smoothed CVaR** surrogate, backprop through the objective |
| `robust_dfl` | same, trained at a stricter α (+0.025) |
| `oracle` | the CVaR LP handed the **true** conditional distribution — the regret denominator |

**Generator** ([`synthetic.py`](src/tailrisk_dfl/synthetic.py)) — configurable signal-to-noise,
Student-*t* tails, skew, factor / block / crisis correlation structure, nonlinear or hidden-crash
misspecification, crash probability, and trading frictions from 5 to 50 bps.

---

## Reproduce

```bash
pip install -e .
```

```bash
OMP_NUM_THREADS=1 python scripts/run_experiment.py --config configs/smoke.json --output outputs/smoke
```

The smoke config finishes in a couple of minutes and exercises the same code path. The full grid
behind the table above:

```bash
OMP_NUM_THREADS=1 python scripts/run_experiment.py --config configs/research_grid_v2.json --output outputs/research_grid_v2 --plots
```

```bash
pytest
```

> **`OMP_NUM_THREADS=1` is not optional.** Without it, torch and numpy contend over OpenMP threads
> and the run segfaults (exit 139) partway through the grid. On `scikit-learn` 1.0.x with `scipy`
> ≥ 1.11 you will also hit the removed `scipy.linalg.solve(sym_pos=)` argument — use a modern
> scikit-learn rather than shimming it.

---

## What this does **not** show

Stated plainly, because the caveats are load-bearing:

- **Synthetic data throughout.** Oracle regret requires the true conditional law, which is exactly
  why the study is simulated — you cannot measure it on historical prices. Nothing here is evidence
  about real markets.
- **One hyperparameter setting per method.** Only τ was corrected; there is no learning-rate or
  width sweep for DFL. A tuned DFL might narrow the gap further. The gap is monotone in τ and did
  not cross over in the range tested, but that range is finite.
- **A structural asymmetry favouring the LP.** The two-stage methods receive `prev_weights` and the
  turnover constraint at decision time; the DFL policy is stateless and sees only a turnover *proxy*
  during training. Treat `high_cost_turnover_constrained` as indicative, not apples-to-apples.
- **Two cells are weak.** `high_cost` on CVaR (*t* = 2.12, 4/5 seeds) and `heavy_tail` on regret
  (*t* = 2.02, 3/5 seeds) are "consistent in direction", not individually significant at 5 seeds.
- **The split is really 75/25.** The config advertises 60/15/25, but the validation block is
  concatenated into training and never used ([`experiment.py`](src/tailrisk_dfl/experiment.py)).
  Flagged rather than quietly fixed — a v3 should spend it on DFL hyperparameter selection.
- `predictable_tail_crash` has a **structural R² floor** near 0.71–0.80: the tanh basis saturates,
  so that regime cannot be pushed to arbitrary nonlinearity.

---

## Repository map

```
src/tailrisk_dfl/
  synthetic.py     configurable return generator (tails, skew, correlation, misspecification)
  optimizer.py     Rockafellar–Uryasev CVaR LP, min-variance, capped-simplex projection
  baselines.py     equal weight, min-variance, two-stage, robust two-stage
  dfl.py           smoothed-CVaR end-to-end learner
  experiment.py    walk-forward runner over regimes × seeds × methods
  evaluation.py    test CVaR, oracle regret, turnover, drawdown, tail-loss frequency
configs/           smoke.json · research_grid.json (v1) · research_grid_v2.json (v2)
outputs/           committed result sets for both grids, plus plots
RESULTS_v2.md              current numbers — cite these
RESULTS_v1_superseded.md   pre-audit numbers, kept deliberately and clearly marked
research_notes.md          iteration log
```

## License

MIT — see [LICENSE](LICENSE).
