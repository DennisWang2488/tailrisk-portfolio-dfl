# v2 results — after the adversarial audit (2026-08-03)

`configs/research_grid_v2.json` · same grid as v1 (25 assets, 8 features, 5 regimes × 5 seeds ×
6 methods, 95% CVaR, 60/15/25 split*) with three fixes from an adversarial code audit, plus two
metric repairs. Raw numbers: `outputs/research_grid_v2/`.

## What changed and why

| Fix | v1 → v2 | Audit finding |
|---|---|---|
| `dfl_smooth_tau` | 0.01 → **1e-4** | τ was the same order as per-period loss sd, so ~90% of the DFL gradient landed outside the tail; the smoothed objective was a 5.7× biased estimate of CVaR |
| `oracle_scenarios` | 1200 → **300** | oracle had 60 tail scenarios vs. competitors' 15; oracle regret conflated estimation error with Monte Carlo error |
| `misspecification_strength` | 0.7/1.0/0.5 → **8/15/5** | row-wise vs. global normalization made the nonlinear term ~5× weaker than nominal; the "misspecified" regimes were 91–93% linearly explainable. Now: nonlinear R²=0.45, crash R²=0.80 (tanh saturates — structural ceiling), high-cost R²=0.60 |
| Sortino | fixed | was np.std of negative returns only, with a /1e-12 divide-by-zero producing ~1e10 values |
| `tail_loss_frequency` | fixed | was each series' own-quantile exceedance = ceil(0.05n)/n for ANY series; now measured against the oracle portfolio's VaR (fixed, shared threshold) |

\* The advertised 60/15/25 split concatenates the validation block into training (see
`experiment.py`); it is effectively 75/25 and the validation block is unused. Not changed in v2 —
flagged for a v3 that uses it for DFL hyperparameter selection.

## Headline: the conclusion survives, the margin honestly shrinks

**Estimate-then-optimize attains the lowest out-of-sample 95% CVaR AND the lowest oracle regret
in 5/5 regimes — including under genuine misspecification.**

Test CVaR, best two-stage vs. best DFL variant, per regime:

| Regime | true-mean linear R² | best TS | best DFL | ratio | paired t | seeds |
|---|---|---|---|---|---|---|
| well_specified_high_sample | 0.93 | −0.00299 | 0.00086 | — | 8.42 | 5/5 |
| heavy_tail_medium_sample | 0.93 | 0.00796 | 0.01158 | 1.45× | 2.29 | 5/5 |
| nonlinear_misspecified | **0.45** | 0.00692 | 0.01108 | 1.60× | 4.94 | 5/5 |
| predictable_tail_crash | 0.80 | 0.01725 | 0.02167 | 1.26× | 5.19 | 5/5 |
| high_cost_turnover_constrained | 0.60 | 0.01817 | 0.02066 | 1.14× | 2.12 | 4/5 |

Oracle regret: two-stage best in 5/5 regimes, paired t = 2.0–9.5 (weakest cell: heavy-tail, 3/5
seeds, t = 2.02).

## What the fixes changed

1. **The τ fix helped DFL a lot — and it still loses.** DFL's own test CVaR improved 26–77% in
   four regimes (76.8% in well-specified). The v1 "2–3× worse" gaps were partly an artifact of a
   mis-scaled smoothing parameter; the honest v2 gaps are 1.14–1.60×. This strengthens the paper's
   claim: the comparison is now against a *competent* end-to-end learner.
2. **Genuine misspecification did not rescue DFL.** At linear R² = 0.45 — the regime end-to-end
   training is supposed to win — two-stage still wins 5/5 seeds at t = 4.94. Estimation-error
   control still dominates at n = 500.
3. **Two-stage is slightly worse in absolute terms in the now-harder regimes** (e.g. nonlinear:
   0.00426 → 0.00692) — as it should be: its model is now actually wrong. The gap narrows but does
   not close, which is exactly the "estimation error vs. objective alignment" tradeoff the study
   is about.
4. **Caveats that remain**: single hyperparameter setting per method (only τ corrected — no lr /
   width sweep for DFL); DFL policy is stateless while the LP receives `prev_weights` and the
   turnover constraint (structural asymmetry in the high-cost regime — treat that regime's
   comparison as indicative, not apples-to-apples); tanh crash regime has a structural R² floor
   ~0.71; two regimes sit at paired t ≈ 2.1–2.3 (p ≈ 0.08–0.10 at n=5 seeds) — describe those as
   "consistent direction" rather than individually significant.

## Reproduce

```bash
python scripts/run_experiment.py --config configs/research_grid_v2.json --output outputs/research_grid_v2 --plots
```
