> # ⚠️ SUPERSEDED — do not cite these numbers
>
> This is the **v1** result set, kept in the repository on purpose. An adversarial
> self-audit found three defects in the configuration that produced it, and the
> magnitudes below are wrong as a result:
>
> 1. `dfl_smooth_tau = 0.01` was the same order as the per-period loss standard
>    deviation, so only 7–41% of the DFL gradient landed on the tail it was
>    supposed to be optimizing. **DFL was crippled by a mis-scaled hyperparameter,
>    not by its own merits.**
> 2. The oracle was solved with 1200 scenarios against competitors' 300, so
>    "oracle regret" mixed estimation error with Monte Carlo error.
> 3. The `misspecification_strength` knob was ~5× weaker than nominal, so the
>    regimes labelled "misspecified" were 91–93% linearly explainable — they were
>    **linear regimes with a misleading name**.
>
> The direction of the finding survived all three fixes; the **"2–3× worse" gaps
> in the table below did not** — they are 1.14–1.60× once DFL is given a
> correctly scaled objective. See **[RESULTS_v2.md](RESULTS_v2.md)** for the
> corrected numbers and [README.md](README.md#the-audit) for the full story.

# Headline results — `research_grid` (v1, superseded)

Config: `configs/research_grid.json` · 25 assets, 8 features, 5 regimes × 5 seeds × 6 methods,
95% CVaR objective, 20% position cap, turnover penalty 0.001, 300 scenarios (oracle: 1200),
60/15/25 train/validation/test split. Raw numbers: `outputs/research_grid/summary.csv`.

Lower is better for test CVaR and oracle regret.

| Regime | Best (test CVaR) | CVaR | Best DFL variant | CVaR | Gap |
|---|---|---:|---|---:|---:|
| `well_specified_high_sample` | two_stage | −0.00299 | dfl | 0.00416 | 2.4× worse |
| `heavy_tail_medium_sample` | robust_two_stage | 0.00796 | robust_dfl | 0.01678 | 2.1× |
| `nonlinear_misspecified` | robust_two_stage | 0.00426 | dfl | 0.01247 | 2.9× |
| `predictable_tail_crash` | robust_two_stage | 0.01668 | robust_dfl | 0.03280 | 2.0× |
| `high_cost_turnover_constrained` | robust_two_stage | 0.01704 | robust_dfl | 0.03253 | 1.9× |

## What the grid says

1. **Estimate-then-optimize wins in 5/5 regimes**, on both test CVaR and oracle regret. The robust
   two-stage variant takes CVaR in 4/5; plain two-stage wins the well-specified high-sample regime.
2. **The decision-focused learners never win**, including in the two regimes designed to favor them —
   `nonlinear_misspecified` (misspecification strength 0.7) and `predictable_tail_crash` (hidden crash
   structure, 5% crash probability, crisis correlation). This is the opposite of the usual DFL argument.
3. **Robustness helps exactly where theory says it should.** Robust two-stage beats plain two-stage
   everywhere except the well-specified high-sample regime, where the extra conservatism only costs.
4. Read as: at n = 500–900 periods with 25 assets, **estimation error dominates objective misalignment**.
   The end-to-end gradient signal through a smoothed CVaR argmin is too noisy to pay for itself at this
   sample size.

## Caveats a reader should hold

- **Synthetic data throughout.** The oracle baseline requires the true conditional law, which is why the
  study is simulated — oracle regret is not measurable on historical prices. It also means these results
  do not establish anything about real markets.
- Single hyperparameter setting per method (`dfl_epochs=250`, `hidden=64`, `lr=5e-3`, `smooth_tau=0.01`).
  A tuned DFL might close some of the gap; nobody ran that sweep.
- 5 seeds per regime. Standard errors are in `summary.csv` — several gaps are large relative to SEM,
  but the per-regime n is small.
- Long-only, single-period, no shorting, no leverage, no financing costs.

## Reproduce

```bash
python scripts/run_experiment.py --config configs/research_grid.json --output outputs/research_grid --plots
```

The smoke config (`configs/smoke.json`) runs in a couple of minutes and exercises the same code path.
