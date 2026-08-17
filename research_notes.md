# Research Notes - DFL vs. Predict-Then-Optimize for CVaR Portfolios

This file is the running log for the auto-research loop. Each agent iteration
appends a block below. Do not overwrite prior blocks.

See `SESSION_PROMPT.md` for the full research mandate, task hierarchy, and
iteration format requirements.

---

## Baseline results (pre-loop, human-seeded)

**Date:** 2026-04-23
**Config:** `configs/smoke.json` (2 regimes, 2 seeds - `predictable_crash_small`, `well_specified_small`)

### Key numbers

| regime | method | test_cvar | oracle_regret | tc_adj_return | cvar_win_rate | regret_win_rate |
|---|---|---|---|---|---|---|
| well_specified_small | two_stage | **0.00325** | 0.00105 | 0.00661 | **100%** | **100%** |
| well_specified_small | robust_two_stage | 0.00567 | 0.00093 | 0.00450 | 0% | 0% |
| well_specified_small | dfl | 0.00614 | 0.00062 | 0.00576 | 0% | 0% |
| well_specified_small | robust_dfl | 0.00604 | 0.00159 | 0.00577 | 0% | 0% |
| well_specified_small | min_variance | 0.00686 | 0.00131 | 0.00117 | 0% | 0% |
| well_specified_small | equal_weight | 0.00839 | 0.00074 | 0.00120 | 0% | 0% |
| predictable_crash_small | robust_two_stage | **0.03260** | -0.00076 | -0.00549 | 50% | 50% |
| predictable_crash_small | two_stage | 0.03210 | -0.00134 | -0.00507 | 50% | 50% |
| predictable_crash_small | min_variance | 0.03509 | 0.00019 | -0.00640 | 0% | 0% |
| predictable_crash_small | dfl | 0.03532 | 0.00186 | -0.00493 | 0% | 0% |
| predictable_crash_small | robust_dfl | 0.03859 | 0.00417 | -0.00619 | 0% | 0% |
| predictable_crash_small | equal_weight | 0.04868 | 0.00123 | -0.00854 | 0% | 0% |

### Initial observations
1. **two_stage dominates well_specified_small** across all metrics (100% win rate on CVaR and oracle regret). This is expected - the model is correctly specified, linear, and two-stage ridge is near-Bayes-optimal here.
2. **DFL underperforms** in both smoke regimes. Whether this is structural or a training issue (only 2 seeds, smoke-scale data) is the key Q2 question.
3. **predictable_crash_small** sees two-stage variants tied; DFL/robust_dfl worse. The crash regime with hidden misspecification should be where DFL's end-to-end training gives an edge - but it doesn't show here at small scale.
4. **Negative oracle regret** for two_stage/robust_two_stage in predictable_crash means they beat the oracle in-sample - likely an artefact of small sample variance.

### Priority for loop iteration 1
1. Run `configs/research_grid.json` (full 5-regime, 5-seed grid) - this is the most important single action.
2. Analyse the results and update this file with regime-by-regime conclusions.
3. Begin `FINDINGS.md` scaffold once results are in.

---

<!-- Agent iterations appended below -->
