# v4: a DFL that actually differentiates through the LP

v3 showed the v2 gap sits on the objective axis: a softmax policy trained on a
smoothed CVaR loses to MSE + bootstrap + LP even with the same linear capacity.
But that softmax policy never touches the optimizer, so v3 could not say
whether *decision-focused learning* loses, or just *that particular way of
doing it*. v4 adds the real thing.

## Method

`dfl_lp` (linear forecaster) and `mlp_dfl_lp` (same MLP as `dfl`):

```
x --f_θ--> μ(x) --(+ bootstrap residuals)--> 50 scenarios --CVaR LP layer--> w
loss = smoothed 95% CVaR of realized w·r over a 256-row minibatch  −  γ·mean
```

- The LP layer is the Rockafellar–Uryasev LP from `optimizer.py` rebuilt in
  cvxpylayers (Clarabel), with a small quadratic term `1e-3·‖w‖²` in the
  training layer only. Without it the LP argmin is piecewise constant in μ and
  the gradient norm is ~1e-6; with it ~1e-2 and the training loss falls
  steadily (QPTL, Wilder et al. 2019).
- f_θ is warm-started from the MSE fit, i.e. from `two_stage` / `mlp_two_stage`.
  Residuals for the bootstrap come from that fit and are held fixed, with a
  fixed scenario draw per row (common random numbers) so the objective is
  deterministic in θ.
- Fine-tune up to 30 epochs, lr ∈ {1e-3, 5e-3} and epoch chosen on the
  validation block, refit on train+val. If validation never improves on the
  MSE init, the selected epoch is 0 and the method *is* two-stage.
- At decision time: the exact HiGHS LP `two_stage` uses, 300 scenarios,
  previous weights, turnover penalty. Only the forecaster differs.

So `dfl_lp` vs `two_stage` isolates one thing: was the forecaster fine-tuned
on decision loss or not. It is "estimate-then-optimize init, integrated
fine-tune", which is the recipe the theory papers recommend.

## Result 1: through-the-LP DFL recovers two-stage, and crushes the softmax policy

Test CVaR, 5 seeds, same paths as v2/v3. Lower is better.

| regime | two_stage | dfl_lp | ratio | *t* | wins | linear_dfl (v3) |
|---|---:|---:|---:|---:|:---:|---:|
| well_specified_high_sample | −0.00299 | −0.00271 | 0.91 | 0.8 | 2/5 | 0.00107 |
| heavy_tail_medium_sample | 0.00849 | 0.00828 | 0.98 | −0.4 | 3/5 | 0.01299 |
| nonlinear_misspecified | 0.00739 | 0.00729 | 0.99 | −0.3 | 3/5 | 0.00945 |
| predictable_tail_crash | 0.01817 | 0.01913 | 1.05 | 1.5 | 1/5 | 0.03258 |
| high_cost_turnover_constrained | 0.01829 | 0.01789 | 0.98 | −1.1 | 3/5 | 0.03112 |

`dfl_lp` is statistically indistinguishable from `two_stage` everywhere
(|*t*| ≤ 1.5), ranks 1st in three regimes on mean CVaR, and is 0.57–0.77× the
softmax-policy `linear_dfl` with 5/5 seeds in every regime. Same linear
capacity, same CVaR objective, same data. The v2/v3 "DFL loses by 1.3–1.8×"
result was about the softmax policy, not about training on decision loss.

Why the difference: the softmax policy learns 25 weights per row from the
~5% tail rows alone. `dfl_lp` starts from a forecaster that used every row,
and the LP supplies the constraint structure (simplex, cap, tail averaging)
for free, so the gradient only has to move μ a little.

## Result 2: with the MLP, fine-tuning through the LP does not help at n ≈ 500

| regime | mlp_two_stage | mlp_dfl_lp | ratio | *t* | wins |
|---|---:|---:|---:|---:|:---:|
| well_specified_high_sample | −0.00226 | −0.00247 | 1.09 | −1.2 | 4/5 |
| heavy_tail_medium_sample | 0.00865 | 0.01111 | 1.29 | 3.5 | 0/5 |
| nonlinear_misspecified | 0.00781 | 0.00807 | 1.03 | 0.6 | 2/5 |
| predictable_tail_crash | 0.01886 | 0.01948 | 1.03 | 0.9 | 2/5 |
| high_cost_turnover_constrained | 0.01769 | 0.01922 | 1.09 | 6.9 | 0/5 |

Neutral in three regimes, clearly worse in the heavy-tail and high-cost
regimes. An MLP has enough freedom to move μ a lot in the direction the
~20-row training tail wants, and the 75-row validation block (≈4 tail rows)
is too small to stop it reliably.

## How often did fine-tuning do anything?

Selected fine-tune epochs (0 = validation never beat the MSE init):

| regime | dfl_lp | mlp_dfl_lp |
|---|---|---|
| well_specified_high_sample | 5, 0, 9, 1, 3 | 0, 5, 4, 0, 0 |
| heavy_tail_medium_sample | 14, 10, 0, 0, 5 | 14, 1, 7, 1, 2 |
| nonlinear_misspecified | 15, 3, 2, 4, 3 | 3, 0, 0, 2, 3 |
| predictable_tail_crash | 0, 1, 0, 1, 5 | 0, 0, 0, 0, 3 |
| high_cost_turnover_constrained | 25, 6, 0, 0, 9 | 2, 1, 2, 0, 1 |

28% of `dfl_lp` cells and 40% of `mlp_dfl_lp` cells selected 0 epochs. In the
crash regime, fine-tuning almost never helped on validation. So at this
sample size the honest summary of true DFL is: "usually a no-op, occasionally a
small gain, occasionally a small loss, net zero against two-stage".
