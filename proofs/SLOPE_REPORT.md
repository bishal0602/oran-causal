# Task 06 — design slope

Status: **needs clarity**. The manuscript statement requires a uniformly bounded
predictable centre. Its proof must group the centred errors by episode rather
than treating overlapping outcome windows as unit-level martingale increments.
The paper and task 05 are unchanged.

All Lean names below are in `OranCausal.Slope` unless otherwise qualified.

## Counterexample and exact identity

`unbounded_centre_counterexample` proves, for every positive integer N and every
Boolean assignment path, that

\[
 |\hat\beta_N|\geq (3^N+3)/N
\]

when outcomes are zero, directions are +1, propensities are 1/2, and the centre
of one-based unit u is 3^u. Lean uses zero-based indices and writes this centre
as `3^(u+1)`. The finite slope is exactly the sum of score-times-residual divided
by the sum of squared scores, not a substitute statistic.

`geometricSum_upper` bounds the magnitude of the preceding signed geometric
sum. `geometricSum_lower` then uses the reverse triangle inequality: the final
term dominates every possible preceding assignment path. `badSlope_eq` connects
this signed sum to the original slope.

`fairAssignments` is the countable product of Bernoulli(1/2) measures on Bool.
`fairAssignments_independent` and `fairAssignments_marginal` verify independence
and fair marginals. With the natural past of these draws, this is an (A1) design.
Take one unit per episode, N=n, and outcomes identically zero. The deterministic
centre is predictable; all outcomes are bounded and observed within their
episode. `zero_outcome_effect` proves the assignment effect is zero.
`counterexample_probability_one` proves the slope has absolute value at least
one with probability one for every N>0, and `counterexample_not_convergent`
proves failure of convergence in probability to zero.

`half_propensity_identity` proves the Horvitz–Thompson identity for every finite
N, every Boolean assignment vector, every outcome and centre, and directions
whose squares are one. It imposes no boundedness, predictability, or stochastic
assumptions. At N=0 Lean totalises division by zero; the statistical use is N>0.
`half_propensity_target` also proves that the propensity-weighted target becomes
the arithmetic mean of the signed assignment effects at propensity 1/2.

## Corrected theorem and source mapping

`EpisodeDesign.corrected_convergence` states convergence in probability of the
actual slope minus the propensity-variance-weighted policy effect to zero.
The generic bounded martingale weak law is now proved in Lean; this theorem
has no weak-law premise.

`EpisodeDesign` represents an arbitrary probability space and an infinite
ordered sequence of episodes. Episode e has K=N_max candidate slots. A slot's
predictable `a` is its direction (+1 or -1) when it exists and zero otherwise.
Consequently `a²` is the existence indicator. Nonexistent slots contribute zero
to the count, score, variance weight, and target. The binary `L` is acceptance.
Dummy draws in empty slots can be added without affecting any statistic.

| Source condition or quantity | Lean representation |
|---|---|
| Natural pre-opening history | `H e j` |
| End-of-episode history | `F (e+1)`; `F e` is the previous episode end |
| H1: outcomes observed by their own episode end | `y_end`; assignment observability is `L_end` |
| H2: at most N_max units per episode | `j : Fin K`, with `a=0` on nonexistent slots |
| H3: appended episode order | `before : F e ≤ H e j`; `after : H e j ≤ F (e+1)` |
| (A1): predictable existence, direction and propensity | `a_pre`, `p_pre`, `a_values`, `overlap` |
| (A1): conditional acceptance probability | `design : E[L\|H]=p`, with `L_values` in {0,1} |
| Bounded outcomes | `y_bound : abs(y) ≤ M` |
| **Added bounded predictable centre** | `c_pre`, `c_bound : abs(c) ≤ C` |
| Score | `q = a(L-p)` |
| Squared-score conditional mean | `w = a² p(1-p)` |
| Numerator | `z = q(y-c)` |
| Conditional assignment contrast | `assignmentEffect` |
| Weighted signed effect | `t = a p(1-p) assignmentEffect` |
| N | `count = total(a²)` |
| Actual slope and target | `beta = total(z)/total(q²)`; `target = total(t)/total(w)` |
| N/n bounded below with probability tending to one | `hcount`: P(N_n < rho*n) tends to zero for fixed rho>0 |
| Upper count bound | `count_cap`: 0 ≤ N_n ≤ n*K, from H2 |

The constants M and C are nonnegative bounds, eta>0 is the overlap constant,
and rho>0 is the lower count rate. These are arbitrary, not study-specific.
The theorem does not assume independent units or independent episodes, a
deterministic realised unit count, or a unit-level adapted outcome process.

For binary assignments with overlap, conditional means given assignment are
joint conditional moments divided by conditional assignment probabilities.
`assignmentEffect_joint_moments` verifies that the implemented effect is

\[
 \tau=\frac{E[Ly\mid H]}p-
       \frac{E[(1-L)y\mid H]}{1-p}.
\]

This is the source definition in `s03_setting.tex`, Equation `eq:tau_u`.
It allows the outcome to depend on subsequent modes within its episode.

## Episode proof

`binary_design_moments` proves the Bernoulli conditional identities using
Mathlib's conditional-expectation linearity and predictable-factor pull-out
rules. In particular E[q(y-c)|H]=t and E[q²|H]=w. Removing the predictable
centre does not change the numerator conditional mean.

`episode_blocking` uses Mathlib's tower property and finite-sum linearity.
If a slot error X has E[X|H]=0, then E[X|F_e]=0. The sum over K slots is
F_(e+1)-measurable and bounded by K times the slot bound. This supplies every
premise of the proved martingale-difference weak law.

`EpisodeDesign.errors_weak_law` applies this argument twice:

\[
 X=q(y-c)-t,\quad |X|\leq 2M+C;
 \qquad Y=q^2-w,\quad |Y|\leq 2.
\]

Thus (total(z)-total(t))/n and (total(q²)-total(w))/n both converge to zero
in probability. The bounds use |q|≤1, |t|≤M and |w|≤1; bounds on conditional
expectations are proved with Mathlib, not imposed as extra model assumptions.

`ratio_error_bound` handles the final algebra. On the high-probability event
N_n≥rho*n, both total(q²)/n and total(w)/n are at least eta²*rho, and
abs(total(t)/n)≤K*M. The first lower bound holds pointwise because overlap
also bounds the *realised* squared score away from zero on existing units.
`convergence_on_good_event` then proves convergence using a union bound and
the two weak laws. The proof uses the conservative overlap bound eta²;
it does not require the sharper eta(1-eta) bound in the manuscript.

## Proved bounded martingale weak law

`bounded_martingale_weak_law` proves `BoundedMartingaleWeakLaw mu` for
an arbitrary probability space. For a filtration F and real increments D_e,
it assumes D_e is F_(e+1)-measurable, integrable, has conditional mean zero
given F_e, and satisfies |D_e| ≤ K almost surely, for one fixed real K.
It concludes n⁻¹ sum_(e<n) D_e → 0 in probability. No independence is assumed.

The Lean proof establishes each step:

1. Boundedness implies membership in L². Integrating the conditional
   expectation gives E[D_e] = 0.
2. For i < j, D_i is F_j-measurable. Pulling it out of the conditional
   expectation gives E[D_i D_j | F_j] = D_i E[D_j | F_j] = 0.
   Integrating proves E[D_i D_j] = 0, hence zero cross covariance.
3. Mathlib's covariance expansion of a finite sum gives
   Var(sum_(e<n) D_e) = sum_(e<n) Var(D_e) ≤ n K².
4. Chebyshev at threshold n ε gives, for n > 0 and ε > 0,
   Pr(|n⁻¹ sum_(e<n) D_e| ≥ ε) ≤ K²/(n ε²).
   This upper bound tends to zero, proving convergence in probability.

The conditional-expectation pull-out and integral identities, finite-sum
variance expansion, and Chebyshev inequality are all proved Mathlib results.
There are no remaining named theorem obligations. `episode_blocking`,
`EpisodeDesign.errors_weak_law`, and `EpisodeDesign.corrected_convergence`
now invoke this proved law without a weak-law parameter. The original
bounded-centre correction and H1–H3 remain explicit.

H1–H3 are the facts explicitly supplied by the user for this task: clipped
post-windows (pinned `app_a.tex` pipeline description), the per-episode unit
cap (its setting proof), and episode/opening-time/cell order. They are explicit
in the formal model; their validity for an implementation is not inferred or
silently assumed by Lean.

## Validation

The pinned Lean/Mathlib v4.35.0 project passed `lake build` with exit code 0
(3123 jobs). Follow-up output is in `task06-followup-build.log`. The only task-06 warnings are two
cosmetic tactic-linter messages at `Slope.lean:505` about using `<;>` where `;`
would suffice; there are no build errors.

The twelve `#print axioms` commands in the module report only `propext`,
`Classical.choice`, and `Quot.sound`, including for the corrected convergence
theorem. Their output is extracted in `task06-followup-axioms.log`, including the newly proved weak law. No `sorry`, `admit`,
new axiom declaration, `native_decide`, unchecked implementation hook, or
check-disabling option occurs in the module. The toolchain and dependency pins
were not changed.

The required workspace command `uv run oran-causal check` passed lint but
stopped at the formatter: `packages/analysis/tests/test_system_stats.py:202`
would be reformatted. That file is outside this task's permitted edit scope
and was left untouched. Type checking and tests after that formatting stage
did not run. Output is in `task06-workspace-check.log`.
