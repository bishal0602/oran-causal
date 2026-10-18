# Task 07 — asymptotic validity of PMRT

Status: **needs clarity**. The existing Lean development is complete as a
collection of composable results, with three named external theorem obligations.
Two qualifications to the pinned claim remain: the martingale assertion needs
integrability, and exact redraw variance one holds only when sigma is positive.
No manuscript file was changed.

All declaration names below are in `OranCausal.Asymptotic`, in
`OranCausal/Asymptotic.lean`, imported by `OranCausal.lean`.
Sources are the pinned `tasks/sources/s04_pmrt.tex` and
`tasks/sources/app_a.tex`, reproduced in the Task 07 specification.

## Statements and source mapping

| Paper quantity or step | Lean representation and result |
|---|---|
| Assignment, propensity, direction, predictable weight | `L`, `p`, `s`, `w`; binary `L`, predictable `p,s,w`, `s²=1`, and `E[L|F_i]=p` are explicit hypotheses |
| Sharp-null use | A2 supplies predictability of the post-window-based weight under the enlarged history; this is represented by `hw`, not a separate simulator model |
| Fresh draws | The conditional Bernoulli mean is the A1 conclusion supplied by Task 03; Task 07 does not reconstruct the fresh-key process |
| Increment and variance | `D_i=(s_i w_i)(L_i-p_i)`; `bernoulli_increment_moments` proves conditional mean zero and conditional second moment `w_i² p_i(1-p_i)` |
| Martingale and predictable quadratic variation | `pmrt_martingale` proves the partial-sum martingale and equality of the sum of conditional squared increments to the displayed design variance |
| Episode growth | One fixed filtration and sequence; deterministic nondecreasing row lengths `m n` tending to infinity. The source uses `m n=n N_max`, with nonexistent units assigned weight zero |
| A4 scale and random limit | Positive deterministic `scale n` tending to infinity; predictable variance divided by `scale n²` tends in probability to `eta²`, with measurable `eta>0` almost surely |
| A4 maximum-weight condition | `bound n` dominates every `|w_i|/scale n` in the row and tends to zero in probability; the row maximum is the intended instance |
| Conditional Lindeberg | `conditional_lindeberg_from_max` proves it from the predictable weight bound; it is not assumed as an extra model condition |
| Observed normal limit | `self_normalized_clt`, conditional on the named `HallHeyde` theorem obligation |
| Conditional redraw moments | `bernoulli_redraw_moments` and `standardized_redraw_moments`; the latter requires `sigma>0` |
| Redraw normal approximation | `redraw_cdf_limit` proves uniform CDF convergence in probability from bounded independent centred rows, variance convergence, and the named `LindebergFellerPolya` obligation |
| Normalized maximum and zero variance | `normalized_max_vanishes`, `normalized_max_identity`, `zero_variance_vanishes`, and `redraw_variance_tends_one` supply the A4 reductions |
| Transformed redraw scores | `redraw_score_cdf_limit`, using continuous score transformation and the named `Polya` obligation |
| Uniform limiting tail probability | `cdf_uniform` proves the probability integral transform; `redraw_tail_uniform` combines it with the uniform CDF approximation |
| Conditional binomial count | `conditional_binomial_from_success_sets` derives count probabilities from the conditional product law of the B success indicators; `conditional_binomial_atom` integrates them |
| Binomial mixture | `binomial_uniform_integral` proves the beta-integral value `1/(B+1)`; `binomial_mixture_limit` derives convergence of each count probability from weak convergence of the tail probability |
| Finite Monte Carlo grid | `conditional_binomial_rejection_limit` proves the exact raw rejection-probability limit `gridCDF B t = floor(t(B+1))/(B+1)` for every fixed B and t in [0,1] |
| Conservative reporting | `asymptotic_rejection_bound` proves the reported-p limsup bound by `gridCDF B t`, and `gridCDF_le` bounds it by t |

The finite-grid proof sums count probabilities directly. It remains valid at
grid discontinuities, including t=0 and t=1; it does not incorrectly infer
CDF convergence at atoms from weak convergence alone. B is any fixed natural
number, including zero. A sequence B depending on n is outside this theorem.

## Named external theorem obligations

These are definitions of propositions passed as explicit hypotheses, never
Lean axioms. The relevant general results were not located in the pinned
Mathlib v4.35.0 library. Task rules permit this treatment; the separate citation
tasks remain responsible for checking the cited works themselves.

1. **`HallHeyde mu`** — the source's Hall–Heyde Corollaries 3.1 and 3.2,
   specialized to initial segments of a single filtration. Square-integrable,
   adapted martingale differences, deterministic diverging scale and row
   lengths, conditional Lindeberg, and predictable variance/scale² converging
   to eta²>0 imply stable convergence of sum/scale to eta Z and of
   sum/sqrt(predictable variance) to Z. Z is standard normal and independent
   in the stable-limit representation. `StableGaussianLimit` explicitly
   quantifies over bounded measurable variables on the original space and
   bounded continuous test functions.
2. **`LindebergFellerPolya nu`** — the source's Durrett Theorem 3.4.10 and
   Exercise 3.2.9. Independent, centred, square-integrable triangular rows
   with total variance tending to one and Lindeberg sums tending to zero
   have a standard-normal distributional limit and uniformly convergent
   CDFs. `redraw_cdf_limit` applies this pathwise on further subsequences;
   the passage back to convergence in probability is proved in Lean.
3. **`Polya`** — weak convergence of probability measures on the real line
   to a law with continuous CDF implies uniform CDF convergence. This is
   the general continuous-limit version of the same cited exercise, used
   for transformed scores.

These obligations supply general limit theorems, not the PMRT p-value result.
The probability integral transform, conditional-binomial reduction,
beta integral, finite-grid limit, and conservative limsup comparison are proved.

## Assembly interfaces and directions

The module exposes separate stages rather than one theorem taking a single
A1–A4 structure. Its explicit interfaces include measurable conditional redraw
laws/tail probabilities and CDF errors, conditional independence and Bernoulli
marginals of redraws, and the conditional product law of redraw successes.
They describe the paper's redraw experiment. Construction of that experiment
as one probability kernel, and discharge of all its measurability obligations,
are not automated by this module. Likewise the finite-sum variance identity
is available separately from the generic CLT theorem. These interfaces must
be discharged when instantiating the stages; they are not hidden conclusions.

Use the negative score so an inclusive exceedance is a lower CDF event.
For the two-sided test the score is `-abs(z)`, with limit law the image of
standard normal under negative absolute value. `lower_absolute_atomless`
and `lower_absolute_normal_continuous` justify its continuous CDF, and
`cdf_mapped_exceedance` identifies the exact inclusive-tail probability.
`continuous_score_limit` supplies the observed transformed limit.

For a fixed direction d and orientation o in {−1,1}, use score `-d o z`.
`fixed_direction_score_limit` also permits a measurable random orientation
fixed across n, using the stable normal limit. This is not permission to
select a new orientation from each evaluation sample. For a random fixed
orientation, the two fixed-sign redraw CDF approximations can be combined
over its two events; that final specialization is not packaged as a separate
end-to-end theorem.

`asymptotic_rejection_bound` yields, for every t in [0,1],

\[
\limsup_n\Pr(p_n\le t)
\le \frac{\lfloor t(B+1)\rfloor}{B+1}\le t.
\]

This is the conclusion to supply to Task 08, conditional on the listed
interfaces and theorem obligations. The reported p-value may be any value
at least the raw plus-one p-value. This covers increasing the count through
the numerical tolerance and reporting p=1 on zero variance or skipped tests.
`reportedP_conservative` proves the zero-variance comparison explicitly.
The source's informal final “vanishing term plus Monte Carlo error” sentence
is represented through the tail-probability approximation and conditional
binomial law, not a separate quantitative Monte Carlo error theorem.

## Qualifications requiring author review

**Integrability in the opening martingale assertion.** A1–A3 alone say that
weights are predictable, not integrable. `pmrt_martingale` explicitly uses
square-integrable weights, as supplied by A4. Without integrability even a
single increment need not be integrable: take U uniform on (0,1) in the initial
history, w=1/U, and an independent fair assignment L. Then w is predictable
but E|w(L−1/2)|=infinity. This is a mathematical counterexample to the
unqualified A1–A3 assertion, not an additional Lean declaration. For a genuine
martingale and finite conditional second moments, the moment condition must
be present. The asymptotic A1–A4 result already includes square integrability.

**Exact variance one on degenerate data.** If sigma=0, the stated convention
sets every standardized redraw to zero. Its variance is zero, not one;
`zero_redraw_moments` proves this directly. A4 implies that the probability
of this event tends to zero, not that it is impossible for every finite n.
For example, start with one zero weight and then append unit weights with
fair assignments: the first row is degenerate while A4 holds asymptotically.
Thus “variance one” needs “on sigma>0”. The p-value asymptotic bound survives
the degenerate case by conservative reporting. No positivity assumption on
every finite-sample sigma was added to the final p-value bound.

## Validation

`lake build` passed with exit code 0 (3302 jobs), recorded in
`task07-resume-build.log`. All fourteen audited declarations use only
`propext`, `Classical.choice`, and `Quot.sound`; their output is in
`task07-axioms.log`. The forbidden-proof-feature scan is clean. Task 07 has
no warnings; the build replays two existing Task 06 tactic-linter warnings.
The earlier successful build remains in
`task07-build.log`; `task07-source-check.log` is an older, superseded diagnostic.
No dependency pins, paper files, or other task implementations were changed.
