# Task 02: re-proposal counterexample

**Status: needs clarity.** Both interpretations requested by the user are proved,
for every real weight `a > 1`. They give different rejection probabilities.

## Results

All declarations below are in `OranCausal.Counterexample`.

1. `exact_tail_rejection`:
   \[
   \Pr(p_\infty\le1/2)=3/4.
   \]
   Here \(p_\infty\) is the exact conditional redraw-tail probability, including
   ties. `exact_tail_violates_validity` proves that this exceeds \(1/2\).
2. `finite_redraw_rejection_limit`:
   \[
   \lim_{B\to\infty}\Pr\left(\frac{1+K_B}{B+1}\le1/2\right)=3/8.
   \]
   Conditional on the observed assignment, the exceedance count has law
   \(K_B\sim\operatorname{Binomial}(B,p_\infty)\).
3. `finite_redraw_rejection_formula` gives the exact finite-\(B\) probability:
   \[
   \Pr(p_B\le1/2)=\frac38
   \begin{cases}
   1,& B\text{ odd},\\
   1-\binom{2n}{n}/4^n,& B=2n.
   \end{cases}
   \]
   This includes \(B=0\), with rejection probability zero.
   `one_redraw_rejection` proves the counterexample to an unqualified
   finite-redraw equality: at \(B=1\), the probability is \(3/8\), not \(3/4\).

## Model and correspondence

| Source concept | Lean definition |
|---|---|
| Fair sequential assignments | `Outcome := Bool × Bool`, with uniform probability `assignmentProbability`; `true` is acceptance |
| Adaptive existence of unit 2 | `hasSecond ω := !ω.1`; the second latent bit is unused if unit 1 is accepted |
| Scores with propensity one half and positive signs | `signed`, `doubledScore`; the latter is twice \(S\) |
| Design variance | `varianceFour`, equal to four times \(\sigma^2\) |
| Standardized statistic | `z = doubledScore / sqrt varianceFour`, hence exactly \(S/\sigma\) |
| Redraw holding the observed unit set fixed | `z a (hasSecond ω) b`: redraw bits `b` vary, existence remains determined by observed `ω` |
| Exact one-sided tail, \(d=o=+1\) | `pInfinity`, computed by enumerating all four equally probable redraw bit pairs |
| Plus-one Monte Carlo p-value | `monteCarloP B k = (1+k)/(B+1)` |
| Conditional iid redraw count | `binomialMass B k p`, the binomial mass \(\binom Bk p^k(1-p)^{B-k}\) |
| Actual finite-redraw rejection probability | `monteCarloRejection`: finite sum over the uniform observed assignment and its conditional binomial count law |

Probabilities are explicit finite sums, not axiomatized expectation operators.
`binomialMass_sum` proves normalization, and `binomialMass_nonneg` proves
nonnegativity on \([0,1]\). `pInfinity_values` proves that the exact tails are
\(1/2\) on the first-accepted and rejected-then-accepted branches, and \(1\)
on the rejected-twice branch. These branches have probabilities \(1/2\),
\(1/4\), and \(1/4\), respectively.

The finite-redraw probability uses the binomial law of the count directly;
it does not construct an infinite sequence of redraws or assert an almost-sure
convergence theorem for the Monte Carlo p-values themselves.

## Why the results differ

The source's exact-tail calculation is correct. Its wording “with \(B\to\infty\)”
must distinguish a test using the exact tail from the limit of tests using
the plus-one Monte Carlo formula. The cutoff \(1/2\) is an atom of the exact
p-value distribution. On branches with exact tail \(1/2\), the finite test
rejects precisely when \(2K_B<B\), whose probability tends to \(1/2\), not one.

The proof uses binomial symmetry and an explicit vanishing bound:
\[
\left(\binom{2n}{n}/4^n\right)^2\le\frac1{n+1}.
\]
There are no assumptions of a central limit theorem, a law of large numbers,
or continuity of a distribution function at the cutoff.

The registered sentence omits the exact-tail qualification and the condition
\(a>1\); the task source supplies the latter. The authors must decide which
p-value the sentence denotes. The original claim and source text were not
edited. Assertions elsewhere in the example about (A1)–(A3) and a growing-unit
martingale limit are not asserted by these two finite-design theorems.

## Validation

Lean and Mathlib remain pinned to `v4.35.0`. `lake build` passed without
warnings or errors; see `task02-build.log`. The measured build wall time was
36.626 seconds with dependencies cached, on the machine documented in
`REPORT.md`.

```text
'OranCausal.Counterexample.exact_tail_rejection' depends on axioms: [propext, Classical.choice, Quot.sound]
'OranCausal.Counterexample.finite_redraw_rejection_limit' depends on axioms: [propext, Classical.choice, Quot.sound]
Build completed successfully (2703 jobs).
```

The source has no forbidden proof constructs.

## Deferred item: fixed cutoffs above one half

The follow-up theorem
`OranCausal.Counterexample.finite_redraw_rejection_limit_between_half_and_three_quarters`
states, for every real a>1 and fixed 1/2<t<3/4,

\[
 \Pr\left\{\frac{1+K_B}{B+1}\leq t\right\}\longrightarrow\frac34.
\]

The stronger theorem `finite_redraw_rejection_limit_above_half` allows any
1/2<t<1. `monteCarloRejectionAt` is the actual finite-redraw rejection
probability at a general threshold, summing over the observed assignment and
its conditional binomial redraw-count law. `monteCarloRejectionAt_half` proves
that at t=1/2 this is exactly the original `monteCarloRejection` definition.

No weak law is assumed. The proof establishes the finite combinatorial identity

\[
 \sum_{k=0}^B {B\choose k}(2k-B)^2=B\,2^B
\]

by induction using Pascal's recurrence. Dividing by 2^B gives the centred
second moment of Binomial(B,1/2). The elementary inequality
(x+1)^2≤2x^2+2 then bounds the second moment of the **plus-one** p-value:

\[
 E[(p_B-1/2)^2]\leq\frac1{2(B+1)}.
\]

`fair_nonrejection_bound` applies the finite-sum form of Chebyshev/Markov to get

\[
 0\leq\Pr(p_B>t\mid p_\infty=1/2)
 \leq\frac1{2(B+1)(t-1/2)^2}\longrightarrow0.
\]

Thus `fair_rejection_above_half` proves rejection probability tends to one on
the branches of total probability 3/4 with exact tail 1/2. On the remaining
branch the redraw count is B and p_B=1, so it never rejects when t<1.
`monteCarloRejectionAt_eq` proves the exact mixture identity for every B.

For t<3/4, the limit strictly exceeds the nominal cutoff. This establishes
the deferred anti-conservatism result for the actual plus-one procedure while
preserving the different, already proved boundary limit 3/8 at t=1/2.
The original task status remains `needs clarity` because the pinned source
still states the incorrect 3/4 limit **at** the boundary. The follow-up result
itself requires no additional hypotheses or unproved probability lemmas.

Follow-up validation: `lake build` passed with exit 0 (3123 jobs), without errors
or task-02 warnings. Existing cosmetic warnings from task 06 were replayed.
All four audited declarations, including both new limit theorems, depend only
on `propext`, `Classical.choice`, and `Quot.sound`. There are no forbidden proof
constructs. Evidence is in the task 02 result table and
`task02-followup-build.log` / `task02-followup-axioms.log`.
