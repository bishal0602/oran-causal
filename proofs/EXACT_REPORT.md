# Task 04: exact redraw p-values

**Status: implemented**, using the source context explicitly accepted by the
author: `t ∈ [0,1]`, fixed direction `d`, and initial-information-measurable
training orientation `o`.

## Main statement

Main declaration: `OranCausal.Exact.exact_redraw_pvalues`.
Its full-slot version is `OranCausal.Exact.fixed_design_exact`.

For arbitrary natural numbers `B` and `M`, let `M` be a finite bound on the
number of possible realised unit slots. Their presence, order/keys,
propensities, signs, weights, and orientation are measurable in the initial
sigma-algebra `G`. The observed and redraw draws are fresh coordinates of a
jointly independent uniform field independent of `G`. Each replica uses the
same propensity for each slot, the same realised unit set, and the same
statistic. For any fixed real direction `d` and any `t ∈ [0,1]`, the theorem
proves both

\[
\mathbb E[\mathbf1\{p\le t\}\mid G]\le t\quad\text{a.s.},\qquad
\mathbb E[\mathbf1\{p_d\le t\}\mid G]\le t\quad\text{a.s.}
\]

Here

\[
S^{(b)}=\sum_{i\text{ present}}s_i(L_i^{(b)}-\pi_i)w_i,\qquad
\sigma^2=\sum_{i\text{ present}}\pi_i(1-\pi_i)w_i^2,\qquad
z^{(b)}=S^{(b)}/\sigma,
\]

and the p-values are exactly the plus-one, tie-inclusive formula in the
source. Replica zero is the observation; replicas `1,…,B` are redraws.
There is no asymptotic or large-`B` step.

## Representation

All declarations below are in `OranCausal.Exact` unless qualified otherwise.

| Concept | Lean representation |
|---|---|
| Initial information | Arbitrary sub-sigma-algebra `G` of an arbitrary probability space |
| Field independent of itself and initial information | `iIndep (Fresh.fieldInfo G U) μ`, with every coordinate's law `Fresh.unitUniform` |
| Observed and redraw draws | A single countable uniform field with distinct keys; a replica label can be included in the field key |
| Initial-measurable unit ordering | `key : ℕ → Ω → K`, every `key n` measurable in `G` |
| Finite number of redraws | Arbitrary `B : ℕ` |
| Finite unit bound | Arbitrary `M : ℕ`, with unit slots `Fin M` |
| Random realised unit set | `active : Fin M → Set Ω`, each presence event measurable in `G` |
| Inactive slots | `maskedWeight active w`; `pmrtZ_masked` proves exact equality with sums over present slots |
| Design quantities | `π`, `sgn`, and `w : Fin M → Ω → ℝ`, each coordinate `G`-measurable |
| Replicate-major propensities | `repeatedPropensity B M π`; the same slot propensity in each replica |
| Bernoulli modes | `modes U key q n ω`, accepting precisely when the selected uniform draw is below `q n ω` |
| Observed/redraw assignment array | `experiment B M X`, using `finProdFinEquiv` to enumerate replicas and slots |
| Standardized statistic | `pmrtZ` |
| One-sided score | `d * o ω * pmrtZ …`, with fixed `d` and `G`-measurable `o` |
| Plus-one p-value | `monteCarloP`; `monteCarloP_eq_rankP` identifies it with the observation's upper rank divided by `B+1` |
| Two-/one-sided p-values | `twoSidedP`, `oneSidedP` |
| Conditional validity | Mathlib conditional expectations of real event indicators, bounded almost surely |

## Assumptions, sources, and generality

Line references are to the task's checked-in files in `tasks/sources/`.
Each is also recorded in the task result table.

| Formal premise or modeling convention | Source |
|---|---|
| Probability space and initial information independent of the assignment field | `app_a.tex:68–75` |
| Countable key space and jointly independent unit-uniform coordinates | `app_a.tex:70–72,104` |
| Distinct observed unit keys | `app_a.tex:78–79` |
| Independent redraw randomness, independent of the observed data, with the same propensities | `app_a.tex:153–155,165–166,195–198` |
| Realised units, order, signs, propensities, and weights measurable initially | `app_a.tex:189–191` |
| Predictable Bernoulli design law (A1) | `s04_pmrt.tex:86–88`; obtained here from the uniform construction and task 03 |
| Sharp-null setting (A2), with assignment-invariant data entering the common fixed statistic | `s04_pmrt.tex:80–89`; `app_a.tex:64–67,83–86,195–198` |
| Finite unit bound at each finite episode count | `app_a.tex:115–119` |
| Score and design-variance formulas | `s04_pmrt.tex:55–60` |
| Plus-one formula, ties included, each fixed `d ∈ {+1,−1}` | `s04_pmrt.tex:71–77` |
| Training orientation fixed in advance | `s04_pmrt.tex:63–65,110` |
| Threshold `t ∈ [0,1]`, all finite episode counts and redraw counts | `app_a.tex:190–191` |

The conditional calculation needs the common fixed statistic, not a separate
potential-outcome predicate. The sharp-null contribution to the source proof
is represented by holding the initial data and the same signs/weights/statistic
fixed across the observation and all assignments. Initial measurability of
those weights is an explicit extra premise of Lemma 2. The formal theorem does
not assume either exchangeability or the full conditional assignment law;
both are established in the proof.

Some source restrictions are unnecessary for this result. The formal theorem
allows propensities throughout `[0,1]`, arbitrary real signs and weights,
arbitrary fixed real `d`, and arbitrary real initial-measurable `o`. It
therefore includes the source's propensity lower bound and `±1` signs,
directions, and orientations. No moment or nonzero-variance assumption is
needed for the bounded rejection indicators. Zero variance gives the common
score zero and p-value one. `B=0`, no realised units, and ties are included.

The episode index `n` affects the design and its finite unit bound only. The
theorem quantifies over every such bound `M`; it does not fix a study-specific
unit count or redraw count. Padding absent units is an exact normal form,
proved by `pmrtZ_masked`, for the source's initially measurable random unit set.

Only the first `(B+1)*M` draw keys must be fresh. The infinite sequence notation
is for convenient indexing; no freshness or conditional draw-law premise is
imposed beyond that finite experiment. Auxiliary uniforms for inactive slots
have zero weight and do not affect either p-value.

## Proof structure

1. `rank_count_bound`: for any finite vector of scores, at most `(B+1)*t`
   slots can have tie-inclusive upper rank at most `(B+1)*t`. The proof chooses
   a minimum-score element of the rejection set; all rejected slots count in
   its upper rank. Ties require no random breaking or continuity assumption.
2. `iid_rank_valid`: product assignment masses are invariant under slot
   swaps. Each slot has the same rejection probability; averaging the
   deterministic counting inequality proves validity.
3. `conditional_prefix_mass`: conditional Bernoulli draw probabilities and
   initially measurable propensities imply the full conditional product law,
   using the tower and pull-out properties. This joint law is proved, not
   postulated.
4. `conditional_experiment_mass`: reindexing the finite prefix into observed
   and redraw assignments yields the conditional iid assignment law.
5. `conditional_finite_event` and `conditional_iid_rank_valid`: finite
   conditional probabilities are evaluated with the initially measurable
   statistic, yielding the almost-sure conditional bound.
6. `fixed_design_exact`: task 03's `Fresh.fresh_draw` proves the conditional
   Bernoulli probabilities from the fresh uniform field. The preceding steps
   apply separately to absolute and oriented scores.
7. `exact_redraw_pvalues`: the same result with an explicit initially
   measurable realised-unit mask.

The rank argument used by the cited Phipson–Smyth result is proved directly;
no cited statistical result is introduced as an axiom.

## Files and validation

`OranCausal/Exact.lean` is imported from `OranCausal.lean`. `Fresh.lean` has a
small reuse-interface extension: `history` is marked `expose`, and the
structural equations `history_zero` and `history_succ` are exported. This
allows the later module to use the history recursion under Lean 4.35's module
system. Task 03's main theorem statement and proof are unchanged.

Lean and Mathlib pins remain `v4.35.0`. On 2026-10-17, `lake build` and the
separate `lake env lean OranCausal/Exact.lean` source recheck passed without
warnings or errors. The proof source contains no forbidden constructs.

```text
'OranCausal.Exact.fixed_design_exact' depends on axioms: [propext, Classical.choice, Quot.sound]
'OranCausal.Exact.exact_redraw_pvalues' depends on axioms: [propext, Classical.choice, Quot.sound]
Build completed successfully (3031 jobs).
```

Logs: `task04-build.log`, `task04-axioms.log`.
Successful build wall time: **29.662 seconds**, compiling the new Exact module
and root module with dependencies cached. Initial downloads and development
checks are excluded. Machine: `DESKTOP-4F2VPO4`, Windows 11 `10.0.26200`,
Intel Core i5-1340P, 16 logical processors, 15.69 GiB RAM, as in `REPORT.md`.
