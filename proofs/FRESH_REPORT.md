# Task 03: fresh draws

## Scope agreed with the author

The author explicitly accepted the full surrounding source setup as context
for the registered sentence. In particular, the random field is jointly
independent and independent of initial information, the keys are selected
recursively without looking ahead, and propensities lie in `[0,1]`.
The source's `λ(C)` denotes `λ(C ∩ (0,1))`, not unrestricted Lebesgue measure.

## Formal statement

The main declaration is `OranCausal.Fresh.fresh_draw`.
On any probability space, let `K` be a countable measurable key space with
measurable singletons. Let `G0` be a sub-sigma-algebra, and let the measurable
field `U : K → Ω → ℝ` have uniform unit-interval marginals. The sigma-algebras
`G0` and `σ(U_k)` for all `k` are jointly independent. Define recursively

\[
G_0=\mathcal G_0,\qquad
V_i(\omega)=U_{\xi_i(\omega)}(\omega),\qquad
G_{i+1}=G_i\vee\sigma(V_i).
\]

Every `ξ_i` is `G_i`-measurable. Fix any index `n` such that
`ξ_i(ω) ≠ ξ_n(ω)` for every `i < n` and every outcome. If `π` is
`G_n`-measurable and `0 ≤ π(ω) ≤ 1`, then

\[
\mathcal L(V_n)=\operatorname{Uniform}(0,1),\qquad
\sigma(V_n)\perp G_n,\qquad
\mathbb E[\mathbf1\{V_n<\pi\}\mid G_n]=\pi\quad\text{almost surely}.
\]

Indices start at zero: Lean's `n` corresponds to the manuscript's `u-1`.
The acceptance event uses a strict inequality, as in the source.

## Mapping and assumptions

All names below have namespace `OranCausal.Fresh`.

| Source concept | Lean representation |
|---|---|
| Probability space | Arbitrary `Ω`, `mΩ : MeasurableSpace Ω`, `μ : Measure Ω`, `IsProbabilityMeasure μ` |
| Countable keys | `K`, `Countable K`, a measurable space and measurable singletons |
| Initial information | `G0`, with `hG0 : G0 ≤ mΩ` |
| Assignment field | `U : K → Ω → ℝ`, each coordinate measurable |
| Full field independence from initial information | `iIndep (fieldInfo G0 U) μ`; `none` indexes `G0`, `some k` indexes `σ(U_k)` |
| Uniform marginal law | `hlaw : ∀ k, μ.map (U k) = unitUniform` |
| Uniform probability measure | `unitUniform := volume.restrict (Ioo 0 1)`; probability-measure instance proved |
| Adaptively selected keys | `key : ℕ → Ω → K`, with `hkey` asserting measurability against the recursively defined `history` |
| History | `history G0 U key` |
| Freshness | `hfresh : ∀ ω i, i < n → key i ω ≠ key n ω` |
| Propensity | Measurable real function `π` with pointwise range in `Icc 0 1` |
| Uniform selected draw | `selected_law` |
| Independence of selected draw and past | `selected_independent` |
| Conditional acceptance probability | Mathlib's Bochner conditional expectation of the acceptance indicator in `fresh_draw` |

The task sentence states measurability of the current key and propensity and
freshness. The surrounding setting and recursive proof supply the remaining
substantive assumptions listed above. The sigma-algebra and random-variable
measurability hypotheses make those assumptions explicit. No extra
independence of the selected key from previous draws is imposed: the key can
depend arbitrarily on the permitted past.

The theorem requires only that the current key is different from every
previous key; it does not additionally require all previous keys to be
pairwise distinct. This includes the source's distinct-key setting.
The propensity may be random and may equal either endpoint zero or one.

## Proof structure

For a fixed target coordinate `k`, `otherInfo` contains all initial
information and all coordinates except `U_k`. Joint independence gives
`coordinate_indep_other`.

`history_localized` proves the source's “recursion never reads `U_k`” step by
induction, rather than assuming it. On the event that all previous keys avoid
`k`, every past event is measurable in `otherInfo`. A localized sigma-algebra
is used to carry this induction through the join defining the next history.
`selection_slice_measurable` then applies freshness to the current key slice.

`fresh_factorization` partitions an arbitrary past event by the countably
many current keys, uses independence on each slice, and sums the disjoint
probabilities. This proves both the selected law and independence.

For the random propensity, `independent_uniform_threshold` uses the product
law and integrates the uniform interval probabilities. Masking the propensity
by a past event gives `ConditionalAcceptance`, the defining conditional
probability identity on every past event. Finally,
`ConditionalAcceptance.condExp_eq` uses uniqueness of conditional expectation
to obtain the ordinary Mathlib conditional-expectation equality.

## Validation

**Implemented**, under the full source context explicitly accepted by the
author. Lean and Mathlib remain pinned to `v4.35.0`.

The direct source check (`lake env lean OranCausal/Fresh.lean`) passed without
warnings or errors. The main theorem's axiom output was:

```text
'OranCausal.Fresh.fresh_draw' depends on axioms: [propext, Classical.choice, Quot.sound]
```

The project build also passed without warnings or errors. Its last lines,
saved in `task03-build.log`, were:

```text
ℹ [3025/3027] Replayed OranCausal.Fresh
info: OranCausal/Fresh.lean:374:0: 'OranCausal.Fresh.fresh_draw' depends on axioms: [propext, Classical.choice, Quot.sound]
✔ [3026/3027] Built OranCausal (23s)
Build completed successfully (3027 jobs).
```

The successful build took **30.950 seconds** wall time, with the Fresh module
and dependencies already compiled and the root module rebuilt. In the
preceding build, Fresh compiled in 28 seconds, but the command timed out
before the root build finished; that interrupted invocation is not reported
as a successful build. Initial downloads and development checks are excluded.
The machine is `DESKTOP-4F2VPO4`, Windows 11 `10.0.26200`, Intel Core i5-1340P,
16 logical processors, 15.69 GiB RAM, as documented in `REPORT.md`.

The source contains none of the forbidden proof constructs. The new module is
imported by `OranCausal.lean`. No claim or source text was changed.
