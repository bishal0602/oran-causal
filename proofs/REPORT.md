# THM-BY-FIN: finite weighted BY bound

## Main theorem

`OranCausal.StepUp.fdr_le_of_selfConsistent`

Let \((\Omega,\mathcal F,\mu)\) be any probability space, \(m\ge1\),
\(q\ge0\), \(w_j\ge0\), and \(\delta\in\mathbb R\). The parameters
\(q,w,\delta\) and the null set \(H_0\subseteq\{0,\ldots,m-1\}\) are fixed.
Let \(p_j:\Omega\to\mathbb R\) and
\(S:\Omega\to\mathcal P(\{0,\ldots,m-1\})\) be measurable. Set

\[
c_m=\sum_{i=1}^m\frac1i,\qquad
t_j(k)=\frac{q w_j k}{m c_m}.
\]

If, for every outcome, \(j\in S\) implies \(p_j\le t_j(|S|)\), and

\[
\mu\{p_j\le t_j(i)\}\le t_j(i)+\delta
\quad(j\in H_0,\ 1\le i\le m),
\]

then

\[
\operatorname{FDR}
=\int_\Omega\frac{|H_0\cap S(\omega)|}{\max(|S(\omega)|,1)}\,d\mu(\omega)
\le\frac q m\sum_{j\in H_0}w_j+|H_0|\delta.
\]

## Claim-to-definition mapping

All names below are in `OranCausal.StepUp`, unless qualified otherwise.

| Registered claim | Lean representation |
|---|---|
| The `m` hypotheses | `Fin m`; arbitrary natural `m` with `hm : 1 ≤ m` |
| Rejection set `S` | `S : Ω → Finset (Fin m)`, with `hS : Measurable S` |
| Self-consistency | Main theorem parameter `hsc` |
| `t_j(k)` | `threshold m q w j k` |
| `c_m` | `harmonic m`; `range m` indexes the terms `1 / (i + 1)` |
| Weights | Fixed function `w : Fin m → ℝ` |
| `H0` | Fixed finite set `H0 : Finset (Fin m)` |
| `delta` | Arbitrary real parameter `δ` |
| Null grid bound | Parameter `hgrid`; probabilities are `μ.real` |
| FDR | `fdr μ S H0`, the Bochner integral of the false-discovery proportion |
| Per-null contribution | `contribution (S ω) j`; `sum_contribution` proves the counting identity |
| Whatever the dependence | Arbitrary `Ω`, sigma-algebra, probability measure `μ`, and jointly defined `p` and `S`; no independence assumptions |

`μ.real A` is Mathlib's real-valued measure of `A`. The probability-measure
instance guarantees finiteness; each grid event is measurable. Integrability
of every rejection contribution is proved from measurability and its bounds
between zero and one.

## Hypotheses and scope

| Hypothesis | Relation to the registered claim |
|---|---|
| Arbitrary measurable space with `μ : Measure Ω` and `IsProbabilityMeasure μ` | Makes the probability and expectation notation explicit. |
| `m : ℕ`, `1 ≤ m` | Positive finite number of hypotheses; positivity is implicit in the displayed denominator, but not expressly stated. |
| Fixed `q : ℝ`, `0 ≤ q` | Nonnegative target level, implicit in its statistical meaning. No assumption `q ≤ 1` is needed. |
| Fixed `w : Fin m → ℝ`, `∀ j, 0 ≤ w j` | Explicit in the quoted manuscript corollary, not in the registered sentence. |
| Fixed real `δ` | Stated parameter. Nonnegativity is not required by this proof. |
| Fixed `H0 : Finset (Fin m)` | The stated true-null subset. |
| `p : Fin m → Ω → ℝ`, `∀ j, Measurable (p j)` | Makes the random-variable meaning of p-values explicit. |
| `S : Ω → Finset (Fin m)`, `Measurable S` | Makes the random rejection set and FDR well-defined. |
| Pointwise self-consistency `hsc` | Exactly the stated rejection condition, for all outcomes and rejected indices. |
| Marginal grid inequalities `hgrid` for null indices only | Exactly the stated probability bounds for `i = 1, …, m`. |

The formalization uses deterministic weights, as required by the source
proof's fixed-threshold argument. It does not quantify over outcome-dependent
weights. Weights selected on independent data can be regarded as fixed after
conditioning, provided the grid bounds hold in the conditional probability
space; that conditioning step is not asserted by this theorem.

There is no weight normalization assumption. The manuscript's
\(\sum_j w_j=m\) is only needed for the further bound by \(q\) when
\(\delta=0\). Nor does this theorem assume that `S` is the maximal step-up
set, that non-null p-values are valid, or that p-values are mutually independent.

The p-values are permitted to be any measurable real functions satisfying
the null grid bounds. Thus ordinary p-values in \([0,1]\) are included without
needing range hypotheses. Zero weights, p-values equal to zero, ties at grid
thresholds, an empty null set, and no rejections are included. In particular,
there is no division by a weight or a threshold increment.

These are explicit domain conventions and a strengthening of the p-value
range, not a change to the bound. The nonnegative level and weight conventions
come from their stated statistical role and the quoted corollary; they are
sufficient conditions, not a claim that each is mathematically necessary.
No counterexample to the registered statement under these conventions was
found. The expression for `m = 0` has a zero denominator in ordinary
mathematics and is outside the formal statement.

## Proof structure

For a nonnegative nondecreasing sequence `f`, the discrete envelope is

\[
B_m(f)=\frac{f(m)}m+
\sum_{i=1}^{m-1}f(i)\left(\frac1i-\frac1{i+1}\right).
\]

`ratio_le_envelope` proves \(f(r)/r\le B_m(f)\) for \(1\le r\le m\).
With \(a=q w_j/(m c_m)\) and
\(f(i)=\mathbf1\{p_j\le ai\}\), self-consistency bounds the rejected
contribution by this envelope. Its expectation is the same linear expression
in the marginal grid probabilities (`integral_envelope`). All coefficients
are nonnegative (`envelope_mono`). Substitution of the grid bounds gives

\[
B_m(i\mapsto ai+\delta)=a c_m+\delta
\]

by `envelope_linear`. Summation over true nulls finishes the proof.
The argument uses only marginal probabilities, so arbitrary dependence is
preserved throughout. The asymptotic assertion is outside this project.

## Reproduction and validation

The project pins Lean `leanprover/lean4:v4.35.0` and Mathlib tag `v4.35.0`
(commit `56a005855ea17e0b0958f73ec801ae140b0a1679`).

From this directory, with the pinned toolchain on PATH:

```text
lake build
lake env lean OranCausal/StepUp.lean
```

On this machine, Lean was installed locally under the ignored `.tools/`
directory. To use that installation in PowerShell, first run:

```powershell
$env:PATH = "$PWD\.tools\lean-4.35.0-windows\bin;$env:PATH"
```

Validation on 2026-10-16 succeeded with no warnings or errors. The final
`lake build` output, also saved in `build.log`, was:

```text
ℹ [2643/2645] Built OranCausal.StepUp (16s)
info: OranCausal/StepUp.lean:245:0: 'OranCausal.StepUp.fdr_le_of_selfConsistent' depends on axioms: [propext, Classical.choice, Quot.sound]
✔ [2644/2645] Built OranCausal (8.8s)
Build completed successfully (2645 jobs).
```

The separate `lake env lean OranCausal/StepUp.lean` recheck exited with code
zero and produced the following output, saved in `axioms.log`:

```text
'OranCausal.StepUp.fdr_le_of_selfConsistent' depends on axioms: [propext, Classical.choice, Quot.sound]
```

The proof sources contain no `sorry`, `admit`, new axiom declarations,
`native_decide`, `implemented_by`, `extern`, or `set_option` commands.

| Measurement | Result |
|---|---|
| Successful `lake build` wall time | **28.792 seconds**; both project modules compiled, dependency artifacts already cached |
| Separate source/axiom recheck wall time | **13.781 seconds** |
| Machine | `DESKTOP-4F2VPO4` |
| OS | Windows 11 Home Single Language, `10.0.26200`, x86-64 |
| CPU | 13th Gen Intel Core i5-1340P, 16 logical processors |
| RAM | 15.69 GiB |
| Lean | `4.35.0`, `x86_64-w64-windows-gnu`, commit `eb6cafb7290e8bd5f2937522d8d804f58f02bafc`, Release |

The reported build time excludes initial toolchain and dependency downloads,
cache extraction, and earlier failed development builds. The manifest pins
all dependency commits; Mathlib's `inputRev` is `v4.35.0` and its resolved
`rev` is the commit recorded above.

## Task 05 — THM-GATE-BUDGET

Status: **implemented** under the development-data source context explicitly
confirmed by the author. The formalization is in `OranCausal/GateBudget.lean`,
imported by `OranCausal.lean`.

### Source context and hypotheses

The tolerance is `ε = (1/20) * G / (3 * max N 1)`, where `N` and `G` are
development accept-all means (`tasks/sources/s04_pmrt.tex:235–239`). The
author confirmed the following interpretation:

| Hypothesis | Formal representation | Source context |
|---|---|---|
| (a) First-order additivity over rejected units and relations | `ratioMovement = (∑ i ∈ units, ∑ r : Fin 3, effect i r) / G` | `s04_pmrt.tex:240–243`; additivity explicitly confirmed by author |
| (b) Ratio denominator is development `G`, with `G > 0` | The same `G` occurs in `tolerance` and the ratio; premise `hG : 0 < G` | `s04_pmrt.tex:238–241`; denominator matching and positivity explicitly confirmed by author |
| (c) Rejected count is at most `max N 1` | `hcount : (units.card : ℝ) ≤ max N 1`; a real-valued mean-count version is also provided | `s04_pmrt.tex:235,238–240`; count budget explicitly confirmed by author |
| Per-edge magnitude bound | `∀ i ∈ units, ∀ r, abs (effect i r) ≤ tolerance N G` | `s04_pmrt.tex:240` |
| Three relations | Relation index `Fin 3` | `s04_pmrt.tex:240` |

These conditions are explicit premises, not conclusions about held-out
episodes. The theorem permits arbitrary real `N` and positive `G`; neither a
study-specific sample size nor a fixed number of rejected units is assumed.
Nonnegativity of `N` is unnecessary because `max N 1` is positive regardless.

### Declarations and proof

All names below belong to `OranCausal.GateBudget`.

- `budget_identity` proves `3 * max N 1 * tolerance N G / G = 1/20`.
- `budget_bound` proves the requested inequality `≤ 0.05`.
- `mean_count_budget` handles an arbitrary real count (including a
  development mean) and a total movement bounded by `3 * count * ε`.
- `additive_guard_budget` proves the absolute ratio-movement bound for a
  finite set of rejected units. Two triangle inequalities bound the sum by
  the sum of per-edge magnitudes; three relations and the count budget then
  reduce it to the identity above.
- `count_overrun_example` records the requested scope note.

### Scope note

For development `G=100`, `N=1`, the tolerance is `5/3`. Two rejected units,
each contributing exactly `ε` on each of three relations, give additive
ratio movement `6 * (5/3) / 100 = 1/10 > 1/20`. The lemma also checks
`2 > max 1 1`: condition (c) fails. All six effects meet the per-edge
bound. This illustrates the role of the count budget and is recorded only
as a scope note for the development-data rationale.

### Validation

- `lake build`: passed without warnings or errors, 3032 jobs; **166.528 s**
  wall time with dependencies cached. Machine/toolchain as above; Lean and
  Mathlib pins remain `v4.35.0`.
- `lake env lean OranCausal/GateBudget.lean`: passed without warnings or errors.
- `#print axioms` for `budget_bound`, `mean_count_budget`,
  `additive_guard_budget`, and `count_overrun_example`: each exactly
  `[propext, Classical.choice, Quot.sound]`.
- `lake env leanchecker --verbose OranCausal.GateBudget`: passed, exit 0,
  `replaying OranCausal.GateBudget`. This replays the module's declarations
  against its imported environment with Lean's kernel.

Logs: `task05-build.log`, `task05-axioms.log`, `task05-leanchecker.log`.

The additional repository-wide `uv run oran-causal check` passed lint,
formatting, and type checks, but did not complete its Python test suite.
The first run reached the command timeout; the background retry exited 255
without a traceback or pytest summary, with its log ending during
`packages/analysis/tests/test_posthoc_gates.py`. The cause is undetermined;
this is not recorded as a passing workspace check. See
`task05-workspace-check.log`. The separate Lean acceptance checks above passed.
