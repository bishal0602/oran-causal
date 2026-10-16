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
