# Task 09: scope of the FDR guarantee (`sec:reading`, `tab:guarantees`)

Status: **needs clarity** (see "Status" below). The proofs are complete and have no `sorry`. The theorems are in
`OranCausal/Scope.lean`, namespace `OranCausal.Scope`, imported by `OranCausal.lean`. Source: the current paper
(`paper/sections/s03_setting.tex`, `s04_pmrt.tex`, `app_a.tex`). The drift from the pinned sources is listed below.

## Decisions taken with the authors

| Question | Decision |
|---|---|
| Null target | Main version: per-cell target of `app:proof_setting` (KPI of every cell of r at every window second). Second version: the summed outcome `y_u` of `sec:estimand` (sum over the cells of r, post minus pre), with a non-cancellation hypothesis and a counterexample without it |
| "Not sign, relation or magnitude" | Structural lemma plus finite counterexamples (sign, magnitude); relation through the propagation theorems. No asymptotic version |
| Topology | Per-witness hypotheses for the core theorems, uniform network corollaries. `nbr c = nbrs c \ {c}`; no irreflexivity or symmetry assumed |
| `tab:guarantees` | Only the Declarations row ("sign, relation, magnitude" not covered) and the Signs row ("None"); the other rows belong to tasks 03, 05, 06, 07, 08 |
| Status | A hypothesis the claim does not state means `needs clarity` (README), noting that the paper's text implies it |

## Formal setting

One family f and one KPI k. `Model E C T A`: neighbour lists `nbrs : C → Finset C`, pre- and post-windows
`pre post : T → Finset T`, potential KPI `kpi e c s a` of cell c at second s of episode e under the whole assignment
a of f's modes (A arbitrary). A key `⟨episode, cell, opening⟩` is a possible unit (the xApp is fixed: one writer per
family). Relations: `own c = {c}`, `nbr c = nbrs c \ {c}`, `far c = univ \ insert c (nbrs c)`. `window t = pre t ∪
post t`. Per-cell `target r ξ a` is the dependent function (cell of r, window second) ↦ KPI; `summed r ξ a = Σ_{c ∈
cells r} (Σ_post − Σ_pre)`. `SharpNull r` (resp. `SummedNull r`): every key's target (resp. summed outcome) is the same
under any two assignments.

## Results

| Declaration | Statement |
|---|---|
| `Declarations.fdp_insert_le`, `fdp_union_le`, `fdr_union_le` | Declaring hypotheses whose sharp null is false adds no false discovery and never raises the FDP / FDR |
| `Declarations.limsup_fdr_le_with_false_nulls` (main, a) | THM-FDR-ASY with H0 = {j : sharp null of j holds}: the weighted BY step-up plus any measurable extra declarations of false sharp nulls has limsup FDR ≤ q. A declared edge is a false discovery only if its sharp null holds; sign, relation and magnitude are not arguments of `fdr` |
| `Model.not_sharpNull_nbr_of_witness` (main, b) | f changes the own KPI of ξ.cell at a window second, and ξ.cell ∈ nbrs c' with c' ≠ ξ.cell ⇒ H₀(f, nbr, k) false |
| `Model.not_sharpNull_nbr_of_own`, `not_sharpNull_nbr_of_own_uniform` | ¬H₀(own) ⇒ ¬H₀(nbr) if every affected cell (resp. every cell) is on another cell's neighbour list |
| `Model.not_sharpNull_far_of_witness` (main, b) | as above with ξ.cell ≠ c' and ξ.cell ∉ nbrs c' ⇒ H₀(f, far, k) false |
| `Model.not_sharpNull_far_of_own`, `not_sharpNull_far_of_own_uniform` | ¬H₀(own) ⇒ ¬H₀(far) if every affected cell (resp. every cell) is far from some cell |
| `Model.not_summedNull_nbr_of_witness`, `not_summedNull_far_of_witness` | summed versions; extra hypothesis: no cell of c''s nbr (far) sum moves against ξ.cell (0 ≤ Δ_d·Δ_c) |
| `Counterexamples.single_cell` | one cell: own null false, nbr and far sharp nulls hold (no such cells) |
| `Counterexamples.two_cells` | two cells on each other's lists: own and nbr false, far sharp null holds (far set empty) |
| `Counterexamples.summed_cancellation` | accept moves cell 0 by +1 and cell 1 by −1, both on cell 2's list: own (summed) and per-cell nbr nulls false, summed nbr null holds |
| `NonCoverage.fdp_le_dirFdp`, `sign_not_covered` | the FDP never exceeds the directional FDP (error if null true or sign wrong); one declared false null with a wrong sign has FDP 0 and directional FDP 1 |
| `NonCoverage.expected_stat` | linear part of S = Σ v_u w_u with w_u = g_u + h_u y_u (g, h predictable; y_u = y0_u + τ_u L(W_u)), independent draws: E S = Σ_u sgn_u h_u τ_u π_u(1 − π_u) |
| `NonCoverage.sign_reversal` | positive gains 1/10 and 1, π = 1/2, τ = (2, −1): average effect 1/2 > 0, E S = −1/5 < 0 |
| `NonCoverage.magnitude_not_covered` | own sharp null false while the unit effects are +1 and −1 for every assignment of the other unit, so their average (and every policy average) is 0 |

## Deviations and qualifications

1. **Topology hypothesis not stated in the claim.** "When f affects its own cells, the neighbour and far hypotheses
   of f are false" needs the affected cell to be on another cell's neighbour list (nbr), and far from some cell
   (far). `single_cell` and `two_cells` show both can fail. The paper's text implies the hypothesis ("A unit's
   neighbour and far cells are the own cells of other units of f"), and it holds in the study's network; the claim
   itself states none.
2. **"Also moves the neighbour and far sums" (sec:reading) needs non-cancellation.** With the per-cell target of
   `app:proof_setting` the propagation needs only deviation 1. With the summed outcome of `sec:estimand` the own
   effect can cancel inside the neighbour or far sum (`summed_cancellation`), so the sentence is false as worded.
   Wording point for the paper: say that the effect moves the KPI of neighbour and far cells (per-cell), or add the
   non-cancellation condition. The sharp null in the paper is stated on the per-cell target, so the conclusion
   ("H₀(f, nbr, k) and H₀(f, far, k) are then false") holds under deviation 1 alone.
3. **Sign and magnitude are shown by counterexamples on special cases.** `stat` is the linear part of the PMRT
   statistic: the clip of the weight, the orientation o and the plus-one p-value are not modelled. The reported sign
   is sign(S·o) with o fixed, so a negative drift with o = +1 gives the wrong sign on average; a probability-of-wrong-
   sign version (asymptotic) was not requested. In `magnitude_not_covered` a unit's effect does not depend on the other
   unit's mode, so every policy-averaged effect τ_π equals the unit average 0; the general τ_π of `def:tau_pi`
   (conditional expectations, limit over episodes) is not formalised.
4. **Interpretive, not theorems.** "Whether PMRT attributes edges to the same relation as the ground truth is an
   empirical question" and "the far hypotheses are no negative control" are readings of results (2) and the
   counterexamples, not mathematical statements. The other rows of `tab:guarantees` are tasks 03, 05, 06, 07, 08.
5. **Inherited from task 08.** `limsup_fdr_le_with_false_nulls` uses `FDRAsymptotic.limsup_fdr_le` and its
   hypotheses (`NullLimit` per true null, deterministic weights with Σ w = m, measurable p-values and tested sets);
   task 08 is `needs clarity`.

## Technical notes

`StepUp.fdr`, `FDRAsymptotic.rejections` and `Exact.assignmentMass` are not exposed across modules, so `Scope.lean`
uses non-public `import all OranCausal.StepUp`, `import all OranCausal.FDRAsymptotic`, `import all OranCausal.Exact`.
No other module was edited. `Key.atCell` is `@[expose]` (used in `@[simp]` lemma).

## Source drift (pinned `8744ef6` vs current paper)

`sec:reading` and `tab:guarantees` are textually unchanged (only `\claim{THM-SCOPE}` and a LaTeX comment added). The
relations, the sharp null and `y_u` in `s03_setting.tex`, and `app:proof_setting`, are unchanged. The other changes in
`s04_pmrt.tex` and `app_a.tex` (bounded slope centre, episode grouping, gate class size) do not touch this task. A
tension present in both versions: `sec:estimand` defines y_u as a sum over the cells of r, `app:proof_setting` the null
target per cell, and `tab:pipeline` builds the weight from per-cell receiver slots (handled by deviation 2).

## Status

`needs clarity`, for deviation 1 (hypothesis not stated in the claim, implied by the paper's text) and deviation 2
(wording of sec:reading for the summed outcome). The authors may set it to `implemented` if they accept them.

## Validation

`lake build` passes (3304 jobs), recorded in `task09-build.log`; the new module has no warnings (the build replays the
two known `Slope.lean:505` linter warnings). `task09-axioms.log` holds 19 audited declarations, all with `propext`,
`Classical.choice`, `Quot.sound` only. The forbidden-feature scan is clean (`task09-forbidden-scan.log`). Per-step
logs: `task09-step1..6-build.log`. Kernel re-check (leanchecker) not run on the laptop: modules for the VPS check are
`OranCausal.Scope` (new) and its imports `OranCausal.StepUp`, `OranCausal.FDRAsymptotic`, `OranCausal.Exact`,
`OranCausal.Asymptotic`. Nothing was committed; no dependency pins or paper files were changed.
