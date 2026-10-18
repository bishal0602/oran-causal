---
id: THM-SCOPE
kind: lean
status: needs clarity
---

# 09 THM-SCOPE: Scope of the FDR guarantee

## Claim (verbatim, `spec/claims.yaml`)

> The FDR bound covers the declared edge, not its sign, relation or magnitude; when f affects its own cells, the neighbour and far hypotheses of f are false and declaring them is correct.

Registered dependencies: `THM-FDR-ASY`.

## Source text (verbatim, comments removed; full files in `sources/`)

### Scope of the guarantees (`s04_pmrt.tex`, `sec:reading`)

```latex
\subsection{Scope of the guarantees}
\label{sec:reading}

A declared edge states that the modes of family $f$ change KPI $k$ somewhere in the cells that relation $r$ covers,
and the FDR bound covers that claim asymptotically. It does not cover the sign: the reported sign is the direction of
the weighted association between the centred modes and the outcome, which need not be the sign of $\tau_{\pi}$. Nor
does it cover which relation carries the effect. A unit's neighbour and far cells are the own cells of other units of
$f$, so an effect of $f$ on its own cells also moves the neighbour and far sums. The hypotheses $H_0(f, \text{nbr}, k)$ and
$H_0(f, \text{far}, k)$ are then false, and declaring them is correct. Whether PMRT attributes edges to the same
relation as the ground truth is therefore an empirical question, and the far hypotheses are no negative control.
Table~\ref{tab:guarantees} summarises what each layer of the framework guarantees and what rests on evidence alone.

\begin{table}[tbp]
\caption{Guarantees of the framework.}
\label{tab:guarantees}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright}p{0.27\textwidth}>{\raggedright}p{0.38\textwidth}>{\raggedright\arraybackslash}p{0.29\textwidth}@{}}
\toprule
Layer and assumptions & Guarantee & Not covered \\
\midrule
Logged design; (A1) & Scores have mean zero given the past (Lemma~\ref{lem:fresh}) & logs without propensities \\
Edge p-value; (A1)--(A4) & Asymptotically valid for the sharp null (Proposition~\ref{prop:pmrt}) & finite samples; weak nulls \\
Declarations; (A1)--(A4) & FDR over sharp nulls at most $q$, asymptotically (Corollary~\ref{cor:fdr}) & sign, relation, magnitude \\
Signs & None & wrong signs \\
Design slope; (A1) & Slope converges to $\tau_{\pi}$ (Lemma~\ref{lem:slope}) & exact predictability of the outcome centre \\
Gate & Undeclared guard edges of a rejected class within tolerance, by an approximate 90~\% bound & coverage of the bound; policy-level safety; the protected slice \\
Decisions & None & other scenarios \\
\botrule
\end{tabular*}
\end{table}
```

## Scope

- Prove the registered statement for arbitrary values of its parameters; do not specialise to the numbers of the study unless the statement itself is about specific numbers.
- You decide the formal definitions and the declaration names. Each definition must mean what the claim and the source text say; definitions the claim relies on are in `sources/`.
- If the claim as written is false, ambiguous, or needs a hypothesis it does not state, do not silently change it. Prove the closest correct version, report the exact difference (with a counterexample where one exists) and set the status to `needs clarity`.
- Parts of this claim may be interpretive rather than mathematical. Formalise what is mathematical; list the rest in the report and say why it is not a theorem.

## Acceptance

As in `README.md`: builds, no `sorry`, axioms limited to `propext`, `Classical.choice`, `Quot.sound`, result table filled in.

## Result

Filled in by the agent.

| Field | Value |
|---|---|
| Status | needs clarity (topology hypothesis not stated in the claim but implied by the paper's text; sec:reading's "moves the neighbour and far sums" needs non-cancellation for the summed outcome; authors may accept as implemented) |
| Main declaration(s) | `OranCausal.Scope.Declarations.limsup_fdr_le_with_false_nulls`; `OranCausal.Scope.Model.not_sharpNull_nbr_of_witness`, `not_sharpNull_far_of_witness` (with `_of_own`, `_of_own_uniform` forms) and summed `not_summedNull_nbr_of_witness`, `not_summedNull_far_of_witness`; counterexamples `OranCausal.Scope.Counterexamples.single_cell`, `two_cells`, `summed_cancellation`; non-coverage `OranCausal.Scope.NonCoverage.sign_not_covered`, `expected_stat`, `sign_reversal`, `magnitude_not_covered` |
| Files | `proofs/OranCausal/Scope.lean`, `proofs/OranCausal.lean` (import), `proofs/SCOPE_REPORT.md` |
| Deviation from the claim | Propagation own -> nbr/far needs the affected cell to be on another cell's neighbour list / far from some cell (counterexamples `single_cell`, `two_cells`). With the per-cell null target of app:proof_setting nothing else is needed; with the summed outcome y_u of sec:estimand the own effect can cancel in the neighbour/far sum (`summed_cancellation`), so a non-cancellation hypothesis is added. "Not sign, relation or magnitude" is shown by a structural lemma and finite counterexamples on the linear part of S (clip, orientation and plus-one p-value not modelled); interpretive sentences listed in the report |
| Hypotheses not stated in the claim | Topology (per witness: ξ.cell ∈ nbrs c', c' ≠ ξ.cell for nbr; ξ.cell ≠ c', ξ.cell ∉ nbrs c' for far; uniform forms as corollaries), implied by "a unit's neighbour and far cells are the own cells of other units of f"; for the summed version, same-sign non-cancellation; for the FDR link, task 08's hypotheses (NullLimit per true null, Σ w = m, measurability) |
| `lake build` | passes, 3304 jobs (`proofs/task09-build.log`) |
| `#print axioms` | `propext`, `Classical.choice`, `Quot.sound` for all 19 audited declarations (`proofs/task09-axioms.log`) |
| Notes | Formalised against the current paper; sec:reading, s03 and app:proof_setting unchanged from the pinned sources apart from \claim tags. Non-public `import all` of StepUp, FDRAsymptotic, Exact (bodies not exposed); no other module edited. Forbidden-feature scan clean (`proofs/task09-forbidden-scan.log`). Leanchecker to be run on the VPS (p1G). Not committed |
