---
id: THM-SCOPE
kind: lean
status: todo
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
| Status | |
| Main declaration(s) | |
| Files | |
| Deviation from the claim | |
| Hypotheses not stated in the claim | |
| `lake build` | |
| `#print axioms` | |
| Notes | |
