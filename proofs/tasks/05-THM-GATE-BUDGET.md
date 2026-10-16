---
id: THM-GATE-BUDGET
kind: lean
status: todo
---

# 05 THM-GATE-BUDGET: Gate tolerance budget

## Claim (verbatim, `spec/claims.yaml`)

> If every unit of a class were rejected and each rejection moved a guard KPI by at most epsilon on each of the three relations, the guard ratio would move by at most 0.05.

Registered dependencies: `DEF-GATE`.

## Source text (verbatim, comments removed; full files in `sources/`)

### Edge-certification gate (`s04_pmrt.tex`, `sec:gate`)

```latex
\subsection{Edge-certification gate}
\label{sec:gate}

A map leaves some guard edges undeclared, and a referee acting on it can harm a guard KPI through such an edge. The
gate prevents this by letting the referee reject a class of requests only when every undeclared guard edge of the
class is certified small. It checks every class $(f, d)$ that the referee can reject against the own, neighbour and far
relations of the two guard KPIs, service violations and RLFs. Rejecting a request of direction $d$ moves a guard KPI
by $-d\,\beta$, so a rejection's harmful direction is $h = -d$. We write the class $(f, +1)$ of requests that raise $f$
as $f$-up, so that sleep-up holds the requests to put a pico to sleep, and $(f, -1)$ as $f$-down.

A guard edge that the map declares stays under the referee's own logic. An undeclared edge is certified when the
one-sided 90~\% upper bound of $h\,\beta$ from the map's own estimator is at most a tolerance $\varepsilon(f, d, k)$,
and is otherwise unresolved; a class with any unresolved edge is always accepted. PMRT's bound uses the design slope
with its martingale standard error, the associational maps' bound the least-squares slope with an independent-error
standard error, and the oracle map's a conservative bound from its stored 95~\% interval; a map without an effect
estimator cannot certify. The tolerance is
\begin{equation}
\varepsilon(f, d, k) = \frac{0.05\, G_k}{3\, \max\{N(f, d), 1\}},
\label{eq:tol}
\end{equation}
with $G_k$ the mean per-episode total of guard KPI $k$ and $N(f, d)$ the mean number of units of the class per
episode, both measured under accept-all on 40 development episodes (Appendix~\ref{app:simparams}). Suppose that every unit of a
class were rejected and that each rejection moved the KPI by at most $\varepsilon$ on each of the three relations. The
guard ratio would then move by at most 0.05, half the margin of 0.10 that a referee may use under the 1.10 limit of
Section~\ref{sec:estimand}. The certificate is thus a first-order budget per edge, not a bound on the referee's guard
ratios: it ignores overlapping windows, classes acting together and declared edges, so whether a referee meets its
guards is checked on held-out seeds. The uncertified classes depend only on the map, the bounds and $\varepsilon$, and
were fixed before the evaluation data were generated.
```

## Scope

- Prove the registered statement for arbitrary values of its parameters; do not specialise to the numbers of the study unless the statement itself is about specific numbers.
- You decide the formal definitions and the declaration names. Each definition must mean what the claim and the source text say; definitions the claim relies on are in `sources/`.
- If the claim as written is false, ambiguous, or needs a hypothesis it does not state, do not silently change it. Prove the closest correct version, report the exact difference (with a counterexample where one exists) and set the status to `needs clarity`.

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
