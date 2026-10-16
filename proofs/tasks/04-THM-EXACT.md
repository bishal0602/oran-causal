---
id: THM-EXACT
kind: lean
status: todo
---

# 04 THM-EXACT: Exact case of the redraw p-value

## Claim (verbatim, `spec/claims.yaml`)

> If, in addition to (A1) and (A2), the realised units, their order, sgn_u, pi_u and w_u are all G_0-measurable, then Pr(p <= t | G_0) <= t for every t, n and B, and likewise for p_d (Lemma 2).

Registered dependencies: `ASM-A1-DESIGN`, `ASM-A2-NULL`, `THM-FRESH`, `CIT-PS`.

## Source text (verbatim, comments removed; full files in `sources/`)

### Lemma `lem:exact` and proof (`app_a.tex`)

```latex
\begin{lemma}[Exact case]
\label{lem:exact}
Suppose, in addition to (A1) and (A2), that the realised units, their order, $\mathrm{sgn}_u$,
$\pi_u$ and $w_u$ are all $\mathcal{G}_0$-measurable. Then $\Pr(p \le t \mid \mathcal{G}_0) \le t$
for every $t \in [0, 1]$, every $n$ and every $B$, and likewise for $p_d$.
\end{lemma}

\begin{proof}
Applying Lemma~\ref{lem:fresh} unit by unit, conditionally on $\mathcal{G}_0$ the modes are
independent with propensities $\pi_u$. The observed assignment and the $B$ redraws are then independent
and identically distributed given $\mathcal{G}_0$, and $z$ is the same fixed function of each, so
the $B + 1$ values $|z|, |z_1|, \dots, |z_B|$ are exchangeable given $\mathcal{G}_0$. The Monte Carlo
p-value of Equation~\eqref{eq:pval}, which counts ties against $|z|$, is then valid for every
$B$~\cite[Sect.~4]{phipson2010permutation}.
\end{proof}
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
