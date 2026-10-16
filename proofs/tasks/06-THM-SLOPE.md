---
id: THM-SLOPE
kind: lean
status: todo
---

# 06 THM-SLOPE: Design slope

## Claim (verbatim, `spec/claims.yaml`)

> Under (A1) with the natural history, bounded outcomes and a predictable centre, beta_hat converges to the pi_u (1 - pi_u)-weighted average of sgn_u tau_u^pi; at pi_u = 0.5 it is the Horvitz-Thompson estimator as an identity (Lemma 3).

Registered dependencies: `ASM-A1-DESIGN`, `DEF-SLOPE`.

## Source text (verbatim, comments removed; full files in `sources/`)

### Lemma `lem:slope` and proof (`app_a.tex`, `app:proof_slope`)

```latex
\subsection{The design slope}
\label{app:proof_slope}

\begin{lemma}[Design slope]
\label{lem:slope}
Assume (A1) with the natural history $\mathcal{F}_{u-1}$ in place of $\mathcal{G}_{u-1}$, bounded
outcomes, a centre $\tilde y_u$ that is $\mathcal{F}_{u-1}$-measurable, $N / n$ bounded away from 0 and
infinity with probability tending to one. Then
$\hat\beta$ of Equation~\eqref{eq:slope_ht} satisfies
\begin{equation*}
\hat\beta - \frac{\sum_u \pi_u (1 - \pi_u)\, \mathrm{sgn}_u\, \tau_u^{\pi}}{\sum_u \pi_u (1 - \pi_u)} \to 0
\end{equation*}
in probability. When $\pi_u = 0.5$ for every unit the right-hand fraction is $\bar\tau_{\pi}$
and Equation~\eqref{eq:slope_ht} holds as an identity.
\end{lemma}

\begin{proof}
Since $\tilde y_u$ is $\mathcal{F}_{u-1}$-measurable and $\mathbb{E}[v_u \mid \mathcal{F}_{u-1}] = 0$,
$\mathbb{E}[v_u e_u \mid \mathcal{F}_{u-1}] = \mathrm{sgn}_u \{(1 - \pi_u)\, \mathbb{E}[y_u \mathbf{1}\{W_u = \text{accept}\} \mid \mathcal{F}_{u-1}]
- \pi_u\, \mathbb{E}[y_u \mathbf{1}\{W_u = \text{reject}\} \mid \mathcal{F}_{u-1}]\}
= \pi_u (1 - \pi_u)\, \mathrm{sgn}_u\, \tau_u^{\pi}$, and
$\mathbb{E}[v_u^2 \mid \mathcal{F}_{u-1}] = \pi_u (1 - \pi_u)$. The sums
$\sum_u \{v_u e_u - \pi_u (1 - \pi_u) \mathrm{sgn}_u \tau_u^{\pi}\}$ and
$\sum_u \{v_u^2 - \pi_u (1 - \pi_u)\}$ are martingales with bounded increments over at most $n N_{\max}$ terms (Appendix~\ref{app:proof_setting}), so
their variances are $O(n)$ and, divided by $N$, they tend to 0 in probability by Chebyshev's inequality. Overlap keeps $N^{-1} \sum_u \pi_u (1 - \pi_u) \ge \pi_{\min}(1 - \pi_{\min})$, and bounded outcomes keep
$N^{-1} \sum_u \pi_u (1 - \pi_u) \mathrm{sgn}_u \tau_u^{\pi}$ bounded, which gives the limit of the ratio. With $\pi_u = 0.5$, $v_u^2 = 1/4$ and
$4 v_u = 2\, \mathrm{sgn}_u (2 L_u - 1)$, which is Equation~\eqref{eq:slope_ht}.
\end{proof}
```

## Scope

- Prove the registered statement for arbitrary values of its parameters; do not specialise to the numbers of the study unless the statement itself is about specific numbers.
- You decide the formal definitions and the declaration names. Each definition must mean what the claim and the source text say; definitions the claim relies on are in `sources/`.
- If the claim as written is false, ambiguous, or needs a hypothesis it does not state, do not silently change it. Prove the closest correct version, report the exact difference (with a counterexample where one exists) and set the status to `needs clarity`.
- If the proof needs a result that Mathlib v4.35.0 does not contain, you may take it as an explicit, named hypothesis of the theorem (never an `axiom`). State it in mathematical form in the report, give the reference it corresponds to, and set the status to `needs clarity` so the authors decide whether that is acceptable.

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
