---
id: THM-BY-FIN
kind: lean
status: implemented
---

# 01 THM-BY-FIN: Finite weighted BY bound

## Claim (verbatim, `spec/claims.yaml`)

> For any rejection set S with p_j <= t_j(|S|) for every j in S, where t_j(k) = q w_j k / (m c_m), the false discovery rate is at most (q/m) sum_{j in H0} w_j + |H0| delta whenever every null p-value satisfies Pr(p_j <= t_j(i)) <= t_j(i) + delta for i = 1..m, whatever the dependence between the p-values.

Registered dependencies: `CIT-BY`, `CIT-BR`.

## Source text (verbatim, comments removed; full files in `sources/`)

### Corollary `cor:fdr` (`s04_pmrt.tex`)

```latex
\begin{corollary}[Asymptotic false discovery rate]
\label{cor:fdr}
Let $\mathcal{H}_0$ hold the true sharp nulls among the $m = 60$ hypotheses. Let (A1) to (A4) hold for each of them,
and let the weighted step-up use weights $\omega_j \ge 0$ with $\sum_j \omega_j = m$ and directions $d_j$ fixed from
data independent of the evaluation data. Then, whatever the dependence between the p-values,
\begin{equation}
\limsup_{n \to \infty} \mathrm{FDR} \le \frac{q}{m} \sum_{j \in \mathcal{H}_0} \omega_j \le q,
\label{eq:fdr}
\end{equation}
where a false discovery is the rejection of a true sharp null.
\end{corollary}
```

### Proof (`app_a.tex`, `app:proof_cor`)

```latex
\subsection{Proof of Corollary~\ref{cor:fdr}}
\label{app:proof_cor}

Let $c_m = \sum_{i=1}^m 1/i$ and $t_{j,i} = q\, \omega_j\, i / (m c_m)$ for $i = 1, \dots, m$, with $t_{j,0} = 0$. The
declaration layer is the weighted step-up with the reshaping function $i \mapsto i / c_m$, which rejects exactly the hypotheses
with $p_j \le t_{j,\rho}$, $\rho$ being the number of rejections, and which controls the FDR at
$q \sum_{j \in \mathcal{H}_0} \omega_j / m$ under arbitrary dependence when the null p-values are
super-uniform~\cite[Thm.~4.2]{blanchard2008fdr}. The p-value used for $j$, $p_{d_j}$ or $p$, is fixed in advance, and
an untested hypothesis is never rejected, which only lowers the FDR. Here the null p-values are super-uniform only in
the limit, so the bound is re-derived. Since $p_j \le t_{j,\rho}$ with $t_{j,i-1} < p_j \le t_{j,i}$ implies
$\rho \ge i$, the argument of~\cite{benjamini2001dependency} gives, whatever the joint law of the p-values and with
$F_j(t) = \Pr(p_j \le t)$,
\begin{equation*}
\mathbb{E}\Bigl[\frac{\mathbf{1}\{p_j \le t_{j,\rho}\}}{\max(\rho, 1)}\Bigr]
\le \sum_{i=1}^{m-1} F_j(t_{j,i}) \Bigl(\frac{1}{i} - \frac{1}{i+1}\Bigr) + \frac{F_j(t_{j,m})}{m}.
\end{equation*}
For $j \in \mathcal{H}_0$, Proposition~\ref{prop:pmrt} gives
$\limsup_n \sup_t \{F_j(t) - t\} \le 0$, uniformly in $t$, so the grid may move with $n$, as it does
through the projected weights. Substituting $F_j(t_{j,i}) \le t_{j,i} + o(1)$,
\begin{equation*}
\limsup_{n} \mathbb{E}\Bigl[\frac{\mathbf{1}\{p_j \le t_{j,\rho}\}}{\max(\rho, 1)}\Bigr]
\le \frac{q\, \omega_j}{m c_m} \Bigl( \sum_{i=1}^{m-1} \frac{1}{i+1} + 1 \Bigr) = \frac{q\, \omega_j}{m},
\end{equation*}
and summing over $j \in \mathcal{H}_0$ gives Equation~\eqref{eq:fdr}. The weights and directions
enter only through the fixed thresholds $t_{j,i}$, which is why they must not depend on the
evaluation data. \hfill$\square$
```

## Scope

- Prove the registered statement for arbitrary values of its parameters; do not specialise to the numbers of the study unless the statement itself is about specific numbers.
- You decide the formal definitions and the declaration names. Each definition must mean what the claim and the source text say; definitions the claim relies on are in `sources/`.
- If the claim as written is false, ambiguous, or needs a hypothesis it does not state, do not silently change it. Prove the closest correct version, report the exact difference (with a counterexample where one exists) and set the status to `needs clarity`.
- The asymptotic step (limit in `n`) is out of scope; it is task 08.

## Acceptance

As in `README.md`: builds, no `sorry`, axioms limited to `propext`, `Classical.choice`, `Quot.sound`, result table filled in.

## Result

Filled in by the agent.

| Field | Value |
|---|---|
| Status | implemented (independently re-checked 2026-10-16) |
| Main declaration(s) | `OranCausal.StepUp.fdr_le_of_selfConsistent` |
| Files | `proofs/OranCausal/StepUp.lean`, `proofs/REPORT.md` |
| Deviation from the claim | none |
| Hypotheses not stated in the claim | `m >= 1`, `q >= 0`, `w_j >= 0`, measurable `p_j` and `S`; `sum w_j = m` and `delta >= 0` not needed |
| `lake build` | clean rebuild passes; `leanchecker` replay exit 0 |
| `#print axioms` | `propext`, `Classical.choice`, `Quot.sound` |
| Notes | Commit `e2b20ee`. 2026-10-18 (task 08): added public lemmas `fdr_nonneg`, `fdr_le_one` to StepUp.lean (additive only; no existing statement, name or visibility changed); lake build passes; `#print axioms` of `fdr_le_of_selfConsistent`, `fdr_nonneg`, `fdr_le_one`: propext, Classical.choice, Quot.sound (proofs/task08-step7-build.log) |
