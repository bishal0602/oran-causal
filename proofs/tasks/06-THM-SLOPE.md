---
id: THM-SLOPE
kind: lean
status: needs clarity
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

Completed with the author-approved bounded-centre correction and episode blocking. The follow-up proves the bounded martingale weak law and removes that premise from corrected convergence.
See `../SLOPE_REPORT.md` for the mathematical statements and source-to-Lean mapping.

| Field | Value |
|---|---|
| Status | `needs clarity` |
| Main declaration(s) | `OranCausal.Slope.unbounded_centre_counterexample`; `counterexample_not_convergent`; `half_propensity_identity`; `half_propensity_target`; `EpisodeDesign.corrected_convergence`; `bounded_martingale_weak_law` (all in `OranCausal.Slope`) |
| Files | `proofs/OranCausal/Slope.lean`; import in `proofs/OranCausal.lean`; `proofs/SLOPE_REPORT.md`; `proofs/task06-build.log`; `proofs/task06-axioms.log`; `proofs/task06-workspace-check.log`; `proofs/task06-followup-build.log`; `proofs/task06-followup-axioms.log` |
| Deviation from the claim | added hypothesis: bounded centre (\|ỹ_u\| ≤ C); the paper's proof needs it for bounded increments; counterexample in `OranCausal.Slope.unbounded_centre_counterexample`. The half-propensity identity is proved without boundedness. Centred errors are grouped by episode, not treated as unit-level martingale differences. |
| Hypotheses not stated in the claim | Explicit source facts authorised by the user: H1, each outcome is observed by its episode end; H2, at most `N_max` units per episode; H3, episodes are appended in order and the previous episode's history is contained in each unit's pre-opening history. Uniform boundedness and predictability of the centre. `BoundedMartingaleWeakLaw μ` is now proved by `bounded_martingale_weak_law` using conditional-expectation orthogonality, variance additivity, the bound n·K², and Chebyshev. No named library obligation or weak-law premise remains. |
| `lake build` | PASS, exit 0, 3123 jobs; no errors. Follow-up: `task06-followup-build.log`. Two cosmetic tactic-linter warnings at `Slope.lean:505` concern `<;>` versus `;`. |
| `#print axioms` | PASS for all twelve audited declarations, including the proved weak law and corrected convergence theorem: only `propext`, `Classical.choice`, `Quot.sound`; no `sorryAx`. See `task06-followup-axioms.log`. |
| Notes | Counterexample uses an explicitly constructed iid Bernoulli(1/2) assignment law, zero outcomes, direction +1, and centre `3^(u+1)` in zero-based Lean indices. Its lower bound holds on every assignment path; nonconvergence in probability is proved. No episode independence is assumed. `uv run oran-causal check` passed lint but stopped at an existing formatting issue in `packages/analysis/tests/test_system_stats.py:202`, outside the permitted edit scope; later checks did not run. Paper and task 05 unchanged; no commit or push. |
