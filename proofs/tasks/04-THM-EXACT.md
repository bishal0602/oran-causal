---
id: THM-EXACT
kind: lean
status: implemented
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
| Status | implemented; the author explicitly accepted the threshold and orientation conditions as source context |
| Main declaration(s) | `OranCausal.Exact.exact_redraw_pvalues`; full-slot version `OranCausal.Exact.fixed_design_exact` |
| Files | `proofs/OranCausal/Exact.lean`, imported by `proofs/OranCausal.lean`; `proofs/EXACT_REPORT.md`; `proofs/task04-build.log`; `proofs/task04-axioms.log`; small history-interface extension in `proofs/OranCausal/Fresh.lean` |
| Deviation from the claim | None under the approved source context. Both conclusions are almost-sure bounds on Mathlib conditional expectations of rejection indicators. The theorem includes an explicit initial-measurable random unit set and proves that padding absent slots with zero weights gives exactly the realised-unit statistic. |
| Hypotheses not stated in the claim | The full source setup is made explicit below. Uniform-field independence plus fresh initial-measurable keys yields the conditional Bernoulli law using task 03; the conditional joint product law and rank symmetry are then proved rather than assumed. |
| Assumption — probability space and initial information | Arbitrary probability space; `G` is an initial sub-sigma-algebra. Source: `sources/app_a.tex:68–75`. |
| Assumption — independent uniform field | Countable keys, measurable independent Uniform(0,1) coordinates, jointly independent of `G`. Source: `sources/app_a.tex:70–72,104`. |
| Assumption — fixed realised units and order | Presence events and ordered draw keys are `G`-measurable; observed keys are distinct. Source: `sources/app_a.tex:78–79,189–191`. |
| Assumption — independent redraw randomness | Redraws use fresh independent uniforms, with the same slot propensities; independent over units, of the data, and across replicas. Source: `sources/app_a.tex:153–155,165–166,195–198`. Replica labels can be included in draw keys. |
| Assumption — initial-measurable design values | Each `sgn_u`, `pi_u`, and `w_u` is `G`-measurable. Source: `sources/app_a.tex:189–191`. |
| Assumption — propensity range and A1 | Formal range `[0,1]` includes the source's stricter `[pi_min,1-pi_min]`. The A1 conditional draw law is derived from the uniform field and task 03. Source: `sources/s04_pmrt.tex:86–88`. |
| Assumption — sharp-null/common statistic | The null's relevant consequence is represented by initial-measurable data and the same fixed statistic for observed and redrawn assignments; no separate potential-outcome predicate is needed once those explicit fixed-design premises hold. Source: `sources/s04_pmrt.tex:80–89`; `sources/app_a.tex:64–67,83–86,195–198`. |
| Assumption — finite unit bound | Arbitrary `M` bounds the realised unit slots; initially measurable absence is represented by zero weights. Source: `sources/app_a.tex:115–119`. Every finite episode count is covered through its bound; no study-specific count is used. |
| Assumption — score and p-value definitions | Exact PMRT score/variance, plus-one denominator `B+1`, and tie-inclusive exceedance counts. Source: `sources/s04_pmrt.tex:55–60,71–77`. |
| Assumption — fixed direction and training orientation | `d` is fixed; `o` is `G`-measurable. The proof allows arbitrary real values and therefore covers the stipulated `±1` case. Source: `sources/s04_pmrt.tex:63–65,73–77,110`; explicitly accepted by the author. |
| Assumption — threshold and redraw count | `t ∈ [0,1]`, arbitrary natural `B`, including zero. Source: `sources/app_a.tex:190–191`; explicitly accepted by the author. |
| `lake build` | Passed without warnings or errors, 3031 jobs; 29.662 seconds wall time with cached dependencies. Separate source recheck also passed. |
| `#print axioms` | Both principal theorems: `[propext, Classical.choice, Quot.sound]` |
| Notes | Ties, zero variance, empty realised unit sets, and all finite `B` are covered. `Fresh.history` is exposed and its zero/successor equations exported for cross-module reuse; task 03's main theorem statement and proof are unchanged. The registry and manuscript source text were not edited. |
