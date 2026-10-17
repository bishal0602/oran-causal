---
id: THM-COUNTEREX
kind: lean
status: needs clarity
---

# 02 THM-COUNTEREX: Re-proposal counterexample

## Claim (verbatim, `spec/claims.yaml`)

> In a two-unit design where a rejected unit is proposed again, Pr(p_{+1} <= 1/2) = 3/4 > 1/2, so re-proposal alone breaks finite-sample validity (Example 1).

Registered dependencies: `DEF-PMRT-PVAL`.

## Source text (verbatim, comments removed; full files in `sources/`)

### Example `ex:adaptive` (`app_a.tex`)

```latex
\begin{example}[Re-proposal alone breaks finite-sample validity]
\label{ex:adaptive}
Let $\pi_u = 0.5$, $\mathrm{sgn}_u = +1$ and $o = +1$. Unit 1 has weight $w_1 = 1$. If it is
rejected, the request is proposed again and opens unit 2 with weight $w_2 = a > 1$; if it is
accepted, there is no unit 2. All weights are fixed in advance, so (A1) to (A3) hold. If unit 1 is
accepted, $z = 1$ and the redraws give $z_b = \pm 1$, so $p_{+1} = 1/2$. If unit 1 is rejected,
$z = (a - 1)/\sqrt{1 + a^2}$ when unit 2 is accepted and $z = -(a + 1)/\sqrt{1 + a^2}$ when it is
rejected; the redraws on the two realised units give $p_{+1} = 1/2$ and $p_{+1} = 1$. Hence, with
$B \to \infty$, $\Pr(p_{+1} \le 1/2) = 1/2 + 1/4 = 3/4 > 1/2$. The redraws treat unit 2 as if it
existed whatever unit 1 received, which is the source of the error, as in randomisation tests under
response-adaptive designs~\cite{simon2011randomization}; as the number of units grows,
the martingale central limit theorem removes it.
\end{example}
```

## Scope

- Prove the registered statement for arbitrary values of its parameters; do not specialise to the numbers of the study unless the statement itself is about specific numbers.
- You decide the formal definitions and the declaration names. Each definition must mean what the claim and the source text say; definitions the claim relies on are in `sources/`.
- If the claim as written is false, ambiguous, or needs a hypothesis it does not state, do not silently change it. Prove the closest correct version, report the exact difference (with a counterexample where one exists) and set the status to `needs clarity`.
- The example states its own numbers; prove it for the design it describes, for every weight `a > 1`, in the limit `B -> infinity` the example uses.

## Acceptance

As in `README.md`: builds, no `sorry`, axioms limited to `propext`, `Classical.choice`, `Quot.sound`, result table filled in.

## Result

Filled in by the agent.

| Field | Value |
|---|---|
| Status | needs clarity for the pinned source's boundary statement; both original interpretations and the deferred off-boundary result are proved |
| Main declaration(s) | `OranCausal.Counterexample.exact_tail_rejection`; `finite_redraw_rejection_limit`; `finite_redraw_rejection_formula`; new `finite_redraw_rejection_limit_above_half` and `finite_redraw_rejection_limit_between_half_and_three_quarters` (all in `OranCausal.Counterexample`) |
| Files | `proofs/OranCausal/Counterexample.lean`, imported by `proofs/OranCausal.lean`; `proofs/COUNTEREX_REPORT.md`; `proofs/task02-build.log`; new `proofs/task02-followup-build.log` and `proofs/task02-followup-axioms.log` |
| Deviation from the claim | Exact-tail result: `Pr(p∞ ≤ 1/2) = 3/4`. For the source's plus-one Monte Carlo p-value, `lim_B Pr(pB ≤ 1/2) = 3/8`, not `3/4`. For odd `B` the finite probability is `3/8`; for `B = 2n` it is `(3/8)(1 - choose(2n,n)/4^n)`. **Deferred item completed:** for every `a > 1` and fixed `1/2 < t < 3/4`, `Pr(pB ≤ t) → 3/4`. The stronger result covers all `1/2 < t < 1`. Thus the plus-one procedure is asymptotically anti-conservative at every requested cutoff, while its boundary limit remains different. |
| Hypotheses not stated in the claim | `a : ℝ`, `1 < a`; fair sequential assignments, first rejection triggers the second unit, fixed weights `1,a`, positive direction and orientation, redraws hold the observed unit set fixed, and conditionally iid redraws. These are specified by the source; `a` is arbitrary. |
| `lake build` | Follow-up PASS, exit 0, 3123 jobs; no errors or task-02 warnings. Existing cosmetic warnings in task 06 were replayed unchanged. See `task02-followup-build.log`. Original validation is retained in `task02-build.log`. |
| `#print axioms` | All four audited limit/exact-tail declarations, including both new limit theorems: only `[propext, Classical.choice, Quot.sound]`; no `sorryAx`. See `task02-followup-axioms.log`. |
| Notes | No weak-law premise or other unproved probability lemma is needed. A finite binomial second-moment identity gives `E[(pB-1/2)^2] ≤ 1/(2(B+1))`; on each fair-tail branch the nonrejection probability is at most `1/(2(B+1)(t-1/2)^2)`, hence tends to zero. The remaining branch has `pB=1` and never rejects for `t<1`. `monteCarloRejectionAt_half` connects the new general-threshold probability to the original definition. The pinned claim and source are unchanged; `needs clarity` concerns their boundary wording, not the completed deferred result. See `COUNTEREX_REPORT.md`. |
