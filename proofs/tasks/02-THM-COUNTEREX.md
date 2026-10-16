---
id: THM-COUNTEREX
kind: lean
status: todo
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
| Status | |
| Main declaration(s) | |
| Files | |
| Deviation from the claim | |
| Hypotheses not stated in the claim | |
| `lake build` | |
| `#print axioms` | |
| Notes | |
