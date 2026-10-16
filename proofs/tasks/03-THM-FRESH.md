---
id: THM-FRESH
kind: lean
status: todo
---

# 03 THM-FRESH: Fresh draws

## Claim (verbatim, `spec/claims.yaml`)

> If xi_u and pi_u are G_{u-1}-measurable and xi_u differs from the earlier keys, then U_{xi_u} is Uniform(0, 1) and independent of G_{u-1}, so Pr(W_u = accept | G_{u-1}) = pi_u (Lemma 1).

## Source text (verbatim, comments removed; full files in `sources/`)

### Lemma `lem:fresh` and proof (`app_a.tex`)

```latex
\begin{lemma}[Fresh draws]
\label{lem:fresh}
If $\xi_u$ and $\pi_u$ are $\mathcal{G}_{u-1}$-measurable and $\xi_u$ differs from
$\xi_1, \dots, \xi_{u-1}$, then $U_{\xi_u}$ is Uniform$(0, 1)$ and independent of
$\mathcal{G}_{u-1}$, so $\Pr(W_u = \text{accept} \mid \mathcal{G}_{u-1}) = \pi_u$.
\end{lemma}

\begin{proof}
Fix a key $\xi$ and let $\mathcal{H}_\xi = \mathcal{G}_0 \vee \sigma(U_{\xi'} : \xi' \ne \xi)$,
which is independent of $U_\xi$. The keys $\xi_1, \dots, \xi_{u-1}$ and the values
$U_{\xi_1}, \dots, U_{\xi_{u-1}}$ are built recursively, each key from $\mathcal{G}_0$ and the
earlier values. On the event $E_\xi = \{\xi_u = \xi\}$ none of the earlier keys equals
$\xi$, so the recursion never reads $U_\xi$, and by induction on $u$ every event
$A \in \mathcal{G}_{u-1}$ satisfies $A \cap E_\xi \in \mathcal{H}_\xi$. Hence, for every Borel
set $C$,
$\Pr(A \cap E_\xi \cap \{U_{\xi_u} \in C\}) = \Pr(A \cap E_\xi)\, \lambda(C)$ with
$\lambda$ the Lebesgue measure, and summing over the countably many keys gives
$\Pr(A \cap \{U_{\xi_u} \in C\}) = \Pr(A)\, \lambda(C)$.
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
