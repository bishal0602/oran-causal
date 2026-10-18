---
id: THM-PMRT-ASY
kind: lean
status: needs clarity
---

# 07 THM-PMRT-ASY: Asymptotic validity of PMRT

## Claim (verbatim, `spec/claims.yaml`)

> Under (A1) to (A3), S is a martingale with predictable quadratic variation sigma^2; with (A4), z converges to N(0, 1), the redraw distribution converges uniformly to Phi in probability, and limsup Pr(p <= t) <= floor(t (B + 1)) / (B + 1) <= t (Proposition 1).

Registered dependencies: `ASM-A1-DESIGN`, `ASM-A2-NULL`, `ASM-A3-PRED`, `ASM-A4-REG`, `THM-FRESH`, `CIT-HH`, `CIT-DUR`.

## Source text (verbatim, comments removed; full files in `sources/`)

### Proposition `prop:pmrt` (`s04_pmrt.tex`)

```latex
\begin{proposition}[Asymptotic validity of PMRT]
\label{prop:pmrt}
Assume (A1) to (A3). Then $S = \sum_u v_u w_u$ is a martingale with respect to $(\mathcal{G}_u)$, and its predictable
quadratic variation is $\sigma^2$. Let (A4) also hold and the number of episodes grow.
\begin{enumerate}
\item The statistic $z = S / \sigma$ converges in distribution to $\mathcal{N}(0, 1)$.
\item Conditionally on the data, every redraw $z_b$ has mean 0 and variance 1, and its distribution function converges
  uniformly to $\Phi$ in probability.
\item For every fixed $B$ and every $t \in [0, 1]$, $\limsup \Pr(p \le t) \le \lfloor t(B + 1) \rfloor / (B + 1) \le t$.
  The same holds for $p_d$ for each $d$ when $o$ is fixed in advance.
\end{enumerate}
The p-values $p$ and $p_d$ are therefore asymptotically valid. The value $p$ differs from
$2\{1 - \Phi(|z|)\}$ only by a term that vanishes in probability and by Monte Carlo error.
\end{proposition}
```

### Setting, assumptions and proof (`app_a.tex`)

```latex
\subsection{Setting}
\label{app:proof_setting}

Fix a hypothesis $(f, r, k)$. For every key $\xi$ defined below, let $Y_\xi$ be the target that a unit of $f$ with
that key would have: KPI $k$ over the cells of relation $r$ in the pre- and post-windows around its opening time. The
sharp null $H_0(f, r, k)$ states that every $Y_\xi$ is the same under any assignment of $f$'s modes. It is stated over
keys rather than realised units, because which units are realised depends on the modes. In
Section~\ref{sec:inference}, $\mathcal{G}_0$ holds all randomness of the system except the draws that set the modes of
$f$, and $\mathcal{G}_u$ adds the modes $W_1, \dots, W_u$. The probability space carries the network's randomness and the
assignment draws of every family. The draws of family $f$ form a field $\{U_\xi\}$ of
independent Uniform$(0, 1)$ variables indexed by keys $\xi$ (episode, cell, xApp, opening
second), independent of $\mathcal{G}_0$, which holds all other randomness. Unit $u$ of
$f$, in information order (episode, opening time, cell), has key $\xi_u$ and mode
$W_u = \text{accept}$ if and only if $U_{\xi_u} < \pi_u$, and
$\mathcal{G}_u = \mathcal{G}_0 \vee \sigma(U_{\xi_1}, \dots, U_{\xi_u})$. This filtration
contains the one of Section~\ref{sec:inference}, built from the modes; every statement below
holds for both, because keys, propensities and weights depend on the draws only through the
modes. Distinct units of
$f$ have distinct keys, because a cell-xApp pair has at most one active unit. Episodes are
appended in a fixed order as $n$ grows, so the units of the first $n$ episodes are an initial
segment of one sequence and $\mathcal{G}_u$ does not depend on $n$. Write
$L_u = L(W_u)$, so that
$v_u = \mathrm{sgn}_u (L_u - \pi_u)$, with conditional variance $\pi_u (1 - \pi_u)$. Under $H_0(f, r, k)$ every $Y_\xi$ is
$\mathcal{G}_0$-measurable, so the target $Y_{\xi_u}$ of unit $u$, and hence the post-window part of $w_u$, is
$\mathcal{G}_{u-1}$-measurable when $\xi_u$ is, although it is observed after the unit opens. Only then may $w_u$ use
$u$'s own post-window, and validity concerns that case alone.

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

In the simulator the assumptions of Lemma~\ref{lem:fresh} hold because the trajectory up to an
opening time is a function of $\mathcal{G}_0$ and of the modes of earlier units. A configuration change takes effect at the next tick, the xApps and the incumbent read delivered reports only, and
a unit of $f$ that opens in the same second at a lower-indexed cell precedes $u$ in information
order. Assumption (A1) states this property for a general system. The propensity is 0.5 under the randomised
policy and within $[0.15, 0.85]$ under the incumbent.
Assumption (A3) holds by the construction of Table~\ref{tab:pipeline}, and (A4) is the one assumption that the
construction leaves open.
 The number $N$ of units of $f$ is
bounded by a deterministic $n N_{\max}$, where $N_{\max}$ is the number of cell-xApp pairs times the number of
minutes in an episode, since a pair opens at most one unit per 60~s. Setting $v_u w_u = 0$ for the indices
$N < u \le n N_{\max}$ keeps every statement below, because the event
$\{u \le N\}$ is $\mathcal{G}_{u-1}$-measurable.

\subsection{Assumptions in detail}
\label{app:assumptions}

Assumption (A4) of Section~\ref{sec:inference} states formally that, as $n$, the number of episodes, grows, there are constants $s_n \to \infty$ with $\sigma^2 / s_n^2 \to \eta^2$ in probability and
$0 < \eta^2 < \infty$ almost surely, that $\max_u |w_u| / s_n \to 0$ in probability, and that each $w_u$ is square
integrable; Section~\ref{sec:inference} states what the four assumptions require in operational terms.

\subsection{Proof of Proposition~\ref{prop:pmrt}}
\label{app:proof_prop}

\emph{Martingale.} By (A1), $\mathbb{E}[L_u \mid \mathcal{G}_{u-1}] = \pi_u$, so
$\mathbb{E}[v_u \mid \mathcal{G}_{u-1}] = 0$ and
$\mathbb{E}[v_u^2 \mid \mathcal{G}_{u-1}] = \pi_u (1 - \pi_u)$, using
$\mathrm{sgn}_u^2 = 1$. By (A3), $w_u$ is $\mathcal{G}_{u-1}$-measurable, so
$\zeta_u = v_u w_u$ satisfies $\mathbb{E}[\zeta_u \mid \mathcal{G}_{u-1}] = 0$ and
$\sum_u \mathbb{E}[\zeta_u^2 \mid \mathcal{G}_{u-1}] = \sum_u \pi_u (1 - \pi_u)\, w_u^2 = \sigma^2$.
These identities are exact for every $n$.

\emph{Part 1.} Let $X_{n,u} = \zeta_u / s_n$. Since $|v_u| \le 1$,
\begin{equation*}
\sum_u \mathbb{E}\bigl[X_{n,u}^2 \mathbf{1}\{|X_{n,u}| > \gamma\} \mid \mathcal{G}_{u-1}\bigr]
\le \frac{\sigma^2}{s_n^2}\, \mathbf{1}\Bigl\{\max_u |w_u| > \gamma s_n\Bigr\} \to 0
\end{equation*}
in probability for every $\gamma > 0$ by (A4), and
$\sum_u \mathbb{E}[X_{n,u}^2 \mid \mathcal{G}_{u-1}] = \sigma^2 / s_n^2 \to \eta^2$ in probability.
The histories $\mathcal{G}_u$ do not depend on $n$, so they are nested in $n$, and the rows have the
deterministic length $n N_{\max}$ (Appendix~\ref{app:proof_setting}). The martingale central limit theorem with a random
limiting variance~\cite[Cor.~3.1]{hall1980martingale} then gives $S / s_n \to \eta Z$ stably, with
$Z \sim \mathcal{N}(0, 1)$ independent of $\eta$, and since $\eta^2 > 0$ almost surely its self-normalised
form~\cite[Cor.~3.2]{hall1980martingale} gives $z = S / \sigma \to Z$ in distribution. The event $\sigma = 0$, on which the code
sets $z = 0$ and $p = 1$, has probability tending to 0.

\emph{Part 2.} Condition on the data. A redraw sets $W_u^{(b)} = \text{accept}$ with probability
$\pi_u$, independently over $u$ and of the data, and keeps $\mathrm{sgn}_u$, $\pi_u$, $w_u$ and
$\sigma$. Thus $z_b = \sigma^{-1} \sum_u w_u v_u^{(b)}$ is a sum of independent terms with
mean 0 and variances $w_u^2 \pi_u (1 - \pi_u) / \sigma^2$ summing to 1. Its Lindeberg
quantity is at most $\mathbf{1}\{\max_u |w_u| > \gamma \sigma\}$, which tends to 0 in
probability because $\max_u |w_u| / s_n \to 0$ and $\sigma / s_n \to \eta > 0$. Along a further subsequence of every subsequence it tends to 0 almost surely for every
rational $\gamma$, and there the Lindeberg--Feller theorem~\cite[Thm.~3.4.10]{durrett2019probability} and
P\'olya's theorem~\cite[Exercise~3.2.9]{durrett2019probability} give $\sup_x |\Pr^*(z_b \le x) - \Phi(x)| \to 0$
almost surely, where $\Pr^*$ is the redraw law; hence the convergence holds in probability.

\emph{Part 3.} The code counts $|z_b| \ge |z| - 10^{-9}(1 + |z|)$; the tolerance
can only increase $p$, so it suffices to prove the claim for the count $Q$ without it, which
is Equation~\eqref{eq:pval}. Given the data, the $B$ redraws are independent and identically
distributed, so $Q$ is Binomial$(B, \Pi_n)$ with $\Pi_n = \Pr^*(|z_b| \ge |z|)$. By part 2,
$|\Pi_n - 2\{1 - \Phi(|z|)\}| \to 0$ in probability, and by part 1,
$2\{1 - \Phi(|z|)\}$ converges in distribution to a Uniform$(0, 1)$ variable $V$; so $\Pi_n \to V$ in
distribution. For each $j \in \{0, \dots, B\}$ the map
$x \mapsto \binom{B}{j} x^j (1 - x)^{B - j}$ is bounded and continuous on $[0, 1]$, so
\begin{equation*}
\Pr(Q = j) = \mathbb{E}\Bigl[\binom{B}{j} \Pi_n^j (1 - \Pi_n)^{B - j}\Bigr]
\to \int_0^1 \binom{B}{j} x^j (1 - x)^{B - j}\, dx = \frac{1}{B + 1}.
\end{equation*}
Thus $p = (1 + Q) / (B + 1)$ converges in distribution to the uniform law on
$\{1/(B+1), \dots, 1\}$. Because $p$ takes values in this fixed finite set,
$\Pr(p \le t) \to \lfloor t (B + 1) \rfloor / (B + 1) \le t$ uniformly in $t$. For $p_d$ replace
$\Pi_n$ by $\Pr^*(d\, o\, z_b \ge d\, o\, z)$, which is within $o_p(1)$ of $1 - \Phi(d\, o\, z)$;
since $|d\, o| = 1$, $d\, o\, z \to \mathcal{N}(0, 1)$ and the same argument applies. The last
sentence of the proposition follows from $|\Pi_n - 2\{1 - \Phi(|z|)\}| \to 0$ in probability and
$\mathbb{E}[Q / B \mid \text{data}] = \Pi_n$.
\hfill$\square$
```

## Scope

- Prove the registered statement for arbitrary values of its parameters; do not specialise to the numbers of the study unless the statement itself is about specific numbers.
- You decide the formal definitions and the declaration names. Each definition must mean what the claim and the source text say; definitions the claim relies on are in `sources/`.
- If the claim as written is false, ambiguous, or needs a hypothesis it does not state, do not silently change it. Prove the closest correct version, report the exact difference (with a counterexample where one exists) and set the status to `needs clarity`.
- If the proof needs a result that Mathlib v4.35.0 does not contain, you may take it as an explicit, named hypothesis of the theorem (never an `axiom`). State it in mathematical form in the report, give the reference it corresponds to, and set the status to `needs clarity` so the authors decide whether that is acceptable.

## Acceptance

As in `README.md`: builds, no `sorry`, axioms limited to `propext`, `Classical.choice`, `Quot.sound`, result table filled in.

## Result

Resumed the previous agent's passing implementation, reviewed its hypotheses,
expanded the axiom audit, and documented the proof and remaining qualifications
in `../ASYMPTOTIC_REPORT.md`.

| Field | Value |
|---|---|
| Status | `needs clarity` |
| Main declaration(s) | `OranCausal.Asymptotic.pmrt_martingale`; `self_normalized_clt`; `standardized_redraw_moments`; `redraw_cdf_limit`; `redraw_score_cdf_limit`; `conditional_binomial_rejection_limit`; `asymptotic_rejection_bound`; `fixed_direction_score_limit`; `zero_variance_vanishes` (all in `OranCausal.Asymptotic`) |
| Files | `proofs/OranCausal/Asymptotic.lean`; import in `proofs/OranCausal.lean`; `proofs/ASYMPTOTIC_REPORT.md`; `proofs/task07-resume-build.log`; `proofs/task07-axioms.log` |
| Deviation from the claim | The A1–A3 martingale assertion explicitly uses square-integrable weights, which A4 supplies but A1–A3 alone do not. Exact redraw variance one is proved on `sigma > 0`; on `sigma = 0`, `zero_redraw_moments` proves variance zero. A4 makes this event negligible, and conservative reporting preserves the final bound. The report gives counterexamples to both unqualified finite-sample statements. |
| Hypotheses not stated in the claim | Named external theorem obligations `HallHeyde mu` (Hall–Heyde Corollaries 3.1/3.2), `LindebergFellerPolya nu` (Durrett Theorem 3.4.10 and Exercise 3.2.9), and `Polya` (general continuous-limit version). No new Lean axioms. Stages explicitly take measurability of conditional redraw quantities and the conditional independent Bernoulli/product law describing the redraw experiment; construction of one probability kernel and end-to-end discharge of these interfaces are not packaged. Exact statements and source mapping are in the report. |
| `lake build` | PASS: exit 0, 3302 jobs, `task07-resume-build.log`. No Task 07 warnings or errors; two existing Task 06 tactic-linter warnings were replayed. |
| `#print axioms` | PASS: all fourteen audited declarations use only `propext`, `Classical.choice`, and `Quot.sound`; no `sorryAx`. See `task07-axioms.log`. Forbidden-proof-feature scan clean. |
| Notes | Proves the beta integral and binomial mixture, the probability integral transform, and the fixed-grid probability limit, including grid endpoints. `asymptotic_rejection_bound` gives limsup Pr(p ≤ t) ≤ floor(t(B+1))/(B+1) ≤ t for fixed B and t in [0,1], conditional on its explicit interfaces. This supplies the super-uniformity conclusion for Task 08. Fixed-sign and fixed random-orientation score-limit helpers are included; no per-sample orientation selection is justified. No paper edits, commits, or pushes. |
