# Task 08: asymptotic FDR control (Corollary `cor:fdr`)

Status: **needs clarity** (see "Status" below). The proofs are complete and have no `sorry`. The theorems are in
`OranCausal/FDRAsymptotic.lean`, namespace `OranCausal.FDRAsymptotic`, imported by `OranCausal.lean`.
Source: the current paper (`paper/sections/s04_pmrt.tex`, `app_a.tex`), as the authors decided. The drift from the
pinned sources is listed below.

## Decisions taken with the authors

| Question | Decision |
|---|---|
| Uniformity and the moving grid | Weights `w n j` form a deterministic sequence in n. Each null has its own fixed B. Uniformity in t is proved, not assumed |
| "Fixed from data independent of the evaluation data" | `w` and `d` are deterministic, so the theorem is conditional on the training data. Independence is what justifies that conditioning; it is not formalised |
| Per-null input | Each true null supplies the conclusions task 07 proves (`NullLimit`). Four theorems derive these from task 07's hypotheses |
| Rejection set | The weighted BY step-up is defined and proved self-consistent, then fed to task 01's bound |
| Untested hypotheses | p-value `+∞`: such a j is outside the tested set `T n ω` and is never rejected |
| H0, m, q | H0 and m fixed in n; m >= 1 arbitrary (not 60); q >= 0 |

## Formal statement

The setting is one probability space and a sample-size index n, as in task 07. For each j there is a used p-value
`p j n` (the tested-or-1 value) and a measurable tested set `T n`. The rejection set is
`rejections q w p T n = stepUp q (w n) (p · n) (T n)`, where `stepUp` takes ρ = max{k <= m : #{j ∈ T : p_j <= t_j(k)}
>= k}, rejects {j ∈ T : p_j <= t_j(ρ)}, and uses t_j(k) = q w_j k / (m c_m).

`NullLimit μ p` holds when some fixed `B` and counts `Q n` satisfy two conditions:
- for every t ∈ [0,1], `Pr((1+Q_n)/(B+1) <= t) → gridCDF B t`, which is the first conclusion of
  `Asymptotic.conditional_binomial_rejection_limit`;
- `p_n >= (1+Q_n)/(B+1)` pointwise, which is the `hconservative` form of `Asymptotic.asymptotic_rejection_bound`.

| Declaration | Statement |
|---|---|
| `limsup_fdr_le` (main) | w_n >= 0, Σ_j w_{n,j} = m for all n, `NullLimit` for every j ∈ H0 ⇒ limsup FDR_n <= limsup_n (q/m) Σ_{H0} w_{n,j} <= q, under any dependence |
| `limsup_fdr_le_of_tendsto` | in addition w_n → w₀ ⇒ limsup FDR_n <= (q/m) Σ_{H0} w₀_j <= q (w₀ >= 0 and Σ w₀ = m are derived) |
| `limsup_fdr_le_const` | weights fixed in n: the paper's statement, limsup FDR_n <= (q/m) Σ_{H0} w_j <= q |
| `fdr_le_add_vanishing` | the core: FDR_n <= (q/m) Σ_{H0} w_{n,j} + δ_n with δ_n >= 0 and δ_n → 0, for any w_n >= 0 |
| `fdr_le_slack` | the finite BY/BR bound at each n (task 01 with δ = `slack n`); measurability of the rejection set is proved |
| `card_stepUp`, `stepUp_selfConsistent`, `mem_stepUp`, `not_mem_rejections_of_untested` | \|S\| = ρ, rejections meet t_j(\|S\|), S = {tested j : p_j <= t_j(ρ)}, untested never rejected |
| `excess_le_latticeSlack`, `tendsto_latticeSlack`, `uniform_excess_of_nullLimit` | lattice lemma: limsup_n sup_{t>=0} (F_n(t) − t) <= 0 for fixed B |
| `nullLimit_of_binomial`, `nullLimit_of_pmrt`, `nullLimit_of_twoSided_pmrt`, `nullLimit_of_directed_pmrt` | `NullLimit` from task 07's hypotheses: generic; the full `asymptotic_rejection_bound` bundle; two-sided p (score −\|z\|); one-sided p_d (score −d o z, d fixed in advance, o fixed across n) |

Proof route: (1) the step-up is self-consistent; (2) task 01's bound holds at every n with the slack δ_n; (3) for
fixed B, task 07's pointwise limit is lifted to uniformity in t; (4) the slack vanishes even though the thresholds
move with n; (5) the limsup follows, with FDR in [0,1] and the level in [0,q]. Directions and the choice between p and
p_d enter only through which task 07 instantiation supplies `NullLimit` for j. With d deterministic, that choice is
fixed in advance.

## Deviations and qualifications

1. **Middle term for moving weights.** The paper's weights move with n (the "projected weights"), but `eq:fdr` writes
   (q/m) Σ_{H0} ω_j as if ω were fixed. The main theorem therefore reads limsup FDR <= limsup_n (q/m) Σ_{H0} w_{n,j}
   <= q. The paper's exact middle term holds for convergent or constant weights (the two corollaries).
2. **Uniformity only for t >= 0.** For t < 0, F(t) − t = −t is positive. This is enough, because every threshold
   t_{j,i} = q w_{n,j} i / (m c_m) is >= 0 when q >= 0 and w >= 0.
3. **Prop. `prop:pmrt` item 3 is pointwise; the proof cites it as uniform.** The appendix says "Proposition gives
   limsup_n sup_t {F_j(t) − t} <= 0, uniformly in t". The proposition (and task 07) states the bound for each fixed t
   and fixed B only. The gap is closed for fixed B by the lattice lemma. The raw p-value lives on {k/(B+1)}, and
   rep >= raw, so Pr(rep <= t) <= Pr(raw <= ⌊t(B+1)⌋/(B+1)), and sup_t reduces to a maximum over B+2 points. B
   growing with n is not covered.
4. **Task 01 file, additive change.** Public lemmas `StepUp.fdr_nonneg` and `StepUp.fdr_le_one` were added to
   `StepUp.lean`, because `fdr`'s body is not exposed across modules and the limsup needs FDR >= 0. No existing
   statement, name or visibility changed. Task 01's acceptance was re-run (build and axioms clean) and noted in its
   Result notes.
5. **Inherited from task 07.** Deriving `NullLimit` from A1 to A4 goes through task 07's interfaces and its named
   obligations (`HallHeyde`, `LindebergFellerPolya`, `Polya`) and its two qualifications. Task 07 is `needs clarity`.
6. **Hypotheses not in the claim.** m >= 1, q >= 0, measurable `p j n` and `T n`, deterministic weights and
   directions, and fixed B per null.

## Source drift (pinned `8744ef6` vs current paper)

`cor:fdr` and `app:proof_cor` are textually unchanged; only `\claim` tags were added. The nearby changes are:
- (A3) now includes "square integrable" (formerly in the A4 paragraph of app_a);
- Prop. item 2 now says variance 1 "where σ > 0" (task 07's qualification);
- s04 says the fitted parts were trained on episodes "separate from the evaluation data" (formerly "fixed before any
  evaluation data were generated");
- step 5 of `tab:pipeline` uses a shrunk running mean.

None of these changes the corollary.

## Status

`needs clarity`, for deviations 1 and 3 (a notational difference and a proof-step difference, to be reflected in the
paper) and dependency 5. The authors may set it to `implemented` if they accept them.

## Validation

`lake build` passes (3303 jobs), recorded in `task08-build.log`. The new module has no warnings; the build replays the
two known task 06 linter warnings. `task08-axioms.log` holds 17 audited declarations (3 in StepUp, 14 in
FDRAsymptotic), all with `propext`, `Classical.choice`, `Quot.sound` only. The forbidden-feature scan is clean
(`task08-forbidden-scan.log`). The per-step logs are `task08-step1..7-build.log`. Nothing was committed; no
dependency pins or paper files were changed.
