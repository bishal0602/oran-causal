# Cross-method discovery study: shared contract (orchestrator-owned; DRAFT, not frozen)

Approved plan (user, 2026-10-02): discovery only on E1-E5; data regimes R1/R2 (all worlds), R3/R4 (E4);
methods below; fidelity gates before any evaluation; one frozen protocol; EVAL on fresh seeds on Kaggle / Colab /
Lightning. Interface: `cdd_oran/xmethod/api.py` (do not edit; propose changes in your hand-back).

## 1. Hard rules for every worker

1. Work only in your own git worktree and branch (`xm/<your-name>`), created from `feat/v2` at the commit given in
   your brief. Commit there. NEVER push. NEVER merge. The orchestrator merges.
2. Never edit frozen files: `docs/benchmark/*.md` that say FROZEN, `docs/benchmark/artifacts/*`, frozen harnesses,
   `cdd_oran/envs/v2/e*.py` SCM mechanisms (add variants in new files instead).
3. Seeds: DEV only, from the block below. Never generate or read EVAL seeds. Never touch any other study's seeds
   (`docs/benchmark/SEED_REGISTRY.json`).
4. Heavy compute (anything over ~10 CPU-min) goes to Kaggle / Colab / Lightning via the existing runners
   (`scratchpad/e6_dev/kaggle_job.py`, `colab_run.py`, `lightning_run.py`; Kaggle CLI `~/.cloudtools/Scripts/kaggle.exe`;
   max 5 concurrent Kaggle sessions shared by ALL workers, so use at most 1 at a time unless your brief says more).
   Use single-quoted command strings for job launchers (a double-quoted `$JOB_SRC` expands locally and breaks the job).
5. Python: in your worktree run `uv sync` once, then `uv run python ...` / `uv run pytest -q tests/<yours>` (add deps with `uv add`, never pip; mention every added dependency in the hand-back). Kaggle/Colab/Lightning runners live in `scratchpad/e6_dev/` and work from your worktree.
6. If anything is ambiguous, or a rule blocks you, STOP and write the question in your status file; do not guess.
7. Status: keep `scratchpad/xmethod/status/<your-name>.md` current (what is done, what is running where, blockers,
   questions). When finished, end it with a `## HAND-BACK` section (deliverables, test results, fidelity-gate
   results, deviations, open questions). Keep it under 60 lines.

## 2. Worlds and regimes

| World | Mechanism isolated | SCM source (read-only) |
|---|---|---|
| E1 | clean identifiable graph | `cdd_oran/envs/v2/e1.py`, `cdd_oran/e1slice/` |
| E2 | gated edge active only in a rare regime | `cdd_oran/envs/v2/e2.py`, `cdd_oran/e2slice/dataset.py` |
| E3 | delayed harm (lagged effects) | `cdd_oran/envs/v2/e3.py` |
| E4 | latent confounding, association has the wrong sign; lambda sweep {0, .5, 1, 1.5} | `cdd_oran/envs/v2/e4.py`, `docs/benchmark/GATE_CONTRACT_E4.md` |
| E5 | E2-E4 mechanisms composed + observation noise | `cdd_oran/envs/v2/e5.py`, `scripts/e5_*` |

| Regime | Rows | Design (api.Design.kind) |
|---|---|---|
| R1 | actions i.i.d. random | `iid` |
| R2 | bounded setpoint (slow, not random) + random dither (Stage 0 F-dither, `docs/benchmark/STAGE0_RESULT.md`) | `dither` |
| R3 (E4) | context-dependent policy on an OBSERVED confounder, probabilities logged | `logged` |
| R4 (E4) | latent confounder, nothing logged (negative control: no method can identify) | `none` |

R3 needs an E4 variant with an observed confounder: a NEW file (e.g. `cdd_oran/xmethod/worlds/e4_logged.py`), the
original E4 SCM unchanged. Every dataset also carries one **placebo action column** `P_placebo` drawn with the same
design as the real actions and with no effect on anything (truth-free negative control, section 5).

## 3. Methods (adapter names)

| Family | Adapter | Owner |
|---|---|---|
| proposed | `pmrt_core` (design-based CRT redrawing the known random part + `cdd_oran/decision/fdr_layer.py`) | xm-pmrt |
| independence tests | `mscr`, `pcorr`, `pdcor`, `rcot2`, `cmi_knn` | xm-citests |
| classic causal discovery | `pc` (causal-learn), `notears` (authors' code) | xm-classic |
| published O-RAN | `shap_dag` (Sharma et al.); `two_tower` = neural relevance gate (ours, adaptation inspired by arXiv:2601.13213, not the published supervised model) | xm-classic |
| simple | `corr`; `granger` (E3 only) | xm-classic |

## 4. Fidelity gates (each adapter, before any EVAL; results in the hand-back)

F1 standard / author implementation where one exists (record package + version or commit).
F2 own port == reference on identical inputs (tolerance stated), e.g. RCoT vs R `RCIT` (R runs on Kaggle/Colab, not
   installed locally), pdCor vs `dcor`.
F3 reproduces a known result (its paper or its package's own tests).
F4 synthetic: finds planted edges; null false-positive rate at nominal on null data (state n, reps, CI).
F5 tuning by the section 5 rule only, on DEV.
F6 a fidelity-table row: source, version, every deviation and why.
(F7, independent audit, is run by the orchestrator after hand-back.)
A gate that fails is reported, not worked around.

## 5. Tuning and declaration rules (same for all)

- Methods with valid p-values declare at q = .05 with BY over the candidate set (and report raw p).
- Methods with scores only (and, as a secondary, the p-value methods): threshold tau = the smallest value such that
  the placebo column `P_placebo` gets at most 1 declaration in total over the DEV datasets of that (world, regime, n)
  cell (the analogue of E6's "far_fpr" rule). Truth is never used for tuning.
- No other per-method tuning on DEV beyond what the brief allows; hyperparameters are the authors' defaults unless
  the brief says otherwise (record any change in F6).

## 6. Seeds (DEV block for this study; registered by the orchestrator)

DEV: 3_000_000 - 3_000_199 for every world (corpus seeds); method RNG streams tag 7801 (pmrt_core), 7802 (citests),
7803 (classic), 7804 (cdl, R-50: torch seed = default_rng([7804, dataset seed, fit index])). EVAL block 3_100_000+
is RESERVED: do not use.

## 7. Sample sizes

n grid {500, 1000, 4000} for all methods; {8000, 24000} additionally for methods that scale (state which in the
hand-back, with measured CPU-s and peak RAM per run).

## 8. Rulings (orchestrator, 2026-10-02)

- R-1 E4 sign sanity: R1 (randomised action) gives the CORRECT sign at every lambda; the wrong sign appears in R4
  (and marginally in R3). The brief's "R1 wrong" was an error.
- R-2 Declarations: pooled BY over all candidates of a dataset (q .05) is primary for p-value methods; native rules
  kept in notes. Score-only rule: tau = 2nd-largest P_placebo score pooled over the cell's DEV datasets; declare iff
  score > tau.
- R-3 Conditioning set for CI tests: all other action columns (incl. P_placebo) + lagged KPIs (+ context in R3).
- R-4 Signs: every method reports a sign. Unsigned statistics take the sign of the partial correlation of
  (source, target) given the same conditioning set (notes['sign_rule'] = 'pcorr_given_Z'); NOTEARS uses its weight.
- R-5 pcorr p-value: partial-correlation t-test, df = n - 2 - |Z|.
- R-6 Primary comparison set and BY family = action->KPI candidates (incl. P_placebo); KPI->KPI candidates are a
  secondary analysis with their own BY family (PMRT core cannot test them: not randomised).
- R-7 R3 policy must be redrawable (categorical with logged per-row probabilities, or per-row parametric with logged
  parameters) and draws each action independently of the others given the observed context.
- R-8 PMRT core adjustment may use concurrent values of other actions whose designs are independent of the focal
  action given context (R1, R2, R3); documented as a reduction from E6.
- R-9 Resampling resolution: every resampling method (incl. pmrt_core) uses max B = 9999 with Besag-Clifford
  sequential stopping (h = 20 exceedances; p = h/k at stop, else (1 + count)/(B + 1)). Reason: at native B (299-2999)
  pooled BY over m = 20-54 candidates cannot declare an isolated edge (floor artifact, as in RCoT v1).
- R-10 Placebo columns: P_placebo (tuning) is always i.i.d. from the action's marginal design and independent of
  everything, in every regime. In R3/R4 an extra P_placebo_conf follows the confounded policy: a diagnostic negative
  control, scored apart, never used for tuning, not in the primary BY family. RNG tags 7800 (harness), 7899 (dummy).
- R-11 F3 reproductions may read stored datasets of completed older studies read-only (e.g. runs/e2slice-recovery, runs/e1slice-v2-envelope).
- R-12 (user-approved 2026-10-02) Seeds: the number of EVAL seeds per cell is set by a power calculation on DEV data
  (smallest difference of interest, e.g. recall gap .15, paired design), identical for all methods; no fixed 50.
- R-13 (user-approved 2026-10-02) Fairness for costly methods: same n grid, seeds, resolution (R-9) and declaration
  rule for all; each method uses its AUTHORS' own null where one exists (RCoT: Strobl et al. approximate null; pdCor:
  check the original paper for a non-permutation test); a per-(method, dataset) compute budget, fixed in the protocol
  (proposal 2 CPU-h) and identical for all methods: a cell a method cannot finish within the budget (measured on DEV)
  is reported as "infeasible at this n (measured cost X)", never silently dropped and never cut by hand.
- R-14 (user-approved 2026-10-02) PMRT core in the E-series runs WITHOUT the Huber clip (huber_c None = pmrt "plain"
  arm). Reason: with the clip, E2 R2 null rate .070 [.055,.087] at .05 (n 1000; finite-sample failure of the fixed-W
  CRT when W depends on the focal action's past dithers); without it .052. Disclosed in the paper.
- R-15 (user-approved 2026-10-02) Experiment D, E6 validity audit (descriptive, post-hoc, disclosed): on the existing
  v4 E6 data, raw-p rejection rates at .05 / .01 on GT-NULL hypotheses for pmrt loadsp_c (frozen primary) vs loadsp
  (no clip), with cluster-bootstrap CIs. Frozen verdicts unchanged; the measured rate is reported next to them.
- R-16 Inside ephemeral cloud kernels, pip install is allowed if pinned to the exact uv.lock version (logged).
- R-10 (amended 2026-10-02, per consult ex3): the tuning placebo P_placebo follows the SAME design as the real
  actions in R1/R2 (R2: its own setpoints + dither, as the harness implements), so it detects R2 failures; in R3/R4
  it is independent of the confounder (i.i.d. from the marginal design). P_placebo_conf as before.
- R-17 (user-approved 2026-10-02, after 3 independent expert opinions, scratchpad/xmethod/consult/) PRIMARY arm =
  EQUAL INFORMATION: every method with a conditioning-set or covariate interface receives exactly PMRT core's
  design covariates Z_eq = R-3 set + setpoints (Design.fixed_part) of all designed actions incl. the focal one +
  all actions at t-1 and t-2 (+ context in R3), via ONE shared helper (cdd_oran/xmethod/covariates.py), one
  pre-specified encoding per method, no DEV tuning of Z. Granger: Z_eq as exogenous regressors (conditional / VARX);
  PC: setpoints and lagged actions as exogenous tier-0 nodes (background knowledge). corr, NOTEARS, SHAP-DAG,
  two-tower have no conditioning interface: native only (disclosed).
- R-18 SECONDARY arm = NATIVE (R-3 conditioning sets as before): "as typically applied, design-blind".
- R-19 PMRT ablation: pmrt_core with R-3 covariates only (no setpoints, no lagged actions), completing the 2x2
  (information x inference engine).
- R-20 Recall / power is compared only among (method, arm, cell) whose null-edge and placebo rates are not
  significantly above .05; validity is reported for all.
- R-21 Before freeze: DEV validity check (placebo + truth-null rates) of every equal-information arm in R2 (and R3),
  n 500 / 1000 / 4000; reported, not tuned on.
- Claim (for the paper): "tests that ignore the randomization design are invalid on RIC-style setpoint + dither
  data; using the design restores validity; PMRT does so by construction, needs no outcome model, admits any
  statistic, and stays valid under logged confounded policies (R3)".
- R-22 Degenerate cells: where a conditional test has an exact fit (deterministic E1 / E3 under Z_eq), affected
  candidates return score NaN, p None, notes['not_testable'] and are not scorable; never a silent p = 1.
- R-23 Scoring of not-testable candidates: counted as NOT declared (true edge = miss, null = no false positive); n_not_testable_true / _null reported per cell.
- R-24 (user-approved 2026-10-02) Observation noise in ALL worlds (E1-E5), same model as E5_NOISE_DETECT:
  y_k <- y_k + N(0, (kappa * sigma_k)^2), sigma_k = the noiseless KPI's spread. Primary kappa = kappa_E6, CALIBRATED from
  E6-P data (not the earlier synthetic .1 / .3): unpredictable share u of each E6 KPI's variation given actions and
  recent state, kappa = sqrt(u / (1 - u)), summary across KPIs fixed in the calibration report. Sensitivity sweep
  kappa in {kappa_E6 / 2, 2 kappa_E6} at one n with fewer seeds. Noiseless versions = sanity check only.
- R-25 Helper usage: equal-information adapters take their WHOLE conditioning set from
  design_covariates(data, focal=<action>) (concurrent='designed', as pmrt); the native arm uses the R-3 set
  (concurrent='all'). Names: lag_kpi:<K>, ctx:<c>, sp:<P>, <P>@t-1, <P>@t-2, concurrent:<P>, <name>:missing.
  pmrt_core 'not_applicable' (no design) is distinct from R-22 'not_testable'; both count as not declared.
- R-26 Compute policy (user facts 2026-10-02: Colab free tier via API; Lightning 3.77 credits; Kaggle free incl.
  ~30 GPU-h/week): shard every job small and resumable (one (world, regime, n, seed, method) per unit, merge
  tolerant of missing / duplicate shards) so any platform can take any shard.
  * Kaggle: long unattended CPU jobs and all final EVAL shards (survives disconnects); also GPU jobs that must run
    unattended. Max 5 concurrent sessions shared by all agents.
  * Colab (free, API): parallel capacity for medium jobs (aim < 3 h per session) and GPU work (T4): GPU ports
    (CMI kNN neighbour search, two-tower torch, XGBoost gpu_hist) and their reference checks. Expect reclaimed
    sessions: checkpoint per shard, relaunch on failure.
  * Lightning (3.77 credits): reserve only; small urgent jobs when Kaggle and Colab are saturated; ask the
    orchestrator before spending more than 1 credit.
  * GPU versions of a method must reproduce the CPU reference (F2-style check, tolerance stated) before use.
- R-27 (user, 2026-10-02) kappa value: kappa_E6 = .25 (pooled one-step OOS unexplained share on E6-P DEV-v4, median
  over KPIs .254, 95 % CI [.229, .337], rounded; results/kappa_calib/). PRIMARY kappa = .25 in every world (E5's
  old .3 default is NOT used in this study; specs pass kappas explicitly); sweep {.125, .5} (the .5 end covers the
  within-cell reading .48); kappa 0 = sanity only. Paper caveat: kappa_E6 is an upper bound on pure measurement noise.
- R-28 Eq arm, lagged-KPI sources (secondary KPI->KPI family): Z = design_covariates(data) base set (no focal)
  minus the source's own lag_kpi column, plus every action at t (as R-3). Same rule in classic (_classic_common
  cond_set, audited) and citests. Listed in PROTOCOL_NOTES for the user's protocol review.
- R-29..R-36 (orchestrator, 2026-10-03 night, user asleep and delegated; from 3 independent protocol reviews in
  consult/protocol_review/review_protocol_{A,B,C}.md; for the user's morning review):
  * R-29 tau (score-only arms): conformal cutoff at level .05: tau = the ceil((M+1)(1-.05))-th smallest of the M
    tune-seed placebo scores of the (arm, cell) (max if that index exceeds M). Replaces "2nd largest" (R-2), which gave
    2/(M+1) (E4 .095). Same rule for PMRT scored with tau (secondary).
  * R-30 three-way validity per (arm, cell, rate): INVALID if the seed-cluster 95 % CI lower bound > .05; VALID if the
    upper bound <= .075 (Bradley liberal band); else INCONCLUSIVE. C1 needs INVALID; C2/C3 need VALID (symmetric
    burden). Cell-count thresholds (F_max) are calibrated by simulation under the shared-seed structure at true
    rate .05, not Binomial(N,.025). R-20 power comparisons use VALID cells only.
  * R-31 C2 split: C2a pmrt_eq; C2b every eq arm (incl. pcorr_eq etc.) whose native partner is INVALID. Pre-registered
    fallback wording if C2b fails: "design-based inference restores validity; adding design covariates does not
    suffice". The eq arms are also evaluated in E4 R3 (C3 reported for every eq arm; the PMRT-specific claim is only
    "by construction", not "only PMRT").
  * R-32 New design-blind baseline robust to serial dependence: pcorr_hac (native arm, R-3 set, Newey-West HAC SE,
    Andrews automatic bandwidth, time-ordered rows). In set D for C1. C1 needs |D_counted| >= 3 and reports a verdict
    per arm.
  * R-33 Secondary arm eq_min for the CI tests and pcorr_hac: R-3 set + sp:<focal> only (the minimal design
    information). Reported with |Z| sizes per world; addresses "equal information != equal benefit".
  * R-34 Seeds: E4 (R3/R4) S_E4 = 200 (validity target); other cells S from the R-12 power calc, floor 40, cap 100
    (cap revisited after the CI-test DEV pilot); E4 excluded from the recall power calc (Q1). Tune seeds = DEV tune
    seeds re-run at the freeze commit (Q2; hash equality vs DEV = reproducibility check).
  * R-35 Integrity: every package pinned to uv.lock on every platform (pin on, citests deps installed); each dataset's
    arms on ONE platform; records stamp code commit, protocol sha and spec sha; EVAL guard keyed to a sha constant in
    the spec + "FROZEN: yes" (Q3: campaign.py owner builds it, audited, in the freeze commit). Feasibility (R-13) is
    fixed from the DEV cost on the Kaggle reference host (T3); in EVAL the budget is a safety cap (2x). Missing,
    infeasible or under-seeded focal cells cap a verdict at PARTIAL, never silently drop out. GPU time reported
    separately (budget = CPU).
  * R-36 Wording: "validity does not depend on the outcome model" (not "needs no outcome model"); "RIC-style" scoped
    to the R2 design tested; section 0 lists everything seen before the freeze. Estimand check before freeze: no
    truth-null candidate has a nonzero lag-1 total effect in the SCM.
- R-37 P_placebo_conf (R-10 diagnostic) is never a conditioner of any other source, in any method or arm (incl. PC
  node sets): the diagnostic must not change the primary analysis. citests already do this; classic aligns
  (_classic_common.cond_set, pc nodes) on xm/hac.
  * R-35a torch: version pinned to uv.lock (2.10.0); CPU/CUDA build may match the session hardware, recorded per record.
- R-38 (night) pcorr_hac small-n: add a fixed-b variant (Kiefer-Vogelsang 2005 Bartlett fixed-b critical values,
  same bandwidth rule). The member of set D for C1 is chosen PRE-FREEZE on synthetic F4 only (AR(1) rho .5/.8 and
  iid, n 500/1000/4000): the variant with the smaller max |rate - .05|; the other is reported as secondary. No
  E-series data used for the choice. eq_min for lagged-KPI sources = native set. R-37 extends to notears, shap_dag,
  two_tower (fit without P_placebo_conf; conf candidates from a second fit). R-29 conformal tau implemented in the
  shared score.placebo_tau / ClassicBase.tune.
- R-39 (night; protocol2 Q6-Q10): Q6 power/recall comparisons gate on NOT-INVALID (R-20's original wording: "not
  significantly above .05"); VALID-only reported as sensitivity. Q7 OK (drop "PARTIAL if only n 500 fails").
  Q8 E4 R3/R4 at S_E4 only for the C3 readers (pmrt_eq, pmrt_r3, every eq arm); other arms at S. Q9 OK (NOT
  EVALUABLE < 90 % cells; C1 pooled-R2 INVALID leg; INVALID precedence). Q10 S_E4 = smallest of {200, 300, 400, 600}
  at which a nominal pmrt_eq passes C3 with probability >= .90 in the fmax simulation; fixed now as a rule (T-register).
- R-40 (night; audit-citests Q1/M1): mscr's "eq" arm adds Z_eq only as single conditioners in its max statistic, so it
  is not equal information: mscr_eq is EXCLUDED from C2b, reported with the label "single-conditioner max statistic;
  cannot condition on the joint design set". pdcor is reported as a dependence test (not a CI test; doc fix P1).
  audit-citests fixes 1-6 go to xm/citests2 (L4 RNG stream change allowed pre-freeze).
  * R-40a pdcor_native is NOT in set D (dependence test, not CI: would strawman C1); mscr admissible under T4; mscr_eq_min labelled too.
  * R-35b EVAL interpreter = Python 3.12 (uv-installed venv in cloud kernels), stamped per record.
- R-41 GPU budget (R-13 analogue): a GPU arm (cmi_knn torch) gets 2 h wall per (method, dataset) on a Kaggle T4;
  over that = "infeasible at this n (measured cost X)". GPU time is reported separately from CPU (Experiment C).
  * R-41a cmi_knn GPU arms may run in a separate Kaggle GPU shard from the dataset's CPU arms (same platform); every record stamps the dataset hash, and eval_analysis requires equal hashes for all arms of a dataset.
- R-42 (user-approved 2026-10-03) PMRT nonlinear statistic. DEV showed pmrt_eq recall .28-.46 in E2/E5 vs SHAP-DAG
  .67-.97 (raw p <= .05: .38-.54), likely because pmrt_core's statistic is linear. PMRT admits any statistic (validity
  comes from redrawing the design). Plan: (1) per-edge diagnosis E2/E5 (what PMRT misses, effect shapes); (2) 2-3
  nonlinear statistics, each a function of (redrawn random part, past-only covariates / residual) so the CRT stays
  exact; SELECTION RULE fixed now: among candidates that pass exactness (synthetic F4 + DEV truth-null and placebo
  rates not INVALID in any cell, R-30) and the R-13 budget at n 4000, pick the highest mean per-edge recall at raw
  p <= .05 over DEV E2/E5 R1/R2 n 1000 + synthetic nonlinear scenarios; ties -> cheaper. DEV seeds only (not
  3_000_160-189, reserved for pmrt-diag); disclosed in PROTOCOL_A s.0. Linear pmrt_core kept as secondary arm.
  (3) R4 (no design) reported as "not applicable" for PMRT, never recall 0; like-for-like table (PMRT scored with the
  same per-edge cutoff as score-only arms, and every p arm at raw p <= .05) promoted to a headline table.
- R-43 T1 power pairs use the primary PMRT arm (pmrt_nl_eq; winner run on DEV T1 cells). Fallback: if no R-42 candidate passes, linear pmrt_eq is primary. V0 shows all recalls with validity flags; V2/V3 gate on not-INVALID.
- R-44 (user, 2026-10-03) compute: before every new job check Colab (CPU / GPU) first for medium jobs; Kaggle for long
  unattended ones (cap 5); when Kaggle is full and Colab unavailable, use Lightning (up to all 3.77 credits; log
  credits per job in the status file). Never let a job wait for a Kaggle slot if another platform is free.
- R-45 (pmrt-diag REPORT): pmrt_core has no flaw; DEV E2 R2 excess = chance + selection (600 regen replicates
  per cell nominal, .049-.053). Wording: PMRT is asymptotically valid (martingale CLT; its predictable adjustment
  uses the focal action's own past dither); exact only when the adjustment is invariant to the focal dither. The
  optional exact `inv` variant is NOT added (applies to memoryless E2 only; costs 1-2.5 pts recall).
- R-46 (user, 2026-10-03) supersedes the reserve framing of R-26/R-44: Kaggle (5), Lightning (several CPU/GPU
  studios; measured ~0.06 credits/h per CPU studio, 5-8x faster than the laptop) and Colab are ALL primary. Every
  shard plan splits across them in parallel from the start. Credits logged per job; ask the orchestrator before
  cumulative Lightning use passes 3 credits. Stop studios when idle.
- R-47 (user, 2026-10-03) CI DEV plan C: full DEV grid for pcorr (7 arms) + rcot2 (3); mscr (3) at n <= 1000;
  pdcor and cmi_knn get T3 cost probes only (no DEV validity / power). Full grid was ~5900 core-h + 372-1482 GPU-h.
  OPEN for the freeze: the EVAL grid for pdcor / cmi_knn / mscr at large n (decide with the user).
- R-48 (user, 2026-10-03) pdCor DROPPED from Study A (all arms): not a valid CI test under conditioning (audit:
  E2 R2 eq null .22, P_placebo_conf 5/5), O(n^2) cost, no role in any verdict. Paper: one "considered, excluded" line.
- R-49 (user) CMI-kNN exact speed-up: reuse the conditioning-set neighbourhoods across local shuffles + batched GPU
  shuffles; same statistic and null as tigramite CMIknn (shuffle test); must reproduce the current backend exactly
  (or a stated float tolerance with identical decisions) before use. No approximate kNN.
- R-49 REVISED (user): CMI-kNN (tigramite CMIknn) is DROPPED from Study A, like pdCor (cost; RCoT and pcorr remain
  as CI tests). No cmi-fast worker.
- R-50 (user, 2026-10-03) ADD CDL (the authors' NaNA 2026 conference method: Wang et al. 2022 Causal Dynamics
  Learning, masked neural predictor, CMI from likelihood ratios, EMA, threshold). Source: branch refactor/codebase
  cdd_oran/models/cdl.py + configs/env_*_cdl.yaml + tests/golden/Environment*_CDL.json. Score-only method: primary
  declaration = conformal placebo tau (R-29) like other score-only arms; the conference's fixed threshold reported
  as secondary. Native arm only (no conditioning interface), R-37 applies.
- R-51 cdl gets the full DEV grid run by dev-runs before the freeze (T1 / T4 (iii) apply, no waiver). T3 host: DEV
  costs from any platform are accepted, converted to the Kaggle reference by a measured speed factor (same
  calibration units run on both hosts, factor recorded).
- R-52 R-42 selection applied: winner gbm (pmrt_core statistic="gbm", pmrt-core-v2; arm pmrt_nl_eq): F4 0/24 and
  DEV 0/108 INVALID cells, recall .749 (rff .666, poly .638, linear .420), cost n 4000 <= 57 CPU-s. Linear stays
  secondary (pmrt_eq). Merged 5e8b4f9.
- R-53 cdl merged: F2 bitwise port, F3 Env II PASS / Env I 4 of 5 seeds (miss at the paper's seed: weak edge, CPU vs
  paper's CUDA), F4 primary family VALID, power 1.00; cost n 24000 514 CPU-s (all n feasible). Its KPI->KPI family
  is "not calibrated by the action placebo" (no claim). Epoch budget ~130 epochs, cap 16k steps.
- R-54 mscr (user, 2026-10-03, CI DEV Q11): kept in Study A as a reported-INVALID arm. All mscr arms (eq too) reject
  truth-null edges in R2 at .93-1.00 (single conditioner, cannot condition on the joint design set; cf. R-40). EVAL
  runs mscr at n <= 1000 only (settles the large-n grid question); its verdicts feed C1 framing, never C2b.
- R-55 Exp C / T3 timing (overnight panel 3/3 unanimous, 2026-10-04): (1) T3 speed factors from a TRIMMED paired
  calibration block (5-6 anchor arms spanning cheap/medium/expensive, n 500/1000/4000, 3 repeats), run as its own job
  under 1 process per vCPU (cgroup quota, never oversubscribed) on Kaggle (reference) and Colab, before EVAL where
  slots allow; full 201-unit plan only if spare capacity. Uncalibrated host -> results provisional; a method within
  the factor's error of 7200 CPU-s counts as over budget. (2) Host type = platform + CPU model (/proc/cpuinfo logged
  per record). (3) Paper runtime table = dedicated uncontended timing run on Kaggle (1 proc/vCPU, few datasets per
  method x n, peak RSS); campaign CPU-s only supplementary with load mix disclosed. (4) Rows pool tune+measure seeds,
  kappas, lambdas; report median, mean, min, max; tune/measure flag kept; per world/regime breakdown in json.
  EVAL sessions run 1 process per vCPU and log CPU model, loadavg, concurrent processes per unit.
- R-56 freeze-prep Q1-Q3 (overnight panel of 3, 2026-10-04): Q1 (3/3) keep the C2b membership rule; the C2 text and
  main table NAME rcot2_eq as an eq arm INVALID in R2 (with its R1 vs R2 rates; cause: RCoT2 miscalibrated already in
  R1); pre-registered unfiltered table of every eq arm's verdict; C2b also reported with / without the R1-failing
  partners. Q2 (2/3) mscr_native stays in set D (R-54); C1 reported with and without mscr (sensitivity line); the
  C1-FAILURE definition (INVALID in R2, not in R1) applies to it unchanged. Q3 (2/3) T6 cap lowered to 60 as a stated
  compute ceiling (free tiers; S 100 ~94 h wall risks incomplete cells): S = clip(S_power, 40, 60), S_power from DEV
  data only, frozen; if S_power > 60 the study runs at 60 and reports achieved power.
- R-57 EVAL interpreter pinned to exactly Python 3.12.14 on every platform (Kaggle, Colab, Lightning, VPS): uv's
  "3.12" now resolves to 3.12.15. campaign.py EVAL_PYTHON = "3.12.14"; pin_check fails a session on any other patch.
  (User: VPS upgraded to match.)
- R-58 EVAL scope trim (user GO 2026-10-04 after a 4-reviewer scope panel: claims auditor, skeptical referee,
  compute economist, paper editor). Pre-registered grid choices (spec max_n / blocks + s.7 text; report says
  "not in grid", never missing / PARTIAL):
  (1) cdl: R1/R2 only, n <= 4000 (training cost; feeds no claim); not in E4 R3/R4, not in the kappa sweep.
  (2) pmrt_nl_eq: R1/R2 at all five n (C2a keeps its 50 cells and F_max).
  (3) E4 R3: C3 readers at S_E4 300. Fallback ONLY if the pmrt_nl_eq n 8000/24000 cost pilot shows E4 R3 n 24000 is
      expensive: drop n 24000 in E4 R3 for ALL C3 readers alike (never PMRT alone); then re-simulate C3 F_max + T9.
  (4) E4 R4: every arm at S (not 300); negative control, no claim.
  (5) mscr_eq_min dropped; mscr arms out of the kappa sweep. (6) Everything else unchanged.
  (7) Freeze does not wait for cdl DEV parts at n > 4000 (T1 uses n <= 4000; cdl capped); they finish as descriptive.
  Expected EVAL ~360-450 CPU-h (was ~1 243), ~17-22 h wall on Kaggle + VPS.
- R-59 Pre-freeze fixes from the independent campaign.py EVAL audit (results/audit/campaign_eval_audit.md, verdict
  FAIL; 2026-10-04). Nothing was frozen on feat/v2 yet, so these are fixes before the freeze, not amendments; the
  freeze is re-hashed after them and the auditor re-checks the diff. Code follows the pre-registered text:
  (F1) eval_analysis checks the spec stamp against integrity.spec_file_sha256 (LF file sha, the rule of
       file_sha256); plus a test whose records come from campaign.run_units, not hand-built.
  (F2) PROTOCOL_A s.7 stands: an EVAL unit over the 2x cap (RLIMIT_CPU 14 400) is recorded infeasible with its CPU-s
       and never re-run; its cell is infeasible. OOM / other signals stay errors (re-run).
  (F3) --skip-complete-from applies the merge's accept rule (mode, protocol sha, spec sha) before skipping a key.
  (F4) integrity stamps cost-table sha + part index / count; merge refuses mixed values within one EVAL run.
  (F5) session exit code non-zero if any part process failed (wait each PID); vps_run records it.
  (F6) EVAL refuses --budget different from the spec; the effective cap is stamped.
  (F7) EVAL refuses --no-isolate; isolation mode is stamped.
  (F8) vps --procs sets the scope CPUQuota (procs x 100 %), so NP follows it. (F9) merge summary adds python,
       lock_sha256 and spec_file_sha256 sets. (F10) no change (no effect on the frozen spec), noted in the audit.
- R-60 Overnight supplementary experiments (user: "run experiments that strengthen the paper"; 4-reviewer panel
  2026-10-04: referee, paper editor, integrity auditor, compute economist). All EXPLORATORY / supplementary: never
  change C1-C3 verdicts or the E6 headline; own specs + output dirs (never specs/eval or EVAL shards); methods and
  procedures exactly as frozen; fresh seed block 3_200_000-3_200_199 (claimed XMETHOD_EXTRAS in SEED_REGISTRY);
  declared in scratchpad/xmethod/EXTRAS_PROTOCOL.md and committed BEFORE launch and before anyone reads EVAL output;
  every declared run reported whatever the outcome; start only after EVAL has launched all 96 parts (Kaggle) / its
  last VPS job is pulled; EVAL requeues keep priority (keep 1 Kaggle slot free until EVAL merge checks pass).
  (X1) R-55 paper timing run (pre-registered, Exp C exp_c_timing.py --block paper): one uncontended Kaggle session,
       mscr_eq_min dropped (R-58(5)); log CPU model (EPYC 7B12 = provisional).
  (X2) Dither dose-response (C1 mechanism): E1, E2, E3 R2, n 1000 and 4000, kappa .25; DITHER_DELTA {.02,.05,.10,
       .20} at the frozen block count, plus blocks {5, 80} at delta .10; arms corr, pc_eq, notears, shap_dag,
       pcorr_native, pcorr_eq, pmrt_eq, pmrt_nl_eq; 20 tune + 40 measure seeds per design (tau re-derived per design
       by the frozen R-29 procedure on its own tune seeds); generator constants overridden by a NEW wrapper module,
       frozen generate.py untouched; delta .10 / frozen blocks = reproduction check of the EVAL design.
  (X3) Design-misspecification stress test (C2a assumption): the data stay as generated, the design TOLD to the
       design-based arms is wrong: dither width x0.5, x0.8, x1.25, x2; setpoint switch times shifted 2 % / 5 % of a
       block; E1 R2 and E2 R2 at n 1000 (+ E4 R3 lam 1 propensity distortion only if code is small); arms pmrt_nl_eq,
       pmrt_eq, pmrt_r3, pcorr_eq (contrast); 60 measure seeds. Reported as a degradation curve (where the known-
       design assumption breaks), not as evidence against C2a.
  Order X1 > X2 > X3; X3 only if capacity remains. Deferred to the user: Exp B effect injection, the pmrt_nl gap
  diagnostic (post-hoc optics), Exp B method-RNG replication, bridge figures (paper phase). Rejected: S extension
  40 -> 100 (S creep), extra kappa sweep (EVAL covers it), anything on seeds 3_100_000-299.
