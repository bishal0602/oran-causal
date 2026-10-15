# Study A protocol: design-based vs design-blind edge tests on E1-E5 (EVAL pre-registration)

FROZEN: yes (2026-10-04; v5, rulings R-1..R-58; worker `protocol`, branch xm/freeze on feat/v2 41d6c37; S = 40)

Freeze procedure. Before any EVAL unit is generated: (1) every T-item of section 13 is filled by its rule and the
value is written here; (2) the line above becomes "FROZEN: yes (<date>)"; (3) `specs/eval/full.json` gets
`protocol_sha256` = the LF sha256 of this file (the campaign EVAL guard and `eval_analysis.py` check it) and its
`pkgs_lock`; (4) the freeze note `scratchpad/xmethod/freeze/FREEZE_NOTE.md`, committed in the freeze commit (whose
message is one subject line), records the LF sha256 of this file, `scratchpad/xmethod/eval_analysis.py`,
`scratchpad/xmethod/eval_report.py`, `specs/eval/full.json`, `cdd_oran/xmethod/campaign.py`,
`cdd_oran/xmethod/dev_power.py` and `results/fmax_sim/dependence.json` and the DEV inputs (contents: T7); "recorded
in the freeze commit" below means in that note or a file it names; (5) the orchestrator
registers the EVAL block 3_100_000-3_100_299 in `docs/benchmark/SEED_REGISTRY.json`. After the freeze this file is
never edited: amendments go to `docs/xmethod/PROTOCOL_A_AMENDMENTS.md` + `specs/eval/amendments.json` (section 14).
Sources: CONTRACT.md, PROTOCOL_NOTES.md, PMRT_CORE.md, FIDELITY_CLASSIC / CITESTS, xm/dev-runs, reviews A-C.

## 0. What was seen before the freeze (R-36)

Only DEV seeds 3_000_000-3_000_199 were generated or read; no EVAL seed (3_100_000+). Seen, bearing on the verdicts:
(a) fidelity gates F1-F7 of every adapter; (b) the kappa calibration (R-27); (c) the dev-runs pilot: corr and
granger_native declare the R2 placebo 2-4x nominal, so C1's direction for them was known; (d) the full DEV run (tune
3_000_000-019, measure 3_000_100-159) incl. every arm's DEV validity (T5); (e) three protocol reviews of the draft and
pilot numbers; (f) the CI-test audit's smoke (n 1000, 5 DEV seeds: mscr invalid in R2 in both arms, pdcor eq E2 R2
null .22); (g) F2-GPU and the cmi_knn GPU cost on one DEV dataset (E5 R1, arm eq; Kaggle T4 wall 172 s at n 1000,
~2.4k-6.1k s at n 4000, ~9k s extrapolated at n 8000); (h) the DEV power read (R-42): pmrt_eq recall .28-.46 in E2 /
E5 vs SHAP-DAG .67-.97 (pmrt_eq at raw p <= .05: .38-.54), attributed (not shown) to pmrt_core's linear statistic;
PMRT admits any statistic, its validity coming from the redraw. Choices made after DEV numbers, reviews or the
audit: no Huber clip in PMRT (R-14, E2 R2 DEV truth-null rates); eq arms primary (R-17, before DEV validity);
NOTEARS kept despite F3 19/20; the R2 design (20 setpoint blocks, +/-.10 dither, Stage 0 on E6) supplies PMRT's randomisation, so favours it
by construction; after the reviews (R-29..R-39): conformal tau, three-way validity, simulated F_max (dependence from
DEV seeds 3_000_100-199), C2a / C2b with fallback wording, eq arms in E4 R3, the HAC baseline pcorr_hac (added after
reviewers predicted dependence-robust design-blind tests might hold level; its F4 was seen: over-rejection at n <=
1000 under strong persistence) and its fixed-b variant (set-D choice on synthetic F4 only, T8), eq_min, S_E4 by
simulation (T9), P_placebo_conf never a conditioner (R-37), the draft's "n 500 PARTIAL" clause dropped; after the
audit (R-40): mscr_eq out of C2b (not equal information), pdcor labelled a dependence test and pdcor_native out of D;
after the GPU cost (R-41): cmi_knn on the torch backend in every arm with a 2 h GPU wall budget; after the DEV power
read (R-42): a nonlinear PMRT statistic picked by a selection rule fixed in R-42 before any candidate was built
(T10; DEV seeds 3_000_100-159 and synthetic data only, never 3_000_160-189, reserved for the pmrt-diag check),
pmrt_nl_eq the primary PMRT arm and the linear pmrt_eq secondary, PMRT power in R4 reported as not applicable
(never recall 0), and the like-for-like table V0 (every p arm at raw p <= .05 and with the conformal tau, next to
the score-only arms) promoted to a headline; after the CI-test audit and the DEV cost estimate (R-47..R-50): the
CI DEV plan C (R-47: full DEV grid for pcorr and rcot2, mscr at n <= 1000 only, cost probes only for pdcor and
cmi_knn; the full grid was ~5900 core-h + 372-1482 GPU-h), pdcor dropped (R-48) and cmi_knn dropped (R-49 revised,
cost; section 4), and the authors' conference method CDL added as a
score-only arm (R-50); after the CI DEV run (plan C, `results/dev/ci_c`) and the R-42 selection (R-51..R-54):
mscr INVALID in every R2 cell in all three arms (truth-null raw .93-1.00), so mscr is kept as a reported-INVALID arm
at n 500 / 1000 only (R-54, T11); pcorr, pcorr_hac and pcorr_hac_fb native INVALID in R2 (truth-null raw .21-.46 at
n 1000) with their eq / eq_min arms at .04-.06; rcot2 INVALID in most R2 cells in every arm (rcot2_eq truth-null
.067-.104 at n 1000; 20 of 25 R2 cells) and in 7 of 25 R1 cells (eq and native; its authors' null is liberal, F4),
so the C1 / C2b directions of these arms were known (section 10); the R-42 winner gbm (R-52: F4 0 / 24 and DEV 0 /
108 INVALID cells, recall .749 vs linear .420); cdl's fidelity, cost and 12-cell integration run (R-53); the DEV
runs of cdl on the full grid (R-51) and of pmrt_nl_eq on the T1 cells (R-43), the T1 input; the DEV cost tables and
the EVAL projection (`scratchpad/xmethod/freeze/`); R-55 (timing: T3 speed factors from a calibration block keyed
by platform + CPU model, 1 process per vCPU in EVAL, the paper's runtime table from a dedicated timing run), set
after the DEV costs were seen (they came from oversubscribed sessions); R-56 (after the freeze-prep DEV preview of
the C1 / C2b legs: C2b membership rule kept, rcot2_eq named, an unfiltered eq-arm table, C2b with / without the
R1-failing partners, C1 with / without mscr, S cap 60); R-58 (EVAL scope trim, user GO after a 4-reviewer scope
panel, set after the DEV costs, the pmrt_nl_eq DEV run and the EVAL projection were seen, ~1 240 CPU-h before):
cdl at R1 / R2 n <= 4000 only (training cost; it feeds no claim), E4 R4 at S for every arm (a negative control with
no claim), mscr_eq_min dropped and mscr out of the kappa sweep, the C3 readers kept at S_E4 in E4 R3 with every n
(the R-58(3) fallback, dropping n 24000 there for all C3 readers alike, was not triggered: the pmrt_nl_eq cost
pilot measured ~100 CPU-s per dataset at E4 R3 n 24000); no verdict's cell set changed except C3 readers' E4 R4
seed count, which no claim reads. Estimand check: PASS, 18 cells.

## 1. Question and claim

Question. On one-step action -> KPI data logged under a known randomisation design, which edge tests hold their
nominal false-positive level, and how much power do the valid ones have?
Claim under test (CONTRACT section 8, R-36 wording): "tests that ignore the randomisation design are invalid on the
setpoint + dither data of the R2 design tested (C1); using the design restores validity (C2a: the design-based
test; C2b: design-covariate adjustment of model-based tests); PMRT does so by construction, its validity does not
depend on the outcome model, it admits any statistic, and it stays valid under logged confounded policies (C3)".
"RIC-style" means only this R2 design. The outcome-model and any-statistic properties are argued from the
construction (PMRT_CORE.md), not tested; fallback wordings in section 10; power descriptive, no superiority claim.

## 2. Worlds and regimes (controlled diagnostic environments)

Each world isolates one mechanism; none models a deployed RAN. SCMs read-only (`cdd_oran/envs/v2/e*.py`), generator
`cdd_oran/xmethod/worlds/generate.py` (`xm-worlds/3`, RNG streams keyed by (seed, world, regime, purpose), not by n:
smaller n are prefixes of the same draws). Candidates true / null (P_placebo -> KPI among the nulls): E1 4/16, E2
16/38, E3 4/21, E4 1/1, E5 6/14; E4 has no real-action null (read-outs: P_placebo, and P_placebo_conf in R3 / R4).

| world | mechanism isolated | regimes |
|---|---|---|
| E1 | clean identifiable linear graph (K2 = P2 + .5 K0, K3 = P3 + .5 K1) | R1, R2 |
| E2 | gated edge, active only in a rare regime | R1, R2 |
| E3 | delayed harm: effects through lagged KPI cascades (depth 3) | R1, R2 |
| E4 | latent confounding; associational sign wrong at large lambda; lambda in {0, .5, 1, 1.5} | R1-R4 |
| E5 | E2-E4 mechanisms composed (subdom .2, chain .2, theta -2.5) | R1, R2 |

| regime | design (`api.Design.kind`) | role |
|---|---|---|
| R1 | `iid`: every action redrawn i.i.d. per row | reference: valid tests hold level |
| R2 | `dither`: 20 setpoints per corpus (+/-25 % envelope), held n/20 rows, + i.i.d. U(+/-.10 range) dither | C1 / C2 test bed |
| R3 | `logged`: E4 behaviour policy on an OBSERVED confounder, per-row probabilities logged (R-7) | C3 |
| R4 | `none`: E4 latent confounder, nothing logged | negative control |

E4 lambda: R1 / R2 at 1 only (inert, R-1); R3 / R4 at all four. Placebos (R-10): `P_placebo` (real actions' design in
R1 / R2, marginal design in R3 / R4) is in the primary BY family and the only tuning column; `P_placebo_conf`
(E4 R3 / R4, confounded policy) is outside every BY family, never tuned on, never a conditioner (R-37).

## 3. Observation noise (R-24, R-27)

y_k <- y_k + N(0, (kappa sigma_k)^2) in every world; PRIMARY kappa .25 (kappa_E6, CI [.229, .337], an upper bound on
measurement noise); sweep {.125, .5} in R2 at n 1000; kappa 0 is a DEV sanity check only.

## 4. Methods

| adapter (version) | family | authors' null / native rule | arms | fidelity (F1-F7) |
|---|---|---|---|---|
| pmrt_core (`pmrt-core-v1`: linear statistic; `pmrt-core-v2`: statistic gbm, module `pmrt-nl-v1`, T10) | proposed, design-based CRT | redraw of the known random part, B <= 9999 Besag-Clifford, no clip; any statistic (R-42) | nl_eq (primary: statistic gbm, T10), eq (linear, secondary), r3 | engine bit-exact vs E6 pmrt `plain`; F4 null .043-.058 (linear); gbm: F4 truth-null .026-.047, 0 / 24 INVALID, DEV 0 / 108 INVALID (R-52) |
| mscr (`xm-mscr-v2-bc`) | CI test; single-conditioner max statistic, the eq arm cannot condition on the joint design set (R-40); reported-INVALID arm, EVAL n 500 / 1000 only (R-54) | stratified random-partition bank, B 9999 BC | eq (not in C2b), native (eq_min dropped, R-58(5)) | F3 exact on stored E2; F4 null .054 / .049 (n 500 / 1000) |
| pcorr (`xm-pcorr-v1`) | CI test | partial-correlation t-test, df n-2-\|Z\| | eq, eq_min, native | F2 / F4 PASS |
| pcorr_hac (`PcorrHac` 1.0, R-32) | design-blind, serial-dependence robust | OLS t, Newey-West HAC SE, Andrews AR(1) bandwidth, time order, t(n-k) | native, eq_min | F2 == statsmodels; F4 iid .049-.054, AR(1) .8: .104 / .084 / .060 (n 500 / 1000 / 4000) |
| pcorr_hac_fb (`inference: fixed_b`, 1.1, R-38) | as pcorr_hac; set-D member (T8) | same statistic and bandwidth, Kiefer-Vogelsang (2005) Bartlett fixed-b p | native, eq_min | F3 vs KV Table I PASS; F4 iid .047-.053, AR(1) .8: .091 / .080 / .056 |
| rcot2 (`xm-rcot-v2-lpd4`) | CI test | Strobl et al. LPB4, num_f 100 | eq, eq_min, native | F4 null .084 / .065: authors' null LIBERAL, kept for faithfulness |
| pc (causal-learn 0.1.4.8) | classic discovery | Fisher-z .05; score = pMax | eq (tier-0 design nodes), native | PASS-WITH-NOTES |
| notears (xunzheng/notears@4a9ab19) | classic discovery | w_threshold .3 | native only | F3 19/20; included, disclosed |
| shap_dag (XGBoost 3.4.1 + shap) | O-RAN (Sharma et al.) | "most influential" | native only | PASS-WITH-NOTES |
| two_tower (E2 port, row-share score) | ADAPTATION inspired by arXiv:2601.13213 | relative gate .10 | native only | not the published supervised model |
| cdl (port of the authors' NaNA 2026 conference method, refactor/codebase@65b56f3 `cdd_oran/models/cdl.py`; version "... port, offline 130-epoch rule (cap 16k steps)", R-50, R-53) | the authors' conference method: Causal Dynamics Learning (Wang et al. 2022), masked neural predictor, per-edge CMI from likelihood ratios, EMA; CPU, training steps min(16000, ceil(130 n / 128)); EVAL grid R1 / R2 n <= 4000 only (R-58(1)) | conference fixed threshold CMI >= .16 (secondary, V0); primary: conformal tau | native only | F2 bitwise port PASS; F3 Env II 16 / 16 PASS, Env I 4 of 5 seeds (paper's seed 9 / 10, CPU vs the paper's CUDA); F4 primary family VALID, power 1.00; KPI -> KPI family not calibrated by the action placebo, no claim (R-53, FIDELITY_CDL.md) |
| corr (scipy pearsonr) | simple | Pearson t-test | native only | PASS |
| granger (own OLS F == statsmodels) | simple, E3 only | F-test, one lag | eq (VARX), native | PASS-WITH-NOTES |

pdcor (partial distance correlation, `xm-pdcor-v2-projperm`) was considered and excluded from every arm (R-48): pdCor
= 0 is not conditional independence, so it is not a valid CI test under conditioning (CI-test audit smoke: E2 R2 eq
truth-null rate .22, P_placebo_conf declared 5 / 5), and its cost is O(n^2). cmi_knn (tigramite CMIknn,
`xm-cmiknn-tigramite-5.2.10.1-bc`) was considered and excluded from every arm (R-49 revised) because of its cost
(Kaggle T4, torch neighbour search: ~2.4k-6.1k s per dataset at n 4000, ~9k s extrapolated at n 8000); pcorr and
rcot2 remain as CI tests.
Hyperparameters = authors' defaults; nothing tuned on DEV except R-14. CI-test adapters, pcorr_hac and cdl enter only
under T4. Conditioning-set sizes |Z| for focal P0 (native / eq_min / eq): R1 E1 8/8/18, E2 14/14/32, E3 9/9/19, E4
2/2/6, E5 8/8/18; R2 E1 8/9/23, E2 14/15/41, E3 9/10/24, E4 2/3/8, E5 8/9/23; E4 R3 4/4/10 (DEV seed 3_000_000, n 500).

## 5. Arms (R-17, R-18, R-19, R-25, R-28, R-33)

- PRIMARY block: `pmrt_nl_eq`, the PMRT arm (spec `focal`; pmrt_core with covariates eq and the statistic chosen by
  the R-42 rule, T10; it is "the PMRT arm" of C2a, C3, V3 and T1; if no candidate passes, pmrt_eq, T10 fallback),
  every `*_eq` arm (WHOLE conditioning set from
  `covariates.design_covariates(data, focal=<action>)`: R-3 set + setpoints of all designed actions + all actions at
  t-1, t-2 + context in R3; PC tier-0 nodes; Granger exogenous regressors; lagged-KPI sources per R-28) and the
  methods without a conditioning interface (corr, notears, shap_dag, two_tower, cdl; native only, disclosed, kept
  apart in every table by their rule).
- SECONDARY block: `pmrt_eq` (pmrt_core's linear statistic, `pmrt-core-v1`; secondary since R-42, labelled);
  `*_native` (R-3 set, design-blind, R-18); `*_eq_min` (R-3 set + `sp:<focal>` only, the minimal design
  information, R-33; = native in R1 and for lagged-KPI sources; pcorr, pcorr_hac(_fb), rcot2; mscr_eq_min dropped,
  R-58(5)); pmrt_r3 (R-19).

## 6. Declaration rules (R-2, R-6, R-9, R-22, R-23, R-29)

- p arms (`declare: by`): BY at q .05 over the dataset's PRIMARY family (action -> KPI incl. P_placebo), by the
  adapter; secondary family (lagged KPI -> KPI) its own BY. pmrt_core counts not-applicable candidates (R4) in its m
  (`notes['by_family_m']`, conservative); the V11 BY re-check uses it.
- tau arms: declare iff score > tau, tau = the ceil((M+1)(1-.05))-th smallest of the (arm, cell)'s M tune-seed
  P_placebo scores (the largest if the index exceeds M): out-of-sample placebo level <= .05 under exchangeability
  (R-29, shared `score.placebo_tau`; E4: M 20, tau = max, level 1/21); the KPI -> KPI family reuses it. p arms (incl.
  the PMRT arms) are also scored with tau (V0, V3) and at raw p <= .05 per edge, no multiplicity (V0; R-42). cdl is
  also scored at the conference's fixed threshold (declare iff CMI >= .16, spec `fixed_threshold`; R-50): a
  secondary scoring (V0 rule `fixed`), not tuned, so all three of its rates enter its validity. Not testable / not
  applicable = NOT declared, counted (R-23). Signs R-4.

## 7. Grid, seeds, budget, platforms

- n grid {500, 1000, 4000, 8000, 24000}; feasibility per (arm, n) fixed from DEV cost (T3); a T3-infeasible cell is
  shown as "infeasible (DEV cost X)" and is not part of the arm's pre-registered cell set.
- Pre-registered grid choices (R-54, R-58; spec blocks + `not_in_grid` rules, generated by
  `scratchpad/xmethod/freeze/r58_spec.py`): every arm has every world / regime / lambda / n / kappa cell of the
  measurement blocks, except (1) cdl: R1 / R2 at n <= 4000 only (not in E4 R3 / R4, not in the kappa sweep; R-58(1));
  (2) mscr_eq and mscr_native: n 500 / 1000 only (R-54), not in the kappa sweep (R-58(5)); mscr_eq_min is dropped
  (R-58(5)). pmrt_nl_eq has R1 / R2 at all five n (R-58(2)). A cell outside an arm's grid is "not in grid" (`nig`):
  it is not a planned cell, so it is never missing, never caps a verdict at PARTIAL and never makes a component NOT
  EVALUABLE (section 10); V11 checks that every gap matches a rule (`grid_gaps_ruled`). R-58(3) fallback (drop E4 R3
  n 24000 for every C3 reader alike, then re-simulate the C3 F_max and T9): not triggered (T3 cost pilot).
- Measurement seeds from the EVAL block: S seeds 3_100_000.. (T1) for every cell, except that the C3 readers
  (pmrt_nl_eq, pmrt_eq, pmrt_r3, every eq arm; R-39) use S_E4 = 300 seeds (T9) in E4 R3; E4 R4 is at S for every arm
  (R-58(4): negative control, no claim); sweep cells the first ceil(S/2). Tune seeds = DEV tune seeds 3_000_000-019
  re-run at the freeze commit (R-34; sha and declarations vs DEV reported, V11). `specs/eval/full.json`: 24 arms;
  at the frozen S 40 (T1): 150 960 units / 11 000 datasets (40 520 tune); S 50 / 60 would have been 168 970 / 11 750
  and 186 980 / 12 500 (projection:
  `scratchpad/xmethod/freeze/eval_projection.json`).
- Budget (R-13, R-35): 7200 CPU-s per (method, dataset) decides feasibility (T3, Kaggle reference host). In EVAL
  RLIMIT_CPU = 14 400 (2x safety cap): a unit over it is recorded infeasible with its CPU-s, never re-run, and its
  cell is infeasible; an OOM kill is an error (re-run). RNG tags 7800 data, 7801 pmrt, 7802 citests, 7803 classic.
- GPU budget (R-41, the R-13 analogue): an arm run on a GPU backend (spec `budget_wall_s`; none since cmi_knn was
  dropped; cdl runs on CPU, R-53: its GPU run neither reproduces the CPU run nor is faster) has a budget of 2 h wall
  (7200 s) per (method, dataset) on a Kaggle T4 (`budget_cpu_s` none); over it = "infeasible at this n (measured cost
  X)", X in GPU wall-s (T3). In EVAL the same 2x safety cap applies (14 400 wall-s), same rule as the CPU cap. GPU
  wall time is reported apart from CPU-s (V10, Experiment C), never summed with it. Arms on CPU backends report any
  `gpu_s` (V10), not budgeted.
- Platforms (R-26, R-35, R-41a, R-46, R-57): packages pinned to uv.lock everywhere (`pkgs_lock`); the EVAL
  interpreter is exactly Python 3.12.14 on every platform (R-35b as amended by R-57: campaign `EVAL_PYTHON`; pin_check
  fails a session on any other patch release, and uv's bare "3.12" is not used); shards split across Kaggle,
  Lightning, Colab and the VPS in parallel (R-46, R-57); a dataset's arms on one platform, in one shard, except that GPU
  arms may run in a separate GPU shard on the same platform (R-41a). EVAL sessions run 1 process per vCPU, never
  oversubscribed (R-55; Kaggle 4, Colab 2, VPS 7), and every record logs its host type = platform + CPU model
  (/proc/cpuinfo; the VPS is host type 'vps' + its CPU model, R-57), the load average and the number of concurrent
  processes. Every record stamps the dataset hash
  (`dataset_sha256`, sha256 over every array and name of the generated dataset), and all arms of a dataset must carry
  the same hash (eval_analysis V11: a mismatch, or an ok record without a hash, makes the run PROVISIONAL; a unit not
  run has no dataset and is only counted). EVAL shards in their own directory (tune keys equal DEV keys).

## 8. Metrics (per (arm, cell) on measurement seeds; cell = (world, regime, lambda, kappa, n))

- Rates: truth-null (primary truth-null candidates), placebo (P_placebo), confounded placebo (P_placebo_conf); for p
  arms raw (p <= .05) and declared (BY), for tau arms declared. Not-testable candidates leave every denominator;
  not-applicable ones (no p) leave the raw denominators and count as not declared (V7).
- 95 % CI: seed-cluster percentile bootstrap (2000 reps, rng 20261002, seeds sorted), Wilson with design effect
  alongside. Pooled rates cluster on (world, seed) (separate RNG streams per world).
- Validity per rate (R-30): INVALID if the CI lower bound > .05 (takes precedence); VALID if the upper bound <= .075
  (Bradley); else INCONCLUSIVE. Cell validity over its rates (p arms: all raw and declared; tau arms: truth-null and
  confounded-placebo declarations; P_placebo is their tuning column, reported only): INVALID if any rate is, VALID
  if all are, else INCONCLUSIVE ("no read" if none, e.g. tau arms in E4 R1 / R2); < 10 seeds: "few seeds".
- Power (R-39: cells not INVALID; VALID-only is the sensitivity read, since at S 40 an exactly nominal cell is VALID
  w.p. only .25-.45): recall per seed (mean, sd), FDP (mean = FDR, CI), pooled sign accuracy. Also counted: not
  testable / not applicable; cost (CPU-s; wall-s for the GPU arms, R-41; peak RSS, GPU-s, infeasible and error units).
- Runtime (Experiment C, R-55): the paper's runtime table comes from a dedicated uncontended timing run on Kaggle (1
  process per vCPU, a few datasets per (method, n), CPU-s and peak RSS), not from the campaign. Its rows pool tune
  and measure seeds, kappas and lambdas and report median, mean, min and max, with the tune / measure flag kept and
  a per (world, regime) breakdown in the json. The campaign CPU-s (V10) are supplementary, reported with their load
  mix (host types, load average, concurrent processes).
- PMRT in R4 (R-42): the real actions have design kind `none`, so a PMRT arm tests no true edge there. Its power
  (recall, sign) is "not applicable" (NA), never recall 0, and is left out of every recall table, mean and paired
  difference (V0, V2, V3, V6). Its placebo rates are still read (P_placebo has an i.i.d. design in R4).

## 9. Analyses

HEADLINE (R-42, descriptive, no verdict): V0 like-for-like table, printed first. Every p arm is scored twice: at raw
p <= .05 per edge (no multiplicity) and with the conformal placebo tau of section 6, like the score-only arms. It
sits next to the score-only (tau) arms. Per cell and scoring: mean per-seed recall, truth-null rate and placebo
rate (declaration rates, CI, three-way validity; a tau scoring's placebo is its tuning column, reported only).
Recall is shown for every counted cell next to its rates. PMRT in R4 is NA (section 8). cdl also has a row at the
conference's fixed threshold (rule `fixed`, secondary; section 6).
PRIMARY (kappa .25): A1 validity (V1), verdicts C1, C2a, C2b, C3 (V4); A2 recall among cells not INVALID (V2), paired
recall differences PMRT arm - arm under the SAME rule (V3: BY arms vs the PMRT arm's BY, tau arms vs the PMRT arm
scored with tau; per-cell paired-t CI, descriptive; E4 rows descriptive). SECONDARY (no verdict): information levels
in R2 with |Z| (V5), KPI -> KPI (V8), E4 by lambda (V6), kappa sweep (V9), not testable (V7), cost (V10), the C2a /
C3 rules for the other PMRT arms (pmrt_eq, pmrt_r3; V4, no effect).

## 10. Support rules (computed by `eval_analysis.py`, kappa .25)

An arm's pre-registered cell set = its grid cells (section 7: cells "not in grid" under R-54 / R-58 are not
part of it) minus T3 cells. A cell is counted if its status is ok or
under-seeded (some planned seeds missing) with >= 10 seeds. Caps (R-35): a component is NOT EVALUABLE if < 90 % of
its planned cells are counted, and SUPPORTED is capped at PARTIAL if any planned cell is not counted (missing,
EVAL-infeasible, untuned, few seeds) or is under-seeded; such cells never drop out silently (listed in V4 / V1).
F_max (R-30) = smallest f with P(#INVALID <= f) >= .95 for an arm whose tests have level exactly .05, by
`eval_analysis.fmax_simulate` (2000 runs, seed 20261003): S seeds shared by all cells; null statistics Brownian in n
(DEV check on E1: corr .68 / .51 / .34 vs sqrt(n1/n2) .71 / .50 / .35); E4 lambdas (.69-.95) and same-dataset
candidates (|r| <= .12) correlated as estimated in `results/fmax_sim/dependence.json` (`fmax_calib.py`, DEV seeds
3_000_100-199, pcorr stand-in; sha in the spec); worlds, regimes independent; the analysis' own bootstrap per cell.
Fixed values (`results/fmax_sim/fmax_table.json`; other cell sets by the same function):

| cell set (rates null_raw, plac_raw) | S 40 | S 50 | S 60 | S 70 | S 80 | S 90 | S 100 |
|---|---|---|---|---|---|---|---|
| R1 + R2, 50 cells (C2a, C2b): F_max (P pass at exact level) | 3 (.93) | 3 (.92) | 3 (.92) | 3 (.91) | 3 (.92) | 4 (.94) | 4 (.94) |
| E3 R1 + R2, 10 cells (granger_eq): F_max (P pass) | 1 (.91) | 2 (.95) | 1 (.93) | 1 (.92) | 1 (.92) | 2 (.95) | 2 (.96) |
| R1, 25 cells (C1 leg iii); E3 R1, 5 cells (granger_native) | 2; 1 | 2; 1 | 2; 1 | 2; 1 | 2; 1 | 2; 1 | 2; 1 |
| R2, 25 cells: P(pooled rate INVALID) at exact level (C1 leg ii false alarm) | .049 | .048 | .036 | .042 | .048 | .042 | .052 |

C3 set (E4 R3, 20 cells, P_placebo and P_placebo_conf raw): F_max 2 at S_E4 200 / 300, P(pass) .82 / .93 (T9;
one candidate per dataset, shared seeds). Chance INVALID per cell at an exact level .02-.026; non-monotone entries
are Monte Carlo steps of an integer quantile (P(#INVALID <= F_max) >= .95 holds for each).

- C1 (design-blind tests invalid on R2). D = the native p arms of the frozen spec with `set_D: true`: corr,
  granger_native, the HAC variant chosen under T8 and the native CI tests admitted by T4 (a dependence test that is
  not CI would make C1 a strawman; R-40, Q11). mscr_native is in D over its own cells (n 500 / 1000; R-54), its
  label printed with its verdict, the FAILURE definition unchanged; C1 is also reported without it (spec
  `c1_sensitivity_drop`; R-56), a sensitivity line with no effect. The HAC variant not chosen gets the same legs
  reported descriptively, no effect. Arm a is a FAILURE iff (i) INVALID in >= half of its planned R2 cells
  (a cell not counted is not INVALID), (ii) its pooled R2 truth-null or placebo raw rate is INVALID and (iii) #INVALID
  R1 cells <= F_max(R1 set); if (iii) fails, "INVALID IN R1" (invalid regardless of the design, e.g. a liberal
  authors' null; not a failure); else NOT A FAILURE. C1 needs >= 3 assessable arms (>= 90 % of R1 and R2 cells
  counted), else NOT EVALUABLE; SUPPORTED if >= half fail, NOT SUPPORTED if none, else PARTIAL; capped; a verdict per
  arm. If the HAC member is NOT A FAILURE, C1 reads "design-blind tests with i.i.d. / exchangeable nulls are invalid
  on the R2 design tested; the serial-dependence-robust design-blind test is not".
- C2a (the design-based test is valid). The PMRT arm (pmrt_nl_eq, spec `focal`) over its R1 + R2 cells: (a)
  #INVALID <= F_max; (b) the pooled truth-null and placebo raw rates are VALID. SUPPORTED iff (a) and (b); capped.
  The same rule for the other PMRT arms (pmrt_eq, linear statistic; pmrt_r3) is reported, no effect (R-42).
- C2b (design covariates restore validity). Every eq p arm except mscr_eq whose native partner is a C1 FAILURE, same
  rule on its R1 + R2 cells, a verdict per arm; SUPPORTED if >= half pass, NOT SUPPORTED if none, else PARTIAL;
  capped. Wording: C2a and C2b SUPPORTED -> "using the design restores validity (design-based inference and
  design-covariate adjustment)"; C2a SUPPORTED, C2b not -> "design-based inference restores validity; adding design
  covariates does not suffice" (+ the arms for which it does). R-40: mscr_eq (spec `c2b: false`) is not equal
  information, so it is excluded; its same-rule R1 + R2 result is reported, no effect, labelled "single-conditioner
  max statistic; cannot condition on the joint design set" (also on mscr_eq_min). R-56 (membership rule kept):
  (1) an eq p arm outside C2b whose native partner is INVALID IN R1 while meeting C1 legs (i) and (ii), and which
  itself meets legs (i) and (ii), is NAMED in the C2 text (wording suffix "<arm> INVALID in R2 as already in R1 (not
  restored by design covariates)") with this pre-registered sentence: "<arm> is INVALID in R2 (pooled truth-null raw
  rate r2 [CI]) as already in R1 (r1 [CI]): the test is miscalibrated without the design (its native partner <arm>
  is INVALID IN R1), so it is outside C2b's membership rule, and adding design covariates does not make it valid".
  DEV (section 0) expects rcot2_eq here (R2 truth-null .067-.104 at n 1000; rcot2_native INVALID in 7 of 25 R1
  cells, F_max 2: RCoT2 is miscalibrated already in R1). (2) V4 prints an unfiltered table of EVERY eq arm: C2b
  membership, its native partner's C1 verdict, the same-rule R1 + R2 verdict (tau arms on truth-null declarations),
  R2 INVALID cells, pooled R1 and R2 truth-null rates. (3) C2b is also reported with the eq arms whose native partner
  is INVALID IN R1 but meets legs (i) and (ii) added to its members (sensitivity, no effect).
- C3 (valid under the logged confounded policy). The PMRT arm over E4 R3 (4 lambda x 5 n, S_E4 300): (a) #INVALID
  <= F_max on P_placebo and P_placebo_conf raw; (b) both pooled raw rates VALID. Reported with the same rule for every
  eq arm (tau arms: P_placebo_conf declarations), no claim effect; if any also passes, the wording adds "so do
  <arms>; the property is not specific to PMRT", and PMRT's property is stated as "by construction" only. The other
  PMRT arms get the same rule reported, no effect, never among those <arms>.
- Claim SUPPORTED iff C1, C2a, C2b and C3 are SUPPORTED; otherwise V4 prints the wording the components license.
  A re-read without n 500 is descriptive only. V0 (headline), the R4 negative control (V6) and V3 sit next to the
  verdicts. The estimand check (`estimand_check.py`, section 0) is re-run at the freeze commit; a failing candidate
  is reclassified.

## 11. Execution and integrity

- Driver `cdd_oran/xmethod/campaign.py` (xm/dev-runs, built and audited for the freeze, R-35): EVAL seeds only if the
  spec's `protocol_sha256` equals this file's LF sha256 and it says "FROZEN: yes"; records stamp code commit,
  `run_mode.protocol_sha256`, `integrity.spec_file_sha256` (the spec file's LF sha256; R-59), pkgs, host, python
  (3.12.14, R-57). Shards on Kaggle,
  Lightning, Colab and the VPS (R-46, R-57);
`campaign merge`.
- `eval_analysis.py` uses only records whose key is an expected unit with the expected role. FINAL (exit 0) needs:
  protocol frozen with the spec's sha; every record from the freeze commit or an amendment listing its key; dirty
  exactly false; stamps equal the spec's shas; no missing, unexpected, wrong-role or duplicate key; full candidate set
  per record; every error unit listed as persistent; dataset sha stamped on every ok record and equal across all arms
  and shards of a dataset (R-41a); one platform per dataset; one pkgs set equal to `pkgs_lock`; F_max inputs' sha as
  in the spec. Else PROVISIONAL (exit 2), failed checks listed. Error units are re-run until none remain; one failing
  every re-run is listed in amendments.json (key, reason) and its cell is under-seeded or few-seeds, which caps the
  verdicts.

## 12. Outputs (`eval_tables.json` + `EVAL_TABLES.md`)

V0 like-for-like headline, printed first (R-42; section 9); V1 validity per (arm, cell) with status; V2 recall / FDP /
sign (not INVALID; NA = not applicable); V3 same-rule differences; V4 verdicts, F_max, wording (+ the other PMRT arms,
the R-56 eq-arm table, named arms and sensitivity lines; no effect); V5 information levels; V6 E4; V7 not testable; V8
KPI -> KPI; V9 kappa sweep (cdl and mscr not in the sweep); V10 campaign cost (supplementary to the timing-run table,
R-55; GPU arms in wall-s, a separate table; R-41); V11 integrity. In every table a cell outside an arm's grid reads
`nig` (not in grid, R-54 / R-58), never missing; the rules and their cell counts are listed with V1 and in the report
(`eval_report.py`, frozen with eval_analysis, T7). Figures (not drawn): validity map, recall vs n, information levels,
E4 by lambda, cost vs n.

## 13. Register of values fixed at the freeze (rules fixed now)

- T1 seed count S: `eval_analysis.py t1 --spec <DEV specs> --merged <files> --cap <T6>` on the full DEV runs (dev-runs
  `results/dev/full/merged.jsonl.gz` + the citests DEV merged file; paths and LF sha256 in the freeze commit). Kappa
  .25 cells at n in {500, 1000, 4000}, E4 excluded (one true edge); every pair PMRT arm vs a primary-block arm under
  the same declaration rule, neither INVALID on DEV, gives the paired-t seed count for a recall gap .15 (alpha .05,
  power .8); S = min(cap, max(40, max count rounded up to a multiple of 10)), cap 60 (T6), i.e. S = clip(S_power, 40,
  60) from DEV data only, frozen (R-56); minimum detectable gap and achieved power (gap .15) at S per pair reported,
  and if S_power > 60 the study runs at 60 and reports the achieved power; the max over noisy DEV sd's biases S upward
  (disclosed). The PMRT arm of the pairs is the primary one (`t1 --focal pmrt_nl_eq`): dev-runs runs the T10 winner on
  the T1 cells (DEV measure seeds) and that DEV run is in the T1 input; under the T10 fallback, pmrt_eq. CI DEV plan C
  (R-47): mscr's DEV run covers n <= 1000 only, so its pairs use n 500 / 1000. cdl (R-51, R-58(7)): its DEV run at
  n <= 4000 (the T1 cells; cdl's EVAL grid) is the T1 input (T4 (iii) applies, no waiver); the freeze does not wait for
  its DEV parts at n > 4000, which finish as descriptive. Interim (focal
  pmrt_eq, 10 arms + 13 CI arms, 108 pairs; results/dev/ci_c/t1.json): S_power 14, S 40, minimum detectable gap .082.
  VALUE: 40. Final T1 (focal pmrt_nl_eq; every DEV record: 10 arms + 13 CI arms + pmrt_nl_eq + cdl, 2086 cells,
  106 840 records, 0 screen issues): 115 pairs, S_power 14 (pc_eq E5 R2 n 4000; cdl's largest pair 11), so S = 40
  (floor), cap 60 not binding; minimum power at S over the pairs .999, maximum minimum detectable gap .083
  (`results/dev/final/` REPORT.md, notes.txt, t1.json, t1_pairs.json; dev-runs, feat/v2 41d6c37).
- T2 tau: from the EVAL tune records (section 6); not a DEV quantity. VALUE: computed in EVAL (no freeze value).
- T3 infeasible (arm, n): max over worlds, regimes, lambdas of DEV `cpu_s` per unit, in Kaggle reference seconds, >
  7200 makes n and every larger n infeasible (spec `max_n`, `t3_cost_cpu_s`). DEV costs from any platform count
  (R-51): a unit run on another host type is converted as cost x f. Host type = platform + CPU model (R-55; logged per
  record). f = sum of Kaggle CPU-s / sum of host CPU-s over a trimmed paired calibration block (R-55): 6 anchor arms
  spanning cheap / medium / expensive (pcorr_eq, rcot2_eq, shap_dag, pmrt_nl_eq, cdl, mscr_eq_min; mscr at n 500 /
  1000 only; the anchor set is fixed and need not match the EVAL arm set: an anchor is a CPU-workload benchmark for
  the host speed factor, pooled over anchors, so mscr_eq_min stays an anchor although R-58(5) dropped it from EVAL),
  each at n 500 / 1000 / 4000, E2 R2 kappa .25, DEV seeds 3_000_000-002 (the 3 repeats), built and run by
  xm-citests (`scratchpad/xmethod/exp_c_timing.py calib`, read by `exp_c_runtime.py calib`) as its own job under 1
  process per vCPU on Kaggle (the reference), on each Colab CPU model and on the VPS (R-57), before EVAL where slots
  allow (the full 201-unit plan only with spare capacity). Its error e = max over anchors of |f_anchor / f - 1|. A DEV
  cost from an uncalibrated host type makes the T3 read provisional; an arm whose converted max is within e of 7200
  (max x f x (1 + e) > 7200) counts as over budget. The block, both costs, f and e are recorded in the freeze commit.
  Without a DEV cost at that n: a cost pilot (DEV seeds 3_000_000-002 in the arm's 3 costliest (world, regime) at that
  n), on Kaggle or on a calibrated host with its f; its max decides. Pilot: none infeasible (max 144 s). GPU arms
  (R-41): the same rule on DEV wall-s per unit (child wall clock when isolated), referenced to a Kaggle T4 > 7200
  (spec `t3_cost_wall_s`), f measured against a Kaggle T4. No GPU arm is planned (R-49 revised, R-53). VALUE: no (arm,
  n) infeasible, no spec `max_n`. Max DEV CPU-s per unit before f (`eval_projection.json` `t3_max_cpu_s_by_arm_n`):
  pmrt_nl_eq 1148 at n 24000 (T3 cost pilot, E2 R1, Kaggle, 1 process per vCPU; n 8000: 500; E4 R3 n 24000: 103;
  `freeze/pmrt_nl_cost_pilot.json`) and 290 at n 4000 (DEV run `results/dev/pmrt_nl`), cdl 439 at n 4000 (its EVAL
  grid ends at n 4000, R-58(1)), mscr_eq 192 (n 1000; mscr_eq_min, 719, dropped, R-58(5)), two_tower 156, shap_dag
  148, pc_eq 80, pmrt_eq 63, pmrt_r3 60, every other arm <= 28 (n 24000). Calibration (R-55; xm-citests
  `results/exp_c/calib/factors.json`, xm/exp-c aee2fab, Python 3.12.14, 51 paired units per host type): Kaggle
  Xeon 2.20GHz = reference (f 1); Colab Xeon 2.20GHz f 1.067, e .101; VPS AMD EPYC-Rome f 2.562, e .253. Converted
  max over the calibrated host types, cost x f x (1 + e): pmrt_nl_eq 1148 (n 24000), cdl 439 (n 4000), mscr_eq 223
  (n 1000, Colab), none > 7200. DEV costs from uncalibrated host types are provisional reads, each far from the
  budget: Kaggle AMD EPYC 7B12 max 268 (cdl n 4000; over only if f > 26.8), Lightning Xeon Platinum 8488C 207 (cdl
  n 4000; f > 34.7), Colab AMD EPYC 7B12 121 (mscr_eq n 1000; f > 59.4). f and e by host type are recorded in the
  freeze commit (T7).
- T4 admitted CI-test arms, HAC arms and cdl: iff (i) the F1-F6 line in `status/xm-citests.md` (HAC: `status/hac.md`;
  cdl: its fidelity line in `status/cdl.md`, R-50) and (ii) the verdict in `status/audit-citests.md` are PASS or
  PASS-WITH-NOTES (audit table: OK, or a doc / protocol FIX whose fix is merged; mscr's M1 protocol fix is R-40, so
  mscr, a faithful port, stays admissible; Q11), and (iii) its DEV run is in the T1 input. rcot2's liberal authors'
  null is a disclosed note, not a failure. Others removed, with reason. VALUE: admitted: pcorr (eq, eq_min, native; F2
  / F4 PASS, audit OK), pcorr_hac and pcorr_hac_fb (+ eq_min; F2 / F3 PASS, `status/hac.md`, `hac2.md`), rcot2 (eq,
  eq_min, native; F2 / F3 PASS, F4 liberal authors' null disclosed, audit OK), mscr (eq, native; F3 / F4 PASS, audit
  M1 fixed by R-40; a reported-INVALID arm at n 500 / 1000, R-54; mscr_eq_min dropped, R-58(5)), cdl (F2 PASS, F3
  PARTIAL at the paper's seed, F4 primary family VALID; R-53; R1 / R2 n <= 4000, R-58(1)); (iii): the CI arms' DEV run
  is plan C (`results/dev/ci_c`, 58 480 / 58 480 ok); cdl's DEV run (`results/dev/cdl`, 5080 / 5080 ok, full grid)
  is in the final T1 input (`results/dev/final`), so (iii) holds (R-58(7)). Removed: pdcor (R-48), cmi_knn (R-49
  revised).
- T5 DEV validity (R-21): computed by `eval_analysis.py` on the DEV records, reported in the freeze commit; not used.
  VALUE: `results/dev/full/REPORT.md` (10 arms, 824 cells: VALID 268, INVALID 249, INCONCLUSIVE 247, NO_READ 60),
  `results/dev/ci_c/REPORT.md` (13 CI arms, 1138 cells: VALID 122, INVALID 444, INCONCLUSIVE 572), and the final
  DEV report over every DEV record incl. cdl and pmrt_nl_eq (`results/dev/final/REPORT.md`, 2086 cells; cdl 100
  cells: INVALID 45, VALID 30, INCONCLUSIVE 13, NO_READ 12; pmrt_nl_eq 24 cells: VALID 11, INCONCLUSIVE 13, 0
  INVALID); paths and LF sha256 in the freeze commit.
- T6 S cap: 100, unless `campaign project` at S 100 with the DEV costs (incl. citests) exceeds the compute the user
  states at the freeze; then the largest multiple of 10 that fits, not below 40 (R-34). Projection
  (`eval_projection.json`, DEV costs, 1 process per vCPU, no Lightning; wall with Kaggle 4 sessions + Colab 3 CPU
  jobs + the VPS (7), then Kaggle + VPS, then Kaggle only; R-58 grid; pmrt_nl_eq DEV run + T3 cost pilot, cdl
  Kaggle costs; host speed by the R-55 factors at f_lo, the VPS ~2.1 Kaggle processes each): S 40 = 150 960 units,
  ~510 CPU-h, ~14 / 17 / 32 h; S 50 = 168 970, ~560 CPU-h, ~16 / 18 / 35 h; S 60 = 186 980, ~620 CPU-h,
  ~17 / 20 / 39 h; pmrt_nl_eq ~61 % at S 40. The stated
  compute ceiling (R-56: free tiers; S 100 risks incomplete cells) fits S 60. VALUE: 60, so S = clip(S_power, 40,
  60) (T1: S = 40).
- T7 freeze-commit contents: campaign EVAL mode (audited), all adapter branches, `uv.lock`, filled `pkgs_lock`, this
  file FROZEN, the estimand-check re-run. VALUE: one commit on feat/v2 containing `cdd_oran/xmethod/campaign.py` and
  `dev_power.py` (xm/dev-runs), every adapter of the spec (`cdd_oran/xmethod/methods/`: pmrt_core, pmrt_nl, pc,
  granger, corr, notears, shap_dag, two_tower, cdl + _cdl_model, mscr, pcorr, pcorr_hac, rcot2 and their shared
  modules), runner / api / score / covariates / worlds, the launchers (`scratchpad/e6_dev/` kaggle_job, colab_run,
  xm_dispatch), `uv.lock`, the spec with `protocol_sha256` and `pkgs_lock`, `eval_analysis.py` and the report
  generator `eval_report.py` (renders section 12 from eval_analysis' outputs; frozen with it), `results/fmax_sim/`,
  the DEV inputs of T1 / T3 / T5 (merged files or their LF sha256), the T3 calibration block (R-55: f and e by host
  type) and cost-pilot records, the R-55 per-record host logging in campaign.py, the
  estimand-check re-run, and this file FROZEN; checklist `scratchpad/xmethod/freeze/FREEZE_CHECKLIST.md`.
- T8 set-D HAC variant (R-38): pcorr_hac vs pcorr_hac_fb on synthetic F4 only (AR(1) rho .5 / .8 and iid, n 500 /
  1000 / 4000): the smaller max |rate - .05| gets `set_D: true`, the other false (secondary). VALUE: pcorr_hac_fb
  (fixed-b .0405 vs t .0535; calib F4, xm/hac2 `results/pcorr_hac_f4b.json`); pcorr_hac (t) secondary.
- T9 S_E4 (R-39): the smallest of {200, 300, 400, 600} at which an exactly nominal PMRT arm passes C3 with
  probability >= .90 in `fmax_simulate` (`results/fmax_sim/s_e4_rule.json`; the simulation does not depend on the
  statistic). VALUE: 300 (P .93; 200: .82); the C3 set stays E4 R3, 4 lambda x 5 n (R-58(3) fallback not triggered).
- T10 PMRT statistic of pmrt_nl_eq (R-42): statistic chosen by the R-42 rule. Among the candidate nonlinear
  statistics (each a function of the redrawn random part and quantities fixed under the redraw, so the CRT stays
  exact) that pass exactness (synthetic F4 + DEV truth-null and placebo rates not INVALID in any cell, R-30) and the
  R-13 budget at n 4000, pick the highest mean per-edge recall at raw p <= .05 over DEV E2 / E5 R1 / R2 n 1000 + the
  synthetic nonlinear scenarios; ties -> the cheaper. DEV seeds 3_000_100-159 only (3_000_160-189 reserved). Worker
  pmrt-nl applies it (status `pmrt-nl.md`). The winner's option and version string go into pmrt_nl_eq's spec config
  (filled below). The linear pmrt_eq stays as the secondary arm. Fallback: if no candidate
  passes the gates, pmrt_nl_eq is removed from the spec and the linear pmrt_eq is the primary PMRT arm (spec
  `focal: pmrt_eq`, analysis primary, its R-42 label dropped) in every rule above that names the PMRT arm. VALUE: gbm
  (R-52; `results/pmrt_nl/SELECTION.md`): spec config `{"covariates": "eq", "statistic": "gbm"}`, version
  `pmrt-core-v2` (module `pmrt-nl-v1`); all three candidates passed, gbm recall .749 (rff .666, poly .638); no
  fallback.
- T11 EVAL grid of the mscr arms at large n (R-47): decided by the user (R-54). Under CI DEV plan C mscr has DEV
  records at n <= 1000 only (the R-47 question also named pdcor and cmi_knn, both dropped since: R-48, R-49
  revised). VALUE: the mscr arms (eq, native; eq_min dropped, R-58(5)) run at n 500 / 1000 only (tune, measure, E4
  R3 / R4) and not in the kappa sweep (R-58(5)); the other cells are "not in grid" (section 7), not a T3
  infeasibility; mscr is a reported-INVALID arm, never in C2b (spec `tbd.mscr`).

## 14. Deviations

After the freeze this file is not edited. Amendments (date, what, why, records touched) go to
`docs/xmethod/PROTOCOL_A_AMENDMENTS.md` and, if they touch records, `specs/eval/amendments.json` (id, commit, keys;
persistent errors: key, reason), before the affected records are analysed where possible. A bug found after EVAL:
fix, re-run every affected unit under a new commit listed there, report original and corrected results. EVAL never
feeds back into a method, tau or this protocol; a second EVAL needs a fresh seed block and a new protocol version.
