# Study A supplementary experiments X2 / X3: declaration (R-60)

DECLARED 2026-10-05, before launch and before any EVAL output was read. The author opened no EVAL
result, shard, merge or table. The commit that adds this file is the
declaration. Later changes go only into the dated "Amendments" section at the end, and the report lists them.

**Status: EXPLORATORY / supplementary.** These runs never change the C1-C3 verdicts, the Study A claim or the E6
headline (PROTOCOL_A is untouched).

## 1. Rules (R-60)

- **Own files.** Specs live in `specs/posthoc/`, with their own output directories. Never
  `specs/eval`, never EVAL shards.
- **Frozen methods and procedures.** Every arm's ref, config and declaration are copied from the frozen
  `specs/eval/full.json` (a test checks this). tau is the frozen R-29 conformal cutoff. Scoring uses the frozen
  `eval_analysis.py`.
- **Wrapper only, no frozen file edited.** `oran_causal.experiments.campaign.extras` overrides two generator constants (X2) or the
  design told to a method (X3) in its own process only. The freeze manifest
  (`data/records/freeze/freeze/FREEZE_MANIFEST.sha256`, LF rule) stays valid, and a test checks it.
- **Seeds.** A fresh block, 3_200_000-3_200_199, claimed as XMETHOD_EXTRAS for E1-E4 in
  `manifests/seeds.json` (section 2). Nothing touches seeds 3_100_000-3_100_299.
- **Launch timing.** Start only after EVAL has launched all 96 parts (Kaggle) and its last VPS job has been pulled.
  EVAL requeues keep priority: keep one Kaggle slot free until the EVAL merge checks pass.
- **Order.** X1 > X2 > X3. X3 runs only if capacity remains. X1 (paper timing, Exp C) is declared by its own
  pre-registered block, not here.
- **Every declared run is reported**, whatever the outcome. Cells that are infeasible, in error or missing are
  printed as such, never dropped. Error records are re-run (campaign resume). A unit over budget is infeasible.

## 2. Seeds (XMETHOD_EXTRAS)

| range | use |
|---|---|
| 3_200_000-3_200_019 | X2 tune (20 seeds; tau of the score-only arms, per design) |
| 3_200_020-3_200_059 | X2 measure (40 seeds per design) |
| 3_200_060-3_200_119 | X3 measure (60 seeds) |
| 3_200_120-3_200_199 | unassigned; any use needs an amendment before its launch |

- **X2 designs share the seed numbers** (common random numbers). The dither of a seed is the same uniform draws
  scaled by delta, so the delta comparison is paired.
- **The wrapper refuses** any other seed for an extras spec. Campaign's own guard still applies to every other spec.

## 3. X2: dither dose-response (C1 mechanism)

**Question.** How do the truth-null and P_placebo rejection rates, and recall, of these arms change with two design
settings?
- The arms are: the design-blind arms corr, pc_eq, notears, shap_dag and pcorr_native; the design-covariate arm
  pcorr_eq; and the design-based arms pmrt_eq and pmrt_nl_eq.
- The settings are: the dither amplitude delta at the frozen 20 setpoint blocks, and the number of setpoint blocks at
  the frozen delta .10.
- No directional hypothesis is tested. The result is descriptive.

**Cells.**
- E1, E2 and E3, regime R2, n 1000 and 4000, kappa .25.
- 6 designs (delta, blocks): (.02, 20), (.05, 20), (.10, 20), (.20, 20), (.10, 5), (.10, 80). One spec each:
  `x2_d02`, `x2_d05`, `x2_d10`, `x2_d20`, `x2_b05`, `x2_b80`.
- Only R2 reads the two constants. The envelope (.25) and everything else stay frozen.

**Reproduction cell.** (.10, 20) is the frozen EVAL R2 design. At equal seed its datasets equal the frozen
generator's bit-for-bit:
- tested in `tests/test_posthoc_extras.py`;
- the analysis regenerates every dataset of `x2_d10` with the frozen generator and requires equal hashes;
- for the other designs it requires every R2 dataset to differ.

**Seeds and declaration.**
- Tune 3_200_000-019 for the score-only arms pc_eq, notears and shap_dag only. Their tau is re-derived per design and
  cell by R-29 on that design's own tune records.
- Measure 3_200_020-059 for all 8 arms.
- p arms declare by pooled BY (q .05), as frozen. No conformal tau is computed for p arms.

**Metrics.** As in section 5.

## 4. X3: design-misspecification stress test (C2a assumption)

**Question.** Where does the known-design assumption break? The data are generated with the frozen design; only the
design TOLD to the method is wrong. The result is reported as a degradation curve, not as evidence against C2a
(which assumes the design is known).

**Cells.**
- E1 R2 and E2 R2 at n 1000.
- E4 R3, lambda 1, at n 1000 (propensity distortion; the code is small, so it is included).
- kappa .25, measure seeds 3_200_060-119 (60). One spec: `x3_told`. The data are generated once per dataset and
  shared by every variant.

**Arms.** pmrt_nl_eq, pmrt_eq, pmrt_r3, and pcorr_eq as the contrast. Each runs with the exact design (arm name
unchanged) and with every variant (arm name suffix).

**Told variants.** Each variant changes every applicable design of the dataset, including P_placebo's dither:
- **Dither width** x0.5, x0.8, x1.25, x2 (`.w050`, `.w080`, `.w125`, `.w200`; E1 / E2 R2): each dither design's law
  U(-w, w) is told as U(-c w, c w). The realised random and fixed parts are unchanged.
- **Switch times late by 2 % / 5 % of a block** (`.s02`, `.s05`; E1 / E2 R2):
  - the told setpoint is the true one shifted s rows later, s = floor(f L + .5) with L = n / blocks = 50 rows, so
    s = 1 and 3 rows (effective 2 % and 6 %; the rounding is declared here);
  - told random part = action - told setpoint, so it is exact wherever the told setpoint is right.
- **Logging lambda** x0.5, x0.8, x1.25, x2 (`.l050`, `.l080`, `.l125`, `.l200`; E4 R3):
  - every logged design's propensity table is recomputed with the told lambda from the observed Z;
  - the i.i.d. P_placebo design is not touched, so its rate is a control.

**Expected invariances** (verified on one dataset before this declaration; any difference is a harness bug, reported
as such):
- pcorr_eq is identical under the width and lambda variants, because it never reads the design law or the
  propensity;
- pcorr_eq changes under the switch-time variants, because its eq set uses the told setpoints as covariates.

**Declaration.** All X3 arms are p arms (BY). There are no tune seeds.

## 5. Metrics (as the frozen eval_analysis defines them)

Per cell (spec / design, arm, world, regime, lambda, n, kappa), from `eval_analysis.screen` and `build_cells` on
the spec's merged records. Only expected units with the expected role enter.

- **Truth-null rate.**
  - Applies to the primary-family (action -> KPI) candidates with zero true effect.
  - "decl" = declared under the arm's primary declaration (BY for p arms; score > tau for tau arms).
  - "raw" = p <= .05 (p arms).
  - Denominator: the testable candidates. Not-testable candidates are left out of every denominator.
- **P_placebo rate** (and P_placebo_conf in E4 R3): the same, over the placebo -> KPI candidates. For tau arms
  P_placebo is the tuning column (reported, not a validity rate).
- **Recall:** the mean over measurement seeds of per-seed recall (true primary edges declared / true primary edges).
- **CIs:** seed-cluster 95 % percentile bootstrap (2000 reps, seed 20261002).
  - A cell's clusters are its seeds.
  - Pooled rates per (spec, arm, regime) use `eval_analysis._pooled`, with clusters = (world, seed), over all cells
    of that arm.
- **The R-30 class** (INVALID if CI low > .05, VALID if CI high <= .075, else INCONCLUSIVE) is printed as a
  descriptive label, not a verdict.
- **Status and provenance:** the cell status (ok / few_seeds / under_seeded / infeasible / missing / untuned), the
  record counts and the provenance check (section 3; X3 requires every record's generated-data hash to equal the
  frozen generator's).

Output: `uv run python -m oran_causal.experiments.posthoc x2x3` writes `extras_tables.json` and `EXTRAS_TABLES.md` (cells, pooled, X2 trend, X3
curve).

## 6. Report wording (exact; the placeholders are filled from extras_tables.json)

**Lead (always, verbatim):** "These supplementary experiments are exploratory: they were declared before launch
(protocols/posthoc/PROTOCOL.md) on a fresh seed block, use the frozen methods and procedures, and do not
change C1-C3." Followed by "(declaration commit <sha>)".

**X2, one sentence per arm and rate** (truth-null decl; truth-null raw and P_placebo raw for p arms):

> "X2 (dither dose-response; R2 of E1-E3, kappa .25, n 1000 and 4000, 40 measurement seeds per design): the pooled
> <rate> of <arm> was <r> [<lo>, <hi>], <r> [...], <r> [...] and <r> [...] at dither delta .02, .05, .10 and .20 (20
> setpoint blocks), so it <trend> with delta, and <r> [...] / <r> [...] with 5 / 80 setpoint blocks at delta .10."

- **`<trend>`** is the declared rule on the four point estimates in increasing delta:
  - "rises" if they never decrease and at least one increases;
  - "falls" if they never increase and at least one decreases;
  - "does not change" if all are equal;
  - otherwise "is not monotone";
  - "not reported" if a design has no rate.
- **Appended sentence:** "The delta .10 / 20-block design is the EVAL R2 design (datasets bit-identical to the frozen
  generator at equal seed); its seeds are disjoint from EVAL."

**X3, one sentence per base arm, regime and family** (width narrower x0.8 then x0.5; width wider x1.25 then x2;
switch late 2 % then 5 %; lambda lower x0.8 then x0.5; lambda higher x1.25 then x2):

> "X3 (design misspecification; data generated with the frozen design, the design told to the method wrong; 60
> measurement seeds): with the <family> the pooled truth-null raw rate of <arm> in <regime> was <r> [...] (exact),
> <r> [...] and <r> [...], and its P_placebo raw rate <r> [...], <r> [...] and <r> [...]; the first variant with an
> INVALID pooled rate is <variant | none>."

- In E4 R3 the sentence also gives the P_placebo_conf raw rate.
- **Closing sentence (verbatim):** "This is a degradation curve showing where the known-design assumption breaks; it
  is not evidence against C2a, which assumes the design is known."
- **Recall** of every cell and pooled arm is in the tables. The text gives recall only alongside the rates above, in
  the same format.

## 7. Cost and launch (DEV cost table; Kaggle-reference CPU-h, an upper estimate)

`specs/posthoc/PROJECTION.md`:
- each X2 spec: 2280 units, 360 datasets, 9.6 CPU-h (pmrt_nl_eq 7.1);
- x3_told: 4560 units, 180 datasets, 19.6 CPU-h (pmrt_nl_eq E4 R3 extrapolated: DEV had no pmrt_nl E4 cost);
- total 77.2 CPU-h.

Launches go through `oran_causal.experiments.launch`.

## Amendments

### 2026-10-05: X4, random accept / defer referee (Study 3 / E6-P certsafe baseline)

**POST HOC, DESCRIPTIVE.** X4 was added on 2026-10-05, after the Study 3 results were known: the certsafe EVAL
(`protocols/study3/PROTOCOL.md`; verdict PARTIAL; CS:PMRT identical to never_sleep on all 160 seeds) and
the report `reports/2026-10-02-v4-pmrt-confounded-certsafe/`. Requested with user approval. X4 tests no hypothesis and has no criterion. It changes no frozen artifact, verdict, protocol, driver or
analyzer of Study 1-3, and no X2 / X3 rule above. The commit that adds this entry is the declaration; it precedes
every X4 episode (the 1-2 seed local plumbing test below is not a result).

**Question.** Where does a referee that defers at random, blind to the request and to any map, land on the Study 3
scale (R, retention, guard ratios, eligibility)? It is a reference point for MG:PMRT's deferral volume.

**Policy** (`X4:rand@<p>`):
- Each xApp request reaching the referee is deferred (decision "reject") independently with probability p and
  accepted otherwise. Every family and direction is treated alike. No writes, no rollbacks.
- Active from t = 0, warm-up included (as MG:PMRT and never_sleep).
- RNG: one stream per episode, `default_rng([seed, 6640])`, one uniform u per request in arrival order
  (obs["requests"] order within a step). Defer iff u < p. All three p share the stream (common random numbers), so the
  three arms make nested decisions until their request streams diverge.
- Tag 6640 is new. It was checked free against `manifests/seeds.json` rng_stream_tags. The registry is not edited:
  its sha256 is pinned in the E6-P record headers.

**p, three arms:**
- **p_PMRT = .352.** This is MG:PMRT's request-level deferral rate in the certsafe EVAL: deferred requests / requests
  reaching the referee = 37 890 / 107 525 = .35238 (records `data/records/study2_3_decision/runs/e6p-cs-eval-1/*/res_*.jsonl`,
  160 seeds, all t), rounded to .352.
- For reference only: the certsafe composition of MG:PMRT counts directional units, not requests. It deferred
  carrier 71, sleep 869, ptx 8313 and prot_min 0 units, and accepted 25 805, a unit-level rate of
  9 253 / 35 058 = .264. X4 defers requests, so the request-level rate is the one matched.
- **.25 and .50.**

**Episodes.**
- Seeds: the Study 3 EVAL seeds, 191100-191259 (160), cfg seed = seed. These are no new seeds; X4 claims none.
- Plant: identical to Study 3: `e6p_screen.make_cfg(e6p_conf.PAIR, e6p_conf.STRATUM, seed, lf)`, lf = L40 from
  `e6p_state.json`, 120 s warm-up unscored + 600 s scored.
- Jobs: 160 seeds x 3 arms = 480. The record schema is that of the certsafe EVAL with sub "x4". Every record carries
  p, realised deferrals per family / phase, requests, cpu_s and the platform.
- Code: one new file, `oran_causal.experiments.posthoc.x4`, plus its test. It imports the frozen driver modules
  and edits none of them.

**Platform.**
- Runs on the VPS (`oran_causal.experiments.launch`, 7 processes in the capped systemd scope).
  Python 3.12.14, numpy 2.4.2 / scipy 1.18.1 (uv.lock pins).
- The stored Study 3 arms ran on Kaggle (Linux, Python 3.12.13, same numpy / scipy).
- **Provenance check, run first in the same VPS job.** noarb and never_sleep on seeds 191100-191101, made with
  the frozen driver's arbiters. Every outcome field is compared with the stored Kaggle record.
  - If all are equal, the plant is bit-identical across the two platforms.
  - Otherwise the differences are reported, and the X4 comparison is labelled cross-platform (still reported).
- **Cost.** X4 cost is VPS CPU-h, converted to Kaggle-reference CPU-h:
  - with `data/records/posthoc_x1/exp_c/calib/factors.json` (pooled "*" factor, VPS Python 3.12.14);
  - and with the direct factor of this workload: stored Kaggle cpu_s / VPS cpu_s of the four provenance jobs.

**Scoring** (the Study 3 code, unchanged):
- Inputs: the stored EVAL records (20 shards; the aliased CS arms filled from their targets as
  `e6p_certsafe_analyze.cmd_eval` does), plus the X4 records.
- Call: `e6p_conf_analyze.arm_stats_multi` (N_BOOT 10 000, `default_rng([6624, 20, 160])`: the same bootstrap
  index matrix as `cs_eval.json`).
- Hence V_AA, V_ref, the reference arm and the denominator are the stored ones. Every stored arm must reproduce
  `cs_eval.json`; any mismatch is a harness bug, reported.
- **Per X4 arm:** V, R, R*, retention, guard ratios (svr, nonprot_embb_viol, ll_viol, rlf), eligible (the point
  rule, unchanged), and 90 % CIs (paired seed bootstrap). Also the realised deferral rate (deferred / requests) and
  deferrals per family.
- **Comparisons:** paired dR (90 % CI) to noarb (accept-all), never_sleep, MG:PMRT, CS:PMRT and CS:GT, all
  descriptive.
- MG:GT is not a Study 3 arm and has no record on these seeds, so the table shows CS:GT and states this.

**Output.**
- `data/records/posthoc_x4/x4/` (records mirror, `x4_tables.json`) and
  `data/records/posthoc_x4/X4_REPORT.md`.
- The report has one table (X4 arms next to never_sleep, MG:PMRT, CS:PMRT, CS:GT, accept-all) and a three-line
  descriptive reading.
- **Never:** a verdict, a claim, or a change to Study 3's E / D1-D3 / S1.
- Missing or failed jobs are reported as such, never dropped. A failed job is re-run (resume by key).

### 2026-10-05: X7, are the associational referees the static subset rule sub:ES+PowerES? (Study 3 / E6-P certsafe)

**POST HOC, DESCRIPTIVE** (advisor question D; user approval). X7 was added after the Study 3 results
were known. In those results the associational referees had R about -10 to -11 and deferred about 100 % of prot_min,
as the Gate A anchor sub:ES+PowerES does; so far only the totals and CIs had been compared.
- No hypothesis, no criterion. No frozen file, artifact or verdict of Study 1-3 is changed.
- The commit that adds this entry is the declaration and precedes every X7 episode.

**Arms compared.**
- Associational referees: four frozen certsafe-artifact jobs:
  - CS:granger@dev ("granger")
  - CS:granger_by
  - CS:two_tower@dev ("two_tower")
  - CS:shap_gbdt@dev ("shap")
  - Each is `DirectionalUnitArbiter(CertSafeMapGateV2(M, uncertified))`, built by `e6p_certsafe.make_arbiter`
    from `artifacts/study3_certsafe.json`.
  - The @plc variants alias to noarb (accept-all) in the artifact and are not compared.
- Reference: the Gate A anchor sub:ES+PowerES (`e6p_screen.make_arbiter` -> `baselines.subset`).
- Seeds and plant: the 160 Study 3 EVAL seeds 191100-191259 and the Study 3 plant, as X4.

**A1. Code check** (no run). The report states, with file:line:
- what a static-rule rejection and a referee deferral do to a request (decision value, ACK / NACK, re-queue);
- when each acts (warm-up);
- how the xApp re-submits;
- the unit / direction mechanics of the referees.

**A2. Per-request agreement**, logged by re-running with decision logging. The stored records have counts only.
- **(i) Reference stream (primary).**
  - One job per seed runs sub:ES+PowerES (its decisions applied).
  - At every second the four referees are queried in shadow on a deep copy of the same obs. Their decisions are
    logged, not applied, so each referee's internal state evolves on the reference's request stream.
  - The episode must be bit-identical to the stored sub:ES+PowerES record (checked).
- **(ii) Own stream (secondary).**
  - Each referee re-runs as itself (decisions applied). The subset rule is evaluated in shadow on the same requests.
  - The episode must be bit-identical to the stored referee record (checked).
- **Agreement** = share of requests (each request reaching the referee = one request unit) with the same decision
  (accept vs reject). Reported:
  - overall and per control parameter (prot_min, ptx, carrier, sleep), and per direction;
  - all t and the scored window (t >= 120 s) separately;
  - with the 2 x 2 decision counts.
  - CI: seed-cluster percentile bootstrap, 2000 reps, `default_rng([20261005, 7])`.

**A3. Attribution** (replays). Per referee, two arms:
- `X7:<ref>|pm`: the referee's own decisions, with every deferral of a non-prot_min request replaced by accept.
- `X7:<ref>|other`: every prot_min deferral replaced by accept.
- The override acts on the output only. The arbiter's internal state (units, duty counters) is the frozen code's,
  unaware of it.
- 8 arms x 160 seeds. Per arm: V, R, R*, retention, guard ratios (svr, nonprot_embb_viol, ll_viol, rlf),
  eligibility, 90 % CIs.

**A4. Paired differences.**
- R and V of each referee and attribution arm minus sub:ES+PowerES, with 90 % CIs, on the paired seed bootstrap.
- Also: the number of seeds whose stored referee record is bit-identical to sub:ES+PowerES in every outcome field,
  and the per-seed dV distribution.

**Scoring.** As X4: the Study 3 code unchanged.
- `e6p_conf_analyze.arm_stats_multi` on the stored 26 arms plus the X7 arms.
- N_BOOT 10 000, `default_rng([6624, 20, 160])`.
- Stored arms must reproduce `cs_eval.json`.
- dV uses the same index matrix (pooled V per resample).

**Run.**
- New file `oran_causal.experiments.posthoc.x7` plus its test; it edits no frozen code.
- VPS (capped scope with 7 processes, Python 3.12.14, same pins), one job at a
  time, coordinated with cdl's X5. Not Kaggle.
- Jobs: 160 (A2 i) + 640 (A2 ii) + 1280 (A3) = 2080. At about 31 VPS CPU-s each, about 18 VPS CPU-h, about 2.6 h
  wall.
- Missing or failed jobs are reported, never dropped. A failed job is re-run (resume by key).

**Output.**
- `data/records/posthoc_x7/x7/` (x7_tables.json, records) and
  `data/records/posthoc_x7/X7_REPORT.md`.
- The report gives the numbers, one suggested Discussion sentence (descriptive analogy or not) and a stated
  confidence. Never a verdict.

### 2026-10-05: X5, PMRT-Lin's E4 R3 C3 failure: chance or real excess? (advisor question B)

**POST HOC, EXPLORATORY (R-60).** Declared on 2026-10-05, after the EVAL results were known. Requested with user
approval. The cells were chosen *because* they failed, which is why step 2
uses fresh data. X5 changes no frozen file, verdict (C1-C3) or EVAL output.

- Step 1 reads the frozen EVAL records read-only.
- Before writing this entry, the author read only these EVAL fields for E4 R3 pmrt_eq / pmrt_r3:
  - `cpu_s` / `gen_cpu_s` / platform, for the cost;
  - the record structure: one P_placebo -> K0 and one P_placebo_conf -> K0 candidate per dataset;
  - the published `results/eval/csv/e4_by_lambda.csv` rates.
- No p-value was read before this entry.

**Question.** In EVAL, pmrt_eq got the INVALID label in three E4 R3 C3 cells:
- (lambda 0, n 8000): P_placebo_conf raw .080;
- (lambda .5, n 8000): P_placebo_conf raw .083;
- (lambda 1, n 24000): P_placebo raw .083.

Each rate is over 300 datasets. Are these chance (expected by multiplicity) or a real excess over .05?

**Step 1: existing EVAL records (descriptive; no new run).** Inputs: every E4 R3 cell (lambda 0 / .5 / 1 / 1.5 x
n 500 / 1000 / 4000 / 8000 / 24000), arms pmrt_eq and pmrt_r3, candidates P_placebo -> K0 and
P_placebo_conf -> K0, measurement-role records only. Reported:
- **(a) Uniformity.** KS distance and p against U(0, 1), and a QQ table (empirical p quantiles at .01, .025, .05, .10,
  .20, .50). CRT p-values are discrete and slightly super-uniform, so KS is read as descriptive.
- **(b) Tail.** Rejection rates at alpha .01, .025, .05 and .10, and observed / expected. The question is whether the
  excess sits in the whole lower tail or only just below .05.
- **(c) Concentration.** Do a few datasets carry the excess?
  - Overlap of the rejecting seeds (p <= .05) between P_placebo and P_placebo_conf in a cell.
  - Overlap across the four lambdas at the same n. EVAL seed numbers are shared across cells.
  - Each overlap is compared with its expectation under independence.
- **(d) Label count by chance.**
  - Compute the probability that one C3 rate at exactly .05 gets the INVALID label. Method: the frozen seed-bootstrap
    rule (2000 reps) applied to 300 independent Bernoulli(.05) datasets, 10 000 simulations, seed 20261005.
  - From it: the expected number of INVALID labels among an arm's 40 E4 R3 C3 rates (20 cells x 2), and P(count >=
    observed).
  - Rates are treated as independent. This is an approximation: P_placebo and P_placebo_conf share each dataset.
- Step 1 decides nothing. It is read together with step 2.

**Step 2: fresh datasets.**
- **Seeds.** 3_300_000-3_301_999 (2000; XMETHOD_DIAG, E4). The same seed numbers are used in every cell (common random
  numbers, as in EVAL).
- **Cells** (E4 R3, kappa .25):
  - failing: (lambda 0, n 8000), (lambda .5, n 8000), (lambda 1, n 24000);
  - adjacent, i.e. the other lambdas at the same n: (lambda 1, n 8000), (lambda 1.5, n 8000), (lambda 0, n 24000),
    (lambda .5, n 24000), (lambda 1.5, n 24000).
- **Arms.** pmrt_eq and pmrt_r3, with ref, config and declaration copied from the frozen `specs/eval/full.json`.
- **Specs.** Two specs, run in this order: `x5_fail` (the 3 failing cells) and `x5_adj` (the 5 adjacent cells).
- **Rates.** Per cell and arm: P_placebo raw and P_placebo_conf raw = #(p <= .05) / #testable datasets (one candidate
  per dataset), with a 95 % Clopper-Pearson CI.
- **Primary test.** The three (cell, rate) pairs that failed for pmrt_eq. For each: a one-sided exact binomial test of
  H0 rate <= .05 against H1 rate > .05, Holm-adjusted over the 3 at family-wise .05. Outcome per pair:
  - **EXCESS** if the Holm-adjusted p < .05;
  - **CHANCE** if not EXCESS and the CP upper bound <= .075 (the R-30 VALID bound);
  - **UNRESOLVED** otherwise.
- **Answer to B:**
  - "chance" if all three are CHANCE;
  - "real excess" if at least one is EXCESS (named);
  - else "unresolved".
- **What each answer means.**
  - Chance: the EVAL failure is consistent with what multiplicity over many cells produces. The frozen C3 verdict
    stands, and the Discussion explains it.
  - Real excess: PMRT-Lin has a genuine size distortion of the measured size in E4 R3 at that setting. The verdict
    still stands (it is frozen), and the Discussion states the magnitude.
  - The adjacent cells show whether an excess is specific to the cell or general at that n.
- **Secondary (descriptive, unadjusted):** every cell x arm x rate, with CP CI, one-sided binomial p and R-30 label.

**Step 3: lagged-action covariates removed.**
- In E4 R3 there are no setpoints. Removing the lagged-action columns from the eq covariate set therefore leaves
  exactly the frozen r3 set (lagged KPI + context). A test checks this on X5 data: pmrt_r3's covariates = pmrt_eq's
  minus the 6 lagged-action columns.
- So step 3 is the pmrt_r3 arm on the same datasets, and needs no extra run.
- It is read only if some primary pair is EXCESS. In that cell, compare pmrt_eq with pmrt_r3 by an exact McNemar test
  (paired over datasets, two-sided .05).
  - pmrt_r3 lower and McNemar significant: the lagged-action covariates are implicated.
  - Otherwise: they are not.
- If nothing is EXCESS, step 3 is reported as not triggered. The pmrt_r3 numbers are still tabled.

**Cost** (EVAL unit costs per platform; one generation per dataset):
- Per seed over the 8 cells: 157.5 VPS CPU-s.
- x5_fail: 50.5 s per seed, 28 VPS CPU-h, about 4 h on 7 processes.
- x5_adj: 107 s per seed, 59 VPS CPU-h, about 8.5 h.
- Total 87.5 VPS CPU-h, about 172 Kaggle-reference CPU-h.

**Platform.**
- VPS (7 processes in the systemd scope), py 3.12.14 pinned venv.
- One job at a time, coordinated with X4, then X7.
- A preflight refusal stops the lane and is reported.

**Pre-run amendment (2026-10-05 ruling; nothing of X5 had run, no step-1 p-value had been read).**
- `x5_adj` is trimmed to 1000 seeds per cell: 3_300_000-3_300_999, the first half of the X5 block (still common random
  numbers with `x5_fail`). SE about .007 at .05, which still separates .05 from .08.
- `x5_fail` keeps 2000 seeds per cell. The primary test, outcomes and step 3 are unchanged (the primary pairs are all in
  `x5_fail`); the adjacent cells' secondary CIs are wider.
- Cost: x5_adj 29.5 VPS CPU-h (about 4.3 h on 7 processes); X5 total 57.5 VPS CPU-h.
- Order on the VPS: x5_fail after xm-citests' X7, then x5_adj.

### 2026-10-05: X6, PMRT-GBM under a wider told dither: the centring mechanism (advisor question C)

**POST HOC, EXPLORATORY (R-60).** Same authorship and rules as X5. In X3, pmrt_nl_eq told the dither x2 had a pooled
truth-null raw rate of .100 [.090, .110], while pmrt_eq became conservative (.000).

**Hypothesis** (advisor, read from `methods/pmrt_nl.py`):
- The GBM statistic is T(V) = sum_t h_t(v_t) w_t / sd, with w_t = y_t - m_t. Here m_t is the design mean, under the
  TOLD law, of the past-only GBM profile f_t(v).
- Under the true design, c_t = E_true[f_t] - E_told[f_t] is not 0. Then E[w_t] is about c_t, so T_obs sits above
  the told-law redraws by about sum_t c_t^2 / sd. This mean shift causes the inflation.
- For PMRT-Lin, T = sum_t v_t w_t, and both laws have the same mean (symmetric, same centre), so c_t = 0. The wider
  redraws only widen the null, which makes it conservative.
- Shifting h_t by a per-row constant moves T_obs and every redraw alike, so the p-value depends on m_t only through
  w_t.

**Design** (trimmed to the cells that answer the question): E1 R2, n 1000, kappa .25. X3 showed the same inflation in
E1 (.100) and E2 (.101), and E1 costs less than half as much per GBM unit.
- **Seeds:** 3_302_000-3_302_199 (200; XMETHOD_DIAG, E1).
- **2 x 2 ablation at told width x2.** Every dither design of the dataset, P_placebo included, is changed:
  - profile grid: always the told law (+-2w), as in the frozen told run;
  - centring law (told / true): the law of m_t, which is used for h and for w_t;
  - redraw law (told / true): the law the CRT draws V from.
- **The four cells:**

  | cell | centring | redraw | note |
  |---|---|---|---|
  | TT | told | told | the frozen told arm `pmrt_nl_eq.w200` |
  | Tt | told | true | |
  | tT | true | told | |
  | tt | true | true | on the told grid |

- **How m_t is computed.** "told" uses the frozen exact formula. "true" uses the Gauss-Legendre mean (64 nodes) of the
  told-grid profile under the true law U(-w, w).
- **Bias log** (every GBM arm, per action and target):
  - c_t = E_true[f_t] - E_told[f_t], both by Gauss-Legendre (64 nodes) on the told grid;
  - z_bias = sum_t c_t w_t / sd, where sd is the frozen T's denominator, i.e. the predicted shift of T_obs over the
    redraws in null-sd units;
  - mean c_t, and corr(c_t, w_t) over the post-burn rows.
  - For narrower told widths, the true support extends past the told grid, so the profile is clamped there (stated).
- **Finer told-width grid** (frozen told runs): x .5, .8, 1 (exact), 1.25, 1.5, 2, for pmrt_nl_eq and for pmrt_eq.
- **Hooks.** In-process overrides in `extras.py` style (`pmrt_nl.design_law` / `gbm_profiles` / `GbmStat` wrapped
  during an X6 unit only, restored on exit). No frozen file is edited.
  - Tests: with logging on, TT's p-values equal the frozen told run's bit-for-bit, and the exact arm's c_t is 0.
- **Arms** (15): pmrt_nl_eq x {exact, .w050, .w080, .w125, .w150, .w200, .w200 Tt, .w200 tT, .w200 tt}, and pmrt_eq x
  {exact, .w050, .w080, .w125, .w150, .w200}. All are BY p arms; there are no tune seeds.

**Metrics.** As in section 5 (frozen eval_analysis), per arm: the truth-null raw rate (12 candidates per seed) and the
P_placebo raw rate (4 per seed), seed-cluster bootstrap CI, R-30 label; plus the bias log.

**Predictions of the hypothesis:**
- **P1 (CRT sanity).** Tt and tt (true redraw) are not INVALID on the truth-null raw rate. A true redraw makes T_obs
  and the redraws exchangeable whatever the statistic. A violation is a harness bug, reported.
- **P2 (reproduction).** TT's truth-null raw rate is INVALID (CI low > .05).
- **P3 (mechanism).** tT's truth-null raw rate has CI high <= .075 (VALID or conservative), i.e. true centring removes
  the inflation.
- **P4 (bias term).**
  - In TT, the mean z_bias over (seed, truth-null candidate) is > 0, with a seed-bootstrap 95 % CI excluding 0.
  - In TT, the rejection rate in the top z_bias tercile exceeds the bottom tercile's (seed-bootstrap CI of the
    difference excluding 0).
  - In tT, |mean z_bias| < 1/4 of TT's.

**Outcomes:**
- **SUPPORTED:** P1-P4 all hold.
- **REFUTED:** P2 holds and tT is INVALID. The inflation then persists under true centring, so it does not come from
  told-law centring.
- **PARTIAL:** any other combination, with the failed predictions named.
- **The finer grid is descriptive:** the rate and mean z_bias against told width, GBM next to Lin.

**Cost** (X3 VPS unit costs: pmrt_nl_eq E1 n 1000 12.7 s, pmrt_eq .5 s):
- 200 x (9 x 12.7 + 6 x .5) = 23 500 VPS CPU-s, i.e. 6.5 VPS CPU-h, about 15 Kaggle-reference CPU-h (pmrt_nl_eq factor
  2.28).
- Colab (TPU v5e-1 host, 4 processes): about 2 h wall.

**Platform.** Colab, through the frozen campaign's colab lane (`--venv-python 3.12.14`, uv venv). Or the VPS, if it
is free first.

**Report** (both X5 and X6): `data/records/posthoc_x5_x6/DIAG_REPORT.md` and `diag_tables.json`. Each gives:
- what was run, and the numbers with CIs;
- which declared outcome happened;
- one suggested Discussion sentence, and its confidence.

### 2026-10-11: X8, stricter-threshold null-outcome check of the frozen v4 PMRT (journal review 1.2, decision D10)

DECLARED 2026-10-11 by the advisor, before the X8 run. Post hoc with respect to the Study 1 freeze; it never changes
the Study 1 verdict. Trigger: the journal assessment (reviews/2026-10-10/assesment.md, point 1.2) asks for evidence on
tail accuracy where the declaration layer acts; the frozen K0n check (protocols/study1/DISCOVERY.md section 5) reports
only rates at p <= .05.

**Already seen before this declaration (disclosed).**
- (a) A reproduction of the frozen K0n at its own settings (40 variants, shifts 1-4, B 999) with the new runner, to
  check the runner: 2400 used p, rate(p <= .05) .0525, family rates carrier .0517 / prot_min .0517 / ptx .060 /
  sleep .0467, 0 declarations, all equal to the frozen analysis_v4.json K0n. These 40 variants are re-run at B 9999
  as part of X8.
- (b) One timing variant at B 9999 (group 0, shift 5): 0 primary declarations, min used p .0017.
- (c) Design fact found while sizing X8: at n_target 60 the primary layer's (loadsp_c + wby1s, q .05) weights lie in
  [.2, 3.94] (sum 60, m 60, c_60 = 4.68), so the rank-1 weighted-BY threshold is at most .05 x 3.94 / (60 x 4.68)
  = 7.0e-4. At the frozen K0n B 999 the smallest attainable p is 1e-3, so the frozen K0n criterion "variants with >= 1
  declaration <= 5 of 40" could only fail through several simultaneous small p's; a single declaration was not
  attainable. X8 runs at B 9999 (min p 1e-4), where it is.

**Method and data (frozen, unchanged).** The frozen v4 PMRT artifact artifacts/study1_pmrt.json (sha256 and code sha256 checked
by e6p_disc_analyze_v4.artifact_status); eval_v4 records (Kaggle outputs e6p-disc-v4ev-1-a / -b, seeds
188100-188699, all 600 required). Runner `oran_causal.experiments.posthoc.x8` re-uses the frozen K0n recipe
(e6p_disc_analyze_v4.run_k0n): groups = the 10 eval_v4 60-slices; variant (g, k) gives episode i of group g the
lab_series of episode (i + k) mod 60; PB.run_integrated with RNG split SPLIT["null"] = 0; PB.declare_all with
n_target 60.

**Changes from the frozen K0n (the only ones).** B 9999 (the frozen discovery B) instead of 999; shifts k = 1..59
(every non-identity cyclic shift) instead of 1..4: 590 variants, about 35 400 used p's in the primary combination.

**Pre-stated checks (primary combination loadsp_c + wby1s; descriptive for the other 8):**
- **X8a (declaration layer).** Fraction of variants with >= 1 declaration <= .075 (under the complete null the BY
  guarantee FDR <= q = .05 is a family-wise bound, so the expected fraction is <= .05; .075 mirrors the K0n tolerance).
- **X8b (tail, 1e-3).** Pooled rate of used p <= 1e-3 <= 2e-3 (twice nominal).
- **X8c (tail, 2e-4).** Pooled rate of used p <= 2e-4 <= 6e-4 (three times nominal; about 7 expected).
- Outcome: **CONSISTENT** if X8a-c all hold; otherwise **NOT CONSISTENT**, naming the failed check. Every outcome is
  reported in the journal (Appendix C) whatever it is.
- **Descriptive:** rates at .05 / .01 / 1e-3 / 2e-4 / 1e-4 per combination and per family, independence binomial
  tails, a variant-cluster bootstrap 95 % upper bound of the 1e-3 rate, and the redraw p2 against the normal
  approximation 2(1 - Phi(|z|)) for each arm statistic.

**Caveats to report.** The 59 variants of a group share that group's actions and logged probabilities (only the
outcome series move), so the evidence is conditional on 10 logged designs, and p's within a variant are dependent
(hence the cluster bootstrap). Null outcomes from other episodes break the outcome-action link but keep the outcome's
own time structure.

**Platform and cost.** Local (16 cores, 6 processes); about 13 s per variant at B 9999, about 2.1 CPU-h in all.
Outputs: `data/records/posthoc_x8/x8/` (res_*.jsonl, summary.json, logs).

### 2026-10-11: X9-X13, journal-revision re-analyses of stored Study 1-3 data (decisions D3, D7; review 1.4, 1.5, 2.B, 2.E)

DECLARED 2026-10-11 by the advisor, before these analyses were run. Post hoc and descriptive: no verdict, no frozen
label, criterion or verdict changes; all frozen values stay primary. No new simulation: inputs are stored artifacts
(analysis JSONs, eval records, GT run records already produced by the frozen runs). Every generator must first
reproduce the frozen value at the frozen setting (named per item) and stops if it does not.

**Already seen before this declaration (disclosed).** The 2026-10-10 review audit read: the Study 3 certsafe arm
values (never-sleep R .256 / V 160.32 / retention 1.146; MG:PMRT R .201 / retention 1.634; CS:GT R .221 / retention
1.294; CS:PMRT = never-sleep; noarb V 175.30); a quick GT delta sweep (labels change with delta; PMRT sign accuracy
stays 1.00); far hypotheses GT-TRUE 8 of 15 (to be recomputed); option_a ptx -> nbr RLF GT mean -0.0611, CI
[-0.0962, -0.0271].

- **X9 (end-to-end decision table; T1.1, T1.2).** Study 3 arms accept-all (noarb), never-sleep, ungated MG:PMRT, certified
  CS:PMRT, oracle CS:GT, plus the gated associational arms: V, R, retention, the four guard ratios with their CI90,
  rejection rate, and paired differences against never-sleep (seed-paired, as the frozen eval). Reproduce: the frozen
  certsafe verdict-table values. Study 2 rows (option_a) where the stored records allow; else omitted with the reason.
- **X10 (gate diagnosis; T1.3).** For each CS:PMRT rejection class: the unresolved guard edge(s), h x beta, UB90, tau,
  the GT effect and label; flag edges whose GT effect is harmful beyond tau. Reproduce: the frozen certification
  outcome (CS:PMRT reduces to never-sleep).
- **X11 (GT sensitivity; T1.4-T1.6).** Relative delta x0.5 / x1 / x2 (pv floor fixed, and floor x0.5 / x2 at delta x1),
  GT CI level 90 / 95 / 99 % if the per-unit GT records are available (Kaggle e6p-disc-v4gt-1-a/b; Study 2 GT run);
  label counts and PMRT / baseline precision, recall, sign accuracy, Studies 1 and 2, with INDET counted wrong / right
  as bounds. Reproduce at x1 / 95 %: PMRT P .92, R .79, labels 29 / 21 / 10 (Study 1 pooled, as the paper).
- **X12 (threshold choice; T1.7).** Every Study 2 baseline at @dev and @plc: edges, right sign, wrong sign, GT-NULL,
  placebo declarations; state that @plc is set on the placebo logs (in-sample for the placebo criterion).
- **X13 (relation attribution; T1.8).** For each declared nbr / far edge (PMRT and baselines, Studies 1 and 2): is the
  same (family, KPI) own edge GT-TRUE, and is the declared edge GT-NULL / INDET? Counts of far hypotheses by GT label.

Outputs: `data/records/posthoc_x9_x16/revision/` (one JSON + CSV per item, REVISION_REPORT.md). Tables that enter
the paper get cddfig generators with tests in `figures/`.

### 2026-10-11: X14, map scores against an "effect exists" ground truth (user decision 2026-10-11; review 1.5)

DECLARED 2026-10-11 by the advisor, before X14 was computed. Post hoc and descriptive: no verdict, no frozen label,
criterion or verdict changes; the frozen material-effect rule stays the primary ground truth.

**Already seen (disclosed).** The X11 results (W1, REVISION_SECTIONS_W1.md), including the labels that change at
delta x2. Every declared edge that turns from TRUE to NULL at delta x2 was TRUE under the frozen rule, so its knockout
CI excluded zero. The existence-precision values themselves have not been computed.

**Rule (existence ground truth).** Same knockout records, pooled means and frozen 95 % episode-cluster bootstrap
intervals as the frozen GT (gt_p.gt_table draws), same support rule:
- EXISTS(sign) when the interval excludes zero, any magnitude, with sign = sign(mean);
- NOT-ESTABLISHED when the interval contains zero;
- UNSUPPORTED as the frozen INDET-unsupported (< 10 units or < 3 episodes), unscored.

**Scores** (PMRT primary and every baseline / map, Study 1 pooled and the 17 slices as stored, Study 2 as stored).
For declared edges:
- confirmed = EXISTS with the declared sign;
- wrong sign = EXISTS with the opposite sign;
- not established = NOT-ESTABLISHED.

Existence precision = confirmed / (confirmed + wrong sign + not established). Recall against the EXISTS set is
reported too, as is a side-by-side with the frozen material-rule precision. Descriptive: the same at 90 % and 99 %
intervals.

**Reproduction first.** The frozen material-rule metrics must be reproduced exactly (as in X11) before any X14 number
is used.

**Outputs.** data/records/posthoc_x9_x16/revision/x14_existence.{json,csv}, a section X14 in
REVISION_SECTIONS_W1.md, and a cddfig generator `tab_appC_existence` (or extra columns in tab_appC_gt_sensitivity if
that reads better; the choice is stated).

### 2026-10-11: X15, why did the ungated PMRT referee's contrast with never-sleep not replicate? (user, 2026-10-11)

DECLARED 2026-10-11 by the advisor, before X15 was computed. Post hoc and diagnostic: no verdict changes.

**Already seen (disclosed).**
- The same arm MG:PMRT (same maps artifact, same plant, same code per protocols/study3/PROTOCOL.md s.4) was run on two
  disjoint seed blocks: Study 2 option (a), 187000-187159, and Study 3, 191100-191259.
  - Study 2: R .470 [.346, .594], dR vs never-sleep +.249 [.139, .362], RLF ratio 1.31.
  - Study 3: R .201 [.025, .363], dR -.055 [-.216, .091], RLF 1.05.
- Accept-all V: 157.67 (Study 2) against 175.30 (Study 3). Never-sleep R .221 against .256.
- A rough independent-blocks z for the dR difference is about 2.6, assuming i.i.d. seeds.

**Questions and analyses** (stored per-seed EVAL records of both blocks; the frozen eval's metric definitions;
seed-bootstrap with the frozen N_BOOT and RNG convention where applicable):
- **Q1, seed blocks.** Are the two blocks exchangeable at the plant level? Per-seed accept-all V, never-sleep V,
  their distributions (mean, sd, quantiles, top-5 seeds), and a two-sample comparison (permutation test on the mean
  and a KS test).
- **Q2, chance or heavy tails.** Per-seed dR(MG:PMRT - never_sleep) in each block:
  - mean, sd, skew, the top-k seed contributions;
  - a two-sample permutation test of the block difference;
  - the leave-top-5-out estimate per block.
  The same for MG:GT, the comparison arm, and for never_sleep vs accept-all (control: does a well-behaved contrast
  replicate?).
- **Q3, mechanism.**
  - Does per-seed dR depend on seed difficulty (accept-all V; surge timing or load if stored)?
  - Does MG:PMRT's per-seed rejection count (ptx-up, sleep) differ between blocks?
  - Is the Study 2 advantage concentrated in seeds where never-sleep does poorly?
- **Q4, best estimate.** The pooled 320-seed estimate of R and dR for MG:PMRT, MG:GT and never_sleep against
  accept-all, with a seed-bootstrap 90 % CI (block-stratified), and the RLF ratio pooled.

**Reading rule (stated in advance).**
- If Q1 shows no block difference and Q2's permutation p >= .01, the non-replication is read as chance plus heavy
  tails, and the pooled estimate is the reported best estimate.
- If Q1 shows a block difference, the result is reported as seed-block sensitivity of the referee, with the
  mechanism from Q3.
- Either way, every number is reported.

**Outputs.** data/records/posthoc_x9_x16/revision/x15_replication.{json,csv} and a section X15 in a new
REVISION_SECTIONS_X15.md.

### 2026-10-12: X16, strictly predictable centre for the design slope and the gate bounds (journal review FU 2.2, N2 5)

**POST HOC, declared before it was run.** Added on 2026-10-12 by the advisor (user approval, "Text + X16 re-analysis"),
after every Study 1-3 result was known, in answer to three reviewers asking that the effect of the approximately
predictable centre be measured (reviews/2026-10-12). X16 re-analyses stored data only. It changes no frozen
artifact, verdict, protocol, driver or analyzer, and no earlier X entry. This entry is written before any X16 number
is computed.

**Question.** The design slope (e6p_conf_analyze.design_slope; certsafe.design_slope_se) uses the residual
r_u = y_u - centre_u from eprocess_units.predictable_residuals, whose running state is updated with every earlier
unit j in information order, even when j's outcome window [t0_j, t0_j + H] (H = 90 s) has not closed at t0_u. How
much do the PMRT map slopes, the gate's PMRT upper bounds and the gate's certification statuses change when the
centre uses only units whose windows have closed?

**Variant (X16:strict).** Identical to predictable_residuals (same strata, shrinkage, pooled slope, ridge, burn-in,
scale), except that unit j enters the running state used for unit u only if t0_j + H <= t0_u. Implemented as a new
function in the X16 script; the frozen function is not edited. The scale sigma_u is not used by the slope and is
reported only.

**Data.** The DISC unit table exactly as `e6p_certsafe_analyze.py bounds` builds it (records 186100-186699, stage
eval, disc_bench.load_pool H 90 / H_pre 90). If the cached unit table is not available locally, X16 stops and
reports; it does not re-simulate.

**Outputs, both centres side by side.**
1. Check: the frozen centre reproduces every MG:PMRT map |beta| and every stored PMRT bound (rel. tol 1e-9).
2. Map slopes: for every MG:PMRT map edge, beta (frozen, strict), relative change, sign agreement with the declared
   sign.
3. Gate bounds: for every (family, rel, guard KPI), beta, se, z and the one-sided 90 % upper bound of h * beta for
   each direction, under both centres; summary of |change in UB| / tolerance.
4. Certification: certsafe.certify on the frozen tolerances (tau from the frozen artifact) with the strict bounds;
   list every (f, d) class whose status (certified / unresolved) changes, and the CS:PMRT uncertified set and
   decision signature under both centres.

**Reading rule (stated in advance).**
- If no class changes status, the CS:PMRT signature is unchanged, so the Study 3 gated referee's decisions, and every
  Study 3 number, are unchanged under the strict centre; the paper states this with the largest map-slope change.
- If any class changes status, the paper reports which, with the bound change; whether to re-simulate Study 3 is a
  separate user decision.
- Every number is reported.

**Outputs.** data/records/posthoc_x9_x16/revision/x16_centre.{json,csv} and REVISION_SECTIONS_X16.md.
