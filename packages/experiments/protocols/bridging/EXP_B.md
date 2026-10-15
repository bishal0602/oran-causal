# Experiment B: the Study A method set on the frozen E6 v4 data (descriptive bridge)

Status: DRAFT plan, code ready, NOT launched (worker exp-b, branch xm/exp-b from feat/v2 5a811de; 2026-10-04).
Open decisions: `scratchpad/xmethod/status/exp-b.md` (## Questions).

## 0. What this is and is not

- Purpose: run the final Study A method set (PROTOCOL_A s.4-5: `pmrt_nl_eq` primary, linear `pmrt_eq` / `pmrt_r3`, the
  CI tests, the HAC baseline, classic discovery, the O-RAN baselines, `corr`, `granger`, `cdl`) on the EXISTING frozen
  E6-P v4 data. This bridges E6 (where the v4 confirmatory study ran) and Study A (E1-E5).
- DESCRIPTIVE only: no verdict and no claim. The v4 protocol (`docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md`, FROZEN),
  its artifacts (`E6P_MSCRPLUS_V4_FROZEN.json`, `E6P_PMRT_V4.json`), its pinned code and its PARTIAL verdict are
  untouched. Experiment B is disclosed as post hoc, like Experiment D (R-15).
- Nothing is regenerated. The v4 episodes are read through the v4 analyzer's own loaders. Nothing frozen is edited.
  No Study A seed (DEV 3_000_000+, EVAL 3_100_000+) is generated or read.
- The PMRT arms here are Study A's `pmrt_core`, not the frozen v4 PMRT. They fit their predictable adjustment
  past-only on each slice, as in E1-E5. The frozen v4 primary used a kernel learned on ev2 and frozen. Its stored
  declarations appear as a reference row only (section 5).

## 1. Data (frozen v4, read-only)

The data lives only in Kaggle kernel outputs (`res_*.jsonl`, record `e6p-disc-rec/1`). There is no local copy.

| stage | seeds | episodes | Kaggle source kernels (bishalpanta/) | use here |
|---|---|---|---|---|
| eval | 188100-188699 | 600 | `e6p-disc-v4ev-1-a`, `-b` | measurement slices |
| placebo | 188700-188739 | 40 | `e6p-disc-v4plc-1-a` | exact sharp null: accept applied whatever the logged mode |
| gt | 188800-188839 | 40 | (stored result only) | truth: `analysis_v4.json["gt"]`, copied verbatim to `scratchpad/xmethod/results/exp_b/gt_v4.json` (source sha256 recorded) |
| dev | 188000-188019 | 20 | `e6p-disc-v4dev-1-a` | not used (Q3) |

Slices are those of the v4 analyzer (`e6p_disc_analyze_v4.slice_plan`; split ids as v4): pooled 600 (split 0),
s60 x 10 (20-29), s120 x 5 (40-44), s300 x 2 (60-61, descriptive in v4 too) and the placebo logs (9). One dataset
= one (slice, family). That gives 19 slices x 4 families = 76 datasets.

Unit counts (H 90 / H_pre 90 unit set; from the stored v4 analysis, re-checked against the built tables in the cloud):

| slice | carrier | sleep | ptx | prot_min |
|---|---|---|---|---|
| s60_0 (typical s60) | 1 935 | 279 | 1 502 | 5 064 |
| s120_0 | 4 027 | 545 | 3 057 | 9 751 |
| s300_0 | 10 094 | 1 538 | 7 611 | 25 197 |
| pooled | 20 276 | 3 072 | 14 988 | 50 731 |
| placebo logs | 670 | 123 | 678 | 3 633 |

(`scratchpad/xmethod/specs/exp_b/counts_v4.json` has every slice.)

## 2. Mapping to the Study A interface (`cdd_oran/xmethod/e6_bridge.py`)

- **Row.** One randomised CRT unit of family f (carrier / sleep / ptx / prot_min). The unit set is the one PMRT
  tested in v4. Rows are in information order (episode, t0).
- **Action.** `A_<f>` = sgn x L. L is the logged mode's level (accept 1 / reject 0). sgn is the proposal's direction,
  sign(prop - cur). This is intention to treat: the LOGGED draw, not the applied mode.
- **Design** (E6 logged everything PMRT needs: the mode draw, its pi0 row and sgn). `logged`, `categorical_rows`
  (the harness R3 convention) over (-1, 0, 1), with per-row propensity [P(L=1) on the unit's sign, P(L=0)]. In v4 that is .5 / .5. The centred action
  equals v4's `v_design`, so a design-based test redraws exactly v4's randomisation.
- **Placebo.** `P_placebo` = sgn x Bernoulli(P(L=1)): the same design, drawn independently (RNG
  [7800, 6, family, dataset seed]). It is the Study A truth-free negative control (R-10).
- **KPIs.** The 15 v4 targets (own / nbr / far x pv / v / e / rlf / load). `X_kpi_lag` is the pre-window sum
  [t0 - 90, t0). Y is the post-window sum [t0, t0 + 90). Y - lag is the v1-v3 unit outcome.
  - E6 holds the UE population fixed, so own + nbr + far load = 27 000 in every window.
  - The far-load LAG is therefore set all-NaN (adapters drop NaN-only lag columns), by a generic rule: a lag
    column that is a linear combination of the earlier ones. It is recorded in `meta.lag_dropped_collinear`.
  - The far-load target stays.
- **Context.** The 36 numeric obs-only ctx keys at t0, `hist` (past design-centred modes, own cell / exposure
  neighbours, 150 s), sgn and t0 / T. These are PMRT's predictable unit features, all pre-assignment. Encoding, per
  dataset, truth-free and in fixed order:
  - all-NaN keys are dropped;
  - partly missing keys are 0-filled, plus a `:missing` indicator (R-25);
  - only columns that add rank to (intercept, actions, kept lags, earlier context) are kept (`full_rank_columns`);
  - a second pass drops the context column with the smallest standardised residual variance given all the others
    while any is below 1e-6 (`drop_near_collinear`).

  Without this, E6's structural redundancies (sum-to-one shares, energy ~ power state, a key observed only as 0)
  make pcorr STOP on every candidate. That was seen in the smoke.
- **Time.** `time_index` = 0..n-1 in information order, contiguous. The eq arms' "actions at t-1, t-2" are the
  previous units of the family. For an episode's first units they come from the previous episode: an independent
  earlier draw, so still pre-assignment. There are no gap rows; Study A's adapters never saw gap indicators, and
  the citests stopped on them.
- **Candidates.** The 30 primary pairs (A_f, K) and (P_placebo, K). There is no lagged-KPI -> KPI family, because v4
  has no truth for it.
- **Not a fork.** campaign.py (xm/dev-runs) does the running. Its `run_units` receives Experiment B's units, and its
  dataset hook is rebound to the bridge (`exp_b.Loader`).

## 3. Arms (spec `scratchpad/xmethod/specs/exp_b/exp_b.json`)

Refs, configs and declaration modes are copied from `specs/eval/full.json`, R-52 (gbm) and R-53 (cdl).

| arm | rule | conditioning / information in E6 | note |
|---|---|---|---|
| pmrt_nl_eq (primary PMRT) | by | redraws the logged design; eq covariates | gbm statistic (R-52) |
| pmrt_eq, pmrt_r3 | by | eq / R-3 set | linear secondary, ablation |
| pcorr_eq, pcorr_native | by | eq / R-3 | |
| rcot2_eq, rcot2_native | by | eq / R-3 | authors' null liberal (disclosed in Study A) |
| mscr_eq, mscr_native | by | single-conditioner max | n <= 1000 only (R-54): 18 of 76 datasets |
| pcorr_hac_fb | by | R-3, time order (episode, t0) | Study A set-D member (T8) |
| pc_eq, pc_native | tau | eq: setpoints none, lagged actions tier 0 / R-3 | |
| notears, shap_dag, two_tower, cdl | tau | no conditioning interface (native) | cdl also `fixed` .16 (secondary) |
| corr | by | none | |
| granger_eq, granger_native | by | one-step rows: VARX / own lag | E3-only in Study A; here each row is one pre -> post step |

Omitted:
- `*_eq_min`: E6 has no dither design, so no setpoint exists and eq_min equals native.
- `pcorr_hac` (t): secondary in Study A; the user's list names fb.
- pdcor, cmi_knn: dropped from Study A (R-48, R-49).

What eq adds over native in E6: the lagged actions (previous units) and the concurrent designed placebo. The R-3 set
already holds the context, because E6 context is logged.

## 4. Truth reference (descriptive)

- **GT-TRUE (29), with sign.** The v4 GT cells (`gt_v4.json`, 40 GT episodes, knock-out contrasts) are the recall
  target.
- **GT-NULL (21).** This is an EQUIVALENCE label (GT CI inside +- delta), not a sharp null. Experiment D found two
  NULL cells with real sub-delta effects. Rates are therefore also given on set B: the 12 NULL cells whose GT CI
  contains 0, the closest available proxy for a true null. INDET (10) cells are unscored.
- **Exact nulls.**
  - The P_placebo column in every dataset (60 per slice).
  - The 60 real hypotheses on the placebo logs.

## 5. Declarations, metrics, outputs (`exp_b table`)

**Rules.** Every rule a method has is applied:
- `by`: the adapter's own BY at q .05 over the dataset's primary family (30 candidates; Study A's rule).
- `raw`: p <= .05 per hypothesis, for p arms.
- `tau`: score > conformal placebo tau (R-29 formula), for every arm.
  - s60 / s120 / s300: tau is leave-one-slice-out over the OTHER slices of the same kind and family. M = 15 x
    (slices - 1), i.e. 135 / 60 / 15 placebo scores.
  - pooled and placebo logs: tau comes from the dataset's own 15 placebo scores, labelled in-sample (Q2).
- `fixed`: cdl's .16.

**Metrics.** Per (arm, slice kind, rule), pooled over the 4 family datasets of each slice:
- recall on GT-TRUE;
- rejection rates on GT-NULL, set B, P_placebo and the placebo logs, with cluster-bootstrap CIs over slices
  (`campaign.cluster_ci`);
- sign accuracy;
- AUROC of the score, TRUE vs NULL;
- status counts (ok / infeasible / error) and cost.

**Reference row.** The frozen v4 primary (loadsp_c + wby1s), recomputed from the stored hypothesis tables with the
same definitions. It is a reference only; nothing is re-run.

**Outputs.** `scratchpad/xmethod/results/exp_b/{merged.jsonl.gz, exp_b_tables.json, EXP_B_TABLES.md}`.

## 6. Cost and launch plan (cloud only; laptop = tiny smoke only; NO Lightning)

**Budget.** The R-13 budget is 7200 CPU-s per (arm, dataset) under campaign's RLIMIT_CPU. A unit over budget is
recorded "infeasible at this n (measured X)", and the same arm is not run at larger n (campaign registry). Units
total 1328: 76 datasets x 17 arms + 2 x 18 mscr.

**Projection** (`exp_b project`): the Study A DEV cost per dataset (max over worlds), interpolated in n, times an E6
size factor per arm (guessed from candidates / conditioners / nodes; replaced by the pilot):

| | core-h |
|---|---|
| all arms | ~71 |
| of which cdl (x20 guess; 5 units over budget) | ~55 |
| everything else | ~16 |

The laptop smoke (section 7) calibrates the factors at small n.

**Phase P0: prep + pilot, one Kaggle session** (4 CPUs, sources = the 3 v4 kernels; R-55: at most one process
per vCPU, campaign's semaphore; orchestrator GO 2026-10-04 when < 4 of the account's Kaggle sessions run).
1. Build the unit tables (minutes; Experiment D needed ~7-15 s per slice after caching).
2. Check the counts against `counts_v4.json` (`counts --check`, exit 3 on a mismatch).
3. Run every arm on s60_0 and the placebo logs (8 datasets): the cost pilot.
4. Pull the tables (~30 MB; `kaggle_job.py pull --pattern '*.npz'`) for the Colab shards.

**Phase P1: the remaining units** across Kaggle and Colab:
- Kaggle runs one session at a time while the cdl DEV run holds slots.
- Colab CPU jobs take the cheap parts with the tables bundled.
- Partitioning uses the pilot's cost table (`--cost-table`, campaign's LPT partition, a dataset's arms on one
  platform).
- The large cdl / pc / shap / two_tower units of pooled and s300 go to Kaggle (11 h session limit).
- Estimate: 1-2 Kaggle sessions + 2-4 Colab jobs, under one day of wall time once slots are free.

**P0 result (2026-10-04, Kaggle xm-expb-p0b, 9f54800).**
- 144 / 144 ok; counts equal the stored v4 analysis in all 19 slices.
- Measured max CPU-s at n <= 5064: cdl 4036, notears 927, mscr 385, pc_eq 222, pmrt_nl_eq 200, the rest <= 41.
- P1 projection (`scratchpad/xmethod/results/exp_b/p1_projection.json`):
  - ~54 core-h within budget (cdl 31, notears 13);
  - 12 cdl units (datasets >= ~9.1k units) projected over the 7200 s budget: R-13 "infeasible at this n".
- The size-factor guesses above are superseded.

**P1 platforms (2026-10-04).** Kaggle `xm-expb-p1a` (parts 0-3) and `xm-expb-p1b` (parts 4-7), Py 3.12.14; the
user's VPS `xm-expb-p1-v1` (parts 8-15, worker exp-c), Py 3.12.13, budget 2784 VPS CPU-s = 7200 / 2.586.
- VPS cost conversion (orchestrator ruling, exp-c vps.md Q4): VPS CPU-s are converted to the Kaggle reference with
  `scratchpad/xmethod/results/exp_b/factors_vps_py3.12.13.json`, a verbatim copy of xm/exp-c aee2fab
  `scratchpad/xmethod/results/exp_c/calib/factors_vps_py3.12.13.json` (sha256 1aa2fab4...; Kaggle reference
  xm-expc-cal-k1 vs VPS cal-v1, host "vps|AMD EPYC-Rome Processor"). Reason: the same interpreter as the run, and the
  factor the budget came from, so a cdl infeasible unit sits at exactly 7200 ref-s.
- Rule: ref CPU-s = VPS CPU-s x the arm's `f` when the file has that exact arm name (cdl 2.368, pmrt_nl_eq 2.331,
  shap_dag 2.340, rcot2_eq 1.992, pcorr_eq 1.975), else the pooled `*` factor 2.687 (no transfer across arm names,
  e.g. mscr_eq is not given mscr_eq_min's 3.013); cdl infeasible units are reported at the budget (7200 ref-s,
  f_hi 2.586). The Py 3.12.14 factors (cal-v2) are not used for these parts.

**P2.** `exp_b merge` (missing / error report), then `exp_b table` locally.

**Commands** (dry run first):
```
uv run python -m cdd_oran.xmethod.exp_b kaggle --spec scratchpad/xmethod/specs/exp_b/exp_b.json --name xm-expb-k1 --parts 4 --dry-run
uv run python -m cdd_oran.xmethod.exp_b colab  --spec ... --name xm-expb-c1 --parts 4 --part-set 0,1 --tables scratchpad/xmethod/results/exp_b/tables
```

## 7. Validation done (laptop, no v4 data)

- **Synthetic tests.** `tests/test_xmethod_exp_b.py` uses a synthetic unit table with a planted sleep -> nbr effect.
  It checks the mapping (design, propensity zeros, time gaps) and runs counts -> run -> merge -> table end to end
  with corr / pcorr_eq / pmrt_eq.
- **Adapter smoke.** All 19 arms ran on the local non-v4 E6 cache (`.tmp/mscr_plus/infra/cache/dev1.npz`, 20 v3-era
  DEV episodes, pi0 .5 / .2 / .3), sleep and carrier. Results are in the status file.

## 8. Integrity

- Records carry campaign's stamps: code commit / dirty, spec sha, pkgs, host, dataset sha256.
- One platform per dataset.
- Dataset seeds are 60 000 000 + 10 x split + family index, disjoint from every Study A block.
- The GT extract records the source file's sha256.
- The cloud prep rebuilds the tables from the frozen JSONL and checks the counts against the stored v4 analysis.

## 9. Results (descriptive; 2026-10-04)

**Runs and coverage.** 1328 / 1328 expected units have a record (P0 `xm-expb-p0b`, Kaggle `xm-expb-p1a` /
`xm-expb-p1b`, VPS `xm-expb-p1-v1`): 1316 ok, 12 infeasible, 0 missing / error / duplicate / conflict, no
mixed-platform dataset, 0 dirty records (commits 9f54800, bd03132, 9bbeeb4, 219bf0b). The prep in each Kaggle session
rebuilt the tables and matched the stored v4 counts in all 19 slices. The 12 infeasible units are all cdl, and they
are exactly the 12 the P0 projection flagged (datasets of n >= 9 751): R-13 "infeasible at this n". mscr runs only at
n <= 1000 (R-54), i.e. the sleep family of the 60- and 120-ep slices (15 of 72 measure datasets).

**Files** (`scratchpad/xmethod/results/exp_b/`): `merged.jsonl.gz` (+ `.summary.json`), `exp_b_tables.json` /
`EXP_B_TABLES.md` (every arm x slice kind x rule, rates with hits / n and slice-cluster CIs, dataset coverage),
`exp_b_costs.json` / `EXP_B_COSTS.md` (VPS converted with `factors_vps_py3.12.13.json`, s.6).

**Headline** (rule "by" = the adapter's BY declaration for p arms; "tau" = LOSO conformal placebo threshold, in-sample
for pooled; INDET excluded; `ds` = measure datasets with an ok record, of 72):

| arm | rule | ds | recall 60 / 120 / 300 / pooled | GT-NULL rate 60 / 120 / 300 / pooled | P_placebo 60 | placebo logs |
|---|---|---|---|---|---|---|
| frozen v4 primary (reference) | | | 0.56 / 0.62 / 0.69 / 0.79 | 0.10 / 0.10 / 0.10 / 0.10 | | 0.00 |
| pmrt_nl_eq | by | 72 | 0.39 / 0.49 / 0.55 / 0.55 | 0.05 / 0.05 / 0.07 / 0.14 | 0.00 | 0.00 |
| pmrt_eq | by | 72 | 0.49 / 0.55 / 0.62 / 0.72 | 0.05 / 0.06 / 0.10 / 0.10 | 0.00 | 0.00 |
| pmrt_r3 | by | 72 | 0.49 / 0.55 / 0.59 / 0.72 | 0.05 / 0.07 / 0.10 / 0.10 | 0.00 | 0.00 |
| pcorr_eq | by | 72 | 0.52 / 0.58 / 0.57 / 0.72 | 0.05 / 0.06 / 0.10 / 0.10 | 0.00 | 0.00 |
| pcorr_hac_fb | by | 72 | 0.52 / 0.57 / 0.57 / 0.69 | 0.05 / 0.06 / 0.10 / 0.10 | 0.00 | 0.00 |
| granger_eq | by | 72 | 0.52 / 0.58 / 0.57 / 0.72 | 0.05 / 0.06 / 0.10 / 0.10 | 0.00 | 0.00 |
| rcot2_eq | by | 72 | 0.38 / 0.41 / 0.48 / 0.48 | 0.05 / 0.07 / 0.07 / 0.10 | 0.01 | 0.03 |
| mscr_eq (sleep only) | by | 15 | 0.63 / 0.82 / - / - | 1.00 / 1.00 / - / - (1 NULL hyp.) | 0.59 | 0.36 |
| pc_eq | tau | 72 | 0.11 / 0.12 / 0.19 / 0.14 | 0.10 / 0.06 / 0.12 / 0.14 | 0.04 | 0.07 |
| notears | tau | 72 | 0.19 / 0.20 / 0.22 / 0.24 | 0.05 / 0.05 / 0.05 / 0.05 | 0.01 | 0.05 |
| shap_dag | tau | 72 | 0.27 / 0.26 / 0.33 / 0.28 | 0.07 / 0.07 / 0.05 / 0.05 | 0.04 | 0.00 |
| two_tower | tau | 72 | 0.27 / 0.29 / 0.22 / 0.34 | 0.10 / 0.10 / 0.07 / 0.10 | 0.05 | 0.15 |
| corr | by | 72 | 0.61 / 0.73 / 0.79 / 0.83 | 0.51 / 0.58 / 0.62 / 0.62 | 0.59 | 0.17 |
| granger_native | by | 72 | 0.62 / 0.70 / 0.74 / 0.76 | 0.41 / 0.55 / 0.69 / 0.76 | 0.40 | 0.22 |
| cdl (partial) | tau | 60 | 0.40 / 0.49 / 0.44 / 0.25 | 0.08 / 0.23 / 1.00 / 1.00 (300, pooled: 1 NULL hyp.) | 0.05 | 0.05 |

Native-covariate variants and the raw / tau / fixed rules are in `EXP_B_TABLES.md`.

**Observations** (descriptive, no test; the v4 confirmatory result is unchanged):
- The covariate-adjusted linear arms (pmrt_eq, pmrt_r3, pcorr, pcorr_hac_fb, granger_eq) behave almost identically:
  BY recall 0.49-0.52 at 60 eps rising to 0.69-0.72 pooled, GT-NULL 0.05-0.10, 0-1 / 600 placebo-column rejections
  at 60 eps and none on the placebo logs. All stay below the frozen v4 primary's recall
  (0.56-0.79).
- pmrt_nl_eq (Study A primary) has lower recall than the linear arms on E6 (0.39-0.55 BY), with GT-NULL 0.05-0.14.
- Unadjusted marginal tests (corr, granger_native) and mscr reject many GT-NULL hypotheses and the P_placebo column
  (0.40-0.59 at 60 eps): consistent with confounding by the logged state, which the adjusted arms remove.
- Graph learners (pc, notears, shap_dag, two_tower) recover 0.11-0.34 of GT-TRUE at a null rate 0.05-0.14.
- cdl and mscr rows have partial coverage; their large-slice / pooled null rates rest on the single sleep NULL
  hypothesis and are not comparable with full-coverage rows.
- GT-NULL is an equivalence label, not a sharp null (s.4); set-B and placebo-log columns are the sharp-null views.
- CIs are a cluster bootstrap over slices (10 / 5 / 2 for 60 / 120 / 300): with 2-5 clusters they are narrow or
  zero-width and should not be read as precision.

**Cost** (Kaggle reference, `EXP_B_COSTS.md`): 82.0 ref core-h in total, cdl 53.7 (24.0 of it = the 12 infeasible
units at 7200 s), notears 12.9, pc_eq 5.5, pmrt_nl_eq 4.1, every other arm <= 2.0.
