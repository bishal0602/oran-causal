# E6-P discovery protocol: MSCR-CRT on P3 surge-L40 (step 1)

FROZEN: yes (2026-09-29)

**Status: FROZEN 2026-09-29, before any DEV, PLACEBO, EVAL or GT outcome was inspected.** Written from
scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md and the committed collection + GT half (3e5ce78). `e6p_discovery.py`
refuses `--stage eval|eval_ext|gt` unless this file's LF-normalised sha256 equals `FROZEN_SHA256` in the driver. Any
edit after the freeze is an amendment and needs fresh EVAL/GT seeds. Open items were resolved in section 11.

## 1. Question and estimand

Does MSCR, used as a design-based conditional randomization test on randomized arbiter logs, recover the true
cause→effect structure of the indirect xApp conflict on the E6-P plant? And does it beat the published and
standard baselines on the same data?

- **Edge space.** 60 hypotheses (f, rel, k):
  - knob family f ∈ {carrier, sleep, ptx, prot_min} (the unit's knob);
  - relation rel ∈ {own = {c}, nbr = N(c) − {c}, far = cells outside N(c)};
  - KPI k ∈ {pv (protected violated UE-s), v (violated UE-s, all slices), e (energy J), rlf (count), load (served
    UE-s)}.
  - N(c) is the unit's logged, obs-only exposure set (`units_p.exposure_sets`, symmetrised CIO graph).
  - far is the negative control.
- **Orientation "dir".** An edge is the effect of a knob-value INCREASE: sleep 1 = asleep, carrier = active
  carriers, ptx dB, prot_min share. The unit treatment is x = level × sgn:
  - level = accept 1 / half 0.5 / reject 0 of the logged π0 mode;
  - sgn = sign(prop − cur) of the unit's opening request, with 0 counted as +1.

  This is the same orientation as the GT (`gt_p.PRIMARY_ORIENT`). The "act" GT table is reported but not scored.
- **Declared estimand mismatch.** The GT contrast continues with every other unit accepted (AA). The CRT runs
  under π0 (other units randomized). This mismatch is declared, not corrected.

## 2. Plant, stages, seeds, tags

Everything below is fixed by the committed driver `scratchpad/e6_dev/e6p_discovery.py`. Its module docstring is
the record contract (schema `e6p-disc-rec/1`).

- **Plant.** `e6p_screen.make_cfg("P3", 3, seed, lf)`, with lf = 1.75546875 (L40). Episodes are 720 s: 120 s
  warm-up plus 600 s scored. The env is `E6Env(wg3=True, trace=True)`, the arbiter `units_p.UnitArbiter` with
  T = 60 s and `open_rule="feasible"`. During warm-up every request is accepted.
- **Logging policy π0.** `collect_p.PI0_HIGH_NO_RB`: accept 0.5 / half 0.2 / reject 0.3 for ES, PowerES and
  SliceGuarantee. Draws use `default_rng([seed, 6612, c, x_idx, t0])`.
- **Stages and seeds.** The block 180000-183999 is registered as `e6p_discovery_episodes`. Unused seeds in the
  block: 183444-183999.

  | stage | seeds | policy | use |
  |---|---|---|---|
  | dev | 183300-183319 (j 0-19) | π0 | τ tuning, pipeline check (allowed before the freeze) |
  | eval | 183320-183379 (j 20-79), fold = (j − 20) // 20 | π0 | the discovery test (FROZEN required) |
  | gt | 183380-183399 (j 80-99) | π0 + knockout labels | ground truth (FROZEN required) |
  | placebo | 183400-183419 | π0 modes drawn and logged, **accept applied** | K0 (sharp null) |
  | prof | 183420-183443 (3 profiles × 8) | only xApp X's units accepted | PACIFISTA native severity |

- **Tags** (`SEED_REGISTRY.json` `rng_stream_tags`):

  | tag | stream |
  |---|---|
  | 6612 | π0 draws |
  | 6613 | GT label sampling |
  | 6614 | region probes (unused here) |
  | 6615 | G0c bootstrap (unused here) |
  | 6616 | analysis draws |
  | 6617 | GT bootstrap |

  Tag 6616 carries these analysis streams:
  - CRT draws `[seed, 6616, family_idx, lag, split]`;
  - the CRT row order `[seed, 6616, 99, family_idx, lag, split]`;
  - the QACM holdout split `[seed, 6616, 7, 7, f, rel, k]`.

  The analysis seed is 0. Splits are 0 = pooled EVAL, 1-3 = folds 0-2, 9 = placebo.

## 3. Unit table (`cdd_oran/decision/crt_units.py`, `build_unit_data`)

- **Rows.** One row per opened unit of a family in section 1 whose windows fit in the series:
  t0 − 90 ≥ 0 and t0 + 90 ≤ 720. Other units are dropped and counted (about 15 % of units, all opened in the last
  90 s).
- **Target.** y(rel, k) = Σ over the relation's cells of [KPI over `lab_series` rows t0 … t0+89, minus the KPI
  over rows t0−90 … t0−1].
  - Rows t0 … t0+89 are env seconds t0+1 … t0+90. Rows t0−90 … t0−1 are seconds t0−89 … t0.
  - H = H_pre = 90. The "load" KPI is the series field `ue`.
- **Conditioners.** These are pre-assignment:
  - every numeric obs-only ctx field of the unit (`UnitArbiter.context`: cur, prop, step, since_change, n_exp,
    is_macro, the 13 own mediators and 17 aggregates over N(c) − c);
  - the pre-window value of every (rel, k) (15 columns).

  NaN values become the column median. Degenerate columns are dropped (`features.degenerate_columns`).
- **Never read.** `gt_static`, `gt_labels` and `lab_outcome`. A test poisons them.

## 4. Primary method: MSCR-CRT (`crt_units`, version `mscr-crt-units-v1`)

- **Null.** The sharp null of no effect of family f's assignments on the whole trajectory.
  - Under this null the unit skeleton is invariant, so conditional draws re-draw every f unit's mode from the
    logged table of its xApp (`pi0_table[x]`).
  - The other families' modes are held fixed, and outcomes are held fixed.
  - p = (1 + #{T_draw ≥ T_obs}) / (B + 1).
- **Statistic.**
  - Primary: the MSCR S* (`crt.MSCRStat`: frozen `mscr._s_star` for the observed value, the vectorised replica
    for the draws, fidelity checked on every hypothesis), with CRTConfig nc = 3, nb = 3, min_stratum = 10.
  - Secondary (reported): `crt.signed_stat`.
  - The 15 hypotheses of a family share one conditioner set and one set of draws, computed in one pass. This is
    bitwise equal to per-hypothesis `MSCRStat.draws`, asserted every run.
  - Sign = sign of the within-episode slope of y on x.
- **B = 9999.** This is an **amendment to the plan's 999**; see section 11.
  - The plan's 999 cannot declare anything unless ≥ 6 hypotheses sit at the floor: BY's rank-1 threshold over
    m = 60 is q / (m c_m) = 1.78e-4, and 1/(B+1) must not exceed it.
  - Draws are consumed in chunks of 1000 of the same stream, which gives the same p-values as one full draw.
- **Declarations.** BY at q = 0.05 over the tested hypotheses of one call. Statuses are declared, not_detected
  or undetermined, with the strict `crt.CLAIMS` wording. There is **no tuning**.
- **Support rule.** A hypothesis is tested only when the family has ≥ 30 units, ≥ 5 accepted, ≥ 5 rejected and
  ≥ 3 episodes. Otherwise it is undetermined, counted as not declared in the scores.
- **Carryover audit (lag 1).** Descriptive only; it never enters the verdict.
  - It tests the previous unit of the same (episode, cell, xApp, knob) against this unit's y and pre-window
    value, conditioning on the previous unit's pre-assignment values.
  - It uses B_audit = 999 and BY over the audit's own tests. At that B, BY is not reachable at rank 1 over 120
    tests, so the share of p ≤ 0.05 is reported next to it.
- **Receiving cell (localisation).** Per (EVAL seed, pico), over its sleep-request units (step > 0), each other
  cell q gets the score Σ_u (x_u − x̄)(y_uq − ȳ_q) of the load change, or Σ_u x_u y_uq when x does not vary in the
  group. The top-1 is the largest positive score.
  - The claim counts only if MSCR-CRT declared sleep → nbr load.
  - It is scored against `gt_static.cand` of that seed; the scorer is the only reader of that field.

## 5. Comparators and baselines (`cdd_oran/decision/baselines_disc.py`)

All unit-table baselines see the same unit table, x and y as the CRT, and the same support rule. The support rule
makes scores NaN, and NaN is never declared.

| id | method | score / sign |
|---|---|---|
| shap_gbdt | Sharma 2025 via `scripts/e2_baseline_shap_dag.fit_shap_importances`; one fit per hypothesis on [x, sgn, pre] | mean\|SHAP(x)\| / sd(y); sign(corr(x, y)) |
| corr | pooled Pearson | \|corr(x, y)\|; sign |
| granger | lagged F-test: OLS post = a + b·pre + c·x, F(1, n−3) on c | −log10 p; sign(c) |
| granger_by | the same p-values, BY at q = 0.05 (untuned; reported, not in P2) | — |
| two_tower | `scripts/e2_baseline_gnn.fit_two_tower`, `TwoTower` re-bound by a wrapper (script unedited): 4 params (x per family), 15 targets, d 16, r 8, hidden 32, 400 epochs, lr 0.01, **l1 0.5** | gate a[target, family]; sign(corr) |
| int | PACIFISTA INT (`envs.e6.published.int_distance`) of y·sgn, accepted vs rejected units | INT; sign(mean_acc − mean_rej) |
| qacm | QACM KPI predictor (`published._ANN` / `_PR`, better holdout R² kept): own-cell post KPI (z) on [level·step, cur, pre (z), own PRB util] | mean \|pred − pred(change 0)\| (z units); sign. OWN ONLY: nbr/far never declared |
| mscr_rowperm | frozen MSCR-v2 `discover_template` on the pooled obs-only panels (`panel_from_rec` + `concat_panels` + `drop_degenerate_p`, `kpi_owner=KPI_OWNER_P`, targets prot_viol / energy_w / rlf, n_perm 2999, thin 60 s) | mapped: own_/nbr_{carrier, sleep, ptx, prot_min} × {pv←prot_viol, e←energy_w, rlf} = 24 of 60 |
| pacifista_native | per-profile ECDFs of per-cell per-second KPIs; σ(a, b) = mean INT; conflict iff σ > 0.25 | xApp level only: not in the edge space |

- **SHAP score.** The score is absolute: the paper's per-KPI relative rule declares ≥ 1 family per target, far
  included.
- **Rowperm, unmapped.** v and load have no panel KPI, and far has no panel aggregate. Unmapped hypotheses are
  excluded from rowperm's scores and counted. Panel nbr means the CIO-neighbour mean, not the N(c) − c sum.
- **Two-tower l1 = 0.5.** The value was fixed on synthetic data only. At the script's 1e-3 the planted gate ranked
  at chance (3/3 seeds); at 0.5 it ranked first (3/3).
- **QACM tuning.** For the τ rule only, QACM is also fitted on DEV's far relation.
- **SHAP fallback.** If the SHAP script cannot be imported, the fallback is sklearn GBDT with permutation
  importance, recorded in the report.
- **τ rule (identical for every tuned baseline).** Tuning happens on DEV only; then τ is frozen and applied to
  pooled EVAL and each fold. A hypothesis is declared iff score > τ.
  - **Primary: "far_fpr".** τ = the 2nd-largest DEV far-relation score, so at most 1 DEV far declaration. That
    budget equals the far budget P1 grants MSCR. If fewer than 2 far hypotheses are scorable, the method is
    untunable and declares nothing.
  - **Why this rule.** It is label-free: DEV has no GT, and the GT seeds are locked. It uses only the negative
    control, so it is equally fair to every method and leaks nothing about which true edges are scored. It
    matches the specificity budget MSCR is held to. The alternative, a physics prior, is the planner's
    expectation of the very edges under test (already corrected once: PowerES is not minor). It would favour
    methods that agree with a possibly wrong prior.
  - **Sensitivity: "physics_f1".** τ is set to maximise DEV F1 against `edge_score.PHYSICS_PRIOR`. It is
    reported and never enters a verdict.
  - **Scale note.** τ from 20 DEV episodes is applied unchanged to pooled EVAL (60 episodes). For
    sample-size-dependent scores (|corr|, INT) this is conservative on pooled EVAL and matched on the 20-episode
    folds.
- **P2 set.** The DEV-tuned baselines are shap_gbdt, corr, granger, two_tower, int and qacm. granger_by and
  mscr_rowperm are reported but are not in P2.

## 6. Ground truth (`cdd_oran/decision/gt_p.py`; frozen constants)

- **Knockout.** CRN knockout on the GT seeds. The labelled rate is 1.0 for carrier, sleep and ptx, and 0.3 for
  prot_min (tag 6613). Modes are accept and reject, with reseeds k ∈ {1, 2, 3} and H = 90 s. Every other unit is
  accepted.
- **Unit value.** The mean over k of Σ over the relation's cells of (accept − reject), times sgn ("dir").
- **Per (f, rel, k).**
  - The pooled mean gets an episode-cluster percentile bootstrap 95 % CI with B = 4000 (tag 6617).
  - δ_k = max(5 % × max over (f, rel) of |mean|, floor), with floor pv = 0.5.
  - **TRUE(sign)** if the CI excludes 0 and |mean| ≥ δ.
  - **NULL** if the CI lies within ±δ.
  - **INDET** otherwise, or when there are < 10 units or < 3 episodes. INDET is excluded from scoring and counted.
- **Receiving cell.** `gt_p.receiving_shares` gives the "dir" load gain shares per (seed, pico). The rate at which
  the GT's own top-1 is `cand` is reported as the ceiling of the localisation metric.

## 7. Metrics (`cdd_oran/decision/edge_score.py`)

- **Scope.** Metrics are computed over TRUE ∪ NULL; INDET and unmapped hypotheses are excluded and counted.
- **Counts.** TP = declared ∧ TRUE (any sign). FP = declared ∧ NULL. FN = TRUE ∧ not declared.
- **Rates.**
  - Precision = TP/(TP+FP); NaN if nothing is declared.
  - Recall = TP/(TP+FN).
  - F1 = 2PR/(P+R), or 0 when TP = 0 and the value is defined.
  - Sign accuracy = the share of TP whose sign equals the GT sign.
- **Slices.** Overall, **indirect = relation nbr**, own and far.
- **far_declared.** The number of declared far hypotheses, whatever their GT status; also far_fp.
- **Top-1.** Top-1 receiving-cell accuracy vs cand (MSCR-CRT only; the others give no per-cell output).
- **Splits.** Pooled EVAL and each EVAL fold (MSCR-CRT is re-run per fold with its own split key).

## 8. Criteria and verdict (`scratchpad/e6_dev/e6p_disc_analyze.py`)

| criterion | rule | data |
|---|---|---|
| K0 validity | rejection count n (tested hypotheses with p_mscr ≤ 0.05) satisfies P(Binom(m, 0.05) ≥ n) ≥ 0.01 (m = 60: INVALID at n ≥ 8) AND ≤ 1 BY declaration of 60 | PLACEBO, logged modes as the replicate |
| K1 support | ≥ 60 sleep units AND ≥ 15 of them logged reject | EVAL unit table |
| G premise | GT (dir) sleep → nbr pv is TRUE(+) | GT |
| P1 | indirect recall ≥ 2/3 AND indirect precision ≥ 0.80 AND overall F1 ≥ 0.60 AND sign accuracy ≥ 0.90 AND ≤ 1 far declaration | MSCR-CRT, pooled EVAL |
| P2 | MSCR indirect F1 ≥ the best DEV-tuned baseline's (NaN counts as 0), pooled AND in ≥ 2 of 3 folds | EVAL |

- **K1 extension.** If K1 fails, one extension to 120 EVAL episodes is allowed, then UNDERPOWERED stands. The
  extension seeds must be fixed by amendment before the run (section 11).
- **Verdict precedence.** INVALID (K0 fails) > NO-CHAIN (G fails) > UNDERPOWERED (K1 fails) > PASS (P1 ∧ P2) /
  PARTIAL (P1 only) / KILL (P1 fails).
- **Consequences.**
  - PARTIAL: step 3 must include the SHAP-map arm.
  - KILL: step 2 uses physics only, as a declared privilege.
  - NO-CHAIN: stop.

## 9. Analysis code path and runtime

```
python scratchpad/e6_dev/e6p_disc_analyze.py analyze --dev DEV.jsonl --eval EVAL.jsonl --gt GT.jsonl \
       --placebo PLACEBO.jsonl --prof PROF.jsonl --json OUT.json
```

- **Code path.**
  1. `crt_units.build_unit_data` and `placebo_rejection_units` (K0).
  2. K1.
  3. `run_crt_units` (pooled with audit, then 3 folds).
  4. `baselines_disc.run_baselines` (τ on DEV) and `mscr_rowperm`.
  5. `pacifista_native`.
  6. `gt_p.summarize`, then `edge_score.gt_reference` / `score_method` / `p1_check` / `p2_check` /
     `top1_accuracy`.
  7. The verdict.
- **Dry run.** Only DEV and PLACEBO are present. The report is labelled "PIPELINE DRY RUN, NOT A VERDICT":
  - DEV stands in for EVAL, and baselines are tuned and scored in-sample;
  - pseudo-folds are episode index % 3;
  - the reference is the physics proxy;
  - min_episodes is set to 1.
- **Smoke records.** These need `--allow-smoke` and are never a verdict.
- **Provisional label.** While `FROZEN: no`, a full-mode verdict is labelled PROVISIONAL.
- **Local runtime projection** (measured on smoke records replicated to 60 EVAL, 20 PLACEBO and 20 DEV
  episodes; 16-thread laptop):

  | component | time |
  |---|---|
  | CRT pooled (8370 units, B 9999) | ~5.6 min (34 s per 1000 draws) |
  | CRT 3 folds | ~5 min |
  | CRT placebo | ~1.3 min |
  | lag-1 audit (B 999) | 0.5 min |
  | SHAP-GBDT (DEV + pooled + folds) | ~4 min |
  | QACM (paper ANN settings) | ~3.5 min |
  | two-tower | ~1 min |
  | rowperm | < 0.5 min |
  | GT bootstrap | < 0.5 min |
  | **total** | **≈ 22 min < 1 h** |

## 10. Claims wording

- **Declared.** "Sharp null of no assigned-mode effect rejected; BY across the 60 tested hypotheses
  (mscr-crt-units-v1); no MSCR FDR claim."
- **Not detected and undetermined.** These never mean "no edge".
- **Rowperm.** Its p-values are approximate (serially dependent, fed-back panels).

## 11. Resolutions at freeze (2026-09-29, before any DEV, PLACEBO, EVAL or GT outcome was inspected)
1. **B = 9999** (not 999). At m = 60, B = 999 cannot reach the BY threshold of ≈ 1.8e-4.
2. **P1 indirect recall ≥ 2/3**, exactly.
3. **K0** uses a one-sided binomial count rule at level 0.01: INVALID if P(Binom(m, 0.05) ≥ n_reject) < 0.01
   (m = 60 → n ≥ 8), or if more than 1 BY declaration is made. The plan's rate ≤ 0.08 rule would falsely
   invalidate a valid test 18 % of the time.
4. **Two-tower l1 = 0.5** and the **SHAP absolute score** are declared design choices, set on synthetic data only.
5. **K1 extension seeds 183460-183519** are EVAL folds 3-5. P2 stays on folds 0-2 plus pooled. The driver is
   amended accordingly.
6. **Verdict precedence:** INVALID > NO-CHAIN > UNDERPOWERED > PASS/PARTIAL/KILL.
7. **Localisation** is as defined in section 4.
8. **far** is scored against the GT as is. A far TRUE is reported.
