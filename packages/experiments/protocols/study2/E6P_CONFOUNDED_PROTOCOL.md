# E6-P option (a) protocol: causal maps from confounded incumbent logs, and the WG3 referee they drive

FROZEN: yes (2026-10-01, freeze 1, after the v4 verdict PARTIAL)

> **Naming (2026-09-30):** the discovery method is **PMRT** (Predictable Matched-Filter Randomization Test; docs/benchmark/METHOD_NAMES.md). It is the method labelled "MSCR+" in protocol v4 and in the frozen artifact E6P_MSCRPLUS_V4_FROZEN.json; this study uses its label-only successor E6P_PMRT_V4.json (docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md).

**Status: FROZEN at freeze 1 (2026-10-01); drafted 2026-09-30.** Written after the kill tests K-A, K-A2 and K-B (section 2) and before any seed of the
study blocks below is simulated. The file must be frozen (this line set to "FROZEN: yes", the file's LF-normalised
sha256 recorded in the freeze commit message, in `.tmp/PLAN.md` AND in `scratchpad/e6_dev/e6p_conf.py` as
`FROZEN_SHA256_CONF`) BEFORE any `disc`, `placebo` or `gt` episode is simulated; those stages refuse to run while
`FROZEN_SHA256_CONF` is None or differs from this file. A SECOND freeze (the maps artifact, section 9) happens after
the discovery analysis and BEFORE any `eval` episode is simulated. Only `dev_conf` (section 6) and smoke runs are
allowed before the first freeze.

Sources: `scratchpad/e6_dev/decision/OPTION_A_PLAN.md` (plan + amendment), `.tmp/PLAN.md` (K-A, K-A2, K-B entries),
`scratchpad/e6_dev/decision/opta_ka_summary.json`, `opta_ka2_summary.json`, `opta_kb.json`,
`docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md` (PMRT frozen artifact, v4 criteria), `cdd_oran/decision/mapgate.py`
(MapGateV2), `cdd_oran/decision/collect_p.py` (IncumbentPolicy / PlaceboIncumbent),
`cdd_oran/decision/pmrt.py` + `fdr_layer.py` (PMRT; formerly `crt_units_plus.py` + `mscr_multi.py`), `scratchpad/e6_dev/e6p_step2_dev.py` (R, eligibility).

## 0. Disclosure: what was developed on which data

### 0.1 Discovery method (MSCR / PMRT)

| attempt | method | data | verdict |
|---|---|---|---|
| v1 (`E6P_DISCOVERY_PROTOCOL.md`) | MSCR-CRT v1 | EVAL 60 eps | KILL (indirect recall 3/8; sign bug) |
| v2 (`..._V2.md`) | MSCR-CRT v2 | eval_v2 480 eps | KILL near miss (premise z 2.67 not declared) |
| v3 (`..._V3.md`) | MSCR-CRT v2, n 1200 | eval_v3 1200 eps | PARTIAL (P1 pass, P2 fail); validity only approximate (agent V: sleep null-outcome rejection .143) |
| PMRT development | 29 statistic arms (agent S), 13 declaration layers (agent M), 9 combinations (agent I) | fitted on ev2 + DEV, SELECTED on ev3 slices vs the pooled GT | `loadsp_c` + `wby1s` chosen; selection optimism about .02-.05 F1 |
| v4 (`..._V4.md`, frozen at 4fc2cd9) | frozen PMRT artifact (sha256 `4735a85a...d14ba`) | fresh 188000-189999, pi0 .5/.5 from t = 0 | PARTIAL (2026-10-01, Kaggle e6p-disc-v4an-1): K0, K0n, G, K1, P1, P3, S pass; P2 fails only vs granger_by at n 60 (2/10 wins; 4/5 at n 120), which declares 13 placebo edges vs PMRT 0; pooled indirect F1 .95, sign accuracy 1.00. Not INVALID, so this study proceeds (lead decision 6) |

PMRT enters this study as the v4 frozen artifact, unchanged: nothing is re-fitted on confounded data. K-B (section 2)
used MSCR-CRT v2, not PMRT; this study is the first time PMRT sees incumbent logs.

### 0.2 Referee and study design

- Step-2 DEV (seeds 184200-184239): K-L0 KILL; the debate agents F/G/H concluded that on P3 surge-L40 the winning
  policy is close to a static envelope and that a map-driven referee is unlikely to beat a good static rule.
- `OPTION_A_PLAN.md` was written after step-2 DEV. It found that the picos fall asleep inside the warm-up, so a
  referee has to act from t = 0.
- K-A (DEV 184200-239): MapGate v1 with the knockout-GT map, KILL (R -.07, RLF ratio 1.21).
- **MapGate v2 was designed AFTER K-A, on the same DEV seeds.** Its constants (priority (pv, v, rlf), theta .05,
  k_conf 1, T 60 s, directional reject units) were fixed before any v2 run, but its structure was chosen knowing the
  v1 failure modes on those seeds.
- K-A2 (same DEV seeds) evaluated v2. Its declared rule said STOP (section 2). The study goes ahead on the map-quality
  contrasts only; the STOP reasons are carried into the claims (section 11).
- The K-B kill rule was amended BEFORE any K-B data existed ("attributable" flips); K-B then said CONTINUE.
- **Choices below that were informed by K-B outcomes (disclosed):**
  - both tau variants for every baseline;
  - the n-free transfer of the placebo tau for granger;
  - a placebo of 200 episodes;
  - the explicit magnitude rule for map betas (K-B's MSCR v2 map deferred carrier-on requests because of a magnitude
    and not a sign: section 4.3).
- **Chosen on DEV 184200-239:** never_sleep as the best DEV static (section 5.2).
- **Not tuned on outcomes:** the incumbent table (Plan agent, before K-B).
- Every seed of the study blocks (186080-187159) is unseen.

## 1. Questions and claims

Setting: the E6-P cell P3 surge-L40. An operator has only logs of a CONFOUNDED incumbent WG3 arbiter:

- the incumbent accepts or rejects xApp requests with probabilities that depend on load and protection pressure;
- its per-request propensities are logged;
- load and protection pressure also drive the KPIs.

**Q-disc.** Does PMRT (design-based, it uses the logged propensities) recover a causal map from these logs that is
valid (placebo / null-outcome) and correct on the cells a referee reads? Do associational baselines produce wrong
maps (DEV-calibrated) or empty maps (placebo-calibrated) from the same logs?

**Q-dec.** A single frozen referee rule, MapGateV2, is identical for every map and acts on-policy from t = 0. Is it
better when driven by the PMRT map than when driven by any associational map? And where does it stand against:

- the incumbent;
- accept-all;
- the static rules;
- the GT map (ceiling)?

Primary claims (section 7): **K0** (+ K0n): PMRT validity on confounded logs; **E**: the PMRT-map referee is
eligible; **D1**: R(PMRT map) - R(best associational map) >= .10 with paired LB90 > 0; **D2**: the PMRT map beats
EACH associational map (one-sided, Holm). PASS = K0 & K0n & E & D1 & D2. Everything else is secondary or descriptive,
including the comparison with never_sleep, which is reported in full whatever it shows (section 7.3).

## 2. Evidence from the kill tests (DEV; cited, not re-used as data)

**K-A** (MapGate v1, `opta_ka_summary.json`, 40 DEV seeds, 5.4 CPU-h)

| arm | R | R 90% CI | eligible |
|---|---|---|---|
| GT (v1) | -.07 | [-.51, .28] | no (RLF 1.21) |
| GT_own | -1.93 | [-2.65, -1.24] | no |
| GT_flip | -1.23 | [-1.96, -.69] | no |
| never_sleep | .217 | [.017, .402] | yes |

KILL by rule. Map quality still mattered hugely (dR(GT - GT_own) +1.86).

**K-A2** (MapGate v2, `opta_ka2_summary.json`, same 40 seeds, 3.7 CPU-h; den = V_AA - V_ref = 47.05)

| arm | R | 90% CI | eligible | dR(GT2 - arm) [90% CI] | paired SD per seed of dR |
|---|---|---|---|---|---|
| GT2 (GT map) | .209 | [-.026, .422] | yes (retention 1.29; guards svr .92, nonprot .91, ll .98, rlf .94) | - | - |
| GT2_own | .282 | [-.024, .589] | no (rlf 1.26) | -.073 [-.35, .20] | 1.05 |
| GT2_flip (carrier) | -.821 | [-1.20, -.48] | yes | +1.03 [.60, 1.49] | 1.71 |
| GT2_rand | -1.475 | [-2.25, -.76] | no | +1.68 [.93, 2.51] | 3.03 |
| blanket2 (no map) | .071 | [-.10, .23] | no (retention .86) | +.139 [-.08, .36] | .85 |
| never_sleep | .217 | [.017, .402] | yes | -.008 [-.16, .15] | .61 |
| noarb | 0 | - | yes | +.209 [-.03, .42] | .86 |
| M2 (static G rule) | .102 | [.017, .188] | yes | +.107 [-.12, .32] | .85 |
| B2 (static envelope) | .361 | [.043, .687] | no (rlf 1.27) | -.152 [-.45, .13] | 1.12 |

The declared rule said **STOP**: R(GT2) - R(GT2_own) < .10, and R(GT2) < R(never_sleep). The map-quality contrasts
passed:

- wrong maps are catastrophic (flip, rand);
- the right map is safe (eligible);
- the right map is about as good as the best static rule it implies (GT2 is about equal to never_sleep).

The paired SDs (last column) were computed for this protocol by `.tmp/opta/protocol/paired_sd.py`, output in
`paired_sd.json`. They are the per-seed SD of the linearised pooled-ratio contrast, which agrees with the bootstrap
SE x sqrt(40) to within 10 %.

**K-B** (`opta_kb.json`, seeds 186000-186079, MSCR-CRT v2, H 90 / H_pre 60): CONTINUE on both labels.

Every unit carried its own `probs` row (6499 / 6499 placebo units, 6701 / 6701 applied), with p_mismatch 0.

| map (applied, 40 eps) | edges | TRUE right / wrong sign / NULL / INDET | placebo declarations | wrong-edge ("~clean") flips / ep | GT+placebo-edge flips / ep | flips vs GT / ep |
|---|---|---|---|---|---|---|
| MSCR-CRT v2 | 12 | 11 / 0 / 1 / 0 | 0 (K0 rate 0) | 0 | 0 | 19.75 |
| corr @DEV | 31 | 14 / 8 / 3 / 6 | 36 | 98.1 | 101.6 | 98.4 |
| granger @DEV | 39 | 14 / 6 / 9 / 10 | 37 | 89.7 | 101.6 | 91.1 |
| granger_by | 34 | 12 / 5 / 8 / 9 | 31 | 81.5 | 101.6 | 82.9 |
| two_tower @DEV | 21 | 7 / 5 / 3 / 6 | 20 | 98.1 | 29.1 | 99.5 |
| shap_gbdt @DEV | 20 | 13 / 3 / 2 / 2 | 10 | 70.1 | 2.2 | 81.2 |
| int @DEV | 20 | 4 / 10 / 4 / 2 | 17 | 26.5 | 4.4 | 22.0 |
| qacm @DEV | 10 | 8 / 0 / 1 / 1 | 6 | 2.1 | 70.2 | 14.3 |
| * @placebo tau | 0-16 (granger 0, two_tower 1, corr 3, int 3, qacm 6, shap 16) | mostly TRUE | <= 1 by construction | 0 (qacm 2.1) | 0 | 12.2-15.1 (missing edges) |

**The lesson carried into this design.** DEV-tau baseline maps are WRONG: 26-98 wrong-edge flips per episode for
corr, granger, granger_by, two_tower, shap and int (qacm 2.1). Placebo-tau baseline maps are nearly EMPTY: they flip
12-15 decisions per episode through missing edges. Both variants are therefore arms.

The MSCR v2 map had no wrong-sign edge, yet it flipped 19.75 decisions per episode against the GT map:

| flips / ep | request | GT map says | MSCR v2 map says | cause |
|---|---|---|---|---|
| 13.55 | carrier+ | accept | defer | a MAGNITUDE effect: carrier -> nbr pv declared +12.06 against own pv -5.41 (GT: +0.83 / -6.78) |
| 3.3 | sleep+ | defer | accept | the missing premise sleep -> nbr pv |
| 2.1 | ptx+ | accept | defer | - |

Consequences:

- map magnitudes are part of the method and are fixed in section 4.3;
- PMRT power at the DISC size matters (section 8).

## 3. Plant, incumbent and logging design

**Plant.** Identical to v1-v4 and K-A/K-A2/K-B:

- `e6p_screen.make_cfg("P3", 3, seed, lf)`, lf = e6p_state L40 = 1.75546875;
- 120 s warm-up (UNSCORED for the plant's SLA counters) + 600 s scored = 720 s;
- `E6Env(cfg, log=False, wg3=True, trace=True)`.

Numerics: Kaggle Linux only (numpy 2.4.2, scipy 1.18.1). Local Windows records are never used for any stage after
`dev_conf`.

**Incumbent (logging policy of DISC and dev_conf; also a reference arm).** `collect_p.IncumbentPolicy`, unchanged
from K-B. It reads only the unit's obs-only ctx and never its own past modes. Pressure is
s = max(own_prb_util, nbr_max_prb_util). The modes are accept / reject only, clipped to [.15, .85].

| request class | condition | P(accept) | otherwise |
|---|---|---|---|
| saving (carrier off / sleep / ptx down) | s < .6 | .85 | .15 |
| restore (carrier on / wake / ptx up) | s < .6 | .35 | .85 |
| SG raise | own_prot_below_frac > .05 | .85 | .30 |
| SG lower | own_prot_below_frac = 0 | .85 | .20 |

- Draws: one uniform per unit from `default_rng([seed, 6622, c, x_idx, int(t0)])`.
- Logged per unit: `probs` = {accept: pa, reject: 1 - pa}, plus `inc` = {cls, key, s, below}.
- Record-level `pi0_table` = null.

**Arbiter wrapper during collection.** `units_p.UnitArbiter`:

- T = 60 s;
- open_rule "feasible";
- active from t = 0 (`run_collection(arb_warmup_s=0.0)`);
- a reject unit rejects every request of its (cell, xApp) for T, as in K-B. This is NOT directional, unlike
  MapGateV2: section 13.

The tap is set to `count_all=True`, so knockout rollouts count warm-up seconds. The analysis reads `lab_series`, which
records every second.

**Placebo.** `collect_p.PlaceboIncumbent`: the same draws and logged rows, but accept is applied (a sharp null).
`finalize_units` relabels each unit: mode / p = the incumbent draw, and applied_mode = accept.

**GT.** The incumbent base path plus knockout labels (`gt_p` / `labels_p.Labeller`, the v4 GT settings):

- ks 1-3, H 90, modes accept vs reject, AA continuation, GT_RATE, sampling tag 6613;
- only units with t0 >= 90 are labelled (the tested population, as v4);
- `counts.skipped_early` records the rest.

## 4. Discovery: methods and maps

All methods run on the SAME unit table: the PMRT window rule, H = 90, H_pre = 90 (`disc_bench.load_pool(H=90,
H_pre=90)`), so units with t0 < 90 are not tested. This keeps the frozen artifact's feature distribution and the
exclusion-by-t0 argument of v4.

The option-(a) plan's H_pre = 60 is NOT used: the artifact was never fitted on units with missing [60, 90) pre bins.
The consequence is disclosed: incumbent units that open in [0, 90) are logged and enter later units only through the
`hist` feature and the skeleton; about 2 sleep units per episode (v4 DEV smoke) are untested.

### 4.1 PMRT (primary method): the v4 artifact

- Artifact: `docs/benchmark/artifacts/E6P_PMRT_V4.json`, sha256 `9991c390df3a53ef12b553990da52c6adab25d4087bb029f1427e20a898e6a91`. It is the label-only
  successor of the frozen v4 artifact `E6P_MSCRPLUS_V4_FROZEN.json` (sha256
  `4735a85a1975edc6ddea412972be2f972a4ade9015844f153ae45d82f47d14ba`): the same learned content (checked field by
  field at build time), new labels and the code sha256s of the renamed modules; it records the old artifact in
  `supersedes` (docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md).
- Combination: `loadsp_c` + `wby1s` (q .05, weights / directions from the artifact, n_target = the number of
  episodes of the analysed set: 600 for DISC, 200 for the placebo).
- p-values: B = 9999. RNG and splits as v4: `default_rng([0, 6616, 3, family_idx, split])`; split 0 for pooled DISC,
  9 for the placebo, 0 for the K0n variants (B 999).
- Support rule as v2: >= 30 units, >= 5 accepted, >= 5 rejected, >= 3 episodes.
- Before any use, the analyzer runs `pmrt_artifacts.py verify`, and it refuses to run on an artifact or code
  sha256 mismatch (as v4).

**Validity under the context-dependent incumbent (code audit, 2026-09-30).** The statistic already uses per-unit rows
everywhere, so NO change to pmrt / fdr_layer / pmrt_bench (formerly crt_units_plus / mscr_multi /
mscr_integrate_bench) is required.

- Every design quantity comes from `UnitData.probs[rows]`, the per-unit row:
  - `v_design` / `v_var` (v_u = sgn_u (L(mode_u) - probs_u . L), Var = probs_u . L^2 - (probs_u . L)^2);
  - the re-draw mean `mu = ud.probs[rows] @ LEVEL_V2_ARR` in `pmrt_bench.integrated_family`;
  - the conditional re-draws `crt_units.PiAssignment.mode_draws` (cumulative sums of `probs[rows]`, one independent
    uniform per unit).
- Both UnitData builders prefer the unit's logged `probs` over a record's `pi0_table`:
  - `disc_bench.episode_arrays` (the path PMRT uses);
  - `crt_units.build_unit_data`.
- The `hist` feature uses the same per-unit rows of past units.

Under H0(f, rel, kpi), the row of unit u is F_{u-1}-measurable: the incumbent reads the obs-only ctx at t0 and no past
mode, and draws from an independent keyed uniform. So E[v_u | F_{u-1}] = 0 exactly, and S = sum v_u w_u stays a
martingale with the CRT's predictable variance.

Validity is ASYMPTOTIC (martingale CLT), exactly as v4. The floor .15 bounds |v_u| <= .85 and Var(v_u) >= .1275.

Three items are REQUIRED in the study code (not in the statistic):

- (a) The records must carry `probs` on every unit of the four families. `pi0_table` must be null. `e6p_opta_kb.
  unit_record` already does both.
- (b) The disc_bench cache does NOT record whether a row came from `probs`. A missing row silently falls back to a
  one-hot row (v = 0: the unit is lost, not invalid). So the analyzer must assert, for every disc / placebo / dev_conf
  record, that every unit of FAMILIES has `probs`, and that `UnitData.meta["p_mismatch"] == 0` (K-B: 0).
- (c) The `wby1s` weights project the ev2 z to n_target under the ev2 design (Var(v) about .21). Under the incumbent,
  Var(v) = .1275 (saving / restore) and .16-.21 (SG), so the projection is optimistic. This affects power only,
  because the weights are fixed before the data.

One wording correction, in documentation only: the `crt_units` docstring calls per-unit-row re-draws "the exact
conditional law". That holds under the whole-trajectory sharp null. For a single (f, rel, kpi) null the PMRT
argument is the asymptotic martingale one.

### 4.2 Associational baselines and the two tau variants

`baselines_disc` on the same unit table, unchanged: corr, granger, shap_gbdt, int, qacm, two_tower (tuned), and
granger_by (BY at q .05, untuned: one variant). Each tuned baseline yields TWO maps, each with one variant of tau:

- **@dev (conventional)** — `baselines_disc.tune_tau` far-FPR rule (max_far = 1). The v1-v4 code path, calibrated on
  **dev_conf**: 20 incumbent episodes, the same regime the operator has. K-B calibrated on the unconfounded step-1
  DEV instead; that variant is reported offline only (section 7.4).
- **@plc (placebo-calibrated, generous)** — tau = the (MAX_FP + 1)-th largest finite score on the confounded placebo
  (200 eps) over the declarable relations, MAX_FP = 1 (K-B `placebo_tau`). A placebo is a sharp-null experiment that
  an operator usually does NOT have, so this variant favours the baselines.
  - Scale-free scores (corr |r|, shap mean|SHAP|/sd(y), int ECDF distance, two_tower gate, qacm z-effect) are used
    as they are. Their null noise shrinks with n, so a tau from 200 episodes is conservative at 600.
  - granger's score (-log10 p of F(1, n - 3)) grows with n under a fixed confounding bias. So its placebo tau is
    transferred n-free: r_tau^2 = F_tau / (F_tau + df_plc), and at DISC it declares iff F / (F + df_disc) >
    r_tau^2. The sign is unchanged.

Associational map set A = {corr, granger, shap_gbdt, int, qacm, two_tower} x {@dev, @plc} + {granger_by}: 13 maps.

### 4.3 Map construction (fixed; identical rules for every map of a kind)

A map M: {(f, rel, kpi): beta} over the declared hypotheses with a nonzero sign. beta is the effect of accepting a
+1 knob-direction step (the "dir" orientation). Only kpi in {pv, v, rlf, e} and rel in {own, nbr, far} are read by
MapGateV2.

- **PMRT map.** Declared edges from the primary combination on the pooled DISC data. beta = the design-based slope
  beta_hat = sum_u v_u r_u / sum_u v_u^2 over the family's tested units, where:
  - v_u is the PMRT design-centred treatment with the unit's own row;
  - r_u = the plain target y(rel, kpi) (H 90 post - pre) minus its PREDICTABLE running centre (the PMRT "pred" arm
    residual, `eprocess_units.predictable_residuals`).

  It is design-unbiased for a linear level effect, because E[v_u c_u] = 0 for any predictable c_u. The sign of the
  declared edge is the PMRT sign; if the beta_hat sign disagrees, the edge keeps the PMRT sign with |beta_hat| (and
  the disagreement is counted and reported). The same y scale as the baselines.
- **Baseline maps.** beta = the method's sign x |naive OLS slope of y on x = level * sgn| over the family's units. This
  is K-B's `map_from_declared`, the associational effect size of the method's own world view.
- **GT map (privileged ceiling).** `map_from_gt(gt_conf cells, true_only=True)`: TRUE edges of the fresh gt_conf "dir"
  table (frozen gt_p rule), beta = the edge mean. This is the construction of K-A / K-A2 M_GT.
- **Random map.** `random_sized_map(M_PMRT, gt_conf cells, tag 6623)` with the new key `default_rng([6623, 3])`
  (K-A2 used [6623, 2]). It has |M_PMRT| random keys of the 60-cell universe, random signs, and |beta| = that GT
  cell's |mean|.
- **Decision signature** (for aliasing and reporting) = for each of the 8 (f, d), the tuple
  (harm, energy == "costly", harmful rels, conflict) of `MapGateV2.classify`. These are the only map-dependent inputs
  of `decide` and of the duty bound. Two maps with equal signatures give identical policies, hence identical
  trajectories on the same seed.

## 5. Decision study: the referee and the arms

### 5.1 The referee (frozen rule)

`mapgate.mapgate_v2_arbiter(M)` = `DirectionalUnitArbiter(MapGateV2(M, theta = .05, k_conf = 1))`:

- T = 60 s, open_rule "feasible";
- warmup_s = 0: it acts on-policy from t = 0;
- a fresh instance per episode;
- the rule, constants and code are identical for every map (sha256 of mapgate.py recorded in the maps artifact).

Scoring is unchanged: the plant's warm-up is unscored.

### 5.2 Arms of the EVAL stage (one job = one (seed, arm); every arm on every EVAL seed)

| group | arm | definition |
|---|---|---|
| Gate A anchors | freeze, sub:ES+PowerES, noarb, sub:ES, sub:PowerES, sub:SliceGuarantee | exactly `e6p_step2_dev` (R, retention, guards need them) |
| references | incumbent | UnitArbiter(IncumbentPolicy(seed)) from t = 0, T 60, open_rule feasible: what the operator runs today |
| | never_sleep | K-A static: reject every pico sleep request (prop > cur) from t = 0. It is also the **best DEV static**: the highest-R eligible static on DEV 184200-239 (never_sleep .217 > M2 .102 > M1 .098; B1 / B2 / blanket2 ineligible). The two named references are one arm. |
| | blanket2 | MapGateV2(blanket_saving_map()): the no-map control |
| | B2 (descriptive) | step-2 static envelope (reject ptx-up, sleep, macro carrier-off after the warm-up). Highest raw DEV R (.361) but DEV-ineligible (rlf 1.27). Included because the static envelope is the known competitor. |
| maps | MG:PMRT | MapGateV2(M_PMRT) |
| | MG:GT | MapGateV2(M_GT_conf): privileged ceiling |
| | MG:rand | MapGateV2(random map): chance control |
| | MG:<b> for b in A | 13 associational maps (section 4.2) |

That is 26 named arms. **Aliasing:** arms whose maps share a decision signature are simulated once. An all-accept
signature is simulated as its own arm unless the pre-freeze repro check shows it bit-identical to noarb. The alias
table is part of the maps artifact.

## 6. Data and seeds (block 186000-187999, registered E6 "e6p_confounded_episodes"; tags 6622-6624)

| stage | seeds | n | policy | use |
|---|---|---|---|---|
| K-B (done) | 186000-186079 | 80 | placebo / incumbent | kill test (section 2); never re-used |
| dev_conf | 186080-186099 | 20 | incumbent | before the freeze: plumbing, unit counts, K1 projection, baselines' @dev tau, alias repro check (2 seeds) |
| disc | 186100-186699 | 600 | incumbent | discovery data: every method's map; K0n |
| placebo | 186700-186899 | 200 | PlaceboIncumbent | PMRT K0; baselines' @plc tau; placebo declarations |
| gt | 186900-186939 | 40 | incumbent base + knockouts | fresh GT: discovery scoring and the GT map |
| reserve | 186940-186999 | - | - | unused unless a disclosed amendment |
| eval | 187000-187159 | 160 | every arm of section 5.2 | the decision test; paired by seed |
| reserve | 187160-187999 | - | - | unused |

- Records follow the K-B format (schema "e6p-disc-rec/1") with sub "conf" for dev_conf / disc / placebo / gt, and the
  e6p-optaka2-rec/1 format for eval with sub "conf".
- The registry note in `SEED_REGISTRY.json` still lists the plan's provisional layout (placebo 186700-739, GT
  186740-759, EVAL 187000-079 + 080-159). It must be updated to this table before the freeze.
- RNG tags:
  - 6622: incumbent draws;
  - 6623: random map, key [6623, 3];
  - 6624: this study's bootstraps: DISC analysis `default_rng([6624, 10, ...])`, EVAL `default_rng([6624, 20,
    n_seeds])`;
  - 6616 / 6613 / 6617: CRT / GT sampling / GT bootstrap, as v4.

## 7. Criteria (pre-specified)

### 7.1 Discovery stage (DISC + placebo + gt)

| id | criterion | rule | data |
|---|---|---|---|
| K0 | PMRT placebo validity | v4 rule: P(Binom(m, .05) >= n_reject) >= .01 over the p's wby1s uses AND <= 1 declaration (n_target 200) | placebo |
| K0n | PMRT null-outcome validity | v4 recipe on DISC: 10 groups of 60 consecutive DISC episodes x 4 cyclic series shifts = 40 variants, B 999. Pooled rate of used p <= .05 is <= .075, every family <= .10, and variants with >= 1 declaration <= 5 of 40 | disc (series shifted) |
| K1 | support | >= 500 tested sleep units with >= 75 rejects in DISC (provisional; re-projected from dev_conf before the freeze, see section 13) | disc |
| G | premise in GT | gt sleep -> nbr pv TRUE(+) (frozen gt_p rule, "dir") | gt |
| X1 | premise recovered | PMRT declares sleep -> nbr pv (+) on pooled DISC | disc vs gt |
| X2 | referee-relevant map quality | over the PMRT declarations with kpi in {pv, v, rlf, e}, scored against gt (INDET excluded): precision (TRUE with the declared sign) >= .80 AND sign accuracy >= .90 | disc vs gt |
| X3 | chain | >= min(3, \|C*\|) of C* declared with the GT sign (C = the v2 chain set) | disc vs gt |

Discovery label (reported; not part of PASS):

| precedence | label | condition |
|---|---|---|
| 1 | INVALID | K0 or K0n fails |
| 2 | NO-CHAIN | G fails |
| 3 | UNDERPOWERED | K1 fails |
| 4 | DISC-PASS | X1, X2 and X3 all hold |
| 5 | DISC-PARTIAL | X2 holds, X1 or X3 fails |
| 6 | DISC-FAIL | otherwise |

**INVALID stops the study before EVAL** (no maps artifact, no eval compute). Any other discovery label proceeds to
the maps freeze and EVAL: the decision claim stands on its own, and a wrong or empty map is exactly what EVAL
measures.

### 7.2 Decision stage (EVAL, 160 seeds)

**Definitions.** These are the Gate A definitions of `e6p_step2_dev.arm_stats`, pooled over the EVAL seeds that have
every anchor and the arm:

- V = 3600 sum prot_viol / sum prot_ue_s;
- V_ref = the min pooled V of the three singles, re-minimised per bootstrap resample;
- R = (V_AA - V) / (V_AA - V_ref);
- retention = (Ef - E) / (Ef - EA);
- guard ratio_k = pooled k / pooled noarb k, for k in svr, nonprot_embb_viol, ll_viol, rlf;
- **eligible** = retention >= .90 and every guard ratio <= 1.10 (point estimates);
- R* = R if eligible, else min(R, 0). An ineligible arm cannot count above accept-all, as Gate A's "ineligible -> 0".

Bootstrap: paired over seeds, N_BOOT = 10000, `default_rng([6624, 20, n_seeds])`. Eligibility is held at its point
value inside the bootstrap.

| id | claim | rule |
|---|---|---|
| E | PMRT referee eligible | MG:PMRT eligible (point); retention and guard 90 % CIs reported |
| D1 | beats the best associational map | Delta1 = R*(PMRT) - max_{b in A} R*(b) >= .10 AND LB90 > 0. LB90 = the 5th percentile of R_boot(PMRT) - max_b R*_boot(b), with the max RE-SELECTED in every resample (accounts for picking the best of 13) |
| D2 | beats each associational map | for every b in A: one-sided paired bootstrap p_b = (1 + #{R_boot(PMRT) - R*_boot(b) <= 0}) / (N_BOOT + 1). Holm at alpha .05 over the DISTINCT signatures of A (aliased maps are one hypothesis: their outcomes are identical by construction). Pass iff every one is rejected. A map with the same signature as MG:PMRT has dR = 0 and fails D2 honestly |
| K0 | validity | section 7.1 (K0 and K0n) |

**Verdict precedence**

| precedence | label | condition |
|---|---|---|
| 1 | INVALID | K0 or K0n fails (EVAL is not run) |
| 2 | NOT ELIGIBLE | E fails |
| 3 | PASS | D1 and D2 |
| 4 | PARTIAL | exactly one of D1 and D2 |
| 5 | FAIL | otherwise |

The verdict is labelled "NOT A VERDICT" unless the data are complete (disc 600, placebo 200, gt 40, dev_conf 20,
eval 160 x every distinct arm, no smoke, Kaggle platform fingerprint equal across stages) and every sha256 matches:
protocol, v4 artifact and code, maps artifact, mapgate.py.

### 7.3 Secondary contrasts (reported in full, never criteria)

**Pre-registered expectation, stated so it cannot be re-framed later.** On DEV the GT map was about equal to
never_sleep: -.008 [-.16, .15].

- The study does NOT expect, and will NOT claim, that the PMRT map beats the best static rule.
- The paired contrast R(PMRT) - R(never_sleep) is reported with its 90 % CI and the non-inferiority read (LB90 >
  -.10), whatever it shows. It is reported in the abstract-level summary alongside D1 / D2.

The other contrasts:

- R(PMRT) - R(c) with 90 % CI for c in {noarb, incumbent, blanket2, never_sleep, B2, MG:GT, MG:rand};
- map efficiency R(PMRT) / R(MG:GT);
- every arm's R, retention, guard ratios with CIs, and the eligibility flag;
- per-family / direction defer counts and the decision table of every map.

### 7.4 Descriptive only

- Discovery tables for every method (declared edges; TRUE-right / wrong-sign / NULL / INDET; precision, sign, chain,
  premise; placebo declarations at each tau).
- K-B-style OFFLINE replay on the DISC logged contexts: flips per episode vs the GT map, "~clean" (wrong-edge),
  "GT+plc" (placebo-edge).
- The @dev maps with the step-1 unconfounded DEV tau (K-B's calibration): offline replay only.
- PMRT beta sign disagreements (section 4.3).
- The other 8 PMRT combinations and MSCR-CRT v2 + BY on DISC.
- Two honest competitors that use the logged propensities (plan section 2), discovery only, no EVAL arm: an IPW-Wald
  test and Granger with ctx conditioners. See section 13.

## 8. Power and sample sizes

**DISC n = 600.**

- PMRT premise declaration on the ev3 bench (old design: pi0 .5/.2/.3 after the warm-up, effective Var(v) about .21):
  2/10 at n 60, 5/10 at n 120, 4/4 at n 300.
- Model: mean premise z = z* kappa sqrt(n / 120), with z* the declaration threshold (about 3-3.5 under wby1s).
  kappa = sqrt(Var_inc / Var_old) x sqrt(r), and r = the ratio of tested sleep units per episode.
  - Var_inc = .1275 gives kappa about .78 at r = 1;
  - K-B had 5.3 sleep units / episode at t0 >= 60, against about 4 / episode in v1-v3 after the warm-up; the t0 >= 90
    cut lowers this, so r is about .7-1.

| kappa | P(declare) at n 300 | n 400 | n 600 | n 800 |
|---|---|---|---|---|
| .78 | .76 | .90 | .99 | 1.00 |
| .65 | .53 | .71 | .91 | .98 |
| .55 | .35 | .50 | .75 | .90 |

(z* = 3; z* = 3.5 gives +.01-.03.)

Discount the bench by about 1.3x (the scout's realism factor plus the ev3 selection optimism): 600 is the smallest
round size with P >= .9 at kappa .65. It is also the plan's DISC size. The dev_conf unit counts re-check r before the
freeze.

**EVAL n = 160** (187000-187159, one fixed-n run, no interim look). The relevant paired SDs per seed are those of
section 2, in R units:

- GT-map vs noarb: .86 (the likely D1 / D2 comparator, because placebo-tau maps are near-empty and act almost like
  noarb);
- GT-map vs never_sleep: .61;
- wrong maps: 1.7-3.0 (easy contrasts).

With SD .86, a single contrast gives:

| n | P(dR_hat >= .10 and LB90 > 0), delta .15 | delta .20 | delta .25 | P(z > 2.45) (Holm, 7 remaining), delta .20 |
|---|---|---|---|---|
| 80 | .47 | .67 | .83 | .36 |
| 120 | .60 | .82 | .94 | .54 |
| 160 | .71 | .90 | .98 | .69 |
| 200 | .79 | .95 | .99 | .80 |

The Holm figure is for the least favourable case. Holm over distinct signatures and highly correlated near-empty maps
make D2 closer to the single-contrast figure.

If the PMRT map reproduces the GT map's decisions (DEV: delta about .21 vs noarb), P(PASS) is about .65-.85 at
n = 160. It is about .5 if the PMRT map recovers only part of the GT map (delta .15). This is stated in advance: a
FAIL at delta <= .15 is NOT evidence that confounding does not matter. The wrong-map contrasts (K-B DEV-tau maps) are
powered at > .99.

## 9. Procedure, freezing and artifacts

1. **Pre-freeze (allowed now).**
   - Build the code of section 12.
   - Run dev_conf (20 eps) and smoke. From it:
     - unit counts and K1 projection;
     - baselines' @dev tau (frozen into the maps step);
     - repro checks: IncumbentPolicy arm vs the dev_conf logging trajectory; an all-accept MapGateV2 vs noarb, bit for
       bit, on 186080-186081 (decides the noarb alias).
   - Update SEED_REGISTRY.json (section 6).
   - Commit the v4 working-copy files this study depends on (collect_p.py PI0_V4 etc.).
2. **Freeze 1.** Set this file to FROZEN: yes; sha256 in the commit message, `.tmp/PLAN.md` and
   `e6p_conf.py FROZEN_SHA256_CONF`.
3. **Collection (Kaggle).** disc, placebo and gt, run concurrently. Every stage refuses to run unless frozen and
   registered.
4. **Discovery analysis (Kaggle job).** `e6p_conf_analyze.py disc`:
   - caches, artifact verify, the probs / p_mismatch assertions, PMRT (pooled DISC, placebo, K0n);
   - baselines x tau variants, the GT table, all maps (section 4.3), signatures, the alias table, offline replay;
   - output: `disc_conf.json` + the discovery label. If INVALID: stop and report.
5. **Freeze 2: maps artifact.**
   - `docs/benchmark/artifacts/E6P_CONF_MAPS.json` (+ `.sha256`; `git add -f`, as `.gitignore` ignores artifacts/).
   - Contents:
     - every EVAL arm's map (keys, beta, provenance);
     - `decision_table_v2`, signature and alias target for every map;
     - MapGateV2 constants and the mapgate.py sha256;
     - the sha256 of `disc_conf.json`, of the analyzer and of the v4 artifact.
   - Committed and its sha written into `e6p_conf.py MAPS_SHA256` BEFORE any eval episode. The eval stage refuses to
     run otherwise.
6. **EVAL (Kaggle).** 160 seeds x distinct arms; job order is seed-major and resumable.
7. **Decision analysis.** `e6p_conf_analyze.py eval`: Gate A stats, R*, E, D1 (re-selected max), D2 (Holm over
   distinct signatures), section 7.3 / 7.4, verdict. Output `eval_conf.json`, `verdict_conf.json`.
8. **Report** every criterion and every descriptive item, next to the v4 verdict.

## 10. Cost (Kaggle CPU; E6-P episode about 65-100 CPU-s, GT episode about .6-.7 CPU-h)

| item | n | CPU-h |
|---|---|---|
| dev_conf (collection, 85 s/ep) | 20 | 0.5 |
| disc | 600 | 14.2 |
| placebo | 200 | 4.7 |
| gt | 40 | 24-28 |
| discovery analysis (PMRT B 9999 pooled + placebo, K0n 40 x B 999, baselines at n 600: SHAP / two-tower dominate, about 15x K-B's 493 s) | - | 3-5 |
| eval, 160 seeds; per seed: anchors about 354 s + references about 294 s + 3 map arms + distinct associational maps at about 72 s | 160 | 58 (about 6 distinct associational signatures) to 80 (no aliasing) |
| eval analysis | - | < .2 |
| **total** | | **about 105-133 (central about 115)** |

Wall clock: about 6-8 h on 5 concurrent Kaggle sessions (4 CPUs each), plus the queue.

**Reduced variant** (if the lead caps compute near the plan's ~80 CPU-h): placebo 100, gt 20, eval 120, which gives
about 78-95 CPU-h.

- D1 / D2 power at delta .20 drops to about .82 / .54.
- GT precision on small edges (carrier -> nbr pv) drops.
- That variant must be chosen BEFORE freeze 1.

## 11. Claims wording

**If PASS.**

> "On simulated confounded logs of an incumbent WG3 arbiter with logged propensities (E6-P cell P3 surge-L40), a
> design-based causal discovery method (PMRT, frozen before the study) produced a map that, fed to a fixed map-driven
> referee (MapGateV2, identical for every map), gave R = x [90 % CI] and was eligible (energy and guardrails). It
> exceeded the best associational-map referee by Delta1 [LB90] and each of 13 associational maps (Holm, one-sided
> .05). Associational maps were either wrong (DEV-calibrated) or nearly empty (placebo-calibrated). The knockout-GT
> map gave R = y. The static never-sleep rule gave R = z (difference w [CI]): the map-driven referee does not beat the
> best static rule; its value is that a design-based map avoids the wrong decisions associational maps make from
> confounded logs."

The never_sleep sentence is mandatory in every outcome.

**Scope.**

- One simulated cell, one incumbent design, one referee rule. The rule was developed on DEV seeds (MapGate v1 -> v2
  after K-A) and frozen before this study.
- PMRT REQUIRES the incumbent's per-request propensities to be logged and bounded away from 0 / 1. Without logged
  propensities the method does not apply.
- Validity is asymptotic (martingale CLT), checked by K0 / K0n, not exact.
- The comparison is against associational methods that ignore the propensities. The IPW-type competitors are
  descriptive (section 7.4).

**Never claim:**

- "beats static rules";
- "generalises beyond E6-P";
- "exact";
- anything from K-A / K-A2 / K-B as confirmatory (they are DEV kill tests on other seeds).

**If PARTIAL / FAIL / NOT ELIGIBLE.** Report the failing criterion first, then the D-contrasts with CIs. Use the
section 8 power statement to separate "underpowered at the realised delta" from "no advantage".

## 12. Code (built 2026-09-30, before freeze 1; exact files and commands)

All paths are repo-relative; local commands run from the repo root with `PYTHONPATH=.` and `.venv/Scripts/python.exe`.
Kaggle collection kernels are built by `scratchpad/e6_dev/cloud.py` through `kaggle_run.py` (BARE script names; 4
shards per kernel); analyses run through `kaggle_job.py`. The cloud bundle of every `e6p_conf*` script carries this
doc, `SEED_REGISTRY.json` and `docs/benchmark/artifacts/*` (v4 PMRT artifact, maps artifact), so every guard also
runs there.

1. **Driver `scratchpad/e6_dev/e6p_conf.py`**, with wrappers `e6p_conf_{dev,disc,placebo,gt,eval}.py` (stage
   `dev_conf` / `disc` / `placebo` / `gt` / `eval` baked in).
   - Commands: `run --stage S --part i/k --out F.jsonl [--smoke] [--short S] [--max-labels N] [--arms a,b]
     [--maps FILE]`; `summary --in FILES [--json OUT] [--allow-smoke]` (unit counts, tested sleep units / rejects per
     episode, the K1 projection to 600 episodes, missing rows, the repro checks); `list`.
   - Collection (dev_conf, disc, placebo, gt): `collect_p.run_collection(open_rule="feasible", arb_warmup_s=0.0,
     count_all=True, arbiter_cls=mapgate.DirectionalUnitArbiter)`. This is lead decision 1 (the DIRECTIONAL
     arbiter); the K-B wrapper of section 3 is not used. Policies `IncumbentPolicy` / `PlaceboIncumbent`. Records
     "e6p-disc-rec/1" via `e6p_opta_kb.unit_record` (`probs`, `inc`), `pi0_table` null, sub "conf", record stages
     dev / eval / placebo / gt, plus `conf_stage`, `directional` and `passed`.
   - The gt hook is the v4 hook (Labeller ks 1-3, H 90, AA continuation, GT_RATE, tag 6613, t0 < 90 not labelled)
     with IncumbentPolicy as the base. Its rollouts fork the directional arbiter, so a knockout "reject" defers the
     opening direction only (the same action as MapGateV2's).
   - dev_conf also runs the repro jobs of section 9.1 on 186080-186081:
     - `incumbent` = the EVAL incumbent arm (DirectionalUnitArbiter(IncumbentPolicy) from t = 0); it must reproduce
       the logging trajectory;
     - `MG:allaccept` (MapGateV2({})) and `noarb`.
     In the smoke (seed 18, 120 s scored) both differences were 0 (bit identical).
   - eval arms: the anchors (`e6p_step2_dev`), incumbent, never_sleep (`e6p_opta_ka`), B2, and the map arms blanket2,
     MG:PMRT, MG:GT, MG:rand and MG:<b> x 13. One job per distinct signature (`alias_table`); an all-accept signature
     is aliased to noarb only if the repro check says so. Records "e6p-optaka2-rec/1" + `signature`, `aliases`,
     `conf_stage`.
   - Guards (`guard_run`; smoke runs are exempt):
     - `check_seed(seed, stage)`: the section 6 ranges; the K-B block, the reserves and every other registered block
       are forbidden;
     - the registry: block, tags 6622-6624, and the section 6 ranges in the note;
     - `FROZEN_SHA256_CONF` (+ "FROZEN: yes") for disc / placebo / gt / eval;
     - `MAPS_SHA256` (LF sha256 of `docs/benchmark/artifacts/E6P_CONF_MAPS.json`) for eval;
     - Linux numerics for the frozen stages;
     - the numeric env (`cloud.numeric_env`) in every header.
2. **Analyzer `scratchpad/e6_dev/e6p_conf_analyze.py`**: `disc`, `build`, `verify`, `eval` (the module docstring is
   the spec).
   - `disc`:
     - the v4 artifact is checked by its sha256 and its embedded code sha256s (`e6p_disc_analyze_v4.artifact_status`).
       `pmrt_artifacts.py verify` needs the params pickle, which no bundle carries;
     - every dev_conf / disc / placebo unit of FAMILIES must carry a valid row (accept / reject, sum 1, entries in
       [.15, .85], p = probs[mode]), pi0_table must be null and every pool must have p_mismatch == 0; otherwise the
       analysis stops;
     - PMRT on pooled DISC, placebo K0, K0n (v4 recipe on the 10 DISC groups), K1, G, X1-X3, the discovery label;
     - baselines @dev (dev_conf far-FPR), @plc (placebo tau; granger: partial r^2 transfer, tau_r2 = the
       (MAX_FP + 1)-th largest placebo r^2), granger_by;
     - maps (section 4.3; `design_slope` for PMRT), signatures / aliases, offline replay;
     - descriptive: the other combinations, MSCR-CRT v2 + BY, IPW-Wald, Granger+ctx, the `--dev-step1` maps;
     - output `disc_conf.json`.
   - `build` writes the maps artifact. It refuses INVALID, failed row checks, a non-full analysis and missing arms.
     `verify` re-derives the signatures and aliases, checks every sha256 and prints the `MAPS_SHA256` line.
   - `eval`: one bootstrap index matrix (N 10000, `default_rng([6624, 20, n_seeds])`) for every arm; R*, E, D1 (max
     re-selected per resample), D2 (Holm over the distinct signatures of A), the never_sleep report, the secondary
     contrasts, the verdict. Outputs `eval_conf.json` and `verdict_conf.json`.
3. **`cdd_oran/decision/mapgate.py`** (additive):
   - `random_sized_map(..., key=2)`; the default is K-A2's (checked against the K-A2 header map);
   - `decision_signature`: the section 4.3 tuple with the inputs `decide` never reads dropped, so equal signatures
     <=> identical policies; `signature_key`, `all_accept_signature`;
   - `map_to_json` / `map_from_json`.
   **`cdd_oran/decision/collect_p.py`** (additive): `run_collection(arbiter_cls=None)`.
4. `design_slope` and the n-free granger transfer live in the analyzer. pmrt (formerly crt_units_plus) is untouched
   beyond the label-only rename: the PMRT v4 artifact's code sha256s match.
5. `docs/benchmark/SEED_REGISTRY.json`: the confounded note carries the section 6 layout; the tag notes of 6623 (key 3)
   and 6624 (DISC [6624, 10, ...], EVAL [6624, 20, n_seeds]) are updated.
6. `scratchpad/e6_dev/cloud.py` bundles this doc, the registry and `docs/benchmark/artifacts/*` for `e6p_conf*`.
7. **Tests `tests/test_opta_study.py`** (17):
   - seed layout and guards; freeze and maps-sha refusals;
   - the directional arbiter and its fork; directional incumbent and placebo collection rows;
   - per-unit rows through the disc_bench cache, and a missing row caught;
   - `design_slope` unbiased where the naive slope is confounded; the granger transfer;
   - signature equality <=> identical decisions; the alias table; artifact build refusals and verify;
   - X / K1 / label precedence; R*; D1 re-selection; Holm / D2; verdict precedence; alias expansion.

**Commands.** The NAMEs are examples; each launch needs a new NAME. At most 5 Kaggle sessions run at once.

    # pre-freeze (allowed now)
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-conf-dev-1 e6p_conf_dev.py 1
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py pull e6p-conf-dev-1 1
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/e6_dev/e6p_conf.py summary --in scratchpad/e6_dev/runs/e6p-conf-dev-1/all.jsonl
    # freeze 1 (user): "FROZEN: yes" here, FROZEN_SHA256_CONF in e6p_conf.py, commit; then
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-conf-disc-1 e6p_conf_disc.py 2
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-conf-placebo-1 e6p_conf_placebo.py 1
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-conf-gt-1 e6p_conf_gt.py 2
    # discovery analysis (Kaggle job)
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py launch e6p-conf-discan-1 --cmd "python scratchpad/e6_dev/e6p_conf_analyze.py disc --dev $JOB_SRC/e6p-conf-dev-2-a --disc $JOB_SRC/e6p-conf-disc-1-a,$JOB_SRC/e6p-conf-disc-1-b --placebo $JOB_SRC/e6p-conf-placebo-1-a --gt $JOB_SRC/e6p-conf-gt-1-a,$JOB_SRC/e6p-conf-gt-1-b --dev-step1 scratchpad/e6_dev/runs/e6p-disc-dev-1/all.jsonl --workers 4 --out $JOB_OUT" --paths docs/benchmark/artifacts/E6P_PMRT_V4.json scratchpad/e6_dev/runs/e6p-disc-dev-1/all.jsonl --sources bishalpanta/e6p-conf-dev-2-a,bishalpanta/e6p-conf-disc-1-a,bishalpanta/e6p-conf-disc-1-b,bishalpanta/e6p-conf-placebo-1-a,bishalpanta/e6p-conf-gt-1-a,bishalpanta/e6p-conf-gt-1-b
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py pull e6p-conf-discan-1
    # freeze 2 (local): build + verify, git add -f the artifact and its .sha256, MAPS_SHA256 in e6p_conf.py, commit
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/e6_dev/e6p_conf_analyze.py build --disc-json scratchpad/e6_dev/runs/e6p-conf-discan-1/out/disc_conf.json
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/e6_dev/e6p_conf_analyze.py verify --disc-json scratchpad/e6_dev/runs/e6p-conf-discan-1/out/disc_conf.json
    # EVAL + decision analysis
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-conf-eval-1 e6p_conf_eval.py 3
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py launch e6p-conf-evalan-1 --cmd "python scratchpad/e6_dev/e6p_conf_analyze.py eval --records $JOB_SRC/e6p-conf-eval-1-a,$JOB_SRC/e6p-conf-eval-1-b,$JOB_SRC/e6p-conf-eval-1-c --disc-json scratchpad/e6_dev/runs/e6p-conf-discan-1/out/disc_conf.json --out $JOB_OUT" --paths docs/benchmark/artifacts/E6P_CONF_MAPS.json scratchpad/e6_dev/runs/e6p-conf-discan-1/out/disc_conf.json --sources bishalpanta/e6p-conf-eval-1-a,bishalpanta/e6p-conf-eval-1-b,bishalpanta/e6p-conf-eval-1-c

## 13. Open items to settle before freeze 1

1. **Directional collection.** DISC uses the K-B wrapper (a reject unit also rejects the opposite direction for
   60 s); MapGateV2 is directional. Keep K-B's collection (the proven bite; PMRT artifact semantics), or collect with
   `DirectionalUnitArbiter` so the discovered effect matches the referee's action? The default here is K-B's.
2. **Placebo size.** 200 with n-free granger transfer (default), or matched to DISC (600, +9.4 CPU-h)?
3. **GT labelling of the warm-up.** Label sleep units in [60, 90) too, for the GT MAP only (not for discovery
   scoring)? This costs about +10 % GT CPU. The default is no (v4 rule, t0 >= 90).
4. **IPW-Wald and Granger+ctx.** Should they be EVAL arms outside A (+2 arms, about 6.4 CPU-h)? The default is
   discovery-only descriptive.
5. **K1 thresholds** after the dev_conf unit counts.
6. **Precondition on v4.** The default: the study runs unless the v4 verdict is INVALID (K0 / K0n). A v4 KILL on
   power alone does not block it, because DISC is 5x the v4 120-slice.

## Lead decisions on the drafter's open questions (2026-09-30, before any study data)
1. DISC uses the DIRECTIONAL arbiter (a reject defers only the opening direction), so the discovered effects match
   MapGateV2's actions.
2. Placebo: 200 episodes, with the Granger partial-correlation tau conversion.
3. Full budget (~115 CPU-h, Kaggle).
4. IPW-Wald and Granger+ctx: discovery-only, descriptive (not EVAL arms).
5. GT: no warm-up labels (t0 >= 90 only), identical for every map.
6. A v4 verdict of INVALID blocks this study; a v4 KILL on power alone does not.
7. PASS also requires K0n (the null-outcome check).
The study freezes only after (a) the v4 verdict is known and (b) the driver and analyzer below are built and tested.

## Implementation notes (builder, before freeze 1; these override any conflicting text above)
1. Directional arbiter everywhere (lead decision 1): DISC collection, the incumbent reference arm (so the repro check
   holds), and the GT knockout rollouts. The GT estimand is therefore "defer the opening direction" (differs from the
   gt_ext GT used in K-A/K-A2; reported side by side where both exist). Section 3's non-directional wording is superseded.
2. Artifact verification: the v4 artifact is checked by its sha256 and embedded code sha256s (the params pickle is not
   bundled); the maps artifact by its own sha256 (MAPS_SHA256, freeze 2).
3. Decision signature: only inputs the MapGateV2 rule actually reads (equal signatures <=> identical policies, tested).
4. Granger placebo tau: the second-largest placebo partial r^2 (at most one placebo declaration in r^2).
5. Precision = TRUE edges with the right sign / declared non-INDET edges; sign accuracy over declared GT-TRUE edges.
6. K1 thresholds (500 sleep units / 75 rejects in DISC) are provisional until the dev_conf projection; they are fixed
   at freeze 1.
7. If a baseline fails on Kaggle, its error is recorded and `build` refuses to write the maps artifact (no silent arm
   drops).
8. dev_conf ran as Kaggle kernel `e6p-conf-dev-2-a` (the `e6p-conf-dev-1` code dataset never became ready); the
   analysis commands in section 12 read `e6p-conf-dev-2-a`. Result: 71 s/ep; 6.5 sleep units/ep (3.9 tested, 1.3
   rejects); K1 projection for 600 DISC episodes 2340 tested sleep units / 780 rejects; repro checks bit-exact.
9. K1 thresholds are FIXED at freeze 1 at the provisional values (>= 500 tested sleep units, >= 75 rejects): the
   dev_conf projection clears them about 4.7x / 10x, so no re-tuning is needed.
10. The primary method runs from `E6P_PMRT_V4.json` (the label-only successor of the frozen v4 artifact; equivalence
    in `E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md`).
