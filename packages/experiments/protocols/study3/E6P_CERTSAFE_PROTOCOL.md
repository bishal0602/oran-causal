# E6-P follow-up study 2b: the certified-safe referee

FROZEN: yes (2026-10-01, freeze 1)

Drafted 2026-10-01, after the option (a) verdict NOT ELIGIBLE and the reviewer synthesis
(`scratchpad/e6_dev/decision/next_after_opta/SYNTHESIS.md`, reviewers A, B, C). The file must be frozen ("FROZEN: yes",
its LF-normalised sha256 in the freeze commit message, in `.tmp/PLAN.md` and in `scratchpad/e6_dev/e6p_certsafe.py`
as `FROZEN_SHA256_CS`) BEFORE any `dev` or `eval` episode of section 6 is simulated and before the `bounds` job runs.
A second freeze (the certsafe artifact, section 9) happens before any `eval` episode. Only local and Kaggle smoke runs
(`--smoke`: seed % 31, never a result) are allowed before freeze 1.

This study does NOT replace option (a). The option (a) protocol (`docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md`, frozen)
and its verdict NOT ELIGIBLE stand and are reported first. This is one disclosed follow-up, run once, with its own
protocol hash and fresh seeds. Whatever it shows, the next step is writing (stop rule, section 11).

## 0. Disclosure: why this study exists, and every look so far

### 0.1 The option (a) result

Option (a) EVAL (seeds 187000-187159, 160 seeds, frozen protocol, maps artifact LF sha256 `6c82357b...8ff8940a`):

- MG:PMRT had the highest R of all arms, +.470 [.346, .594], and cut SLA violations to .86x;
- but its RLF guard ratio was 1.31 [1.10, 1.56] > 1.10, so it was ineligible, R* = 0, and D1 / D2 failed by rule;
- verdict NOT ELIGIBLE. MG:GT +.244 (eligible), never_sleep +.221 (eligible).

### 0.2 The failure was foreseeable

On DEV (K-A2, seeds 184200-184239) the GT map with its nbr / far edges removed (GT2_own) was already RLF-ineligible
(1.26). A map that is silent on guard-KPI spillover edges was known to make MapGateV2 unsafe before option (a) EVAL
ran. Option (a) did not act on it.

### 0.3 The mechanism (found after option (a) EVAL; disclosed)

MapGateV2 reads an undeclared edge as a zero edge. The reviewer synthesis attributed the RLF excess to the missing
sleep -> nbr / far rlf edges. The builder of this protocol checked the frozen decision tables and the option (a)
EVAL defer counts (`scratchpad/e6_dev/decision/conf_eval.json`):

| request | MG:PMRT | MG:GT |
|---|---|---|
| sleep+ (fall asleep) | defer | defer |
| ptx+ (raise power) | defer | defer [bounded] (duty bound: released every other unit) |
| ptx units deferred (160 seeds) | 8674 | 2370 (+ 1786 released) |

Both maps defer sleep, so the sleep -> nbr rlf edge cannot explain the difference. The difference is ptx+. The GT
map declares ptx -> nbr rlf = -.061 (raising power lowers neighbour RLF), so deferring ptx+ is a guard conflict and
MapGateV2's duty bound releases it. The PMRT map does not declare that edge (PMRT z -2.2 plain, -.7 loadsp, not
declared), so MapGateV2 deferred every ptx-up request, which raised neighbour RLF. (On step-2 DEV, the static rule B1,
which rejects ptx-up requests after the warm-up, had an RLF ratio of 1.27: 54.7 vs 43.2 RLF per episode.)

General form of the flaw. The guard ratios compare an arm with accept-all (noarb). Accepting is the guard reference.
The referee's only deviation from that reference is a DEFERRAL. A deferral is safe for the guards only if its effect
on every guard KPI is known to be small. MapGateV2 checked this only for declared edges. The rule below checks it for
every guard edge, for every map, without naming any edge.

Knowledge disclosed: the designer knows which edge failed. The rule is stated in terms of the guard definition only,
is applied identically to every arm, and has no tuning constant chosen from outcomes (tau is mechanical, section 3).

### 0.4 Rejected alternatives (forking paths)

- 2a (an RLF-specific matched filter or rare-event statistic in PMRT): tunes the detector toward known misses.
  Rejected (reviewers A, B, C).
- 2d (more discovery data): n would be chosen after seeing z near 1. Rejected.
- Loosening the 1.10 guard: not done. All option (a) definitions stay unchanged (section 7).

### 0.5 Every look at E6-P referee data (to be listed in the paper)

| look | data | what was learned |
|---|---|---|
| step-2 DEV | 184200-184239 | K-L0 KILL; static rules (B1 RLF 1.27) |
| K-A (MapGate v1) | same DEV | KILL; GT_own RLF ineligible |
| K-A2 (MapGate v2) | same DEV | STOP rule; GT2_own RLF 1.26 |
| K-B | 186000-186079 | CONTINUE (map quality) |
| option (a) DISC / placebo / GT | 186100-186939 | DISC-PASS; PMRT 22 edges, 0 wrong sign, 0 placebo |
| option (a) EVAL | 187000-187159 | NOT ELIGIBLE (RLF 1.31) |
| this study, builder projection | dev_conf 186080-186099 (20 eps) + option (a) disc z | section 8 (not binding) |
| this study, DEV | 191000-191039 | tau inputs only |
| this study, EVAL | 191100-191259 | the decision test |

## 1. Question and claims

Can a map-driven referee whose deferrals are gated by the map method's OWN uncertainty be eligible (energy and
guardrails) and still beat (a) the associational-map referees under the same gate and (b) the best static rule?

Primary claims (section 7): **E** (CS:PMRT eligible), **D1**, **D2** (unchanged definitions, applied to the gated
arms), **D3** (CS:PMRT beats never_sleep). PASS = E & D1 & D2 & D3 (and K0, K0n of option (a), already passed).

## 2. Plant, maps, referee (unchanged from option (a))

- Plant: identical to option (a) (`e6p_screen.make_cfg("P3", 3, seed, lf)`, lf = L40, 120 s warm-up unscored + 600 s
  scored). Kaggle Linux numerics (numpy 2.4.2, scipy 1.18.1) for every stage after smoke.
- Maps: the frozen option (a) maps artifact `docs/benchmark/artifacts/E6P_CONF_MAPS.json`, LF sha256
  `6c82357b913d5ee11d501ff1209ec5e2068732e85181dcd27a5216109ff8940a` (`e6p_conf.MAPS_SHA256`). No re-discovery, no
  map is changed. Every driver stage refuses to run if that sha differs.
- Discovery data for the bounds: the option (a) DISC records (186100-186699, 600 episodes), read again by the
  `bounds` job; no new discovery episode.
- Referee: MapGateV2 (`cdd_oran/decision/mapgate.py`, unchanged; sha256 `64ad9d1b...26dd9`, the value pinned in the
  maps artifact) with theta .05, k_conf 1, T 60 s, open_rule "feasible", DirectionalUnitArbiter from t = 0.

## 3. The certified-safe gate (frozen rule; `cdd_oran/decision/certsafe.py`)

Notation. A request has family f in {carrier, sleep, ptx, prot_min} and direction d = sign(prop - cur) in {+1, -1}.
beta(f, rel, k) is the effect of ACCEPTING a +1 knob step on KPI k summed over the relation's cells, over the H = 90 s
unit window (the map's "dir" orientation, option (a) section 4.3). Guard KPIs of the map language: GUARD_KPIS =
(v, rlf) (= MapGateV2's GUARDS_V2: v proxies svr / non-protected eMBB, rlf is rlf). Relations: own, nbr, far.

**Harm direction.** Relative to accepting (the guard reference), deferring a request of direction d changes guard KPI
k by -d * beta(f, rel, k). So the harmful direction of a deferral is h = -d, and its harm is h * beta.

**Classes checked.** Every (f, d) that MapGateV2(M) can defer in some context, i.e. its decision-signature entry is
not "accept". Classes that are never deferred are not checked (accepting is the reference).

**Three-way edge status** for each checked (f, d) and each (rel, k), rel in (own, nbr, far), k in (v, rlf):

| status | condition | consequence |
|---|---|---|
| DECLARED | (f, rel, k) is a key of the map | MapGateV2's own logic (priority, guard conflict, duty bound), unchanged |
| CERTIFIED-SAFE | not declared, the arm's method gives a bound, and UB90(h * beta) <= tau(f, d, k) | none |
| UNRESOLVED | not declared, and no bound or UB90 > tau(f, d, k) | the class is uncertified |

**Gate.** An uncertified (f, d) is never deferred: the referee abstains and accepts, as the guard reference does.
Every other decision is MapGateV2's (`CertSafeMapGateV2(M, uncertified)`; with an empty set it is MapGateV2, tested
bit for bit). The uncertified set is a function of the map, the bounds and tau only, so it is frozen in the
certsafe artifact before EVAL.

**Bounds (UB90).**

- Slope bound (PMRT, associational): UB90 = h * beta_hat + z90 * SE, z90 = 1.2815515655446004 (one-sided 90 %), a
  Wald inversion of the slope's statistic.
- CI bound (GT): UB = max(h * lo, h * hi) over the stored 95 % episode-cluster percentile CI (a one-sided 97.5 % bound,
  more conservative than 90 %).

**tau (derived mechanically from the 1.10 guard).**

    tau(f, d, k) = (RHO - 1) * G_k / (max(N(f, d), 1) * 3),   RHO = 1.05

- G_k = mean per-episode scored guard total under accept-all on the DEV block (k = rlf: the plant's `rlf`; k = v:
  `viol_ue_s`), measured on the arm `CAL:allaccept` (decides exactly as accept-all; bit-identity with noarb is
  checked on 191000-191001).
- N(f, d) = mean number of units per episode of class (f, d) under accept-all on DEV (counted by the gate,
  `dir_<f>_<d>`; all units from t = 0, which is conservative: larger N gives smaller tau).
- 3 = the number of relations (the class budget is split equally over own / nbr / far).

Derivation. If all N(f, d) units of a class are deferred and each deferral raises KPI k by at most tau on each of the
3 relations, the episode total rises by at most N * 3 * tau = .05 G_k, i.e. the guard ratio moves to at most 1.05.
RHO = 1.05 keeps half of the 1.10 guard's .10 margin for estimation error and for the declared edges, which
MapGateV2 handles as before (reviewer B). Linearisation note: unit effects are window effects that overlap in time; the
rule is a first-order budget, not a guarantee; the 1.10 point guard and S1 (section 7) still apply on EVAL.

Builder's provisional values (NOT the frozen ones; G from step-2 DEV noarb 184200-184239: rlf 43.15 / episode, v
20318 / episode; N from the option (a) dev_conf incumbent logs): tau_rlf(ptx, +1) = .055, tau_rlf(sleep, +1) = .189,
tau_rlf(carrier, -1) = .027, tau_rlf(prot_min, -1) = .019 per edge. The frozen values come from the DEV block
(section 6) through `e6p_certsafe_analyze.py calib`.

### 3.1 Bounds per arm (the SAME rule for every arm)

Rule: an undeclared guard edge can be certified only by a one-sided bound from the arm's OWN effect estimator for that
edge. An arm whose method has no effect estimator with a standard error cannot certify, so every undeclared guard edge
of a deferrable class is UNRESOLVED.

| arm | effect estimator (the one that sets the map's beta in option (a) section 4.3) | bound |
|---|---|---|
| CS:PMRT | PMRT design-based slope beta_hat = sum v r / sum v^2 (v = design-centred treatment with the unit's own logged row, r = y - its predictable running centre) | SE = sqrt(sum v^2 (r - beta_hat v)^2) / sum v^2 (martingale / HC0, the same martingale CLT that PMRT's validity rests on) |
| CS:<b>, 13 associational maps | naive OLS slope of y on x = level * sgn (K-B `map_from_declared`) | iid OLS SE |
| CS:GT | knockout GT cell mean (privileged ceiling) | stored 95 % CI |
| CS:rand, CS:blanket2 | none | none: cannot certify |

Notes.

- The PMRT bound uses the plain design slope, not the matched-filter (loadsp) statistic, because only the plain slope
  is in effect units (the filter weights mix cells and times; its statistic has no edge-effect scale). The bounds job
  checks that the slope reproduces every MG:PMRT map |beta| exactly (rel. tol 1e-9) on the DISC unit table, and
  reports beta_hat / SE next to the option (a) `plain_c` and `loadsp_c` z (descriptive).
- The associational bound is the method's own world view. Under confounding the naive slope is biased and its iid SE
  is far too small, so it can certify false safety. That is the comparison (reviewer B's prediction: the gate either
  defers almost nothing or certifies false safety for these maps). The bounds job checks that the naive slope
  reproduces every associational map |beta|.
- All 13 associational arms share one bound table (their betas come from the same estimator); they differ in their
  declared sets.

## 4. Arms of the EVAL stage (one job = one (seed, arm))

| group | arm | definition |
|---|---|---|
| Gate A anchors | freeze, sub:ES+PowerES, noarb, sub:ES, sub:PowerES, sub:SliceGuarantee | exactly `e6p_step2_dev` |
| references | incumbent | as option (a) (descriptive) |
| | never_sleep | as option (a); the D3 comparator (best DEV static) |
| | MG:PMRT | option (a)'s ungated arm on the new seeds (descriptive replication: the paired effect of the gate) |
| gated maps | CS:PMRT, CS:GT, CS:rand, CS:blanket2, CS:<b> for the 13 associational maps | CertSafeMapGateV2(M, uncertified) |

26 named arms. Aliasing as option (a): CS arms with equal certsafe signatures are simulated once (equal signature <=>
identical policy, tested); an all-accept signature is aliased to noarb iff the DEV repro check is bit-identical. B2
(option (a)'s descriptive static envelope) is dropped to save compute; it is not part of any criterion.

## 5. Records

Schema `e6p-optaka2-rec/1` (as option (a) EVAL), sub `certsafe`, `cs_stage` dev / eval; headers carry the protocol
freeze status, the maps artifact status, the certsafe artifact status and sha256, the sha256 of mapgate.py and
certsafe.py, and the numeric platform. Gate counters: `uncert_<f>` (deferrals withheld) and `dir_<f>_<d>` (units by
opening direction) in `policy_counts.policy`.

## 6. Seeds (block 191000-191999, registered E6 "e6p_certsafe_episodes")

| stage | seeds | n | arms | use |
|---|---|---|---|---|
| dev | 191000-191039 | 40 | CAL:allaccept on every seed; noarb on 191000-191001 | G_k, N(f, d), noarb alias. Nothing else is computed on DEV |
| reserve | 191040-191099 | - | - | unused unless a disclosed amendment |
| eval | 191100-191259 | 160 | every arm of section 4 | the decision test, paired by seed |
| reserve | 191260-191999 | - | - | unused |

The block was checked free of every registered block of every world in `SEED_REGISTRY.json` (incl. the option (a)
block 186000-187999, the v4 block 188000-189999, the confirmation reserves 155200-155399 and XTRUCE 190200-190399)
and of the E6 TEST range (>= 960000), and of every literal seed in scripts/, cdd_oran/, scratchpad/, tests/, on
2026-10-01. The driver asserts every seed against this layout and forbids every other block. No new RNG tag: the EVAL
bootstrap reuses `default_rng([6624, 20, n_seeds])` of option (a) by design (definitions unchanged).

## 7. Criteria (pre-specified)

Definitions are those of option (a) section 7.2 and `e6p_step2_dev`, unchanged: V, V_ref (re-minimised per resample),
R, retention, guard ratios for svr, nonprot_embb_viol, ll_viol, rlf; eligible = retention >= .90 and every guard ratio
<= 1.10 (point); R* = R if eligible else min(R, 0); paired bootstrap over seeds, N_BOOT 10000,
`default_rng([6624, 20, n_seeds])`, eligibility held at its point value.

| id | claim | rule |
|---|---|---|
| E | gated PMRT referee eligible | CS:PMRT eligible (point) |
| D1 | beats the best gated associational map | R*(CS:PMRT) - max_b R*(CS:b) >= .10 AND LB90 > 0, max re-selected per resample (option (a) D1) |
| D2 | beats each gated associational map | one-sided paired bootstrap p per distinct certsafe signature of the 13 CS:<b>, Holm alpha .05, all rejected (option (a) D2) |
| D3 | beats the best static rule | R*(CS:PMRT) - R*(never_sleep) > 0 AND paired LB90 (5th percentile) > 0 |
| S1 | RLF upper CI (secondary, reviewer B) | the 95th bootstrap percentile of CS:PMRT's RLF ratio <= 1.20 |
| K0, K0n | validity | carried from option (a) (both passed); INVALID if either is false in `disc_conf.json` |

Verdict precedence: INVALID > NOT ELIGIBLE (E fails) > PASS (D1, D2 and D3) > PARTIAL (one or two of D1, D2, D3) >
FAIL (none). S1 does not change the label; a PASS or PARTIAL with S1 failing is reported as "<label> (S1 failed)" and
no safety claim may be made without stating it. The label is "NOT A VERDICT" unless: 160 EVAL seeds with every arm, no
smoke records, the protocol frozen, the certsafe artifact verified and equal to `CERTSAFE_SHA256`, one platform
fingerprint across the EVAL shards, and the certsafe.py sha256 of every EVAL header equal to the artifact's.

Secondary contrasts (reported in full, never criteria): R(CS:PMRT) - R(c) with 90 % CI for c in {noarb, incumbent,
never_sleep, MG:PMRT, CS:GT, CS:rand, CS:blanket2}; every arm's R, retention, guard ratios with CIs and eligibility;
uncertified classes and withheld deferrals per arm; the MG:PMRT replication of option (a)'s RLF ratio.

## 8. Expected outcome and power (stated in advance)

Builder's projection (not binding; computed before freeze 1 from the 20 dev_conf episodes, SE scaled by
sqrt(2773 / 79499) to the DISC unit count, beta_hat approximated by the option (a) `plain_c` z times that SE, tau as in
section 3):

| CS:PMRT class | undeclared guard edge that decides | projected UB90 vs tau | status |
|---|---|---|---|
| sleep+ | sleep nbr rlf, far rlf, far v | .095 vs .189, .044 vs .189, 14 vs 89 | certified: still deferred |
| ptx+ | ptx nbr rlf | .163 vs .055 | uncertified: never deferred (the option (a) failure edge) |
| carrier- | carrier nbr rlf | .080 vs .027 | uncertified |
| prot_min- | prot_min far rlf | .070 vs .019 | uncertified |

So CS:PMRT is expected to act close to never_sleep (it defers sleep only, through 60 s directional units instead of
per request). Pre-registered expectations:

- E likely holds (never_sleep was eligible in option (a), RLF .99).
- D3 is unlikely: the expected R difference to never_sleep is near 0, and the paired SD per seed of GT-map-type arms
  vs never_sleep was .61 (option (a) section 2), so at n 160 (SE about .048) LB90 > 0 has 80 % power only for a true difference of about .12. If CS:PMRT
  only ties never_sleep, that is reported as is: the map then adds nothing over the static rule it implies.
- D1 / D2 depend on whether the OLS bounds certify the associational maps' deferrals. With certification, those maps
  keep their wrong deferrals (option (a): R -9.8 to -11.5); without it they collapse toward noarb (R about 0). In both
  cases a CS:PMRT near never_sleep (+.22 in option (a)) clears the .10 margin if the associational arms sit at or below
  0; the option (a) D1 / D2 power table (section 8 there) applies.

## 9. Procedure, freezing and artifacts

1. Pre-freeze (allowed now): code (section 12), tests, local smoke, optional Kaggle smoke (`--smoke`).
2. **Freeze 1.** This file "FROZEN: yes"; LF sha256 in the commit message, `.tmp/PLAN.md` and
   `e6p_certsafe.py FROZEN_SHA256_CS`; commit.
3. **DEV (Kaggle kernel) and bounds (Kaggle job), concurrently.** Both refuse to run unless frozen (bounds: it refuses
   a non-full result unless `--allow-partial`, which is smoke only).
4. **calib** (local, light) on the pulled DEV records -> `certsafe_calib.json`.
5. **Freeze 2.** `build` writes `docs/benchmark/artifacts/E6P_CERTSAFE.json` (+ `.sha256`; `git add -f`): tau, the
   bound tables, per arm the source map, bound method, edge statuses, uncertified classes, certsafe signature,
   decision table and alias target, the jobs, the sha256 of the bounds / calib JSONs, of disc_conf.json, of the maps
   artifact, of mapgate.py, certsafe.py, the driver and the analyzer. `verify` re-derives everything and prints the
   `CERTSAFE_SHA256` line; commit it into the driver. `build` refuses partial inputs, a beta-reproduction failure, a
   changed maps artifact and a disc_conf.json other than the one the maps artifact pins.
6. **EVAL (Kaggle).** 160 seeds x distinct arms, seed-major, resumable.
7. **Decision analysis** (`e6p_certsafe_analyze.py eval`, Kaggle job): outputs `eval_certsafe.json`,
   `verdict_certsafe.json`.
8. **Report** option (a) first, then this study, every criterion and every look (section 0.5).

## 10. Cost (Kaggle CPU; option (a) EVAL measured 60-65 CPU-s per arm-episode)

| item | n | CPU-h |
|---|---|---|
| DEV: 40 x CAL:allaccept + 2 x noarb | 42 eps | 0.8 |
| bounds job (DISC caches for 600 eps + slopes) | - | 0.3 |
| calib, build, verify (local) | - | < 0.05 |
| EVAL: per seed 6 anchors + 3 references + J distinct CS jobs (J <= 17 after aliasing) at about 64 s | 160 seeds | (9 + J) x 2.84: 43 (J 6), 51 (J 9), 74 (J 17) |
| EVAL analysis (Kaggle job) | - | < 0.2 |
| **total** | | **about 45-75, central about 50** |

Wall clock: about 2.5-4 h on 5 concurrent EVAL kernels (4 CPUs each), plus the queue; DEV and bounds about 30 min.
Reduced variant (to be chosen before freeze 1): drop the descriptive references incumbent and MG:PMRT (-5.7 CPU-h).
Reviewer C's estimate (about 20 CPU-h) assumed few gated arms; every CS:<b> arm on 160 seeds costs 2.8 CPU-h.

## 11. Claims wording and stop rule

Stop rule: this is the last E6-P referee study. If it is NOT ELIGIBLE, FAIL or PARTIAL, there is no further rescue;
the paper reports option (a) and this study as they are.

- **If PASS:** "Gated by its own design-based uncertainty (a deferral is allowed only when every guard edge is
  declared or bounded below a tolerance derived from the guard), the PMRT-map referee was eligible, beat every
  associational-map referee under the same gate (D1, D2) and beat the best static rule (D3: x [LB90])." Always add:
  this rule was designed after option (a) failed, and option (a) itself was NOT ELIGIBLE.
- **If E holds and D3 fails:** "Under the gate the PMRT referee is safe but reduces to the static never-sleep rule."
  Report D1 / D2 as measured.
- **Never claim:** "safe superiority" from option (a); "beats static rules" without D3; anything from the builder's
  projection (section 8) or DEV as a result; generalisation beyond E6-P.
- Scope as option (a) section 11 (one simulated cell, one incumbent, logged propensities required, asymptotic
  validity).

## 12. Code (built 2026-10-01, before freeze 1)

All paths repo-relative; local commands from the repo root with `PYTHONPATH=.` and `.venv/Scripts/python.exe`.

1. `cdd_oran/decision/certsafe.py` (new; mapgate.py untouched): `tau_table`, `bound_ub`, `classify_edge`,
   `can_defer`, `certify`, `CertSafeMapGateV2` (MapGateV2 subclass; empty uncertified set = MapGateV2),
   `certsafe_arbiter`, `certsafe_signature`, the estimators `design_slope_se` / `naive_slope_se`, `gt_ci_bounds`,
   JSON helpers.
2. Driver `scratchpad/e6_dev/e6p_certsafe.py` (+ wrappers `e6p_certsafe_dev.py`, `e6p_certsafe_eval.py`): stages dev /
   eval, guards (seed layout and forbidden blocks; `FROZEN_SHA256_CS` for both stages; option (a) `MAPS_SHA256`;
   `CERTSAFE_SHA256` for eval; registry block / tag / layout; Linux numerics; smoke exempt), aliasing over certsafe
   signatures, `smoke_artifact` for plumbing.
3. Analyzer `scratchpad/e6_dev/e6p_certsafe_analyze.py`: `bounds`, `calib`, `build`, `verify`, `eval` (reuses
   `e6p_conf_analyze.arm_stats_multi`, `d1_check`, `d2_check`, `holm` unchanged).
4. `scratchpad/e6_dev/cloud.py`: bundles this doc for `e6p_certsafe*` kernels (additive).
5. `docs/benchmark/SEED_REGISTRY.json`: block 191000-191999 and its note; the 6624 note records the reuse.
6. Tests `tests/test_certsafe_study.py`: tau derivation, bounds, three-way classification, certification on the frozen
   maps, default MapGateV2 bit-identity (mapgate.py sha256 equals the maps artifact's pin; identical stateful
   decisions on every frozen map), gate decisions and signatures, estimators (design slope = the option (a) map beta;
   SE calibrated; naive slope confounded), seed layout vs every registered block, freeze guards, driver-sha pin, calib
   and artifact round trip and refusals, D3 / S1 / verdict precedence.

**Commands.** NAMEs are examples; each launch needs a new NAME; at most 5 Kaggle sessions at once. Commands whose
`--cmd` contains `$JOB_SRC` / `$JOB_OUT` use single quotes so the local shell does not expand them.

    # pre-freeze smoke (local)
    PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_certsafe_study.py -q
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/e6_dev/e6p_certsafe.py run --stage dev --smoke --short 60 --part 0/1 --out .tmp/cs_dev_smoke.jsonl
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/e6_dev/e6p_certsafe.py run --stage eval --smoke --short 60 --part 0/1 --arms CS:GT,MG:PMRT --out .tmp/cs_eval_smoke.jsonl
    # freeze 1 (user): "FROZEN: yes" here, FROZEN_SHA256_CS in e6p_certsafe.py, commit; then DEV + bounds
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-cs-dev-1 e6p_certsafe_dev.py 1
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py launch e6p-cs-bounds-1 --cmd 'python scratchpad/e6_dev/e6p_certsafe_analyze.py bounds --disc $JOB_SRC/e6p-conf-disc-1-a,$JOB_SRC/e6p-conf-disc-1-b --disc-json scratchpad/e6_dev/decision/conf_disc_conf.json --workers 4 --out $JOB_OUT' --paths docs/benchmark/artifacts/E6P_CONF_MAPS.json scratchpad/e6_dev/decision/conf_disc_conf.json --sources bishalpanta/e6p-conf-disc-1-a,bishalpanta/e6p-conf-disc-1-b
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py pull e6p-cs-dev-1 1
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py pull e6p-cs-bounds-1
    # calib + freeze 2 (local): build, verify, git add -f the artifact and its .sha256, CERTSAFE_SHA256 in e6p_certsafe.py, commit
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/e6_dev/e6p_certsafe_analyze.py calib --records scratchpad/e6_dev/runs/e6p-cs-dev-1 --out scratchpad/e6_dev/runs/e6p-cs-dev-1
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/e6_dev/e6p_certsafe_analyze.py build --bounds scratchpad/e6_dev/runs/e6p-cs-bounds-1/out/certsafe_bounds.json --calib scratchpad/e6_dev/runs/e6p-cs-dev-1/certsafe_calib.json --disc-json scratchpad/e6_dev/decision/conf_disc_conf.json
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/e6_dev/e6p_certsafe_analyze.py verify
    # EVAL + decision analysis
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-cs-eval-1 e6p_certsafe_eval.py 5
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py launch e6p-cs-evalan-1 --cmd 'python scratchpad/e6_dev/e6p_certsafe_analyze.py eval --records $JOB_SRC/e6p-cs-eval-1-a,$JOB_SRC/e6p-cs-eval-1-b,$JOB_SRC/e6p-cs-eval-1-c,$JOB_SRC/e6p-cs-eval-1-d,$JOB_SRC/e6p-cs-eval-1-e --disc-json scratchpad/e6_dev/decision/conf_disc_conf.json --out $JOB_OUT' --paths docs/benchmark/artifacts/E6P_CONF_MAPS.json docs/benchmark/artifacts/E6P_CERTSAFE.json scratchpad/e6_dev/decision/conf_disc_conf.json --sources bishalpanta/e6p-cs-eval-1-a,bishalpanta/e6p-cs-eval-1-b,bishalpanta/e6p-cs-eval-1-c,bishalpanta/e6p-cs-eval-1-d,bishalpanta/e6p-cs-eval-1-e
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py pull e6p-cs-evalan-1

## 13. Open items for the lead (settle before freeze 1)

1. **Deferral direction.** The synthesis wrote "UNRESOLVED (action deferred)", i.e. it gates ACCEPTS. Because the
   guards are ratios to accept-all, an accept cannot move them; only deferrals can, and the option (a) excess came from
   deferring ptx+ (section 0.3). This protocol gates deferrals. Gating accepts as well would defer more and could not
   fix the observed failure.
2. **Guard KPIs.** (v, rlf), MapGateV2's guard proxies. RLF only would change nothing in the projection (every
   uncertified class is already uncertified on an rlf edge).
3. **Budget split.** The class budget .05 is split over the 3 relations but not over the classes; two certified classes
   could in the worst case use .05 each. Alternative: split over the deferrable classes too (much smaller tau; the
   projection then also leaves sleep+ uncertified, i.e. CS:PMRT is about accept-all).
4. **Expected tie with never_sleep** (section 8). Is a study whose likely result is "eligible, ties never_sleep" worth
   about 50 CPU-h, or is the reduced variant preferred?

## 14. Lead decisions (2026-10-01, before freeze 1; user chose the full study)
1. Deferral direction: the gate certifies DEFERRALS; an uncertified class is accepted (item 13.1 as written).
2. Guard KPIs: (v, rlf), as built.
3. Budget split: per relation only (.05 per class). Worst case two certified classes use .05 each, which stays within
   the unchanged 1.10 guard; the class-split alternative leaves nothing certifiable and would make the study moot.
4. Full variant (all arms, about 45-75 CPU-h); EVAL split over 5 Kaggle kernels (a-e) instead of 3 (parallelism only;
   the job list is seed-major and identical).
5. Stop rule (section 11) confirmed: after this study, write up whatever it shows; no further referee iterations.
