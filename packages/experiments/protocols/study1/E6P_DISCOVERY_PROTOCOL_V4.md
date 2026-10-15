# E6-P discovery protocol v4: frozen MSCR+ on small fresh slices (step 1, fourth attempt)

FROZEN: yes (2026-09-30)

**Status: DRAFT (agent I, 2026-09-30; collection code, artifact and analyzer implemented by the v4 builder, same
day).** It must be frozen (this line set to "FROZEN: yes", the file's LF-normalised sha256 recorded in the commit message,
in `.tmp/PLAN.md` AND in `scratchpad/e6_dev/e6p_discovery.py` as `FROZEN_SHA256_V4`) BEFORE any eval_v4 / placebo_v4 /
gt_v4 episode is simulated: those stages refuse to run while `FROZEN_SHA256_V4` is None or differs from this file's
sha256, and the analyzer labels its verdict PROVISIONAL unless both hold. The frozen artifact (section 3) is already in
`docs/benchmark/artifacts/` (committed with this file at the freeze). The freeze commit is the commit that sets this file to FROZEN: yes (see `git log -- docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md`).

## 0. Disclosure: attempt history and what was selected on which data

| attempt | method | data | verdict |
|---|---|---|---|
| v1 (`E6P_DISCOVERY_PROTOCOL.md`, f9d87740) | MSCR-CRT v1 | EVAL 60 eps (183320-379), GT 183380-399 | KILL: indirect recall 3/8, sign bug |
| v2 (`..._V2.md`, fea6488c) | MSCR-CRT v2 | eval_v2 480 eps, gt_ext, placebo_ext | KILL (near miss): premise z 2.67 not declared; P2v2 fail |
| v3 (`..._V3.md`, 1107dc99) | MSCR-CRT v2, n = 1200 | eval_v3 1200 eps (182000-183199), GT = gtx-3, K0 = plx-c2 | PARTIAL: P1v2 PASS (premise z 6.1, chain 4/4, ind recall 7/8, 0 placebo decl.); P2v2 FAIL (folds 1/4) |

**Validity finding on v2 (agent V, after v3).** The unit skeleton depends on the same family's past modes (a
rejected request is re-proposed about 60 s later: P(reject | successor) about .99 carrier / 1.00 sleep). The v2 CRT
re-draws modes i.i.d. on the realised skeleton and demeans over the realised unit set (episode x sgn FE), so it is
only approximately valid. On null outcomes (same-seed all-accept twin, time-shifted other episodes' series) v2
rejected SLEEP hypotheses at .143 (nominal .05); other families .06-.07. The placebo cannot detect this (the placebo
applies accept, so no re-proposal follows a logged reject). Claims of "exact" validity for v1-v3 are withdrawn; v3's
PARTIAL stands only as an approximate-validity result.

**What MSCR+ changed and where it was selected.** After v3 the user asked for a data-efficient MSCR. Development
used ev2 (480) + DEV (20) for fitting and the ev3 records (1200) as a bench of disjoint slices (60 x 10, 120 x 10,
300 x 4 folds, 1200 pooled) scored against the POOLED GT (gt-1 + gtx-3, 40 eps):
- agent S (statistic): 29 arms; plain_c / loadsp_c chosen AFTER seeing ev3 (optimism about .02-.05 F1);
- agent M (declaration layer): 13 procedures; wby1s / dagger1s chosen after seeing ev3 (on v2 statistics);
- agent I (integration, this draft): 3 statistics x 3 layers = 9 combinations scored on the same ev3 slices; the
  recommendation rule was written BEFORE that run (`.tmp/mscr_plus/I/PRESPEC.md`).
ev3 and the pooled GT are therefore SELECTION data. v4 uses fresh seeds, a fresh GT and a fresh placebo; nothing
learned is re-fitted on v4 data. Any v4 claim must report v1-v3 and this selection history.

## 1. Question

Does the frozen MSCR+ recover the indirect conflict chain (pico sleep -> nbr load -> nbr protected violations,
PowerES ptx -> nbr load) on SMALL slices of fresh randomized logs (60 and 120 episodes), while staying valid, and does
it match or beat the associational baselines slice by slice?

## 2. Logging design (changed from v1-v3)

- **pi0 = accept .5 / reject .5** (no "half") for ES, PowerES and SliceGuarantee (`collect_p.PI0_V4`). v1-v3 used
  accept .5 / half .2 / reject .3, where half was applied as accept 88-100 % of the time (effective 70/30 split,
  Var(v) .21); .5/.5 gives Var(v) .25 (about +9 % z at fixed n) and removes the mis-dosed mode.
- **Randomized FROM t = 0.** The arbiter's all-accept warm-up is removed for the logging policy
  (`run_collection(arb_warmup_s=0.0)`); the plant's warm-up (120 s unscored for the plant's own SLA counters) is
  unchanged. Reason: the picos fall asleep inside the warm-up (share asleep 0 at 60 s, .83 at 120 s), so v1-v3
  never randomized the sleep decisions that start the chain. Draw key unchanged: `[seed, 6612, c, x_idx, t0]`.
- **Tap:** `count_all=True`, so the tap's cumulative arrays (hence the GT knockout contrasts and the privileged
  per-unit `kpi`) include warm-up seconds. The analysis reads `lab_series`, which always has every second.
- **Early units (H_pre).** The frozen statistic keeps the unit window rule t0 - 90 >= 0 and t0 + 90 <= T
  (`load_pool(H = 90, H_pre = 90)`; its pre bins span up to 150 s, the [90, 150) bin imputed when missing, as fitted).
  Units with t0 < 90 are randomized and logged but NOT tested; they enter later units only as past (the `hist`
  feature, the skeleton). This keeps the sharp-null argument (exclusion by t0, a pre-assignment quantity) and the
  fitted feature distribution. Sleep decisions at t0 in [90, 120) are tested; those in [60, 90) are not.
  A shorter H_pre is NOT used (the frozen adjustment was never fitted on units with missing [0, 90) pre bins).
  **Measured consequence (local DEV smoke, seeds 183300-183301, full episodes, `.tmp/mscr_plus/I/unitcount.json`):**
  randomizing from t = 0 opens +40-50 % units per episode (127 -> 192, 112 -> 162), but almost all new units have
  t0 < 90 (carrier 11-14, ptx 18, prot_min 5-9, sleep 2 per episode) and are untestable; units in [90, 120) are 0-4
  per episode. Testable units rise only +11-25 % (110 -> 137, 102 -> 113), partly because the randomized warm-up changes
  the later trajectory. So under H_pre = 90 v4 does NOT test the warm-up sleep decisions (about 2 per episode, t0 < 90);
  the design gain is mainly the .5/.5 split and the extra post-warm-up units. Two seeds only: re-measure on dev_v4.
- **GT labelling** is restricted to units with t0 >= 90 (the tested population; saves the labelling cost of the
  untestable warm-up units).
- Units in [90, 120) have covariates (t0 / T, ctx, pre-window in the ramp) outside the training range (all training
  units had t0 >= 120). This can cost power; it does not affect validity (the weights stay predictable).

## 3. Method: frozen MSCR+ (`mscr-crt-units-plus-v0` + `mscr-multi-v1`)

**Frozen artifact.** `docs/benchmark/artifacts/E6P_MSCRPLUS_V4_FROZEN.json` (schema `e6p-mscrplus-frozen/1`, 1.36 MB,
LF sha256 **4735a85a1975edc6ddea412972be2f972a4ade9015844f153ae45d82f47d14ba**, also in the `.sha256` file next to it and
as `ARTIFACT_SHA256` in the analyzer). It embeds the sha256 of the 8 code files it depends on (`code_sha256`:
crt_units_plus, mscr_multi, eprocess_units, crt_units, crt_units_v2, disc_bench, mscr_integrate_bench,
mscr_plus_artifacts); the builder re-ran `build` on 2026-09-30 from the same params / prior and got a BYTE-IDENTICAL file
(same sha as agent I's draft in `.tmp/mscr_plus/I/artifact/`), and `verify` passed locally (placebo1 2435 / plxc2 2440 /
dev1 2630 units: statistic bit-identical to the params pickle; layer weights / directions identical to the prior json).
Any edit of one of those 8 files before the freeze changes their sha256: then rebuild, re-verify and replace the sha
here and in the analyzer (the analyzer refuses to run on a sha or code mismatch). Built by
`scratchpad/e6_dev/mscr_plus_artifacts.py build` from
- the params fitted by agent S on ev2 (sub "v2", 480 eps) + DEV (20 eps): Kaggle kernel `bishalpanta/mscrplus-s-2`,
  `out/params.pkl` (sha256 recorded inside the artifact);
- the ev2 prior (the same statistic run on ev2, split 0): Kaggle kernel `bishalpanta/e6p-mscri-2`,
  `out/prior_ev2.json` (sha256 recorded inside the artifact).
It holds EVERY learned piece as plain JSON (no pickle): per hypothesis the ridge adjustment (fill, mx, sx, keep, B),
the kernels of "plain" and "loadsp" (with / without the [90, 150) bin), the slot count, training signs, running-centre
intercepts, the Huber scale model; per family the receiver table rho; the training ctx keys; and for the layer the
ev2 signed z per statistic arm, n_prior 480, q .05, floor .2, cap 5, thr 3 (plus the implied directions and the
weights at n 60 / 120 / 600, for inspection). `verify` checks that the artifact reproduces the pickle's statistic bit
for bit (done on placebo1 / plxc2 / dev1: identical). The code is frozen at the freeze commit (sha256 of the modules in the
artifact's `code_sha256`).

**Primary combination: `loadsp_c` + `wby1s`** (chosen by the pre-specified rule of `.tmp/mscr_plus/I/PRESPEC.md`
from the ev3 bench, Kaggle `e6p-mscri-2`: best eligible score .705; the only other combination within noise, max + wby1s
.702, is later in the simplicity order; section 0). Only this combination enters the criteria. The other 8 combinations are
descriptive.

- Statistic (per family f, hypothesis (f, rel, kpi); `mscr_integrate_bench.integrated_family`): design-centred
  treatment v_u = sgn_u (L(mode_u) - E_pi0 L), L(accept) = 1, L(reject) = 0; predictable weight w_u = Huber-clipped
  running-centred kernel score of the adjusted per-slot post-window target; S = sum_u v_u w_u, z = S / sqrt(sum_u
  Var_pi0(v_u) w_u^2); T = |z| (two-sided) or the oriented z (one-sided, effect direction = sign(z) x the kernel's
  training orientation); max arm = max over plain_c / loadsp_c on the same draws.
- p-values: B = 9999 conditional re-draws of f's modes from the logged pi0 (other families fixed),
  RNG `default_rng([0, 6616, 3, family_idx, split])`, p = (1 + #{T_b >= T_obs}) / (B + 1); split = 0 pooled, 20 + i for
  60-slice i, 40 + i for 120-slice i (fixed here); also fixed by the implementation: 60 + m for the descriptive 300-slice
  m, 9 for placebo_v4 (as the ev3 bench), 0 for every K0n null-outcome variant (B 999).
- Layer (`mscr_multi.declare`, q = .05, n_target = the slice size): `wby1s` (weighted BY, Roeder-Wasserman weights of
  the projected ev2 z, cap 5, floor .2; one-sided in the ev2 direction where |z_ev2| >= 3: 33 of 60 hypotheses) as specified in mscr_multi's docstring;
  weights / directions from the artifact only.
- Support rule as v2 (>= 30 units, >= 5 accepted, >= 5 rejected, >= 3 episodes).

**Validity argument.** Every weight is a function of the past (earlier units' modes and outcomes in information order,
the unit's own pre-window, obs-only ctx at t0, the fixed artifact), so under H0(f, rel, kpi) S is a martingale with
the CRT's predictable variance; the CRT p is asymptotically valid despite the skeleton's dependence on past modes
(martingale CLT), not exact. The layer is valid under arbitrary dependence given valid p's and weights / directions
fixed before the data (the artifact). Checked empirically in v4 by K0 and K0n (section 5).

## 4. Data (fresh seed block 188000-189999, to be registered as E6 dev_reserved "e6p_discovery_v4")

The block is disjoint from 150000-150399, 155000-155399, 160000-179999, 180000-187999 (incl. 186000-187999 being
registered for option (a)) and 190000-190399.

| stage | seeds | n | policy | use |
|---|---|---|---|---|
| dev_v4 | 188000-188019 | 20 | pi0 v4 | plumbing smoke, unit counts, baseline tau (DEV far-FPR rule, as v1-v3); allowed before the freeze |
| eval_v4 | 188100-188699 | 600 | pi0 v4 | the test; j = seed - 188100; slice60 = j // 60 (0-9), slice120 = j // 120 (0-4), pooled = all 600 |
| placebo_v4 | 188700-188739 | 40 | placebo of pi0 v4 (logged, accept applied) | K0 |
| gt_v4 | 188800-188839 | 40 | pi0 v4 base path + knockouts (gt_p, accept vs reject, AA continuation, ks 1-3, H 90) | fresh GT |
| reserve | 188900-189999 | - | - | unused unless a later disclosed amendment |

Collection: Kaggle only (`e6p_discovery.py` stages above), frozen stages refuse to run unless this file is frozen.
Local Windows records are never used for eval / GT / placebo. Records: stage dev / eval / placebo / gt, sub "v4",
`fold` = j // 120 (eval), plus `slice60` = j // 60 and `slice120` = j // 120 (eval, else null), `arb_warmup_s` 0.0,
`count_all` true, `pi0_table` = PI0_V4; gt_v4 `counts.skipped_early` = sampled units with t0 < 90 not labelled. The
analyzer recomputes the slices from the seed (j = seed - 188100), not from the record fields.

## 5. Criteria (all pre-specified; C = the chain set of v2: sleep -> nbr load (+), sleep -> nbr pv (+), sleep -> nbr
v (+), ptx -> nbr load (-); premise = sleep -> nbr pv (+); C* = members TRUE with that sign in gt_v4)

| id | criterion | rule | data |
|---|---|---|---|
| K0 | placebo validity | primary combination: P(Binom(m, .05) >= n_reject) >= .01 on the p's the layer uses AND <= 1 declaration (placebo pooled, n_target 40) | placebo_v4 |
| K0n | null-outcome validity | agent V / S recipe on eval_v4: groups of 60 episodes, 4 cyclic shifts of other episodes' series (40 variants), B 999: pooled rate of used p <= .05 is <= .075, every family <= .10, variants with >= 1 declaration <= 5 of 40 | eval_v4 (series shifted) |
| K1 | support | >= 60 tested sleep units with >= 15 rejects in EVERY 120-slice | eval_v4 |
| G | premise in GT | gt_v4 sleep -> nbr pv TRUE(+) (frozen GT rule, "dir") | gt_v4 |
| P1 | chain / premise at n = 120 | in >= 3 of the 5 slices: premise declared (+) AND >= min(3, |C*|) members of C* declared with the GT sign | eval_v4 slices |
| P2 | small-slice advantage | MSCR+ indirect (nbr) F1 >= the baseline's in >= 6 of 10 60-slices AND >= 3 of 5 120-slices, separately for EACH baseline (corr, granger with DEV-v4 tau; granger_by) | eval_v4 slices vs gt_v4 |
| P3 | pooled | on all 600: P1v2 rule of v2 (premise declared +, >= min(3, |C*|) of C*, overall precision >= .80, sign accuracy >= .90) | eval_v4 |
| S | sign | overall sign accuracy >= .90 pooled AND mean over the 120-slices >= .90 | eval_v4 |

Ties count for MSCR+ in P2 (F1 >= baseline); an undefined F1 (no GT-TRUE nbr edge scorable) is a loss.

**Operational details fixed by the analyzer (`scratchpad/e6_dev/e6p_disc_analyze_v4.py`, written before the freeze).**
- "p's the layer uses" = the p that wby1s actually thresholds per hypothesis (one-sided in the artifact direction where
  |z_ev2| >= 3 and the one-sided p exists, else two-sided), over the hypotheses with a finite p (support rule met).
  K0: m = their number, n_reject = #{p <= .05}, pass iff binom.sf(n_reject - 1, m, .05) >= .01 and <= 1 declaration;
  m = 0 fails.
- K0n: the 10 groups are the 10 eval_v4 60-slices; variant k (k = 1..4) gives episode i of the group the lab_series of
  episode (i + k) mod 60 of the same group (units, modes and skeleton unchanged), statistic B 999, split 0, layer
  n_target 60; rates pooled over all used p of all 40 variants (per family: the p's of that family's hypotheses).
- K1 counts sleep units of the MSCR+ unit table (window rule H 90 / H_pre 90) and those with logged mode reject.
- P1 / P3 hit = declared with the expected (= GT) sign; |C*| = number of C members with GT TRUE and the expected sign.
- P2: an undefined baseline F1 counts as 0 (edge_score.p2_check convention); DEV tau = far-FPR rule on dev_v4 (all 20
  episodes, H 90 / H_pre 90 unit table).
- S: the 120-slice mean is over the slices with a defined sign accuracy (>= 1 true positive); none defined -> S fails.
- Verdict label: "NOT A VERDICT" unless the data are complete (eval 600 = 188100-188699, placebo 40, gt 40, dev 20, all
  sub "v4", no smoke), K0n ran, and the artifact / code sha256 match; "(PROVISIONAL: protocol not frozen)" unless this
  file says "FROZEN: yes" and its sha256 equals `FROZEN_SHA256_V4`.

**Verdict precedence:** INVALID (K0 or K0n fails) > NO-CHAIN (G fails) > UNDERPOWERED (K1 fails) > PASS (P1 and P2 and P3
and S) / PARTIAL (P3 and S, but P1 or P2 fails) / KILL.

**Power statement (disclosed; from the ev3 bench at the OLD design, selection-optimistic by about .02-.05 F1).**
loadsp_c + wby1s on ev3: premise declared AND >= 3 chain hits in 6 of 10 120-slices (4 of 10 60-slices) ->
P(P1) = P(>= 3 of 5 | .6) = .68 (.50 at .5 per slice). P2: F1 >= baseline in 9 / 7 / 7 of 10 60-slices and 9 / 9 / 9 of 10
120-slices (corr / granger / granger_by), i.e. per-slice win rates .7-.9 -> P(>= 6 of 10 | .7) = .85, P(>= 3 of 5 |
.9) = .99. The v4 design (Var(v) +19 %, +11-25 % testable units) should help a little; the frozen weights were projected
for the old design. A KILL on P1 alone at this power is NOT evidence against the chain.

**Descriptive only:** the other 8 MSCR+ combinations; MSCR-CRT v2 + BY; slices of 300 (2); recall over all GT-TRUE nbr
edges; per-hypothesis z / p; baselines' placebo declarations with their DEV tau; tested units with t0 in [90, 120).

## 6. Analysis procedure
1. Build the disc_bench cache of eval_v4 / placebo_v4 / dev_v4 and the GT reference of gt_v4: done INSIDE the analyzer
   with the same functions as `disc_bench_run.py cache` / `gtref` (`disc_bench.build_cache` per JSONL shard, Hs 30 / 60 /
   90 / 150, stage filter; `disc_bench.gt_reference_files` over all gt_v4 episodes, gt_p frozen rule, "dir").
2. `mscr_plus_artifacts.py verify` against the committed artifact (sha must match; bit-identity to the params pickle on
   the small placebo1 / plxc2 / dev1 caches). The analyzer also refuses to run if the artifact sha256 differs from
   `ARTIFACT_SHA256` or any embedded code sha256 differs from the running file.
3. The v4 analyzer `scratchpad/e6_dev/e6p_disc_analyze_v4.py` (reuses `mscr_integrate_bench.run_integrated` /
   `declare_all` with the artifact loaded by `mscr_plus_artifacts.load_artifact`, the slices of section 4, the criteria
   of section 5) runs on Kaggle; outputs `analysis_v4.json` (every criterion, every slice, per-hypothesis z / p, all
   descriptive items) and `verdict_v4.json`.
4. Report every criterion, every slice, and the descriptive items.

**Exact commands** (Windows repo root; kernel slugs follow from the NAMEs; `kaggle_run.py` uses 4 shards per kernel):

    # before the freeze (1 session, ~10 min):
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-disc-v4dev-1 e6p_disc_dev_v4.py 1
    # after the freeze (5 sessions at once: eval 2 kernels ~1.9 h, gt 2 kernels ~3.3 h, placebo 1 kernel ~0.3 h):
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-disc-v4ev-1 e6p_disc_eval_v4.py 2
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-disc-v4gt-1 e6p_disc_gt_v4.py 2
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_run.py launch e6p-disc-v4plc-1 e6p_disc_placebo_v4.py 1
    # analysis (1 session, after all collection kernels finished):
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py launch e6p-disc-v4an-1 --paths docs/benchmark/artifacts --sources bishalpanta/e6p-disc-v4ev-1-a,bishalpanta/e6p-disc-v4ev-1-b,bishalpanta/e6p-disc-v4plc-1-a,bishalpanta/e6p-disc-v4gt-1-a,bishalpanta/e6p-disc-v4gt-1-b,bishalpanta/e6p-disc-v4dev-1-a,bishalpanta/mscrplus-s-2,bishalpanta/e6p-mscri-2,bishalpanta/disc-bench-1 --cmd 'A=docs/benchmark/artifacts/E6P_MSCRPLUS_V4_FROZEN.json; python scratchpad/e6_dev/mscr_plus_artifacts.py verify --artifact $A --params $JOB_SRC/mscrplus-s-2/out/params.pkl --prior $JOB_SRC/e6p-mscri-2/out/prior_ev2.json --small $JOB_SRC/disc-bench-1/out/cache_small && python scratchpad/e6_dev/e6p_disc_analyze_v4.py analyze --artifact $A --eval $JOB_SRC/e6p-disc-v4ev-1-a,$JOB_SRC/e6p-disc-v4ev-1-b --placebo $JOB_SRC/e6p-disc-v4plc-1-a --gt $JOB_SRC/e6p-disc-v4gt-1-a,$JOB_SRC/e6p-disc-v4gt-1-b --dev $JOB_SRC/e6p-disc-v4dev-1-a --out $JOB_OUT --workers 4'
    .venv/Scripts/python.exe scratchpad/e6_dev/kaggle_job.py pull e6p-disc-v4an-1

Analyzer defaults = the protocol values (B 9999, B-null 999, null groups 60 x 4 shifts, slices 60 / 120 / 300; the
descriptive shap_gbdt / int / qacm / two_tower run on the pooled, 300- and 120-slices, errors recorded, never a
criterion input). `--dry-run --slice-sizes a,b,c` (pseudo slices by seed rank, EVAL := DEV) is for plumbing only; the
builder's local dry run on the v1 DEV / placebo-1 / gt-1 records (B 199) ran end to end in 15 s (peak RSS 190 MB).

## 7. Cost
- E6-P episode about 70-100 CPU-s on Kaggle (collection only): eval_v4 600 -> ~14 CPU-h; placebo_v4 40 + dev_v4 20 ->
  ~1.5 CPU-h.
- GT episode about 0.6-0.7 CPU-h -> gt_v4 40 -> ~26 CPU-h, plus the extra labelled warm-up units (GT_RATE = 1 for
  carrier / sleep / ptx) only in [90, 120) (0-4 per episode, section 2; labelling t0 < 90 would add about +50 %
  labelled units, which is why it is excluded). Local DEV smoke: collection CPU per episode unchanged (71-82 s
  current vs 81 s v4, local).
- Analysis (MSCR+ on 16 slices + pooled, B 9999; K0n 40 variants B 999; baselines): < 1 CPU-h; the descriptive
  MSCR-CRT v2 + BY (17 splits, B 9999) and shap_gbdt / two_tower (8 splits) add an unmeasured 1-3 h on one Kaggle session.
- Builder's local smoke (v4 stages, `--smoke --short 120`, 240 s episodes, DEV seeds): 17-18 CPU-s per collection
  episode (73-86 units, 31-43 of them with t0 < 90); gt_v4 with `--max-labels 2`: 2 labels in 113 CPU-s (~57 s per
  label), 40 sampled units skipped as t0 < 90; peak RSS 295 MB.
- Kaggle plan (5 concurrent CPU sessions, 4 shards each): dev_v4 1 kernel before the freeze; after the freeze eval_v4 2
  kernels (75 episodes per shard, ~1.9 h), gt_v4 2 kernels (5 episodes per shard, ~3.3 h), placebo_v4 1 kernel (10 per
  shard, ~0.3 h) = 5 sessions; then the analysis job (1 session). Commands in section 6.
- Total about 42-50 CPU-h, all Kaggle (5 shared CPU sessions).

## 8. Claims wording
- Declared: "Sharp null of no assigned-mode effect rejected (MSCR+ frozen artifact, predictable design-based score,
  asymptotically valid CRT); weighted BY (one-sided where pre-directed) at q .05 across the tested hypotheses."
- Scope: a fourth attempt on fresh seeds after a disclosed development phase that selected the method on ev3; not a
  confirmation of v3.

## 9. Collection-code changes required before dev_v4 (apply after the option-(a) edits of collect_p.py land)

**IMPLEMENTED 2026-09-30 (v4 builder), uncommitted at the time of writing:** items 1-4 below as specified, plus
`docs/benchmark/artifacts/E6P_MSCRPLUS_V4_FROZEN.json` (+ `.sha256`) already in place (item 5; NOTE: `.gitignore` line
36 `artifacts/` ignores this directory, so the freeze commit needs `git add -f docs/benchmark/artifacts/`), `cloud.py` also bundles
`docs/benchmark/artifacts/*` for `e6p_disc*` scripts, the registry entry is `E6.dev_reserved.e6p_discovery_v4_episodes`
(checked free of every E6 / XTRUCE block), v4 stages also require that registry entry (`registry_check(stage)`), the
v4 records carry `slice60` / `slice120`, the gt_v4 hook counts `skipped_early`, and `FROZEN_SHA256_V4 = None` in
`e6p_discovery.py` (set it at the freeze). Tests: `tests/test_disc_v4.py`. The v1-v3 stages, their seed block, their
freeze check (E6P_DISCOVERY_PROTOCOL.md / FROZEN_SHA256) and their records are unchanged.

1. `cdd_oran/decision/collect_p.py`: add `PI0_V4 = {x: {"accept": 0.5, "reject": 0.5} for x in ("ES", "PowerES",
   "SliceGuarantee")}` (+ `PI0["v4"]`, `__all__`). `run_collection(arb_warmup_s=..., count_all=...)` is already in the
   option-(a) working copy; no other collect_p change.
2. `scratchpad/e6_dev/e6p_discovery.py`: stages dev_v4 / eval_v4 / placebo_v4 / gt_v4 in `STAGES` and `jobs()` with the
   seeds and slice fields of section 4 (records stage dev / eval / placebo / gt, sub "v4", fold = j // 120);
   `check_seed` accepts 188000-189999 for these stages only; `make_policy` uses `CP.PI0_V4` when sub == "v4" (placebo:
   `PlaceboPolicy(sd, tables=PI0_V4)`); `episode_job` passes `arb_warmup_s=0.0, count_all=True` for sub "v4" and logs
   `arb_warmup_s`, `count_all` and `pi0_table = PI0_V4` in the record; the GT hook skips units with t0 < 90 for sub
   "v4"; `FROZEN_STAGES` += eval_v4, gt_v4, placebo_v4 with a per-stage protocol check against this file
   (`FROZEN_SHA256_V4`).
3. Wrappers `e6p_disc_{dev,eval,placebo,gt}_v4.py`; `cloud.py` bundles `docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md`
   (line ~183 list).
4. `docs/benchmark/SEED_REGISTRY.json`: register 188000-189999 "e6p_discovery_v4" (E6 dev_reserved, note as above).
5. At the freeze: copy the rebuilt artifact to `docs/benchmark/artifacts/` with its `.sha256`.
No change to crt_units.py / units_p.py / disc_bench.py (a .5/.5 table is handled by the existing pi0_table path).
