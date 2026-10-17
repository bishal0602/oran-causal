# Provenance: analysis.posthoc and experiments.posthoc

Source commit for every `CDD-ORAN@0c89106` path below is `0c89106` on `feat/v2`. Items X1 to X16 are the post hoc checks of the journal paper (Appendix C and ESM). Stored outputs live in `data/records/posthoc_*`.

## Analysis

All under `packages/analysis/src/oran_causal/analysis/posthoc/`.

| New path | Source | Note |
|---|---|---|
| `paths.py`, `common.py` | helpers shared by the old scripts (`ci_close`, `js_default`, `sub_filter`, record-path lookup) | Written new; no global state |
| `refs.py` | none | `RefereeTools` protocol. Analysis cannot import methods, so X10, X12 and X16 take the method symbols as injected callables |
| `x1_runtime.py` | `scratchpad/xmethod/exp_c_runtime.py` | `build` produces the paper table; needs `specs/exp_c/timing.json` as spec; `generated_utc` and input labels are pinned |
| `x2x3.py` | `cdd_oran/xmethod/extras.py` (`analyse` half) | Stored spec order d02,d05,d10,d20,b05,b80,x3_told; protocol sha256 pinned from the stored record (it predates the amendments) |
| `x4.py` | `scratchpad/e6_dev/x4_random_defer.py` (analysis half) | `bad_lines` written as 0 (new loader raises on bad JSON) |
| `x5x6.py` | `cdd_oran/xmethod/diag.py` (analysis half) | x5 step 1, x5, x6 and the diag tables; sha256 pinned from the stored record |
| `x7.py` | `scratchpad/e6_dev/x7_subset_check.py` (analysis half) | As X4 |
| `x8.py` | `scratchpad/e6_dev/k0n_tail_check.py` (`summarize`) | Run half is in experiments |
| `x9.py` .. `x15.py` | `scratchpad/xmethod/revision/x9_decision_e2e.py` .. `x15_replication.py` | Read frozen records, no simulation. X9, X10, X15 use `analysis.system` statistics and loaders |
| `x16.py` | `scratchpad/xmethod/x16_centre.py` | `sha256` keys now name the new artifacts; `data.files` relative to the records; `data.disc` argv string dropped |

## Experiments

All under `packages/experiments/src/oran_causal/experiments/posthoc/`.

| New path | Source | Note |
|---|---|---|
| `x1_timing.py` | `scratchpad/xmethod/exp_c_timing.py` | `make_spec`, units, assignment, host, load, worker, run. Lock, pin check, cloud and launch code is superseded by `experiments.launch` and not ported; `session_command` replaces `cloud_cmd` |
| `x4.py` | `scratchpad/e6_dev/x4_random_defer.py` (run half) | Built on `experiments.system` common and study2/study3 |
| `x7.py` | `scratchpad/e6_dev/x7_subset_check.py` (run half) | As X4 |
| `x8.py` | `scratchpad/e6_dev/k0n_tail_check.py` (run half) | Reuses the study1 discovery helpers of `experiments.system` |
| `tools.py` | none | `referee_tools()` wires `RefereeTools` to the method symbols (map decoders, `certify_map`, `can_defer`, `edge_quality`, signatures, `design_slope_bound`, guard KPIs, `LEVEL_V2_ARR`) |
| `extras_specs.py` | `cdd_oran/xmethod/extras.py` (`build_specs`, `build_cost_agg`, `write_specs`) | Declared X2/X3/X5/X6 specs rebuilt from `specs/eval/full.json`; `check` compares them with `specs/extras/*.json` (floats rtol 1e-9; the cost tables' `about` line names the old `scratchpad/xmethod/results/dev` paths) |
| `extras_run.py` | `cdd_oran/xmethod/extras.py` (`dither_constants`, `generate`, `tell`, `_told_run_one`, `_x6_run_one`, `x6_hooks`, `installed`) | The monkeypatching of campaign and runner is gone: the campaign runner takes `Providers.dataset` and `Providers.execute`; only the generator's dither constants and the three pmrt_nl functions of X6 are patched, in a context manager (`patched`). `x6_hooks` is rewritten for the new `GbmProfiles` / `Focal` API of `core.pmrt.nonlinear` (same quantities) |
| `__main__.py` | none | Dispatcher: `x1-runtime x2x3 x4 x5x6 x7 x8 x9 .. x16`, `x1-timing x4-run x7-run x8-run` |

## Not ported

- `extras.py` command line (`specs`, `projection`, cloud commands): specs are checked by `extras_specs check`, runs go through `python -m oran_causal.experiments.campaign run --spec specs/extras/<s>.json` and `experiments.launch`.
- `exp_c_timing.py` lock, pin check, cloud and launch code.

## Deviations

- Stored files are CRLF (Windows text mode); the new writers write LF. Gates compare modulo CR, and as parsed JSON.
- `x4`/`x7` `bad_lines` is a constant 0 (stored value 0).
- X15 `Q3.study3.ols_dR_on_V_AA.slope`: `np.polyfit` differs in the last ulp between numpy 2.4.2 (stored) and 2.5.4. The computation is kept; the test uses rtol 1e-12 for that key.
- X16: floats differ about 1e-14 relative from the stored record (stored on Python 3.12 / numpy 2.4.2). The gate is rtol 1e-9, exact for structure, bools and strings; old-code `sha256`, `env` and `timing_s` keys are excluded.
- X2/X3 provenance checks pass on Linux and fail locally on Windows for 122 of 360 datasets (platform floats in the generator); gate on the VPS.

## Extras run path

An extras spec has an `extras` block. `campaign/extras.py` (new; validation, the seed blocks `XMETHOD_EXTRAS` 3200000-3200199 and `XMETHOD_DIAG` 3300000-3302199, the `integrity.extras` stamp), `campaign/spec.py` (extras seeds, never EVAL), `campaign/integrity.py` (`accept_rule`: DEV records of the same spec sha), `campaign/run.py` (`Providers.execute`, `run_part` keeps the caller's stamp) and `campaign/__main__.py` (`run` of an extras spec calls `run_extras_part`) carry it. Smoke check, 2026-10-17, win32, in-process: trimmed copies of `x2_d05`, `x3_told`, `x5_adj`, `x6_gbm` (2 seeds, n 1000) ran end to end; edges and X6 bias logs equal the stored records at rtol 1e-6; the E4 R3 `dataset_sha256` differs from the stored one (platform floats in the generator, not investigated).

## Fixtures and tests

- `packages/analysis/tests/test_posthoc_units.py`, `test_posthoc_gates.py` (gates on the frozen records; slow: X2/X3 all specs, X5/X6 stages).
- `packages/experiments/tests/test_posthoc_extras.py` (specs equal the frozen files; seed and design guards; `tell`; X2 design generation; X6 hooks; campaign CLI runs of tiny X2 and X3 specs).
- `packages/experiments/tests/test_posthoc_drivers.py`, `test_posthoc_referee_gates.py` (slow: X16).
- `packages/experiments/tests/data/posthoc_jobs_golden.json` (job lists of X4 and X7, from the old code), `posthoc_records_golden_win32.json` (3 X4 and 4 X7 smoke episode records, seed 191100, 20 s, generated with the old code on win32). A Linux golden `posthoc_records_golden_linux.json` was made from the old code on the VPS.
