# Provenance: shap_gbdt rerun with real TreeSHAP

The frozen Study 1 and Study 2 `shap_gbdt` results took the sklearn fallback (GBDT + permutation importance): on Kaggle the SHAP script was not importable (`ImportError('Numba needs NumPy 2.0 or less. Got NumPy 2.4.')`). This rerun recomputes the baseline with HistGBDT + `shap.TreeExplainer` on the frozen inputs, rebuilds the referee artifacts on the new maps under new names, and reruns the affected referee EVAL jobs. No frozen file is changed.

## Code

| Path | Source | Note |
|---|---|---|
| `packages/methods/src/oran_causal/methods/system/baselines.py` (`score_shap`, `score_gbdt_perm`, `IMPORTANCE`, `SEEDED`) | the old `shap_gbdt` scorer with its `ImportError` fallback | Split in two: `shap_gbdt` is real TreeSHAP only (an `ImportError` propagates); the old fallback is the separate method `gbdt_perm`. Every score dict and baseline record stores `importance` (`tree_shap` / `permutation`) |
| `packages/experiments/src/oran_causal/experiments/system/shap_rerun.py` | none | Commands `study1`, `study2`, `maps`, `bounds`, `certsafe`, `eval`, `report`, `x7`. `report` stores workspace-relative paths (`portable`). `report` keeps the frozen records of every job whose decision signature is unchanged; a job whose signature a different frozen job already simulates takes that job's frozen records (`frozen_twins`: Study 3 `CS:shap_gbdt@dev` takes `CS:qacm@dev`) |
| `packages/analysis/src/oran_causal/analysis/system/shap_splice.py` | the record-merge part of `shap_rerun.eval_report` | `ShapSources` is the one table of TreeSHAP record paths; `OVERLAY_FILES` lists the recorded files with `shap_gbdt` entries and `overlaid` maps each to its overlay copy (the one rule for figures and `spec/values.toml`); `frozen_twins`, `changed_jobs`, `merged_records` (frozen EVAL shards of the unchanged jobs + the new jobs, alias fill), `spliced_v4` (the `shap_gbdt` entries of a Study 1 analysis), `splice_fd` (`figures_data.json` with the TreeSHAP entries; every recomputed block is checked against the Study 2 / 3 reports) |
| `packages/experiments/src/oran_causal/experiments/system/shap_overlay.py` | none | `python -m oran_causal.experiments.system.shap_overlay` writes `data/records/shap_rerun/overlay/`: exactly the files of `OVERLAY_FILES` at their record-set paths, recomputed with the TreeSHAP entries (X9, X11-X14 called on the spliced inputs; X7 with the `CS:shap_gbdt@plc` referee) |
| `packages/figures/src/oran_causal/figures/paths.py` (`resolve`) | none | `record()` reads a file of `OVERLAY_FILES` from its overlay copy only (missing copy: `FileNotFoundError`) and every other file from its record set |
| `analysis.posthoc` `x7.analyse_records(refs=)`, `x9.study3_of` / `study2_of` / `assemble`, `x11.run(study_set=)`, `x13.run(inputs=)`, `x14.run(study_set=)`; `experiments.posthoc.x7` `jobs(refs=, arms=)`, `run(maps_doc=, art_doc=)`; `analysis.system.report_data` `map_quality`, `opt_a_eval`, `cs_eval`, `certification` | the load-and-compute functions | Split so the computation runs on given inputs; the defaults read the recorded files as before |
| `shap_rerun x7` (`X7_REF = CS:shap_gbdt@plc`, `X7_REFS`, `X7_ARMS`) | `posthoc x7` | Reruns the X7 static-subset shadow and the `CS:shap_gbdt@plc` referee arms (own stream, protected-share-only and without-it replays) on seeds 191100-191259 with the TreeSHAP certsafe artifact |
| `study2.run(maps_doc=)`, `study3.run(art_doc=, maps_doc=)`, `study3.guard`, `artifacts.build_certsafe(maps_doc=)`, `artifacts.verify_maps(doc=)`, `artifacts.verify_certsafe(maps_doc=, doc=)` | the frozen-path arguments | A pinned `FrozenDoc` replaces the path, so the guard checks the SHAP artifacts against their own pins; the defaults stay the frozen artifacts |

## Inputs

| Input | Where |
|---|---|
| Study 1 caches and `analysis_v4.json` | `data/records/study1_raw/e6p-disc-v4an-pmrt-1/out/` |
| Study 2 discovery records and frozen `disc_conf.json` | `data/records/study2_raw/e6p-conf-{dev-2-a,disc-1-a,disc-1-b,placebo-1-a,discan-2}` |
| Study 2 step-1 DEV records (`--dev-step1` of the frozen run) | `CDD-ORAN` working tree (untracked): `scratchpad/e6_dev/runs/e6p-disc-dev-1/all.jsonl`, sha256 `2cafc6ac2efd373ad67d61802e8c576aa14c02431ebe9ff134ca71f21c3e3c34`, copied to `data/records/shap_rerun/inputs/e6p-disc-dev-1/all.jsonl`. Seeds 183300-183319; not the v4 DEV cache (seeds 188000-188019) |
| Frozen bounds, calibration, EVAL records | `data/records/study2_3_decision/runs/e6p-cs-bounds-1`, `e6p-cs-dev-1`, `e6p-conf-eval-1`, `e6p-cs-eval-1` |

## Outputs

| Path | What |
|---|---|
| `packages/experiments/artifacts/E6P_CONF_MAPS_SHAP.json` (+ `.sha256`) | `E6P_CONF_MAPS.json` rebuilt from `disc_conf_shap_gbdt.json`; key `shap_rerun` names the file it replaces |
| `packages/experiments/artifacts/E6P_CERTSAFE_SHAP.json` (+ `.sha256`) | `E6P_CERTSAFE.json` rebuilt on the SHAP maps |
| `data/records/shap_rerun/overlay/` | the derived files with the TreeSHAP entries at their record-set paths: `system_artifacts/report_2026-10-02/figures_data.json`, `system_artifacts/decision/` (Study 1 analysis, Study 2 discovery, evaluations, verdicts, bounds, map-quality listing), the Study 1 analysis and Study 2 discovery at their run paths, `posthoc_x9_x16/revision/` (X9, X11-X14), `posthoc_x7/x7/x7_tables.json` |
| `data/records/shap_rerun/x7/` | X7 records of the `CS:shap_gbdt@plc` referee and the static-subset shadow (640 jobs) |
| `data/records/shap_rerun/` | discovery outputs, bounds, EVAL records of the changed jobs, the evaluation reports; manifest `data/manifests/shap_rerun.sha256` |

## Gates

Same-conclusion bar (floats within rtol 1e-6, decisions and declarations equal):

- `gbdt_perm` control equals the frozen records: Study 1 all slices and the placebo; Study 2 maps, placebo maps, alias table and replay (the DEV tau of the permutation path drifts from 0.016347 to 0.016773 on the current stack with the same declarations).
- Recomputed certsafe bound tables equal the frozen tables within rtol.
- Determinism: one EVAL episode (Study 2, seed 187000, `MG:shap_gbdt@dev`) rerun twice on the VPS gives the same record (`checks/determinism.jsonl`).
- Stack control: the frozen `shap_gbdt@dev` arm rerun on the VPS with the frozen artifacts equals the frozen Kaggle records (Study 2 seeds 187000/040/080/120, Study 3 seeds 191100/140/180/220; `checks/stack_s{2,3}.jsonl`).
- Frozen twin: on the VPS, seed 191101 of the new `CS:shap_gbdt@dev` equals the frozen-artifact `CS:qacm@dev` (`checks2/`); the frozen Kaggle `CS:qacm@dev` record of that seed differs by one UE-second (stack drift), so the report uses the frozen records.
- X7: 640 jobs (160 seeds x `X7:sub|shadow`, `X7:CS:shap_gbdt@plc|own/pm/other`), no duplicates, one lock (0 pin mismatches) on every Colab run. With the recorded jobs of the kept referees, `x7_tables.json` equals the recorded X7 tables on every Granger, Granger + BY, two-tower and shadow entry (rtol 1e-6); only the SHAP-GBDT entries, the commit and platform lists and the cpu-hour totals change.
- Overlay: `test_system_shap_overlay.py` (`slow`) rewrites the overlay into a temporary directory and compares it with the stored copy; `test_system_shap_splice.py` checks that every `spec/values.toml` record alias follows `overlaid`; every record file the figure build reads that contains `shap_gbdt` is in `OVERLAY_FILES`; the map-quality listing reproduces the frozen `conf_map_quality.txt` lines of every unchanged arm. Every splice check against the Study 2 and Study 3 reports passes (`shap_splice.splice_fd`).
- Unaffected arms keep their frozen records (`study2_3_decision`, `study1_raw`, `study2_raw`, `system_artifacts` verify against their manifests); `data/records/shap_rerun/` verifies against `data/manifests/shap_rerun.sha256` (248 files).

## Launches

| Platform | Name | Work |
|---|---|---|
| Kaggle | `ocz-shap-smoke-k1` | Study 2 smoke (`--smoke --short 120`) |
| Kaggle | `ocz-shap-s3-k1` ... `ocz-shap-s3-k4` | Study 3 EVAL parts 0-15 of 20 |
| VPS | `/opt/oran-causal/p1d/shap` | Study 1, Study 2 discovery, bounds, Study 2 EVAL parts 0-2 of 7, checks |
| instabox | `/data/home/oran-causal/p1d/out` | Study 2 EVAL parts 3-6 of 7, Study 3 EVAL parts 16-19 of 20 |
| Colab | `ocz-x7-smoke-c1` | X7 smoke (`x7 --smoke --short 120`, part 0 of 1) |
| Colab | `ocz-x7-c1`, `ocz-x7-c2`, `ocz-x7-c3` | X7 parts 0-3, 4-7, 8-11 of 40 |
| Colab | `ocz-x7-c5`, `ocz-x7-c6`, `ocz-x7-c7` | X7 parts 12-17, 18-23, 24-29 of 40 |
| Colab | `ocz-x7-c8`, `ocz-x7-c9` | X7 parts 30-34, 35-39 of 40 |
