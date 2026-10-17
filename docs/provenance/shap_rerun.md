# Provenance: shap_gbdt rerun with real TreeSHAP

The frozen Study 1 and Study 2 `shap_gbdt` results took the sklearn fallback (GBDT + permutation importance): on Kaggle the SHAP script was not importable (`ImportError('Numba needs NumPy 2.0 or less. Got NumPy 2.4.')`). This rerun recomputes the baseline with HistGBDT + `shap.TreeExplainer` on the frozen inputs, rebuilds the referee artifacts on the new maps under new names, and reruns the affected referee EVAL jobs. No frozen file is changed.

## Code

| Path | Source | Note |
|---|---|---|
| `packages/methods/src/oran_causal/methods/system/baselines.py` (`score_shap`, `score_gbdt_perm`, `IMPORTANCE`, `SEEDED`) | the old `shap_gbdt` scorer with its `ImportError` fallback | Split in two: `shap_gbdt` is real TreeSHAP only (an `ImportError` propagates); the old fallback is the separate method `gbdt_perm`. Every score dict and baseline record stores `importance` (`tree_shap` / `permutation`) |
| `packages/experiments/src/oran_causal/experiments/system/shap_rerun.py` | none | Commands `study1`, `study2`, `maps`, `bounds`, `certsafe`, `eval`, `report`. `report` keeps the frozen records of every job whose decision signature is unchanged; a job whose signature a different frozen job already simulates takes that job's frozen records (`frozen_twins`: Study 3 `CS:shap_gbdt@dev` takes `CS:qacm@dev`) |
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
| `data/records/shap_rerun/` | discovery outputs, bounds, EVAL records of the changed jobs, the evaluation reports; manifest `data/manifests/shap_rerun.sha256` |

## Gates

Same-conclusion bar (floats within rtol 1e-6, decisions and declarations equal):

- `gbdt_perm` control equals the frozen records: Study 1 all slices and the placebo; Study 2 maps, placebo maps, alias table and replay (the DEV tau of the permutation path drifts from 0.016347 to 0.016773 on the current stack with the same declarations).
- Recomputed certsafe bound tables equal the frozen tables within rtol.
- Determinism: one EVAL episode (Study 2, seed 187000, `MG:shap_gbdt@dev`) rerun twice on the VPS gives the same record (`checks/determinism.jsonl`).
- Stack control: the frozen `shap_gbdt@dev` arm rerun on the VPS with the frozen artifacts equals the frozen Kaggle records (Study 2 seeds 187000/040/080/120, Study 3 seeds 191100/140/180/220; `checks/stack_s{2,3}.jsonl`).
- Frozen twin: on the VPS, seed 191101 of the new `CS:shap_gbdt@dev` equals the frozen-artifact `CS:qacm@dev` (`checks2/`); the frozen Kaggle `CS:qacm@dev` record of that seed differs by one UE-second (stack drift), so the report uses the frozen records.
- Unaffected arms keep their frozen records (`study2_3_decision`, `study1_raw`, `study2_raw`, `system_artifacts` verify against their manifests); `data/records/shap_rerun/` verifies against `data/manifests/shap_rerun.sha256` (137 files).

## Launches

| Platform | Name | Work |
|---|---|---|
| Kaggle | `ocz-shap-smoke-k1` | Study 2 smoke (`--smoke --short 120`) |
| Kaggle | `ocz-shap-s3-k1` ... `ocz-shap-s3-k4` | Study 3 EVAL parts 0-15 of 20 |
| VPS | `/opt/oran-causal/p1d/shap` | Study 1, Study 2 discovery, bounds, Study 2 EVAL parts 0-2 of 7, checks |
| instabox | `/data/home/oran-causal/p1d/out` | Study 2 EVAL parts 3-6 of 7, Study 3 EVAL parts 16-19 of 20 |
