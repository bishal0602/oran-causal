# Provenance: methods.controlled, methods.registry

Source commit for every `CDD-ORAN@0c89106` path below is `0c89106` on `feat/v2`. The method code at that commit is identical to the code at `5a95186`, which produced the Study 4 EVAL records (`git diff --stat 5a95186 0c89106` touches only `cdd_oran/xmethod/diag.py` and `extras.py`). Behaviour is unchanged unless the note says otherwise.

## Modules

| New path | Source | Note |
|---|---|---|
| `packages/methods/src/oran_causal/methods/registry.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/runner.py` (`REGISTRY`, `load_method`); `cdd_oran/xmethod/methods/classic.py` (`_Lazy`) | One lazy `ARMS` mapping and `resolve`. `LEGACY` maps every `cdd_oran.xmethod.methods.<name>:<Cls>` ref and `cdd_oran.xmethod.runner:DummyMethod`. `pmrt_core` and `pmrt_nl` resolve to `oran_causal.core.pmrt.rows:PmrtCore` and `oran_causal.core.pmrt.nonlinear:PmrtNl` (they already implement `Method`, so there is no adapter). `RETIRED`: the `cmi_knn` and `pdcor` refs of the dev specs raise `RetiredArmError` |
| `packages/methods/src/oran_causal/methods/controlled/_classic.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/_classic_common.py` | `cond_set` takes `CovariateSpec`. `by_declare` is core `weighted_step_up` on the finite p-values. `sign_pcorr_given_Z` renamed `sign_pcorr_given_z`; `_threads` renamed `thread_limit`. `peak_rss_mb` dropped (no arm uses it); `placebo_tau` delegates to `oran_causal.core.scoring.placebo_tau` |
| `packages/methods/src/oran_causal/methods/controlled/_citests.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/_citests_common.py` | `SequentialP` and `sequential_p_from_draws` come from `oran_causal.core.pvalues`; BY from `_classic.by_declare`. `tau_from_scores` dropped (tau is `oran_causal.core.scoring.placebo_tau`) |
| `packages/methods/src/oran_causal/methods/controlled/_fixedb.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/_fixedb.py` | As-is |
| `packages/methods/src/oran_causal/methods/controlled/corr.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/corr.py` | As-is |
| `packages/methods/src/oran_causal/methods/controlled/granger.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/granger.py` | As-is |
| `packages/methods/src/oran_causal/methods/controlled/pcorr.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pcorr.py`; `cdd_oran/analysis/auto_threshold.py` (`largest_gap`); `cdd_oran/e1slice/discovery_v2.py` (`partial_correlation_scores`, `EPS_COND`, `FROZEN_FLOOR`) | Extracts kept next to their only user |
| `packages/methods/src/oran_causal/methods/controlled/pcorr_hac.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pcorr_hac.py` | As-is |
| `packages/methods/src/oran_causal/methods/controlled/pc.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pc.py` | Settings in `PcSettings`; `run_pc(d, names, tiers, forbidden, settings)`; `forbidden_edges` |
| `packages/methods/src/oran_causal/methods/controlled/notears.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/notears.py` | `notears_linear_bk` keeps the authors' code (lint exemptions only) |
| `packages/methods/src/oran_causal/methods/controlled/_vendor/notears/LICENSE` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/_vendor/notears/LICENSE` | Byte-identical (sha256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`) |
| `packages/methods/src/oran_causal/methods/controlled/_vendor/notears/VENDORED.md` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/_vendor/notears/VENDORED.md` | Byte-identical; names the upstream commit `4a9ab19` |
| `packages/methods/src/oran_causal/methods/controlled/_vendor/notears/linear.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/_vendor/notears/linear.py` | Byte-identical (sha256 `39644acd5b5d5ebc5fb5be5783365137eaa3b309df73bc0a75bad1942a0a426f`); excluded from ruff and ty and marked `-text` |
| `packages/methods/src/oran_causal/methods/controlled/shap_dag.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/shap_dag.py`; `scripts/e2_baseline_shap_dag.py` (`fit_shap_importances`) | |
| `packages/methods/src/oran_causal/methods/controlled/two_tower.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/two_tower.py`; `scripts/e2_baseline_gnn.py` (`TwoTower`, `fit_two_tower`) | The `_sized` rebinding of the script's `TwoTower` becomes explicit `TowerSizes` and `TrainConfig`. `two_tower_mask` not ported (unused by the arm) |
| `packages/methods/src/oran_causal/methods/controlled/_cdl_model.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/_cdl_model.py` | `CdlSettings`; `CDL(settings, device)`. Only `fit_step` and `cmi_step` kept; `predict_next_state`, save/load and the `get_*graph` helpers dropped (unused) |
| `packages/methods/src/oran_causal/methods/controlled/cdl.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/cdl.py` | |
| `packages/methods/src/oran_causal/methods/controlled/_rcot.py` | `CDD-ORAN@0c89106:cdd_oran/e2slice/discovery_rcot.py` (`RCoTDiscoveryConfig`, RFF residuals, statistic, HBE p-value, `_gamma_sf`, helpers); `cdd_oran/e2slice/discovery_rcot_v2.py` (`frozen_config_v2`) | Only what `rcot2` uses |
| `packages/methods/src/oran_causal/methods/controlled/rcot2.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/rcot2.py` | `bh_per_target` is core BH; `block_perm_sequential` takes `h` keyword-only |
| `packages/methods/src/oran_causal/methods/controlled/mscr_bank.py` | `CDD-ORAN@0c89106:cdd_oran/discovery/mscr.py` | Execution settings in `Execution(n_jobs, device, block_elems)`; BY and the Besag-Clifford p-value from core |
| `packages/methods/src/oran_causal/methods/controlled/mscr_torch.py` | `CDD-ORAN@0c89106:cdd_oran/discovery/mscr_torch.py` | Torch optional (`cpu` / `cu128` extras) |
| `packages/methods/src/oran_causal/methods/controlled/mscr.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/mscr.py` | |
| `packages/methods/src/oran_causal/methods/controlled/dummy.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/runner.py` (`DummyMethod`, `_by_declare`) | BY via core; the old `c_m` used `np.sum`, so a threshold can differ by 1 ulp |

Not ported: `cdd_oran/xmethod/methods/cmi_knn.py` and `pdcor.py` (dropped by the port map; their refs are `RETIRED`).

## Tests

| New path | Source | Note |
|---|---|---|
| `packages/methods/tests/conftest.py` | `CDD-ORAN@0c89106:tests/test_xmethod_classic.py`, `tests/test_xmethod_citests.py` (synthetic-data helpers) | Session fixtures `make_classic`, `make_ci` |
| `packages/methods/tests/test_classic.py` | `CDD-ORAN@0c89106:tests/test_xmethod_classic.py` | |
| `packages/methods/tests/test_pcorr_hac.py` | `CDD-ORAN@0c89106:tests/test_xmethod_pcorr_hac.py` | |
| `packages/methods/tests/test_citests.py` | `CDD-ORAN@0c89106:tests/test_xmethod_citests.py` | rcot2, pcorr and mscr parts; cmi_knn and pdcor tests dropped |
| `packages/methods/tests/test_cdl.py` | `CDD-ORAN@0c89106:tests/test_xmethod_cdl.py` | Hermetic; skips when torch is absent |
| `packages/methods/tests/test_mscr_bank.py` | `CDD-ORAN@0c89106:tests/test_discovery_mscr.py` | Plus execution-invariance tests of `Execution` |
| `packages/methods/tests/test_extracts.py` | `CDD-ORAN@0c89106:tests/test_e1slice_discovery_v2.py`, `tests/test_e2slice_discovery_rcot.py`, `tests/test_e2slice_discovery_rcot_v2.py` | Only the tests of the extracted helpers |
| `packages/methods/tests/test_registry.py` | new | Every `ref` in `packages/experiments/specs/**` resolves or is retired; `import oran_causal.methods.registry` loads no torch, shap, causal-learn or statsmodels; arm modules import only core, sim and methods |
| `packages/methods/tests/test_eval_golden.py` | new | Reruns the arms on the fixture below |

## Test fixture

| New path | Source | Generator |
|---|---|---|
| `packages/methods/tests/data/eval_golden.json.gz` | `data/records/study4_eval/eval/merged.jsonl.gz` (frozen Study 4 EVAL records, campaign `xm-campaign/1`, code `5a95186`, Python 3.12.14, numpy 2.4.2, torch 2.10.0+cpu, glibc 2.39) | `gen_fixture.py` (session scratchpad), run as `uv run python gen_fixture.py data/records/study4_eval/eval/merged.jsonl.gz packages/methods/tests/data/eval_golden.json.gz`. 68 records: seed 3000000, n = 500, cells E1-R1, E3-R2 and E4-R2 (lambda 1), every arm; each keeps `method`, `job`, `config`, `dataset_sha256`, `edges` and the native list |

## Gate against the EVAL records

One seed (3000000) of every (environment, regime, lambda, kappa, n) cell with n <= 1000, every arm, record config: 1,099 runs on Ubuntu 24.04 (glibc 2.39), Python 3.14.4, numpy 2.5.4, torch 2.11. Every regenerated `dataset_hash` equals the recorded `dataset_sha256`, and no declaration or sign differs. The cdl and two_tower score drift (no decision change in 64 of 66 cells) depends on the CPU that produced the record: cells recorded on AMD hosts are bit-exact, and cells recorded on Kaggle's Intel Xeon are not. The old CDD-ORAN code run on the same host shows the same drift. The report of round 5, phase 4a has the per-arm table.
