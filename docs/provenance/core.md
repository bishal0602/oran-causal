# Provenance: core

Source commit: CDD-ORAN `0c89106` (`feat/v2`). Behaviour is unchanged unless the note says otherwise.

## Modules

| New path | Source | Note |
|---|---|---|
| `packages/core/src/oran_causal/core/multiplicity.py` | `CDD-ORAN@0c89106:cdd_oran/decision/fdr_layer.py` | Pure procedures: `c_harm`, RW weights, `cap_weights`, `weighted_step_up`, DAGGER, graphical Holm, p-filter, e-BH. `check_weights` extracted. The `max_iter` argument of `p_filter` is fixed at 1000 (the only value used) |
| `packages/core/src/oran_causal/core/declare.py` | `CDD-ORAN@0c89106:cdd_oran/decision/fdr_layer.py` | `HypStats`, `Prior`, `prior_weights`, `prior_directions`, `declare`. Hypotheses, gate DAG, graphical-Holm design and p-filter groups are explicit inputs (`Structure`); settings move into `WeightConfig` and `DeclareConfig`. `gate_parents`, `gate_graph`, `stats_from_json`, `stats_from_crt_v2` are not ported here (phases 6-7) |
| `packages/core/src/oran_causal/core/design.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pmrt_core.py` (assignment); `cdd_oran/decision/pmrt.py` (`v_design`, `v_var`) | `v_design` and `v_var` take the per-mode dose `levels` explicitly |
| `packages/core/src/oran_causal/core/pvalues.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pmrt_core.py` (`crt` counting), `pmrt_nl.py` (`crt_stat` counting), `_citests_common.py` (`SequentialP`, `sequential_p_from_draws`) | One `ExceedanceCounter` replaces the two inline Besag-Clifford loops |
| `packages/core/src/oran_causal/core/ridge.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pmrt_core.py` (`_ridge_fit`, `_mad`, `predictable_weights`); `cdd_oran/decision/pmrt.py` (`_ridge_gcv`); `cdd_oran/decision/eprocess_units.py` (`predictable_residuals`) | Settings in `AdjustConfig` and `ResidualConfig` (defaults = `EProcConfig` fields used) |
| `packages/core/src/oran_causal/core/covariates.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/covariates.py` | Keyword flags become `CovariateSpec`; `include_setpoints` and `setpoints` merge into `setpoints="all"|"focal"|"none"`; `EQ` and `R3` presets |
| `packages/core/src/oran_causal/core/scoring.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/score.py`; `cdd_oran/decision/edge_score.py` (`confusion`, `score_method`) | `confusion` and `score_method` take the hypothesis list explicitly. `p1_check`, `p2_check`, `top1_accuracy`, references are not ported here (phase 9) |
| `packages/core/src/oran_causal/core/gate.py` | `CDD-ORAN@0c89106:cdd_oran/decision/certsafe.py` | `tau_table`, `bound_ub`, `classify_edge`, `certify`, `design_slope_se`. Families, relations, guard KPIs and directions are a `GateSpace`; `certify` takes `can_defer` (MapGateV2 stays in methods); `design_slope_se` takes arrays in information order |
| `packages/core/src/oran_causal/core/pmrt/engine.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pmrt_core.py` (`crt`), `pmrt_nl.py` (`crt_stat`) | Settings in `RedrawConfig`; assignment in `Redraw` |
| `packages/core/src/oran_causal/core/pmrt/candidates.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pmrt_core.py`, `pmrt_nl.py` (`run` bodies) | Shared candidate, BY-family and result helpers of the two row statistics |
| `packages/core/src/oran_causal/core/pmrt/rows.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pmrt_core.py` | `PmrtCore`; the `rngs`/`W_override` arguments become `Overrides` |
| `packages/core/src/oran_causal/core/pmrt/nonlinear.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pmrt_nl.py` | `PmrtNl`; `gbm_profiles` returns `GbmProfiles` and takes `Focal` laws |
| `packages/core/src/oran_causal/core/pmrt/units.py` | `CDD-ORAN@0c89106:cdd_oran/decision/pmrt.py` | Plant-free statistic only: `PmrtConfig`, `slot_matrix`, `demean`, `running_center`, `lw_cov`, `kernels`, `rank1_pos`, `fit_core`, `variant_scores`, `fit_hmodel`, `h_weights`, `lin_z`, `modified`, `center_all`, `cv_select`, `fit_gb`. `fit_core` and `cv_select` take `TrainSet`/`CvData` |

## Not ported in phase 2 (phase 7)

From `CDD-ORAN@0c89106:cdd_oran/decision/pmrt.py`: `PmrtData`, `load_pmrt_pool`, `pool_rows`, `_bins`, `_hist`, `pmrt_data`, `subset_pmrt`, `cache_records`, `pmrt_data_from_records`, `hyp_design`, `align_ctx`, `strata`, `fit_pmrt`, `family_columns`, `family_tests_pmrt`, `run_pmrt`, `edges_pmrt`, params I/O. They need the unit table (`crt_units.UnitData`, `disc_bench.Pool`) and the relation slot counts.

## Test fixtures

| New path | Source | Generator |
|---|---|---|
| `packages/core/tests/data/fdr_step_up.npz` | `CDD-ORAN@0c89106:cdd_oran/decision/fdr_layer.py` run on numpy 2.4.2 | `gen_fdr.py` (session scratchpad), run as `PYTHONPATH=<CDD-ORAN> <CDD-ORAN>/.venv/Scripts/python.exe -B gen_fdr.py out.npz`, then split by part and re-encoded (`dc_e` int64 to int32, values unchanged) |
| `packages/core/tests/data/fdr_dagger.npz` | same | same |
| `packages/core/tests/data/fdr_declare_p.npz` | same | same; 2,000 stat sets, p-value procedures and prior weights |
| `packages/core/tests/data/fdr_declare_e.npz` | same | same; first 1,000 stat sets, e-value procedures |
| `packages/core/tests/data/pmrt_rows_golden.json.gz` | `CDD-ORAN@0c89106:cdd_oran/xmethod/methods/pmrt_core.py`, `pmrt_nl.py` on numpy 2.4.2 (CDD-ORAN environment) | `gen_pmrt.py` (session scratchpad): three synthetic datasets (iid, dither, logged) x six configs |
| `packages/core/tests/data/pmrt_rows_golden_np25.json.gz` | same old code run on numpy 2.5.4 (oran-causal environment, `PYTHONPATH=<CDD-ORAN>`) | same generator |
