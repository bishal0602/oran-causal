# Provenance: experiments harness, campaign and Study 4 driver

Source commit `CDD-ORAN@0c89106` unless stated otherwise. Ported code keeps behaviour; splits follow port_map §2.1 and §2.5.

| New path | Source | Note |
|---|---|---|
| `packages/experiments/src/oran_causal/experiments/paths.py` | new | Locations of protocols, specs, artifacts, manifests |
| `.../harness/runner.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/runner.py` | Job planning, `run_part`, runner merge. `REGISTRY`/`load_method` → `oran_causal.methods.registry.resolve`; `_by_declare` and `DummyMethod` moved to methods (p1D). `job_key`/`method_config` take a dataset tuple. Cloud launchers (`kaggle`, `colab`, `_cloud_cmd`) not ported (phase 8) |
| `.../harness/records.py` | `cdd_oran/xmethod/runner.py` (`run_one`, `_clean`, `_versions`, `_done_keys`) | `run_one(method, Job)`; record fields unchanged; runner version `oran-causal-runner/1` |
| `.../harness/isolation.py` | `cdd_oran/xmethod/runner.py` (`PeakRSS`); `cdd_oran/xmethod/campaign.py` (`_run_isolated`, `classify_kill`) | `run_isolated(task, limit_s, wall_s)` |
| `.../campaign/spec.py` | `cdd_oran/xmethod/campaign.py` (`Unit`, EVAL guard, `validate_spec`, `expand`, `spec_sha256`) | Protocol texts injected (`ProtocolTexts`); default is the shipped `protocols/study4/PROTOCOL_A.md` |
| `.../campaign/shard.py` | `cdd_oran/xmethod/campaign.py` (`unit_cost`, `partition`, `complete_dataset_keys`) | Plus `assignment` (key → part) |
| `.../campaign/run.py` | `cdd_oran/xmethod/campaign.py` (`run_units`, `run_part`, budgets, infeasibility records, registry) | `run_units` split into small steps; `RunOptions`, `Providers`, `Shard`, `Overrun`; campaign version `oran-causal-campaign/1` |
| `.../campaign/merge.py` | `cdd_oran/xmethod/campaign.py` (`_index`, `merge_index`, `merge_records`, `merge`, `iter_jsonl`) | Streaming summary via `MergeTally`; summary written with LF |
| `.../campaign/integrity.py` | `cdd_oran/xmethod/campaign.py` (`accept_rule`, integrity stamp, `check_eval_preconditions`, `check_one_per_cpu`, `MERGE_SETS`) | EVAL guard: the interpreter must match the workspace pin (`.python-version`, `requires-python` of `uv.lock`) instead of exactly `3.12.14`; `FROZEN_EVAL_PYTHON` keeps `3.12.14` as provenance of the frozen records (user decision, round 5) |
| `.../campaign/provenance.py` | `cdd_oran/xmethod/campaign.py` (`code_info`, `dirty_paths`, `lock_pins`, `pin_check`, `is_torch_build`, `pkg_versions`) | The lock is the workspace `uv.lock` (`lock_versions`, `read_lock`), not a `uv export` file; `PythonPin`, `workspace_python` new. Env vars renamed `XM_CODE_COMMIT/DIRTY`, `XM_LOCK_FILE` → `ORAN_CAUSAL_CODE_COMMIT/DIRTY`, `ORAN_CAUSAL_LOCK_FILE`; dirt scope `packages/` and `packages/experiments/specs/`. `write_lock`, `_req_blocks` not ported (phase 8 bundle) |
| `.../campaign/host.py` | `cdd_oran/xmethod/campaign.py` (`host_info`, `cpu_quota`, `campaign_procs`, `proc_visible`, `load_info`) | `XM_PLATFORM` → `ORAN_CAUSAL_PLATFORM`; process marker `oran_causal.experiments` |
| `.../campaign/aggregate.py` | `cdd_oran/xmethod/campaign.py` (`aggregate`, `seed_row`, `summarise_cell`, `cluster_ci`, `wilson_deff`, `_rate`, `to_result`, `record_unit`) | DEV aggregate; not in the plan tree's file list |
| `.../campaign/cost.py` | `cdd_oran/xmethod/dev_power.py`; `cdd_oran/xmethod/campaign.py` (`cost_table_from_agg`, `measured_infeasible`, `_fit_cost`, `project`) | |
| `.../campaign/fmax.py` | `scratchpad/xmethod/fmax_calib.py` | Calibration only (`dependence`, `z_stats`, `pair_type_corr`, `table`, `se4`); the simulation is `oran_causal.analysis.controlled.fmax.fmax_simulate` |
| `.../campaign/kappa.py` | `scratchpad/xmethod/kappa_calib.py` (fit part) | `build` (episode design matrices) needs the system decision layer: phase 6/7 |
| `.../campaign/estimand.py` | `scratchpad/xmethod/estimand_check.py` | `_WORLDS` lookups replaced by `truth.ENV_CLASSES` and `e4_logged.GRID` |
| `.../controlled/study4.py` | `cdd_oran/xmethod/campaign.py` (`main`: list / run / merge) | Plain functions: `plan`, `run_shard`, `run_study`, `integrity_problems` |
| `packages/experiments/tests/test_campaign_*.py`, `test_study4.py` | `CDD-ORAN@0c89106:tests/test_xmethod_campaign.py` | 36 of 51 ported; dropped: cloud/launch/lock-export tests (phase 8) |
| `packages/experiments/tests/test_harness.py` | `tests/test_xmethod_harness.py` (runner tests), `tests/test_xmethod_noise.py` (runner tests) | |
| `packages/experiments/tests/test_fmax_estimand_kappa.py` | new | Golden checks against frozen outputs |
| `packages/experiments/tests/data/fmax_dependence.json` | `data/records/study4_dev/fmax_sim/dependence.json` (`CDD-ORAN@0c89106:scratchpad/xmethod/results/fmax_sim/dependence.json`) | Byte copy, sha `402802b1…` = spec `fmax_dependence_sha256` |
| `packages/experiments/tests/data/estimand_check.json` | `data/records/study4_dev/estimand_check.json` | Byte copy |
| `packages/experiments/tests/data/eval_record_fields.json` | first record of `data/records/study4_eval/eval/merged.jsonl.gz` | Field names only, extracted with a one-off script |
