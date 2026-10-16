# Provenance: sim.controlled, sim.controlled.datasets, core.data

Source commit for every `CDD-ORAN@0c89106` path below is `0c89106` on `feat/v2`.

| New path | Source | Note |
|---|---|---|
| `packages/core/src/oran_causal/core/data.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/api.py` | Behaviour as-is; module docstring and field comments rewritten as Google docstrings; `Method.tune`/`run` docstrings added |
| `packages/sim/src/oran_causal/sim/controlled/base.py` | `CDD-ORAN@0c89106:cdd_oran/envs/v2/base.py` | `V2Env` renamed `ControlledEnv`; `true_adj_matrix` moved here from the five subclasses (identical bodies); abstract `adjacency_edges` added |
| `packages/sim/src/oran_causal/sim/controlled/e1.py` | `CDD-ORAN@0c89106:cdd_oran/envs/v2/e1.py` | `E1V2Env` renamed `E1Env`; `adjacency_edges` exposed as a property over the module constant `ADJACENCY_EDGES` |
| `packages/sim/src/oran_causal/sim/controlled/e2.py` | `CDD-ORAN@0c89106:cdd_oran/envs/v2/e2.py` | `E2V2Env` renamed `E2Env`; `_TRUE_ADJACENCY` and `_DECOY_OMITTED_EDGE` made module constants |
| `packages/sim/src/oran_causal/sim/controlled/e3.py` | `CDD-ORAN@0c89106:cdd_oran/envs/v2/e3.py` | `E3V2Env` renamed `E3Env`; `TRUE_ADJACENCY` and `TRUNCATED_OMITTED_EDGES` made module constants |
| `packages/sim/src/oran_causal/sim/controlled/e4.py` | `CDD-ORAN@0c89106:cdd_oran/envs/v2/e4.py` | `E4V2Env` renamed `E4Env`; `MODE_*`, `LATENT_EDGES`, `TRUE_ADJACENCY` and `DO_GRID_N` are module constants (`MODE_*` also kept on the class) |
| `packages/sim/src/oran_causal/sim/controlled/e5.py` | `CDD-ORAN@0c89106:cdd_oran/envs/v2/e5.py` | `E5V2Env` renamed `E5Env`; design-constant comments dropped, mechanism summarised in the module docstring |
| `packages/sim/src/oran_causal/sim/controlled/datasets/generate.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/worlds/generate.py` | `generate_dataset` split into `_generate_interventional`, `_generate_logged`, `_apply_obs_noise`, `_assemble`; `lam`, `kappa` keyword-only; `_with_placebo` renamed `with_placebo`; `GENERATOR_VERSION` string unchanged because `dataset_hash` covers `meta` |
| `packages/sim/src/oran_causal/sim/controlled/datasets/e4_logged.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/worlds/e4_logged.py` | `E4LoggedEnv` now extends `E4Env` |
| `packages/sim/src/oran_causal/sim/controlled/datasets/truth.py` | `CDD-ORAN@0c89106:cdd_oran/xmethod/worlds/generate.py` | `truth_for`, `REGIMES_OF`, `candidates_for`, `roles_for`, sign and role tables, `PLACEBO*`, `E4_LAMBDAS`, split out so analysis needs no generator |
| `packages/sim/tests/test_controlled_tape.py` | `CDD-ORAN@0c89106:tests/test_v2_crn.py` | Ported, `E1V2Env` -> `E1Env` |
| `packages/sim/tests/test_controlled_truth.py` | `CDD-ORAN@0c89106:tests/test_e1_scm_gate.py`, `test_e2_decision_gate.py`, `test_e3_scm_gate.py`, `test_e4_structural_gate.py`, `test_e5_confound.py`, `test_e5_chain.py` | Environment-truth assertions kept (mechanism, latency, adjacency, standardisation, E4 and E5 confounder semantics); `scripts.*` gate harnesses, mutation-rejection tests and world-model regret tests dropped |
| `packages/sim/tests/test_datasets.py` | `CDD-ORAN@0c89106:tests/test_xmethod_noise.py` | World part only (kappa default, noise sd, common random numbers, E5 single noise layer, invalid kappa); runner tests and the `design_covariates` exact-fit test are not ported here. Alignment, placebo, design and truth tests are new |
| `packages/sim/tests/data/study4_eval_datasets.json` | `data/records/study4_eval/eval/merged.jsonl.gz` (frozen Study 4 EVAL records, campaign `xm-campaign/1`, spec `eval_full`, code `5a95186`) | One row per (world, regime, lambda, kappa, n) combination: the lowest seed present, its recorded `dataset_sha256`, and `y_mean`/`y_std` of the regenerated `Y` |

## Fixture generator

The fixture holds 100 combinations. The key fields come from the `job` object of each record, and the lowest seed per combination was selected. `y_mean` and `y_std` were computed by the ported generator on Ubuntu 24.04 (glibc 2.39), Python 3.14.4, numpy 2.5.4, after asserting that every regenerated `dataset_hash` equals the recorded `dataset_sha256`.

```text
docker run ubuntu:24.04: uv venv --python 3.14; uv pip install numpy==2.5.4
python make_fixture.py   # imports oran_causal.sim.controlled.datasets.generate, writes study4_eval_datasets.json
```

The `LIBM_REFERENCE` constant in `test_datasets_records.py` is the sha256 of `exp`/`erfc` outputs over a fixed grid, computed on the same Ubuntu 24.04 image (identical under Python 3.12 and 3.14).
