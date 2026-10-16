# Provenance: sim.system.decision (the decision layer) and `core.data.UnitTable`

Source commit for every `CDD-ORAN@0c89106` path below is `0c89106` on `feat/v2`; `e6_dev/` stands for `CDD-ORAN@0c89106:scratchpad/e6_dev/`. Constants, RNG keys and operation order are unchanged. Unit tables rebuilt from the study-1 records, the hypothesis space and gate structure, and the knockout labels and ground truth equal the old code's bit for bit (gates 1 to 3 of the phase 6b report). Only the sim-side parts are ported here; the estimators, the CRT statistics and the benchmark drivers follow in the methods and experiments phases.

| New path | Source | Note |
|---|---|---|
| `packages/sim/src/oran_causal/sim/system/decision/vocab.py` | `CDD-ORAN@0c89106:cdd_oran/decision/gt_p.py` (`FAMILIES`, `RELATIONS`, `PRIMARY_ORIENT`), `edge_score.py` (`KPIS`, `HYPOTHESES`, `PHYSICS_PRIOR`), `fdr_layer.py` (`KPI_GATE`, `GATE_DELTA`, `gate_parents`, `gate_graph`, `pfilter_groups`), `crt_units.py` (`MODES`, `LEVEL`), `labels_p.py` (`CELL_KPIS`) | Plant-free vocabulary; new `structure()` returns the `core.declare.Structure` that core's structured procedures take explicitly; `LEVEL_ARR`, `ORIENTS` constants |
| `packages/sim/src/oran_causal/sim/system/decision/gt.py` | `CDD-ORAN@0c89106:cdd_oran/decision/gt_p.py` | Knockout ground truth; the rule thresholds are a frozen `GtRule` (`GT_RULE`) passed to `classify`, `gt_table` and `summarize` instead of keyword arguments; `cluster_boot` returns a `BootStat` named tuple; `classify(stat, delta, rule)` |
| `packages/sim/src/oran_causal/sim/system/decision/units.py` | `CDD-ORAN@0c89106:cdd_oran/decision/units_p.py`, `plans.py` (`QUANTUM`, `half_step`), `adapters/e6.py` (`knob_region`) | `UnitArbiter` keeps the old constructor; `prev_t` starts at 0.0 instead of None (read only after the first observation); `_report_mediators`, `_static` split out of `mediators` |
| `packages/sim/src/oran_causal/sim/system/decision/collect.py` | `CDD-ORAN@0c89106:cdd_oran/decision/collect_p.py`; `e6_dev/e6p_opta_kb.py` (`incumbent_consts`) | `run_collection(cfg, policy, options, labeller, env)`: the keyword settings `T`, `open_rule`, `arb_warmup_s`, `count_all`, `churn_cap`, `arbiter_cls` moved into a frozen `Collection` dataclass (`COLLECTION` = the old defaults); `enc`/`dec` and `SERIES_FIELDS` moved to `records.py`; `install_tap` sets `plant.__dict__["take_counters"]` as before |
| `packages/sim/src/oran_causal/sim/system/decision/labels.py` | `CDD-ORAN@0c89106:cdd_oran/decision/labels_p.py` | The paused state `(env, obs, snap)` is one `Paused` named tuple: `rollout(at, policy, k, H, ext)`, `Labeller.label(at, unit, modes)`, `Labeller.region_probe(at, unit, site, mode, label)`; `Labeller.aa` and `Labeller.hold` are the public rollout helpers; `aa_rollout` reads `env.pending_requests` instead of `env._pending[2]` |
| `packages/sim/src/oran_causal/sim/system/decision/scenario.py` | `e6_dev/e6p_screen.py` (`make_cfg`, `check_frozen`, `FROZEN_E6P`, `E6_FROZEN`, `PAIRS`, `STRATA`, `TARGET_L`, `deployed`), `e6_dev/e6p_state_json.py` (`LOAD_FACTOR`) | `check_frozen` raises `ValueError` instead of `SystemExit`; the default-field list is `DEFAULT_FIELDS`; `PAIR`, `STRATUM` = P3, surge-L40 (the study plant); `SMOKE_S` |
| `packages/sim/src/oran_causal/sim/system/decision/records.py` | `CDD-ORAN@0c89106:cdd_oran/decision/crt_units.py` (`build_unit_data`, `H_UNIT`, `HS`), `disc_bench.py` (`CACHE_SCHEMA`, `iter_records`, `episode_arrays`, `build_cache`, `load_pool`, the GT decode as `gt_episodes`, the privileged keys of its docstring as `PRIVILEGED_KEYS`), `collect_p.py` (`enc`, `dec`, `SERIES_FIELDS`); `e6_dev/e6p_discovery.py` (`SCHEMA`, `STEP_S`, `unit_record`); `e6_dev/e6p_opta_kb.py` (`unit_record` with probabilities, renamed `incumbent_unit_record`) | Plant-free; `build_unit_data` returns `core.data.UnitTable`; `build_cache` drops its `log` argument; the pool is a frozen `Pool` |
| `packages/sim/src/oran_causal/sim/system/decision/synthetic.py` | `CDD-ORAN@0c89106:cdd_oran/decision/crt_units.py` (`synthetic_records`) | Own module because it imports `collect` (and so the plant); settings in a frozen `SyntheticSpec` (`DEFAULT_SPEC`) |
| `packages/core/src/oran_causal/core/data.py` (`UnitTable`) | `CDD-ORAN@0c89106:cdd_oran/decision/crt_units.py` (`UnitData`) | Frozen dataclass; `families` and `levels` stored as fields; properties `n`, `level`, `x`; `rows_of`, `subset` (indices or a mask, `prev` remapped) |
| `packages/sim/tests/test_decision_vocab.py` | new | Hypothesis space, priors, gate DAG, Holm graph and p-filter groups against `decision_vocab.json` and `packages/core/tests/data/fdr_declare_p.npz` |
| `packages/sim/tests/test_decision_gt.py` | `CDD-ORAN@0c89106:tests/test_e6p_gt.py` (GT rules) + new golden test | `classify`, `gt_table`, `receiving_shares` on the synthetic contrasts; `summarize` on four stored GT episodes against the old output |
| `packages/sim/tests/test_decision_gt_plant.py` | `CDD-ORAN@0c89106:tests/test_e6p_gt.py` (plant mechanisms, `Labeller(cells=True)`) | `Plant._gains` is `gains` in the new plant |
| `packages/sim/tests/test_decision_labels.py` | `CDD-ORAN@0c89106:tests/test_e6p_labels.py` | Ported; the five tests on the labelled episode are `slow` (45 s fixture) |
| `packages/sim/tests/test_decision_units.py` | `CDD-ORAN@0c89106:tests/test_e6p_units.py` + `half_step` golden test | Two lock-equals-reject episode tests are `slow` |
| `packages/sim/tests/test_decision_collect.py` | `CDD-ORAN@0c89106:tests/test_e6p_collect.py` | Ported |
| `packages/sim/tests/test_decision_records.py` | `CDD-ORAN@0c89106:tests/test_disc_bench.py`, records parts of `tests/test_crt_units.py` and `tests/test_crt_units_v2.py` | Synthetic `lab_series` compared decoded: zlib-ng in the Windows Python 3.14 build compresses to other bytes, the decoded arrays are identical |
| `packages/sim/tests/test_decision_scenario.py` | new | Frozen study config, `check_frozen` rejections, smoke mode, `deployed` |
| `packages/core/tests/test_unit_table.py` | new | `UnitTable` fields, `subset`, `rows_of` |

`CDD-ORAN@0c89106:tests/test_e6p_mechanics.py` has no decision imports; it is ported with the plant (`test_system_mechanics.py`, [sim_system.md](sim_system.md)).

Not ported here: `edge_score.gt_reference`, `physics_reference`, `p1_check`, `p2_check`, `top1_accuracy` (analysis, phase 9); `disc_bench.bench`, `draw_subsets`, `subset_metrics` and the method wrappers (experiments); `crt_units` CRT statistics and `PiAssignment`, `fdr_layer.stats_from_*` (methods, phase 7).

## Fixtures

Generated by the old code with CDD-ORAN's environment (Windows 11, Python 3.12.13, numpy 2.4.2), all under `packages/sim/tests/data/`.

| File | Content | sha256 |
|---|---|---|
| `decision_vocab.json` | Old `edge_score` `HYPOTHESES`, `KPIS`, `PHYSICS_PRIOR`; `fdr_layer` `gate_parents`, `gate_graph` (w0, G), `pfilter_groups`; `gt_p` `FAMILIES`, `RELATIONS`, `PRIMARY_ORIENT`; `crt_units` `MODES`, `LEVEL`; `labels_p.CELL_KPIS` | `b82f581a008a9171e3be6c5e79cb977332f70bc47cd8ad33ba5f68c75a767957` |
| `decision_gt_golden.npz` | The first 4 stored v4 GT episodes by seed from `data/records/study1_raw/e6p-disc-v4gt-1-a/res_0..3.jsonl` (375 labels, float32 deltas) and old `gt_p.summarize(n_boot=1000)` on them (JSON) | `6618ad6855d401e9792cb2460067593bc3be2a3f8de59ccfe4cc04650856a746` |
| `decision_units_golden.npz` | Old `crt_units.synthetic_records(rng(11), n_ep=4, units_per_ep=30, effect=60.0)` and old `build_unit_data` on them (H 90, and H 60 / 150) | `785b3ae23e73a213e2230d0b0614a3200f46622193ab283800bb34dc54bfb11c` |
| `decision_half_step.json` | Old `plans.half_step`, `e6_dev/e6p_screen.half_step` and `units_p.half_step_p` decisions on a grid of knobs, currents and proposals | `ba87ee96e6595cffdff3d73955e9088c39014325982a9f45076a4800446b9782` |

```text
PYTHONPATH=<CDD-ORAN> <CDD-ORAN>/.venv/Scripts/python.exe -B gen_vocab.py packages/sim/tests/data/decision_vocab.json
PYTHONPATH=<CDD-ORAN> <CDD-ORAN>/.venv/Scripts/python.exe -B gen_fixtures.py data/records/study1_raw packages/sim/tests/data
```

`gen_vocab.py` and `gen_fixtures.py` live in the phase 6b worker scratchpad.

## Re-collection on another host

Re-running a stored study-1 episode reproduces its plant trajectory, per-second series, SLA totals and knockout labels bit for bit on Linux (Ubuntu 24.04, glibc 2.39). The obs-side KPM mediators `own_act_ue`, `nbr_sum_act_ue` and `nbr_sum_prot_act_ue` of a few unit contexts differ by one active-UE tick in one report (0.008 to 0.016 in a five-report mean). The old code at `0c89106` on that host gives the same values as the new code under both the new stack and the records' stack (Python 3.12.13, numpy 2.4.2, scipy 1.18.1), so the residual comes from the host (the records were produced on Kaggle, Linux 6.12, glibc 2.35), not the port. Unit tables, caches and every reported number are built from the stored records, which the new code reads bit for bit.
