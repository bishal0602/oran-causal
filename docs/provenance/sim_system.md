# Provenance: sim.system (the system-level plant)

Source commit for every `CDD-ORAN@0c89106` path below is `0c89106` on `feat/v2`. The plant's numerics are unchanged: same operation order, dtypes and RNG streams. On one host and stack the old and new code give byte-identical per-second state digests (gate 2 of the phase 6a report).

| New path | Source | Note |
|---|---|---|
| `packages/sim/src/oran_causal/sim/system/config.py` | `CDD-ORAN@0c89106:cdd_oran/envs/e6/config.py` | `validate_scenario` split into `_validate_surge` and `_validate_mistune`; provenance tags kept as short comments |
| `packages/sim/src/oran_causal/sim/system/geometry.py` | `CDD-ORAN@0c89106:cdd_oran/envs/e6/geometry.py` | `Layout._cands` renamed `cands`; `_hotspots` split out of `Layout.__init__`; `MACRO_BORESIGHTS` made a module constant; `antenna_db` `hbs`, `hut` keyword-only |
| `packages/sim/src/oran_causal/sim/system/sim.py` | `CDD-ORAN@0c89106:cdd_oran/envs/e6/sim.py` | `_rng` renamed `tape_rng(seed, stream, t=0, salt=None)` (`TAPE_TAG`, `RESEED_TAG` constants); `Plant.tick` split into `_hidden`, `_arrivals`, `_radio`, `_mobility`, `_schedule` (+ `_protected_share`), `_queues`, `_count`, `_energy`, `_score_second`; `Plant.__init__` split into `_init_*`; `_gains` renamed `gains`, `_new_counters` renamed `new_counters`; `_Scenario` renamed `Scenario`, `_Scenario._dist` renamed `dist`; `make_scenario(cfg, None)` raises `ValueError` after validation |
| `packages/sim/src/oran_causal/sim/system/ric.py` | `CDD-ORAN@0c89106:cdd_oran/envs/e6/ric.py` | `_quantise` renamed `quantise`; `knob_get` and `knob_set` dispatch through tables (`_CELL_ATTR`, `_QUANT`); `KPM._make` split into `_make_fast`, `_make_p`, `_make_prot`, `_pool`; `Knob` type alias, `MOB_KEYS` constant |
| `packages/sim/src/oran_causal/sim/system/xapps.py` | `CDD-ORAN@0c89106:cdd_oran/envs/e6/xapps.py` | `Report`, `Request` aliases; `MRO._pair`, `TS._pair`, `SliceSLA._cell` split out; `SliceSLA` all-NaN `nanmax` warning suppressed locally (the test suite turns warnings into errors) |
| `packages/sim/src/oran_causal/sim/system/xapps_p.py` | `CDD-ORAN@0c89106:cdd_oran/envs/e6/xapps_p.py` | `ESPico._pico`, `Coverage._cell`, `SliceGuarantee._cell` split out |
| `packages/sim/src/oran_causal/sim/system/env.py` | `CDD-ORAN@0c89106:cdd_oran/envs/e6/env.py` | `E6Env` options keyword-only; `_apply` renamed `apply`; `step_apply` split into `_rollback`, `_decide`, `_accept`, `_write`; new `pending_requests` property (replaces `env._pending[2]`); `Obs`, `Decision`, `Arbiter` aliases, `STAT_KEYS`, `MAX_DEFER_S` |
| `packages/sim/src/oran_causal/sim/system/baselines.py` | `CDD-ORAN@0c89106:cdd_oran/envs/e6/baselines.py` | `subset` and `region_subset` return `Subset` / `RegionSubset` instances (attribute `keep`) instead of closures; `_static_carriers`, `_static_cio`, `CellPriorityLock._blocked` split out |
| `packages/sim/src/oran_causal/sim/system/trace.py` | `CDD-ORAN@0c89106:cdd_oran/decision/trace.py` | Moved from the decision package to the plant; `TraceRecorder.finish` split into `_row_arrays`, `_event_arrays`, `_kpm_arrays`, `_meta`; unused `_ST` dropped |
| `packages/sim/tests/test_system_scenarios.py` | `CDD-ORAN@0c89106:tests/test_e6_scenarios.py` | Ported with the renames above |
| `packages/sim/tests/test_system_baselines.py` | `CDD-ORAN@0c89106:tests/test_e6_baselines.py` | Ported |
| `packages/sim/tests/test_system_env_wg3.py` | `CDD-ORAN@0c89106:tests/test_e6_env_wg3.py` | Ported; `pending_requests` test is new |
| `packages/sim/tests/test_system_mechanics.py` | `CDD-ORAN@0c89106:tests/test_e6p_mechanics.py` | Ported (all tests are plant mechanics) |
| `packages/sim/tests/test_system_trace.py` | `CDD-ORAN@0c89106:tests/test_decision_trace.py` | TraceRecorder parts only (trace off, untraced copy, row and config rebuild, labels, ACK/NACK accounting, npz round trip); the collector (`decision.collect`) is replaced by a local deterministic WG3 exerciser |
| `packages/sim/tests/test_system_golden.py` | new | Short-episode score test against the old plant (counts rtol 3% / atol 2, energy rtol 1e-3), the DEF-SIM-CONST constants, and the `slow` golden-episode test against stored EVAL records |
| `packages/sim/tests/data/system_golden_records.json` | `CDD-ORAN@0c89106:scratchpad/e6_dev/runs/e6p-cs-eval-1/{a..e}/res_*.jsonl` (certsafe EVAL, schema `e6p-optaka2-rec/1`, Kaggle, Python 3.12.13, numpy 2.4.2, scipy 1.18.1, glibc 2.35; record code `8ef2f3c`, plant files identical to `0c89106`) | Seeds 191100 and 191101 x arms `noarb`, `freeze`, `sub:ES`, `sub:PowerES`, `sub:SliceGuarantee`, `sub:ES+PowerES`: the plant `sla` fields and `st_*` counters copied verbatim |
| `packages/sim/tests/data/system_short_episodes.json` | Old plant `CDD-ORAN@0c89106:cdd_oran/envs/e6/*`, `cdd_oran/decision/trace.py` run by the trajectory harness | The `score()` dict of `golden_short_noarb`, `golden_short_sub` and `e6p_mixed_trace` from the Windows 11 run (Python 3.14.4, numpy 2.5.4, scipy 1.18.1) |

## Fixture generators

The episode arms and configuration follow `CDD-ORAN@0c89106:scratchpad/e6_dev/e6p_certsafe.py` (`arm_job`) with `e6p_screen.make_cfg`, `run_env` and `outcome`: pair P3, stratum surge-L40, `load_factor = 1.7554687500000001`, `E6Env(cfg, log=False, wg3=True)` stepped to `total_s`; `freeze` runs with no xApps.

```text
# scores of the old plant
PYTHONPATH=<CDD-ORAN> <python> -B -W ignore trajcmp.py old OUT.json
# fixtures
python make_fixtures.py <CDD-ORAN> <win32 OUT.json> <linux OUT.json> packages/sim/tests/data
```

`trajcmp.py` and `make_fixtures.py` live in the phase 6a worker scratchpad; `test_system_golden.py` carries the same digest function (`_digest`, `run_short`). The plant's float results depend on the platform libm, so the short-episode test compares the stored `win32` scores within a tolerance; the `slow` golden-episode test compares the stored EVAL outcomes at rtol 5% (energy 1e-3).
