# Addendum to protocol v4: label-only rename "MSCR+" -> PMRT

Status: label change only. It does not amend any rule, threshold, seed range, statistic or declaration layer of
`E6P_DISCOVERY_PROTOCOL_V4.md`. That file stays frozen and unedited (commit 4fc2cd9, LF sha256
`6857466c91afa296a5cbaabd8fe0be9cb036dd35358736e855adfc540faf84ec`).

## Why
Protocol v4 calls its method "MSCR+". The method is no longer MSCR (the Max-Stratified Correlation-Ratio test of
E2-E5). It is a design-based conditional randomization test with predictable outcome adjustment and a matched-filter
aggregation, now named **PMRT, the Predictable Matched-Filter Randomization Test** (definition and component map in
`docs/benchmark/METHOD_NAMES.md`). The freeze protects the method's behaviour, not its labels. This addendum records
the rename and the proof that behaviour is unchanged.

## What changed
| frozen path (4fc2cd9) | frozen sha256 | renamed path | new sha256 |
|---|---|---|---|
| `cdd_oran/decision/crt_units_plus.py` | `c78276b5c5f533ca...` | `cdd_oran/decision/pmrt.py` | `3e71759881555268...` |
| `cdd_oran/decision/mscr_multi.py` | `3c701a156162fb63...` | `cdd_oran/decision/fdr_layer.py` | `326e14620c8304eb...` |
| `cdd_oran/decision/eprocess_units.py` | `17eba255168b3b81...` | (same path, docstring only) | `7a6492367a22de79...` |
| `cdd_oran/decision/crt_units.py` | `e5b85b90e56c6ebd...` | (unchanged) | `e5b85b90e56c6ebd...` |
| `cdd_oran/decision/crt_units_v2.py` | `9c47e939d140e922...` | (unchanged) | `9c47e939d140e922...` |
| `cdd_oran/decision/disc_bench.py` | `46912b47041df1bb...` | (same path, docstring only) | `21dcfefe1c8bacb0...` |
| `scratchpad/e6_dev/mscr_integrate_bench.py` | `bebe76407fac8266...` | `scratchpad/e6_dev/pmrt_bench.py` | `2629612436f570e5...` |
| `scratchpad/e6_dev/mscr_plus_artifacts.py` | `d40833af3cbbe5d8...` | `scratchpad/e6_dev/pmrt_artifacts.py` | `5d60e77a493b8924...` |

Full sha256s: `code_sha256` / `legacy_code_sha256` inside `E6P_PMRT_V4.json`.

- Identifiers: `PlusConfig` -> `PmrtConfig`, `PlusData` -> `PmrtData`, `plus_data` -> `pmrt_data`,
  `load_plus_pool` -> `load_pmrt_pool`, `fit_plus` -> `fit_pmrt`, `family_tests_plus` -> `family_tests_pmrt`,
  `run_crt_units_plus` -> `run_pmrt`, `edges_plus` -> `edges_pmrt`, `PLUS_STREAM` -> `PMRT_STREAM` (value **3**,
  unchanged, so every RNG stream is the same), `plus_config` -> `pmrt_config`. Version string `pmrt-v1`. The old
  string `mscr-crt-units-plus-v0` is kept as `LEGACY_VERSION` and written as `legacy_version` in outputs.
- `disc_bench.py` and `eprocess_units.py` are AST-identical to the frozen files once docstrings are removed.
- The old import paths `crt_units_plus` / `mscr_multi` are deprecation shims that re-export the new modules (same
  objects). The byte-exact frozen code is `git show 4fc2cd9:<path>`.
- Artifacts:

  | artifact | schema | sha256 (LF) |
  |---|---|---|
  | `E6P_MSCRPLUS_V4_FROZEN.json` (frozen, unedited) | `e6p-mscrplus-frozen/1` | `4735a85a1975edc6ddea412972be2f972a4ade9015844f153ae45d82f47d14ba` |
  | `E6P_PMRT_V4.json` (successor) | `e6p-pmrt/1` | `9991c390df3a53ef12b553990da52c6adab25d4087bb029f1427e20a898e6a91` |

  `E6P_PMRT_V4.json` was built by `pmrt_artifacts.py build --supersedes E6P_MSCRPLUS_V4_FROZEN.json` from the same
  params pickle (sha `0ef7cf7b...`, Kaggle `mscrplus-s-2`) and prior (sha `b6db889c...`, Kaggle `e6p-mscri-2`). The
  builder refuses unless every learned field (config, rho, ctx keys, per-hypothesis ridge / kernels / h-model, arms,
  layer prior, directions, weights) equals the frozen artifact's. It records `supersedes` (the frozen artifact's sha256),
  `legacy_code_sha256` and `renamed_from`.
- `e6p_disc_analyze_v4.py` accepts either artifact. The frozen one passes its code check through the PMRT artifact's
  supersession record (`code_via: supersession`); the PMRT one passes directly.

## Equivalence proof
Procedure (`scratchpad/e6_dev/pmrt_equivalence.py`):
1. **Local, small caches** (`local`): export the frozen tree from 4fc2cd9 with `git archive`. Run, in a separate
   interpreter per tree, the integrated statistic (every per-hypothesis z, beta, p2, p_plus, p_minus, sign, status) and
   all 9 statistic x layer declaration combinations (weights, directions, p used, declared, sign) on placebo1 and
   plxc2 (split 9) and dev1 (split 0). Compare the canonical JSON outputs byte for byte, then leaf by leaf.
2. **Full v4 data** (`reports`): the v4 analysis is run once with the frozen code (the verdict run) and once with the
   renamed code + `E6P_PMRT_V4.json` on the same inputs. The two reports must agree on every leaf except provenance and
   timing fields (`VOLATILE`: argv, artifact, protocol, timing, timing_s, wall_s, memory). String leaves are compared
   after mapping the label "MSCR+" to "PMRT" (the P2 rule text names the method).

Results:
- `pmrt_artifacts.py verify --legacy`: statistic bit-identical to the params pickle on placebo1 (2435 units), plxc2
  (2440) and dev1 (2630); layer weights / directions identical to the prior json; learned content identical to the
  frozen artifact.
- `pmrt_equivalence.py local` (2026-09-30), at B = 999 and B = 9999: **byte-identical**, 16929 leaves, 0 differences,
  max abs difference 0.0 (60 hypotheses per cache; primary loadsp_c + wby1s declares 0 / 0 / 4 on placebo1 / plxc2 /
  dev1, identically in both trees).
- `tests/test_pmrt_equivalence.py` repeats the local check at B = 199.
- `reports` on a local end-to-end analyzer dry run (v1 DEV / placebo-1 / gt-1 records, B 199, pseudo slices 5 / 10 /
  20, frozen main tree vs renamed tree): `analysis_v4.json` 16245 leaves and `verdict_v4.json` 209 leaves, 0
  differences. Before the label mapping, the only differences were timing and the P2 rule text ("count for MSCR+").
- **Full-data rerun (2026-10-01, Kaggle `e6p-disc-v4an-pmrt-1`, renamed code + `E6P_PMRT_V4.json`, same inputs as
  the verdict run `e6p-disc-v4an-1`):**
  - `pmrt_artifacts.py verify --legacy` passed on Kaggle (statistic bit-identical to the params pickle on the three
    small caches; learned content identical to the frozen artifact).
  - `verdict_v4.json`: **identical** (269 leaves, 0 differences; verdict PARTIAL in both).
  - `analysis_v4.json`: 44663 leaves, **14 differences, all in two descriptive baselines**: `desc_tau/two_tower/tau`
    (0.0058573 vs 0.0058578) and QACM's declarations on slice `s300_1` (9 vs 10 edges). Zero differences in every
    other leaf, including all 36667 PMRT / criterion leaves (every statistic x layer combination, per-hypothesis
    tables, K0, K0n, G, K1, P1-P3, S) and the corr / granger / granger_by / v2 baselines.
  - Cause: both baselines fit small seeded neural networks (two-tower; QACM's ANN candidate) on CPU. Their code did
    not change in the rename (`baselines_disc.py`, `envs/e6/published.py`, `scripts/e2_baseline_gnn.py` identical to
    34a2c15); the two jobs ran on different Kaggle machines (wall 3763 s vs 5046 s), and float reductions in the
    network training are not bit-reproducible across hardware. They are descriptive items, never criterion inputs.
  - Strict reading of the rule below: the check as written ("every leaf except VOLATILE") FAILS on these 14 leaves.
    Disclosed clarification, made after seeing the result: the rule's purpose is to detect a behaviour change in the
    renamed code, so it applies to leaves computed by code the rename touched; the descriptive neural baselines are
    excluded. Under that reading the rerun is equivalent and the rename stands. The frozen-code result remains the
    official v4 verdict either way.

## Rule
The v4 verdict is the one produced by the frozen code at 4fc2cd9. The renamed-code rerun serves only as the
equivalence check. If it differs on any non-volatile leaf, the rename is reverted, and the frozen code's result stands.
From this addendum on, new documents, the option (a) study and the paper say PMRT. "MSCR" refers only to the E2-E5 test.
