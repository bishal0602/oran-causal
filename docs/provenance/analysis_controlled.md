# Provenance: analysis.controlled

Source commit for every `CDD-ORAN@0c89106` path below is `0c89106` on `feat/v2`.

## Source files

| New path | Source | Note |
|---|---|---|
| `packages/analysis/src/oran_causal/analysis/records.py` | `CDD-ORAN@0c89106:scratchpad/xmethod/eval_analysis.py` (load, `file_sha256`, `_status`, `unit_of`, `to_result`), `eval_report.py` (`iter_lines`), `cdd_oran/xmethod/runner.py` (`job_key`, `cell_key`) | Streaming reader; key builders copied so analysis imports neither runner nor experiments |
| `.../analysis/controlled/protocol.py` | `scratchpad/xmethod/eval_analysis.py` (constants) | `ANALYSIS_VERSION` stays `xm-eval-analysis/5`; it appears in the frozen outputs |
| `.../controlled/spec.py` | `eval_analysis.py` (`planned_units`, `expected_units`, `not_in_grid`, `arm_kind`, `focal_arm`, `power_not_applicable`, `cost_field`, `native_partner`, `analysis_block`, `arm_label`) | `planned_units` split into `_block_arms` and `_block_cells` |
| `.../controlled/validity.py` | `eval_analysis.py` (`cluster_ci`, `design_effect`, `wilson_deff`, `classify`, `rate`, `combine`, `mean_ci`) | As-is |
| `.../controlled/fmax.py` | `eval_analysis.py` (`fmax_simulate`, `_boot_mult`, `_kinds`, `_psd_unit`, `_cand_corr`, `load_dependence`) | Not `fmax_calib.py`: the calibration script stays out. `fmax_simulate` is a pure NumPy simulation of the validity rule that `verdicts` needs; it runs no method. Split into plan, batch, pooled flags and summary; same RNG draw order |
| `.../controlled/cells.py` | `eval_analysis.py` (`seed_row`, `summarise`, `screen`, `build_cells`, `raw_declare`, `fixed_declare`, `counts`, `valid`, `powered`, `cell_class`, `conformal_tau`) | Scoring and the placebo threshold come from `core.scoring`; `build_cells` split into group, seed rows, status, entry |
| `.../controlled/power.py` | `eval_analysis.py` (`paired_diffs`, `paired_seeds`, `min_detectable_gap`, `achieved_power`, `t1_seed_count`) | T1 seed-count rule as-is |
| `.../controlled/verdicts.py` | `eval_analysis.py` (`validity_component`, `c1_arm`, `_majority`, `verdicts`) | `verdicts` (125 lines) split into `_c1`, `_c2b`, `_eq_table`, `_c3`, `_c2_wording`; shared arguments bundled in `Evidence` |
| `.../controlled/integrity.py` | `eval_analysis.py` (`integrity`, `protocol_check`, `load_amendments`, `by_declare`) | `integrity` (113 lines) split into code, stamp, candidate, dataset, package and BY checks; BY re-check uses `core.multiplicity.weighted_step_up` with NaN-padded untested hypotheses, and a differential test pins it to the old step-up |
| `.../controlled/tables.py` | `eval_analysis.py` (`assemble`, `like_for_like`, `cost_table`, `markdown`, `_clean`, `write`) | `markdown` (205 lines) split into one function per table section |
| `.../controlled/report.py` | `scratchpad/xmethod/eval_report.py` | Chunked analysis, DEV spec rendering, CSVs, `REPORT.md`, `report.json`, command line. `ReportRequest.generated_utc` and `input_labels` pin the recorded timestamp and input paths so a frozen report can be regenerated; `relabel_tune` is from `eval_analysis.py` |

## Tests

| New path | Source | Note |
|---|---|---|
| `packages/analysis/tests/conftest.py` | `tests/test_xmethod_eval_analysis.py` (synthetic study) | Fixtures instead of a loaded module |
| `.../tests/test_controlled_analysis.py` | `tests/test_xmethod_eval_analysis.py` | Dropped: `E.main` command-line test (the analysis command line is `report.main`), the fallback-tau test, `campaign.expand` equivalence, `campaign.run_units` stamp tests, `r58_spec.py` generator equality, the protocol register and `pmrt_core` file checks, the spec-wide `cdl.py` and `PROTOCOL_A.md` text checks |
| `.../tests/test_controlled_report.py` | `tests/test_xmethod_eval_report.py` | Plus a FINAL / pinned-timestamp round trip |
| `.../tests/test_controlled_golden.py` | new | Old `analyse` and `markdown` output on real EVAL records (below) |
| `.../tests/test_controlled_t0_gate.py` | new | T0 gate, `slow` |

## Test data

| Path under `packages/analysis/tests/data/` | Source | Command / note |
|---|---|---|
| `eval_full_spec.json` | `CDD-ORAN@0c89106:scratchpad/xmethod/specs/eval/full.json` | Byte copy of `packages/experiments/specs/eval/full.json` |
| `pilot_subset.jsonl.gz` | `CDD-ORAN@0c89106:tests/data/xmethod_eval_pilot_subset.jsonl.gz` | Byte copy |
| `dependence.json` | `data/records/study4_dev/fmax_sim/dependence.json` | Byte copy; sha256 `402802b1...` is the one the EVAL spec freezes |
| `eval_subset_records.jsonl.gz`, `eval_subset_spec.json` | `data/records/study4_eval/eval/merged.jsonl.gz` | 480 records: arms `pmrt_nl_eq`, `pcorr_native`, `pcorr_eq`, `pc_native`; worlds E1 and E4; regimes R1, R2, R3 (E4 lambda 1); n 500; 12 seeds per role. Records are the frozen ones, unmodified; the spec is the frozen spec reduced to those blocks |
| `expected_tables.json.gz`, `expected_tables.md.gz` | Old `eval_analysis.analyse` and `markdown` on the two files above | Run with `CDD-ORAN/.venv` (numpy 2.4.2): `analyse(records, spec, "93856c230d84abda2ff3b894218dd6cb0ecd47f9", dep=load_dependence(dependence.json), spec_sha=None, protocol={"ok": True})`, then `json.dumps(_clean(out), indent=1, sort_keys=True)` |

## How the frozen report was produced

From `scratchpad/xmethod/status/freeze.md` in CDD-ORAN (chunked path, run on the freeze branch):

```text
eval_report.py --spec specs/eval/full.json --merged results/eval/merged.jsonl.gz
  --freeze-commit 93856c230d84abda2ff3b894218dd6cb0ecd47f9 --amendments specs/eval/amendments.json
  --dev-merged results/dev/{full,ci_c,pmrt_nl,cdl}/merged.jsonl.gz --out results/eval
```

The new equivalent, as run by `test_controlled_t0_gate.py`:

```text
ReportRequest(spec=packages/experiments/specs/eval/full.json,
  merged=[data/records/study4_eval/eval/merged.jsonl.gz],
  freeze_commit=93856c230d84abda2ff3b894218dd6cb0ecd47f9,
  amendments=packages/experiments/specs/eval/amendments.json,
  protocol=packages/experiments/protocols/study4/PROTOCOL_A.md,
  dependence=data/records/study4_dev/fmax_sim/dependence.json,
  dev_merged=data/records/study4_dev/dev/{full,ci_c,pmrt_nl,cdl}/merged.jsonl.gz,
  generated_utc="2026-10-05T07:33:00Z", input_labels={spec and merged -> recorded scratchpad paths})
```

## Not ported

| Source | Reason |
|---|---|
| `eval_analysis.py` `main`, `t1` subcommand, `--as-tune` / `--min-flag-seeds` handling of the one-shot CLI | The report command line (`controlled.report.main`) covers the same options except the one-shot `eval_tables.json` writer, kept as `tables.write_tables`; the `t1` subcommand is `power.t1_seed_count` |
| `fmax_calib.py` | Produces the dependence and `F_max` tables: an experiment, not an analysis |
