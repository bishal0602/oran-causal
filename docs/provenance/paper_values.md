# Provenance: generated paper values and claim tags (phase 10, part 1)

No CDD-ORAN source: this is new code. The manuscript is the journal copy `8744ef6` under `paper/`; the numbers it prints from `\val{key}` come from the frozen records under `data/records`, each pinned by sha256 to `data/manifests`.

| New path | Role | Note |
|---|---|---|
| `spec/values.toml` | The value table: `[records]` aliases and one `[[values]]` entry per key (`from = "alias:/pointer"`, or `args` + `expr`, plus `fmt`) | 199 keys. Hand-edited source of truth; keys are stable names (`s1.map.pmrt.prec`, `s2.dec.pmrt.rlf_lo`, `x8.redraws`) |
| `packages/analysis/src/oran_causal/analysis/paper_values.py` | Generator and `--check` | `uv run python -m oran_causal.analysis.paper_values [--check]`. Pointers are RFC 6901 plus row selectors (`field=value&field=value` picks one row, `[...]`/`[*]` maps over rows); `.csv` records are read as rows. Formats: Python spec then `|` filters `up` (round half up on the decimal repr, as the authors rounded 0.625 to 0.63), `x100`, `thousands`, `math`. Reads each record once and refuses it when its sha256 differs from the manifest |
| `paper/generated/values.tex` | `\defval{key}{text}` lines, sorted by key | Generated and committed; regeneration from the records is byte-identical (`test_regeneration_from_the_frozen_records_is_byte_identical`) |
| `paper/generated/values.json` | Sidecar: per key `text`, `fmt`, `expr` and, per argument, `record`, `sha256`, `pointer` and the raw number | Read by trace rule F10 |
| `spec/paper_exempt.yaml` | Decimals the prose prints without `\val` (file, literal, count, reason) | 41 entries: constants of the method or design, simulator coordinates, and results with no record pointer yet. F11 fails on any other bare decimal and on a stale entry |
| `paper/claims.sty` | Copy of `spec/claims.sty` (the paper builds standalone on Overleaf) | Equality is a test |
| `paper/tools/flatten.py` | Submission flattener | Now expands `\val{key}` to the literal of `values.tex`, drops `\claim{ID}`, `\usepackage{claims}`, `\showclaimsfalse` and `\input{generated/values}` |
| `packages/trace/src/oran_causal/trace/paper.py` | Source-based manuscript scan for the paper rules | Follows the `\input` tree of `sn-article.tex` and `supplementary.tex`; no LaTeX build needed |
| `packages/trace/src/oran_causal/trace/rules.py` | F8 `\claim` ids registered, F9 every empirical/theorem claim tagged, F10 `\val` keys generated and backed, F11 no bare decimals, F14 post hoc wording | F10: used keys defined; `values.tex` equals the sidecar; sidecar record hashes equal the manifests; plain pointers re-resolve against the record to the stored value; claim `evidence.values` names exist. F14: a `posthoc: true` claim is tagged under a heading (or a section whose lead) containing "post hoc" or "chosen after the protocols were fixed" |

Tests: `packages/analysis/tests/test_paper_values.py` (format, expressions, pointers, hermetic generation and `--check`, spec errors, `claims.sty` copy, committed files vs the spec, byte-identical regeneration when `data/records` exists), `packages/analysis/tests/test_paper_flatten.py`, `packages/trace/tests/test_paper_rules.py`.

## Claim tags

`\claim{ID}` follows the `\label` of the first non-table, non-figure location in the claim's `paper:` list (the abstract is not tagged); all 80 claims are tagged. Post hoc claims are also tagged at each of their `sec:`/`app:` locations. The tags are invisible (`\showclaimsfalse`); `\showclaimstrue` prints them in the margin.

## Records read

Sha256 of every record is in `paper/generated/values.json`. Aliases: `fd` = `system_artifacts/report_2026-10-02/figures_data.json`, `v4a` = `system_artifacts/decision/v4_analysis.json`, `x8` = `posthoc_x8/x8/summary.json`, `x9`, `x11`, `x14`, `x15`, `x16` = `posthoc_x9_x16/revision/x{9_decision_e2e,11_gt_sensitivity,14_existence,15_replication,16_centre}.json`, `s4` = `study4_eval/eval/report.json`, `s4r` = `study4_eval/eval/csv/recall_vs_n.csv`, `br` = `bridging/exp_b/exp_b_tables.json`.
