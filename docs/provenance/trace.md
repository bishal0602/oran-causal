# Provenance: trace member and spec

Journal repo `xApp-Journal-Discover-Telecommunications`: commit `8744ef6` (the working tree read on 2026-10-15 over `ec32c22`, committed unchanged on 2026-10-16). Each file is also pinned by sha256.

| New path | Source | Note |
|---|---|---|
| `spec/claims.yaml` | journal manuscript files listed below | 80 claims from a manual inventory of the manuscript (2026-10-15). Statements paraphrase the manuscript; paper labels are its `\label`s, plus `abstract` |
| `spec/claims.yaml` evidence | `CDD-ORAN@0c89106:reports/2026-10-02-v4-pmrt-confounded-certsafe/figures_data.json` (sha256 `85fe7cff…b74e`) | Every pointer and `expect` value checked against this file on 2026-10-15 |
| `spec/claims.yaml` evidence | `CDD-ORAN@0c89106:scratchpad/e6_dev/decision/v4_analysis.json` (sha256 `38d7d28c…b62a`) | Recall 0.79 = 23/29, derived (`derive`) |
| `spec/claims.yaml` code symbols | migration port map (old symbol → new module) | All 93 symbols of active claims import (rule F12, 2026-10-17) |
| `spec/claims.schema.yaml` | migration design (2026-10-15) | Paper-label pattern and `additionalProperties: false` added |
| `spec/claims.sty`, `spec/notation.sty` | migration design (2026-10-15) | New. Notation symbols are the paper's (`s03_setting.tex`, `s04_pmrt.tex`, `app_a.tex`) |
| `spec/main.tex`, `spec/sections/*.tex` | `spec/claims.yaml` | Statements generated from the registry; headings and labels from migration design (2026-10-15) |
| `spec/sections/03-estimands.tex` equation | journal `s03_setting.tex` `eq:tau_u` | verbatim |
| `spec/sections/04-pmrt.tex` equations | journal `s04_pmrt.tex` `eq:score`, `eq:stat` | verbatim |
| `spec/sections/05-inference.tex` p-values | journal `s04_pmrt.tex` `eq:pval` | verbatim |
| `spec/sections/05-inference.tex` sequential stopping | `CDD-ORAN@0c89106:docs/xmethod/PMRT_CORE.md` "Statistic" step 4 | Reworded to name the action generically |
| `spec/sections/05-inference.tex` tolerance | journal `app_a.tex` proof of Proposition 1, part 3 | verbatim |
| `spec/sections/06-declaration.tex` directions, untested | journal `app_a.tex` `tab:pipeline` step 8; `s04_pmrt.tex` after `cor:fdr` | verbatim / condensed |
| `spec/sections/07-referee.tex` equation | journal `s04_pmrt.tex` `eq:slope_ht` | verbatim |
| `spec/sections/08-gate.tex` equation | journal `s04_pmrt.tex` `eq:tol` | verbatim |
| `spec/sections/10-metrics.tex` recovery | journal `s05_evaluation.tex` (§ Baselines and metrics) | verbatim |
| `spec/sections/12-assumptions.tex` (A1), (A3), (A4) | journal `s04_pmrt.tex` assumption list; `app_a.tex` `app:assumptions` | verbatim; lead-in of (A4) trimmed |
| `spec/sections/13-theorems.tex` BY inequality, `eq:fdr` | journal `app_a.tex` `app:proof_cor`; `s04_pmrt.tex` `eq:fdr` | verbatim |
| `packages/trace/src/oran_causal/trace/{pointers,symbols}.py`, rules F7 and F12 | new code | F7 follows each evidence `pointer` (RFC 6901, `.json` / `.json.gz`) in `data/records/<record>` and compares `expect`: mappings by subset, numbers equal or equal after rounding half up to the registry's decimals, integers exact. F12 imports `module:attr.sub` of every active claim. Evidence with only `derive` is not read |
| `packages/trace/**` | new code | Design: migration design (2026-10-15). Nothing copied |

## Manuscript files read (working tree at HEAD ec32c22)

| File | sha256 |
|---|---|
| `sn-article.tex` | `8575f597f556e569e302ee00234c7d5f3727543eb6eaf96a6b9f03d93b774cd8` |
| `sections/s01_introduction.tex` | `be50cc1b9342c5f1af69409d66d017082785a7fa6684834f923243f6a202b425` |
| `sections/s02_related.tex` | `bf1390771e8e2badf15e30c845811f7fe1ca9b9f30eac4ba805fd36a2cf3e54d` |
| `sections/s03_setting.tex` | `db3788b5a8d35ffb05b34a5d5f42a79db3e8687ee31d8488757bed85dc30c8c1` |
| `sections/s04_pmrt.tex` | `158dee32a7071f3dfb3c4543e8342559d25bcb7aa72c2bfbae78fd7bd503b849` |
| `sections/s05_evaluation.tex` | `9496db23ccb14d0ad4d1585a4982e9b0cafe8f415a7ed6503bf8439ad305e178` |
| `sections/s06_discovery.tex` | `728fffc5fb57fef429bb5b418eae0651ab9da99fc33688a78014572d0e38cba8` |
| `sections/s10_discussion.tex` | `dc44f6a21e15963b0d7e15753289a5be525797a767b7a0183101e4624867a1e0` |
| `sections/s11_conclusion.tex` | `d865cfb093b79f6a6cd02d96eb4718450ded2ade90c1fcaf0d96698178db935b` |
| `sections/app_a.tex` | `ba32fdf072c9f92b47fef9cdd10abf4f446a2aa49445f8a98065a248beea1d68` |
| `sections/app_b.tex` | `dd72a204defb7e8a798ea8d15c6a474a17d283488a814faa35e4c8e5c226d709` |
| `sections/app_c.tex` | `90e155129d76c03a2304a62053ab9ae8c0c34e2806a0190c53d63c2b9ec1ef47` |
