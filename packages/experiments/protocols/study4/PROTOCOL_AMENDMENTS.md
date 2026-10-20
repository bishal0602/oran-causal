# PROTOCOL_A amendments (Study A EVAL; PROTOCOL_A section 14)

protocols/study4/PROTOCOL.md (LF sha256 c5f7a4fe4dff4bfdeef3ccebac2dcc3748dfa9c2d0c3d17f2ac9d6246379fa22) is frozen and not edited.
Amendments are listed here, and the ones that touch records also in `specs/eval/amendments.json`
(id, commit, keys). Every sha256 below is over the file with LF line endings.

## A-1 (2026-10-05): EVAL records produced at commit 5a95186, not at the freeze commit

- **Ruling:** Q1 (a), 2026-10-05. Written before any EVAL record was analysed.
- **What.**
  - Every EVAL record stamps code commit 5a9518680ca2d74bed0c02d5157cbd53b4b6aa6a (dirty false).
  - The freeze commit is 93856c230d84abda2ff3b894218dd6cb0ecd47f9.
  - PROTOCOL_A s.11 accepts a record for FINAL if it comes from the freeze commit or from an amendment listing its
    key; A-1 lists every key.
- **Commit chain.** 5a95186 ("dev-runs: EVAL cost table (96-part LPT), dispatcher EVAL outputs") has
  parent 75a8fff (a merge of 93856c2 into the DEV-runs branch). `git diff 93856c2 5a95186` lists
  exactly three files:

  | file | change | LF sha256 at 93856c2 -> 5a95186 |
  |---|---|---|
  | xm_dispatch.py | launcher, +12 / -7 lines (below) | b9376bada11760696141ecc1f1d0aa6e59b615449171aaaa5b98f3f44edf8dc4 -> 58b022796c6d0cacc8ada44df7fc91e033792ff12935e491be203a0ab557ced6 |
  | artifacts/eval_cost_table.json | new: the EVAL run's 96-part LPT cost table | absent -> cf05e8809ef1741a3840881ea9b466b78e1b280289c4be8b8a5f82290356307a |
  | `_ref/__pycache__/eval_analysis` bytecode (.pyc) | new, stray bytecode (from the merge 75a8fff) | absent -> 65b9c3030b0f4c849e2bc6d6bad3e01891a7beda78d40994aae403824ccfaf74 (raw bytes) |

- **The xm_dispatch.py diff** (dispatcher only):
  - xm-eval-* Kaggle jobs count in the Kaggle session cap, next to xm-dev-*.
  - New option `--kaggle-mine-max` (my Kaggle sessions at once, <= the existing cap), stored in the dispatch state.
  - The pulled-results glob also collects EVAL result files: eval_res_*.jsonl, recursively under Kaggle out/, VPS
    out/ and Colab xm_eval_out/.
  - No change to unit selection, partitioning, methods, seeds or analysis.
- **The cost table** only assigns units to the 96 parts. Its canonical sha256 (campaign.spec_sha256) is
  9a68459241f5b5711ebd6768baf999e362d9e252b4bcee1bf0384b2373caf14a. This equals the single `cost_table_sha256`
  stamped on every record (R-59 F4) and listed in `results/eval/merged.jsonl.summary.json`.
- **The .pyc** is under `_ref/`. That directory is outside the freeze manifest and inputs (FREEZE_NOTE,
  N3), and nothing imports it.
- **Unchanged.** These are byte-identical at 93856c2 and 5a95186, with LF sha256 equal to FREEZE_NOTE:
  - campaign.py: ab87a268ffbce0dc60c87b479ffec9e48dfa6604f5706695313b745189837450
  - eval_analysis.py: 73685abb3e539a3125ead531ab63555c0292c6ef04bfd969ac9182035b8278df
  - eval_report.py: 8099ca4a43afaa2791c275a45093f5c771124fdcbcb66c7cd3393b9fdd6cb185
  - specs/eval/full.json: a02fd14833bb3a51bfb4136e78653b3b5c867bf0f494f344b3ad8abababdcc07
  - protocols/study4/PROTOCOL.md: c5f7a4fe4dff4bfdeef3ccebac2dcc3748dfa9c2d0c3d17f2ac9d6246379fa22
  - the method adapters (25 files): git tree 8a69550d22185691bdd5e06067b24177b30c841c at both commits.
  - No other code, config, protocol or spec file differs, and uv.lock and
    pyproject.toml are unchanged.
  - Of the 301 files in `data/records/freeze/freeze/FREEZE_MANIFEST.sha256`, xm_dispatch.py is the only one whose
    sha differs at 5a95186.
- **Why.** The dispatcher needed EVAL job accounting (the Kaggle session cap and the EVAL result paths) to run the
  campaign. The cost table is the per-run partition input that campaign stamps (R-59 F4).
- **Records touched.**
  - All 150 960 EVAL records, i.e. every expected unit of `specs/eval/full.json`.
  - Keys: `specs/eval/amendments.json`, amendment A-1, 150 960 keys, file LF sha256
    43be9758506c94dbed5072ca23c0c427386c53d38383f4308fbb588175fc5f03.
  - No re-run and no record changed. The analysis runs with
    `--freeze-commit 93856c230d84abda2ff3b894218dd6cb0ecd47f9 --amendments specs/eval/amendments.json`.
