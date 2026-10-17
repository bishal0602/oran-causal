# oran-causal

Causal discovery for xApp conflicts in O-RAN: the PMRT method, both simulators, every experiment, the figures and the manuscript. This repository is the code of the paper. CDD-ORAN (commit `0c89106`) is reference only; each ported file is listed in `docs/PROVENANCE.md`.

## Quick start

```bash
uv python install 3.14
uv sync --all-packages --extra cpu
uv run oran-causal check
uv run oran-causal reproduce --tier t0
```

Use `--extra cu128` instead of `--extra cpu` on CUDA hosts. The two extras conflict by design.

`oran-causal check` runs `ruff check`, `ruff format --check`, `ty check`, `paper-values --check`, `pytest` and `oran-causal-trace check --strict`, and stops at the first failure.

## Records

Frozen outputs of every study are record sets in `data/records/<set>/`, each pinned by `data/manifests/<set>.sha256`. They are not in git. Zenodo DOIs are not minted yet (`data/zenodo.yaml`); until then copy the sets into `data/records/` and check them with `uv run oran-causal data fetch --verify-only`.

## Reproduce tiers

`uv run oran-causal reproduce --tier t0..t3`. Costs are reference CPU-hours (the Kaggle CPU host).

| Tier | Needs | Does |
|---|---|---|
| t0 | frozen records (`--records DIR`) | Rebuilds every analysis output, table and figure from the records and diffs it with the stored files |
| t1 | stored Study 1, 3 and 4 records, a CPU | Smoke run of every stage, under 10 minutes |
| t2 | VPS and Kaggle, about 460 CPU-h (about 10 h wall); `--dry-run` prints the plan | End-to-end re-run of Studies 1-4 |
| t3 | VPS, Kaggle and Colab, about 1000 CPU-h (about 30 h wall); `--dry-run` prints the plan | Full re-run including the post hoc items X1-X16 |

Launchers: `oran-causal launch`, `colab`, `kaggle`, `vps`, `dispatch`. `oran-causal verify --records DIR` checks a re-run against the frozen conclusions.

## Paper

- `uv run oran-causal paper-values [--check]` regenerates `paper/generated/values.tex` and `values.json` from `spec/values.toml` and `data/records`. Edit the spec, never the generated files.
- `uv run oran-causal-figures list | build [NAME ...]` builds the figures into `paper/`.
- `uv run python paper/build.py [sn-article|supplementary]` builds the PDF into `paper/build/`.

## Traceability

`spec/claims.yaml` registers every claim of the paper. `uv run oran-causal-trace check [--strict] [--verbose]` checks the registry against the tests, record hashes and manuscript tags. In `--strict` mode a failing rule blocks, except F4 (claims without a passing default-run test, unverified cited locators). F5 (Lean), F7 (record pointers) and F12 (code symbols) are not implemented and report "not checkable". `uv run oran-causal-trace matrix` rewrites `docs/TRACEABILITY.md` from the last full test run; rerun it after the tests change.

## Layout

| Path | Contents |
|---|---|
| `packages/core` | The method: design, PMRT engine, p-values, multiplicity, gate. No simulator, no torch |
| `packages/sim` | System-level and controlled simulators |
| `packages/methods` | Study arms and baselines; torch via the `cpu` or `cu128` extra |
| `packages/experiments` | Frozen protocols, drivers, launchers and the `oran-causal` CLI |
| `packages/analysis` | Records to verdicts and paper values |
| `packages/figures` | Paper figures |
| `packages/trace` | Claims registry checker and pytest plugin |
| `paper/` | Manuscript (Springer `sn-jnl`) and `paper/generated/` |
| `spec/` | Claims registry, printed-value spec |
| `tests/` | Cross-package tests, including the import layering |
| `docs/` | Provenance and per-area provenance sections, `TRACEABILITY.md` |
