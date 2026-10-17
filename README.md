# oran-causal

Causal discovery for xApp conflicts in O-RAN: the PMRT method, both simulators, every experiment, the figures and the manuscript. Migrated from CDD-ORAN (commit 0c89106); each ported file is listed in `docs/PROVENANCE.md`.

## Bootstrap

```bash
uv python install 3.14
uv sync --all-packages --extra cpu
uv run oran-causal check
```

Use `--extra cu128` instead of `--extra cpu` on CUDA hosts. The two extras conflict by design.

## Check

`uv run oran-causal check` runs `ruff check`, `ruff format --check`, `ty check` and `pytest` in order and stops at the first failure.

## Reproduce tiers

| Tier | Scope | Status |
|---|---|---|
| t0 | Regenerate figures and tables from frozen records | implemented: `uv run oran-causal reproduce --tier t0 [--records DIR]` |
| t1 | CPU smoke run, under 10 minutes | implemented: `uv run oran-causal reproduce --tier t1` |
| t2 | End-to-end re-run on a VPS | dry run only: `uv run oran-causal reproduce --tier t2 --dry-run [--platform P]` |
| t3 | Full re-run across VPS, Kaggle and Colab | dry run only: `uv run oran-causal reproduce --tier t3 --dry-run [--platform P]` |

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
| `paper/` | Manuscript (Springer `sn-jnl`); build with `uv run python paper/build.py [sn-article\|supplementary]`, output in `paper/build/` |
| `tests/` | Cross-package tests, including the import layering |
| `docs/` | Provenance and per-area provenance sections |
