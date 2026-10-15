# Provenance: workspace skeleton

| New path | Source | Note |
|---|---|---|
| `LICENSE` | `CDD-ORAN@0c89106:LICENSE` | Verbatim copy, same holder |
| `pyproject.toml` (torch cu128 index, line-length 100) | `CDD-ORAN@0c89106:pyproject.toml` | Index moved to explicit cpu and cu128 extras; numpy and scipy pins relaxed to constraint ranges |
| `packages/figures/pyproject.toml` (dependency list) | `figures/pyproject.toml` (workspace-level `figures/` project `cddfig`) | numpy floor relaxed, playwright moved to the root `browser` group, analysis and sim added |
