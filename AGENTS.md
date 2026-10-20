# AGENTS.md

Rules for people and agents working in this workspace.

| Topic | Rule |
|---|---|
| Tooling | `uv` only: `uv run`, `uv add`, `uv lock`, `uv sync`. No pip, mypy or black |
| Python | 3.14. Dependency bounds live in the root `constraint-dependencies` |
| Lint and types | `ruff` (lint and format) and `ty`; `uv run oran-causal check` must pass |
| Imports | Absolute only (`oran_causal.<member>`); relative imports are banned |
| Not now | No Docker, devcontainer, GitHub Actions or release workflows |

## Layering

| Member | May import |
|---|---|
| core | nothing in the workspace, and not torch |
| sim | core |
| methods | core, sim |
| analysis | core, sim |
| figures | analysis, sim |
| trace | nothing |
| experiments | any member |

`tests/test_layering.py` enforces this with fresh-process imports. Edit the table and the test together.

## Comments

- No narration comments and no commented-out code.
- Allowed: license headers, constraints imposed by external tools, public API docstrings (Google style), paper or issue references.
- Encode constraints as types, tests or lint rules instead of prose.
- The `comment-sicko` review agent lives in the user's root agents directory.

## Claims

Tests that back a paper claim use `@pytest.mark.claim("<ID>")`. The registry is `spec/claims.yaml`. Spec references do not go into code comments.

## Commits

- Subject under 70 characters (100 at most if needed). No body.
- No `Co-Authored-By` or other attribution trailer.
