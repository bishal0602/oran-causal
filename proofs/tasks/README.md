# Proof and citation tasks

One file per registered theorem or cited result (`spec/claims.yaml`). Work through them in number order, one at a time.

## Status

Each file starts with `status:`. Set it when you stop work on the file and fill in its `Result` table.

| Status | Meaning |
|---|---|
| `todo` | Not started |
| `implemented` | Acceptance below met |
| `needs clarity` | The claim is false, ambiguous, missing a hypothesis, or needs a result Mathlib lacks; the authors must decide. The table says what and why |
| `error` | Attempted and failed (build, proof or source access); the table says where and what was tried |

`implemented` is re-checked independently before it is accepted.

## Sources

`sources/` holds the manuscript files the claims come from, with LaTeX comments removed (drafting notes only; the text is unchanged). Originals: journal working tree over HEAD `ec32c22`, pinned by sha256 of the original file:

| File | sha256 of original |
|---|---|
| `s03_setting.tex` | `db3788b5a8d35ffb05b34a5d5f42a79db3e8687ee31d8488757bed85dc30c8c1` |
| `s04_pmrt.tex` | `158dee32a7071f3dfb3c4543e8342559d25bcb7aa72c2bfbae78fd7bd503b849` |
| `app_a.tex` | `ba32fdf072c9f92b47fef9cdd10abf4f446a2aa49445f8a98065a248beea1d68` |
| `references.bib` (`sn-bibliography.bib`) | `3097bde1a465370e3bd71f4fdf1a1aa5219378dc84d7a0bf9dfc2837a10a0585` |

Use only these files, the cited works themselves and Mathlib.

## Lean tasks (`kind: lean`)

| Item | Requirement |
|---|---|
| Project | `proofs/` (Lean v4.35.0, Mathlib v4.35.0; do not change the pins) |
| Files | one module per task under `proofs/OranCausal/`, imported from `proofs/OranCausal.lean` |
| Build | `lake build` from `proofs/` with no errors and no `sorry` |
| Axioms | `#print axioms` on each main theorem lists only `propext`, `Classical.choice`, `Quot.sound` |
| Forbidden | `sorry`, `admit`, new `axiom` declarations, `native_decide`, `implemented_by` / `extern` in the proof path, `set_option` that disables checks |

## Citation tasks (`kind: citation`)

Check the cited work itself against the manuscript's use; the task file lists the fields to report.

## Rules

- Edit only `proofs/` (and the status and result of the task files). Do not commit.
- Do not change a claim to make it provable; report the difference instead.
