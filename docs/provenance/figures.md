# Provenance: figures

Source: the workspace `figures/` project (uv project `cddfig`) at git `257dc48` plus its uncommitted working-tree edits (`entities.py`, `fig_appC_x3_misspec`, `fig_study1_subsamples`, `fig_study2_deferrals` and others, last modified 2026-10-14). All data come from CDD-ORAN at `0c89106`, reached through `data/records/` (see `data.md`).

| New path | Source | Note |
|---|---|---|
| `packages/figures/src/oran_causal/figures/**` | `figures/src/cddfig/**` | Moved as-is; relative imports made absolute; `ruff format` and safe fixes applied; long docstring and comment paragraphs rewrapped |
| `packages/figures/src/oran_causal/figures/paths.py` | `figures/src/cddfig/paths.py` | Rewritten: reads `data/records` and `packages/experiments/protocols`, writes to the journal repo or to `ORAN_CAUSAL_FIGURES_OUT`; `record()` and `protocol()` raise when a path is missing |
| `packages/figures/src/oran_causal/figures/cli.py`, `registry.py` | `figures/src/cddfig/cli.py`, `registry.py` | Console script `oran-causal-figures list \| build [NAME ...] [--preview] [--out DIR]` |
| `packages/figures/src/oran_causal/figures/plotted.py` | new | Rows, digest, PDF layout and table digest of what a figure plots; used by the tests |
| `packages/figures/src/oran_causal/figures/figs/fig_envs.py` | `figures/src/cddfig/figs/fig_envs.py` | Sim constants are imported from `oran_causal.sim.controlled` instead of regex-read from CDD-ORAN source text |
| `packages/figures/data/fig_idea_redraws.json` | `figures/data/fig_idea_redraws.json` | Byte-identical (sha256 `446027d9...`), `-text` in `.gitattributes`. Made offline by `figures/scripts/fig_idea_redraws.py`, which needs the old CDD-ORAN code and is not ported |
| `packages/figures/tests/*.py` | `figures/tests/*.py` | Ported; record-, journal- and DESIGN-dependent tests are marked `slow` by fixture |
| `packages/figures/tests/test_{cli,paths,plotted,synthetic_records,t0_gate}.py`, `conftest.py` | new | CLI, path mapping, digest and gate tests |
| `packages/figures/tests/data/plotted_golden.json` | digests of the OLD cddfig run on CDD-ORAN | See the generator below |

Dropped: `figures/_bak_*`, `figures/preview/`, `figures/fonts/` (IBM Plex, unused), `figures/scripts/`.

## Path mapping (CDD-ORAN path read by cddfig -> record set)

| CDD-ORAN path | `data/records/` |
|---|---|
| `scratchpad/xmethod/results/eval` | `study4_eval/eval` |
| `scratchpad/xmethod/results/exp_b` | `bridging/exp_b` |
| `scratchpad/xmethod/results/exp_c/paper` | `posthoc_x1/exp_c/paper` |
| `.../extras/extras_tables.json` | `posthoc_x2_x3/` |
| `.../extras/x4` | `posthoc_x4/x4` |
| `.../extras/diag` | `posthoc_x5_x6/diag` |
| `.../extras/x7` | `posthoc_x7/x7` |
| `.../extras/x8` | `posthoc_x8/x8` |
| `.../extras/revision` | `posthoc_x9_x16/revision` |
| `scratchpad/e6_dev/decision` | `system_artifacts/decision` |
| `reports/2026-10-02-v4-pmrt-confounded-certsafe` | `system_artifacts/report_2026-10-02` |
| `docs/` protocols for Studies 2, 3 and 4 | `packages/experiments/protocols/{study2,study3,study4}` |

## Gate

Each figure and table was built once by the old cddfig from CDD-ORAN and once by the new package from `data/records`, and the data each draws were compared (arrays and JSON written before drawing, not PDF bytes). Results are in `r6/figures.md`. The digests in `plotted_golden.json` were generated from the old project:

```
cd figures && .venv/Scripts/python.exe -B <scratchpad>/golden.py
```

run from outside the tree, with `PYTHONHASHSEED=0` (the old code iterated a set for marker order in `fig_study1_discovery`; the new `plotted.py` sorts marker-only lines).

## Source state and pins

The port is of the workspace `figures/` working tree over git `257dc48` with uncommitted edits (modified and untracked files below). Each edited or untracked source file is pinned by the sha256 of its working-tree copy; unlisted files equal `257dc48`.

| File (under `figures/`) | sha256 |
|---|---|
| `data/fig_idea_redraws.json` | `446027d9305c4ac94602ab39b418f79da35aa9d60008c53a3d79732766b6c323` |
| `scripts/fig_idea_redraws.py` | `9ee13f2ff4ecedd64f7f1e87f5e9152bac4876924362794cb13fb9dc4e70ffb4` |
| `src/cddfig/entities.py` | `ab24ed25bae13d0af7736a0392aa386f0b0730bb971ede4b135aae13b88c6e57` |
| `src/cddfig/figs/fig_appC_x3_misspec.py` | `cf27f8d2b0608fbd5c4ad88381c86e38cf3bd83565e51c856f1eb039660c6b78` |
| `src/cddfig/figs/fig_conflict_types.py` | `73c075ee9e8a7825227669456fee5b20358de14a06c6e5395f83cddea3a250d3` |
| `src/cddfig/figs/fig_decisions.py` | `1fac2c99bde1e2aa04fd26486f3f1f3f0d8cf125065547f6c944f12b961945e0` |
| `src/cddfig/figs/fig_envs.py` | `ce5a68d03201ad913141dd31ad0fe081086f695a8145079b4f5c90af37a48cfa` |
| `src/cddfig/figs/fig_idea.py` | `55b41f653706f44f96269a92a1cfcb748aa08d1a892f40b544084fb07197b740` |
| `src/cddfig/figs/fig_logging_validity.py` | `7c8a8822d06bdbaa15cf2df754959cf18d7a17597db2d352f585ea00c53718fd` |
| `src/cddfig/figs/fig_maps.py` | `40377513242409199bcd2a95a4055197ed13c23d35588dbac040a254a7e4ca62` |
| `src/cddfig/figs/fig_maps_esm.py` | `72c9d76b901b2dc864c0c30f19869f3eed7538fe294075e8643bd2b49b4534f4` |
| `src/cddfig/figs/fig_placebo.py` | `990d44ed6ea08ecad9e808e4faf0186f4d00e59bf0e5255d7e92341b6698f30c` |
| `src/cddfig/figs/fig_study1_subsamples.py` | `553a505b057c2d4eb470d3fe44d3289852ecece10435d7dd5d830374e5bf3f4f` |
| `src/cddfig/figs/fig_study2_deferrals.py` | `8d37d293956aaf420116785b98fab5fcf9dc71ce85374e4b802c946bd6bc9be2` |
| `src/cddfig/figs/fig_study3_certification.py` | `7a4a6341c0873033deda8d7406b157e8ee2c14bf0085aaa9ac34b1832c505626` |
| `src/cddfig/figs/fig_study4_recall.py` | `ec9e72d2a3d5abb70ad9236c61cab23bfc6e076f1e7d879caeffeaf6d01d8820` |
| `src/cddfig/figs/fig_study4_validity_env.py` | `d6b74c5e6384b3e40067ecbc8f1241d0b93a82abc8e3c39c5b77e6b6e9899470` |
| `src/cddfig/figs/tab_decision_all.py` | `cf69585f7c50d25df21c08c063288a0f77625e93a106ae66a62c0d2dad73c5c2` |
| `src/cddfig/methods.py` | `7b9e863340b3cf16fd360a9182aa5f60d26c6b2e669612b4a54f67adddade4ba` |
| `src/cddfig/palette.py` | `8bed9aa491a681944ae3be322a4230da529634c94e4deee9b1c806d26230ba77` |
| `tests/test_entities.py` | `ba92f84237534359182b111ca493d49073ea1b2092dc0d6c84cbf55d45008e41` |
| `tests/test_fig_decisions.py` | `ea6606a66430a09b4001ec489d72d68e15e767a3806ac78bd61abd95675cc0f0` |
| `tests/test_fig_envs.py` | `17c8cab15ebe9a9ad7f64dc4bb41562b00bfed4d39939fcf5e8af41ef5fb7f39` |
| `tests/test_tab_decision_all.py` | `f5dcf59af01f4a6dc96fb9c5bb281d3d9c84acfa6521be60c27bf00e926387fe` |
