# Provenance

oran-causal starts with fresh history. Code, protocols and records come from these sources, which stay unchanged.

| Source | Reference |
|---|---|
| CDD-ORAN | `github.com/Adhyadeshdahal/CDD-ORAN`, branch `feat/v2` at `0c89106` |
| Study 4 freeze and production | freeze `93856c2`; EVAL records produced at `5a95186` (amendment A-1) |
| Figures project (`cddfig`) | workspace `figures/` |
| Manuscript | journal repository at `8744ef6`, moved into `paper/` |
| Raw system-simulator runs | Kaggle kernel outputs, `bishalpanta/*` |

Per-area tables, one row per migrated file or record set:

| Area | File |
|---|---|
| Record sets and frozen experiment inputs | [provenance/data.md](provenance/data.md) |
| Workspace configuration | [provenance/skeleton.md](provenance/skeleton.md) |
| Claims registry and spec | [provenance/trace.md](provenance/trace.md) |
| Lean proof | [provenance/proofs.md](provenance/proofs.md) |
| Method core (`oran_causal.core`) | [provenance/core.md](provenance/core.md) |
| Study 4 arms and registry (`oran_causal.methods`) | [provenance/methods_controlled.md](provenance/methods_controlled.md) |
| Study 4 harness, campaign and driver (`oran_causal.experiments`) | [provenance/experiments_controlled.md](provenance/experiments_controlled.md) |
| Figures and tables (`oran_causal.figures`, was cddfig) | [provenance/figures.md](provenance/figures.md) |
| Study 4 analysis (`oran_causal.analysis.controlled`) | [provenance/analysis_controlled.md](provenance/analysis_controlled.md) |
| Controlled environments and datasets (`oran_causal.sim.controlled`, `core.data`) | [provenance/sim_controlled.md](provenance/sim_controlled.md) |
| System-level plant (`oran_causal.sim.system`) | [provenance/sim_system.md](provenance/sim_system.md) |
| System decision layer (`oran_causal.sim.system.decision`, `core.data.UnitTable`) | [provenance/sim_decision.md](provenance/sim_decision.md) |
| System PMRT pipeline and frozen artifact (`oran_causal.methods.system.pmrt`, `.artifact`, the `core.pmrt.units` driver) | [provenance/methods_system_pmrt.md](provenance/methods_system_pmrt.md) |
| System MSCR-CRT v2, discovery baselines, published comparators, referees and maps (`oran_causal.methods.system.{mscr_crt,baselines,published,referees,maps}`) | [provenance/methods_system.md](provenance/methods_system.md) |
| Cloud and VPS launchers (`oran_causal.experiments.launch`) | [provenance/experiments_launch.md](provenance/experiments_launch.md) |
| Manuscript, build entry point and figure design rules (`paper/`, `packages/figures/DESIGN.md`) | [provenance/paper.md](provenance/paper.md) |
| Studies 1-3 analysis (`oran_causal.analysis.system`) | [provenance/analysis_system.md](provenance/analysis_system.md) |
| Studies 1-3 drivers, discovery tests and referee artifacts (`oran_causal.experiments.system`) | [provenance/experiments_system.md](provenance/experiments_system.md) |
| Experiment B bridging study (`oran_causal.experiments.bridging`, `oran_causal.analysis.bridging`) | [provenance/bridging.md](provenance/bridging.md) |
