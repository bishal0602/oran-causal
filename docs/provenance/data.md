# Provenance: data

Record sets (`data/records/<set>/`, manifests in `data/manifests/`). The Kaggle bundles of study1_raw and study2_raw listed 144 and 174 `__pycache__/*.pyc` files that were never downloaded; they are not records and are left out of the manifests.

| set | origin | files | bytes | gate |
|---|---|---|---|---|
| study1_raw | kaggle:bishalpanta/e6p-disc-v4ev-1-a<br>kaggle:bishalpanta/e6p-disc-v4ev-1-b<br>kaggle:bishalpanta/e6p-disc-v4plc-1-a<br>kaggle:bishalpanta/e6p-disc-v4gt-1-a<br>kaggle:bishalpanta/e6p-disc-v4gt-1-b<br>kaggle:bishalpanta/e6p-disc-v4dev-1-a<br>kaggle:bishalpanta/e6p-disc-v4an-1<br>kaggle:bishalpanta/e6p-disc-v4an-pmrt-1 | 958 | 752,158,165 | PASS: 600 eval episodes (seeds 188100-188699), 89,067 H90 units; placebo 40, GT 40, dev 20; v4an outputs = frozen v4_analysis/v4_verdict (bit) |
| study1_train | kaggle:bishalpanta/e6p-disc-ev2-2-a<br>kaggle:bishalpanta/e6p-disc-ev2-2-b<br>kaggle:bishalpanta/mscrplus-s-2<br>kaggle:bishalpanta/e6p-mscri-2<br>kaggle:bishalpanta/disc-bench-1<br>CDD-ORAN working tree (untracked):scratchpad/e6_dev/runs/e6p-disc-ev3-1 | 963 | 1,869,240,588 | PASS: ev2 480 episodes (2x240), ev3 1,200 episodes = close records |
| study2_raw | kaggle:bishalpanta/e6p-conf-disc-1-a<br>kaggle:bishalpanta/e6p-conf-disc-1-b<br>kaggle:bishalpanta/e6p-conf-placebo-1-a<br>kaggle:bishalpanta/e6p-conf-gt-1-a<br>kaggle:bishalpanta/e6p-conf-gt-1-b<br>kaggle:bishalpanta/e6p-conf-dev-2-a<br>kaggle:bishalpanta/e6p-conf-discan-2 | 1012 | 735,880,597 | PASS: 600 disc episodes (seeds 186100-186699), 79,499 H90 units; placebo 200, GT 40, dev 20 = close records; disc_conf.json = frozen conf_disc_conf.json (sha dafca2bf = figures_data) |
| study2_3_decision | CDD-ORAN working tree (untracked):scratchpad/e6_dev/runs/e6p-conf-eval-1<br>CDD-ORAN working tree (untracked):scratchpad/e6_dev/runs/e6p-conf-evalan-1<br>CDD-ORAN working tree (untracked):scratchpad/e6_dev/runs/e6p-conf-dev-2<br>CDD-ORAN working tree (untracked):scratchpad/e6_dev/runs/e6p-cs-dev-1<br>CDD-ORAN working tree (untracked):scratchpad/e6_dev/runs/e6p-cs-eval-1<br>CDD-ORAN working tree (untracked):scratchpad/e6_dev/runs/e6p-cs-evalan-1<br>CDD-ORAN working tree (untracked):scratchpad/e6_dev/runs/e6p-cs-bounds-1 | 1279 | 74,266,995 | PASS: 32 res_*.jsonl byte-identical to Kaggle e6p-conf-eval-1-{a,b,c}, e6p-cs-eval-1-{a..e} |
| study4_eval | CDD-ORAN@0c89106:scratchpad/xmethod/results/eval | 15 | 115,403,836 | PASS: 150,960 records = summary (150,960 expected, 0 duplicates) |
| study4_dev | CDD-ORAN@0c89106:scratchpad/xmethod/results/dev<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/eval_dev<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/fmax_sim<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/kappa_calib<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/estimand_check.json<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/cov_equiv.jsonl<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/pmrt_equiv_B9999.json | 109 | 149,958,034 | PASS: 7 merged files = their summaries |
| bridging | CDD-ORAN@0c89106:scratchpad/xmethod/results/exp_b | 15 | 2,587,826 | PASS: 1,328 records (1,316 ok, 12 infeasible) = summary |
| posthoc_x1 | CDD-ORAN@0c89106:scratchpad/xmethod/results/exp_c | 51 | 11,171,465 | PASS: copied (no merged summary) |
| posthoc_x2_x3 | CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/x2_*<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/x3_told<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/EXTRAS_REPORT.md<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/EXTRAS_TABLES.md<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/extras_tables.json<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/provenance_platform.json<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/analyse_vps.log | 19 | 21,204,891 | PASS: 7 merged files = their summaries |
| posthoc_x4 | CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/x4<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/X4_REPORT.md | 10 | 605,049 | PASS: copied |
| posthoc_x5_x6 | CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/diag<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/DIAG_REPORT.md<br>CDD-ORAN-wt/xm-classic (xm/diag@2809c2a, untracked):scratchpad/xmethod/results/extras/{x5_adj,x5_fail,x6_gbm} | 13 | 9,813,257 | PASS: x5_adj 10,000, x5_fail 12,000, x6_gbm 3,000 records = summaries; x6 summaries equal the tracked copies |
| posthoc_x7 | CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/x7<br>CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/X7_REPORT.md | 16 | 3,584,893 | PASS: copied |
| posthoc_x8 | CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/x8 (res_*.jsonl untracked) | 15 | 25,664,184 | PASS: copied |
| posthoc_x9_x16 | CDD-ORAN@0c89106:scratchpad/xmethod/results/extras/revision | 21 | 1,890,462 | PASS: copied |
| system_artifacts | CDD-ORAN@0c89106:scratchpad/e6_dev/decision<br>CDD-ORAN@0c89106:reports/2026-10-02-v4-pmrt-confounded-certsafe | 71 | 4,377,826 | PASS: v4_analysis/v4_verdict equal the Kaggle v4an-1 outputs |
| freeze | CDD-ORAN@0c89106:scratchpad/xmethod/freeze | 9 | 92,480 | PASS: FREEZE_MANIFEST 301/301 and FREEZE_CALIB 36/36 at 93856c2 (LF-normalised) |

Frozen experiment inputs (byte-identical copies, git attribute `-text`).

| new path | source | note |
|---|---|---|
| packages/experiments/artifacts/E6P_CERTSAFE.json | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_CERTSAFE.json | byte-identical |
| packages/experiments/artifacts/E6P_CERTSAFE.json.sha256 | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_CERTSAFE.json.sha256 | byte-identical |
| packages/experiments/artifacts/E6P_CONF_MAPS.json | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_CONF_MAPS.json | byte-identical |
| packages/experiments/artifacts/E6P_CONF_MAPS.json.sha256 | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_CONF_MAPS.json.sha256 | byte-identical |
| packages/experiments/artifacts/E6P_MSCRPLUS_V4_FROZEN.json | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_MSCRPLUS_V4_FROZEN.json | byte-identical |
| packages/experiments/artifacts/E6P_MSCRPLUS_V4_FROZEN.json.sha256 | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_MSCRPLUS_V4_FROZEN.json.sha256 | byte-identical |
| packages/experiments/artifacts/E6P_PMRT_V4.json | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_PMRT_V4.json | byte-identical |
| packages/experiments/artifacts/E6P_PMRT_V4.json.sha256 | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_PMRT_V4.json.sha256 | byte-identical |
| packages/experiments/artifacts/E6P_PMRT_V4_EQUIV_FULL_analysis.json | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_PMRT_V4_EQUIV_FULL_analysis.json | byte-identical |
| packages/experiments/artifacts/E6P_PMRT_V4_EQUIV_FULL_verdict.json | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_PMRT_V4_EQUIV_FULL_verdict.json | byte-identical |
| packages/experiments/artifacts/E6P_PMRT_V4_EQUIV_LOCAL.json | CDD-ORAN@0c89106:docs/benchmark/artifacts/E6P_PMRT_V4_EQUIV_LOCAL.json | byte-identical |
| packages/experiments/artifacts/fmax_table.json | CDD-ORAN@0c89106:scratchpad/xmethod/results/fmax_sim/fmax_table.json | byte-identical |
| packages/experiments/artifacts/fmax_table.json.sha256 | new | sidecar sha256 |
| packages/experiments/manifests/seeds.json | CDD-ORAN@0c89106:docs/benchmark/SEED_REGISTRY.json | byte-identical, unconverted |
| packages/experiments/protocols/bridging/EXP_B.md | CDD-ORAN@0c89106:docs/xmethod/EXP_B.md | byte-identical |
| packages/experiments/protocols/posthoc/EXTRAS_PROTOCOL.md | CDD-ORAN@0c89106:scratchpad/xmethod/EXTRAS_PROTOCOL.md | byte-identical |
| packages/experiments/protocols/SHA256SUMS | new | sha256 of the protocol copies |
| packages/experiments/protocols/study1/E6P_DISCOVERY_PROTOCOL_V4.md | CDD-ORAN@0c89106:docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md | byte-identical |
| packages/experiments/protocols/study1/E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md | CDD-ORAN@0c89106:docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md | byte-identical |
| packages/experiments/protocols/study2/E6P_CONFOUNDED_PROTOCOL.md | CDD-ORAN@0c89106:docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md | byte-identical |
| packages/experiments/protocols/study3/E6P_CERTSAFE_PROTOCOL.md | CDD-ORAN@0c89106:docs/benchmark/E6P_CERTSAFE_PROTOCOL.md | byte-identical |
| packages/experiments/protocols/study4/CONTRACT.md | CDD-ORAN@0c89106:scratchpad/xmethod/CONTRACT.md | byte-identical |
| packages/experiments/protocols/study4/PROTOCOL_A.md | CDD-ORAN@0c89106:docs/xmethod/PROTOCOL_A.md | byte-identical |
| packages/experiments/protocols/study4/PROTOCOL_A_AMENDMENTS.md | CDD-ORAN@0c89106:docs/xmethod/PROTOCOL_A_AMENDMENTS.md | byte-identical |
| packages/experiments/protocols/system/E6P_DISCOVERY_PROTOCOL.md | CDD-ORAN@0c89106:docs/benchmark/E6P_DISCOVERY_PROTOCOL.md | byte-identical |
| packages/experiments/protocols/system/E6P_SCREEN_PROTOCOL.md | CDD-ORAN@0c89106:docs/benchmark/E6P_SCREEN_PROTOCOL.md | byte-identical |
| packages/experiments/protocols/system/E6P_SPEC.md | CDD-ORAN@0c89106:docs/benchmark/E6P_SPEC.md | byte-identical |
| packages/experiments/specs/dev/cdl.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/cdl.json | byte-identical |
| packages/experiments/specs/dev/cdl_le4000.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/cdl_le4000.json | byte-identical |
| packages/experiments/specs/dev/ci_c.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/ci_c.json | byte-identical |
| packages/experiments/specs/dev/full.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/full.json | byte-identical |
| packages/experiments/specs/dev/full_ci.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/full_ci.json | byte-identical |
| packages/experiments/specs/dev/pilot.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/pilot.json | byte-identical |
| packages/experiments/specs/dev/pilot_ci_cpu.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/pilot_ci_cpu.json | byte-identical |
| packages/experiments/specs/dev/pilot_ci_gpu.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/pilot_ci_gpu.json | byte-identical |
| packages/experiments/specs/dev/pmrt_nl.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/pmrt_nl.json | byte-identical |
| packages/experiments/specs/dev/pmrt_nl_cost.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/pmrt_nl_cost.json | byte-identical |
| packages/experiments/specs/dev/pmrt_nl_cost_e4.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/pmrt_nl_cost_e4.json | byte-identical |
| packages/experiments/specs/dev/smoke.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/smoke.json | byte-identical |
| packages/experiments/specs/dev/smoke_ci.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/smoke_ci.json | byte-identical |
| packages/experiments/specs/dev/vps_smoke.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/dev/vps_smoke.json | byte-identical |
| packages/experiments/specs/eval/amendments.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/eval/amendments.json | byte-identical |
| packages/experiments/specs/eval/full.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/eval/full.json | byte-identical |
| packages/experiments/specs/exp_b/counts_v4.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/exp_b/counts_v4.json | byte-identical |
| packages/experiments/specs/exp_b/exp_b.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/exp_b/exp_b.json | byte-identical |
| packages/experiments/specs/exp_b/exp_b_pilot.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/exp_b/exp_b_pilot.json | byte-identical |
| packages/experiments/specs/exp_b/exp_b_rest.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/exp_b/exp_b_rest.json | byte-identical |
| packages/experiments/specs/exp_c/timing.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/exp_c/timing.json | byte-identical |
| packages/experiments/specs/extras/dev_cost_agg.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/dev_cost_agg.json | byte-identical |
| packages/experiments/specs/extras/diag_cost_agg.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/diag_cost_agg.json | byte-identical |
| packages/experiments/specs/extras/projection.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/projection.json | byte-identical |
| packages/experiments/specs/extras/PROJECTION.md | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/PROJECTION.md | byte-identical |
| packages/experiments/specs/extras/x2_b05.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x2_b05.json | byte-identical |
| packages/experiments/specs/extras/x2_b80.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x2_b80.json | byte-identical |
| packages/experiments/specs/extras/x2_d02.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x2_d02.json | byte-identical |
| packages/experiments/specs/extras/x2_d05.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x2_d05.json | byte-identical |
| packages/experiments/specs/extras/x2_d10.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x2_d10.json | byte-identical |
| packages/experiments/specs/extras/x2_d20.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x2_d20.json | byte-identical |
| packages/experiments/specs/extras/x3_told.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x3_told.json | byte-identical |
| packages/experiments/specs/extras/x5_adj.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x5_adj.json | byte-identical |
| packages/experiments/specs/extras/x5_fail.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x5_fail.json | byte-identical |
| packages/experiments/specs/extras/x6_gbm.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/extras/x6_gbm.json | byte-identical |
| packages/experiments/specs/smoke_dummy.json | CDD-ORAN@0c89106:scratchpad/xmethod/specs/smoke_dummy.json | byte-identical |
