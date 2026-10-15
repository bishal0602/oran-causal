# R-60 extras: cost projection (DEV cost table)

Exploratory supplementary runs (EXTRAS_PROTOCOL.md); does not change C1-C3. Kaggle-reference CPU-h from `scratchpad/xmethod/specs/extras/dev_cost_agg.json` (DEV means; told arms = base arm; n without DEV data: campaign's power-law fit). Session wall ~ CPU-h / processes (one process per vCPU).

| spec | units | datasets | CPU-h | gen CPU-h | top arms (CPU-h) | wall h at 4 procs | basis not measured |
|---|---|---|---|---|---|---|---|
| x2_d02 | 2280 | 360 | 9.6 | 0.01 | pmrt_nl_eq 7.1, shap_dag 1.2, pc_eq 0.9 | 2.4 | none |
| x2_d05 | 2280 | 360 | 9.6 | 0.01 | pmrt_nl_eq 7.1, shap_dag 1.2, pc_eq 0.9 | 2.4 | none |
| x2_d10 | 2280 | 360 | 9.6 | 0.01 | pmrt_nl_eq 7.1, shap_dag 1.2, pc_eq 0.9 | 2.4 | none |
| x2_d20 | 2280 | 360 | 9.6 | 0.01 | pmrt_nl_eq 7.1, shap_dag 1.2, pc_eq 0.9 | 2.4 | none |
| x2_b05 | 2280 | 360 | 9.6 | 0.01 | pmrt_nl_eq 7.1, shap_dag 1.2, pc_eq 0.9 | 2.4 | none |
| x2_b80 | 2280 | 360 | 9.6 | 0.01 | pmrt_nl_eq 7.1, shap_dag 1.2, pc_eq 0.9 | 2.4 | none |
| x3_told | 4560 | 180 | 19.6 | 0.01 | pmrt_nl_eq 2.9, pmrt_nl_eq.s02 2.0, pmrt_nl_eq.s05 2.0 | 4.9 | pmrt_nl_eq, pmrt_nl_eq.l050, pmrt_nl_eq.l080, pmrt_nl_eq.l125, pmrt_nl_eq.l200 |
| x5_fail | 12000 | 6000 | 63.8 | 8.04 | pmrt_eq 28.0, pmrt_r3 27.8 | 16.0 | none |
| x5_adj | 10000 | 5000 | 70.1 | 8.85 | pmrt_eq 30.7, pmrt_r3 30.6 | 17.5 | none |
| x6_gbm | 3000 | 200 | 18.6 | 0.00 | pmrt_nl_eq 2.0, pmrt_nl_eq.w050 2.0, pmrt_nl_eq.w080 2.0 | 4.7 | none |

Total 229.7 CPU-h. Arms without any cost: none.
