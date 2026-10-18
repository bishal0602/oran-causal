---
id: CIT-RW
kind: citation
status: implemented
---

# 15 CIT-RW: citation check

## Claim (verbatim, `spec/claims.yaml`)

> Roeder-Wasserman weights for weighted multiple testing.

Locator given in the manuscript: no locator in the manuscript. Used at: `sec:inference`, `tab:pipeline`.

## Bibliography entry (verbatim, `sn-bibliography.bib`)

```bibtex
@article{roeder2009weighted,
  author  = {Roeder, Kathryn and Wasserman, Larry},
  title   = {Genome-Wide Significance Levels and Weighted Hypothesis Testing},
  journal = {Statistical Science},
  year    = {2009}, volume = {24}, number = {4}, pages = {398--413},
  doi     = {10.1214/09-STS289}
}
```

## Task

- Find every place the manuscript cites `roeder2009weighted` in `sources/` and read the sentence it supports.
- Obtain the cited work itself (publisher, DOI, author's copy). Do not rely on summaries or secondary sources.
- Check that the locator exists, that the cited result says what the manuscript uses it for, and under which conditions.
- If the work cannot be obtained, set the status to `needs clarity` and say what was tried.
- Where the result is short and finite, a Lean statement of it is welcome but not required.

## Result

Filled in by the agent.

| Field | Value |
|---|---|
| Status | `implemented` (work obtained, locator identified; locator to be added to the paper, see Notes) |
| Source consulted (edition, DOI or URL, pages) | Roeder and Wasserman, Statist. Sci. 24(4):398-413 (2009), doi:10.1214/09-STS289; IMS electronic reprint arXiv:1010.4637v1 (16 pp.; "This reprint differs from the original in pagination and typographic detail", so journal page numbers are not given below) |
| Locator in the manuscript correct? | No locator given: `paper/sections/s04_pmrt.tex:124-126` ("Hypothesis j enters with a weight omega_j ... weights and directions are derived from the training episodes alone~\cite{roeder2009weighted}"); ESM `tab:s_pipeline` row 8 (`paper/supplementary.tex:121`, "Roeder--Wasserman omega_j: floor 0.2, cap 5, mean 1") names them without citation (the ESM has no bibliography) |
| Correct locator | Theorem 3.1 (Section 3, eqs. (3)-(4)) for the weight formula; Section 5 "Estimated weights" for estimating the weights on a separate subset of the data |
| Source statement (verbatim, short) | "Theorem 3.1. Given theta = (xi_1, ..., xi_m), the optimal weight vector w = (w_1, ..., w_m) that maximizes the average power subject to w_j >= 0 and [mean of w] = 1 is w = (rho_c(xi_1), ..., rho_c(xi_m)), where (3) rho_c(xi) = (m/alpha) Phibar(xi/2 + c/xi) I(xi > 0), and c = c(theta) is defined by the condition (4) (1/m) sum_j rho_c(xi_j) = 1." Sect. 5: "split the data into subsets, using one subset to estimate xi_i, and hence w(xi_i), and the second to conduct a weighted test of the hypothesis" |
| Does the source support the manuscript's use? | Yes for the form of the weights (spec:fdr.weights and `oran_causal.core.multiplicity:rw_weights` implement (3)-(4) with sum m) and for fixing them from separate data (Sect. 5; Sect. 6.2 "Independent data weights"). The optimality in Thm 3.1 is for the weighted Bonferroni test of one-sided normal statistics T_j ~ N(xi_j, 1) at level alpha; Sect. 2.2 only remarks that "similar results hold for FDR" (citing Genovese, Roeder and Wasserman 2006) |
| Conditions the source needs that the manuscript does not state | Not conditions for validity (any fixed weights with mean 1 keep the BY/BR bound, CIT-BR), only for optimality: Bonferroni rather than BY, one-sided normal statistics, known xi. The paper's adaptations are its own and are not in the source: alpha = q k_0 / c_m from an anticipated BY rejection count, xi_j = sqrt(max(z_prior^2 - 1, 0)) sqrt(n_target/n_prior), and the mean-preserving cap at (cap - floor)/(1 - floor) followed by omega_j = floor + (1 - floor) w_j, floor 0.2, cap 5 (spec:fdr.weights) |
| Notes | Proposed locator: s04_pmrt.tex:126 `\cite[Thm.~3.1 and Sect.~5]{roeder2009weighted}`; ESM tab:s_pipeline row 8 "Roeder--Wasserman $\omega_j$ (Roeder and Wasserman 2009, Thm. 3.1)"; spec/claims.yaml CIT-RW locator "Thm. 3.1; Sect. 5". Drift: pinned s04_pmrt.tex:126 also referred to Table tab:pipeline and the pinned app_a.tex:56 table row cited the work; both now in the ESM without citation |
