---
id: CIT-BY
kind: citation
status: implemented
---

# 10 CIT-BY: citation check

## Claim (verbatim, `spec/claims.yaml`)

> The Benjamini-Yekutieli step-up controls the FDR under arbitrary dependence.

Locator given in the manuscript: no locator in the manuscript. Used at: `sec:inference`, `app:proof_cor`.

## Bibliography entry (verbatim, `sn-bibliography.bib`)

```bibtex
@article{benjamini2001dependency,
  author  = {Benjamini, Yoav and Yekutieli, Daniel},
  title   = {The Control of the False Discovery Rate in Multiple Testing under Dependency},
  journal = {The Annals of Statistics},
  year    = {2001}, volume = {29}, number = {4}, pages = {1165--1188},
  doi     = {10.1214/aos/1013699998}
}
```

## Task

- Find every place the manuscript cites `benjamini2001dependency` in `sources/` and read the sentence it supports.
- Obtain the cited work itself (publisher, DOI, author's copy). Do not rely on summaries or secondary sources.
- Check that the locator exists, that the cited result says what the manuscript uses it for, and under which conditions.
- If the work cannot be obtained, set the status to `needs clarity` and say what was tried.
- Where the result is short and finite, a Lean statement of it is welcome but not required.

## Result

Filled in by the agent.

| Field | Value |
|---|---|
| Status | `implemented` (work obtained, locator verified) |
| Source consulted (edition, DOI or URL, pages) | Benjamini and Yekutieli, Ann. Statist. 29(4):1165-1188 (2001), doi:10.1214/aos/1013699998; full text from the first author's copy http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_yekutieli_ANNSTAT2001.pdf (journal typeset, pages 1165-1188; Project Euclid PDF link returned an HTML stub) |
| Locator in the manuscript correct? | No locator given (current paper: `paper/sections/s02_related.tex:26-27`, `paper/sections/s04_pmrt.tex:123-124`; the ESM proof idea of `lem:s_by`, `paper/supplementary.tex:251-252`, names "the Benjamini--Yekutieli argument" without a citation, the ESM has no bibliography) |
| Correct locator | Theorem 1.3 (statement p. 1169; proof in Section 5, pp. 1182-1183, eqs. (27)-(29)) |
| Source statement (verbatim, short) | "Theorem 1.3. When the Benjamini Hochberg procedure is conducted with q/(sum_{i=1}^m 1/i) taking the place of q in (1), it always controls the FDR at level less than or equal to (m_0/m) q." (p. 1169; text extraction garbled the formula, restored from the proof on p. 1182) |
| Does the source support the manuscript's use? | Yes for s02 ("BY control keeps the FDR ... under arbitrary dependence"): "always" = any joint distribution of the test statistics (p. 1182, "for any joint distribution of the test statistics"), level m_0 q/m <= q. For s04 the weighted reshaped form is Blanchard-Roquain's (CIT-BR); BY covers the unweighted case (w_j = 1) and the argument (27)-(29) that `lem:s_by` reuses with slack |
| Conditions the source needs that the manuscript does not state | Null p-values P_i = 1 - F_{H_i0}(X_i) with Pr(P_i <= t) = t (proof eq. (27)); <= t suffices (Section 5, discrete/composite case, eq. (21) and Theorem 5.2). The paper's p-values are super-uniform only in the limit; the paper does not apply Thm 1.3 directly but re-derives the bound with slack delta (`lem:s_by`, Lean `OranCausal.StepUp.fdr_le_of_selfConsistent`), so no unstated condition is used |
| Notes | Drift: pinned `app_a.tex:240` cited BY for the argument inside the proof; the current ESM keeps the argument but drops the citation. Locator still to be added to the paper (orchestrator applies all locator proposals together): `\cite[Thm.~1.3]{benjamini2001dependency}` at s02_related.tex:27 and s04_pmrt.tex:124, ESM lem:s_by "(Benjamini and Yekutieli 2001, proof of Thm. 1.3)"; spec/claims.yaml locator still "no locator in the manuscript" |
