---
id: CIT-BY
kind: citation
status: todo
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
| Status | |
| Source consulted (edition, DOI or URL, pages) | |
| Locator in the manuscript correct? | |
| Correct locator | |
| Source statement (verbatim, short) | |
| Does the source support the manuscript's use? | |
| Conditions the source needs that the manuscript does not state | |
| Notes | |
