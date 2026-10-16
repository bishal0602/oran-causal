---
id: CIT-BR
kind: citation
status: todo
---

# 11 CIT-BR: citation check

## Claim (verbatim, `spec/claims.yaml`)

> The weighted step-up with a reshaping function controls the FDR at q sum_{H0} w_j / m under arbitrary dependence when the null p-values are super-uniform.

Locator given in the manuscript: Thm. 4.2. Used at: `app:proof_cor`.

## Bibliography entry (verbatim, `sn-bibliography.bib`)

```bibtex
@article{blanchard2008fdr,
  author  = {Blanchard, Gilles and Roquain, {\'E}tienne},
  title   = {Two Simple Sufficient Conditions for {FDR} Control},
  journal = {Electronic Journal of Statistics},
  year    = {2008}, volume = {2}, pages = {963--992},
  doi     = {10.1214/08-EJS180}
}
```

## Task

- Find every place the manuscript cites `blanchard2008fdr` in `sources/` and read the sentence it supports.
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
