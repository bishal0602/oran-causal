---
id: CIT-DUR
kind: citation
status: todo
---

# 14 CIT-DUR: citation check

## Claim (verbatim, `spec/claims.yaml`)

> The Lindeberg-Feller theorem and Polya's theorem on uniform convergence of distribution functions.

Locator given in the manuscript: Thm. 3.4.10; Exercise 3.2.9. Used at: `app:proof_prop`.

## Bibliography entry (verbatim, `sn-bibliography.bib`)

```bibtex
@book{durrett2019probability,
  author    = {Durrett, Rick},
  title     = {Probability: Theory and Examples},
  edition   = {5},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  year      = {2019},
  doi       = {10.1017/9781108591034}
}
```

## Task

- Find every place the manuscript cites `durrett2019probability` in `sources/` and read the sentence it supports.
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
