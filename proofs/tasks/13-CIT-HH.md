---
id: CIT-HH
kind: citation
status: todo
---

# 13 CIT-HH: citation check

## Claim (verbatim, `spec/claims.yaml`)

> The martingale central limit theorem with a random limiting variance and its self-normalised form.

Locator given in the manuscript: Cor. 3.1, Cor. 3.2. Used at: `app:proof_prop`.

## Bibliography entry (verbatim, `sn-bibliography.bib`)

```bibtex
@book{hall1980martingale,
  author    = {Hall, Peter and Heyde, Christopher C.},
  title     = {Martingale Limit Theory and Its Application},
  publisher = {Academic Press},
  address   = {New York},
  year      = {1980},
  doi       = {10.1016/C2013-0-10818-5}
}
```

## Task

- Find every place the manuscript cites `hall1980martingale` in `sources/` and read the sentence it supports.
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
