---
id: CIT-PS
kind: citation
status: todo
---

# 12 CIT-PS: citation check

## Claim (verbatim, `spec/claims.yaml`)

> The Monte Carlo p-value that counts ties against the observed statistic is valid for every B when the B + 1 values are exchangeable.

Locator given in the manuscript: Sect. 4. Used at: `lem:exact`.

## Bibliography entry (verbatim, `sn-bibliography.bib`)

```bibtex
@article{phipson2010permutation,
  author  = {Phipson, Belinda and Smyth, Gordon K.},
  title   = {Permutation {P}-values Should Never Be Zero: Calculating Exact {P}-values When Permutations Are Randomly Drawn},
  journal = {Statistical Applications in Genetics and Molecular Biology},
  year    = {2010}, volume = {9}, number = {1}, pages = {Article 39},
  doi     = {10.2202/1544-6115.1585}
}
```

## Task

- Find every place the manuscript cites `phipson2010permutation` in `sources/` and read the sentence it supports.
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
