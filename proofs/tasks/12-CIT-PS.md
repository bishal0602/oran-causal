---
id: CIT-PS
kind: citation
status: implemented
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
| Status | `implemented` (work obtained, locator verified; claim and text to be restricted to the source, see Notes) |
| Source consulted (edition, DOI or URL, pages) | Phipson and Smyth, Stat. Appl. Genet. Mol. Biol. 9(1), Article 39 (2010), doi:10.2202/1544-6115.1585; full text from the authors' copy arXiv:1603.05766v1 (header: "Published 31 October 2010; corrected 9 February 2011", "Please cite as ... Article 39"). The De Gruyter PDF link returned HTTP 202 with an empty body; journal pagination not seen |
| Locator in the manuscript correct? | The locator exists: Section 4 "Exact p-values for Monte Carlo tests" (arXiv copy p. 6). Cited at `paper/sections/s04_pmrt.tex:117-118` (`\cite[Sect.~4]{phipson2010permutation}`, exact case); the ESM `lem:s_exact` (supplementary.tex:201-209) has no citation |
| Correct locator | Sect. 4 (with Sect. 3 for the uniformity of B used there) |
| Source statement (verbatim, short) | "Suppose m such datasets are simulated under the null hypothesis, each yielding a distinct test statistic tsim. ... For simplicity, we assume that the test statistic t is continuous, so that p_inf is uniform on the unit interval and all the tsim are distinct. ... Hence the exact Monte Carlo p-value is p_u = P(B <= b) = (b + 1)/(m + 1)." (Sect. 4) |
| Does the source support the manuscript's use? | Partly. It supports the (1 + #)/(B + 1) form and its exactness for every number m (= B) of Monte Carlo draws. It does not treat ties (it assumes a continuous statistic with all values distinct) and does not state the result under exchangeability (it assumes independent datasets simulated under H0); the paper's statistic on a finite randomisation can tie, and the paper counts ties against the observed value |
| Conditions the source needs that the manuscript does not state | (a) Continuous statistic, distinct values (no ties): the paper's claim and DEF-PMRT-PVAL count ties (`|z_b| >= |z|`); the tie-inclusive version is valid (conservative) but is not what Sect. 4 proves. (b) Independent draws under H0 rather than exchangeability: in the paper's exact case (lem:s_exact) the observed assignment and redraws are conditionally i.i.d. given G_0, which meets (b); the claims.yaml wording "exchangeable" is more general than the source |
| Notes | The Lean development does not rely on the source: `OranCausal.Exact.conditional_iid_rank_valid` proves Pr(rank-p <= t | G) <= t for conditionally i.i.d. draws with ties counted against (`upperRank` uses `s j <= s k`). Drift: pinned `app_a.tex:195-200` argued "independent and identically distributed given G_0 ... exchangeable" before citing Sect. 4; the article (s04_pmrt.tex:117-118) now says only "exchangeable". Resolution (authors, option A): the tie-inclusive validity is the paper's own result (Exact.lean); the article cites Sect. 4 only for the continuous case and the CIT-PS claim is restricted to "For a continuous statistic and B independent draws under the null, the Monte Carlo p-value (b + 1)/(B + 1) is exact for every B." (text applied by the orchestrator) |
