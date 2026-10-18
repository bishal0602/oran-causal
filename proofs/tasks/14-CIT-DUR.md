---
id: CIT-DUR
kind: citation
status: implemented
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
| Status | `implemented` (work obtained, both locators verified in the author's 5th-edition text) |
| Source consulted (edition, DOI or URL, pages) | Durrett, Probability: Theory and Examples, 5th ed., author's copy "Version 5 January 11, 2019" from the author's site https://services.math.duke.edu/~rtd/PTE/PTE5_011119.pdf (490 pp.). Edition check: the bib entry is the CUP 5th edition (2019, doi:10.1017/9781108591034, ISBN 9781108473682); Crossref lists the same 9 chapters plus "Measure Theory Details" in the same order as the author's copy (Ch. 3 "Central Limit Theorems": CUP pp. 98-177, author's copy pp. 113-204), so the edition matches; pagination differs, numbering (5th-ed. single sequence, per its preface) is taken from the author's copy. The CUP print itself was not seen |
| Locator in the manuscript correct? | Yes: `\cite[Thm.~3.4.10 and Exercise~3.2.9]{durrett2019probability}` at `paper/sections/s04_pmrt.tex:113-114` |
| Correct locator | Theorem 3.4.10 (Section 3.4.2, author's copy p. 148) and Exercise 3.2.9 (end of Section 3.2, author's copy p. 124) |
| Source statement (verbatim, short) | "Theorem 3.4.10. The Lindeberg-Feller theorem. For each n, let X_{n,m}, 1 <= m <= n, be independent random variables with E X_{n,m} = 0. Suppose (i) sum_{m=1}^n E X_{n,m}^2 -> sigma^2 > 0 (ii) For all eps > 0, lim_n sum_{m=1}^n E(|X_{n,m}|^2; |X_{n,m}| > eps) = 0. Then S_n = X_{n,1} + ... + X_{n,n} => sigma chi as n -> inf." / "3.2.9. If F_n => F and F is continuous then sup_x |F_n(x) - F(x)| -> 0." |
| Does the source support the manuscript's use? | Yes. Given the data, z_b = sigma^{-1} sum_u w_u v_u^(b) is a row of independent centred terms with variances summing to 1, and the Lindeberg sum vanishes (pinned app_a.tex:152-161, along a.s. subsequences); Thm 3.4.10 gives z_b => N(0, 1) and Ex. 3.2.9 (Phi continuous) upgrades it to uniform convergence of the redraw distribution function |
| Conditions the source needs that the manuscript does not state | None of substance. (a) Durrett's rows have length n; the paper's rows have a data-dependent length fixed given the data (padding with zeros is the standard reduction; Lean's `LindebergFellerPolya` states rows of length m(n)). (b) Durrett's hypotheses are deterministic limits; the paper's hold in probability and are passed through the subsequence argument (pinned app_a.tex:157-161), not stated in the current text. (c) Ex. 3.2.9 is an exercise (no proof in the book) |
| Notes | Lean: named obligations `OranCausal.Asymptotic.LindebergFellerPolya` (docstring "Durrett, Theorem 3.4.10, followed by Pólya (Exercise 3.2.9)": independent centred L2 rows, total variance -> 1, Lindeberg -> 0, concluding convergence in distribution and cdfDistance -> 0) and `Polya` (continuous limit cdf, weak convergence -> sup distance -> 0) match the source with sigma = 1. Drift: pinned app_a.tex:157-161 gave the subsequence argument and both locators in the proof; the current ESM prop:s_pmrt proof idea (supplementary.tex:196-199) names the theorems without locators; the article keeps both locators |
