---
id: CIT-HH
kind: citation
status: needs clarity
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
| Status | `needs clarity`: no legitimate full text of the book was reachable, so Cor. 3.1 and 3.2 could not be read |
| Source consulted (edition, DOI or URL, pages) | Not obtained. Tried (2026-10-18): doi.org/10.1016/C2013-0-10818-5 (resolves to the Academic Press 1980 book, ISBN 978-0-12-319350-6, Crossref type edited-book); ScienceDirect book page and chapter PDFs (Ch. 3 "The Central Limit Theorem", pp. 51-96, doi:10.1016/B978-0-12-319350-6.50009-8; "Index to Theorems, Corollaries, and Examples", pp. 299-300): HTTP 403; Elsevier API: 400 (key needed); Google Books (ISBN 0123193508): "No preview available"; Internet Archive: no item; Elsevier shop page: table of contents only (3.1 Introduction, 3.2 The basic CLT, 3.3 Progress toward a general CLT, ...) |
| Locator in the manuscript correct? | Not verifiable: the manuscript cites `\cite[Cor.~3.1 and 3.2]{hall1980martingale}` at `paper/sections/s04_pmrt.tex:112-113`; Chapter 3 is the CLT chapter (pp. 51-96), consistent with the locator, but the corollaries themselves were not seen |
| Correct locator | Unknown until the book is read |
| Source statement (verbatim, short) | Not available |
| Does the source support the manuscript's use? | Not checked. Use in the manuscript: martingale CLT with a random limiting variance (stable convergence S/s_n -> eta Z) and its self-normalised form (S/sigma -> N(0,1)) for the PMRT statistic (article s04_pmrt.tex:112-113; ESM prop:s_pmrt proof idea, supplementary.tex:192-199, names the theorem without locator) |
| Conditions the source needs that the manuscript does not state | To compare once the book is read: Lean's named obligation `OranCausal.Asymptotic.HallHeyde` (Asymptotic.lean:485-511, docstring "Hall--Heyde, Corollaries 3.1/3.2") assumes one filtration with initial segments m(n) (nested sigma-fields), deterministic scales -> infinity, square-integrable martingale differences, predictable variance / scale^2 -> eta^2 in probability, conditional Lindeberg -> 0 in probability, eta > 0 a.s. (for both conclusions), and concludes stable convergence of S/scale to eta Z and of S/sqrt(V) to N(0,1). The ESM (A4) supplies s_n -> inf, sigma^2/s_n^2 -> eta^2 with 0 < eta^2 < inf a.s., max|w_u|/s_n -> 0 |
| Notes | Drift: pinned app_a.tex:144-150 listed the hypotheses checked (conditional Lindeberg via (A4), conditional variance -> eta^2, nested histories, deterministic row length) next to Cor. 3.1 and Cor. 3.2; the current ESM keeps only the name. Ways to close: a library copy or scan of pp. 51-96 (or at least the corollaries' pages), or institutional ScienceDirect access |
