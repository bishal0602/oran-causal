---
id: CIT-BR
kind: citation
status: implemented
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
| Status | `implemented` (work obtained, locator verified) |
| Source consulted (edition, DOI or URL, pages) | Blanchard and Roquain, Electron. J. Statist. 2:963-992 (2008), doi:10.1214/08-EJS180; full text from arXiv:0802.1406v2 (21 Oct 2008), which carries the EJS header, volume and page numbers (Project Euclid PDF link returned an HTML stub) |
| Locator in the manuscript correct? | Yes: `\cite[Thm.~4.2]{blanchard2008fdr}` at `paper/sections/s04_pmrt.tex:142-143` (sec:inference, after Corollary cor:fdr); also cited without locator at `s04_pmrt.tex:124` together with BY |
| Correct locator | Theorem 4.2, p. 976 (proof: Proposition 2.7, p. 968; Lemma 3.2(iii), p. 972; Proposition 3.7, p. 974; BY instance in Section 4.2.1, p. 976) |
| Source statement (verbatim, short) | "Theorem 4.2. Consider R the step-up procedure associated to the factorized threshold collection Delta(h, r) = alpha pi(h) beta_nu(r), where the shape function beta_nu can be written in the form (6). Then R has its (Lambda-weighted) FDR controlled at level Pi(H0) alpha." (p. 976); (6): beta_nu(r) = int_0^r x dnu(x), nu any probability on (0, inf) (p. 972) |
| Does the source support the manuscript's use? | Yes. With Lambda the counting measure, alpha = q, pi(h) = w_h / m (in [0, 1] since w >= 0, sum w = m) and nu({k}) = 1/(k c_m) on {1..m} (p. 976: "beta(r) = r / gamma_m ... we recover the distribution-free procedure of Benjamini and Yekutieli (2001)"), the thresholds are q w_j i / (m c_m) (= ESM lem:s_by t_j(i)) and the level is Pi(H0) alpha = q sum_{H0} w_j / m, under unspecified dependence (Prop. 3.7: "for any multiple testing procedure R", no dependence assumption) |
| Conditions the source needs that the manuscript does not state | (a) Null p-values super-uniform, P(p_h <= t) <= t (standing assumption, Section 2.1, p. 966): stated by the claim; the paper's p-values satisfy it only in the limit, which is why lem:s_by re-derives the bound with slack delta. (b) pi is a fixed (non-random) function: the paper fixes weights and directions from data independent of the evaluation data (cor:fdr), i.e. conditionally fixed; not stated as a conditioning step but implied |
| Notes | Thm 4.2 is stated for the step-up procedure; the ESM lemma lem:s_by and Lean `OranCausal.StepUp.fdr_le_of_selfConsistent` assume only self-consistency, which BR also covers (Prop. 2.7 + Prop. 3.7, any self-consistent R; Thm 4.5 for step-up-down). Drift: pinned `app_a.tex:237` cited Thm 4.2 inside the proof with the full condition list; the current ESM has no citation and the article cites it at s04_pmrt.tex:143. spec/claims.yaml CIT-BR `paper: [cor:s_fdr]` while the \claim{CIT-BR} marker sits in sec:inference of the article |
