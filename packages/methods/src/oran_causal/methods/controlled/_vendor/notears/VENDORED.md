# Vendored: NOTEARS linear (Zheng, Aragam, Ravikumar, Xing, NeurIPS 2018)

- Source: https://github.com/xunzheng/notears, file `notears/linear.py`, branch `master`,
  commit `4a9ab19fe502e392503da331773c5223d82d3666` (fetched 2026-10-02).
- Licence: Apache-2.0 (`LICENSE`, copied verbatim).
- `linear.py` is byte-identical to upstream (no edits). The adapter
  (`oran_causal/methods/controlled/notears.py`) imports `notears_linear` from here; its background-knowledge variant
  `notears_linear_bk` is a separate function in the adapter whose only change is the L-BFGS-B bounds list (tested
  equal to this file when no edge is forbidden).
