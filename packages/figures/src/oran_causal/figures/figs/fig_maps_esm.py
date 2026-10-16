"""ESM version of fig_maps on the confounded logs (Online Resource; F5 appendix version).

Same encoding as fig_maps (ground truth, PMRT, and the two baselines with the highest F1 on these
logs, DEV-tuned thresholds as in tab:maps (b); asserted below), scored against the Study 2 knockout
ground truth. Not every method: eight facets do not fit the page, and the QACM map has no per-edge
record in the analyzer output.

Source: conf_disc_conf.json gt.cells (ground truth), maps.<map> ([family, relation, kpi, effect];
the drawn sign is the sign of the effect) and edge_quality.<map>.edges (the analyzer's own scoring).
Every tile category is re-derived from the ground truth and asserted against edge_quality and the
counts of tab:maps (b).
"""

from oran_causal.figures.data import load_json
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs.fig_maps import _tile, draw
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_maps_esm",
    width="full",
    paper=False,
    sources=[ST.SRC_CONF, ST.SRC_FD],
    caption_hint="Causal maps learned from confounded logs, against the knockout ground truth.",
)

PMRT = "MG:PMRT"
BASELINES = {
    "MG:corr@dev": "Correlation",
    "MG:granger@dev": "Granger",
    "MG:granger_by": "Granger + BY",
    "MG:shap_gbdt@dev": "SHAP-GBDT",
    "MG:two_tower@dev": "Two-tower",
    "MG:int@dev": "INT",
}
# tab:maps (b), Study 2 DEV-tuned: Ed., R, W, N
EXPECT = {PMRT: (22, 18, 0, 3), "MG:granger@dev": (32, 12, 6, 12), "MG:granger_by": (47, 16, 6, 19)}
N_TRUE = 29
EQ = {"right": "true_right", "wrong": "true_wrong", "false": "null"}


def data() -> tuple[dict, list[str]]:
    d = load_json(*ST.CONF_PARTS)
    gt = {(c["family"], c["relation"], c["kpi"]): c for c in d["gt"]["cells"]}
    assert len(gt) == 60
    assert d["gt"]["counts"]["TRUE"] == N_TRUE
    eq = d["edge_quality"]
    f1 = {}
    for b in BASELINES:
        tp = eq[b]["n_true_right"] + eq[b]["n_true_wrong"]
        f1[b] = 2 * tp / (2 * tp + eq[b]["n_null"] + (N_TRUE - tp))
    top = sorted(BASELINES, key=lambda b: -f1[b])[:2]
    assert top == ["MG:granger_by", "MG:granger@dev"], f1
    maps = {"GT": {h: _tile(gt[h], None) for h in gt}}
    for code in [PMRT, *top]:
        dec = {(f, r, k): (1 if v > 0 else -1) for f, r, k, v in d["maps"][code]}
        cells = {h: _tile(gt[h], dec.get(h, 0)) for h in gt}
        n = {
            c: sorted(h for h, v in cells.items() if v[0] == c)
            for c in ("right", "wrong", "false", "missed")
        }
        for c, key in EQ.items():  # same scoring as the analyzer
            assert n[c] == sorted(tuple(e) for e in eq[code]["edges"][key]), (code, c)
        assert (len(dec), len(n["right"]), len(n["wrong"]), len(n["false"])) == EXPECT[code], code
        assert len(n["missed"]) == N_TRUE - len(n["right"]) - len(n["wrong"]), code
        maps[code] = cells
    return maps, top


def build():
    maps, top = data()
    facets = [("GT", "Ground truth"), (PMRT, "PMRT")] + [(c, BASELINES[c]) for c in top]
    return draw(maps, facets, hero=PMRT)
