"""Study 1 data for fig_study1_discovery. Read-only from the record sets through oran_causal.figures.data; every derived quantity is
re-computed from the stored per-edge declarations and asserted against the stored metrics at build time.

FD  = reports/2026-10-02-v4-pmrt-confounded-certsafe/figures_data.json, key "v4"
V4  = scratchpad/e6_dev/decision/v4_analysis.json (gt.cells, slices_result.<slice>.metrics.<code>.declared,
      placebo_descriptive.<code>.declared)
Scoring (cdd_oran/decision/edge_score.py): TP = declared and TRUE (any sign), FP = declared and NULL, INDET excluded;
sign accuracy = share of TP whose declared sign equals the GT sign.
"""

from oran_causal.figures.data import load_json
from oran_causal.figures.figs import _studies as ST

PRIMARY = "loadsp_c+wby1s"
PMRT9 = [f"{s}+{l}" for s in ("loadsp_c", "plain_c", "max") for l in ("wby1s", "by", "dagger1s")]
BASELINES = ["corr", "granger", "granger_by", "shap_gbdt", "two_tower", "qacm", "int"]
SLICE_ARMS = [PRIMARY, "corr", "granger", "granger_by"]
FAMS = ("carrier", "prot_min", "ptx", "sleep")
RELS = ("own", "nbr", "far")
KPIS = ("pv", "v", "e", "rlf", "load")
CHAIN = ["sleep|nbr|pv", "sleep|nbr|load", "sleep|nbr|v", "ptx|nbr|load"]
TOL = 1e-6


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOL


def score(declared: dict[str, int], gt: dict) -> dict:
    """edge_score.confusion over all hypotheses and over nbr only."""
    out = {}
    for scope in ("ov", "ind"):
        tp = fp = fn = sok = 0
        for h, c in gt.items():
            if scope == "ind" and c["relation"] != "nbr":
                continue
            if c["status"] == "INDET":
                continue
            d = h in declared
            if d and c["status"] == "TRUE":
                tp += 1
                sok += declared[h] == c["sign"]
            elif d:
                fp += 1
            elif c["status"] == "TRUE":
                fn += 1
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 0.0 if tp == 0 else 2 * p * r / (p + r)
        out[scope] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "sign_acc": sok / tp if tp else None,
        }
    return out


def load() -> dict:
    fd = ST.fd()["v4"]
    v4 = load_json(*ST.V4_PARTS)
    gt = {f"{c['family']}|{c['relation']}|{c['kpi']}": c for c in v4["gt"]["cells"]}
    assert len(gt) == 60
    counts = {s: sum(c["status"] == s for c in gt.values()) for s in ("TRUE", "NULL", "INDET")}
    assert counts == fd["gt"]["counts"], counts
    nbr_true = sorted(h for h, c in gt.items() if c["relation"] == "nbr" and c["status"] == "TRUE")
    assert nbr_true == sorted(fd["gt"]["nbr_true"]), nbr_true

    arms = PMRT9 + BASELINES
    pooled, slices, placebo = {}, {}, {}
    names = [s for s in fd["slices"]["names"] if s != "pooled"]
    for code in arms:
        dec = dict(v4["slices_result"]["pooled"]["metrics"][code]["declared"])
        sc = score(dec, gt)
        st = fd["pooled"][code]
        assert len(dec) == st["n_declared"], code
        assert _close(sc["ov"]["precision"], st["ov_precision"]), (code, sc["ov"], st)
        assert _close(sc["ov"]["sign_acc"], st["sign_acc"]), code
        assert _close(sc["ind"]["f1"], st["ind_f1"]) and _close(
            sc["ind"]["recall"], st["ind_recall"]
        ), code
        assert _close(sc["ind"]["precision"], st["ind_precision"]), code
        pooled[code] = dict(st, declared=dec, score=sc)
        pl = v4["placebo_descriptive"][code]["declared"]
        assert len(pl) == fd["placebo_declarations"][code], code
        placebo[code] = list(pl)
    for code in SLICE_ARMS:
        slices[code] = {}
        for s in names:
            m = v4["slices_result"][s]["metrics"][code]
            dec = dict(m["declared"])
            sc = score(dec, gt)
            st = fd["per_slice"][code][s]
            assert _close(sc["ind"]["f1"], st["ind_f1"]) and _close(
                sc["ov"]["precision"], st["ov_precision"]
            ), (code, s)
            assert len(dec) == st["n_declared"], (code, s)
            slices[code][s] = dict(st, n_eps=fd["slices"]["n_episodes"][s])
    # stored ranges (S1.10) reproduce from the per-slice values
    for code in SLICE_ARMS:
        rg = fd["ranges"][code]
        allp = [slices[code][s]["ov_precision"] for s in names]
        alln = [slices[code][s]["n_declared"] for s in names]
        allp_p = [*allp, fd["per_slice"][code]["pooled"]["ov_precision"]]
        alln_p = [*alln, fd["per_slice"][code]["pooled"]["n_declared"]]
        assert _close(min(allp_p), rg["ov_precision_all18"][0]) and _close(
            max(allp_p), rg["ov_precision_all18"][1]
        ), code
        assert [min(alln_p), max(alln_p)] == rg["n_declared_all18"], code
        for n in (60, 120, 300):
            f = [slices[code][s]["ind_f1"] for s in names if slices[code][s]["n_eps"] == n]
            assert _close(min(f), rg[f"ind_f1_{n}"][0]) and _close(max(f), rg[f"ind_f1_{n}"][1]), (
                code,
                n,
            )
    # P2 win counts reproduce: PMRT ind_f1 >= baseline per slice
    for b in ("corr", "granger", "granger_by"):
        for n, key in ((60, "wins60"), (120, "wins120")):
            ss = [s for s in names if slices[PRIMARY][s]["n_eps"] == n]
            w = sum(slices[PRIMARY][s]["ind_f1"] >= slices[b][s]["ind_f1"] for s in ss)
            assert w == fd["P2"][b][key], (b, n, w)
    # headline numbers of PAPER_PLAN S1.3 / S1.7
    p3 = fd["P3_values"]
    assert _close(p3["ind_f1"], pooled[PRIMARY]["ind_f1"])
    assert p3["n_declared"] == len(pooled[PRIMARY]["declared"])
    assert all(len(placebo[c]) == 0 for c in PMRT9)
    p3_thr = load_json(*ST.DECISION, "v4_verdict.json")["P3"]["rule"]["precision"]
    return {
        "fd": fd,
        "gt": gt,
        "pooled": pooled,
        "slices": slices,
        "placebo": placebo,
        "slice_names": names,
        "p3": p3_thr,
        "p2": fd["P2"],
    }
