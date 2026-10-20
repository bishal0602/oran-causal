import numpy as np
import pytest

from oran_causal.sim.system.decision import gt
from oran_causal.sim.system.decision.vocab import CELL_KPIS, FAMILIES, KPIS, RELATIONS

NC = 8
EXP = {0: [0, 1, 2, 3], 4: [2, 3, 4, 5]}


def stat(mean: float, ci: tuple[float, float], n: int, n_eps: int) -> gt.BootStat:
    return gt.BootStat(mean, ci, n, n_eps)


@pytest.mark.claim("DEF-GT-LABEL")
def test_classify_rules() -> None:
    assert gt.classify(stat(1.0, (0.5, 1.5), 20, 5), 0.5)["status"] == "TRUE"
    assert gt.classify(stat(-1.0, (-1.5, -0.5), 20, 5), 0.5)["sign"] == -1
    assert gt.classify(stat(0.3, (0.1, 0.45), 20, 5), 0.5)["status"] == "NULL"
    assert gt.classify(stat(0.0, (-0.4, 0.4), 20, 5), 0.5)["status"] == "NULL"
    assert gt.classify(stat(0.0, (-2.0, 2.0), 20, 5), 0.5)["status"] == "INDET"
    assert gt.classify(stat(0.4, (0.1, 0.7), 20, 5), 0.5)["status"] == "INDET"
    assert gt.classify(stat(9.0, (8.0, 10.0), 5, 5), 0.5)["status"] == "INDET"
    assert gt.classify(stat(9.0, (8.0, 10.0), 50, 2), 0.5)["status"] == "INDET"


@pytest.mark.claim("DEF-GT-LABEL")
def test_cluster_boot_resamples_episodes() -> None:
    rng = np.random.default_rng(0)
    st = gt.cluster_boot([1.0, 1.0, 3.0, 3.0], [0, 0, 1, 1], rng, 200)
    assert st.mean == 2.0
    assert st.n == 4
    assert st.n_eps == 2
    assert 1.0 <= st.ci[0] <= st.ci[1] <= 3.0
    empty = gt.cluster_boot([], [], rng)
    assert np.isnan(empty.mean)
    assert empty.n == 0


def _label(rng: np.random.Generator, c: int, knob: str, step: float, eff: tuple) -> dict:
    load_nbr, pv_nbr = eff
    kp = list(CELL_KPIS)
    d = np.zeros((3, len(kp), NC))
    nbr = [q for q in EXP[c] if q != c]
    for k in range(3):
        d[k, kp.index("load"), c] = -3 * load_nbr
        d[k, kp.index("load"), nbr] = load_nbr + rng.normal(0, 0.2, len(nbr))
        d[k, kp.index("pv"), nbr] = pv_nbr / len(nbr) + rng.normal(0, 0.01, len(nbr))
        d[k, kp.index("e"), c] = rng.normal(0, 50.0)
    return {
        "c": c,
        "knob": knob,
        "exp": EXP[c],
        "step": step,
        "kpis": kp,
        "delta": d * np.sign(step),
    }


def _episodes(n_ep: int = 6, per: int = 8) -> list[dict]:
    rng = np.random.default_rng(0)
    eps = []
    for e in range(n_ep):
        labs = [_label(rng, 0, "sleep", 1.0, (1.0, 2.0)) for _ in range(per)]
        labs += [_label(rng, 0, "sleep", -1.0, (1.0, 2.0)) for _ in range(per // 2)]
        labs += [_label(rng, 4, "prot_min", 0.05, (0.0, 0.3)) for _ in range(per)]
        labs += [_label(rng, 4, "ptx", -3.0, (1.0, 0.0)) for _ in range(1 if e == 0 else 0)]
        eps.append(
            {"seed": 1000 + e, "n_cells": NC, "gt_static": {"cand": {"0": 3}}, "gt_labels": labs}
        )
    return eps


@pytest.mark.claim("DEF-GT-LABEL")
def test_gt_table_true_null_indet_on_synthetic_contrasts() -> None:
    rows = gt.unit_rows(_episodes())
    assert gt.relation_cells(0, EXP[0], NC) == {"own": [0], "nbr": [1, 2, 3], "far": [4, 5, 6, 7]}
    rule = gt.GtRule(n_boot=500)
    tab = gt.gt_table(rows, "dir", rule)
    cell = {(c["family"], c["relation"], c["kpi"]): c for c in tab["cells"]}
    assert len(tab["cells"]) == len(FAMILIES) * len(RELATIONS) * len(KPIS)
    assert cell[("sleep", "nbr", "load")]["status"] == "TRUE"
    assert cell[("sleep", "nbr", "load")]["sign"] == 1
    assert cell[("sleep", "own", "load")]["status"] == "TRUE"
    assert cell[("sleep", "own", "load")]["sign"] == -1
    assert cell[("sleep", "nbr", "pv")]["status"] == "TRUE"
    for k in KPIS:
        assert cell[("sleep", "far", k)]["status"] == "NULL"
    assert tab["delta"]["pv"] == 0.5
    assert cell[("prot_min", "nbr", "pv")]["status"] == "NULL"
    assert cell[("sleep", "own", "e")]["status"] == "INDET"
    assert cell[("ptx", "nbr", "load")]["status"] == "INDET"
    assert "unsupported" in cell[("ptx", "nbr", "load")]["why"]
    assert cell[("carrier", "own", "pv")]["n"] == 0
    assert sum(tab["counts"].values()) == len(tab["cells"])
    act = {
        (c["family"], c["relation"], c["kpi"]): c for c in gt.gt_table(rows, "act", rule)["cells"]
    }
    assert act[("sleep", "nbr", "load")]["mean"] < cell[("sleep", "nbr", "load")]["mean"]
    again = gt.gt_table(rows, "dir", rule)
    assert np.array_equal(
        [c["ci"] for c in again["cells"]], [c["ci"] for c in tab["cells"]], equal_nan=True
    )


def test_receiving_shares_top1_vs_candidate() -> None:
    eps = _episodes()
    rs = gt.receiving_shares(gt.unit_rows(eps), {e["seed"]: e["gt_static"] for e in eps})
    assert len(rs) == len(eps)
    assert all(r["pico"] == 0 and r["n_units"] == 8 for r in rs)
    for r in rs:
        assert r["cand"] == 3
        assert r["top1"] in (1, 2, 3)
        assert sum(r["shares"].values()) == pytest.approx(1.0)
        assert r["top1_is_cand"] == (r["top1"] == 3)
        assert r["pico_load_delta"] < 0


@pytest.mark.claim("DEF-EST-TAUAA")
def test_unit_delta_is_accept_minus_reject() -> None:
    reject = np.arange(2 * len(CELL_KPIS) * 3, dtype=float).reshape(2, len(CELL_KPIS), 3)
    label = {"cell": {"accept": np.zeros_like(reject), "reject": reject}}
    assert np.array_equal(gt.unit_delta(label), -reject)
    with pytest.raises(ValueError, match="per-cell"):
        gt.unit_delta({"d": {}})
