import json
from pathlib import Path

import pytest

from oran_causal.analysis import paper_derived as pd
from oran_causal.analysis import paper_values as pv

REPO = Path(__file__).resolve().parents[3]

CELLS = {
    "a": {"material": "NULL", "status": "EXISTS"},
    "b": {"material": "NULL", "status": "NOT_ESTABLISHED"},
    "c": {"material": "TRUE", "status": "EXISTS"},
    "d": {"material": "NULL", "status": "NOT_ESTABLISHED"},
}


def _slice(p_a: float, p_b: float, p_d: float) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    return {
        "hyp": {name: {"primary": {"p": p}} for name, p in (("a", p_a), ("b", p_b), ("d", p_d))}
    }


SLICES = {
    "pooled": _slice(0.0, 0.0, 0.0),
    "s60_0": _slice(0.01, 0.5, 0.04),
    "s60_1": _slice(0.5, 0.5, 0.05),
    "s120_0": _slice(0.0, 0.0, 0.0),
}


def test_null_counts_follow_the_labels_and_the_interval() -> None:
    assert pd.null_hypothesis_count(CELLS) == 3
    assert pd.unestablished_null_count(CELLS) == 2


def test_null_rejection_rates_use_only_the_60_episode_slices() -> None:
    assert pd.null_rejection_rate(CELLS, SLICES) == pytest.approx(3 / 6)
    assert pd.unestablished_null_rejection_rate(CELLS, SLICES) == pytest.approx(2 / 4)


def test_derived_functions_are_available_to_value_expressions() -> None:
    names = {"cells": CELLS, "slices": SLICES}
    assert pv.evaluate("null_hypothesis_count(cells)", names) == 3
    assert pv.evaluate("min(flat(a, b))", {"a": [[0.4, [0.2]]], "b": 0.3}) == 0.2


def test_subtree_arguments_are_pinned_without_a_stored_value(tmp_path: Path) -> None:
    sidecar = json.loads((REPO / "paper" / "generated" / "values.json").read_text("utf-8"))
    argument = sidecar["values"]["s1.null.rate"]["args"]["c"]
    assert set(argument) == {"record", "sha256", "pointer"}


PRINTED = {
    "s1.null.count": "21",
    "s1.null.open.count": "12",
    "s1.null.rate": "0.15",
    "s1.null.open.rate": "0.075",
    "s1.attr.declared": "13",
    "s1.attr.true": "12",
    "s2.attr.declared": "7",
    "s2.plc.rec_max": "0.21",
    "s4.conf.corr_min": "1.00",
    "s4.conf.gbm_min": "0.02",
    "s4.conf.gbm_max": "0.06",
    "s4.conf.corr_all": "0.77",
    "s4.conf.rcot2_all": "0.23",
    "s4.conf.pcorr_all": "0.056",
    "s4.dither.min": "0.042",
    "s4.dither.max": "0.054",
    "s4.cdl.invalid": "9",
    "s4.cdl.conditions": "30",
    "x11.delta_half.changed": "10",
    "x11.delta_dbl.changed": "16",
    "x11.other.changed_max": "2",
    "x11.pmrt.sign_acc_min": "1.00",
    "x14.gby.wrong_s1": "24",
    "x14.gby.wrong_s2": "21",
    "x15.p.noarb": "0.15",
    "x15.p.never_sleep": "0.17",
    "x15.r.never_sleep_s2": "0.22",
    "x15.r.never_sleep_s3": "0.26",
    "x15.rej.s2": "0.37",
    "x15.rej.s3": "0.35",
    "x15.sd.s2": "0.85",
    "x15.sd.s3": "1.1",
    "x15.block_diff": "0.30",
    "x15.perm_p": "0.009",
    "x15.trim.s2": "0.21",
    "x15.trim.s3": "0.041",
    "s3.gate.g_v": r"19\,953.65",
    "s3.gate.g_rlf": "35.025",
}


@pytest.mark.parametrize(("key", "text"), sorted(PRINTED.items()))
def test_traced_values_equal_the_numbers_the_manuscript_prints(key: str, text: str) -> None:
    tex = (REPO / "paper" / "generated" / "values.tex").read_text(encoding="utf-8")
    assert rf"\defval{{{key}}}{{{text}}}" in tex.splitlines()
