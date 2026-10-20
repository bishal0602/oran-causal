"""Shared sources for the Studies 1-3 figures and tables (system-level simulator; record sets
read-only).

FD = the report's asserted figure data (55 assertions against the analyzer JSONs); the decision
JSONs are the analyzer outputs it was built from. Every number a figure shows is read from one of
these files at build time.
"""

from oran_causal.figures.data import load_json, source_note

REPORT = ("system_artifacts", "report_2026-10-02")
FD_PARTS = (*REPORT, "figures_data.json")
DECISION = ("system_artifacts", "decision")
V4_PARTS = (*DECISION, "v4_analysis.json")  # Study 1 analyzer output (GT cells, slices, placebo)
CONF_PARTS = (*DECISION, "conf_disc_conf.json")  # Study 2 discovery analyzer output
CONF_EVAL_PARTS = (*DECISION, "conf_eval.json")  # Study 2 referee evaluation
CS_EVAL_PARTS = (*DECISION, "cs_eval.json")  # Study 3 referee evaluation

SRC_FD = source_note(*FD_PARTS)
SRC_V4 = source_note(*V4_PARTS)
SRC_CONF = source_note(*CONF_PARTS)
SRC_CONF_EVAL = source_note(*CONF_EVAL_PARTS)
SRC_CS_EVAL = source_note(*CS_EVAL_PARTS)

_cache: dict = {}


def fd() -> dict:
    if "fd" not in _cache:
        _cache["fd"] = load_json(*FD_PARTS)
    return _cache["fd"]


def v4_gt_cells() -> dict[tuple[str, str, str], dict]:
    """Study 1 knockout ground truth: (family, relation, kpi) -> cell (status TRUE / NULL / INDET,
    sign of the effect of a +1 parameter step, mean, CI, n).
    """
    if "gt" not in _cache:
        cells = load_json(*V4_PARTS)["gt"]["cells"]
        _cache["gt"] = {(c["family"], c["relation"], c["kpi"]): c for c in cells}
    return _cache["gt"]
