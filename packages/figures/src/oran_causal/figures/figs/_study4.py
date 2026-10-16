"""Shared names and read-only loaders for the Study 4 / bridging-analysis figures.

Paper names follow TERMINOLOGY.md s.6.1 / 6.5 / 12 (never the code names). Method styling (colour =
method, line style = family, marker = base method, one hero per figure) lives in
``oran_causal.figures.methods`` (DESIGN s.12, Q8 mapping 2026-10-06); figures pass arm codes.
"""

import gzip
import json
from collections import defaultdict

import numpy as np

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures.data import load_csv, load_json, source_note
from oran_causal.figures.paths import record

EVAL = ("study4_eval", "eval")
BRIDGE = ("bridging", "exp_b")
SRC_RECALL = source_note(*EVAL, "csv", "recall_vs_n.csv")
SRC_EVAL_MERGED = source_note(*EVAL, "merged.jsonl.gz")
SRC_REPORT = source_note(*EVAL, "report.json")
SRC_BRIDGE_TABLES = source_note(*BRIDGE, "exp_b_tables.json")
SRC_BRIDGE_MERGED = source_note(*BRIDGE, "merged.jsonl.gz")
SRC_CELLS = source_note(*EVAL, "csv", "validity_cells.csv")
WORLDS = ["E1", "E2", "E3", "E4", "E5"]
NS = [500, 1000, 4000, 8000, 24000]

# every Study 4 arm with a paper name (TERMINOLOGY s.6.5; names from oran_causal.figures.methods);
# arms without one (eq_min variants, pc_eq, mscr_eq) are never shown
PAPER = {
    a: M.name(a)
    for a in (
        "pmrt_nl_eq",
        "pmrt_eq",
        "pmrt_r3",
        "pcorr_eq",
        "granger_eq",
        "rcot2_eq",
        "corr",
        "pcorr_native",
        "pcorr_hac",
        "pcorr_hac_fb",
        "granger_native",
        "rcot2_native",
        "mscr_native",
        "pc_native",
        "notears",
        "shap_dag",
        "two_tower",
        "cdl",
    )
}
# C1's design-blind set D (REPORT s.2, report.json V4_verdicts.C1.D)
D_ARMS = ["corr", "granger_native", "mscr_native", "pcorr_hac_fb", "pcorr_native", "rcot2_native"]
# the recall figure's series (Study 4 R2 and the bridging panel): the design-based and
# design-adjusted arms
RECALL_ARMS = ["pmrt_nl_eq", "pmrt_eq", "pcorr_eq", "granger_eq"]
HERO = "pmrt_nl_eq"  # the one hero series of every Study 4 figure (emphasis only, DESIGN s.12)
KAPPA = 0.25  # Study 4 primary kappa (validity_cells / report: every primary cell)
PLACEBO = "P_placebo"  # the truth-free placebo column (Study 4 and the bridge)
SLICE_EPISODES = {
    "60": 60,
    "120": 120,
    "300": 300,
    "pooled": 600,
}  # pooled = all 600 eval episodes (EXP_B s.2)


def series(ax, x, y, lo, hi, arm: str, arms, step_pt: float = L.DODGE_PT, invalid_x=(), **kw):
    """One Study 4 series in its paper style (oran_causal.figures.methods): dodged by the arm's
    place in the figure's legend order (the same offset in every panel), CI band for design-based /
    design-adjusted arms, INVALID crosses in the strip. Returns the dodge transform.
    """
    arms = M.order(arms)
    tr = L.dodge(ax, arms.index(arm), len(arms), step_pt)
    hero = arm == HERO
    if lo is not None and M.family(arm).paper != "Design-blind methods":
        M.band(ax, x, lo, hi, arm, hero=hero, transform=tr)
    M.plot(ax, x, y, arm, hero=hero, label=M.name(arm), transform=tr, **kw)
    if len(invalid_x):
        M.invalid(ax, invalid_x, arm, transform=tr)
    return tr


def recall_r2():
    """Study 4 recall vs n, regime R2 (one row per arm x world x n; INVALID cells carry no recall)."""
    r = load_csv(*EVAL, "csv", "recall_vs_n.csv")
    return r[(r.regime == "R2") & r.arm.isin(RECALL_ARMS)].copy()


def cells():
    """Study 4 validity cells (kappa .25, every arm x world x regime x lam x n): INVALID / VALID /
    INCONCLUSIVE / NO_READ with the truth-null, placebo and confounded-null rates and CIs.
    """
    return load_csv(*EVAL, "csv", "validity_cells.csv")


def world_null_rate(c, arm: str, world: str, regimes) -> tuple[float, str] | None:
    """One environment's raw null-edge rejection rate of ``arm`` over the given regimes: the mean of
    its cells' rates (every cell of a world has the same number of edges x 40 seeds, so this is the
    pooled rate). Truth-null edges; E4 R1 / R2 have no truth-null candidate (REPORT V0), so E4 falls
    back to the placebo edge. Returns (rate, kind).
    """
    s = c[(c.arm == arm) & (c.world == world) & c.regime.isin(list(regimes))]
    if world == "E4":
        s = s[s.lam == 1.0]  # the E4 R1 / R2 cells (lam 1)
    for col, kind in (("null_raw_rate", "truth-null"), ("plac_raw_rate", "placebo")):
        v = s[col].dropna()
        if len(v) and len(v) == len(s):
            return float(v.mean()), kind
    return None


def bridge_tables() -> dict:
    return load_json(*BRIDGE, "exp_b_tables.json")


def _stream(path, keep_line, keep_rec):
    """Records of a merged .jsonl.gz whose raw line passes ``keep_line`` (cheap) and parsed record
    ``keep_rec``.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if keep_line(line):
                r = json.loads(line)
                if keep_rec(r):
                    yield r


def _arm_of(line: str) -> str:
    return line[9 : line.index("|", 9)]  # '{"key": "<arm>|...'


def placebo_p_study4(arms, regime: str = "R2") -> dict[str, np.ndarray]:
    """Placebo-edge p-values of Study 4 measure datasets (kappa .25, the given regime), per arm,
    from the EVAL merge.
    """
    out = defaultdict(list)
    tag = f"|{regime}|"
    recs = _stream(
        record(*EVAL, "merged.jsonl.gz"),
        lambda ln: _arm_of(ln) in arms and tag in ln[:90],
        lambda r: (
            r["status"] == "ok"
            and r["role"] == "measure"
            and r["job"]["regime"] == regime
            and r["job"]["kappa"] == KAPPA
        ),
    )
    for r in recs:
        out[r["arm"]] += [
            e["p"] for e in r["edges"] if e["source"] == PLACEBO and e.get("p") is not None
        ]
    return {a: np.asarray(v, float) for a, v in out.items()}


def bridge_split(seed: int) -> int:
    """v4 split of a bridge dataset (e6_bridge.dataset_seed = 60 000 000 + 10 x split + family index)."""
    return (int(seed) - 60_000_000) // 10


def placebo_p_bridge(arms, splits=range(20, 30)) -> dict[str, np.ndarray]:
    """Placebo-edge p-values of the bridge datasets in ``splits`` (default: the ten 60-episode
    slices, v4 splits 20-29), per arm, from the bridge merge.
    """
    out = defaultdict(list)
    want = set(splits)
    recs = _stream(
        record(*BRIDGE, "merged.jsonl.gz"),
        lambda ln: _arm_of(ln) in arms,
        lambda r: r["status"] == "ok" and bridge_split(r["job"]["seed"]) in want,
    )
    for r in recs:
        out[r["arm"]] += [
            e["p"] for e in r["edges"] if e["source"] == PLACEBO and e.get("p") is not None
        ]
    return {a: np.asarray(v, float) for a, v in out.items()}


def ecdf(p: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Share of p-values <= each grid value."""
    s = np.sort(p)
    return np.searchsorted(s, grid, side="right") / len(s)
