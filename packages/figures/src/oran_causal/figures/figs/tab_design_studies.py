"""Study map: Studies 1-4 and the bridging analysis, with their data, episodes / seeds, analysed units and verdict.
Booktabs LaTeX + csv; not a figure (no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_design_studies``. Kept separate
from tab_design_criteria (PAPER_PLAN s.4 [DECIDE] merge).

No protocol commit column: commit hashes appear only in the data and code availability statement (TERMINOLOGY s.8,
PAPER_PLAN s.5.1 / question 6), although the PAPER_PLAN s.4 row lists "protocol commit".

Sources and keys:
  Study 1   FD v4.slices.n_episodes.pooled, v4.gt.n_episodes, v4.pooled_units, v4.verdict; v4_analysis.json
            data.dev.n, data.placebo.n (DEV and placebo episodes; FD does not copy them).
  Study 2   FD option_a.disc.n_episodes.{dev_conf,disc,placebo,gt}.n, option_a.disc.units, option_a.disc.label,
            option_a.eval.{n,arms,run.n_jobs,verdict}.
  Study 3   FD certsafe.calib.n_dev, certsafe.eval.{n,arms,run.n_jobs,verdict}.
  Study 4   report.json tables.V1_validity (primary block: environments, regimes, n, cells, planned seeds, tuning
            seeds), tables.V11_integrity.by_role_status (measure / tune records), tables.V4_verdicts.{claim,components}.
  Bridging  exp_b_tables.json status (records per method: ok / infeasible); the logs are Study 1's evaluation and
            placebo episodes (EXP_B s.1; counts from v4_analysis.json data).
Re-aggregation: counts of arms, cells and records, the n range, and the sums of the bridging status counts.
"""

from oran_causal.figures.data import load_json
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.figs import _tabm as TM

NAME = "tab_design_studies"
SOURCES = [ST.SRC_FD, ST.SRC_V4, T.SRC_REPORT, T.SRC_BRIDGE_TABLES]
LABELS = {
    "1": "Study 1: discovery from randomised logs",
    "2": "Study 2: discovery from confounded logs and the map-driven referee",
    "3": "Study 3: edge-certified referee",
    "4": "Study 4: validity across controlled diagnostic environments (E1--E5)",
    "B": "Bridging analysis: the Study 4 methods on the Study 1 simulator logs",
}  # TERMINOLOGY s.9; data are Study 1 only (V2 B15, 2026-10-11)


def _study123() -> list[dict]:
    fd = ST.fd()
    v4, oa, cs = fd["v4"], fd["option_a"], fd["certsafe"]
    data = TM.load(ST.V4_PARTS)["data"]
    ne = oa["disc"]["n_episodes"]
    disc = oa["disc"]["label"]
    if not disc.startswith("DISC-"):
        raise AssertionError(f"unexpected discovery label {disc}")
    return [
        {
            "study": "1",
            "data": "System-level simulator, surge scenario; randomised logs with logged propensities",
            "episodes": (
                f"DEV {data['dev']['n']}; evaluation {v4['slices']['n_episodes']['pooled']}; placebo "
                f"{data['placebo']['n']}; ground truth {v4['gt']['n_episodes']}"
            ),
            "units": f"{TM.n(v4['pooled_units'])} decision units (evaluation)",
            "verdict": v4["verdict"],
        },
        {
            "study": "2",
            "data": "Same scenario; confounded logs of the incumbent policy; referee episodes",
            "episodes": (
                f"DEV {ne['dev_conf']['n']}; discovery {ne['disc']['n']}; placebo {ne['placebo']['n']}; "
                f"ground truth {ne['gt']['n']}; referee evaluation {oa['eval']['n']} seeds"
            ),
            "units": (
                f"{TM.n(oa['disc']['units'])} decision units (discovery); {len(oa['eval']['arms'])} referee arms, "
                f"{TM.n(oa['eval']['run']['n_jobs'])} simulator runs"
            ),
            "verdict": f"discovery {disc[len('DISC-') :]}; referee {oa['eval']['verdict']}",
        },
        {
            "study": "3",
            "data": "Same scenario; the Study 2 maps under the certification gate",
            "episodes": (
                f"calibration (DEV) {cs['calib']['n_dev']}; referee evaluation {cs['eval']['n']} seeds"
            ),
            "units": (
                f"{len(cs['eval']['arms'])} referee arms, {TM.n(cs['eval']['run']['n_jobs'])} simulator runs"
            ),
            "verdict": cs["eval"]["verdict"],
        },
    ]


def _study4() -> dict:
    tab = load_json(*T.EVAL, "report.json")["tables"]
    p = [r for r in tab["V1_validity"] if r["block"] == "primary"]
    if {r["kappa"] for r in p} != {tab["primary_kappa"]}:
        raise AssertionError("primary block mixes kappa values")
    worlds = sorted({r["world"] for r in p})
    regimes = sorted({r["regime"] for r in p})
    ns = sorted({r["n"] for r in p})
    cells = {(r["world"], r["regime"], r["lam"], r["n"]) for r in p}
    seeds = sorted({r["planned_seeds"] for r in p})
    big = {(r["world"], r["regime"]) for r in p if r["planned_seeds"] != seeds[0]}
    tune = {r["n_tune"] for r in p}
    if len(seeds) != 2 or big != {("E4", "R3")} or len(tune) != 1:
        raise AssertionError(f"seed layout changed: {seeds}, {big}, {tune}")
    st = tab["V11_integrity"]["by_role_status"]
    v = tab["V4_verdicts"]
    comps = ", ".join(v["components"])
    if set(v["components"].values()) != {v["claim"]}:
        raise AssertionError("components differ from the claim verdict; list them")
    return {
        "study": "4",
        "data": (
            f"Controlled diagnostic environments {worlds[0]}--{worlds[-1]}; regimes {regimes[0]}--"
            f"{regimes[-1]}; {TM.n(ns[0])}--{TM.n(ns[-1])} rows per dataset"
        ),
        "episodes": (
            f"{seeds[0]} seeds per method and condition ({seeds[1]} for some methods in E4 R3); "
            f"{tune.pop()} tuning seeds per condition"
        ),
        "units": (
            f"{len(cells)} conditions; {TM.n(st['measure:ok'])} method-dataset analyses (plus "
            f"{TM.n(st['tune:ok'])} for tuning)"
        ),
        "verdict": f"{v['claim']} ({comps})",
    }


def _bridge() -> dict:
    status = T.bridge_tables()["status"]
    total = sum(sum(s.values()) for s in status.values())
    infeasible = sum(s.get("infeasible", 0) for s in status.values())
    data = TM.load(ST.V4_PARTS)["data"]
    return {
        "study": "B",
        "data": "Study 1 logs, re-analysed with the Study 4 methods",
        "episodes": f"Study 1 evaluation {data['eval']['n']} and placebo {data['placebo']['n']} (no new episodes)",
        "units": f"{TM.n(total)} method-dataset analyses ({infeasible} infeasible)",
        "verdict": "none (post hoc, descriptive)",
    }


def rows() -> list[dict]:
    out = [*_study123(), _study4(), _bridge()]
    for r in out:
        r["label"] = LABELS[r["study"]]
    return out


def latex(rs: list[dict]) -> list[str]:
    out = TM.header(
        NAME,
        SOURCES,
        [
            "No protocol commit column: commit hashes only in the data and code availability "
            "statement (TERMINOLOGY s.8)."
        ],
    )
    out += [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}p{2.7cm} p{2.4cm} p{2.6cm} p{2.3cm} p{1.9cm}@{}}",
        r"\toprule",
        r"Study & Data & Episodes / seeds & Analysed & Verdict \tabularnewline",
        r"\midrule",
    ]
    for i, r in enumerate(rs):
        if r["study"] == "B":
            out.append(r"\midrule")
        out.append(
            rf"\raggedright {r['label']} & \raggedright {r['data']} & \raggedright {r['episodes']} & "
            rf"\raggedright {r['units']} & \raggedright {r['verdict']} \tabularnewline"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


def write() -> list[str]:
    rs = rows()
    return TM.write(NAME, latex(rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)
