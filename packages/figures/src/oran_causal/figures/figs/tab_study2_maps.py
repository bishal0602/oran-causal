"""Study 2 map quality per map (table form of fig_study2_maps): declared edges of each referee map
scored against the knockout ground truth of the confounded logs (TRUE with the right / wrong sign,
NULL, INDET) and the edges the same pipeline declared on placebo logs. Booktabs LaTeX + csv; not a
figure (no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_study2_maps``.

Source: FD ``option_a.map_quality.<arm>`` (edges, true_right, true_wrong, null, indet, unknown,
placebo_edges), shown as stored. ``blanket2`` has no paper name and is left out; ``unknown`` is 0
for every map (checked) and not shown; the oracle and random maps have no placebo map (stored null,
shown as --). Map quality was not a criterion fixed before evaluation: the table is descriptive.
Every row's columns are checked to add up (true_right + true_wrong + null + indet + unknown =
edges). Rows in the paper-wide method order (oran_causal.figures.methods.order); a map's DEV-tuned
threshold before its placebo-calibrated one.
"""

from oran_causal.figures import methods as M
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _tabm as TM

NAME = "tab_study2_maps"
SOURCES = [ST.SRC_FD]
SKIP = {"blanket2"}  # no paper name
COLS = [
    ("true_right", "TRUE, right sign"),
    ("true_wrong", "TRUE, wrong sign"),
    ("null", "NULL"),
    ("indet", "INDET"),
    ("placebo_edges", "Placebo edges"),
]
FAMILY_LABEL = {
    "based": "Design-based test",
    "blind_test": "Design-blind methods",
    "blind_learned": "Design-blind methods",
    "oracle": "Reference maps",
    "random": "Reference maps",
}


def rows() -> list[dict]:
    mq = ST.fd()["option_a"]["map_quality"]
    codes = [c for c in mq if c not in SKIP]
    codes = M.order(
        sorted(
            codes, key=lambda c: (c.split("@")[0], {"dev": 0, "plc": 1}.get(c.partition("@")[2], 0))
        )
    )
    out = []
    for c in codes:
        q = mq[c]
        if q["unknown"] != 0:
            raise AssertionError(f"{c}: {q['unknown']} unknown-status edges; add the column")
        if (
            sum(q[k] for k in ("true_right", "true_wrong", "null", "indet", "unknown"))
            != q["edges"]
        ):
            raise AssertionError(f"{c}: status columns do not add up to {q['edges']} edges")
        arm = M.resolve(c)
        out.append(
            {
                "arm": c,
                "map": TM.map_name(c),
                "group": FAMILY_LABEL[arm.method.family],
                "edges": q["edges"],
                **{k: q[k] for k, _ in COLS},
            }
        )
    return out


def _cell(v) -> str:
    return "--" if v is None else str(v)


def latex(rs: list[dict]) -> list[str]:
    out = TM.header(
        NAME,
        SOURCES,
        [
            "Edges of each referee map vs the knockout ground truth; placebo edges = edges the "
            "same pipeline declared on placebo logs; -- = no placebo map."
        ],
    )
    out += [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l r r r r r r@{}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{TRUE} & & & Placebo \\",
        r"\cmidrule(lr){3-4}",
        r"Map & Edges & right sign & wrong sign & NULL & INDET & edges \\",
        r"\midrule",
    ]
    group = None
    for r in rs:
        if r["group"] != group:
            if group is not None:
                out.append(r"\midrule")
            group = r["group"]
            out.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{group}}}}} \\")
        cells = " & ".join(_cell(r[k]) for k, _ in COLS)
        out.append(rf"\quad {TM.tex(r['map'])} & {r['edges']} & {cells} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


def write() -> list[str]:
    rs = rows()
    return TM.write(NAME, latex(rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)
