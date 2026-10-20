"""Check X1, runtime (Appendix C.1; booktabs LaTeX + csv): a timing run specified in the frozen
Study 4 protocol (R-55), not post hoc. Median / max CPU seconds per (method, data set) by n on
one uncontended Kaggle reference host, single thread, data generation excluded. Not a figure (no
SPEC): run ``uv run python -m oran_causal.figures.figs.tab_appC_runtime``.

Source: exp_c_runtime.json (the json behind EXP_C_RUNTIME.md; X1 timing run),
key ``table``: one row per arm x n with ``cost_ref.{median, max}`` (Kaggle-reference CPU-s),
``n_ok`` / ``n_units`` and ``hosts``; ``budget_cpu_s``, ``ref_host``. Rows only select stored
values. Of the 24 timed arms, the 18 with a paper name are shown; the eq_min variants, pc_eq and
mscr_eq (no paper name) are left out. MSCR runs at n <= 1000 only (not in its grid above). Every
unit ran on the reference host (no speed factor applied; checked).
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_runtime"
RT = ("posthoc_x1", "exp_c", "paper", "exp_c_runtime.json")
SOURCES = [source_note(*RT)]


def rows() -> tuple[list[dict], dict]:
    t = load_json(*RT)
    if t["factors"] or len(t["hosts"]) != 1 or t["hosts"][0]["host"] != t["ref_host"]:
        raise AssertionError("runtime table is not single-host on the reference host")
    out = []
    for r in t["table"]:
        if not A.has_name(r["arm"]):
            continue
        out.append(
            {
                "arm": r["arm"],
                "method": A.M.name(r["arm"]),
                "n": r["n"],
                "units": r["n_units"],
                "ok": r["n_ok"],
                "median_cpu_s": r["cost_ref"]["median"],
                "max_cpu_s": r["cost_ref"]["max"],
            }
        )
    timed = sorted({r["arm"] for r in t["table"]})
    return out, {
        "n_grid": t["n_grid"],
        "budget": t["budget_cpu_s"],
        "timed": timed,
        "ref_host": t["ref_host"],
    }


def latex(rs: list[dict], meta: dict) -> str:
    ns = meta["n_grid"]
    by = {(r["arm"], r["n"]): r for r in rs}
    shown = sorted({r["arm"] for r in rs})
    left = [a for a in meta["timed"] if a not in shown]
    units = sorted({r["units"] for r in rs})
    out = A.header(
        NAME,
        SOURCES,
        [
            "Check X1 (runtime): timing run specified in the Study 4 protocol; descriptive, no verdict. Median / max CPU-s per (method, data set), Kaggle reference host"
            f" ({meta['ref_host']}), single thread, data generation excluded; {'/'.join(map(str, units))} units per method and n;"
            f" budget {meta['budget']:.0f} CPU-s per unit; -- = not in the method's grid.",
            f"Shown: {len(shown)} of {len(meta['timed'])} timed arms; left out (no paper name): {', '.join(left)}.",
        ],
        module=NAME,
    )
    ncol = 1 + len(ns)
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l" + " r" * len(ns) + "@{}}",
        r"\toprule",
        rf" & \multicolumn{{{len(ns)}}}{{c}}{{Median / max CPU-s per data set}} \\",
        rf"\cmidrule(lr){{2-{ncol}}}",
        "Method & " + " & ".join(rf"$n = {A.thousands(n)}$" for n in ns) + r" \\",
    ]
    for fam, codes in A.blocks(shown):
        out += [r"\midrule", A.group_row(fam, ncol)]
        for code in codes:
            cells = []
            for n in ns:
                r = by.get((code, n))
                cells.append(
                    A.DASH if r is None else f"{A.sig3(r['median_cpu_s'])}/{A.sig3(r['max_cpu_s'])}"
                )
            out.append(rf"\quad {A.short(code)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, meta = rows()
    return A.write(NAME, latex(rs, meta), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)
