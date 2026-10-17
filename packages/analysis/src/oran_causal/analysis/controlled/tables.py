"""The Study 4 result tables V0 to V11 as JSON-ready dicts, and their markdown rendering.

V0 like-for-like headline, V1 validity, V2 recall, V3 paired recall difference, V4 verdicts,
V5 information levels in R2, V6 E4, V7 not testable, V8 secondary KPI family, V9 kappa
sweep, V10 cost, V11 integrity. Everything is a pure function of the cell summaries.
"""

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.controlled.cells import Cell, Cells, Rows, counts, powered
from oran_causal.analysis.controlled.power import paired_diffs
from oran_causal.analysis.controlled.protocol import (
    ALPHA,
    ANALYSIS_VERSION,
    BOOT_REPS,
    BOOT_SEED,
    COUNTED,
    FMAX_NSIM,
    FMAX_SEED,
    FOCAL,
    MIN_FLAG_SEEDS,
    PRIMARY_KAPPA,
    RATE_KEYS,
    SWEEP_KAPPAS,
    SWEEP_N,
    VALID_UB,
)
from oran_causal.analysis.controlled.spec import (
    Spec,
    arm_kind,
    cost_field,
    focal_arm,
    native_partner,
    not_in_grid,
)
from oran_causal.analysis.controlled.verdicts import Verdict, rate_text, verdicts
from oran_causal.analysis.records import PathLike, Record, status_of, unit_of

Table = list[dict[str, Any]]
Tables = dict[str, Any]
STATE_TAG = {
    "INVALID": "inv",
    "NO_READ": "nr",
    "infeasible": "inf",
    "infeasible_t3": "inf",
    "untuned": "unt",
    "missing": "mis",
    "few_seeds": "few",
}
RATE_MARK = {"VALID": "", "INVALID": "*", "INCONCLUSIVE": "~", None: "?"}
CELL_MARK = {"VALID": "", "INVALID": "*", "INCONCLUSIVE": "~", "NO_READ": "~", "few_seeds": "?"}


def brief(x: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduces a rate to its rate, interval, count and validity."""
    return (
        None
        if not x
        else {"rate": x["rate"], "ci": x["ci"], "n": x["n"], "validity": x["validity"]}
    )


def cell_row(key: str, cell: Cell) -> dict[str, Any]:
    """Returns the flat V1 row of a cell."""
    primary = cell.get("primary") or {}
    fields = (
        "arm",
        "kind",
        "block",
        "declare",
        "world",
        "regime",
        "lam",
        "n",
        "kappa",
        "status",
        "tau",
        "tau_is_pos_inf",
        "n_tune",
        "n_measure",
        "planned_seeds",
        "n_infeasible",
        "n_error",
    )
    row = {"key": key, **{f: cell.get(f) for f in fields}}
    row["validity"] = primary.get("validity")
    row["n_seeds"] = primary.get("n_seeds")
    for rate_key in RATE_KEYS:
        row[rate_key] = brief(primary.get(rate_key))
    row["power_not_applicable"] = bool(cell.get("power_not_applicable"))
    for f in ("infeasible_cost_cpu_s", "infeasible_cost_wall_s"):
        if cell.get(f) is not None:
            row[f] = cell[f]
    return row


def _like_rules(cell: Cell) -> list[tuple[str, dict[str, Any] | None]]:
    if cell["declare"] == "by":
        return [("raw_p", cell.get("like_raw")), ("tau", cell.get("secondary_tau"))]
    rules = [("tau", cell.get("primary"))]
    if "like_fixed" in cell:
        rules.append(("fixed", cell.get("like_fixed")))
    return rules


def like_for_like(cells: Cells, focal: str = FOCAL) -> Table:
    """V0, the headline table: every arm per cell under a common scoring, side by side.

    P arms appear twice: raw p <= .05 per edge (`raw_p`) and with the conformal placebo tau
    (`tau`) of the score-only arms; score-only arms with their tau, and an arm with a spec
    `fixed_threshold` also at that threshold (`fixed`, secondary). Per row: mean per-seed
    recall (None where power is not applicable), truth-null and placebo declaration rates
    with interval and three-way validity. The placebo is the tuning column of a tau scoring.
    """
    out = []
    for key, cell in sorted(cells.items()):
        for rule, summary in _like_rules(cell):
            ok = cell["status"] in COUNTED and bool(summary) and summary["validity"] != "few_seeds"
            state = (
                cell["status"]
                if cell["status"] not in COUNTED
                else "untuned"
                if not summary
                else summary["validity"]
            )
            s = summary or {}
            out.append(
                {
                    "key": key,
                    "cell": key.split("|", 1)[1],
                    "arm": cell["arm"],
                    "focal": cell["arm"] == focal,
                    "kind": cell["kind"],
                    "block": cell["block"],
                    "declare": cell["declare"],
                    "rule": rule,
                    **{f: cell[f] for f in ("world", "regime", "lam", "n", "kappa", "status")},
                    "state": state,
                    "n_seeds": s.get("n_seeds", 0),
                    "power_not_applicable": bool(cell.get("power_not_applicable")),
                    "recall": s.get("recall") if ok else None,
                    "null": brief(s.get("null_decl")) if ok else None,
                    "placebo": brief(s.get("plac_decl")) if ok else None,
                    "placebo_conf": brief(s.get("conf_decl")) if ok else None,
                    "placebo_is_tuning_column": rule == "tau",
                }
            )
    return out


def cost_table(records: list[Record], cells: Cells, spec: Spec) -> Table:
    """V10 per `(arm, world, regime, n)`.

    `budget` is the arm's budget field: GPU arms are budgeted and read in wall seconds,
    reported apart from the CPU seconds of the CPU arms.
    """
    groups: dict[tuple, dict[str, Any]] = defaultdict(
        lambda: {"cpu": [], "wall": [], "rss": [], "gpu": [], "infeasible": [], "errors": 0}
    )
    for record in records:
        u = unit_of(record)
        g = groups[(u["arm"], u["world"], u["regime"], u["n"])]
        status = status_of(record)
        if status == "ok":
            _add_costs(g, record)
        elif status == "infeasible":
            g["infeasible"].append(
                {
                    "seed": u["seed"],
                    "cpu_s": record.get("cpu_s"),
                    "wall_s": record.get("wall_s"),
                    "reason": record.get("reason"),
                }
            )
        else:
            g["errors"] += 1
    arms = spec["arms"]
    out = [_cost_row(k, v, arms) for k, v in sorted(groups.items()) if k[0] in arms]
    t3 = {
        (c["arm"], c["world"], c["regime"], c["n"]): c
        for c in cells.values()
        if c["status"] == "infeasible_t3"
    }
    out += [_t3_cost_row(k, c, arms) for k, c in sorted(t3.items())]
    return out


def _wall(record: Record) -> float | None:
    x = (
        record.get("child_wall_s")
        if record.get("child_wall_s") is not None
        else record.get("wall_s")
    )
    return None if x is None else float(x)


def _add_costs(group: dict[str, Any], record: Record) -> None:
    for field, dst in (("cpu_s", "cpu"), ("peak_rss_mb", "rss"), ("gpu_s", "gpu")):
        if record.get(field) is not None:
            group[dst].append(float(record[field]))
    wall = _wall(record)
    if wall is not None:
        group["wall"].append(wall)


def _cost_row(key: tuple, v: dict[str, Any], arms: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": key[0],
        "world": key[1],
        "regime": key[2],
        "n": key[3],
        "budget": cost_field(arms[key[0]]),
        "n_runs": len(v["cpu"]),
        "cpu_s_mean": float(np.mean(v["cpu"])) if v["cpu"] else None,
        "cpu_s_max": max(v["cpu"]) if v["cpu"] else None,
        "wall_s_mean": float(np.mean(v["wall"])) if v["wall"] else None,
        "wall_s_max": max(v["wall"]) if v["wall"] else None,
        "peak_rss_mb_max": max(v["rss"]) if v["rss"] else None,
        "gpu_s_mean": float(np.mean(v["gpu"])) if v["gpu"] else None,
        "n_infeasible": len(v["infeasible"]),
        "infeasible": v["infeasible"][:5],
        "n_errors": v["errors"],
        "t3": False,
    }


def _t3_cost_row(key: tuple, cell: Cell, arms: dict[str, Any]) -> dict[str, Any]:
    field = cost_field(arms[key[0]])
    return {
        "arm": key[0],
        "world": key[1],
        "regime": key[2],
        "n": key[3],
        "budget": field,
        "n_runs": 0,
        "cpu_s_mean": None,
        "cpu_s_max": None,
        "wall_s_mean": None,
        "wall_s_max": None,
        "peak_rss_mb_max": None,
        "gpu_s_mean": None,
        "n_infeasible": 0,
        "infeasible": [],
        "n_errors": 0,
        "t3": True,
        f"t3_dev_cost_{field}": cell.get(f"infeasible_cost_{field}"),
    }


def _v2(prim: Cells) -> Table:
    out = []
    for key, cell in sorted(prim.items()):
        p = cell.get("primary") or {}
        state = cell["status"] if cell["status"] not in COUNTED else (p.get("validity") or "none")
        ok = powered(cell)
        out.append(
            {
                "key": key,
                "arm": cell["arm"],
                "block": cell["block"],
                "world": cell["world"],
                "regime": cell["regime"],
                "lam": cell["lam"],
                "n": cell["n"],
                "state": state,
                "power_not_applicable": bool(cell.get("power_not_applicable")),
                "recall": p.get("recall") if ok else None,
                "fdp": p.get("fdp") if ok else None,
                "sign_acc": p.get("sign_acc") if ok else None,
            }
        )
    return out


def _v5(prim: Cells, spec: Spec) -> Table:
    with_info = {
        native_partner(a, spec) for a, d in spec["arms"].items() if arm_kind(d) in ("eq", "eq_min")
    }
    return [
        cell_row(k, e) | {"recall": (e.get("primary") or {}).get("recall")}
        for k, e in sorted(prim.items())
        if e["regime"] == "R2" and (e["kind"] in ("pmrt", "eq", "eq_min") or e["arm"] in with_info)
    ]


def _v6(prim: Cells) -> Table:
    out = []
    for key, cell in sorted(prim.items()):
        if cell["world"] != "E4":
            continue
        p = cell.get("primary") or {}
        out.append(
            cell_row(key, cell)
            | {
                "recall": p.get("recall"),
                "sign_acc": p.get("sign_acc"),
                "wrong_sign_rate": p.get("wrong_sign_rate"),
            }
        )
    return out


def _v7(cells: Cells) -> Table:
    return [
        {
            "key": k,
            "nt_true": e["primary"]["nt_true"],
            "nt_null": e["primary"]["nt_null"],
            "n_not_applicable": e["primary"]["n_not_applicable"],
            "n_seeds": e["primary"]["n_seeds"],
        }
        for k, e in sorted(cells.items())
        if e.get("primary")
        and (e["primary"]["nt_true"] or e["primary"]["nt_null"] or e["primary"]["n_not_applicable"])
    ]


def _v8(prim: Cells) -> Table:
    return [
        {
            "key": k,
            "state": e["primary"]["validity"] if counts(e) else e["status"],
            **e["primary"]["secondary"],
        }
        for k, e in sorted(prim.items())
        if e.get("primary") and e["primary"]["secondary"]["recall"] is not None
    ]


def _v9(cells: Cells) -> Table:
    return [
        {
            "key": k,
            "arm": e["arm"],
            "world": e["world"],
            "regime": e["regime"],
            "lam": e["lam"],
            "n": e["n"],
            "kappa": e["kappa"],
            "status": e["status"],
            "validity": (e.get("primary") or {}).get("validity"),
            "recall": (e.get("primary") or {}).get("recall"),
        }
        for k, e in sorted(cells.items())
        if e["regime"] == "R2" and e["n"] == SWEEP_N and e["kappa"] in SWEEP_KAPPAS
    ]


def assemble(
    cells: Cells,
    rows: Rows,
    records: list[Record],
    spec: Spec,
    *,
    dep: dict[str, Any] | None,
    v11: dict[str, Any],
) -> Tables:
    """Builds every table V0 to V10 from the cells and rows, with the given V11.

    Args:
        cells: Cell summaries of all arms.
        rows: Per-seed rows by cell.
        records: Screened records (V10 reads only their status and cost fields).
        spec: Campaign spec.
        dep: Dependence table for `F_max`, or None.
        v11: Integrity result.

    Returns:
        The tables with the analysis constants.
    """
    focal = focal_arm(spec)
    prim = {k: e for k, e in cells.items() if e["kappa"] == PRIMARY_KAPPA}
    return {
        "analysis": ANALYSIS_VERSION,
        "spec_name": spec.get("name"),
        "focal": focal,
        "alpha": ALPHA,
        "valid_ub": VALID_UB,
        "boot_reps": BOOT_REPS,
        "boot_seed": BOOT_SEED,
        "min_flag_seeds": MIN_FLAG_SEEDS,
        "primary_kappa": PRIMARY_KAPPA,
        "fmax_nsim": FMAX_NSIM,
        "fmax_seed": FMAX_SEED,
        "V0_like_for_like": like_for_like(prim, focal),
        "V1_validity": [cell_row(k, e) for k, e in sorted(prim.items())],
        "V2_recall": _v2(prim),
        "V3_paired": paired_diffs(prim, focal=focal),
        "V3_paired_valid_only": paired_diffs(prim, "VALID", focal),
        "V4_verdicts": verdicts(cells, rows, spec, dep),
        "V5_information_R2": _v5(prim, spec),
        "V6_E4": _v6(prim),
        "V7_not_testable": _v7(cells),
        "V8_secondary": _v8(prim),
        "V9_kappa_sweep": _v9(cells),
        "V10_cost": cost_table(records, cells, spec),
        "V11_integrity": v11,
        "secondary_tau_of_p_arms": {
            k: e["secondary_tau"]["validity"]
            for k, e in sorted(prim.items())
            if e.get("secondary_tau")
        },
        "not_in_grid": not_in_grid(spec),
    }


def fmt(x: Any, digits: int = 2) -> str:
    """Formats a number without the leading zero below 1; `-` for None and NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    s = f"{x:.{digits}f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def world_regime(e: dict[str, Any]) -> str:
    """Returns the `E4 R3 l1` style label of a cell's world, regime and lambda."""
    return f"{e['world']} {e['regime']}" + ("" if e["lam"] is None else f" l{e['lam']:g}")


def arm_order(out: Tables) -> list[str]:
    """Returns the arms in table order: the focal arm, then PMRT arms, then by name."""
    seen: list[str] = []
    for r in out["V1_validity"]:
        if r["arm"] not in seen:
            seen.append(r["arm"])
    focal = out.get("focal") or FOCAL
    return sorted(seen, key=lambda a: (a != focal, "pmrt" not in a, a))


def _pooled_text(pooled: dict[str, Any]) -> str:
    return "; ".join(
        f"{k} {fmt(p['rate'], 3)} [{fmt(p['ci'][0], 3)}, {fmt(p['ci'][1], 3)}] {p['validity']}"
        for k, p in pooled.items()
        if p
    )


def nig_index(out: Tables) -> set[tuple]:
    """Returns `(arm, world-regime label, n, kappa)` of the cells outside an arm's grid."""
    return {
        (c["arm"], world_regime(c), c["n"], c["kappa"])
        for c in (out.get("not_in_grid") or {}).get("cells", [])
    }


def _rate_cell(x: dict[str, Any] | None) -> str:
    return "-" if not x else fmt(x["rate"], 3) + RATE_MARK.get(x["validity"], "?")


def _like_cell(r: dict[str, Any] | None) -> str:
    if r is None:
        return "-"
    if r["recall"] is None and r["null"] is None and r["placebo"] is None:
        return STATE_TAG.get(r["state"], r["state"])
    recall = "NA" if r["power_not_applicable"] else fmt((r["recall"] or {}).get("mean"))
    return f"{recall} / {_rate_cell(r['null'])} / {_rate_cell(r['placebo'])}"


LIKE_INTRO = (
    "Every p arm scored at raw p <= .05 per edge (raw_p) and with the conformal placebo tau (tau), "
    "next to the "
    "score-only arms (tau; cdl also at the conference's fixed threshold, fixed). Rates are "
    "declaration rates; * "
    "INVALID, ~ INCONCLUSIVE; a tau row's placebo is its tuning column (reported, not in its "
    "validity). NA = not "
    "applicable (PMRT in R4: no known design, never recall 0); inv INVALID, inf infeasible (EVAL "
    "or T3), mis "
    "missing, unt untuned, few < 10 seeds; nig not in the arm's grid (pre-registered grid choice, "
    "R-54 / R-58)."
)


def md_like(rows: Table, arms: list[str], nig: set | None = None) -> list[str]:
    """V0 markdown: one table per `(world, regime, lambda)`, rows arm x rule, columns n."""
    nig = nig or set()
    lines = [
        "## V0 like-for-like (headline, R-42): recall / truth-null rate / placebo rate per cell",
        "",
        LIKE_INTRO,
    ]
    ns = sorted({r["n"] for r in rows})
    index = {(world_regime(r), r["arm"], r["rule"], r["n"]): r for r in rows}
    order = sorted(
        {
            (r["world"], r["regime"], -1.0 if r["lam"] is None else r["lam"], world_regime(r))
            for r in rows
        }
    )
    for *_, label in order:
        lines += [
            "",
            f"### {label}",
            "",
            "| arm | rule | " + " | ".join(f"n {n}" for n in ns) + " |",
            "|---|---|" + "---|" * len(ns),
        ]
        for arm in arms:
            for rule in ("raw_p", "tau", "fixed"):
                present = [(label, arm, rule, n) in index for n in ns]
                is_nig = [(arm, label, n, PRIMARY_KAPPA) in nig for n in ns]
                if not any(present):
                    if rule == "raw_p" and any(is_nig):
                        lines.append(
                            f"| {arm} | - | "
                            + " | ".join("nig" if g else "-" for g in is_nig)
                            + " |"
                        )
                    continue
                cells = [
                    "nig" if not pres and g else _like_cell(index.get((label, arm, rule, n)))
                    for pres, g, n in zip(present, is_nig, ns, strict=True)
                ]
                lines.append(f"| {arm} | {rule} | " + " | ".join(cells) + " |")
    return lines


def _md_c1(v4: Verdict, tag: Callable[[str], str]) -> list[str]:
    c1 = v4["C1"]
    lines = [
        "## V4 claim verdicts",
        "",
        f"- **Claim: {v4['claim']}** (components {v4['components']})",
        f"- C1: **{c1['verdict']}** ({c1['n_failures']} / {len(c1['D_counted'])} counted arms of D "
        f"are "
        f"design-blind failures; |D| = {len(c1['D'])}). Wording: {c1['wording']}.",
    ]
    if c1.get("sensitivity_without"):
        sw = c1["sensitivity_without"]
        lines.append(
            f"- C1 sensitivity without {', '.join(sw['dropped'])} (R-56, no effect): "
            f"**{sw['verdict']}** "
            f"({sw['n_failures']} / {len(sw['D_counted'])} counted arms fail)"
        )
    lines += [
        f"- Arm labels (R-40; every table): {a}: {x}" for a, x in v4.get("arm_labels", {}).items()
    ]
    for a, v in [*c1["arms"].items(), *c1.get("descriptive", {}).items()]:
        nd = " (not in D, descriptive)" if a not in c1["arms"] else ""
        lines.append(
            f"  - {a}{tag(a)}{nd}: **{v['verdict']}**; R2 INVALID "
            f"{v['r2_invalid']}/{v['r2_planned']} "
            f"(counted {v['r2_counted']}), pooled R2 {_pooled_text(v['pooled_r2'])}; R1 INVALID "
            f"{v['r1_invalid']}/{v['r1_counted']} (F_max {v['r1_f_max']})"
        )
    return lines


def _md_component(name: str, x: Verdict, indent: str = "") -> list[str]:
    lines = [f"{indent}- {name}: **{x['verdict']}**"]
    if "n_cells" in x:
        not_counted = f"; not counted {len(x['not_counted'])}" if x["not_counted"] else ""
        lines.append(
            f"{indent}  - cells {x['n_cells']}/{x['n_planned']}, INVALID {x['n_invalid']} (F_max "
            f"{x['f_max']}), VALID {x['n_valid']}, INCONCLUSIVE {x['n_inconclusive']}; pooled "
            f"{_pooled_text(x['pooled'])}{not_counted}"
        )
    return lines


def _md_eq_table(v4: Verdict, tag: Callable[[str], str]) -> list[str]:
    lines = [
        "",
        "Every eq arm, same rule on R1 + R2, unfiltered (R-56):",
        "",
        "| eq arm | C2b membership | native C1 | verdict | INVALID / cells (F_max) | R2 INVALID / "
        "planned | "
        "pooled R1 truth-null | pooled R2 truth-null |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a, x in v4["eq_arms_table"].items():
        r2 = "-" if x["r2_invalid"] is None else f"{x['r2_invalid']}/{x['r2_planned']}"
        r1t = rate_text((x["pooled_r1"] or {}).get("null_raw")) if x["pooled_r1"] else "-"
        r2t = rate_text((x["pooled_r2"] or {}).get("null_raw")) if x["pooled_r2"] else "-"
        lines.append(
            f"| {a}{tag(a)} | {x['membership']} | {x['native_c1'] or '-'} | {x['verdict']} | "
            f"{x['n_invalid']}/{x['n_cells']} ({x['f_max']}) | {r2} | {r1t} | {r2t} |"
        )
    lines.append("")
    return lines


def _md_c2c3(v4: Verdict, tag: Callable[[str], str]) -> list[str]:
    focal = v4["focal"]
    lines = _md_component(f"C2a {focal} valid in R1 / R2", v4["C2a"])
    lines.append(f"- C2b eq arms whose native partner is a C1 failure: **{v4['C2b']['verdict']}**")
    for a, x in v4["C2b"]["arms"].items():
        lines += _md_component(f"{a}{tag(a)} (native {x['native']})", x, "  ")
    for a, x in v4["C2b"].get("excluded", {}).items():
        lines += _md_component(
            f"{a} EXCLUDED from C2b (R-40) [{x['label']}], same rule reported (no effect)", x, "  "
        )
    w2 = v4["C2b"].get("with_r1_failing_partners")
    if w2:
        lines.append(
            f"- C2b with the R1-failing partners (R-56, sensitivity, no effect): "
            f"**{w2['verdict']}** (added: "
            f"{', '.join(w2['added']) or 'none'})"
        )
    lines.append(f"- C2 wording: {v4['C2_wording']}.")
    lines += [f"  - {x['sentence']}." for x in v4.get("C2_named", [])]
    if v4.get("eq_arms_table"):
        lines += _md_eq_table(v4, tag)
    lines += _md_component(f"C3 {focal} valid in E4 R3", v4["C3"])
    if v4["C3"].get("wording"):
        lines.append(f"  - wording: {v4['C3']['wording']}.")
    for a, x in v4.get("pmrt_secondary", {}).items():
        lines += _md_component(
            f"{a}{tag(a)} (secondary PMRT arm, same rules reported, no claim effect): C2a", x["C2a"]
        )
        lines += _md_component(f"{a}: C3", x["C3"], "  ")
    eq = "; ".join(f"{a}{tag(a)} {x['verdict']}" for a, x in v4["C3_eq_arms"].items()) or "none"
    lines.append("- C3 reported for every eq arm (no claim effect): " + eq)
    return lines


def md_verdicts(v4: Verdict) -> list[str]:
    """V4 markdown: claim, C1, C2a, C2b, C3 with the sensitivity lines."""
    labels = v4.get("arm_labels", {})

    def tag(arm: str) -> str:
        return f" [{labels[arm]}]" if arm in labels else ""

    return [*_md_c1(v4, tag), *_md_c2c3(v4, tag)]


def md_validity(out: Tables, arms: list[str], nig: set) -> list[str]:
    """V1 markdown: INVALID / VALID / counted cells per world-regime and the listings."""
    lines = [
        "",
        "## V1 validity: INVALID / VALID / counted cells per world-regime (all n)",
        "",
        "| world regime | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    tally: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for r in out["V1_validity"]:
        c = tally[(world_regime(r), r["arm"])]
        if r["status"] in COUNTED and r["validity"] not in (None, "few_seeds"):
            c[2] += 1
            c[0] += r["validity"] == "INVALID"
            c[1] += r["validity"] == "VALID"
        else:
            c[3] += 1
    for label in sorted({world_regime(r) for r in out["V1_validity"]}):
        cells = []
        for arm in arms:
            c = tally.get((label, arm))
            if not c:
                cells.append("nig" if any(x[0] == arm and x[1] == label for x in nig) else "-")
                continue
            head = f"**{c[0]}**" if c[0] else "0"
            cells.append(head + f"/{c[1]}/{c[2]}" + (f" ({c[3]}x)" if c[3] else ""))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "(Nx) = planned cells not counted (missing, infeasible, T3-infeasible, untuned or < 10 "
        "seeds); nig = "
        "not in the arm's grid (pre-registered grid choice, R-54 / R-58: not planned, never "
        "missing / PARTIAL).",
    ]
    lines += _md_not_in_grid(out)
    lines += _md_invalid_cells(out)
    return lines


def _md_not_in_grid(out: Tables) -> list[str]:
    grid = out.get("not_in_grid") or {}
    if not (grid.get("by_rule") or grid.get("unexplained")):
        return []
    lines = ["", "Not in grid (pre-registered grid choices):", ""]
    for r in grid.get("by_rule", []):
        where = ", ".join(f"{k} {v}" for k, v in r["where"].items() if v)
        lines.append(
            f"- {r['ruling']}: {', '.join(r['arms'])} ({where}): {r['n_cells']} cells; "
            f"{r['reason']}"
        )
    if grid.get("unexplained"):
        lines.append(
            f"- UNEXPLAINED gaps ({len(grid['unexplained'])}): "
            + ", ".join(grid["unexplained"][:10])
        )
    return lines


def _md_invalid_cells(out: Tables) -> list[str]:
    flagged = [r for r in out["V1_validity"] if r["validity"] == "INVALID"]
    lines = ["", f"INVALID cells ({len(flagged)}; first 30):"]
    for r in flagged[:30]:
        bad = [
            f"{k} {fmt(r[k]['rate'], 3)} [{fmt(r[k]['ci'][0], 3)}, {fmt(r[k]['ci'][1], 3)}]"
            for k in RATE_KEYS
            if r.get(k) and r[k]["validity"] == "INVALID"
        ]
        lines.append(f"- {r['key']}: " + "; ".join(bad))
    uncounted = [r for r in out["V1_validity"] if r["status"] not in COUNTED]
    if uncounted:
        lines += ["", f"Cells not counted ({len(uncounted)}; first 30):"]
        for r in uncounted[:30]:
            costs = "".join(
                f" (cost {fmt(r[f'infeasible_cost_{f}'], 0)} {unit})"
                for f, unit in (("cpu_s", "CPU-s"), ("wall_s", "GPU wall-s"))
                if r.get(f"infeasible_cost_{f}") is not None
            )
            lines.append(f"- {r['key']}: {r['status']}{costs}")
    return lines


def md_recall(out: Tables, arms: list[str], nig: set) -> list[str]:
    """V2 markdown: recall among cells not INVALID, per world-regime and n."""
    ns = sorted({r["n"] for r in out["V2_recall"]})
    lines = [
        "",
        "## V2 recall among cells not INVALID (n " + " / ".join(map(str, ns)) + "; inv INVALID, "
        "inf infeasible, mis missing, unt untuned, few < 10 seeds; NA not applicable, R-42)",
        "",
        "| world regime | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    index = {(r["arm"], world_regime(r), r["n"]): r for r in out["V2_recall"]}
    for label in sorted({world_regime(r) for r in out["V2_recall"]}):
        cells = []
        for arm in arms:
            parts = [_recall_part(index.get((arm, label, n))) for n in ns]
            parts = [
                "nig" if x == "-" and (arm, label, n, PRIMARY_KAPPA) in nig else x
                for x, n in zip(parts, ns, strict=True)
            ]
            cells.append("-" if all(x == "-" for x in parts) else "/".join(parts))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _recall_part(r: dict[str, Any] | None) -> str:
    if r is None:
        return "-"
    if r["recall"]:
        return fmt(r["recall"]["mean"])
    if r.get("power_not_applicable") and r["state"] not in STATE_TAG:
        return "NA"
    return STATE_TAG.get(r["state"], "-")


def md_paired(out: Tables, focal: str) -> list[str]:
    """V3 markdown: counts of cells where the focal arm is higher, lower or indistinguishable."""
    lines = [
        "",
        f"## V3 paired recall difference {focal} - arm (neither INVALID, same declaration rule; "
        f"per-cell 95 % "
        "paired-t CI, descriptive)",
        "",
        "| arm | rule | block | cells pmrt higher | pmrt lower | CI covers 0 |",
        "|---|---|---|---|---|---|",
    ]
    by_arm: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    for d in out["V3_paired"]:
        c = by_arm[(d["arm"], d["rule"], d["block"])]
        c[0 if d["ci"][0] > 0 else 1 if d["ci"][1] < 0 else 2] += 1
    lines += [
        f"| {a} | {r} | {b} | {c[0]} | {c[1]} | {c[2]} |" for (a, r, b), c in sorted(by_arm.items())
    ]
    return lines


def _grid_cell(r: dict[str, Any], value: str) -> str:
    if r["status"] not in COUNTED:
        return STATE_TAG.get(r["status"], r["status"])
    return value + CELL_MARK.get(r.get("validity"), "?")


def md_grid(
    title: str,
    rows: Table,
    arms: list[str],
    row_key: Callable[[dict[str, Any]], str],
    value: Callable[[dict[str, Any]], str],
) -> list[str]:
    """Renders a row-key by arm grid of cell values with validity marks."""
    cols = [a for a in arms if any(r["arm"] == a for r in rows)]
    grid = {(row_key(r), r["arm"]): _grid_cell(r, value(r)) for r in rows}
    lines = ["", title, "", "| row | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    ordered = sorted(
        rows,
        key=lambda r: (
            r["world"],
            r["regime"],
            -1 if r["lam"] is None else r["lam"],
            r["kappa"],
            r["n"],
        ),
    )
    for rk in dict.fromkeys(row_key(r) for r in ordered):
        lines.append(f"| {rk} | " + " | ".join(grid.get((rk, a), "-") for a in cols) + " |")
    return lines


def _recall_value(r: dict[str, Any]) -> str:
    return fmt((r.get("recall") or {}).get("mean"))


def _e4_value(r: dict[str, Any]) -> str:
    placebo = fmt(((r.get("conf_raw") or r.get("conf_decl")) or {}).get("rate"))
    wrong = "NA" if r.get("power_not_applicable") else fmt(r["wrong_sign_rate"])
    return f"{placebo}/{wrong}"


def md_information(out: Tables, arms: list[str]) -> list[str]:
    """V5 to V9 markdown: information levels, E4, not-testable and secondary counts, kappa sweep."""
    lines = md_grid(
        "## V5 information levels in R2 (pmrt, eq, eq_min, native): recall (* INVALID, ~ "
        "INCONCLUSIVE)",
        out["V5_information_R2"],
        arms,
        lambda r: f"{r['world']} n{r['n']}",
        _recall_value,
    )
    lines += md_grid(
        "## V6 E4: placebo_conf rate (raw p, else declared) / wrong-sign rate of the true edge (* "
        "INVALID)",
        [r for r in out["V6_E4"] if r["n"] in (1000, 4000, 24000)],
        arms,
        lambda r: f"{r['regime']} l{r['lam']:g} n{r['n']}",
        _e4_value,
    )
    lines += [
        "",
        "## V7 not testable / not applicable",
        "",
        f"{len(out['V7_not_testable'])} (arm, cell) with counts > 0 (JSON V7).",
        "",
        "## V8 secondary KPI -> KPI family",
        "",
        f"{len(out['V8_secondary'])} (arm, cell) rows (JSON V8).",
    ]
    lines += md_grid(
        "## V9 kappa sweep (R2, n 1000): recall",
        out["V9_kappa_sweep"],
        arms,
        lambda r: f"{r['world']} k{r['kappa']:g}",
        _recall_value,
    )
    return lines


def _cost_cells(rs: list[dict[str, Any]], field: str, ns: list[int]) -> list[str]:
    cells = []
    for n in ns:
        means = [r[f"{field}_mean"] for r in rs if r["n"] == n and r[f"{field}_mean"] is not None]
        maxes = [r[f"{field}_max"] for r in rs if r["n"] == n and r[f"{field}_max"] is not None]
        if means:
            cells.append(f"{np.mean(means):.1f} / {max(maxes):.1f}")
        else:
            cells.append("T3" if any(r["n"] == n and r["t3"] for r in rs) else "-")
    return cells


def md_cost(out: Tables, arms: list[str]) -> list[str]:
    """V10 markdown: cost per arm and n, CPU arms apart from GPU arms."""
    ns = sorted({r["n"] for r in out["V10_cost"]})
    budget = {r["arm"]: r["budget"] for r in out["V10_cost"]}
    lines: list[str] = []
    titles = (
        ("cpu_s", "CPU-s per dataset"),
        ("wall_s", "GPU arms (R-41): wall-s per dataset on the GPU host, reported apart from CPU"),
    )
    for field, title in titles:
        table_arms = [a for a in arms if budget.get(a, "cpu_s") == field]
        if not table_arms:
            continue
        lines += [
            "",
            f"## V10 cost ({title}; mean / max over worlds, regimes; peak RSS MB; infeasible "
            f"units; "
            "T3 = infeasible from DEV cost)",
            "",
            "| arm | " + " | ".join(f"n {n}" for n in ns) + " | RSS | infeasible |",
            "|---|" + "---|" * (len(ns) + 2),
        ]
        for arm in table_arms:
            rs = [r for r in out["V10_cost"] if r["arm"] == arm]
            rss = max([r["peak_rss_mb_max"] or 0 for r in rs], default=0)
            cells = _cost_cells(rs, field, ns)
            lines.append(
                f"| {arm} | "
                + " | ".join(cells)
                + f" | {rss:.0f} | {sum(r['n_infeasible'] for r in rs)} |"
            )
    return lines


def md_integrity(v11: dict[str, Any]) -> list[str]:
    """V11 markdown: label, failed checks and the counts behind them."""
    b = v11["by_recheck"]
    return [
        "",
        "## V11 integrity",
        "",
        f"- label {v11['label']}; failed checks: {', '.join(v11['failed']) or 'none'}",
        f"- freeze commit {v11['freeze_commit']}; amendments {v11['amendments']}; records used "
        f"{v11['n_records_used']} / expected {v11['n_expected']}; missing {v11['n_missing']}; "
        f"unexpected "
        f"{v11['n_unexpected']}; role mismatch {v11['n_role_mismatch']}; duplicates "
        f"{v11['n_duplicates']}",
        f"- commit violations {v11['n_commit_violations']}; not clean {v11['n_not_clean']}; stamp "
        f"violations "
        f"{v11['n_stamp_violations']}; candidates incomplete {v11['n_candidates_incomplete']}; "
        f"pkgs sets "
        f"{v11['pkgs_sets']}",
        f"- role:status {v11['by_role_status']}; errors {len(v11['errors'])} (not listed "
        f"persistent "
        f"{len(v11['errors_not_listed_persistent'])}); infeasible {len(v11['infeasible'])}; "
        f"dataset-hash "
        f"mismatches {v11['n_dataset_hash_mismatch']} (R-41a: equal across all arms of a dataset); "
        f"ok records "
        f"without a dataset hash {v11['n_dataset_hash_unstamped']} (not-ok "
        f"{v11['n_dataset_hash_unstamped_not_ok']}); "
        f"multi-platform datasets "
        f"{len(v11['multi_platform_datasets'])}",
        f"- BY recheck (p arms): {b['agree']} / {b['records']} records agree; tune reproducibility "
        f"vs DEV: "
        f"{v11['tune_reproducibility_vs_dev']}",
    ]


def markdown(out: Tables) -> str:
    """Renders all tables as markdown."""
    v11 = out["V11_integrity"]
    arms = arm_order(out)
    focal = out.get("focal") or FOCAL
    nig = nig_index(out)
    lines = [
        f"# Study A EVAL tables ({v11['label']})",
        "",
        f"`{out['analysis']}`, spec `{out['spec_name']}`, primary kappa {out['primary_kappa']}, "
        f"primary PMRT arm "
        f"{focal}; commits {', '.join(c[:9] for c in v11['commits'])}. Protocol: "
        f"docs/xmethod/PROTOCOL_A.md.",
        "",
    ]
    lines += [*md_like(out.get("V0_like_for_like", []), arms, nig), ""]
    lines += md_verdicts(out["V4_verdicts"])
    lines += md_validity(out, arms, nig)
    lines += md_recall(out, arms, nig)
    lines += md_paired(out, focal)
    lines += md_information(out, arms)
    lines += md_cost(out, arms)
    lines += md_integrity(v11)
    return "\n".join(lines) + "\n"


def clean(x: Any) -> Any:
    """Makes a table tree JSON-safe: NaN and infinity become None, NumPy scalars Python values."""
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.floating | np.integer):
        return clean(x.item())
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def to_json(payload: Any) -> str:
    """Serialises a table tree the way the reports store it: sorted keys, 1-space indent."""
    return json.dumps(clean(payload), indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_tables(out: Tables, out_dir: PathLike) -> tuple[str, str]:
    """Writes `eval_tables.json` and `EVAL_TABLES.md`; returns their sha256."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    js, md = to_json(out), markdown(out)
    (directory / "eval_tables.json").write_text(js, encoding="utf-8", newline="\n")
    (directory / "EVAL_TABLES.md").write_text(md, encoding="utf-8", newline="\n")
    return hashlib.sha256(js.encode()).hexdigest(), hashlib.sha256(md.encode()).hexdigest()
