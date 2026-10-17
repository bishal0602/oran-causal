"""`verify <study>`: the same-conclusion rules on a re-run's records against the frozen ones.

A re-run directory has the layout of `data/records`. The rules are fixed in the plan:

- Study 1: PARTIAL, failing only P2 and only against Granger + BY; the chain is 4/4 with the
  frozen signs; no PMRT placebo declaration.
- Study 2: discovery PASS; referee NOT ELIGIBLE; the ptx -> nbr RLF edge stays undeclared.
- Study 3: PARTIAL; gated PMRT equals never-sleep on every seed (D3 difference 0, CI [0, 0]).
- Study 4: the C1, C2a, C2b and C3 verdicts and the PMRT-Lin secondary verdicts match; each
  INVALID count within F_max stays within it; every pooled rate falls in its frozen 95 % CI.
- Studies 2 and 3: every point estimate falls in its frozen 90 % interval; Study 1: the G cell
  and every knockout GT cell mean fall in their frozen intervals.
"""

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from oran_causal.analysis.system import study1, study2
from oran_causal.experiments.reproduce.checks import Check, Status
from oran_causal.experiments.reproduce.context import Context
from oran_causal.experiments.reproduce.t0 import study4_report

STUDY1_ANALYSIS = "study1_raw/e6p-disc-v4an-pmrt-1/out/analysis_v4.json"
STUDY2_DISC = "study2_raw/e6p-conf-discan-2/out/disc_conf.json"
STUDY2_EVAL = "study2_3_decision/runs/e6p-conf-evalan-1/out/eval_conf.json"
STUDY3_EVAL = "study2_3_decision/runs/e6p-cs-evalan-1/out/eval_certsafe.json"
STUDY4_REPORT = "study4_eval/eval/report.json"
STUDY4_MERGED = "study4_eval/eval/merged.jsonl.gz"
STUDY1_CRITERIA = ("K0", "K0n", "G", "K1", "P1", "P2", "P3", "S")
PTX_NBR_RLF = ("ptx", "nbr", "rlf")
STUDY4_COMPONENTS = ("C1", "C2a", "C2b", "C3")

type Tree = Mapping[str, Any]
type Rules = Callable[[Context, Path], list[Check]]


def _rule(item: str, target: str, ok: bool, detail: str) -> Check:
    return Check(item, target, Status.PASS if ok else Status.DIFF, detail)


def _load(root: Path, rel: str) -> dict[str, Any]:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _inside(x: Any, ci: Any) -> bool:
    return isinstance(x, int | float) and ci[0] <= x <= ci[1]


def _pairs(new: Tree, old: Tree, path: str = "") -> Iterator[tuple[str, Tree, Tree]]:
    yield path, new, old
    for k, v in old.items():
        if isinstance(v, Mapping) and isinstance(new.get(k), Mapping):
            yield from _pairs(new[k], v, f"{path}/{k}")


def interval_misses(new: Tree, old: Tree) -> tuple[int, list[str]]:
    """Point estimates of `new` outside the frozen 90 % intervals of `old`.

    An interval is a key `<name>_ci90` (scalar or per-KPI mapping) beside `<name>`, or `ci90`
    beside `dR`.

    Returns:
        The number of estimates checked and the paths of those outside.
    """
    n, bad = 0, []
    for path, a, b in _pairs(new, old):
        for k, ci in b.items():
            name = "dR" if k == "ci90" else k.removesuffix("_ci90")
            if not k.endswith("ci90") or name not in a:
                continue
            values = a[name] if isinstance(ci, Mapping) else {"": a[name]}
            cis = ci if isinstance(ci, Mapping) else {"": ci}
            for kpi, c in cis.items():
                n += 1
                if not _inside(values.get(kpi), c):
                    bad.append(f"{path}/{name}{'.' + kpi if kpi else ''}")
    return n, bad


def _interval_check(item: str, new: Tree, old: Tree) -> Check:
    n, bad = interval_misses(new, old)
    return _rule(item, "point estimates", not bad, f"{n - len(bad)}/{n} in frozen 90 % CI {bad}")


def _gt_cells(rep: Tree, frozen: Tree) -> Check:
    def cells(r: Tree) -> dict[tuple[str, str, str], Tree]:
        return {(c["family"], c["relation"], c["kpi"]): c for c in r["gt"]["cells"]}

    new, old = cells(rep), cells(frozen)
    bad = [k for k, c in old.items() if not _inside(new.get(k, {}).get("mean"), c["ci"])]
    return _rule(
        "study1", "GT cell means", not bad, f"{len(old) - len(bad)}/{len(old)} in CI {bad}"
    )


def _study1(ctx: Context, records: Path) -> list[Check]:
    rep = study1.recompute(_load(records, STUDY1_ANALYSIS))
    frozen = _load(ctx.records, STUDY1_ANALYSIS)
    failing = [k for k in STUDY1_CRITERIA if not rep[k]["pass"]]
    per = rep["P2"]["per_baseline"]
    against = [b for b, v in per.items() if not v["ok"]]
    p3 = rep["P3"]["values"]
    signs = {h: v["z"] > 0 for h, v in rep["P3"]["chain"].items()}
    frozen_signs = {h: v["z"] > 0 for h, v in frozen["P3"]["chain"].items()}
    k0 = (rep["K0"]["n_declared"], rep["K0n"]["n_variants_with_decl"])
    sign_text = "equal" if signs == frozen_signs else "differ"
    verdict = rep["verdict"]["verdict"]
    g, g_ci = rep["G"]["cell"]["mean"], frozen["G"]["cell"]["ci"]
    return [
        _rule("study1", "verdict", verdict == "PARTIAL", verdict),
        _rule(
            "study1",
            "failing",
            (failing, against) == (["P2"], ["granger_by"]),
            f"{failing} {against}",
        ),
        _rule(
            "study1",
            "chain",
            p3["chain_hits"] == p3["chain_true"] == 4 and signs == frozen_signs,
            f"{p3['chain_hits']}/{p3['chain_true']}, signs {sign_text}",
        ),
        _rule("study1", "placebo declarations", k0 == (0, 0), f"K0 {k0[0]}, K0n variants {k0[1]}"),
        _rule("study1", "G estimate", _inside(g, g_ci), f"{g:.3f} in {g_ci}"),
        _gt_cells(rep, frozen),
    ]


def _study2(ctx: Context, records: Path) -> list[Check]:
    disc = study2.recompute_disc(_load(records, STUDY2_DISC))
    rep, frozen = _load(records, STUDY2_EVAL), _load(ctx.records, STUDY2_EVAL)
    label, verdict = disc["label"]["label"], rep["verdict"]["verdict"]
    pmrt = disc["maps"]["MG:PMRT"]
    edge = any(tuple(e[:3]) == PTX_NBR_RLF for e in pmrt)
    return [
        _rule("study2", "discovery", label == "DISC-PASS", label),
        _rule("study2", "referee", verdict == "NOT ELIGIBLE", verdict),
        _rule("study2", "ptx -> nbr RLF", not edge, "declared" if edge else "undeclared"),
        _interval_check("study2", rep, frozen),
    ]


def _study3(ctx: Context, records: Path) -> list[Check]:
    rep, frozen = _load(records, STUDY3_EVAL), _load(ctx.records, STUDY3_EVAL)
    d3, verdict = rep["D3"], rep["verdict"]["verdict"]
    same = d3["delta3"] == 0.0 and d3["ci90"] == [0.0, 0.0] and d3["ref"] == "never_sleep"
    return [
        _rule("study3", "verdict", verdict == "PARTIAL", verdict),
        _rule(
            "study3", "gated PMRT = never-sleep", same, f"delta3 {d3['delta3']}, CI {d3['ci90']}"
        ),
        _interval_check("study3", rep, frozen),
    ]


def study4_verdicts(new: Tree, old: Tree) -> dict[str, tuple[Any, Any]]:
    """The Study 4 verdicts that differ, as (re-run, frozen) by name."""
    pairs = {c: (new["components"][c], old["components"][c]) for c in STUDY4_COMPONENTS}
    pairs["claim"] = (new["claim"], old["claim"])
    for arm, blocks in old["pmrt_secondary"].items():
        for block, v in blocks.items():
            if isinstance(v, Mapping) and "verdict" in v:
                pairs[f"{arm}.{block}"] = (
                    new["pmrt_secondary"][arm][block]["verdict"],
                    v["verdict"],
                )
    return {k: v for k, v in pairs.items() if v[0] != v[1]}


def study4_bounds(new: Tree, old: Tree) -> tuple[list[str], list[str], int, int]:
    """INVALID counts that left F_max and pooled rates outside the frozen 95 % CI.

    Returns:
        Paths over F_max, paths outside the CI, and the numbers of counts and rates checked.
    """
    over, outside, n_f, n_r = [], [], 0, 0
    for path, a, b in _pairs(new, old):
        if "n_invalid" in b and "f_max" in b and b["n_invalid"] <= b["f_max"]:
            n_f += 1
            if a.get("n_invalid", b["f_max"] + 1) > b["f_max"]:
                over.append(path)
        if {"rate", "ci", "hits"} <= b.keys() and path.split("/")[-2].startswith("pooled"):
            n_r += 1
            if not _inside(a.get("rate"), b["ci"]):
                outside.append(path)
    return over, outside, n_f, n_r


def _study4_tables(ctx: Context, records: Path) -> dict[str, Any] | Check:
    if (records / STUDY4_REPORT).exists():
        return _load(records, STUDY4_REPORT)["tables"]
    out = ctx.item_dir("verify_study4")
    code = study4_report(records / STUDY4_MERGED, ctx.records / "study4_dev", out)
    if code != 0:
        return Check("study4", "run_report", Status.ERROR, f"exit {code}")
    return json.loads((out / "report.json").read_text(encoding="utf-8"))["tables"]


def _study4(ctx: Context, records: Path) -> list[Check]:
    tables = _study4_tables(ctx, records)
    if isinstance(tables, Check):
        return [tables]
    new = tables["V4_verdicts"]
    old = _load(ctx.records, STUDY4_REPORT)["tables"]["V4_verdicts"]
    differ = study4_verdicts(new, old)
    over, outside, n_f, n_r = study4_bounds(new, old)
    return [
        _rule("study4", "verdicts", not differ, f"claim {new['claim']}; differ {differ}"),
        _rule("study4", "INVALID within F_max", not over, f"{n_f - len(over)}/{n_f} {over}"),
        _rule("study4", "pooled rates", not outside, f"{n_r - len(outside)}/{n_r} in CI {outside}"),
    ]


RULES: dict[str, Rules] = {
    "study1": _study1,
    "study2": _study2,
    "study3": _study3,
    "study4": _study4,
}


def verify(study: str, records: Path, ctx: Context) -> list[Check]:
    """The same-conclusion rows of `study` for the re-run in `records` (frozen: `ctx.records`).

    Study 4 reads the re-run's `report.json` or, without one, builds it from the re-run's
    merged records and the frozen DEV inputs.

    Raises:
        KeyError: An unknown study.
    """
    if study not in RULES:
        raise KeyError(f"unknown study {study!r}; known: {list(RULES)}")
    return RULES[study](ctx, records)
