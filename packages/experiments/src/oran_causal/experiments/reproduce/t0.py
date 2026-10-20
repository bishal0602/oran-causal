"""Tier t0: rebuild every analysis output from the frozen records and diff it with the stored files.

Items: `study4` (the Study 4 EVAL report), `system` (Studies 1-3 analyses, verdicts and
`figures_data.json`), `bridging` (Appendix B tables and costs), `posthoc` (X1-X16) and
`figures` (paper figures and tables against `paper/`). Each item writes under `<out>/<item>`.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis import recorded
from oran_causal.analysis.bridging import costs, tables
from oran_causal.analysis.controlled.report import ReportRequest, run_report
from oran_causal.analysis.system import report_data, study1, study2, study3
from oran_causal.analysis.system.stats import alias_fill, dumps, load_jobs
from oran_causal.experiments.paths import ARTIFACTS, SPECS, STUDY4_PROTOCOL
from oran_causal.experiments.reproduce.checks import (
    Check,
    Status,
    compare_bytes,
    compare_text,
)
from oran_causal.experiments.reproduce.context import Context, Item, guarded, missing
from oran_causal.experiments.reproduce.t0_figures import figures
from oran_causal.experiments.reproduce.t0_posthoc import posthoc
from oran_causal.experiments.system import artifacts

STUDY4_FREEZE_COMMIT = "93856c230d84abda2ff3b894218dd6cb0ecd47f9"
STUDY4_FROZEN_UTC = "2026-10-05T07:33:00Z"
STUDY4_MERGED = "study4_eval/eval/merged.jsonl.gz"
STUDY4_DEV_SPECS = ("full", "ci_c", "pmrt_nl", "cdl")
STUDY1_RUNS = ("e6p-disc-v4an-pmrt-1", "e6p-disc-v4an-1")
STUDY1_MSCR_TIES = ("count for MSCR+; undefined MSCR+ F1", "count for PMRT; undefined PMRT F1")
BRIDGING_FACTORS = "factors_vps_py3.12.13.json"
BRIDGING_BUDGET = 7200.0


def study4(ctx: Context) -> list[Check]:
    """The Study 4 EVAL report (`report.json`, `REPORT.md`, `csv/*`) from the merged records."""
    ev, dev = ctx.records / "study4_eval" / "eval", ctx.records / "study4_dev"
    skips = missing("study4", ctx, STUDY4_MERGED, "study4_dev/fmax_sim")
    if skips:
        return skips
    out = ctx.item_dir("study4")
    code = study4_report(ev / "merged.jsonl.gz", dev, out)
    if code != 0:
        return [Check("study4", "run_report", Status.ERROR, f"exit {code}")]
    names = ["report.json", "REPORT.md", *(f"csv/{p.name}" for p in sorted((ev / "csv").iterdir()))]
    checks = [compare_bytes("study4", out / n, ev / n) for n in names]
    return [*checks, _study4_verdicts(out, ev)]


def study4_report(merged: Path, dev: Path, out: Path) -> int:
    """Writes the Study 4 EVAL report of `merged` into `out` with the frozen DEV inputs in `dev`.

    Returns:
        The report's exit code.
    """
    spec = SPECS / "eval" / "full.json"
    request = ReportRequest(
        spec=spec,
        merged=[merged],
        out_dir=out,
        freeze_commit=STUDY4_FREEZE_COMMIT,
        amendments=SPECS / "eval" / "amendments.json",
        protocol=STUDY4_PROTOCOL,
        dependence=dev / "fmax_sim" / "dependence.json",
        dev_merged=[dev / "dev" / n / "merged.jsonl.gz" for n in STUDY4_DEV_SPECS],
        generated_utc=STUDY4_FROZEN_UTC,
        input_labels={str(merged): recorded.record_label(STUDY4_MERGED)},
        cite_recorded=True,
    )
    return run_report(request, lambda _: None)


def _study4_verdicts(out: Path, ev: Path) -> Check:
    got = json.loads((out / "report.json").read_text(encoding="utf-8"))["tables"]["V4_verdicts"]
    want = json.loads((ev / "report.json").read_text(encoding="utf-8"))["tables"]["V4_verdicts"]
    keys = ("C1", "C2a", "C2b", "C3", "components", "claim")
    bad = [k for k in keys if got[k] != want[k]]
    if bad:
        return Check("study4", "verdicts", Status.DIFF, f"differ: {', '.join(bad)}")
    return Check("study4", "verdicts", Status.PASS, f"claim {got['claim']}")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _study1(ctx: Context) -> list[Check]:
    checks = []
    for run in STUDY1_RUNS:
        d = ctx.records / "study1_raw" / run / "out"
        raw, verdict = _read(d / "analysis_v4.json"), _read(d / "verdict_v4.json")
        rep = study1.recompute(json.loads(raw))
        if run == "e6p-disc-v4an-1":
            raw, verdict = raw.replace(*STUDY1_MSCR_TIES), verdict.replace(*STUDY1_MSCR_TIES)
        target = f"study1 {run}"
        checks.append(compare_text("system", f"{target} analysis_v4.json", dumps(rep), raw))
        vtext = dumps(study1.verdict_record(rep))
        checks.append(compare_text("system", f"{target} verdict_v4.json", vtext, verdict))
    return checks


def _disc_path(ctx: Context) -> Path:
    return ctx.records / "study2_raw" / "e6p-conf-discan-2" / "out" / "disc_conf.json"


def _study2_disc(ctx: Context) -> Check:
    raw = _read(_disc_path(ctx))
    return compare_text(
        "system", "study2 disc_conf.json", dumps(study2.recompute_disc(json.loads(raw))), raw
    )


def _verify_check(name: str, artifact: str, fresh: dict[str, Any], stored: dict[str, Any]) -> Check:
    cited = recorded.recorded(f"artifacts/{artifact}")
    same = fresh["checks"] == stored["checks"] and stored["sha256"] == cited.sha256
    if same and fresh["ok"] and fresh["matches_pin"]:
        return Check("system", name, Status.PASS, "artifact re-verified; checks and sha256 equal")
    return Check("system", name, Status.DIFF, f"ok {fresh['ok']}, checks or sha256 differ")


def _referee(ctx: Context, study: Any, run: str, artifact: str, key: str) -> list[Check]:
    runs = ctx.records / "study2_3_decision" / "runs"
    stored_dir = runs / run.replace("eval", "evalan") / "out"
    raw, verdict = _read(stored_dir / f"eval_{key}.json"), _read(stored_dir / f"verdict_{key}.json")
    stored = json.loads(raw)
    art = json.loads(_read(ARTIFACTS / artifact))
    files = sorted((runs / run).glob("*/res_*.jsonl"), key=lambda p: p.as_posix())
    filt = study2.EVAL_FILTER if study is study2 else study3.job_filter("eval")
    jobs, heads = load_jobs(files, filt)
    alias_fill(jobs, art["alias_of"])
    disc = json.loads(_read(_disc_path(ctx)))
    if study is study2:
        fresh, vkey = artifacts.verify_maps(), "maps_verify"
    else:
        fresh, vkey = artifacts.verify_certsafe(), "artifact_verify"
    inputs = study.EvalInputs(disc, art, stored[vkey], True, stored["argv"])
    rep = study.evaluate(jobs, heads, inputs) | {"timing_s": stored["timing_s"]}
    name = "study2" if study is study2 else "study3"
    return [
        _verify_check(f"{name} {vkey}", artifact, fresh, stored[vkey]),
        compare_text("system", f"{name} eval_{key}.json", dumps(rep), raw),
        compare_text(
            "system", f"{name} verdict_{key}.json", dumps(study.verdict_record(rep)), verdict
        ),
    ]


def system(ctx: Context) -> list[Check]:
    """Studies 1-3: re-derived criteria, referee evaluations, verdicts and `figures_data.json`."""
    fd_rel = "system_artifacts/report_2026-10-02/figures_data.json"
    skips = missing(
        "system",
        ctx,
        "study1_raw",
        "study2_raw/e6p-conf-discan-2",
        "study2_3_decision/runs",
        fd_rel,
    )
    if skips:
        return skips
    checks = [*_study1(ctx), _study2_disc(ctx)]
    checks += _referee(ctx, study2, "e6p-conf-eval-1", "study2_maps.json", "conf")
    checks += _referee(ctx, study3, "e6p-cs-eval-1", "study3_certsafe.json", "certsafe")
    out = ctx.item_dir("system") / "figures_data.json"
    report_data.write(out)
    return [*checks, compare_bytes("system", out, ctx.records / fd_rel)]


def bridging(ctx: Context) -> list[Check]:
    """Appendix B: the bridging tables and costs from the merged records."""
    exp_b = ctx.records / "bridging" / "exp_b"
    v4 = ctx.records / "study1_raw" / STUDY1_RUNS[0] / "out" / "analysis_v4.json"
    skips = missing(
        "bridging", ctx, "bridging/exp_b/merged.jsonl.gz", v4.relative_to(ctx.records).as_posix()
    )
    if skips:
        return skips
    out = ctx.item_dir("bridging")
    spec = json.loads(_read(SPECS / "bridging" / "exp_b.json"))
    merged = exp_b / "merged.jsonl.gz"
    tables.tables(spec, merged, exp_b / "gt_v4.json", out, json.loads(_read(v4)))
    costs.costs(
        merged,
        exp_b / BRIDGING_FACTORS,
        BRIDGING_BUDGET,
        out,
        label=recorded.record_label(f"bridging/exp_b/{BRIDGING_FACTORS}"),
    )
    names = (tables.TABLES_JSON, tables.TABLES_MD, costs.COSTS_JSON, costs.COSTS_MD)
    return [compare_bytes("bridging", out / n, exp_b / n) for n in names]


ITEMS: dict[str, Item] = {
    "study4": study4,
    "system": system,
    "bridging": bridging,
    "posthoc": posthoc,
    "figures": figures,
}


def run(
    ctx: Context, items: Sequence[str] = tuple(ITEMS), registry: dict[str, Item] = ITEMS
) -> list[Check]:
    """Runs the named items in order.

    Raises:
        KeyError: An unknown item.
    """
    unknown = sorted(set(items) - set(registry))
    if unknown:
        raise KeyError(f"unknown t0 items {unknown}; known: {list(registry)}")
    return [c for name in items for c in guarded(name, registry[name], ctx)]
