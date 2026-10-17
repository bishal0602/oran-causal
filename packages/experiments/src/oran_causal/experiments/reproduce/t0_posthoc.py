"""Tier t0, post hoc analyses X1-X16: each rebuilds its stored outputs from the record sets.

The stored files were written in Windows text mode, so they are compared with CRLF read as LF.
Four outputs carry platform floats and are compared as the post hoc gates do: X2/X3
provenance (exact on Linux only), the X8 binomial tails (exact on Windows only, else relative
1e-12), the X15 OLS slope (relative 1e-12) and X16 (relative 1e-9, its environment keys and CSV
not compared).
"""

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from oran_causal.analysis.controlled.tables import clean
from oran_causal.analysis.posthoc import (
    x1_runtime,
    x2x3,
    x4,
    x5x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
)
from oran_causal.experiments.paths import SPECS
from oran_causal.experiments.posthoc.tools import referee_tools
from oran_causal.experiments.reproduce.checks import (
    Check,
    Status,
    close_tree,
    compare_bytes,
    compare_json_close,
)
from oran_causal.experiments.reproduce.context import Context, guarded, missing

X1_LEGACY_DIR = (
    r"D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results"
    r"\exp_c\paper\raw\xm-expc-paper-k1" + "\\"
)
X2X3_SPECS = ("x2_d02", "x2_d05", "x2_d10", "x2_d20", "x2_b05", "x2_b80", "x3_told")
X15_SLOPE = "ols_dR_on_V_AA"
X16_SKIP = ("sha256", "env", "timing_s", "data")
LINUX = sys.platform.startswith("linux")
WINDOWS = sys.platform == "win32"


def _same(item: str, out: Path, stored: Path, names: Sequence[str]) -> list[Check]:
    return [compare_bytes(item, out / n, stored / n, eol=True) for n in names]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def x1(ctx: Context) -> list[Check]:
    """X1 runtime table of the paper run."""
    d = ctx.records / "posthoc_x1" / "exp_c"
    out = ctx.item_dir("posthoc/x1")
    files = sorted((d / "paper" / "raw" / "xm-expc-paper-k1").glob("res_*.jsonl"))
    recs, info = x1_runtime.load(files, {str(f.absolute()): X1_LEGACY_DIR + f.name for f in files})
    stored = json.loads((d / "paper" / "exp_c_runtime.json").read_text(encoding="utf-8"))
    specs = x1_runtime.load_specs([SPECS / "exp_c" / "timing.json"])
    doc = x1_runtime.build(recs, info, generated_utc=stored["generated_utc"], arm_specs=specs)
    md = (d / "paper" / "EXP_C_RUNTIME.md").read_text(encoding="utf-8")
    title = md.split("\n", 1)[0].split("(", 1)[1].rsplit(")", 1)[0]
    _write(out / "exp_c_runtime.json", json.dumps(doc, indent=1, allow_nan=False))
    _write(out / "EXP_C_RUNTIME.md", x1_runtime.render_md(doc, title))
    return _same("posthoc X1", out, d / "paper", ("exp_c_runtime.json", "EXP_C_RUNTIME.md"))


def x2_x3(ctx: Context) -> list[Check]:
    """X2 dose-response and X3 misspecification tables."""
    d = ctx.records / "posthoc_x2_x3"
    out = ctx.item_dir("posthoc/x2x3")
    stored = json.loads((d / "extras_tables.json").read_text(encoding="utf-8"))
    pairs = [(n, d / n / "merged.jsonl.gz") for n in X2X3_SPECS]
    doc = x2x3.analyse(pairs, protocol_sha256=stored["protocol_sha256"])
    _write(out / "extras_tables.json", json.dumps(clean(doc), indent=1, sort_keys=True))
    _write(out / "EXTRAS_TABLES.md", x2x3.analyse_md(doc))
    if LINUX:
        return _same("posthoc X2/X3", out, d, ("extras_tables.json", "EXTRAS_TABLES.md"))
    mine = json.loads((out / "extras_tables.json").read_text(encoding="utf-8"))
    for tree in (mine, stored):
        for c in tree["checks"].values():
            c.pop("provenance")
    diffs = close_tree(mine, stored, 0.0)
    status = Status.DIFF if diffs else Status.PASS
    note = "equal without checks.*.provenance (exact on Linux only)"
    return [Check("posthoc X2/X3", "extras_tables.json", status, diffs[0] if diffs else note)]


def x4_x7(ctx: Context) -> list[Check]:
    """X4 random-rejection and X7 subset-equivalence tables."""
    runs = ctx.records / "study2_3_decision" / "runs" / "e6p-cs-eval-1"
    cs_eval = ctx.records / "system_artifacts" / "decision" / "cs_eval.json"
    factors = ctx.records / "posthoc_x1" / "exp_c" / "calib" / "factors.json"
    d4, d7 = ctx.records / "posthoc_x4" / "x4", ctx.records / "posthoc_x7" / "x7"
    out4, out7 = ctx.item_dir("posthoc/x4"), ctx.item_dir("posthoc/x7")
    x4.write(x4.analyse(runs, d4 / "records", cs_eval=cs_eval, factors=factors), out4)
    x7.write(x7.analyse(runs, d7 / "records", cs_eval=cs_eval, factors=factors), out7)
    return [
        *_same("posthoc X4", out4, d4, ("x4_tables.json",)),
        *_same("posthoc X7", out7, d7, ("x7_tables.json",)),
    ]


def x5_x6(ctx: Context) -> list[Check]:
    """X5 / X6 diagnostics: the three stages and the combined tables."""
    d = ctx.records / "posthoc_x5_x6"
    out = ctx.item_dir("posthoc/x5x6")
    x5x6.write_stage(x5x6.x6(d / "x6_gbm" / "merged.jsonl.gz"), "x6", out)
    res = x5x6.x5(d / "x5_fail" / "merged.jsonl.gz", d / "x5_adj" / "merged.jsonl.gz")
    x5x6.write_stage(res, "x5", out)
    res = x5x6.x5_step1(
        ctx.records / "study4_eval" / "eval" / "merged.jsonl.gz",
        dependence=ctx.records / "study4_dev" / "fmax_sim" / "dependence.json",
    )
    x5x6.write_stage(res, "x5_step1", out)
    stored = json.loads((d / "diag" / "diag_tables.json").read_text(encoding="utf-8"))
    x5x6.combine_tables(out, stored["protocol_sha256"])
    names = (
        "diag_x6.json",
        "diag_x5.json",
        "diag_x5_step1.json",
        "diag_tables.json",
        "DIAG_TABLES.md",
    )
    return _same("posthoc X5/X6", out, d / "diag", names)


def x8_summary(ctx: Context) -> list[Check]:
    """X8 placebo tail summary (bytes on Windows, floats to relative 1e-12 elsewhere)."""
    d = ctx.records / "posthoc_x8" / "x8"
    out = ctx.item_dir("posthoc/x8")
    _write(out / "summary.json", json.dumps(x8.summarize(d), indent=1))
    if WINDOWS:
        return _same("posthoc X8", out, d, ("summary.json",))
    return [compare_json_close("posthoc X8", out / "summary.json", d / "summary.json", 1e-12)]


def revision(ctx: Context) -> list[Check]:
    """X9-X14: the revision checks over the frozen record sets."""
    rev = ctx.records / "posthoc_x9_x16" / "revision"
    out = ctx.item_dir("posthoc/revision")
    tools = referee_tools()
    x9.write(x9.run(ctx.records), out)
    x10.write(x10.analyse(ctx.records, x10.ARTIFACTS, tools), out)
    x11.write(x11.run(ctx.records), out)
    x12.write(x12.run(ctx.records, tools=tools), out)
    x13.write(x13.run(ctx.records), out)
    x14.write(x14.run(ctx.records), out)
    names = (
        "x9_decision_e2e.json",
        "x9_decision_e2e.csv",
        "x10_gate_diag.json",
        "x10_gate_diag.csv",
        "x11_gt_sensitivity.json",
        "x11_gt_sensitivity.csv",
        "x11_gt_labels.csv",
        "x12_threshold.json",
        "x12_threshold.csv",
        "x13_attribution.json",
        "x13_attribution.csv",
        "x14_existence.json",
        "x14_existence.csv",
    )
    return _same("posthoc X9-X14", out, rev, names)


def x15_x16(ctx: Context) -> list[Check]:
    """X15 replication (one OLS slope to 1e-12) and X16 centre (floats to 1e-9)."""
    rev = ctx.records / "posthoc_x9_x16" / "revision"
    out = ctx.item_dir("posthoc/revision")
    x15.write(x15.run(ctx.records), out)
    checks = _same("posthoc X15", out, rev, ("x15_replication.csv",))
    checks.append(_x15_json(out, rev))
    x16.write(x16.run(ctx.records, tools=referee_tools(), cache_dir=out / "x16_cache"), out)
    checks.append(
        compare_json_close(
            "posthoc X16", out / "x16_centre.json", rev / "x16_centre.json", 1e-9, X16_SKIP
        )
    )
    return checks


def _x15_json(out: Path, rev: Path) -> Check:
    mine = json.loads((out / "x15_replication.json").read_text(encoding="utf-8"))
    stored = json.loads((rev / "x15_replication.json").read_text(encoding="utf-8"))
    exact = [d for d in close_tree(mine, stored, 0.0) if X15_SLOPE not in d]
    near = close_tree(mine, stored, 1e-12)
    if exact or near:
        return Check("posthoc X15", "x15_replication.json", Status.DIFF, (exact or near)[0])
    return Check(
        "posthoc X15", "x15_replication.json", Status.PASS, f"exact except {X15_SLOPE} (rtol 1e-12)"
    )


PARTS: tuple[tuple[str, tuple[str, ...], Callable[[Context], list[Check]]], ...] = (
    ("X1", ("posthoc_x1",), x1),
    ("X2/X3", ("posthoc_x2_x3",), x2_x3),
    ("X4/X7", ("posthoc_x4", "posthoc_x7", "study2_3_decision"), x4_x7),
    ("X5/X6", ("posthoc_x5_x6", "study4_eval", "study4_dev"), x5_x6),
    ("X8", ("posthoc_x8",), x8_summary),
    ("X9-X14", ("posthoc_x9_x16", "study1_raw", "study2_raw", "study2_3_decision"), revision),
    ("X15/X16", ("posthoc_x9_x16", "study2_raw", "study2_3_decision"), x15_x16),
)


def posthoc(ctx: Context) -> list[Check]:
    """Every post hoc part whose record sets are present."""
    checks = []
    for name, sets, part in PARTS:
        checks += missing(f"posthoc {name}", ctx, *sets) or guarded(f"posthoc {name}", part, ctx)
    return checks
