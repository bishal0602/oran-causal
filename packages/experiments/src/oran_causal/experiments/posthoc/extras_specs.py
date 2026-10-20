"""The declared X2, X3, X5 and X6 campaign specs, rebuilt from the frozen EVAL spec.

Arms are copied from `specs/eval/full.json` (ref, config, declare); the experiment designs are those
of `protocols/posthoc/PROTOCOL.md`. The frozen copies live in `specs/posthoc/`; `diff_frozen`
checks the rebuilt specs against them.

    python -m oran_causal.experiments.posthoc.extras_specs check [--dir DIR]
    python -m oran_causal.experiments.posthoc.extras_specs write --dir DIR [--records DIR]
"""

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.posthoc.paths import RECORDS, write_text
from oran_causal.experiments.campaign.extras import EXTRAS_SEEDS, X5_SEEDS, X6_SEEDS
from oran_causal.experiments.paths import SPECS

type Spec = dict[str, Any]

PROTOCOL_REL = "protocols/posthoc/PROTOCOL.md"
FROZEN_SPEC = SPECS / "eval" / "full.json"
FROZEN_DIR = SPECS / "posthoc"
KAPPA = 0.25
X2_TUNE = (EXTRAS_SEEDS.start, EXTRAS_SEEDS.start + 19)
X2_MEASURE = (EXTRAS_SEEDS.start + 20, EXTRAS_SEEDS.start + 59)
X3_MEASURE = (EXTRAS_SEEDS.start + 60, EXTRAS_SEEDS.start + 119)
X5_ADJ_SEEDS = (X5_SEEDS[0], X5_SEEDS[0] + 999)
DEV_AGGS = ("full", "ci_c", "pmrt_nl")
COST_AGG = "dev_cost_agg.json"
DIAG_COST_AGG = "diag_cost_agg.json"

X2_ARMS = (
    "corr",
    "pc_eq",
    "notears",
    "shap_dag",
    "pcorr_native",
    "pcorr_eq",
    "pmrt_eq",
    "pmrt_nl_eq",
)
X2_WORLDS, X2_NS = ("E1", "E2", "E3"), (1000, 4000)
X2_DESIGNS = {
    "x2_d02": (0.02, 20),
    "x2_d05": (0.05, 20),
    "x2_d10": (0.10, 20),
    "x2_d20": (0.20, 20),
    "x2_b05": (0.10, 5),
    "x2_b80": (0.10, 80),
}
X3_ARMS = ("pmrt_nl_eq", "pmrt_eq", "pmrt_r3", "pcorr_eq")
X3_DITHER = {
    "w050": {"width": 0.5},
    "w080": {"width": 0.8},
    "w125": {"width": 1.25},
    "w200": {"width": 2.0},
    "s02": {"shift": 0.02},
    "s05": {"shift": 0.05},
}
X3_LOGGED = {
    "l050": {"lam": 0.5},
    "l080": {"lam": 0.8},
    "l125": {"lam": 1.25},
    "l200": {"lam": 2.0},
}
X3_N = 1000
X5_ARMS = ("pmrt_eq", "pmrt_r3")
X5_FAIL = {8000: [0.0, 0.5], 24000: [1.0]}
X5_ADJ = {8000: [1.0, 1.5], 24000: [0.0, 0.5, 1.5]}
X6_WORLD, X6_N = "E1", 1000
X6_WIDTHS = {"w050": 0.5, "w080": 0.8, "w125": 1.25, "w150": 1.5, "w200": 2.0}
X6_ABLATION = {
    "w200_Tt": ("told", "true"),
    "w200_tT": ("true", "told"),
    "w200_tt": ("true", "true"),
}
KEEP = ("ref", "config", "declare", "analysis", "label", "worlds", "native_partner")


def _arm(frozen: Mapping[str, Any], name: str, told: Mapping[str, Any] | None = None) -> Spec:
    arm = {k: frozen[name][k] for k in KEEP if k in frozen[name]}
    if told is not None:
        arm.update(base=name, told=dict(told))
    return arm


def _block(
    role: str,
    worlds: Iterable[str],
    ns: Iterable[int],
    seeds: Sequence[int],
    arms: Iterable[str],
    regime: str = "R2",
    lams: Sequence[float] | None = None,
) -> Spec:
    block: Spec = {
        "role": role,
        "worlds": list(worlds),
        "regimes": [regime],
        "ns": list(ns),
        "kappas": [KAPPA],
        "seeds": list(seeds),
        "arms": list(arms),
    }
    if lams is not None:
        block["e4_lams"] = {regime: list(lams)}
    return block


def _x2(name: str, delta: float, n_blocks: int, arms: Mapping[str, Any], budget: float) -> Spec:
    frozen_design = (delta, n_blocks) == (0.10, 20)
    note = (
        f"R-60 X2 (EXPLORATORY; does not change C1-C3): dither delta {delta:g}, "
        f"{n_blocks} setpoint blocks"
        + (" = the frozen EVAL R2 design (reproduction cell)" if frozen_design else "")
    )
    tau_arms = [a for a in X2_ARMS if arms[a]["declare"] == "tau"]
    return {
        "name": f"extras_{name}",
        "budget_cpu_s": budget,
        "note": f"{note}; declared in {PROTOCOL_REL}",
        "extras": {
            "experiment": "X2",
            "design": {"delta": delta, "n_blocks": n_blocks},
            "protocol": PROTOCOL_REL,
        },
        "arms": {a: _arm(arms, a) for a in X2_ARMS},
        "blocks": [
            _block("tune", X2_WORLDS, X2_NS, X2_TUNE, tau_arms),
            _block("measure", X2_WORLDS, X2_NS, X2_MEASURE, X2_ARMS),
        ],
    }


def _x3(arms: Mapping[str, Any], budget: float) -> Spec:
    out = {a: _arm(arms, a) for a in X3_ARMS}
    dither, logged = list(X3_ARMS), list(X3_ARMS)
    for variants, names in ((X3_DITHER, dither), (X3_LOGGED, logged)):
        for tag, told in variants.items():
            for a in X3_ARMS:
                out[f"{a}.{tag}"] = _arm(arms, a, told)
                names.append(f"{a}.{tag}")
    note = (
        "R-60 X3 (EXPLORATORY; does not change C1-C3): data as generated (frozen design), the "
        "design TOLD to the arm misspecified (arm suffix: w = dither width x, s = switch-time "
        "shift, l = E4 R3 lambda x; no suffix = exact); a degradation curve, not evidence "
        f"against C2a; declared in {PROTOCOL_REL}"
    )
    return {
        "name": "extras_x3_told",
        "budget_cpu_s": budget,
        "note": note,
        "extras": {"experiment": "X3", "design": None, "protocol": PROTOCOL_REL},
        "arms": out,
        "blocks": [
            _block("measure", ["E1", "E2"], [X3_N], X3_MEASURE, dither),
            _block("measure", ["E4"], [X3_N], X3_MEASURE, logged, "R3", [1.0]),
        ],
    }


def _x5(
    name: str, cells: Mapping[int, list[float]], seeds: Sequence[int], what: str, **kw: Any
) -> Spec:
    arms, budget = kw["arms"], kw["budget"]
    note = (
        f"X5 (POST HOC, EXPLORATORY; changes no frozen verdict): fresh-data re-run of {what}; "
        f"frozen arms; declared in {PROTOCOL_REL} (Amendments, 2026-10-05)"
    )
    return {
        "name": f"extras_{name}",
        "budget_cpu_s": budget,
        "note": note,
        "extras": {"experiment": "X5", "design": None, "protocol": PROTOCOL_REL},
        "arms": {a: _arm(arms, a) for a in X5_ARMS},
        "blocks": [
            _block("measure", ["E4"], [n], seeds, X5_ARMS, "R3", lams) for n, lams in cells.items()
        ],
    }


def _x6(arms: Mapping[str, Any], budget: float) -> Spec:
    out = {"pmrt_nl_eq": _arm(arms, "pmrt_nl_eq"), "pmrt_eq": _arm(arms, "pmrt_eq")}
    for tag, width in X6_WIDTHS.items():
        for a in ("pmrt_nl_eq", "pmrt_eq"):
            out[f"{a}.{tag}"] = _arm(arms, a, {"width": width})
    for tag, (centre, redraw) in X6_ABLATION.items():
        told = {"width": 2.0, "centre": centre, "redraw": redraw}
        out[f"pmrt_nl_eq.{tag}"] = _arm(arms, "pmrt_nl_eq", told)
    note = (
        "X6 (POST HOC, EXPLORATORY; changes no frozen verdict): PMRT-GBM told a wider dither; "
        "told-width grid (pmrt_nl_eq, pmrt_eq) + 2 x 2 centring law x redraw law at told x2 "
        "(suffix _<centre><redraw>, T = told, t = true; .w200 = TT); bias log per gbm unit; "
        f"declared in {PROTOCOL_REL} (Amendments)"
    )
    return {
        "name": "extras_x6_gbm",
        "budget_cpu_s": budget,
        "note": note,
        "extras": {"experiment": "X6", "design": None, "protocol": PROTOCOL_REL},
        "arms": out,
        "blocks": [_block("measure", [X6_WORLD], [X6_N], X6_SEEDS, list(out))],
    }


def build_specs(frozen_spec: Path = FROZEN_SPEC) -> dict[str, Spec]:
    """The declared specs, keyed by file stem; arms copied from the frozen EVAL spec."""
    frozen = json.loads(frozen_spec.read_text(encoding="utf-8"))
    arms, budget = frozen["arms"], frozen["budget_cpu_s"]
    out = {name: _x2(name, d, nb, arms, budget) for name, (d, nb) in X2_DESIGNS.items()}
    out["x3_told"] = _x3(arms, budget)
    out["x5_fail"] = _x5(
        "x5_fail",
        X5_FAIL,
        X5_SEEDS,
        "the 3 E4 R3 cells where pmrt_eq was INVALID in EVAL",
        arms=arms,
        budget=budget,
    )
    out["x5_adj"] = _x5(
        "x5_adj",
        X5_ADJ,
        X5_ADJ_SEEDS,
        "the other lambdas at the same n (adjacent passing cells)",
        arms=arms,
        budget=budget,
    )
    out["x6_gbm"] = _x6(arms, budget)
    return out


def build_cost_agg(specs: Mapping[str, Spec], dev: Mapping[str, Any]) -> Spec:
    """The DEV cost table of the arms in `specs`.

    The frozen arms' DEV means; a told arm takes its base arm's entries.

    Args:
        specs: Specs whose arms are costed.
        dev: The `cost` entries of the DEV aggregates, `arm|world|regime|...` keyed.
    """
    out = {}
    for spec in specs.values():
        for a, d in spec["arms"].items():
            base = d.get("base", a)
            for k, v in dev.items():
                arm, _, rest = k.partition("|")
                if arm == base:
                    out[f"{a}|{rest}"] = v
    names = ", ".join(f"data/records/study4_dev/dev/{n}/agg.json" for n in DEV_AGGS)
    return {
        "about": f"R-60 extras DEV cost table: 'cost' of {names}; told arms = their base arm",
        "cost": dict(sorted(out.items())),
    }


def dev_costs(records: Path = RECORDS) -> dict[str, Any]:
    """The `cost` entries of the stored DEV aggregates."""
    cost: dict[str, Any] = {}
    for name in DEV_AGGS:
        agg = records / "study4_dev" / "dev" / name / "agg.json"
        cost.update(json.loads(agg.read_text(encoding="utf-8"))["cost"])
    return cost


def _diff(a: Any, b: Any, path: str, rtol: float) -> Iterable[str]:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield f"{path}/{k}: only in {'rebuilt' if k in a else 'frozen'}"
            else:
                yield from _diff(a[k], b[k], f"{path}/{k}", rtol)
    elif isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            yield f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            yield from _diff(x, y, f"{path}[{i}]", rtol)
    elif (
        isinstance(a, float | int)
        and isinstance(b, float | int)
        and not isinstance(a, bool | type(None))
    ):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=0.0):
            yield f"{path}: {a!r} != {b!r}"
    elif a != b:
        yield f"{path}: {a!r} != {b!r}"


def diff_frozen(
    specs: Mapping[str, Spec], directory: Path = FROZEN_DIR, rtol: float = 1e-9
) -> list[str]:
    """Differences between the rebuilt `specs` and the frozen spec files (floats within `rtol`)."""
    out = []
    for name, spec in specs.items():
        path = directory / f"{name}.json"
        if not path.exists():
            out.append(f"{name}: no frozen file")
            continue
        frozen = json.loads(path.read_text(encoding="utf-8"))
        out += list(_diff(spec, frozen, name, rtol))
    return out


def write_all(directory: Path, records: Path = RECORDS) -> list[Path]:
    """Writes the specs and the two DEV cost tables into `directory` (LF, indent 1)."""
    specs = build_specs()
    cost = dev_costs(records)
    written = []
    for name, spec in specs.items():
        written.append(directory / f"{name}.json")
        write_text(written[-1], json.dumps(spec, indent=1) + "\n")
    for file, experiments in ((COST_AGG, ("X2", "X3")), (DIAG_COST_AGG, ("X5", "X6"))):
        part = {k: v for k, v in specs.items() if v["extras"]["experiment"] in experiments}
        written.append(directory / file)
        write_text(written[-1], json.dumps(build_cost_agg(part, cost), indent=1) + "\n")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """`check` (rebuilt specs equal the frozen files, exit 1 otherwise) or `write`."""
    ap = argparse.ArgumentParser(prog="python -m oran_causal.experiments.posthoc.extras_specs")
    ap.add_argument("command", choices=("check", "write"))
    ap.add_argument("--dir", type=Path, default=FROZEN_DIR)
    ap.add_argument("--records", type=Path, default=RECORDS)
    a = ap.parse_args(argv)
    if a.command == "write":
        for p in write_all(a.dir, a.records):
            print(p)
        return 0
    problems = diff_frozen(build_specs(), a.dir)
    cost = dev_costs(a.records)
    for file, experiments in ((COST_AGG, ("X2", "X3")), (DIAG_COST_AGG, ("X5", "X6"))):
        part = {k: v for k, v in build_specs().items() if v["extras"]["experiment"] in experiments}
        frozen = json.loads((a.dir / file).read_text(encoding="utf-8"))
        problems += list(_diff(build_cost_agg(part, cost), frozen, file, 1e-9))
    print("\n".join(problems) or "extras specs equal the frozen files")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
