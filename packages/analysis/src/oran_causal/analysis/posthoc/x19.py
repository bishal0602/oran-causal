"""X19: summary of the compute and logging cost records (posthoc PROTOCOL, X19).

Reads the worker records (`res_*.jsonl`) of `oran_causal.experiments.posthoc.x19` and writes
`x19_cost.json` and `x19_cost.md`: per timed cell the median, minimum and maximum CPU seconds of
each phase over the repetitions, the wall seconds and the peak RSS; the referee latencies; the
logging volume. A host other than the X1 reference host is flagged; its costs convert with X1's
speed factors. The run fails (exit 1) when a timed discovery run did not reproduce the stored
hypothesis table, as amendment A1 requires (numbers within relative tolerance 1e-9, declarations
exactly), after writing the files.

    python -m oran_causal.analysis.posthoc.x19 --runs DIR [--out-dir DIR]
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc.paths import RECORDS, write_text
from oran_causal.analysis.posthoc.x1_runtime import REF_HOST, host_type
from oran_causal.analysis.records import iter_records

VERSION = "x19-cost/1"
OUT_DIR = RECORDS / "posthoc_x19"
SIZES = {"pooled": 600, "s60_0": 60, "s120_0": 120, "s300_0": 300}

type Rec = dict[str, Any]


def load(runs: Path) -> list[Rec]:
    """The ok and error records of every `res_*.jsonl` under `runs`, last record per key."""
    by_key: dict[str, Rec] = {}
    for f in sorted(runs.rglob("res_*.jsonl")):
        for r in iter_records(f):
            if "key" in r:
                by_key[r["key"]] = r
    return [by_key[k] for k in sorted(by_key)]


def _stats(xs: Sequence[float | None]) -> dict[str, Any]:
    v = [float(x) for x in xs if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": float(np.median(v)), "min": min(v), "max": max(v)}


def _cell(r: Rec) -> str:
    u = r["unit"]
    return f"{u['kind']}|{u['study']}|{u['slice']}"


def summarise_cell(rs: Sequence[Rec]) -> dict[str, Any]:
    """Phase, total and memory statistics of the repetitions of one cell."""
    ok = [r for r in rs if r.get("status") == "ok"]
    phases = sorted({p for r in ok for p in r.get("phase_cpu_s", {})})
    out: dict[str, Any] = {
        "reps_ok": len(ok),
        "reps_error": len(rs) - len(ok),
        "cpu_s": _stats([r.get("cpu_s") for r in ok]),
        "wall_s": _stats([r.get("wall_s") for r in ok]),
        "phase_cpu_s": {p: _stats([r["phase_cpu_s"].get(p) for r in ok]) for p in phases},
        "peak_rss_mb": _stats([r.get("peak_rss_mb") for r in ok]),
        "rss_scope": sorted({str(r.get("rss_scope")) for r in ok}),
        "hosts": sorted({host_type(r) for r in rs}),
    }
    if ok:
        first = ok[0]
        for k in ("n_episodes", "n_units", "n_tested", "B", "n_contexts", "n_train_units"):
            if k in first:
                out[k] = first[k]
    return out


def repro(rs: Sequence[Rec]) -> dict[str, Any]:
    """Reproduction status of the discovery repetitions of one cell."""
    ok = [r for r in rs if r.get("status") == "ok"]
    rep = [r.get("repro") or {} for r in ok]
    return {
        "exact": bool(ok) and all(x.get("exact") for x in rep),
        "close": bool(ok) and all(x.get("close") for x in rep),
        "declared_equal": bool(ok) and all(r.get("repro_declared") for r in ok),
        "max_n_diff": max((int(x.get("n_diff", 0)) for x in rep), default=None),
        "first_diffs": next((x.get("first") for x in rep if not x.get("exact")), None),
    }


def _latency(rs: Sequence[Rec], arm: str) -> dict[str, Any]:
    ok = [r for r in rs if r.get("status") == "ok" and arm in r]
    return {
        "median_us": _stats([r[arm].get("median_us") for r in ok]),
        "p99_us": _stats([r[arm].get("p99_us") for r in ok]),
        "n": ok[0][arm].get("n") if ok else None,
    }


def summarise(recs: Sequence[Rec]) -> dict[str, Any]:
    """The X19 summary of the worker records."""
    cells: dict[str, list[Rec]] = defaultdict(list)
    for r in recs:
        cells[_cell(r)].append(r)
    disc, fit, referee, logging = {}, {}, {}, {}
    for c, rs in sorted(cells.items()):
        kind, study, sl = c.split("|")
        if kind == "disc":
            disc[f"{study}|{sl}"] = {
                "episodes_planned": SIZES.get(sl),
                **summarise_cell(rs),
                "repro": repro(rs),
            }
        elif kind == "fit":
            ok = [r for r in rs if r.get("status") == "ok"]
            fit = {
                **summarise_cell(rs),
                "fit_check": ok[0].get("fit_check") if ok else None,
                "prior_check": ok[0].get("prior_check") if ok else None,
            }
        elif kind == "referee":
            ok = [r for r in rs if r.get("status") == "ok"]
            referee = {
                **summarise_cell(rs),
                "MG:PMRT": _latency(rs, "MG:PMRT"),
                "CS:PMRT": _latency(rs, "CS:PMRT"),
                "certify_median_us": _stats(
                    [(r.get("certify") or {}).get("median_us") for r in ok]
                ),
                "certify_repro": bool(ok)
                and all((r.get("certify") or {}).get("repro") for r in ok),
            }
        elif kind == "logging":
            ok = [r for r in rs if r.get("status") == "ok"]
            logging = ok[-1].get("volume", {}) if ok else {"error": "no ok logging record"}
    hosts = sorted({host_type(r) for r in recs})
    return {
        "version": VERSION,
        "records": len(recs),
        "errors": sorted(r["key"] for r in recs if r.get("status") != "ok"),
        "hosts": hosts,
        "reference_host": REF_HOST,
        "on_reference_host": hosts == [REF_HOST],
        "reproduced": bool(disc)
        and all(v["repro"]["close"] and v["repro"]["declared_equal"] for v in disc.values()),
        "reproduced_exactly": bool(disc) and all(v["repro"]["exact"] for v in disc.values()),
        "disc": disc,
        "fit": fit,
        "referee": referee,
        "logging": logging,
    }


def _fmt(s: Mapping[str, Any] | None, nd: int = 1) -> str:
    if not s or not s.get("n"):
        return "-"
    return f"{s['median']:.{nd}f} [{s['min']:.{nd}f}, {s['max']:.{nd}f}]"


def markdown(s: Mapping[str, Any]) -> str:
    """The report text of a summary."""
    lines = [
        "# X19: compute and logging cost of PMRT and the referee",
        "",
        f"Records {s['records']}; errors {len(s['errors'])}; hosts {', '.join(s['hosts'])}.",
        f"Reference host ({s['reference_host']}): {'yes' if s['on_reference_host'] else 'NO'}.",
        "Stored PMRT hypothesis tables reproduced (A1: numbers within relative 1e-9, declarations"
        f" exact): {'yes' if s['reproduced'] else 'NO'}; bit-exact:"
        f" {'yes' if s['reproduced_exactly'] else 'NO'}.",
        "",
        "## Discovery (frozen PMRT, B 9999; CPU-s median [min, max] over repetitions)",
        "",
        "| study | slice | episodes | units | load | features | test | declare | compute | "
        "wall | peak RSS MB | exact |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, v in s["disc"].items():
        study, sl = k.split("|")
        ph = v["phase_cpu_s"]
        lines.append(
            f"| {study} | {sl} | {v.get('n_episodes', '-')} | {v.get('n_units', '-')} | "
            f"{_fmt(ph.get('load'))} | {_fmt(ph.get('features'))} | {_fmt(ph.get('test'))} | "
            f"{_fmt(ph.get('declare'), 2)} | {_fmt(v['cpu_s'])} | {_fmt(v['wall_s'])} | "
            f"{_fmt(v['peak_rss_mb'], 0)} | {'yes' if v['repro']['exact'] else 'NO'} |"
        )
    f = s.get("fit") or {}
    if f:
        ph = f["phase_cpu_s"]
        fc, pc = f.get("fit_check") or {}, f.get("prior_check") or {}
        lines += [
            "",
            "## Training fit (steps 1-7) and prior run (step 8)",
            "",
            f"Training units {f.get('n_train_units', '-')}. CPU-s: fit {_fmt(ph.get('fit'))}; "
            f"features {_fmt(ph.get('features'))}; prior features "
            f"{_fmt(ph.get('prior_features'))}; prior test {_fmt(ph.get('prior_test'))}; "
            f"total {_fmt(f['cpu_s'])}; peak RSS {_fmt(f['peak_rss_mb'], 0)} MB.",
            f"Fitted parameters against the artifact (information only): {fc.get('exact')} of "
            f"{fc.get('hypotheses')} hypotheses exact, {fc.get('close_1e-6')} within 1e-6. "
            f"Prior z exact: {pc.get('exact')} (close {pc.get('close')}).",
        ]
    r = s.get("referee") or {}
    if r:
        lines += [
            "",
            "## Referee",
            "",
            f"Contexts {r.get('n_contexts', '-')}. Per-decision latency, microseconds (median "
            "of the repetitions' medians and 99th percentiles):",
            "",
            f"- MapGateV2(MG:PMRT): median {_fmt(r['MG:PMRT']['median_us'], 2)}, "
            f"p99 {_fmt(r['MG:PMRT']['p99_us'], 2)}",
            f"- CertSafeMapGateV2(CS:PMRT): median {_fmt(r['CS:PMRT']['median_us'], 2)}, "
            f"p99 {_fmt(r['CS:PMRT']['p99_us'], 2)}",
            f"- One gate certification: {_fmt(r['certify_median_us'], 1)} us; uncertified set "
            f"as stored: {'yes' if r.get('certify_repro') else 'NO'}",
        ]
    lg = s.get("logging") or {}
    if lg and "error" not in lg:
        lines += [
            "",
            "## Logging volume",
            "",
            "| pool | episodes | hours | units/ep-h | units/cell-h | unit bytes (fixed) | "
            "unit bytes (compressed) | units/cell-day (projection) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, v in lg.items():
            if name == "series":
                continue
            ra, sz, pr = v["rates"], v["unit_record"], v["projection"]
            lines.append(
                f"| {name} | {ra['episodes']} | {ra['hours']:.1f} | "
                f"{ra['units_per_ep_hour']:.1f} | {ra['units_per_cell_hour']:.2f} | "
                f"{sz['fixed_bytes']} | {sz['compressed_bytes_per_unit']:.1f} | "
                f"{pr['units_per_cell_day']:.1f} |"
            )
        se = lg.get("series") or {}
        eps = se.get("episodes") or []
        if eps:
            raw = np.median([e["raw_bytes_per_cell_s"] for e in eps])
            sto = np.median([e["stored_bytes_per_cell_s"] for e in eps])
            lines += [
                "",
                f"KPI series: {se['needed_f32_bytes_per_cell_s']} bytes per cell-second for the "
                f"{len(se['kpis_needed'])} KPIs PMRT reads (float32); "
                f"{se['all_f32_bytes_per_cell_s']} for all {len(se['fields_all'])} stored fields; "
                f"stored records: {raw:.1f} raw and {sto:.1f} encoded bytes per cell-second "
                f"(median of {len(eps)} episodes).",
            ]
        lines += ["", "Per-cell-day figures are arithmetic on the simulated scenario's rates only."]
    if s["errors"]:
        lines += ["", "## Errors", "", *[f"- {k}" for k in s["errors"]]]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Writes the summary; exit 1 when a stored hypothesis table did not reproduce under A1."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    s = summarise(load(Path(a.runs)))
    out = Path(a.out_dir)
    write_text(out / "x19_cost.json", json.dumps(s, indent=1, default=str) + "\n")
    write_text(out / "x19_cost.md", markdown(s))
    print(f"[x19] {out}: {s['records']} records, reproduced {s['reproduced']}")
    return 0 if s["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
