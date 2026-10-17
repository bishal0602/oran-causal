"""X8: stricter-threshold check of the frozen v4 PMRT on the K0n null-outcome variants.

The run half (every cyclic shift of every 60-episode group at B 9999) is in
`oran_causal.experiments.posthoc.x8`; this module summarises its per-variant records.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom, norm

from oran_causal.analysis.posthoc.paths import PRIMARY

ALPHAS = (0.05, 0.01, 1e-3, 2e-4, 1e-4)
CRIT = {"var_decl_frac": 0.05, "var_decl_frac_tol": 0.075, "rate_1e3": 2e-3, "rate_2e4": 6e-4}
TOL = 1 + 1e-9
N_BOOT = 2000


def read_variants(directory: str | Path) -> list[dict[str, Any]]:
    """Reads `res_*.jsonl` of a run directory; the first record of a (group, shift) wins.

    Args:
        directory: Directory with the per-part record files.

    Returns:
        Unique variant records in file order.
    """
    rows = []
    for path in sorted(Path(directory).glob("res_*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    seen: set[tuple[int, int]] = set()
    unique = []
    for r in rows:
        if (r["group"], r["shift"]) not in seen:
            seen.add((r["group"], r["shift"]))
            unique.append(r)
    return unique


def _combo(rows: list[dict[str, Any]], c: str) -> dict[str, Any]:
    p = np.array([v[0] for r in rows for v in r["decl"][c].values() if v[0] is not None], float)
    n_decl = [sum(v[1] for v in r["decl"][c].values()) for r in rows]
    o: dict[str, Any] = {
        "n_p": len(p),
        "min_p": float(p.min()),
        "n_decl": int(sum(n_decl)),
        "variants_with_decl": int(sum(x > 0 for x in n_decl)),
    }
    o["var_decl_frac"] = o["variants_with_decl"] / len(rows)
    for al in ALPHAS:
        kk = int(np.sum(p <= al * TOL))
        o[f"n_le_{al:g}"] = kk
        o[f"rate_le_{al:g}"] = kk / len(p)
        o[f"binom_sf_{al:g}"] = float(binom.sf(kk - 1, len(p), al))
    per = np.array(
        [
            [
                sum(1 for v in r["decl"][c].values() if v[0] is not None and v[0] <= 1e-3 * TOL),
                sum(1 for v in r["decl"][c].values() if v[0] is not None),
            ]
            for r in rows
        ],
        float,
    )
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(per), len(per))
        bs.append(per[ix, 0].sum() / max(per[ix, 1].sum(), 1))
    o["rate_le_0.001_boot95"] = float(np.quantile(bs, 0.95))
    return o


def _arms_normal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, list[list[float]]] = {}
    for r in rows:
        for hv in r["hyp"].values():
            for s, sv in hv.get("stats", {}).items():
                if sv.get("z_raw") is None:
                    continue
                arms.setdefault(s, [[], []])
                arms[s][0].append(sv["p2"])
                arms[s][1].append(float(2 * norm.sf(abs(sv["z_raw"]))))
    out = {}
    for s, (pr, pn) in arms.items():
        a, b = np.array(pr), np.array(pn)
        out[s] = {
            "n": len(a),
            **{f"redraw_le_{al:g}": float(np.mean(a <= al * TOL)) for al in ALPHAS},
            **{f"normal_le_{al:g}": float(np.mean(b <= al)) for al in ALPHAS},
        }
    return out


def summarize(directory: str | Path) -> dict[str, Any]:
    """Rates of used p at each threshold per combination, with the pre-stated X8 checks.

    Args:
        directory: Directory with the per-part `res_*.jsonl` files of the run.

    Returns:
        The summary: variants, B, criteria, per-combination rates, the redraw-versus-normal
        comparison of each arm statistic and, when the primary combination ran, its three checks.
    """
    rows = read_variants(directory)
    out: dict[str, Any] = {
        "n_variants": len(rows),
        "B": sorted({r["B"] for r in rows}),
        "criteria": CRIT,
        "combos": {},
    }
    if not rows:
        return out
    for c in rows[0]["decl"]:
        out["combos"][c] = _combo(rows, c)
    out["arms_p2_vs_normal"] = _arms_normal(rows)
    pr = out["combos"].get(PRIMARY)
    if pr:
        out["primary"] = PRIMARY
        out["checks"] = {
            "X8a_var_decl": {
                "value": pr["var_decl_frac"],
                "pass": pr["var_decl_frac"] <= CRIT["var_decl_frac_tol"],
            },
            "X8b_rate_1e3": {
                "value": pr["rate_le_0.001"],
                "pass": pr["rate_le_0.001"] <= CRIT["rate_1e3"],
            },
            "X8c_rate_2e4": {
                "value": pr["rate_le_0.0002"],
                "pass": pr["rate_le_0.0002"] <= CRIT["rate_2e4"],
            },
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Summarises an X8 run directory and optionally writes the summary JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = summarize(a.dir)
    if a.json:
        with Path(a.json).open("w", newline="\n") as fh:
            json.dump(out, fh, indent=1)
    print(
        json.dumps(
            {k: out[k] for k in ("n_variants", "B", "primary", "checks") if k in out}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
