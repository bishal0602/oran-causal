"""X8: stricter-threshold check of the frozen v4 PMRT on the K0n null-outcome variants (run half).

The frozen PMRT artifact is run on the K0n null-outcome variants of the Study 1 EVAL episodes,
extended from the frozen 4 cyclic shifts per group to every shift k = 1..G-1, and run at the frozen
discovery B (9999) instead of the K0n B (999), so the declaration layer can act at its own
thresholds. The recipe is the Study 1 K0n recipe: groups are the 10 EVAL 60-episode slices,
episode i of a group takes the `lab_series` of episode (i + k) mod G, the RNG split is the null
one and n_target is G. Summarising the records is `oran_causal.analysis.posthoc.x8`.

    python -m oran_causal.experiments.posthoc.x8 run --eval DIR --out OUT/res_I.jsonl --part I/N
        [--B 9999] [--shifts 1-59] [--groups 0-9]

A run appends one JSON line per variant `{group, shift, B, n_units, wall_s, hyp, decl}`; variants
already in the file are skipped, so a cut run resumes.
"""

import argparse
import json
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.experiments.system import study1_discovery as s1
from oran_causal.methods.system import artifact as pmrt_artifact
from oran_causal.methods.system import pmrt as pm

GROUP = 60
Log = Callable[[str], None]


def rng_list(s: str) -> list[int]:
    """Expands `"0-9,12"` to `[0, 1, ..., 9, 12]`."""
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def variants(groups: Sequence[int], shifts: Sequence[int]) -> list[tuple[int, int]]:
    """The `(group, shift)` variants, group-major."""
    return [(g, k) for g in groups for k in shifts]


def done_keys(path: Path) -> set[tuple[int, int]]:
    """The `(group, shift)` variants already in a result file."""
    keys: set[tuple[int, int]] = set()
    if path.is_file():
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
                keys.add((r["group"], r["shift"]))
            except json.JSONDecodeError, KeyError:
                continue
    return keys


def shard(todo: Sequence[tuple[int, int]], index: int, count: int) -> list[tuple[int, int]]:
    """The variants of shard `index` of `count`: variant j belongs to it iff j % count == index."""
    return [v for j, v in enumerate(todo) if j % count == index]


def groups_of(seeds: Sequence[int], size: int = GROUP) -> dict[int, list[int]]:
    """Record indices of each EVAL group of `size` consecutive seeds."""
    base, n = s1.SEEDS["eval"]
    return {
        gi: [j for j, s in enumerate(seeds) if (s - base) // size == gi] for gi in range(n // size)
    }


def hyp_block(runr: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    """Status and per-statistic test outputs of every hypothesis of a run."""
    hyp: dict[str, Any] = {}
    for h, o in runr.items():
        if o.get("status") != "tested":
            hyp[h] = {"status": o.get("status")}
            continue
        hyp[h] = {
            "status": "tested",
            "stats": {
                s: {x: o[s].get(x) for x in ("p2", "p_plus", "p_minus", "z_raw", "z")}
                for s in pm.STATS
                if s in o
            },
        }
    return hyp


def decl_block(declared: Mapping[Any, Mapping[Any, Mapping[str, Any]]]) -> dict[str, Any]:
    """`{combination: {hypothesis: [p, declared, sign]}}` of the determined hypotheses."""
    decl = {}
    for (s, lay), d in declared.items():
        decl[pm.cname(s, lay)] = {
            "|".join(h): [
                float(v["p"]) if np.isfinite(v.get("p", np.nan)) else None,
                bool(v.get("declared")),
                int(v.get("sign", 0) or 0),
            ]
            for h, v in d.items()
            if v.get("status") != "undetermined"
        }
    return decl


def run_variant(
    recs: Sequence[dict],
    idx: Sequence[int],
    k: int,
    params: Mapping[str, Any],
    priors: Mapping[str, Any],
    cfg: Any,
    tmp: Path,
) -> dict[str, Any]:
    """PMRT on one null-outcome variant: group `idx` with its `lab_series` shifted by `k`."""
    sub = [recs[j] for j in idx]
    rr = [dict(r, lab_series=sub[(j + k) % len(sub)]["lab_series"]) for j, r in enumerate(sub)]
    pd = pm.pmrt_data_from_records(rr, tmp)
    runr = pm.run_integrated(pd, params, cfg, s1.SPLIT["null"])
    return {
        "n_units": int(pd.n),
        "hyp": hyp_block(runr),
        "decl": decl_block(pm.declare_all(runr, priors, len(sub))),
    }


def run(
    eval_files: Sequence[Path],
    out: Path,
    *,
    part: tuple[int, int] = (0, 1),
    B: int = 9999,
    group: int = GROUP,
    groups: Sequence[int] = tuple(range(10)),
    shifts: Sequence[int] = tuple(range(1, 60)),
    artifact: Path = s1.ARTIFACT,
    log: Log = print,
) -> int:
    """Runs this shard's variants into `out`; returns the number of variants run.

    Raises:
        SystemExit: The artifact does not match its pin, or the EVAL records are incomplete.
    """
    status = s1.artifact_status(artifact)
    if not status["ok"]:
        raise SystemExit(f"artifact sha256 / verification mismatch: {status}")
    params, priors = pmrt_artifact.load_artifact(artifact)
    cfg = pmrt_artifact.pmrt_config(params, B)
    have = done_keys(out)
    todo = [v for v in shard(variants(groups, shifts), *part) if v not in have]
    log(
        f"part {part[0]}/{part[1]}: {len(todo)} variants to run ({len(have)} done), B {B}, "
        "artifact sha ok"
    )
    if not todo:
        return 0
    recs = s1.load_records_light(eval_files, "eval", False)
    seeds = [int(r["seed"]) for r in recs]
    e0, en = s1.SEEDS["eval"]
    if sorted(seeds) != list(range(e0, e0 + en)):
        raise SystemExit(f"eval_v4 incomplete: {len(seeds)} records")
    members = groups_of(seeds, group)
    tmp = Path(tempfile.mkdtemp(prefix="k0n_tail_"))
    out.parent.mkdir(parents=True, exist_ok=True)
    primary = pm.cname(*s1.PRIMARY)
    for gi, k in todo:
        t = time.time()
        body = run_variant(recs, members[gi], k, params, priors, cfg, tmp)
        full = {"group": gi, "shift": k, "B": B, "wall_s": round(time.time() - t, 1), **body}
        order = ("group", "shift", "B", "n_units", "wall_s", "hyp", "decl")
        rec: dict[str, Any] = {key: full[key] for key in order}
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        pc = rec["decl"][primary]
        ps = [v[0] for v in pc.values() if v[0] is not None]
        log(
            f"  g{gi} shift{k}: units {rec['n_units']}, "
            f"primary decl {sum(v[1] for v in pc.values())}, "
            f"min p {min(ps or [float('nan')]):.2g} ({rec['wall_s']} s)"
        )
    return len(todo)


def expand(spec: str) -> list[Path]:
    """The `res_*.jsonl` shards of comma-separated directories or files."""
    out: list[Path] = []
    for p in (Path(x) for x in spec.split(",") if x):
        if p.is_dir():
            out += sorted(f for f in p.rglob("res_*.jsonl") if "bundle" not in f.parts)
        else:
            out.append(p)
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--eval", required=True, help="EVAL shard files or directories, comma-separated")
    r.add_argument("--out", required=True)
    r.add_argument("--part", default="0/1")
    r.add_argument("--B", type=int, default=9999)
    r.add_argument("--group", type=int, default=GROUP)
    r.add_argument("--groups", default="0-9")
    r.add_argument("--shifts", default="1-59")
    r.add_argument("--artifact", default=str(s1.ARTIFACT))
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: `run` a shard of the X8 variants."""
    a = _parser().parse_args(argv)
    i, n = (int(x) for x in a.part.split("/"))
    run(
        expand(a.eval),
        Path(a.out),
        part=(i, n),
        B=a.B,
        group=a.group,
        groups=rng_list(a.groups),
        shifts=rng_list(a.shifts),
        artifact=Path(a.artifact),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
