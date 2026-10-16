"""Episode records of the system-level discovery studies and the unit tables built from them.

A collection episode is one JSONL record of schema `SCHEMA`. Arrays are encoded by `enc`
(base64 of zlib of little-endian float32 C-order bytes, plus the shape). The fields read here are
`units` (c, x, knob, t0, mode, p, applied_mode, step, exp, ctx, optional probs), `pi0_table`,
`lab_series` (per-second per-cell plant series, the unit outcomes), `n_cells`, `seed`, `fold`,
`stage`. The privileged fields `PRIVILEGED_KEYS` are never read by the unit-table builders.

`build_unit_data` turns records into a `UnitTable`: one row per unit of a knob family whose
windows fit in the series. The target of (relation, KPI) is the KPI summed over the relation's
cells in rows t0 .. t0+H-1 minus rows t0-H_pre .. t0-1. Conditioners are the unit's numeric
obs-only context plus the pre-window value of every (relation, KPI).

The cache (`episode_arrays`, `build_cache`) keeps compact per-unit arrays of one record file
for several window lengths; `load_pool` and `Pool.unit_data` rebuild the same `UnitTable` bit
for bit from cache files.
"""

import base64
import dataclasses
import json
import os
import time
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.core.data import UnitTable
from oran_causal.sim.system.decision.gt import relation_cells
from oran_causal.sim.system.decision.vocab import FAMILIES, KPIS, LEVEL_ARR, MODES, RELATIONS

SCHEMA = "e6p-disc-rec/1"
CACHE_SCHEMA = "disc-bench-cache/1"
STEP_S = 10
H_UNIT = 90
HS = (30, 60, 90, 150)
SERIES_FIELDS = ("pv", "v", "e", "rlf", "prb_used", "prb_cap", "ue", "prot_ue")
SERIES_OF = {"pv": "pv", "v": "v", "e": "e", "rlf": "rlf", "load": "ue"}
PRIVILEGED_KEYS = ("gt_static", "gt_labels", "lab_outcome")
EP_FIELDS = ("seed", "sub", "fold", "stage", "policy", "smoke", "n_units_logged", "T")

Record = Mapping[str, Any]
Paths = str | Sequence[str]


def enc(a: np.ndarray) -> dict[str, Any]:
    """Encodes an array as `{"b64", "shape", "dtype": "float32"}`."""
    arr = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    return {
        "b64": base64.b64encode(zlib.compress(arr.tobytes(), 6)).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": "float32",
    }


def dec(d: Mapping[str, Any]) -> np.ndarray:
    """Decodes an `enc` dict to a float32 array of the recorded shape."""
    raw = zlib.decompress(base64.b64decode(d["b64"]))
    return np.frombuffer(raw, dtype="<f4").reshape(d["shape"]).copy()


def unit_record(i: int, u: Mapping[str, Any], n_applied: int) -> dict[str, Any]:
    """Returns the record entry of opened unit `u` at position `i`."""
    ctx = u["ctx"]
    return {
        "i": i,
        "c": int(u["c"]),
        "x": u["x"],
        "x_idx": int(u["x_idx"]),
        "knob": u["knob"],
        "t0": float(u["t0"]),
        "mode": u["mode"],
        "p": float(u["p"]),
        "applied_mode": u.get("applied_mode", u["mode"]),
        "cur": ctx["cur"],
        "prop": ctx["prop"],
        "step": ctx["step"],
        "exp": [int(v) for v in u["exp"]],
        "n_req": int(u["n_req"]),
        "n_applied": n_applied,
        "rb_issued": bool(u.get("rb_issued", False)),
        "ctx": ctx,
        "lab_kpi": u.get("kpi"),
    }


def incumbent_unit_record(i: int, u: Mapping[str, Any], n_applied: int) -> dict[str, Any]:
    """Returns `unit_record` plus the unit's logged probability row and incumbent class."""
    r = unit_record(i, u, n_applied)
    r["probs"] = {m: float(p) for m, p in u["probs"].items()}
    r["inc"] = u.get("inc")
    return r


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _step(u: Mapping[str, Any]) -> float:
    return float(u.get("step", u["ctx"].get("step", 0.0)))


def _probs_row(u: Mapping[str, Any], table: Mapping[str, Mapping[str, float]] | None) -> np.ndarray:
    row = u.get("probs")
    if row is not None:
        return np.array([float(row.get(m, 0.0)) for m in MODES])
    if table is not None:
        tab = table[u["x"]]
        return np.array([float(tab.get(m, 0.0)) for m in MODES])
    return np.array([1.0 if m == u["mode"] else 0.0 for m in MODES])


def _check_row(u: Mapping[str, Any]) -> None:
    row = u["probs"]
    pr = np.array([float(row.get(m, 0.0)) for m in MODES])
    if set(row) - set(MODES) or abs(pr.sum() - 1.0) > 1e-6:
        raise ValueError(f"unit probs row {row} is not a distribution over {MODES}")


def _series(rec: Record, kpis: Sequence[str]) -> tuple[np.ndarray, list[int]]:
    ls = rec["lab_series"]
    data = dec(ls["data"]).astype(np.float64)
    return data, [list(ls["fields"]).index(SERIES_OF[k]) for k in kpis]


def _episode_meta(rec: Record) -> dict[str, Any]:
    return {
        "seed": int(rec["seed"]),
        "stage": rec.get("stage"),
        "fold": rec.get("fold"),
        "policy": rec.get("policy"),
        "smoke": bool(rec.get("smoke")),
        "n_units_logged": len(rec["units"]),
    }


@dataclasses.dataclass
class _Acc:
    kpis: tuple[str, ...]
    relations: tuple[str, ...]
    cols: dict[str, list[Any]] = dataclasses.field(
        default_factory=lambda: {
            k: []
            for k in (
                "episode",
                "seed",
                "fold",
                "family",
                "xapp",
                "c",
                "t0",
                "mode",
                "applied",
                "p",
                "probs",
                "step",
                "sgn",
                "prev",
            )
        }
    )
    ycell: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=dict)
    precell: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=dict)
    masks: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=dict)
    ctxs: list[Mapping[str, Any]] = dataclasses.field(default_factory=list)
    eps: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    dropped: dict[str, int] = dataclasses.field(default_factory=lambda: {"window": 0, "family": 0})
    p_mismatch: int = 0
    n_rows: int = 0
    n_probs_rows: int = 0
    n_cells: int | None = None
    last: dict[tuple[int, str, str], int] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ycell = {k: [] for k in self.kpis}
        self.precell = {k: [] for k in self.kpis}
        self.masks = {r: [] for r in self.relations}


@dataclasses.dataclass(frozen=True)
class _Windows:
    data: np.ndarray
    fidx: list[int]
    H: int = 0
    H_pre: int = 0
    hs: tuple[int, ...] = ()


def _add_unit(acc: _Acc, rec: Record, e: int, u: Mapping[str, Any], w: _Windows) -> None:
    t0 = round(float(u["t0"]))
    post = w.data[t0 : t0 + w.H][:, w.fidx, :].sum(0)
    pre = w.data[t0 - w.H_pre : t0][:, w.fidx, :].sum(0)
    for j, k in enumerate(acc.kpis):
        acc.ycell[k].append(post[j] - pre[j])
        acc.precell[k].append(pre[j])
    nc = int(rec["n_cells"])
    rel = relation_cells(int(u["c"]), u["exp"], nc)
    for r in acc.relations:
        m = np.zeros(nc, bool)
        m[rel[r]] = True
        acc.masks[r].append(m)
    if u.get("probs") is not None:
        _check_row(u)
        acc.n_probs_rows += 1
    pr = _probs_row(u, rec.get("pi0_table"))
    if abs(pr[MODES.index(u["mode"])] - float(u["p"])) > 1e-6:
        acc.p_mismatch += 1
    key = (int(u["c"]), u["x"], u["knob"])
    acc.cols["prev"].append(acc.last.get(key, -1))
    acc.last[key] = acc.n_rows
    step = _step(u)
    values = {
        "episode": e,
        "seed": int(rec["seed"]),
        "fold": -1 if rec.get("fold") is None else int(rec["fold"]),
        "family": FAMILIES.index(u["knob"]),
        "xapp": u["x"],
        "c": int(u["c"]),
        "t0": t0,
        "mode": MODES.index(u["mode"]),
        "applied": MODES.index(u.get("applied_mode", u["mode"])),
        "p": float(u["p"]),
        "probs": pr,
        "step": step,
        "sgn": float(np.sign(step)) or 1.0,
    }
    for k, v in values.items():
        acc.cols[k].append(v)
    acc.ctxs.append(u["ctx"])
    acc.n_rows += 1


def _add_episode(acc: _Acc, e: int, rec: Record, H: int, H_pre: int) -> None:
    nc = int(rec["n_cells"])
    if acc.n_cells is None:
        acc.n_cells = nc
    elif nc != acc.n_cells:
        raise ValueError("episodes with different cell counts cannot be pooled")
    data, fidx = _series(rec, acc.kpis)
    w = _Windows(data, fidx, H, H_pre)
    acc.last = {}
    acc.eps.append(_episode_meta(rec))
    for u in rec["units"]:
        if u["knob"] not in FAMILIES:
            acc.dropped["family"] += 1
            continue
        t0 = round(float(u["t0"]))
        if t0 - H_pre < 0 or t0 + H > data.shape[0]:
            acc.dropped["window"] += 1
            continue
        _add_unit(acc, rec, e, u, w)


def _conditioners(
    ctxs: Sequence[Mapping[str, Any]], pre: Mapping[tuple[str, str], np.ndarray], acc: _Acc
) -> dict[str, np.ndarray]:
    keys = sorted({k for c in ctxs for k, v in c.items() if _is_num(v)})
    z = {
        f"ctx_{k}": np.array(
            [float(c.get(k, np.nan)) if _is_num(c.get(k)) else np.nan for c in ctxs]
        )
        for k in keys
    }
    z.update({f"pre_{r}_{k}": pre[(r, k)] for r in acc.relations for k in acc.kpis})
    return z


def _table(acc: _Acc, H: int, H_pre: int) -> UnitTable:
    n_cells = acc.n_cells or 0
    n = acc.n_rows
    arr = {k: np.asarray(v) for k, v in acc.cols.items() if k not in ("probs", "xapp")}
    probs = np.asarray(acc.cols["probs"], float).reshape(n, len(MODES))
    ycell = {k: np.asarray(v, float).reshape(n, n_cells) for k, v in acc.ycell.items()}
    precell = {k: np.asarray(v, float).reshape(n, n_cells) for k, v in acc.precell.items()}
    masks = {r: np.asarray(v, bool).reshape(n, n_cells) for r, v in acc.masks.items()}
    y = {(r, k): (ycell[k] * masks[r]).sum(1) for r in acc.relations for k in acc.kpis}
    pre = {(r, k): (precell[k] * masks[r]).sum(1) for r in acc.relations for k in acc.kpis}
    z = _conditioners(acc.ctxs, pre, acc)
    return UnitTable(
        episode=arr["episode"].astype(int),
        seed=arr["seed"].astype(int),
        fold=arr["fold"].astype(int),
        family=arr["family"].astype(int),
        xapp=np.asarray(acc.cols["xapp"], dtype=object),
        c=arr["c"].astype(int),
        t0=arr["t0"].astype(int),
        mode=arr["mode"].astype(int),
        applied=arr["applied"].astype(int),
        p=arr["p"].astype(float),
        probs=probs,
        step=arr["step"].astype(float),
        sgn=arr["sgn"].astype(float),
        y=y,
        pre=pre,
        ycell=ycell,
        rel_mask=masks,
        z=z,
        prev=arr["prev"].astype(int),
        n_cells=n_cells,
        H=H,
        H_pre=H_pre,
        relations=acc.relations,
        kpis=acc.kpis,
        families=FAMILIES,
        levels=LEVEL_ARR,
        meta={
            "episodes": acc.eps,
            "dropped": acc.dropped,
            "p_mismatch": acc.p_mismatch,
            "n_units": n,
            "n_probs_rows": acc.n_probs_rows,
            "conditioners": list(z),
        },
    )


def build_unit_data(
    records: Iterable[Record],
    H: int = H_UNIT,
    relations: Sequence[str] = RELATIONS,
    kpis: Sequence[str] = KPIS,
    H_pre: int | None = None,
) -> UnitTable:
    """Builds the unit table of parsed episode records.

    Args:
        records: Episode records of schema `SCHEMA`, in episode order.
        H: Post-window length in seconds.
        relations: Relations to build targets for.
        kpis: KPIs to build targets for.
        H_pre: Pre-window length in seconds; defaults to `H`.

    Returns:
        The pooled unit table; units of other families or whose windows do not fit are dropped
        and counted in `meta["dropped"]`.

    Raises:
        ValueError: Records with different cell counts, or a unit probability row that is not a
            distribution over the modes.
    """
    H = int(H)
    H_pre = H if H_pre is None else int(H_pre)
    acc = _Acc(kpis=tuple(kpis), relations=tuple(relations))
    for e, rec in enumerate(records):
        _add_episode(acc, e, rec, H, H_pre)
    return _table(acc, H, H_pre)


def iter_records(paths: Paths, stages: Iterable[str] | None = None) -> Iterator[dict[str, Any]]:
    """Yields the episode records of JSONL `paths` (a list or a comma-separated string)."""
    stage_set = set(stages) if stages is not None else None
    for path in _path_list(paths):
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or '"episode"' not in line[:200]:
                    continue
                r = json.loads(line)
                if r.get("kind") != "episode" or r.get("schema") != SCHEMA:
                    continue
                if stage_set is not None and r.get("stage") not in stage_set:
                    continue
                yield r


def _path_list(paths: Paths) -> list[str]:
    if isinstance(paths, str):
        return [p for p in paths.split(",") if p]
    return list(paths)


def _empty_arrays(us: Sequence[Mapping[str, Any]], nc: int, hs: Sequence[int], k: int) -> dict:
    n = len(us)
    out: dict[str, Any] = {
        "c": np.zeros(n, np.int16),
        "t0": np.zeros(n, np.int32),
        "mode": np.zeros(n, np.int8),
        "applied": np.zeros(n, np.int8),
        "p": np.zeros(n),
        "probs": np.zeros((n, len(MODES))),
        "step": np.zeros(n),
        "sgn": np.zeros(n),
        "family": np.zeros(n, np.int8),
        "xapp": np.array([str(u["x"]) for u in us], dtype="<U24"),
        "exp": np.zeros((n, nc), bool),
    }
    for h in hs:
        out[f"post_H{h}"] = np.zeros((n, k, nc))
        out[f"pre_H{h}"] = np.zeros((n, k, nc))
    return out


def _fill_unit(out: dict, i: int, u: Mapping[str, Any], table: Any, w: _Windows) -> None:
    t0 = round(float(u["t0"]))
    step = _step(u)
    out["c"][i], out["t0"][i] = int(u["c"]), t0
    out["mode"][i] = MODES.index(u["mode"])
    out["applied"][i] = MODES.index(u.get("applied_mode", u["mode"]))
    out["p"][i], out["probs"][i], out["step"][i] = float(u["p"]), _probs_row(u, table), step
    out["sgn"][i] = float(np.sign(step)) or 1.0
    out["family"][i] = FAMILIES.index(u["knob"])
    out["exp"][i, [int(v) for v in u["exp"]]] = True
    T = w.data.shape[0]
    for h in w.hs:
        if t0 + h <= T and t0 >= 0:
            out[f"post_H{h}"][i] = w.data[t0 : t0 + h][:, w.fidx, :].sum(0)
        if t0 - h >= 0:
            out[f"pre_H{h}"][i] = w.data[t0 - h : t0][:, w.fidx, :].sum(0)


def episode_arrays(
    rec: Record, Hs: Sequence[int] = HS, kpis: Sequence[str] = KPIS
) -> dict[str, Any]:
    """Returns the compact per-unit cache arrays of one episode record.

    Every unit of a knob family gets a row whatever its window. `post_H{h}` and `pre_H{h}` hold
    the per-cell KPI sums over rows t0 .. t0+h-1 and t0-h .. t0-1 (zero where the window does not
    fit). `_ctx` holds the numeric context values and `_ep` the episode metadata.
    """
    nc = int(rec["n_cells"])
    data, fidx = _series(rec, kpis)
    w = _Windows(data, fidx, hs=tuple(Hs))
    table = rec.get("pi0_table")
    us = [u for u in rec["units"] if u["knob"] in FAMILIES]
    out = _empty_arrays(us, nc, Hs, len(kpis))
    ctx = []
    for i, u in enumerate(us):
        _fill_unit(out, i, u, table, w)
        ctx.append({k: (float(v) if _is_num(v) else None) for k, v in u["ctx"].items()})
    out["_ctx"] = ctx
    out["_ep"] = {
        "seed": int(rec["seed"]),
        "sub": str(rec.get("sub") or ""),
        "fold": -1 if rec.get("fold") is None else int(rec["fold"]),
        "stage": str(rec.get("stage")),
        "policy": str(rec.get("policy")),
        "smoke": bool(rec.get("smoke")),
        "n_units_logged": len(rec["units"]),
        "T": data.shape[0],
        "n_cells": nc,
    }
    return out


def _read_parts(paths: Paths, stages: Iterable[str] | None, hs: Sequence[int]) -> list[dict]:
    parts: list[dict] = []
    seen: dict[str, int] = {}
    for rec in iter_records(paths, stages):
        key = json.dumps(rec.get("key")) + str(bool(rec.get("smoke")))
        a = episode_arrays(rec, hs)
        if key in seen:
            parts[seen[key]] = a
        else:
            seen[key] = len(parts)
            parts.append(a)
    return parts


def _ctx_matrix(parts: Sequence[dict], n: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = sorted({k for p in parts for c in p["_ctx"] for k in c})
    kx = {k: j for j, k in enumerate(keys)}
    ctx = np.full((n, len(keys)), np.nan)
    num = np.zeros((n, len(keys)), bool)
    i = 0
    for p in parts:
        for c in p["_ctx"]:
            for k, v in c.items():
                if v is not None:
                    ctx[i, kx[k]] = v
                    num[i, kx[k]] = True
            i += 1
    return ctx, num, keys


def build_cache(
    paths: Paths,
    out_path: str | os.PathLike[str],
    stages: Iterable[str] | None = None,
    Hs: Sequence[int] = HS,
    src: str = "",
) -> dict[str, Any]:
    """Streams JSONL `paths` into one compressed npz cache.

    Episodes are deduplicated by record key, the last one winning.

    Returns:
        Summary `{"out", "episodes", "units", "mb", "wall_s", "stages", "subs"}`.

    Raises:
        ValueError: No episode record of `stages`, or mixed cell counts.
    """
    t = time.time()
    parts = _read_parts(paths, stages, Hs)
    if not parts:
        raise ValueError(f"no episode records (stages {stages}) in {paths}")
    ncs = {p["_ep"]["n_cells"] for p in parts}
    if len(ncs) != 1:
        raise ValueError(f"mixed cell counts {ncs}")
    arrs: dict[str, Any] = {
        name: np.concatenate([p[name] for p in parts])
        for name in [k for k in parts[0] if not k.startswith("_")]
    }
    arrs["ep"] = np.concatenate([np.full(len(p["c"]), e, np.int32) for e, p in enumerate(parts)])
    ctx, num, keys = _ctx_matrix(parts, len(arrs["ep"]))
    arrs["ctx"], arrs["ctx_num"], arrs["ctx_keys"] = ctx, num, np.array(keys, dtype="<U64")
    for f in EP_FIELDS:
        arrs[f"ep_{f}"] = np.array([p["_ep"][f] for p in parts])
    arrs["src"] = np.array(src or os.path.basename(str(paths)))
    arrs["schema"] = np.array(CACHE_SCHEMA)
    arrs["Hs"] = np.array(Hs, np.int32)
    arrs["n_cells"] = np.array(ncs.pop(), np.int32)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrs)
    return {
        "out": str(out_path),
        "episodes": len(parts),
        "units": len(arrs["ep"]),
        "mb": round(out.stat().st_size / 2**20, 1),
        "wall_s": round(time.time() - t, 1),
        "stages": sorted(set(arrs["ep_stage"].tolist())),
        "subs": sorted(set(arrs["ep_sub"].tolist())),
    }


def _prev_rows(ep: np.ndarray, c: np.ndarray, x: np.ndarray, f: np.ndarray) -> np.ndarray:
    prev = np.full(len(ep), -1, int)
    last: dict[tuple[int, int, str, int], int] = {}
    for i in range(len(ep)):
        key = (int(ep[i]), int(c[i]), str(x[i]), int(f[i]))
        prev[i] = last.get(key, -1)
        last[key] = i
    return prev


@dataclasses.dataclass(frozen=True)
class Pool:
    """Concatenated cache files for one (H, H_pre); episode ids are global (0 .. n_eps - 1).

    Attributes:
        H: Post-window length.
        H_pre: Pre-window length.
        n_cells: Number of cells.
        kpis: KPIs of the post / pre arrays.
        eps: Episode metadata arrays (seed, sub, fold, stage, policy, smoke, n_units_logged, T,
            src, file).
        u: Per-unit arrays; `gep` is the global episode id, `post` / `pre` are (n, K, C).
        ctx_keys: Context column names of `u["ctx"]`.
    """

    H: int
    H_pre: int
    n_cells: int
    kpis: tuple[str, ...]
    eps: dict[str, np.ndarray]
    u: dict[str, np.ndarray]
    ctx_keys: list[str]

    @property
    def n_eps(self) -> int:
        """Number of episodes."""
        return len(self.eps["seed"])

    def episodes(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Returns the global episode ids, optionally filtered by a boolean mask."""
        ids = np.arange(self.n_eps)
        return ids if mask is None else ids[np.asarray(mask, bool)]

    def _order(self, episodes: Sequence[int] | None, sort: bool) -> np.ndarray:
        e = self.eps
        ids = np.arange(self.n_eps) if episodes is None else np.asarray(episodes, int)
        if sort:
            key = lambda i: (int(e["seed"][i]), str(e["sub"][i]))  # noqa: E731
            ids = np.array(sorted(ids.tolist(), key=key), int)
        return ids

    def _rows(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = self.u
        rows_all = np.nonzero(pos[u["gep"]] >= 0)[0]
        rows_all = rows_all[np.argsort(pos[u["gep"][rows_all]], kind="stable")]
        t0 = u["t0"][rows_all].astype(int)
        T = self.eps["T"][u["gep"][rows_all]]
        keep = (t0 - self.H_pre >= 0) & (t0 + self.H <= T)
        return rows_all[keep], keep

    def _targets(self, rows: np.ndarray, relations: Sequence[str]) -> tuple[dict, ...]:
        u, n = self.u, len(rows)
        c = u["c"][rows].astype(int)
        own = np.zeros((n, self.n_cells), bool)
        own[np.arange(n), c] = True
        exp = u["exp"][rows]
        rel_all = {"own": own, "nbr": exp & ~own, "far": ~exp}
        masks = {r: rel_all[r] for r in relations}
        post = u["post"][rows].astype(np.float64)
        pre_c = u["pre"][rows].astype(np.float64)
        ycell = {k: post[:, j, :] - pre_c[:, j, :] for j, k in enumerate(self.kpis)}
        precell = {k: pre_c[:, j, :] for j, k in enumerate(self.kpis)}
        y = {(r, k): (ycell[k] * masks[r]).sum(1) for r in relations for k in self.kpis}
        pre = {(r, k): (precell[k] * masks[r]).sum(1) for r in relations for k in self.kpis}
        return y, pre, ycell, masks

    def _z(self, rows: np.ndarray, pre: dict, relations: Sequence[str]) -> dict[str, np.ndarray]:
        num = self.u["ctx_num"][rows]
        cv = self.u["ctx"][rows]
        z = {
            f"ctx_{k}": np.where(num[:, j], cv[:, j], np.nan)
            for j, k in enumerate(self.ctx_keys)
            if num[:, j].any()
        }
        z.update({f"pre_{r}_{k}": pre[(r, k)] for r in relations for k in self.kpis})
        return z

    def _episode_meta(self, ids: np.ndarray) -> list[dict[str, Any]]:
        e = self.eps
        return [
            {
                "seed": int(e["seed"][i]),
                "stage": str(e["stage"][i]),
                "fold": int(e["fold"][i]),
                "sub": str(e["sub"][i]),
                "policy": str(e["policy"][i]),
                "smoke": bool(e["smoke"][i]),
                "n_units_logged": int(e["n_units_logged"][i]),
                "src": str(e["src"][i]),
            }
            for i in ids
        ]

    def unit_data(
        self,
        episodes: Sequence[int] | None = None,
        relations: Sequence[str] = RELATIONS,
        sort: bool = True,
    ) -> UnitTable:
        """Returns the unit table of the given global episode ids (default: all).

        Episodes are ordered by (seed, sub) when `sort` is set, as the study analyzers read
        records; the table then equals `build_unit_data` on those records bit for bit.
        """
        ids = self._order(episodes, sort)
        pos = np.full(self.n_eps, -1)
        pos[ids] = np.arange(len(ids))
        rows, keep = self._rows(pos)
        u = self.u
        gep = u["gep"][rows]
        ep = pos[gep]
        c_, x_, f_ = u["c"][rows], u["xapp"][rows], u["family"][rows]
        y, pre, ycell, masks = self._targets(rows, relations)
        z = self._z(rows, pre, relations)
        mode = u["mode"][rows].astype(int)
        probs = u["probs"][rows]
        n = len(rows)
        p_mismatch = int(np.sum(np.abs(probs[np.arange(n), mode] - u["p"][rows]) > 1e-6))
        return UnitTable(
            episode=ep.astype(int),
            seed=self.eps["seed"][gep].astype(int),
            fold=self.eps["fold"][gep].astype(int),
            family=f_.astype(int),
            xapp=x_.astype(object),
            c=c_.astype(int),
            t0=u["t0"][rows].astype(int),
            mode=mode,
            applied=u["applied"][rows].astype(int),
            p=u["p"][rows].astype(float),
            probs=probs.astype(float),
            step=u["step"][rows].astype(float),
            sgn=u["sgn"][rows].astype(float),
            y=y,
            pre=pre,
            ycell=ycell,
            rel_mask=masks,
            z=z,
            prev=_prev_rows(ep, c_, x_, f_),
            n_cells=self.n_cells,
            H=self.H,
            H_pre=self.H_pre,
            relations=tuple(relations),
            kpis=self.kpis,
            families=FAMILIES,
            levels=LEVEL_ARR,
            meta={
                "episodes": self._episode_meta(ids),
                "dropped": {"window": int((~keep).sum()), "family": None},
                "p_mismatch": p_mismatch,
                "n_units": n,
                "conditioners": list(z),
                "global_episodes": ids.tolist(),
            },
        )


def _check_cache(path: str, z: Any, H: int, H_pre: int) -> None:
    if str(z["schema"]) != CACHE_SCHEMA:
        raise ValueError(f"{path}: schema {z['schema']}")
    if H not in z["Hs"] or H_pre not in z["Hs"]:
        raise ValueError(f"{path}: H {H} / H_pre {H_pre} not cached ({z['Hs'].tolist()})")


def _open_caches(paths: list[str], H: int, H_pre: int) -> tuple[list[Any], int, list[str]]:
    loaded, n_cells, keys = [], None, []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        _check_cache(path, z, H, H_pre)
        nc = int(z["n_cells"])
        if n_cells is None:
            n_cells = nc
        elif nc != n_cells:
            raise ValueError("cache files with different cell counts cannot be pooled")
        loaded.append(z)
        keys.extend(z["ctx_keys"].tolist())
    return loaded, int(n_cells or 0), sorted(set(keys))


_UNIT_KEYS = ("c", "t0", "mode", "applied", "p", "probs", "step", "sgn", "family", "xapp", "exp")


def _append_file(
    path: str,
    z: Any,
    acc: dict[str, dict[str, list]],
    spec: tuple[int, int, list[int], list[str], int],
) -> int:
    H, H_pre, kidx, ctx_keys, off = spec
    eps, us = acc["eps"], acc["u"]
    ne = len(z["ep_seed"])
    for k in EP_FIELDS:
        eps[k].append(z[f"ep_{k}"])
    eps["src"].append(np.full(ne, str(z["src"])))
    eps["file"].append(np.full(ne, os.path.basename(path)))
    us["gep"].append(z["ep"].astype(np.int64) + off)
    for k in _UNIT_KEYS:
        us[k].append(z[k])
    us["post"].append(z[f"post_H{H}"][:, kidx, :])
    us["pre"].append(z[f"pre_H{H_pre}"][:, kidx, :])
    n = len(z["ep"])
    cx = np.full((n, len(ctx_keys)), np.nan)
    cn = np.zeros((n, len(ctx_keys)), bool)
    j = [ctx_keys.index(k) for k in z["ctx_keys"].tolist()]
    cx[:, j], cn[:, j] = z["ctx"], z["ctx_num"]
    us["ctx"].append(cx)
    us["ctx_num"].append(cn)
    return off + ne


def _dedupe(
    eps: dict[str, np.ndarray], us: dict[str, np.ndarray], stages: Iterable[str] | None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    stage_set = set(stages) if stages is not None else None
    seen: set[tuple] = set()
    keep = np.ones(len(eps["seed"]), bool)
    keys = zip(
        eps["seed"].tolist(),
        eps["sub"].tolist(),
        eps["stage"].tolist(),
        eps["smoke"].tolist(),
        strict=True,
    )
    for e, key in enumerate(keys):
        keep[e] = key not in seen and (stage_set is None or key[2] in stage_set)
        seen.add(key)
    if keep.all():
        return eps, us
    new = np.cumsum(keep) - 1
    ku = keep[us["gep"]]
    us = {k: v[ku] for k, v in us.items()}
    us["gep"] = new[us["gep"]]
    return {k: v[keep] for k, v in eps.items()}, us


def load_pool(
    paths: Paths,
    H: int = 90,
    H_pre: int | None = None,
    kpis: Sequence[str] = KPIS,
    stages: Iterable[str] | None = None,
) -> Pool:
    """Concatenates cache npz files for one (H, H_pre).

    Args:
        paths: Cache files (a list or a comma-separated string).
        H: Post-window length; must be cached.
        H_pre: Pre-window length; defaults to `H`; must be cached.
        kpis: KPIs to keep.
        stages: Record stages to keep; None keeps all.

    Returns:
        The pool. An episode (seed, sub, stage, smoke) present in several files is kept once,
        from the first file.

    Raises:
        ValueError: Wrong schema, uncached window, or mixed cell counts.
    """
    plist = _path_list(paths)
    H = int(H)
    H_pre = H if H_pre is None else int(H_pre)
    kidx = [list(KPIS).index(k) for k in kpis]
    loaded, n_cells, ctx_keys = _open_caches(plist, H, H_pre)
    acc: dict[str, dict[str, list]] = {
        "eps": {k: [] for k in (*EP_FIELDS, "src", "file")},
        "u": {k: [] for k in ("gep", *_UNIT_KEYS, "post", "pre", "ctx", "ctx_num")},
    }
    off = 0
    for path, z in zip(plist, loaded, strict=True):
        off = _append_file(path, z, acc, (H, H_pre, kidx, ctx_keys, off))
        z.close()
    eps = {k: np.concatenate(v) for k, v in acc["eps"].items()}
    us = {k: np.concatenate(v) for k, v in acc["u"].items()}
    eps, us = _dedupe(eps, us, stages)
    return Pool(
        H=H, H_pre=H_pre, n_cells=n_cells, kpis=tuple(kpis), eps=eps, u=us, ctx_keys=ctx_keys
    )


def gt_episodes(paths: Paths) -> list[dict[str, Any]]:
    """Returns the decoded GT episodes of JSONL `paths`, deduplicated by (seed, sub) and sorted."""
    eps = [
        {
            "seed": r["seed"],
            "n_cells": r["n_cells"],
            "gt_static": r["gt_static"],
            "gt_labels": [dict(label, delta=dec(label["delta"])) for label in r["gt_labels"]],
            "sub": r.get("sub"),
        }
        for r in iter_records(paths, {"gt"})
    ]
    uniq = {(e["seed"], e.get("sub")): e for e in eps}
    return [uniq[k] for k in sorted(uniq, key=lambda k: (k[0], str(k[1])))]
