"""Versioned per-second propose/apply/outcome trace of an episode (`E6Env(..., trace=True)`).

Schema `e6-trace/1`. One row per control second t (t = 1 .. total_s). Arrays are columnar; the
variable-length tables (requests, rollbacks, writes, locks, config deltas, KPM reports) carry their
own time column.

Runtime arrays (what a near-RT RIC arbiter sees at second t, before or while it decides):

- `t` (T,) i4, `scored` (T,) bool: second index; scored means the second counts in the episode
  score (after warm-up).
- `n_req` (T,) i2: requests offered this second, re-offered deferred ones included.
- `n_locked` (T,) i2: WG3 locks active at decision time.
- `churn_cap` (T,) i4: churn cap in force for this second's apply (-1 means none).
- `changes` (T,) i4: cumulative applied knob changes after this second's apply (warm-up
  included).
- `stats` (T, S) i4: cumulative `env.stats` after apply, columns `meta["stats"]`.
- `cfg_snap_t` (P,), `cfg_snap` (P, K) f8: full pre-action configuration every `snapshot_s`
  seconds (first second included).
- `cfg_d_t`, `cfg_d_k`, `cfg_d_v`: knobs whose pre-action value changed since the previous second
  (any cause: RIC apply, OAM scenario rollout); knob index into `meta["knobs"]`.
- `rq_*` (Q,): one row per offered request: t, xapp (index into `meta["xapps"]`), ver, knob, cur,
  prop, age (deferral age, 0 for fresh), dec (`meta["dec_codes"]`), dec_val (modify value or lock
  seconds, NaN otherwise), out (`meta["out_codes"]`: ok is ACK and changed, noop is ACK already
  in force, actuator / churn / locked are NACK by actuator limit / churn cap / active lock,
  reject / lock_set / expired are arbiter reject / arbiter lock / deferral over 10 s, deferred is
  parked), applied (value in force after the decision).
- `rb_t`, `rb_k`, `rb_out`: rollback requests (`meta["rb_codes"]`).
- `wr_t`, `wr_k`, `wr_v`, `wr_out`: arbiter writes attempted (`meta["wr_codes"]`).
- `lk_t`, `lk_k`, `lk_until`: WG3 locks set.
- `kpm_<gran>_{rx,t0,t1,arrived}`: delivered KPM reports per granularity (rx is the second
  delivered to the RIC); `kpm_<gran>_<field>` (R_g, ...) f4 is the report payload, fields in
  `meta["kpm_fields"]`.

Label arrays (privileged plant counters, prefix `lab_`, never features). Row i is the window
(t[i] - 1, t[i]], the second that elapsed before the decisions of t[i]; cells are the serving cell
at the end of that second:

- `lab_viol` (T, C, 3) u2: violated UE-s per cell per slice (LL, eMBB, BE), every second.
- `lab_outage` (T, C) u2, `lab_ue` (T, C, 3) u2: outage UE-s; UEs served per cell per slice.
- `lab_rlf` (T, C) u2, `lab_energy_j` (T, C) f4: radio link failures, energy per cell.
- `lab_severe` (T, C, 3) u1: severe incidents started (cell-slice over 20 % violated for 10 s;
  scored seconds only).

The episode score counts viol, outage, energy and severe over `scored` rows only
(`plant.sla["rlf"]` counts all seconds).
"""

import dataclasses
import json
from collections.abc import Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oran_causal.sim.system.env import E6Env

SCHEMA = "e6-trace/1"
DEC_CODES = ("accept", "reject", "modify", "defer", "lock")
OUT_CODES = (
    "ok",
    "noop",
    "actuator",
    "churn",
    "locked",
    "reject",
    "lock_set",
    "deferred",
    "expired",
)
ACK_CODES = ("ok", "noop")
RB_CODES = ("applied", "noop", "actuator", "churn")
WR_CODES = ("ok", "actuator", "churn")
LABEL_PREFIX = "lab_"
OUT = {c: i for i, c in enumerate(OUT_CODES)}
_REPORT_TIMES = ("t0", "t1", "arrived")
_RQ_DTYPES = (
    ("t", np.int32),
    ("xapp", np.int8),
    ("ver", np.int8),
    ("knob", np.int16),
    ("cur", float),
    ("prop", float),
    ("age", np.int8),
    ("dec", np.int8),
    ("dec_val", float),
    ("out", np.int8),
    ("applied", float),
)
_LABEL_SUMS = (
    ("viol", "lab_viol"),
    ("outage", "lab_outage"),
    ("ue_s", "lab_ue"),
    ("rlf", "lab_rlf"),
    ("energy_j", "lab_energy_j"),
    ("severe", "lab_severe"),
)


def _dec_code(d: Any) -> tuple[int, float]:
    if isinstance(d, tuple):
        return DEC_CODES.index(d[0]), float(d[1])
    return DEC_CODES.index(d), np.nan


class Trace:
    """Columnar episode trace.

    Args:
        arrays: array name to array.
        meta: JSON-able metadata with a `schema` entry.
    """

    def __init__(self, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        self.arrays, self.meta = arrays, meta

    def __getitem__(self, k: str) -> np.ndarray:
        return self.arrays[k]

    def __contains__(self, k: str) -> bool:
        return k in self.arrays

    def features(self) -> dict[str, np.ndarray]:
        """Return the runtime arrays (everything but the labels)."""
        return {k: v for k, v in self.arrays.items() if not k.startswith(LABEL_PREFIX)}

    def labels(self) -> dict[str, np.ndarray]:
        """Return the privileged label arrays."""
        return {k: v for k, v in self.arrays.items() if k.startswith(LABEL_PREFIX)}

    def to_npz(self, path: str | PathLike[str]) -> None:
        """Write the trace to a compressed `.npz` file."""
        arrays: dict[str, Any] = {"__meta__": np.array(json.dumps(self.meta)), **self.arrays}
        np.savez_compressed(path, **arrays)

    @classmethod
    def from_npz(cls, path: str | PathLike[str]) -> Trace:
        """Read a trace written by `to_npz`.

        Raises:
            ValueError: the file does not carry an `e6-trace/` schema.
        """
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            if not str(meta.get("schema", "")).startswith("e6-trace/"):
                raise ValueError(f"not an E6 trace: schema {meta.get('schema')!r}")
            return cls({k: z[k] for k in z.files if k != "__meta__"}, meta)

    def row(self, t: float) -> int:
        """Return the row index of second t."""
        return int(np.searchsorted(self.arrays["t"], t))

    def config_at(self, t: float) -> np.ndarray:
        """Return the pre-action configuration at second t (`meta["knobs"]` order).

        Raises:
            ValueError: no configuration snapshot exists at or before t.
        """
        a = self.arrays
        i = int(np.searchsorted(a["cfg_snap_t"], t, side="right")) - 1
        if i < 0:
            raise ValueError(f"no configuration snapshot at or before t={t}")
        v = a["cfg_snap"][i].copy()
        m = (a["cfg_d_t"] > a["cfg_snap_t"][i]) & (a["cfg_d_t"] <= t)
        v[a["cfg_d_k"][m]] = a["cfg_d_v"][m]
        return v


def region_labels(
    trace: Trace,
    regions: Sequence[int] | np.ndarray,
    t0: float,
    t1: float,
    scored_only: bool = False,
) -> dict[str, np.ndarray]:
    """Sum the per-cell labels over the window (t0, t1] and group cells by region.

    Args:
        trace: episode trace.
        regions: region id per cell.
        t0: window start (exclusive).
        t1: window end (inclusive).
        scored_only: keep scored seconds only.

    Returns:
        `region_ids` (R,) and, per region, `viol` (R, 3), `outage`, `ue_s` (R, 3), `rlf`,
        `energy_j` and `severe` (R, 3). Privileged labels.
    """
    a = trace.arrays
    reg = np.asarray(regions)
    ids = np.array(sorted({int(x) for x in reg}))
    m = (a["t"] > t0) & (a["t"] <= t1)
    if scored_only:
        m &= a["scored"]
    onehot = (reg[None, :] == ids[:, None]).astype(float)
    out = {"region_ids": ids}
    for name, key in _LABEL_SUMS:
        s = a[key][m].astype(float).sum(0)
        out[name] = np.tensordot(onehot, s, axes=(1, 0))
    return out


class TraceRecorder:
    """Recorder the env feeds through its hooks when `trace=True`; `env.copy()` drops it.

    Args:
        env: the episode being recorded.
        snapshot_s: seconds between full configuration snapshots.
    """

    def __init__(self, env: E6Env, snapshot_s: int = 60) -> None:
        self.env, self.snapshot_s = env, int(snapshot_s)
        self.kidx = {k: i for i, k in enumerate(env.knobs)}
        self.xidx = {x.name: i for i, x in enumerate(env.xapps)}
        self.stat_keys = list(env.stats)
        row_keys = ("t", "scored", "n_req", "n_locked", "churn_cap", "changes", "stats")
        self.rows: dict[str, list[Any]] = {k: [] for k in row_keys}
        self.lab: dict[str, list[np.ndarray]] = {key: [] for _, key in _LABEL_SUMS}
        self.snap_t: list[int] = []
        self.snap: list[np.ndarray] = []
        self.cd: tuple[list[int], list[int], list[float]] = ([], [], [])
        self.rq: dict[str, list[Any]] = {k: [] for k, _ in _RQ_DTYPES}
        self.rb: tuple[list[Any], ...] = ([], [], [])
        self.wr: tuple[list[Any], ...] = ([], [], [], [])
        self.lk: tuple[list[Any], ...] = ([], [], [])
        self.kpm: dict[str, dict[str, list[Any]]] = {}
        self.last_cfg: np.ndarray | None = None
        self._scored = False

    def plant_second(self) -> None:
        """Record the elapsed second's labels (after its ticks, before the KPM reads them)."""
        p = self.env.plant
        nc, cell = p.nc, p.serv
        viol = np.zeros((nc, 3), np.uint16)
        ue = np.zeros((nc, 3), np.uint16)
        for s_, flag in enumerate((p.sec_viol_ll, p.sec_viol_embb, p.sec_viol_be)):
            viol[:, s_] = np.bincount(cell[flag], minlength=nc)
            ue[:, s_] = np.bincount(cell[p.sl == s_], minlength=nc)
        self._scored = bool(p.sec_scored)
        lab = self.lab
        lab["lab_viol"].append(viol)
        lab["lab_ue"].append(ue)
        lab["lab_outage"].append(np.bincount(cell[p.sec_out], minlength=nc).astype(np.uint16))
        lab["lab_rlf"].append(p.ctr["rlf"].astype(np.uint16))
        lab["lab_energy_j"].append(p.ctr["energy_j"].astype(np.float32))
        lab["lab_severe"].append(((p.cell_viol_run == 10) & self._scored).astype(np.uint8))

    def begin(self, now: float, new_reports: list[dict[str, Any]], reqs: list[Any]) -> None:
        """Record the pre-action configuration, locks, churn cap and delivered reports."""
        env = self.env
        t = int(now)
        self._config(t, np.array(list(env.config().values()), float))
        rows = self.rows
        rows["t"].append(t)
        rows["scored"].append(self._scored)
        rows["n_req"].append(len(reqs))
        rows["n_locked"].append(sum(1 for u in env.lock_until.values() if now < u))
        rows["churn_cap"].append(-1 if env.churn_cap is None else int(env.churn_cap))
        for rep in new_reports:
            self._report(t, rep)

    def _config(self, t: int, cfg: np.ndarray) -> None:
        if self.last_cfg is None or (t - 1) % self.snapshot_s == 0:
            self.snap_t.append(t)
            self.snap.append(cfg)
        if self.last_cfg is not None:
            for k in np.nonzero(np.abs(cfg - self.last_cfg) > 1e-12)[0]:
                self.cd[0].append(t)
                self.cd[1].append(int(k))
                self.cd[2].append(float(cfg[k]))
        self.last_cfg = cfg

    def _report(self, t: int, rep: dict[str, Any]) -> None:
        g = self.kpm.setdefault(rep["gran"], {"rx": [], "t0": [], "t1": [], "arrived": []})
        g["rx"].append(t)
        for k in _REPORT_TIMES:
            g[k].append(float(rep[k]))
        for k, v in rep.items():
            if k != "gran" and k not in _REPORT_TIMES:
                g.setdefault(k, []).append(np.asarray(v, np.float32))

    def request(  # noqa: PLR0913, PLR0917
        self, now: float, r: dict[str, Any], d: Any, out: str, applied: float, age: int
    ) -> None:
        """Record one offered request, its decision and its outcome."""
        dc, dv = _dec_code(d)
        values = (
            int(now),
            self.xidx[r["xapp"]],
            int(r["ver"]),
            self.kidx[r["knob"]],
            float(r["cur"]),
            float(r["prop"]),
            int(age),
            dc,
            dv,
            OUT[out],
            float(applied),
        )
        for (k, _), x in zip(_RQ_DTYPES, values, strict=True):
            self.rq[k].append(x)

    def rollback(self, now: float, k: Any, out: str) -> None:
        """Record one rollback request and its outcome."""
        self.rb[0].append(int(now))
        self.rb[1].append(self.kidx[k])
        self.rb[2].append(RB_CODES.index(out))

    def write(self, now: float, k: Any, v: float, out: str) -> None:
        """Record one arbiter write and its outcome."""
        row = (int(now), self.kidx[k], float(v), WR_CODES.index(out))
        for lst, x in zip(self.wr, row, strict=True):
            lst.append(x)

    def lock(self, now: float, k: Any, until: float) -> None:
        """Record one WG3 lock."""
        for lst, x in zip(self.lk, (int(now), self.kidx[k], float(until)), strict=True):
            lst.append(x)

    def end(self, now: float) -> None:
        """Record the cumulative counters after the second's apply."""
        self.rows["changes"].append(self.env.stats["changes"])
        self.rows["stats"].append([self.env.stats[k] for k in self.stat_keys])

    def finish(self) -> Trace:
        """Return the recorded trace."""
        a = self._row_arrays()
        for k, dt in _RQ_DTYPES:
            a["rq_" + k] = np.array(self.rq[k], dt)
        self._event_arrays(a)
        kpm_fields = self._kpm_arrays(a)
        n = len(self.rows["t"])
        for k, lst in self.lab.items():
            a[k] = np.stack(lst[:n])
        return Trace(a, self._meta(kpm_fields))

    def _row_arrays(self) -> dict[str, np.ndarray]:
        rows, n_knobs = self.rows, len(self.env.knobs)
        return {
            "t": np.array(rows["t"], np.int32),
            "scored": np.array(rows["scored"], bool),
            "n_req": np.array(rows["n_req"], np.int16),
            "n_locked": np.array(rows["n_locked"], np.int16),
            "churn_cap": np.array(rows["churn_cap"], np.int32),
            "changes": np.array(rows["changes"], np.int32),
            "stats": np.array(rows["stats"], np.int32).reshape(-1, len(self.stat_keys)),
            "cfg_snap_t": np.array(self.snap_t, np.int32),
            "cfg_snap": np.array(self.snap, float).reshape(-1, n_knobs),
            "cfg_d_t": np.array(self.cd[0], np.int32),
            "cfg_d_k": np.array(self.cd[1], np.int16),
            "cfg_d_v": np.array(self.cd[2], float),
        }

    def _event_arrays(self, a: dict[str, np.ndarray]) -> None:
        tables = (
            ("rb_", self.rb, ("t", "k", "out"), (np.int32, np.int16, np.int8)),
            ("wr_", self.wr, ("t", "k", "v", "out"), (np.int32, np.int16, float, np.int8)),
            ("lk_", self.lk, ("t", "k", "until"), (np.int32, np.int16, float)),
        )
        for pre, src, cols, dts in tables:
            for c, lst, dt in zip(cols, src, dts, strict=True):
                a[pre + c] = np.array(lst, dt)

    def _kpm_arrays(self, a: dict[str, np.ndarray]) -> dict[str, list[str]]:
        kpm_fields = {}
        for g, d in self.kpm.items():
            kpm_fields[g] = [k for k in d if k != "rx" and k not in _REPORT_TIMES]
            a[f"kpm_{g}_rx"] = np.array(d["rx"], np.int32)
            for k in _REPORT_TIMES:
                a[f"kpm_{g}_{k}"] = np.array(d[k], float)
            for k in kpm_fields[g]:
                a[f"kpm_{g}_{k}"] = np.stack(d[k])
        return kpm_fields

    def _meta(self, kpm_fields: dict[str, list[str]]) -> dict[str, Any]:
        env = self.env
        cfg = dataclasses.asdict(env.cfg)
        return {
            "schema": SCHEMA,
            "cfg": json.loads(json.dumps(cfg, default=str)),
            "wg3": bool(env.wg3),
            "knobs": [[k[0], *map(int, k[1:])] for k in env.knobs],
            "xapps": [x.name for x in env.xapps],
            "stats": self.stat_keys,
            "dec_codes": list(DEC_CODES),
            "out_codes": list(OUT_CODES),
            "ack_codes": list(ACK_CODES),
            "rb_codes": list(RB_CODES),
            "wr_codes": list(WR_CODES),
            "kpm_fields": kpm_fields,
            "snapshot_s": self.snapshot_s,
            "slices": ["LL", "eMBB", "BE"],
            "cell_region": [int(x) for x in env.plant.lay.cell_site],
            "label_prefix": LABEL_PREFIX,
            "label_note": "lab_* arrays are privileged plant counters: training/eval targets, "
            "never features",
        }
