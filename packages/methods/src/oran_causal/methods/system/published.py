"""Published xApp-conflict methods on the system simulator, as arbiters that see only `obs`.

QACM (Wadud et al., IEEE TGCN 8(3) 2024, Alg. 1 / Eq. (3)) with the authors' rule-based CDC
(Wadud et al., INFOCOM'25 workshop, Sec. IV) and a KPI predictor (ANN vs PR, Sec. VI-B) fitted on
logs; the PACIFISTA INT distance (del Prever et al., IEEE TMC 2025, Table 1). The study plant uses
`MANIFEST_P`, `QOS_P` and `KPIS_P`. Learned pieces fit on `env.log` lists of logged DEV episodes.
"""

import bisect
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, cast

import numpy as np

from oran_causal.sim.system import config as C
from oran_causal.sim.system.ric import LIMITS, quantise

__all__ = [
    "KPIS_P",
    "KPI_GRAN",
    "MANIFEST",
    "MANIFEST_P",
    "QACM",
    "QOS",
    "QOS_P",
    "ROLES",
    "KPIPredictor",
    "KPIView",
    "clip_step",
    "int_distance",
    "knob_cells",
    "kpi_of",
    "one_step_values",
    "param_groups",
    "run_logged",
    "transitions",
    "violated",
]

Knob = tuple
Qos = tuple[float, int] | None

MANIFEST: dict[str, dict[str, Any]] = {
    "MRO": {"icp": ("cio", "ttt", "hys"), "kpi": "ho_fail", "span_s": 30.0},
    "TS": {"icp": ("cio",), "kpi": "embb_thp", "span_s": 10.0},
    "ES": {"icp": ("carrier",), "kpi": "energy", "span_s": 10.0},
    "SLICE": {"icp": ("ll_ratio",), "kpi": "ll_delay", "span_s": 1.0},
}
QOS: dict[str, Qos] = {
    "ll_delay": (C.LL_DELAY_TARGET_S, 1),
    "embb_thp": (C.EMBB_THP_TARGET_BPS, 0),
    "ho_fail": (0.02, 1),
    "energy": None,
}
KPI_GRAN = {
    "ll_delay": "fast",
    "util": "fast",
    "embb_thp": "thp",
    "ho_fail": "mob",
    "energy": "energy",
    "prot_below": "fast",
    "edge_sinr": "fast",
}

_P0 = C.E6PConfig()
MANIFEST_P: dict[str, dict[str, Any]] = {
    "ES": {"icp": ("carrier", "sleep"), "kpi": "energy", "span_s": 10.0},
    "PowerES": {"icp": ("ptx",), "kpi": "energy", "span_s": _P0.pes_cadence_s},
    "Coverage": {"icp": ("ptx",), "kpi": "edge_sinr", "span_s": _P0.cov_cadence_s},
    "SliceGuarantee": {"icp": ("prot_min",), "kpi": "prot_below", "span_s": _P0.sg_cadence_s},
}
QOS_P: dict[str, Qos] = dict(
    QOS, prot_below=(_P0.sg_viol_hi, 1), edge_sinr=(_P0.cov_sinr_low_db, 0)
)
KPIS_P = ("energy", "prot_below", "edge_sinr")
ROLES = ("self", "src", "dst")

_FIELD = {
    "ll_delay": "ll_delay_p95",
    "util": "prb_util",
    "embb_thp": "embb_thp_p5",
    "energy": "energy_j",
    "prot_below": "prot_below_frac",
    "edge_sinr": "edge_sinr_p",
}


def kpi_of(rep: Mapping[str, Any], name: str) -> np.ndarray:
    """Per-cell KPI from one KPM report of the matching granularity (NaN = no samples)."""
    if name in _FIELD:
        return np.asarray(rep[_FIELD[name]], float)
    if name == "ho_fail":
        tl = rep["too_late"].sum(1)
        den = rep["ho_att"].sum(1) + tl
        return np.where(den >= 3, tl / np.maximum(den, 1), np.nan)
    raise KeyError(name)


def violated(v: float, qos: Qos) -> bool:
    """True iff v breaks the QoS threshold (delta 1: above q, delta 0: below q)."""
    if qos is None or not np.isfinite(v):
        return False
    q, delta = qos
    return v > q if delta == 1 else v < q


class KPIView:
    """Latest delivered KPM report per granularity, as an arbiter sees it."""

    def __init__(self) -> None:
        self.rep: dict[str, Mapping[str, Any]] = {}

    def ingest(self, reps: Iterable[Mapping[str, Any]]) -> None:
        """Keeps the report with the latest window end per granularity."""
        for r in reps:
            old = self.rep.get(r["gran"])
            if old is None or r["t1"] >= old["t1"]:
                self.rep[r["gran"]] = r

    def get(self, name: str) -> np.ndarray | None:
        """Per-cell KPI from the latest report of its granularity, None before the first."""
        rep = self.rep.get(KPI_GRAN[name])
        return None if rep is None else kpi_of(rep, name)


def knob_cells(k: Knob) -> tuple[tuple[int, str], ...]:
    """Cells a knob acts on with their role: CIO[s, n] -> (s, src), (n, dst); else (c, self)."""
    return ((k[1], "src"), (k[2], "dst")) if k[0] == "cio" else ((k[1], "self"),)


def clip_step(k: Knob, cur: float, v: float) -> float:
    """Quantises, clamps to the range and clips to one actuator step (no dwell check)."""
    lo, hi, mstep, _ = LIMITS[k[0]]
    v = min(max(quantise(k, v), lo), hi)
    if k[0] == "ttt":
        i0, i1 = C.TTT_SET_MS.index(int(cur)), C.TTT_SET_MS.index(int(v))
        return float(C.TTT_SET_MS[i0 + int(np.clip(i1 - i0, -1, 1))])
    return float(cur + np.clip(v - cur, -mstep, mstep)) if mstep is not None else float(v)


def one_step_values(k: Knob, cur: float) -> list[float]:
    """Every actuator-grid value reachable from `cur` in one change (QACM Alg. 1's range N)."""
    typ = k[0]
    lo, hi, mstep, _ = LIMITS[typ]
    if typ == "ttt":
        i = C.TTT_SET_MS.index(int(cur))
        return [float(v) for v in C.TTT_SET_MS[max(i - 1, 0) : i + 2]]
    if typ == "sleep":
        return [0.0, 1.0]
    grid = {
        "cio": 1.0,
        "hys": 0.5,
        "ll_ratio": 0.05,
        "carrier": 1.0,
        "ptx": _P0.ptx_grid_db,
        "prot_min": _P0.prot_min_grid,
    }[typ]
    n = round(cast(float, mstep) / grid)
    vals = {quantise(k, cur + j * grid) for j in range(-n, n + 1)}
    return sorted(v for v in vals if lo - 1e-9 <= v <= hi + 1e-9)


def param_groups(manifest: Mapping[str, Mapping[str, Any]] = MANIFEST) -> dict[str, set[str]]:
    """Wadud'25 Alg. 1: PkG[kpi] = every ICP (knob type) of the xApps that own that KPI."""
    pkg: dict[str, set[str]] = {}
    for d in manifest.values():
        pkg.setdefault(d["kpi"], set()).update(d["icp"])
    return pkg


def run_logged(
    cfg: C.E6Config, arbiter: Any = None, **kw: Any
) -> tuple[dict[str, Any], list[dict]]:
    """Runs one logged episode; returns (score, env.log)."""
    from oran_causal.sim.system.env import E6Env

    env = E6Env(cfg, log=True, **kw)
    return env.run(arbiter), env.log


def _replay(log: list[dict]) -> Iterator[tuple[float, dict, dict, KPIView]]:
    view = KPIView()
    prev = None
    for e in log:
        view.ingest(e["reports"])
        if prev is not None:
            yield e["t"], prev, e["config"], view
        prev = e["config"]


def _report_index(log: list[dict]) -> dict[str, tuple[list[float], dict[float, Mapping]]]:
    by_t0: dict[str, dict[float, Mapping]] = {}
    for e in log:
        for r in e["reports"]:
            by_t0.setdefault(r["gran"], {})[r["t0"]] = r
    return {g: (sorted(d), d) for g, d in by_t0.items()}


def _change_rows(
    before: dict, after: dict, rng: np.random.Generator, null_ratio: float
) -> list[tuple[Knob, float, float]]:
    changed = [k for k in after if abs(after[k] - before[k]) > 1e-9]
    rows = [(k, after[k] - before[k], before[k]) for k in changed]
    by_typ: dict[str, int] = {}
    for k in changed:
        by_typ[k[0]] = by_typ.get(k[0], 0) + 1
    for typ, n in by_typ.items():
        pool = [k for k in after if k[0] == typ and k not in changed]
        m = min(len(pool), round(n * null_ratio))
        rows += [(pool[j], 0.0, before[pool[j]]) for j in rng.choice(len(pool), m, replace=False)]
    return rows


def _target(idx: dict, name: str, t: float) -> np.ndarray | None:
    g = KPI_GRAN[name]
    if g not in idx:
        return None
    keys, d = idx[g]
    j = bisect.bisect_left(keys, t - 1e-9)
    return None if j == len(keys) else kpi_of(d[keys[j]], name)


def _add_samples(out: dict, rows: list, name: str, now_tgt: tuple, util: np.ndarray) -> None:
    now, tgt = now_tgt
    for k, dv, v0 in rows:
        for c, role in knob_cells(k):
            x = [role == r_ for r_ in ROLES] + [dv, v0, now[c], util[c]]
            if np.all(np.isfinite(x)) and np.isfinite(tgt[c]):
                X, y = out.setdefault((k[0], name), ([], []))
                X.append(x)
                y.append(tgt[c])


def transitions(
    episodes: Iterable[list[dict]], kpis: Sequence[str], null_ratio: float = 1.0, seed: int = 0
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(knob type, kpi) -> (X, y) regression samples from logged episodes.

    One sample per (applied knob change at second t, acted-on cell, kpi): features [role one-hot,
    delta, value before, kpi now, prb_util now] as the arbiter saw them at t; target = the kpi of
    the first report of its granularity whose window starts at or after t. `null_ratio` times as
    many unchanged knobs of the same type (delta 0) are sampled per second.
    """
    rng = np.random.default_rng(seed)
    out: dict = {}
    for log in episodes:
        idx = _report_index(log)
        for t, before, after, view in _replay(log):
            util = view.get("util")
            if util is None:
                continue
            rows = _change_rows(before, after, rng, null_ratio)
            for name in kpis:
                now = view.get(name)
                tgt = None if now is None else _target(idx, name, t)
                if now is not None and tgt is not None:
                    _add_samples(out, rows, name, (now, tgt), util)
    return {key: (np.asarray(X, float), np.asarray(y, float)) for key, (X, y) in out.items()}


class _ANN:
    """QACM Sec. VI-B ANN: 4 x 128 tanh with dropout 0.2, linear output, Adam, MSE, 10 epochs."""

    def __init__(self, d: int, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        layers, w = [], d
        for _ in range(4):
            layers += [torch.nn.Linear(w, 128), torch.nn.Tanh(), torch.nn.Dropout(0.2)]
            w = 128
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(w, 1)).double()
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch: int = 10) -> _ANN:
        import torch

        g = torch.Generator().manual_seed(self.seed)
        opt = torch.optim.Adam(self.net.parameters())
        Xt, yt = torch.as_tensor(X), torch.as_tensor(y)[:, None]
        self.net.train()
        for _ in range(epochs):
            for b in torch.randperm(len(Xt), generator=g).split(batch):
                opt.zero_grad()
                torch.nn.functional.mse_loss(self.net(Xt[b]), yt[b]).backward()
                opt.step()
        self.net.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(X, float))).numpy()[:, 0]


class _PR:
    """QACM Sec. VI-B polynomial regression (degree 2, least squares)."""

    w: np.ndarray

    @staticmethod
    def _phi(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        cols = [np.ones(len(X))] + [X[:, i] for i in range(X.shape[1])]
        cols += [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
        return np.stack(cols, 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _PR:
        P = self._phi(X)
        self.w = np.linalg.solve(P.T @ P + 1e-6 * np.eye(P.shape[1]), P.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._phi(X) @ self.w


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return 1.0 - float(np.sum((y - p) ** 2)) / max(float(np.sum((y - y.mean()) ** 2)), 1e-12)


class KPIPredictor:
    """Per (knob type, kpi) regression of the next-window cell KPI on the pending change.

    KPIs are z-scored (QACM Sec. VI-A). Both the ANN and PR are fitted and the higher held-out
    R^2 is kept; fewer than `min_samples` samples gives persistence (predict the current value).
    """

    def __init__(
        self,
        kpis: Sequence[str] = ("ll_delay", "embb_thp", "ho_fail", "energy"),
        null_ratio: float = 1.0,
        min_samples: int = 30,
        holdout: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.kpis, self.null_ratio, self.min_samples = kpis, null_ratio, min_samples
        self.holdout, self.seed = holdout, seed
        self.models: dict[tuple[str, str], Any] = {}
        self.report: dict[tuple[str, str], dict[str, Any]] = {}
        self.zs: dict[str, tuple[float, float]] = {}
        self.fscale: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, episodes: Iterable[Iterable[dict]]) -> KPIPredictor:
        """Fits the z-score statistics and one model per (knob type, kpi) on logged episodes."""
        eps = [list(e) for e in episodes]
        data = transitions(eps, self.kpis, self.null_ratio, self.seed)
        for name in self.kpis:
            vals = [
                kpi_of(r, name)
                for log in eps
                for e in log
                for r in e["reports"]
                if r["gran"] == KPI_GRAN[name]
            ]
            v = np.concatenate(vals) if vals else np.array([])
            v = v[np.isfinite(v)]
            self.zs[name] = (
                (float(v.mean()), float(max(v.std(), 1e-9))) if len(v) > 1 else (0.0, 1.0)
            )
        for (typ, name), (X, y) in data.items():
            if name in self.zs:
                self._fit_one(typ, name, X, y)
        return self

    def _fit_one(self, typ: str, name: str, X: np.ndarray, y: np.ndarray) -> None:
        Xz, yz = self._feat(name, X), self.z(name, y)
        mu, sd = Xz[:, 3:].mean(0), Xz[:, 3:].std(0) + 1e-9
        self.fscale[(typ, name)] = (mu, sd)
        Xs = np.concatenate([Xz[:, :3], (Xz[:, 3:] - mu) / sd], 1)
        if len(yz) < self.min_samples:
            self.report[(typ, name)] = {"n": len(yz), "model": "persistence"}
            return
        perm = np.random.default_rng(self.seed).permutation(len(yz))
        n_te = max(int(len(yz) * self.holdout), 1)
        te, tr = perm[:n_te], perm[n_te:]
        cands = {
            "ANN": _ANN(Xs.shape[1], self.seed).fit(Xs[tr], yz[tr]),
            "PR": _PR().fit(Xs[tr], yz[tr]),
        }
        r2 = {m: _r2(yz[te], f.predict(Xs[te])) for m, f in cands.items()}
        best = max(r2, key=lambda m: r2[m])
        self.models[(typ, name)] = cands[best]
        self.report[(typ, name)] = {"n": len(yz), "model": best, "r2": r2}

    def z(self, name: str, v: Any) -> np.ndarray:
        """z-scores a raw KPI value with the fitted statistics."""
        mu, sd = self.zs[name]
        return (np.asarray(v, float) - mu) / sd

    def _feat(self, name: str, X: np.ndarray) -> np.ndarray:
        X = np.array(X, float)
        X[:, 5] = self.z(name, X[:, 5])
        return X

    def predict_z(self, typ: str, name: str, X: np.ndarray) -> np.ndarray:
        """z-scored next-window KPI for raw rows [role one-hot, delta, value, kpi now, util now]."""
        Xz = self._feat(name, X)
        m = self.models.get((typ, name))
        if m is None:
            return Xz[:, 5]
        mu, sd = self.fscale[(typ, name)]
        return m.predict(np.concatenate([Xz[:, :3], (Xz[:, 3:] - mu) / sd], 1))


class QACM:
    """QACM as the CMC behind the authors' rule-based CDC, one conflicting knob at a time.

    CDC (Wadud'25 Sec. IV, pre-action): a pending request of xApp a on knob k conflicts with every
    other deployed xApp whose KPI violates its QoS threshold on a cell k acts on, and with every
    other xApp requesting k in the same second; `anticipate` also involves non-violated xApps.
    Fewer than two involved xApps accept (Eq. 3j). CMC (Alg. 1): candidates on the one-step grid
    within [min, max] of the current value and the step-clipped proposals; fCost = sum_i w_i d_i
    zeta - (sum_i s_i)^2, the first strict minimum wins. The optimum maps to accept (a proposal),
    reject (the current value) or ("modify", value).
    """

    def __init__(
        self,
        predictor: Any,
        *,
        manifest: Mapping[str, Mapping[str, Any]] = MANIFEST,
        qos: Mapping[str, Qos] = QOS,
        weights: Mapping[str, float] | None = None,
        zeta: float = 1e3,
        anticipate: bool = False,
        order: str = "prop_first",
    ) -> None:
        self.pred, self.manifest, self.qos, self.weights = predictor, manifest, qos, weights
        self.zeta, self.anticipate, self.order = zeta, anticipate, order
        self.view = KPIView()
        self.stats = {"conflicts": 0, "accept": 0, "reject": 0, "modify": 0}

    def involved(self, k: Knob, requesters: Iterable[str], deployed: Iterable[str]) -> set[str]:
        """CDC: the set X' of xApps in conflict over knob k (requesters included)."""
        X = set(requesters)
        for b in deployed:
            d = self.manifest.get(b)
            if b in X or d is None:
                continue
            v = self.view.get(d["kpi"])
            if v is None:
                continue
            q = self.qos.get(d["kpi"])
            if self.anticipate or any(violated(v[c], q) for c, _ in knob_cells(k)):
                X.add(b)
        return X

    def utilities(
        self, k: Knob, cur: float, cands: Sequence[float], xapp: str
    ) -> np.ndarray | None:
        """U_i(p_l) per candidate; None if the xApp has no KPI signal on the acted-on cells."""
        name = self.manifest[xapp]["kpi"]
        now, util = self.view.get(name), self.view.get("util")
        if now is None or util is None:
            return None
        rows, ncell = [], 0
        for c, role in knob_cells(k):
            if not (np.isfinite(now[c]) and np.isfinite(util[c])):
                continue
            ncell += 1
            rows += [[role == r_ for r_ in ROLES] + [p - cur, cur, now[c], util[c]] for p in cands]
        if not ncell:
            return None
        u = self.pred.predict_z(k[0], name, np.asarray(rows, float)).reshape(ncell, len(cands))
        return u.mean(0)

    def _grid(self, k: Knob, cur: float, props: list[float]) -> list[float]:
        lo, hi = min([cur, *props]), max([cur, *props])
        grid = [v for v in one_step_values(k, cur) if lo - 1e-9 <= v <= hi + 1e-9]
        if self.order == "prop_first":
            head = list(dict.fromkeys([*props, cur]))
            grid = head + [v for v in grid if all(abs(v - h) > 1e-9 for h in head)]
        return grid

    def alg1(self, k: Knob, cur: float, props: list[float], X: set[str]) -> float:
        """QACM Alg. 1 over the candidate grid; returns p_opt."""
        grid = self._grid(k, cur, props)
        w = {x: (self.weights or {}).get(x, 1.0) for x in X}
        tot = sum(w.values()) or 1.0
        cost = np.zeros(len(grid))
        sat = np.zeros(len(grid))
        for x in sorted(X):
            name = self.manifest[x]["kpi"]
            q = self.qos.get(name)
            U = self.utilities(k, cur, grid, x) if q is not None else None
            if q is None or U is None:
                sat += 1
                continue
            qz = float(self.pred.z(name, q[0]))
            d = np.maximum(qz - U, 0.0) if q[1] == 0 else np.maximum(U - qz, 0.0)
            cost += w[x] / tot * d * self.zeta
            sat += d <= 0
        f = cost - sat**2
        best, fmin = grid[0], np.inf
        for p, fc in zip(grid, f, strict=True):
            if fmin > fc:
                best, fmin = p, fc
        return best

    def __call__(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Decides the second's requests."""
        self.view.ingest(obs["new_reports"])
        reqs = obs["requests"]
        dec: list[Any] = ["accept"] * len(reqs)
        deployed = [x for x in obs["static"]["xapps"] if x in self.manifest]
        by_knob: dict[Knob, list[int]] = {}
        for i, r in enumerate(reqs):
            by_knob.setdefault(r["knob"], []).append(i)
        for k, idx in by_knob.items():
            self._resolve(obs, k, idx, deployed, dec)
        return {"decisions": dec, "writes": []}

    def _resolve(
        self, obs: Mapping[str, Any], k: Knob, idx: list[int], deployed: list[str], dec: list
    ) -> None:
        reqs = obs["requests"]
        X = self.involved(k, [reqs[i]["xapp"] for i in idx], deployed)
        if len(X) < 2:
            self.stats["accept"] += len(idx)
            return
        self.stats["conflicts"] += 1
        cur = float(obs["config"].get(k, reqs[idx[0]]["cur"]))
        props = [clip_step(k, cur, reqs[i]["prop"]) for i in idx]
        p = self.alg1(k, cur, props, X)
        hit = [i for i, pc in zip(idx, props, strict=True) if abs(pc - p) < 1e-9]
        for i in idx:
            dec[i] = "accept" if i in hit else "reject"
        if abs(p - cur) > 1e-9 and not hit:
            dec[idx[-1]] = ("modify", p)
        for i in idx:
            self.stats[dec[i] if isinstance(dec[i], str) else "modify"] += 1


def int_distance(a: Any, b: Any) -> float:
    """PACIFISTA INT = sqrt((1/L) integral |F1(x) - F2(x)| dx), L = the range of both samples."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    xs = np.union1d(a, b)
    if len(xs) < 2:
        return 0.0
    f1 = np.searchsorted(a, xs, side="right") / len(a)
    f2 = np.searchsorted(b, xs, side="right") / len(b)
    return float(np.sqrt(np.sum(np.abs(f1 - f2)[:-1] * np.diff(xs)) / (xs[-1] - xs[0])))
