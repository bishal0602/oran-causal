"""Simple baseline arbiters of the system-level simulator; each uses only `obs`.

- no arbitration: `arbiter=None` in `E6Env.run` (accept all, last writer wins);
- `freeze`: reject every request, so the network keeps its initial 3GPP-default configuration;
- `priority`: per knob and second, keep only the request of the highest-priority xApp;
- `KnobLock`: SON-coordination lease; the first xApp to change a knob owns it for `lease_s`;
- `subset(keep)`: reject every request from xApps outside `keep` (all xApps stay deployed);
- `CellPriorityLock(order)`: a cell changed by a higher-priority xApp in the last `hold_s`
  rejects lower-priority requests on any knob of that cell;
- `region_subset(keep_by_region, site)` and its hindsight search `region_hindsight`;
- static configuration: `apply_static(env, spec)` before the run plus freeze; `tuned_static` is
  the hindsight coordinate descent over `tuned_static_grid()` (an upper reference, not
  deployable);
- `SMORestore`: an alarm-triggered SMO restore outside the near-RT RIC for scenario S2.
"""

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.sim.system import config as C
from oran_causal.sim.system.env import Arbiter, Decision, E6Env, Obs
from oran_causal.sim.system.ric import LIMITS, Knob, knob_set, quantise
from oran_causal.sim.system.sim import MistuneScenario

PRIORITY = ("SLICE", "MRO", "TS", "ES")
UNRANKED = 99


def freeze(obs: Obs) -> Decision:
    """Reject every request."""
    return {"decisions": ["reject"] * len(obs["requests"]), "writes": []}


def priority(obs: Obs, order: Sequence[str] = PRIORITY) -> Decision:
    """Per knob, accept only the request of the highest-ranked xApp in `order`."""
    rank = {x: i for i, x in enumerate(order)}
    best: dict[Knob, int] = {}
    for i, r in enumerate(obs["requests"]):
        k = r["knob"]
        held = obs["requests"][best[k]]["xapp"] if k in best else None
        if k not in best or rank.get(r["xapp"], UNRANKED) < rank.get(held, UNRANKED):
            best[k] = i
    keep = set(best.values())
    n = len(obs["requests"])
    return {"decisions": ["accept" if i in keep else "reject" for i in range(n)], "writes": []}


class KnobLock:
    """Knob lease: the first xApp to change a knob owns it for `lease_s` seconds."""

    def __init__(self, lease_s: float = 60.0) -> None:
        self.lease_s = lease_s
        self.owner: dict[Knob, tuple[str, float]] = {}

    def __call__(self, obs: Obs) -> Decision:
        """Accept the owner's requests and unowned knobs; reject the rest."""
        now, dec = obs["t"], []
        for r in obs["requests"]:
            k = r["knob"]
            own = self.owner.get(k)
            if own is None or own[1] < now or own[0] == r["xapp"]:
                self.owner[k] = (r["xapp"], now + self.lease_s)
                dec.append("accept")
            else:
                dec.append("reject")
        return {"decisions": dec, "writes": []}


def knob_cells(k: Knob) -> tuple[int, ...]:
    """Cells a knob acts on: `(s, n)` for CIO[s, n], `(c,)` for every per-cell knob."""
    return (k[1], k[2]) if k[0] == "cio" else (k[1],)


class CellPriorityLock:
    """Cell-priority lock, like NIST ns3-oran `SingleCommandPerNode`.

    A cell whose knob was changed by an xApp of priority p in the last `hold_s` seconds rejects
    the requests of every lower-priority xApp on any knob of that cell. Same-second requests are
    processed in priority order, so a higher-priority change blocks lower-priority requests on its
    cells at once; same-rank requests never block each other. An accepted request counts as a
    change only once the next `obs["config"]` shows the knob moved away from the request's `cur`.

    Args:
        order: xApp names, highest priority first; unknown xApps rank last.
        hold_s: hold time in seconds.
    """

    def __init__(self, order: Sequence[str], hold_s: float = 60.0) -> None:
        self.rank = {x: i for i, x in enumerate(order)}
        self.hold_s = float(hold_s)
        self.hold: dict[int, tuple[int, float]] = {}
        self.pending: list[tuple[Knob, float, int, float]] = []

    def _mark(self, cell: int, rk: int, t: float) -> None:
        h = self.hold.get(cell)
        if h is None or h[1] + self.hold_s <= t or rk <= h[0]:
            self.hold[cell] = (rk, t)

    def _blocked(self, k: Knob, rk: int, now: float, now_hold: Mapping[int, int]) -> bool:
        blocked = False
        for c in knob_cells(k):
            h = self.hold.get(c)
            if h is not None and now - h[1] < self.hold_s and h[0] < rk:
                blocked = True
            if c in now_hold and now_hold[c] < rk:
                blocked = True
        return blocked

    def __call__(self, obs: Obs) -> Decision:
        """Confirm last second's changes, then decide this second's requests by priority."""
        now, cfg = obs["t"], obs["config"]
        for k, cur, rk, t in self.pending:
            if abs(float(cfg.get(k, cur)) - cur) > 1e-9:
                for c in knob_cells(k):
                    self._mark(c, rk, t)
        self.pending = []
        reqs = obs["requests"]
        dec: list[str | None] = [None] * len(reqs)
        now_hold: dict[int, int] = {}
        for i in sorted(range(len(reqs)), key=lambda j: self.rank.get(reqs[j]["xapp"], UNRANKED)):
            r = reqs[i]
            rk = self.rank.get(r["xapp"], UNRANKED)
            if self._blocked(r["knob"], rk, now, now_hold):
                dec[i] = "reject"
                continue
            dec[i] = "accept"
            if abs(r["prop"] - r["cur"]) > 1e-9:
                self.pending.append((r["knob"], float(r["cur"]), rk, now))
                for c in knob_cells(r["knob"]):
                    now_hold[c] = min(now_hold.get(c, UNRANKED), rk)
        return {"decisions": dec, "writes": []}


BASELINES: dict[str, Callable[[], Any]] = {
    "noarb": lambda: None,
    "freeze": lambda: freeze,
    "priority": lambda: priority,
    "lock": KnobLock,
}


class RegionSubset:
    """Accept a request iff its xApp is kept in the region of the knob's own cell `k[1]`.

    Args:
        keep_by_region: region id to the xApps kept there.
        site: region id per cell.
    """

    def __init__(self, keep_by_region: Mapping[int, Iterable[str]], site: Sequence[int]) -> None:
        self.keep = {int(g): frozenset(v) for g, v in keep_by_region.items()}
        self.site = np.asarray(site)

    def __call__(self, obs: Obs) -> Decision:
        """Decide every request by its region's kept set."""
        keep = self.keep
        dec = [
            "accept" if r["xapp"] in keep[int(self.site[r["knob"][1]])] else "reject"
            for r in obs["requests"]
        ]
        return {"decisions": dec, "writes": []}


def region_subset(keep_by_region: Mapping[int, Iterable[str]], site: Sequence[int]) -> Arbiter:
    """A fixed per-region xApp subset held for the whole episode (all xApps stay deployed)."""
    return RegionSubset(keep_by_region, site)


def all_subsets(xapps: Sequence[str]) -> list[tuple[str, ...]]:
    """Every subset of `xapps` in `xapps` order, empty first, by size then mask order."""
    out = [
        tuple(x for x, m in zip(xapps, mask, strict=True) if m)
        for mask in itertools.product((0, 1), repeat=len(xapps))
    ]
    return sorted(out, key=len)


def region_hindsight(
    evaluate: Callable[[dict[int, tuple[str, ...]]], tuple[float, bool]],
    regions: Iterable[int],
    xapps: Sequence[str],
    all_regions: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Hindsight per-region static subset by one coordinate-descent pass (not deployable).

    Starts from accept-all, then visits `regions` in index order: at region g every other subset
    of `xapps` is evaluated with the other regions held, and the region moves to the best
    admissible candidate if it beats the incumbent. A candidate beats the incumbent iff it is
    admissible and the incumbent is not, or its value is strictly lower; ties keep the incumbent,
    and among candidates the earlier one in `all_subsets` order. Evaluations are cached.

    Args:
        evaluate: maps a keep-by-region assignment to `(value, admissible)`.
        regions: regions to optimise.
        xapps: deployed xApps.
        all_regions: every region of the plant (default `regions`); the others keep every xApp.

    Returns:
        `{"best", "best_val", "best_ok", "path": [{"keep", "val", "ok"}], "n_eval"}`.
    """
    regions = [int(g) for g in sorted(regions)]
    cache: dict[tuple, tuple[float, bool]] = {}
    path: list[dict[str, Any]] = []

    def ev(kb: Mapping[int, Sequence[str]]) -> tuple[float, bool]:
        key = tuple(sorted((g, tuple(v)) for g, v in kb.items()))
        if key not in cache:
            val, ok = evaluate({g: tuple(v) for g, v in kb.items()})
            cache[key] = (float(val), bool(ok))
            keep = {g: tuple(v) for g, v in kb.items()}
            path.append({"keep": keep, "val": cache[key][0], "ok": cache[key][1]})
        return cache[key]

    def better(a: tuple[float, bool], b: tuple[float, bool]) -> bool:
        return a[1] and (not b[1] or a[0] < b[0])

    full = tuple(xapps)
    best = {int(g): full for g in (all_regions if all_regions is not None else regions)}
    cur = ev(best)
    for g in regions:
        win, wv = None, cur
        for s in all_subsets(xapps):
            if s == best[g]:
                continue
            v = ev({**best, g: s})
            if better(v, wv):
                win, wv = s, v
        if win is not None:
            best, cur = {**best, g: win}, wv
    return {"best": best, "best_val": cur[0], "best_ok": cur[1], "path": path, "n_eval": len(cache)}


class Subset:
    """Reject every request from xApps outside `keep`.

    The plant only changes through accepted requests, so this reproduces running `keep` alone.
    `keep` = all deployed is accept-all; `keep` = () is freeze.
    """

    def __init__(self, keep: Iterable[str]) -> None:
        self.keep = frozenset(keep)

    def __call__(self, obs: Obs) -> Decision:
        """Accept the requests of kept xApps."""
        keep = self.keep
        dec = ["accept" if r["xapp"] in keep else "reject" for r in obs["requests"]]
        return {"decisions": dec, "writes": []}


def subset(keep: Iterable[str]) -> Arbiter:
    """Arbiter that keeps only the requests of the xApps in `keep`."""
    return Subset(keep)


STATIC_DEFAULT: dict[str, Any] = {
    "cio_to_pico": 0.0,
    "hys": 2.0,
    "ttt": 320,
    "ll_ratio": 0.0,
    "carriers": "all",
}


def _check(key: str, v: float) -> None:
    typ = {"cio_to_pico": "cio"}.get(key, key)
    lo, hi = LIMITS[typ][:2]
    if not lo <= v <= hi:
        raise ValueError(f"static {key}={v} outside actuator range [{lo}, {hi}]")
    if key == "ttt" and int(v) not in C.TTT_SET_MS:
        raise ValueError(f"static ttt={v} not in {C.TTT_SET_MS}")
    if abs(quantise((typ,), v) - v) > 1e-9:
        raise ValueError(f"static {key}={v} not on the actuator grid")


def _static_carriers(env: E6Env, v: Any) -> list[tuple[Knob, float]]:
    lay = env.plant.lay
    n = {"all": C.MACRO_NTRX, "one": 1}.get(v, v)
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= C.MACRO_NTRX:
        raise ValueError(f"static carriers={v!r} must be 'all', 'one' or 1..{C.MACRO_NTRX}")
    return [(("carrier", c), float(n)) for c in range(lay.n_cells) if lay.is_macro[c]]


def _static_cio(env: E6Env, v: float) -> list[tuple[Knob, float]]:
    lay = env.plant.lay
    out: list[tuple[Knob, float]] = []
    for m in range(lay.n_cells):
        if not lay.is_macro[m]:
            continue
        for q in lay.neighbours[m]:
            if not lay.is_macro[q]:
                out += [(("cio", m, q), v), (("cio", q, m), -v)]
    return out


def apply_static(env: E6Env, spec: Mapping[str, Any]) -> list[tuple[Knob, float]]:
    """Set a static network-wide configuration on `env.plant` before the first second.

    Keys (all optional; missing keys keep the initial value): `cio_to_pico` (dB, cell range
    expansion, CIO[macro -> pico] = +x and CIO[pico -> macro] = -x), `hys` (dB), `ttt` (ms, one
    of `TTT_SET_MS`), `ll_ratio`, `carriers` (`"all"`, `"one"` or an int). Values are checked
    against the actuator range and grid; step and dwell limits do not apply at t = 0.

    Returns:
        The (knob, value) pairs set.

    Raises:
        RuntimeError: after the first control second.
        KeyError: for an unknown key.
        ValueError: for a value off the actuator range or grid.
    """
    if env.sec != 0:
        raise RuntimeError("apply_static must be called before the first control second")
    bad = set(spec) - set(STATIC_DEFAULT)
    if bad:
        raise KeyError(f"unknown static keys {sorted(bad)}")
    out: list[tuple[Knob, float]] = []
    for key, v in spec.items():
        if key == "carriers":
            out += _static_carriers(env, v)
            continue
        fv = float(v)
        _check(key, fv)
        if key == "cio_to_pico":
            out += _static_cio(env, fv)
        else:
            out += [((key, c), fv) for c in range(env.plant.lay.n_cells)]
    for k, val in out:
        knob_set(env.plant, k, val, 0.0)
    return out


def static_policy(env: E6Env, spec: Mapping[str, Any]) -> Arbiter:
    """`apply_static` plus the freeze arbiter: the network runs the whole episode at `spec`."""
    apply_static(env, spec)
    return freeze


def tuned_static_grid() -> dict[str, tuple[Any, ...]]:
    """Coordinate-descent search space; the first value of each dimension is the default."""
    return {
        "cio_to_pico": (0.0, 3.0, 6.0),
        "hys": (2.0, 1.0, 4.0),
        "ttt": (320, 160, 640),
        "ll_ratio": (0.0, 0.1, 0.2),
        "carriers": ("all", "one"),
    }


def tuned_static(
    evaluate: Callable[[dict[str, Any]], float],
    grid: Mapping[str, Sequence[Any]] | None = None,
    max_passes: int = 2,
) -> dict[str, Any]:
    """Hindsight coordinate descent over static configurations, minimising `evaluate(spec)`.

    Chosen per seed on the scored tape itself, so it is an upper reference for what any fixed
    configuration can achieve, not a deployable method. Starts at the default, sweeps the
    dimensions in order, keeps a change only if it strictly lowers the objective, and stops after
    `max_passes` passes or a pass without improvement. Each spec is evaluated once.

    Returns:
        `{"best", "best_val", "path": [{"spec", "val", "improved"}], "n_eval"}`.
    """
    grid = grid or tuned_static_grid()
    cache: dict[tuple, float] = {}
    path: list[dict[str, Any]] = []

    def ev(spec: dict[str, Any]) -> float:
        key = tuple(sorted(spec.items()))
        if key not in cache:
            cache[key] = float(evaluate(dict(spec)))
            path.append({"spec": dict(spec), "val": cache[key]})
        return cache[key]

    best = {k: vals[0] for k, vals in grid.items()}
    best_val = ev(best)
    path[-1]["improved"] = True
    for _ in range(max_passes):
        improved = False
        for k, vals in grid.items():
            for v in vals:
                if v == best[k]:
                    continue
                cand = dict(best, **{k: v})
                n = len(path)
                val = ev(cand)
                if len(path) > n:
                    path[-1]["improved"] = val < best_val
                if val < best_val:
                    best, best_val, improved = cand, val, True
        if not improved:
            break
    return {"best": best, "best_val": best_val, "path": path, "n_eval": len(cache)}


SMO_TL_RATIO = 0.02
SMO_N_REPORTS = 2
SMO_MIN_TL, SMO_MIN_EVENTS = 2, 3


class SMORestore:
    """Alarm-triggered SMO restore: post-rollout verification with fallback, outside the RIC.

    Declared a priori, not tuned: the SMO uses MRO's own nominal too-late target (2 % of HO
    attempts plus too-late events) over the rollout cluster, sustained over MRO's 60 s window (two
    consecutive 30 s mobility reports), with MRO's minimum evidence (>= 2 too-late, >= 3 events).

    It wraps an arbiter `inner` (None = accept all) and passes its decisions through. After the
    S2 rollout it watches the delivered mobility reports whose window starts at or after the push;
    when the cluster too-late ratio exceeds `ratio` on `n_reports` consecutive reports, it restores
    the pre-rollout Hys and TTT on the whole cluster once, as an OAM write. No-op outside
    `scenario="mistune"`.
    """

    def __init__(
        self,
        env: E6Env,
        inner: Arbiter | None = None,
        ratio: float = SMO_TL_RATIO,
        n_reports: int = SMO_N_REPORTS,
    ) -> None:
        self.env, self.inner, self.ratio, self.n_reports = env, inner, ratio, n_reports
        self.streak = 0
        self.fired_t: float | None = None

    def alarm(self, rep: dict[str, Any], scn: MistuneScenario) -> bool:
        """Update the streak with one mobility report; True when the restore should fire."""
        cl = scn.cluster
        tl = float(rep["too_late"][cl].sum())
        den = float(rep["ho_att"][cl].sum()) + tl
        bad = tl >= SMO_MIN_TL and den >= SMO_MIN_EVENTS and tl / den > self.ratio
        self.streak = self.streak + 1 if bad else 0
        return self.streak >= self.n_reports

    def __call__(self, obs: Obs) -> Decision:
        """Pass the inner decisions through and fire the restore on a sustained alarm."""
        if self.inner is not None:
            dec = self.inner(obs)
        else:
            dec = {"decisions": ["accept"] * len(obs["requests"]), "writes": []}
        p = self.env.plant
        scn = p.scn
        if self.fired_t is None and isinstance(scn, MistuneScenario) and scn.push_t is not None:
            for rep in obs["new_reports"]:
                if rep["gran"] == "mob" and rep["t0"] >= scn.push_t - 1e-9 and self.alarm(rep, scn):
                    scn.oam_set(p, scn.pre_hys, scn.pre_ttt, obs["t"], "restore")
                    self.fired_t = obs["t"]
                    break
        return dec
