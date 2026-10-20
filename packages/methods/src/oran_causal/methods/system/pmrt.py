"""PMRT on the system-level decision units: data plumbing, conditional redraws and declarations.

`load_pmrt_pool` and `pmrt_data` turn unit-table caches into the per-unit features of
`oran_causal.core.pmrt.units.PmrtData`: per cell and KPI, post bins [0, 30) [30, 60) [60, 90)
[90, 150) s after t0 and pre bins of the same lengths before it (NaN where a window does not fit
the episode), the observation-only context, the past design-centred modes of every family at
the unit's cell and its exposure neighbours, and t0 / T.

A family's test redraws its modes from the logged tables with every other family fixed, B draws
from `default_rng([seed, CRT_TAG, PMRT_STREAM, family index, split])`, chunked and
chunk-invariant; p = (1 + #{T_b >= T_obs - tol}) / (B + 1). `run_pmrt` covers every arm and arm
combination; `run_integrated` is the frozen Study 1 statistic (arms `plain_c`, `loadsp_c` and
their max, with one-sided p-values in the effect direction), declared by `declare_all`.
"""

import dataclasses
import json
import os
import pickle
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.core.data import UnitTable
from oran_causal.core.declare import DeclareConfig, HypStats, Prior, declare
from oran_causal.core.design import v_design, v_var
from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.core.pmrt import units as pu
from oran_causal.core.pmrt.units import PmrtConfig, PmrtData
from oran_causal.core.ridge import ResidualConfig
from oran_causal.methods.system.mscr_crt import (
    CRT_TAG,
    LEVEL_V2_ARR,
    SPLIT_POOLED,
    PiAssignment,
)
from oran_causal.sim.system.decision import records
from oran_causal.sim.system.decision.vocab import FAMILIES, HYPOTHESES, structure

PMRT_STREAM = 3
BENCH_TAG = 6690
HS4 = records.HS
HIST_WIN = 150
COL_ARMS = ("plain", "plain_c", "loadsp_c")
STATS = ("plain_c", "loadsp_c", "max")
MAX_OF = ("plain_c", "loadsp_c")
LAYERS = ("by", "dagger1s", "wby1s")
COMBOS = [(s, lay) for s in STATS for lay in LAYERS]
N_PRIOR = 480
Q, FLOOR, THR = 0.05, 0.2, 3.0
F32 = np.float32


@dataclasses.dataclass(frozen=True)
class EProcConfig:
    """Settings of the predictable-residual e-process on decision units.

    Attributes:
        q: BY level.
        lam: Betting rule, `"grapa"` or `"mix"`.
        n_grid: Mixture grid size: |lambda| = lam_max 2^-j, j < n_grid, both signs.
        lam_frac: lam_max = lam_frac / max |v|.
        c0: Pseudo-count of the GRAPA denominator.
        clip: s = clip(r / (clip sigma), -1, 1).
        burn: First units of a family with s = 0.
        prior_n: Pseudo-units shrinking a stratum's running means to the same-sign means.
        ridge: Ridge on the pooled within-stratum pre slope, relative to its scale.
        use_pre: Adjust for the pre-window value.
        min_units: Minimum units of a family.
        min_accept: Minimum accepted units.
        min_reject: Minimum rejected units.
        min_episodes: Minimum episodes.
    """

    q: float = 0.05
    lam: str = "grapa"
    n_grid: int = 8
    lam_frac: float = 0.9
    c0: float = 1.0
    clip: float = 2.0
    burn: int = 10
    prior_n: float = 20.0
    ridge: float = 1e-6
    use_pre: bool = True
    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3

    def residual(self) -> ResidualConfig:
        """The running-residual part of the settings."""
        return ResidualConfig(self.burn, self.prior_n, self.ridge, self.use_pre)


@dataclasses.dataclass(frozen=True)
class PmrtPool:
    """A unit-table pool with the cumulative window sums of every `HS4` length.

    Attributes:
        pool: The pool of `records.load_pool`.
        post_cum: Post-window sums per unit row of `pool.u`, shape `[N, K, C, 4]` (float32).
        pre_cum: Pre-window sums, same shape.
    """

    pool: records.Pool
    post_cum: np.ndarray
    pre_cum: np.ndarray


def _path_list(paths: records.Paths) -> list[str]:
    if isinstance(paths, str):
        return [p for p in paths.split(",") if p]
    return [str(p) for p in paths]


def _file_rows(path: str, seen: set, stages: set[str] | None) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as z:
        post = np.stack([z[f"post_H{h}"] for h in HS4], -1).astype(F32)
        pre = np.stack([z[f"pre_H{h}"] for h in HS4], -1).astype(F32)
        keys = list(
            zip(
                z["ep_seed"].tolist(),
                z["ep_sub"].tolist(),
                z["ep_stage"].tolist(),
                z["ep_smoke"].tolist(),
                strict=True,
            )
        )
        ep = z["ep"]
    keep = np.array([(k not in seen) and (stages is None or k[2] in stages) for k in keys], bool)
    seen.update(keys)
    return post, pre, keep[ep]


def load_pmrt_pool(
    paths: records.Paths, stages: Sequence[str] | None = None, H: int = records.H_UNIT
) -> PmrtPool:
    """Loads cache files for H = H_pre = `H` plus the window sums of every `HS4` length.

    Args:
        paths: Cache files (a list or a comma-separated string).
        stages: Record stages to keep; None keeps all.
        H: Unit window length.

    Returns:
        The pool; rows and episode dedupe (first file wins) as `records.load_pool`.

    Raises:
        AssertionError: The window sums do not align with the pool rows.
    """
    plist = _path_list(paths)
    pool = records.load_pool(plist, H=H, H_pre=H, stages=stages)
    st = set(stages) if stages is not None else None
    seen: set = set()
    parts = [_file_rows(p, seen, st) for p in plist]
    keep = np.concatenate([k for _, _, k in parts])
    post = np.concatenate([p for p, _, _ in parts])[keep]
    pre = np.concatenate([q for _, q, _ in parts])[keep]
    if len(post) != len(pool.u["gep"]):
        raise AssertionError("row alignment with load_pool failed")
    return PmrtPool(pool, post, pre)


def pool_rows(pool: records.Pool, episodes: Sequence[int] | None = None) -> tuple[np.ndarray, ...]:
    """Rows of `pool.u` in `Pool.unit_data` order.

    Returns:
        (rows kept by the window rule, every row of the episodes).
    """
    e = pool.eps
    ids = np.arange(pool.n_eps) if episodes is None else np.asarray(episodes, int)
    ids = np.array(sorted(ids.tolist(), key=lambda i: (int(e["seed"][i]), str(e["sub"][i]))), int)
    pos = np.full(pool.n_eps, -1)
    pos[ids] = np.arange(len(ids))
    u = pool.u
    rows_all = np.nonzero(pos[u["gep"]] >= 0)[0]
    rows_all = rows_all[np.argsort(pos[u["gep"][rows_all]], kind="stable")]
    t0 = u["t0"][rows_all].astype(int)
    T = e["T"][u["gep"][rows_all]]
    keep = (t0 - pool.H_pre >= 0) & (t0 + pool.H <= T)
    return rows_all[keep], rows_all


def _bins(cum: np.ndarray, ok: np.ndarray) -> np.ndarray:
    b = np.empty_like(cum)
    b[..., 0] = cum[..., 0]
    b[..., 1:] = cum[..., 1:] - cum[..., :-1]
    b[~ok[:, None, None, :].repeat(cum.shape[1], 1).repeat(cum.shape[2], 2)] = np.nan
    return b


@dataclasses.dataclass(frozen=True)
class _HistRows:
    va: np.ndarray
    t: np.ndarray
    c: np.ndarray
    f: np.ndarray


def _hist_episode(u: Mapping[str, np.ndarray], rk: np.ndarray, a: _HistRows, nf: int) -> np.ndarray:
    tk = u["t0"][rk].astype(int)
    ck = u["c"][rk].astype(int)
    dt = tk[:, None] - a.t[None, :]
    past = (dt > 0) & (dt <= HIST_WIN)
    same = ck[:, None] == a.c[None, :]
    nbr = u["exp"][rk][:, a.c] & ~same
    out = np.zeros((len(rk), 2 * nf))
    for fi in range(nf):
        wf = past & (a.f == fi)[None, :]
        out[:, fi] = (wf & same) @ a.va
        out[:, nf + fi] = (wf & nbr) @ a.va
    return out


def _hist(pool: records.Pool, rows: np.ndarray, rows_all: np.ndarray) -> np.ndarray:
    u = pool.u
    nf = len(FAMILIES)
    out = np.zeros((len(rows), 2 * nf))
    va = v_design(
        u["mode"][rows_all].astype(int), u["probs"][rows_all], u["sgn"][rows_all], LEVEL_V2_ARR
    )
    gep_a = u["gep"][rows_all]
    t_a, c_a = u["t0"][rows_all].astype(int), u["c"][rows_all].astype(int)
    f_a = u["family"][rows_all].astype(int)
    pos_k: dict[int, list[int]] = {}
    for i, e in enumerate(u["gep"][rows].tolist()):
        pos_k.setdefault(e, []).append(i)
    for e, ks in pos_k.items():
        ia = np.nonzero(gep_a == e)[0]
        k = np.array(ks)
        out[k] = _hist_episode(u, rows[k], _HistRows(va[ia], t_a[ia], c_a[ia], f_a[ia]), nf)
    return out


def pmrt_data(ppool: PmrtPool, episodes: Sequence[int] | None = None) -> PmrtData:
    """Unit features of the given global episode ids (default: all).

    The unit table equals `ppool.pool.unit_data(episodes)`.

    Raises:
        AssertionError: The feature rows do not align with the unit table.
    """
    pool = ppool.pool
    ud = pool.unit_data(episodes)
    rows, rows_all = pool_rows(pool, episodes)
    u = pool.u
    if not (
        np.array_equal(u["t0"][rows], ud.t0) and np.array_equal(u["c"][rows].astype(int), ud.c)
    ):
        raise AssertionError("PmrtData rows do not align with the UnitTable")
    t0 = u["t0"][rows].astype(int)
    T = pool.eps["T"][u["gep"][rows]]
    ok_post = np.stack([t0 + h <= T for h in HS4], 1)
    ok_pre = np.stack([t0 - h >= 0 for h in HS4], 1)
    return PmrtData(
        ud=ud,
        post=_bins(ppool.post_cum[rows], ok_post),
        pre=_bins(ppool.pre_cum[rows], ok_pre),
        ctx=np.where(u["ctx_num"][rows], u["ctx"][rows], np.nan),
        ctx_keys=list(pool.ctx_keys),
        hist=_hist(pool, rows, rows_all),
        tfrac=t0 / T.astype(float),
    )


def _ctx_arrays(parts: Sequence[Mapping[str, Any]], n: int) -> dict[str, np.ndarray]:
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
    return {"ctx": ctx, "ctx_num": num, "ctx_keys": np.array(keys, dtype="<U64")}


def cache_records(
    recs: Sequence[records.Record], out_path: str | os.PathLike[str], src: str = "records"
) -> str:
    """Writes in-memory episode records as one uncompressed unit-table cache.

    Episodes are kept in the order given, without dedupe.

    Returns:
        `out_path`.
    """
    parts = [records.episode_arrays(r) for r in recs]
    arrs: dict[str, Any] = {
        name: np.concatenate([p[name] for p in parts])
        for name in parts[0]
        if not name.startswith("_")
    }
    arrs["ep"] = np.concatenate([np.full(len(p["c"]), e, np.int32) for e, p in enumerate(parts)])
    arrs.update(_ctx_arrays(parts, len(arrs["ep"])))
    for f in records.EP_FIELDS:
        arrs[f"ep_{f}"] = np.array([p["_ep"][f] for p in parts])
    arrs["src"] = np.array(src)
    arrs["schema"] = np.array(records.CACHE_SCHEMA)
    arrs["Hs"] = np.array(records.HS, np.int32)
    arrs["n_cells"] = np.array(parts[0]["_ep"]["n_cells"], np.int32)
    np.savez(out_path, **arrs)
    return str(out_path)


def pmrt_data_from_records(
    recs: Sequence[records.Record], tmpdir: str | os.PathLike[str] | None = None
) -> PmrtData:
    """Unit features of in-memory records, through one temporary cache file."""
    d = Path(tmpdir) if tmpdir is not None else Path(tempfile.mkdtemp(prefix="pmrt_"))
    f = d / f"rec_{os.getpid()}_{time.time_ns()}.npz"
    cache_records(recs, f)
    try:
        return pmrt_data(load_pmrt_pool([str(f)]))
    finally:
        f.unlink(missing_ok=True)


def _targets(ud: UnitTable) -> list[tuple[str, str]]:
    return [(r, k) for r in ud.relations for k in ud.kpis]


def _support(ud: UnitTable, family: str, cfg: PmrtConfig) -> tuple[dict[str, int], bool]:
    rows = ud.rows_of(family)
    lv = ud.levels[ud.mode[rows]]
    n_acc, n_rej = int((lv == 1.0).sum()), int((lv == 0.0).sum())
    n_ep = len(np.unique(ud.episode[rows]))
    counts = {"n": len(rows), "n_accept": n_acc, "n_reject": n_rej, "n_episodes": n_ep}
    ok = (
        len(rows) >= cfg.min_units
        and n_acc >= cfg.min_accept
        and n_rej >= cfg.min_reject
        and n_ep >= cfg.min_episodes
    )
    return counts, ok


def stream(cfg: PmrtConfig, family_idx: int, split: int) -> np.random.Generator:
    """The redraw RNG of one family and split."""
    return np.random.default_rng([int(cfg.seed), CRT_TAG, PMRT_STREAM, int(family_idx), int(split)])


def redraws(
    ud: UnitTable, rows: np.ndarray, cfg: PmrtConfig, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Design-centred scores of `cfg.B` mode redraws of `rows`, in chunks.

    Each unit's mode is redrawn from its logged table; a chunk is `V = sgn (LEVEL_V2[M] - mu)` of
    shape `[b, len(rows)]`, with at most `cfg.chunk` draws and `cfg.max_chunk_bytes` of modes.
    """
    pa = PiAssignment(ud)
    mu = ud.probs[rows] @ LEVEL_V2_ARR
    sgv = ud.sgn[rows]
    ch = int(max(1, min(cfg.chunk, cfg.max_chunk_bytes // (8 * max(len(rows), 1)))))
    done = 0
    while done < cfg.B:
        b = min(ch, cfg.B - done)
        v = LEVEL_V2_ARR[pa.mode_draws(rng, rows, b)]
        v -= mu[None, :]
        v *= sgv[None, :]
        yield v
        done += b


@dataclasses.dataclass(frozen=True)
class _ArmObs:
    w: np.ndarray
    sd: np.ndarray
    s: np.ndarray
    t: np.ndarray
    combos: dict[str, list[int]]
    nt: int
    na: int

    @property
    def z(self) -> np.ndarray:
        return self.s / self.sd

    def combo_t(self, ix: Sequence[int]) -> np.ndarray:
        return np.array([self.t[[t * self.na + i for i in ix]].max() for t in range(self.nt)])


def _observe(ud: UnitTable, rows: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows], LEVEL_V2_ARR)
    var = v_var(ud.probs[rows], LEVEL_V2_ARR)
    sd = np.sqrt(var @ (w * w))
    return v @ w, np.where(sd > 0, sd, 1.0)


def _arm_counts(
    obs: _ArmObs, draws: Iterator[np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    tol = 1e-9 * (1.0 + obs.t)
    tc_obs = {c: obs.combo_t(ix) for c, ix in obs.combos.items()}
    cnt = np.zeros(obs.w.shape[1], np.int64)
    cntc = {c: np.zeros(obs.nt, np.int64) for c in obs.combos}
    for v in draws:
        t = np.abs(v @ obs.w) / obs.sd[None, :]
        del v
        cnt += np.count_nonzero(t >= (obs.t - tol)[None, :], axis=0)
        t3 = t.reshape(len(t), obs.nt, obs.na)
        for c, ix in obs.combos.items():
            tc = t3[:, :, ix].max(2)
            cntc[c] += np.count_nonzero(tc >= (tc_obs[c] - 1e-9 * (1 + tc_obs[c]))[None, :], axis=0)
    return cnt, cntc


def _arm_result(
    base: dict[str, Any], t: int, meta: Mapping[str, Any], obs: _ArmObs, counts: tuple
) -> dict[str, Any]:
    cnt, cntc, b1 = counts
    arms = pu.ARMS
    ai = {a: i for i, a in enumerate(arms)}
    z = obs.z
    res = dict(base, status="tested", best=meta["best"], z_plain_train=meta["z_plain_train"])
    res["arms"] = {}
    plain_sign = float(np.sign(obs.s[t * obs.na + ai["plain"]]))
    for a, i in ai.items():
        j = t * obs.na + i
        sdir = meta["sdir"].get(a, 0.0)
        sgn_ = float(np.sign(obs.s[j])) * sdir if sdir else plain_sign
        res["arms"][a] = {"p": float((1 + cnt[j]) / b1), "z": float(z[j]), "sign": int(sgn_)}
    for c, ix in obs.combos.items():
        jbest = ix[int(np.argmax(obs.t[[t * obs.na + i for i in ix]]))]
        res["arms"][c] = {
            "p": float((1 + cntc[c][t]) / b1),
            "z": float(obs.t[t * obs.na + jbest]),
            "sign": res["arms"][arms[jbest]]["sign"],
            "via": arms[jbest],
        }
    return res


def family_tests_pmrt(
    pd: PmrtData,
    family: str,
    params: Mapping[str, Any],
    cfg: PmrtConfig | None = None,
    split: int = SPLIT_POOLED,
) -> list[dict[str, Any]]:
    """CRT of one family for every (target, arm) and arm combination on one set of draws.

    Returns:
        One dict per target (relation, KPI): support counts, `status` and, when tested, per arm
        and combination `p`, `z`, `sign` (and the attaining arm `via` of a combination).
    """
    cfg = cfg or PmrtConfig()
    ud = pd.ud
    counts, ok = _support(ud, family, cfg)
    base = {"version": pu.PMRT_VERSION, "family": family, **counts, "B": cfg.B}
    if not ok:
        return [
            dict(base, relation=r, kpi=k, status="undetermined", reason="support")
            for r, k in _targets(ud)
        ]
    rows, w, targets, meta = pu.family_columns(pd, family, params, cfg)
    s_obs, sd = _observe(ud, rows, w)
    ai = {a: i for i, a in enumerate(pu.ARMS)}
    cidx = {c: [ai[a] for a in arms] for c, arms in pu.COMBOS.items()}
    obs = _ArmObs(w, sd, s_obs, np.abs(s_obs / sd), cidx, len(targets), len(pu.ARMS))
    rng = stream(cfg, ud.families.index(family), split)
    cnt, cntc = _arm_counts(obs, redraws(ud, rows, cfg, rng))
    out = []
    for t, (rel, kpi) in enumerate(targets):
        tb = dict(base, relation=rel, kpi=kpi)
        m = meta[t]
        if m is None:
            out.append(dict(tb, status="undetermined", reason="degenerate_outcome"))
        else:
            out.append(_arm_result(tb, t, m, obs, (cnt, cntc, cfg.B + 1)))
    return out


def run_pmrt(
    pd: PmrtData,
    params: Mapping[str, Any],
    cfg: PmrtConfig | None = None,
    split: int = SPLIT_POOLED,
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Every family's CRT (`family_tests_pmrt`) over the family x relation x KPI hypotheses."""
    cfg = cfg or PmrtConfig()
    t0 = time.time()
    res = []
    for f in pd.ud.families if families is None else families:
        res.extend(family_tests_pmrt(pd, f, params, cfg, split))
    names = sorted({a for r in res if r["status"] == "tested" for a in r["arms"]})
    return {
        "version": pu.PMRT_VERSION,
        "legacy_version": pu.LEGACY_VERSION,
        "results": res,
        "arms": names,
        "wall_s": round(time.time() - t0, 2),
        "n_units": int(pd.n),
    }


def edges_pmrt(run: Mapping[str, Any], arm: str, q: float = 0.05) -> dict[tuple, dict[str, Any]]:
    """BY at `q` over the tested hypotheses of one arm or combination.

    Returns:
        Per (family, relation, KPI): `declared`, `sign`, `p`, `status`, `z_approx` and, when
        tested, `score` = -log10 p.
    """
    tested = [r for r in run["results"] if r["status"] == "tested" and arm in r["arms"]]
    p = np.array([r["arms"][arm]["p"] for r in tested], float)
    dec = weighted_step_up(p, None, q, reshape=True) if tested else []
    out = {}
    for r in run["results"]:
        out[(r["family"], r["relation"], r["kpi"])] = {
            "declared": False,
            "sign": 0,
            "p": float("nan"),
            "status": r["status"],
            "z_approx": float("nan"),
        }
    for r, d in zip(tested, dec, strict=True):
        a = r["arms"][arm]
        out[(r["family"], r["relation"], r["kpi"])] = {
            "declared": bool(d),
            "sign": int(a["sign"]),
            "p": a["p"],
            "status": "declared" if d else "not_detected",
            "z_approx": a["z"],
            "score": float(-np.log10(a["p"])),
        }
    return out


def _js(o: object) -> object:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def params_to_json(params: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe summary of fitted params, without the boosted models."""
    out = {k: params[k] for k in ("version", "config", "n_train_units", "n_train_eps")}
    out["hyp"] = {
        h: {
            "best": p["best"],
            "cv": p["cv"],
            "z_in": p["z_in"],
            "z_plain_train": p["core"]["z_plain"],
            "lam": p["core"]["lam"],
            "lw_shrink": p["core"]["lw_shrink"],
            "mu_full": np.asarray(p["core"]["mu_full"]).tolist(),
            "se_mu": np.asarray(p["core"]["se_mu"]).tolist(),
        }
        for h, p in params["hyp"].items()
    }
    out["rho"] = params["rho"]
    return out


def save_params(params: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Pickles fitted params to `path` and writes their JSON summary next to it."""
    p = Path(path)
    with p.open("wb") as fh:
        pickle.dump(params, fh)
    with p.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(params_to_json(params), fh, indent=1, default=_js)


def load_params(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Loads pickled params (trusted files only)."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


class _Stub:
    def __init__(self, *_a: object, **_k: object) -> None:
        pass

    def __setstate__(self, _state: object) -> None:
        pass

    def __call__(self, *_a: object, **_k: object) -> _Stub:
        return _Stub()


class _NoGbUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.split(".", maxsplit=1)[0] in ("sklearn", "_loss"):
            return _Stub
        return super().find_class(module, name)


def load_params_nogb(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Loads pickled params with every boosted model replaced by None (no sklearn coupling)."""
    with Path(path).open("rb") as fh:
        params = _NoGbUnpickler(fh).load()
    for hp in params["hyp"].values():
        hp["gb"] = None
    return params


def lean_columns(
    pd: PmrtData,
    family: str,
    params: Mapping[str, Any],
    cfg: PmrtConfig,
    arms: Sequence[str] = COL_ARMS,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], list[dict[str, Any] | None]]:
    """`family_columns` restricted to `arms`: only their kernels, no boosted model.

    `params` may be a frozen artifact holding only those kernels.

    Returns:
        `rows` in information order; `W`, column `t * len(arms) + a`; the targets; per target
        the arm orientations `o` and `z_plain_train`, or None when it is not testable.
    """
    ud = pd.ud
    rows = pu.unit_order(ud, ud.rows_of(family))
    rho = {g: np.array(params["rho"][g]) for g in params["rho"]}
    ep, sg = ud.episode[rows], ud.sgn[rows]
    need = sorted({pu.split_arm(a)[0] for a in arms})
    targets = _targets(ud)
    cols, meta = [], []
    for rel, kpi in targets:
        hp = params["hyp"].get(f"{family}|{rel}|{kpi}")
        if hp is None or not np.all(np.isfinite(ud.y[(rel, kpi)][rows])):
            cols.append(np.zeros((len(rows), len(arms))))
            meta.append(None)
            continue
        core = hp["core"]
        p, x, okp = pu.hyp_design(pd, rows, (family, rel, kpi), rho, params["ctx_keys"])
        lean = {**core, "K": {nm: core["K"][nm] for nm in need}}
        r = pu.center_all(core, pu.variant_scores(lean, p, x, okp), ep, sg, cfg)
        w = {}
        for a in arms:
            v, mod = pu.split_arm(a)
            w[a] = pu.modified(r[v], mod, pu.Fitted(core, hp["h"][v], cfg), x)
        cols.append(np.column_stack([w[a] for a in arms]))
        o = {a: pu.arm_direction(pu.split_arm(a)[0], core) for a in arms}
        meta.append({"o": o, "z_plain_train": float(core["z_plain"])})
    return rows, np.column_stack(cols), targets, meta


@dataclasses.dataclass(frozen=True)
class _IntObs:
    w: np.ndarray
    sd: np.ndarray
    z: np.ndarray
    o: np.ndarray
    nt: int

    @property
    def mx(self) -> list[int]:
        return [COL_ARMS.index(a) for a in MAX_OF]

    def z3(self) -> np.ndarray:
        return self.z.reshape(self.nt, len(COL_ARMS))

    def o3(self) -> np.ndarray:
        return self.o.reshape(self.nt, len(COL_ARMS))


def _max_obs(obs: _IntObs) -> dict[str, np.ndarray]:
    mx = obs.mx
    z3, o3 = obs.z3()[:, mx], obs.o3()[:, mx]
    return {
        "t": np.abs(z3).max(1),
        "ok": (o3 != 0).all(1),
        "plus": (o3 * z3).max(1),
        "minus": (-o3 * z3).max(1),
    }


def _integrated_counts(obs: _IntObs, draws: Iterator[np.ndarray]) -> dict[str, np.ndarray]:
    na, mx = len(COL_ARMS), obs.mx
    zo = obs.o * obs.z
    mo = _max_obs(obs)
    tol = 1e-9 * (1.0 + np.abs(obs.z))
    tolm = 1e-9 * (1.0 + mo["t"])
    ob = obs.o3()[None, :, :][:, :, mx]
    c = {k: np.zeros(obs.w.shape[1], np.int64) for k in ("2", "p", "m")}
    c.update({k: np.zeros(obs.nt, np.int64) for k in ("2m", "pm", "mm")})
    for v in draws:
        zb = (v @ obs.w) / obs.sd[None, :]
        del v
        c["2"] += np.count_nonzero(np.abs(zb) >= (np.abs(obs.z) - tol)[None, :], axis=0)
        zob = zb * obs.o[None, :]
        c["p"] += np.count_nonzero(zob >= (zo - tol)[None, :], axis=0)
        c["m"] += np.count_nonzero(zob <= (zo + tol)[None, :], axis=0)
        z3b = zb.reshape(len(zb), obs.nt, na)[:, :, mx]
        c["2m"] += np.count_nonzero(np.abs(z3b).max(2) >= (mo["t"] - tolm)[None, :], axis=0)
        c["pm"] += np.count_nonzero((ob * z3b).max(2) >= (mo["plus"] - tolm)[None, :], axis=0)
        c["mm"] += np.count_nonzero((-ob * z3b).max(2) >= (mo["minus"] - tolm)[None, :], axis=0)
    return c


def _integrated_target(
    t: int, meta: Mapping[str, Any], obs: _IntObs, c: Mapping[str, np.ndarray], b1: float
) -> dict[str, Any]:
    na, nan = len(COL_ARMS), float("nan")
    z = obs.z
    out: dict[str, Any] = {"status": "tested", "z_plain_train": meta["z_plain_train"]}
    plain_sign = float(np.sign(z[t * na + COL_ARMS.index("plain")]))
    for a in MAX_OF:
        j = t * na + COL_ARMS.index(a)
        oa = obs.o[j]
        sgn_ = float(np.sign(z[j])) * oa if oa else plain_sign
        out[a] = {
            "p2": (1 + c["2"][j]) / b1,
            "p_plus": (1 + c["p"][j]) / b1 if oa else nan,
            "p_minus": (1 + c["m"][j]) / b1 if oa else nan,
            "z": float(abs(z[j]) * (sgn_ or 1.0)),
            "z_raw": float(z[j]),
            "sign": int(sgn_),
            "o": oa,
        }
    ok = bool(_max_obs(obs)["ok"][t])
    jb = MAX_OF[int(np.argmax(np.abs(obs.z3()[t, obs.mx])))]
    out["max"] = {
        "p2": (1 + c["2m"][t]) / b1,
        "p_plus": (1 + c["pm"][t]) / b1 if ok else nan,
        "p_minus": (1 + c["mm"][t]) / b1 if ok else nan,
        "z": out[jb]["z"],
        "sign": out[jb]["sign"],
        "via": jb,
    }
    return out


def integrated_family(
    pd: PmrtData, family: str, params: Mapping[str, Any], cfg: PmrtConfig, split: int
) -> dict[tuple[str, str], dict[str, Any]]:
    """The frozen Study 1 statistic of one family.

    Per arm of `MAX_OF` and their max: the two-sided `p2` and the one-sided `p_plus` / `p_minus`
    in the effect direction (NaN when the arm orientation is 0), the signed `z`, `sign` and,
    for the max, the attaining arm `via`.

    Returns:
        Per target (relation, KPI) its `status` and, when tested, one entry per `STATS`.
    """
    ud = pd.ud
    _, ok = _support(ud, family, cfg)
    if not ok:
        return {t: {"status": "undetermined", "reason": "support"} for t in _targets(ud)}
    rows, w, targets, meta = lean_columns(pd, family, params, cfg)
    nt = len(targets)
    s_obs, sd = _observe(ud, rows, w)
    o = np.array([[(m["o"][a] if m else 0.0) for a in COL_ARMS] for m in meta]).ravel()
    obs = _IntObs(w, sd, s_obs / sd, o, nt)
    rng = stream(cfg, ud.families.index(family), split)
    c = _integrated_counts(obs, redraws(ud, rows, cfg, rng))
    out = {}
    for t, tg in enumerate(targets):
        m = meta[t]
        if m is None:
            out[tg] = {"status": "undetermined", "reason": "degenerate_outcome"}
        else:
            out[tg] = _integrated_target(t, m, obs, c, cfg.B + 1.0)
    return out


def run_integrated(
    pd: PmrtData, params: Mapping[str, Any], cfg: PmrtConfig, split: int
) -> dict[str, dict[str, Any]]:
    """`integrated_family` of every family, keyed `"family|relation|kpi"`."""
    res = {}
    for f in pd.ud.families:
        for (rel, kpi), o in integrated_family(pd, f, params, cfg, split).items():
            res[f"{f}|{rel}|{kpi}"] = o
    return res


def hypstats(run: Mapping[str, Mapping[str, Any]], stat: str) -> dict[tuple, HypStats]:
    """Declaration inputs of one statistic of `run_integrated` output."""
    out = {}
    for h, o in run.items():
        key = tuple(h.split("|"))
        if o.get("status") != "tested":
            out[key] = HypStats()
            continue
        s = o[stat]
        out[key] = HypStats(
            p2=float(s["p2"]),
            sign=int(s["sign"]),
            p_plus=float(s["p_plus"]),
            p_minus=float(s["p_minus"]),
        )
    return out


def prior_from_run(
    run: Mapping[str, Mapping[str, Any]], stat: str, n_prior: int = N_PRIOR
) -> Prior:
    """Prior of one statistic from its `run_integrated` output on independent episodes."""
    d = {
        h: (
            {"z": float(o[stat]["z"]), "status": "tested"}
            if o.get("status") == "tested"
            else {"status": "undetermined"}
        )
        for h, o in run.items()
    }
    return Prior.from_stats(d, n_prior=n_prior, source=f"ev2 {stat}")


def declare_all(
    run: Mapping[str, Mapping[str, Any]], priors: Mapping[str, Prior], n_target: int
) -> dict[tuple[str, str], dict]:
    """Declarations of every (statistic, layer) of `COMBOS`, with `z_approx` per hypothesis."""
    out = {}
    st = structure()
    for s, lay in COMBOS:
        cfg = DeclareConfig(Q, priors[s], n_target, FLOOR, THR, st)
        dec = declare(hypstats(run, s), lay, HYPOTHESES, cfg)
        for h, v in dec.items():
            o = run.get("|".join(h), {})
            v["z_approx"] = o.get(s, {}).get("z") if o.get("status") == "tested" else None
        out[(s, lay)] = dec
    return out


def cname(stat: str, layer: str) -> str:
    """Name of a (statistic, layer) combination."""
    return f"{stat}+{layer}"


def select(pool: records.Pool, spec: str) -> np.ndarray:
    """Global ids of the episodes matching any `key=value` of the comma-separated `spec`."""
    m = np.zeros(pool.n_eps, bool)
    for kv in [x for x in spec.split(",") if x]:
        k, v = kv.split("=", 1)
        m |= np.array([str(x) == v for x in pool.eps[k]])
    return np.nonzero(m)[0]


def subsets(
    pool: records.Pool, ids: np.ndarray, n: int, R: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Disjoint evaluation subsets of `n` episodes.

    n = 300 over four folds of 300 gives the folds (splits 1 .. 4); otherwise up to `R` random
    disjoint draws (splits 20 + i) from `default_rng([BENCH_TAG, n, R, seed])`.
    """
    e = pool.eps
    if n == 300 and all((e["fold"][ids] == k).sum() == 300 for k in range(4)):
        return [
            {"name": f"v3fold{k}", "episodes": ids[e["fold"][ids] == k], "split": 1 + k}
            for k in range(4)
        ]
    perm = np.random.default_rng([BENCH_TAG, int(n), int(R), int(seed)]).permutation(ids)
    return [
        {"name": f"rand{i}", "episodes": np.sort(perm[i * n : (i + 1) * n]), "split": 20 + i}
        for i in range(R)
        if (i + 1) * n <= len(perm)
    ]
