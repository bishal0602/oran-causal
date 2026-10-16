from typing import Any

import numpy as np
import pytest

from oran_causal.sim.system import config as C
from oran_causal.sim.system.decision import collect as CP
from oran_causal.sim.system.decision import gt as G
from oran_causal.sim.system.decision import labels as LP
from oran_causal.sim.system.decision.vocab import CELL_KPIS
from oran_causal.sim.system.env import E6Env
from oran_causal.sim.system.ric import knob_set

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def cfg_p3(seed: int = 7, warm: float = 60.0, scored: float = 60.0) -> C.E6Config:
    return C.E6Config(
        seed=seed,
        mix="ES",
        scenario="surge",
        load_factor=1.0,
        n_ue=300,
        n_pico=3,
        mobility="ped",
        warmup_s=warm,
        scored_s=scored,
        e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")),
    )


@pytest.fixture(scope="module")
def env60() -> E6Env:
    env = E6Env(cfg_p3(), log=False, wg3=True)
    while env.sec < 60:
        env.step(None)
    return env


def _ticks(plant: Any, n: int) -> None:
    for _ in range(n):
        plant.tick()


def test_forced_pico_sleep_moves_its_ues_to_the_best_remaining_cell(env60: E6Env) -> None:
    env = env60.copy()
    p = env.plant
    picos = [c for c in range(p.nc) if not p.lay.is_macro[c]]
    pico = max(picos, key=lambda c: int((p.serv == c).sum()))
    ues = np.nonzero(p.serv == pico)[0]
    assert len(ues) > 0
    assert not p.asleep[pico]
    now = float(env.sec)
    off = p.asleep.copy() | (p.waking_until > now)
    off[pico] = True
    want = np.argmax(np.where(off[None, :], -np.inf, p.l3[ues]), 1)
    before = np.bincount(p.serv, minlength=p.nc)
    knob_set(p, ("sleep", pico), 1.0, now)
    assert p.asleep[pico]
    ok = p.int_until[ues] - now <= C.RLF_OUTAGE_S - 1e-9
    assert ok.any()
    assert np.array_equal(p.serv[ues][ok], want[ok])
    assert not (p.serv[ues][ok] == pico).any()
    after = np.bincount(p.serv, minlength=p.nc)
    assert after[pico] <= before[pico] - int(ok.sum())
    moved = np.bincount(want[ok], minlength=p.nc)
    lost = int((~ok).sum())
    assert all(after[c] >= before[c] + moved[c] - lost for c in range(p.nc) if c != pico)
    assert int(ok.sum()) + lost == len(ues)


def test_forced_macro_carrier_off_changes_no_serving_cell(env60: E6Env) -> None:
    a, b = env60.copy(), env60.copy()
    pa, pb = a.plant, b.plant
    macros = [c for c in range(pa.nc) if pa.lay.is_macro[c] and pa.n_car[c] == pa.n_trx[c]]
    m = max(macros, key=lambda c: int((pa.serv == c).sum()))
    knob_set(pb, ("carrier", m), 1.0, float(b.sec))
    assert pb.n_car[m] == pa.n_car[m] - 1
    assert np.array_equal(pa.serv, pb.serv)
    assert np.array_equal(pa.gains(), pb.gains())
    _ticks(pa, 1)
    _ticks(pb, 1)
    assert np.array_equal(pa.serv, pb.serv)
    assert np.array_equal(pa.l3, pb.l3)
    assert pb.rho[m] >= pa.rho[m]
    for _ in range(C.TICKS_PER_CONTROL - 1):
        pa.tick()
        pb.tick()
        assert np.array_equal(pa.gains(), pb.gains())


def test_forced_ptx_drop_moves_the_a3_border_and_serving_cells(env60: E6Env) -> None:
    a, b = env60.copy(), env60.copy()
    pa, pb = a.plant, b.plant
    macros = [c for c in range(pa.nc) if pa.lay.is_macro[c]]
    m = max(macros, key=lambda c: int((pa.serv == c).sum()))
    knob_set(pb, ("ptx", m), pb.ptx_off[m] - 9.0, float(b.sec))
    ga, gb = pa.gains(), pb.gains()
    assert np.allclose(gb[:, m], ga[:, m] - 9.0)
    other = np.arange(pa.nc) != m
    assert np.array_equal(ga[:, other], gb[:, other])
    assert (np.argmax(gb, 1) == m).sum() < (np.argmax(ga, 1) == m).sum()
    n = 5 * C.TICKS_PER_CONTROL
    _ticks(pa, n)
    _ticks(pb, n)
    assert int((pb.serv == m).sum()) < int((pa.serv == m).sum())
    left = np.nonzero((pa.serv == m) & (pb.serv != m))[0]
    assert len(left) > 0
    assert not (pb.serv[left] == m).any()


@pytest.mark.claim("DEF-EST-TAUAA")
def test_labeller_cells_option_adds_per_cell_contrasts_and_keeps_the_default() -> None:
    H, got = 10, {}
    lab_c = LP.Labeller(ks=(1, 2), H=H, cells=True)
    lab_0 = LP.Labeller(ks=(1, 2), H=H)

    def hook(env: E6Env, obs: Any, snap: Any, opened: list[dict[str, Any]]) -> None:
        for u in opened:
            if (
                "u" in got
                or u["knob"] not in ("carrier", "ptx", "prot_min")
                or u["t0"] + H > env.total_s
            ):
                continue
            at = LP.Paused(env, obs, snap)
            got.update(
                u=u, c=lab_c.label(at, u, modes=G.GT_MODES), d=lab_0.label(at, u, modes=G.GT_MODES)
            )

    res = CP.run_collection(
        cfg_p3(seed=9, scored=40.0),
        CP.RandomizedUnitPolicy(9, tables=CP.PI0_HIGH_NO_RB),
        CP.Collection(open_rule="feasible"),
        labeller=hook,
    )
    assert "u" in got
    u, lc, l0 = got["u"], got["c"], got["d"]
    assert "cell" not in l0
    assert "cell_kpis" not in l0
    for key in ("modes", "H", "ks", "raw", "d", "mean"):
        assert lc[key] == l0[key]
    assert lc["cell_kpis"] == list(CELL_KPIS)
    assert lc["cell"]["reject"].shape == (2, len(CELL_KPIS), res["env"].plant.nc)
    assert not lc["cell"]["accept"].any()
    for i in range(2):
        for j, kpi in enumerate(LP.KPIS):
            got_sum = lc["cell"]["reject"][i, j, u["exp"]].sum()
            assert got_sum == pytest.approx(lc["d"]["reject"]["exp"][kpi][i])
    assert np.array_equal(G.unit_delta(lc), -lc["cell"]["reject"])
    with pytest.raises(ValueError, match="per-cell"):
        G.unit_delta(l0)
