"""X7 run half: are the associational referees the static subset rule `sub:ES+PowerES`.

Post hoc and descriptive, on the Study 3 EVAL seeds 191100-191259; the frozen certified-safe driver
is used and nothing frozen is edited. Job kinds (record arm names, sub `x7`):

* `shadow`: `sub:ES+PowerES` applied, the four referees queried in shadow on a deep copy of every
  observation.
* `own`: the referee applied (the frozen arm, `Counting` wrapper as stored), the subset rule in
  shadow.
* `pm`: the referee's decisions with every non-prot_min deferral replaced by accept.
* `other`: the referee's decisions with every prot_min deferral replaced by accept.

Per-request tallies are `record["agree"][shadow]["<family>|<dir>|<pre|post>"] = [aa, ar, ra, rr]`.
Scoring the records is `oran_causal.analysis.posthoc.x7`.

    python -m oran_causal.experiments.posthoc.x7 run --part i/k --out F.jsonl
        [--seeds a,b] [--kinds shadow,own,pm,other] [--smoke] [--short S]

`run` takes the referees (`refs`), a subset of the job arms (`arms`) and the maps / certsafe
artifacts as arguments; the defaults are the frozen ones.
"""

import argparse
import copy
import dataclasses
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.posthoc.x7 import (
    DRIVER,
    FAMS,
    KINDS,
    REF,
    REFS,
    SUB,
    arm_name,
    job_list,
)
from oran_causal.experiments.campaign.host import PLATFORM_ENV
from oran_causal.experiments.campaign.provenance import COMMIT_ENV, DIRTY_ENV
from oran_causal.experiments.launch.fingerprint import numeric_env
from oran_causal.experiments.system import study2, study3
from oran_causal.experiments.system.common import (
    LOAD_FACTOR_L40,
    FrozenDoc,
    Job,
    Part,
    Record,
    cfg_seed,
    header,
    outcome,
    run_env,
    run_jobs,
)
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision.scenario import PAIR, STRATUM, make_cfg

STAGE = "eval"
Obs = dict[str, Any]
Decide = Callable[[Obs], dict[str, Any]]


def _direction(r: Mapping[str, Any]) -> int:
    return int((r["prop"] > r["cur"]) - (r["prop"] < r["cur"]))


def _key(r: Mapping[str, Any], t: float) -> str:
    return f"{r['knob'][0]}|{_direction(r):+d}|{'pre' if t < rf.WARM else 'post'}"


def _rej(d: str) -> int:
    return int(d == "reject")


class Shadowed:
    """Applies `primary`'s decisions and queries every shadow referee on a copy of the observation.

    The copy is taken before the primary sees the observation; the per-request decision pairs
    are tallied in `agree`. Nothing is added to the applied decisions.
    """

    def __init__(self, primary: Decide, shadows: Mapping[str, Decide]) -> None:
        self.primary, self.shadows = primary, shadows
        self.agree: dict[str, dict[str, list[int]]] = {nm: {} for nm in shadows}

    def __call__(self, obs: Obs) -> dict[str, Any]:
        """The primary's decisions, with the shadows tallied."""
        copies = {nm: copy.deepcopy(obs) for nm in self.shadows}
        out = self.primary(obs)
        for nm, sh in self.shadows.items():
            sd = sh(copies[nm])["decisions"]
            tab = self.agree[nm]
            for r, a, b in zip(obs["requests"], out["decisions"], sd, strict=True):
                c = tab.setdefault(_key(r, obs["t"]), [0, 0, 0, 0])
                c[2 * _rej(a) + _rej(b)] += 1
        return out


class Restrict:
    """`inner`'s decisions with every deferral outside the `keep` families replaced by accept.

    The inner arbiter's state is untouched; only its output is overridden.
    """

    def __init__(self, inner: Decide, keep: frozenset[str]) -> None:
        self.inner, self.keep = inner, frozenset(keep)
        self.n = {"overridden": 0, "kept": 0}

    def __call__(self, obs: Obs) -> dict[str, Any]:
        """The inner decisions, restricted."""
        out = self.inner(obs)
        dec = list(out["decisions"])
        for i, (r, d) in enumerate(zip(obs["requests"], dec, strict=True)):
            if d == "reject":
                if r["knob"][0] in self.keep:
                    self.n["kept"] += 1
                else:
                    dec[i] = "accept"
                    self.n["overridden"] += 1
        return {**out, "decisions": dec}


def anchor(arm: str, warmup_s: float) -> Decide:
    """Arbiter of an anchor arm that is a policy (not accept-all)."""
    arb = rf.anchor_arbiter(arm, warmup_s)
    if arb is None:
        raise ValueError(f"{arm} is accept-all")
    return arb


def raw_ref(ref: str, art: Mapping[str, Any]) -> Any:
    """The referee `ref` of the artifact without its `Counting` wrapper."""
    t = art["tables"][ref]
    return rf.certsafe_arbiter(t["map"], t["uncertified"], rf.THETA, rf.K_CONF)


def make(
    kind: str,
    ref: str | None,
    sd: int,
    warmup_s: float,
    art: Mapping[str, Any],
    refs: Sequence[str] = REFS,
) -> Any:
    """The arbiter of a job kind; the shadow job queries `refs`.

    Raises:
        ValueError: Unknown kind.
    """
    if kind == "shadow":
        return Shadowed(anchor(REF, warmup_s), {r: raw_ref(r, art) for r in refs})
    if ref is None:
        raise ValueError(f"job kind {kind} needs a referee")
    if kind == "own":
        primary = rf.cs_arbiter(ref, sd, warmup_s, art)
        if primary is None:
            raise ValueError(f"{ref} is accept-all")
        return Shadowed(primary, {REF: anchor(REF, warmup_s)})
    if kind not in ("pm", "other"):
        raise ValueError(kind)
    keep = frozenset({"prot_min"}) if kind == "pm" else frozenset(FAMS) - {"prot_min"}
    return rf.Counting(Restrict(raw_ref(ref, art), keep), is_unit=False)


def policy_block(arb: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The `policy_counts` and `agree` fields of an arbiter after its episode."""
    if isinstance(arb, Shadowed):
        p = arb.primary
        return (rf.policy_counts(p) if isinstance(p, rf.Counting) else {}), arb.agree
    pc = dict(arb.n)
    pol = getattr(arb.inner.inner, "policy", None)
    pc["policy"] = dict(getattr(pol, "n", None) or {})
    pc["restrict"] = dict(arb.inner.n)
    return pc, None


def job(
    seed: int,
    kind: str,
    ref: str | None,
    art: Mapping[str, Any],
    *,
    smoke: bool = False,
    short: float | None = None,
    refs: Sequence[str] = REFS,
) -> Record:
    """One episode, with the certified-safe job record layout and sub `x7`."""
    study3.check_stage_seed(seed, STAGE)
    sd = cfg_seed(seed, smoke)
    cfg = make_cfg(PAIR, STRATUM, sd, LOAD_FACTOR_L40)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make(kind, ref, sd, cfg.warmup_s, art, refs)
    t_cpu = time.process_time()
    env = run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    pc, agree = policy_block(arb)
    return {
        "kind": "job",
        "schema": study2.SCHEMA_EVAL,
        "key": [seed, arm_name(ref, kind)],
        "seed": seed,
        "cfg_seed": sd,
        "arm": arm_name(ref, kind),
        "sub": SUB,
        "cs_stage": STAGE,
        "smoke": smoke,
        "short": short,
        "episode_s": env.total_s,
        "warmup_s": cfg.warmup_s,
        "load_factor": LOAD_FACTOR_L40,
        **outcome(env),
        "policy_counts": pc,
        "agree": agree,
        "x7": {"kind": kind, "ref": ref},
        "cpu_s": round(cpu, 2),
    }


def jobs(
    art: Mapping[str, Any],
    seeds: Sequence[int] | None = None,
    kinds: Iterable[str] = KINDS,
    *,
    smoke: bool = False,
    short: float | None = None,
    refs: Sequence[str] = REFS,
    arms: frozenset[str] | None = None,
) -> list[Job]:
    """The resumable jobs of the run in shard order (seed-major); `arms` keeps those job arms."""
    todo = job_list(list(seeds) if seeds else study3.stage_seeds(STAGE), tuple(kinds), refs)
    return [
        Job(
            (s, arm_name(r, k)),
            study3.check_stage_seed(s, STAGE),
            lambda s=s, k=k, r=r: job(s, k, r, art, smoke=smoke, short=short, refs=refs),
        )
        for s, k, r in todo
        if arms is None or arm_name(r, k) in arms
    ]


def head(
    part: Part,
    kinds: Sequence[str],
    *,
    smoke: bool,
    short: float | None,
    environ: Mapping[str, str] | None = None,
    refs: Sequence[str] = REFS,
    maps_doc: FrozenDoc = study2.MAPS,
    art_doc: FrozenDoc = study3.CERTSAFE,
) -> Record:
    """The header line of a run call."""
    env = os.environ if environ is None else environ
    return header(
        DRIVER,
        STAGE,
        part,
        smoke,
        schema=study2.SCHEMA_EVAL,
        sub=SUB,
        cs_stage=STAGE,
        platform=numeric_env(),
        host=env.get(PLATFORM_ENV),
        code_commit=env.get(COMMIT_ENV),
        code_dirty=env.get(DIRTY_ENV),
        maps=maps_doc.status(),
        artifact=art_doc.status(),
        certsafe_freeze=study3.PROTOCOL.status(),
        consts={
            "ref": REF,
            "refs": list(refs),
            "kinds": list(kinds),
            "load_factor": LOAD_FACTOR_L40,
            "smoke": {"short": short} if smoke else None,
        },
    )


def run(
    part: Part,
    out: Path,
    *,
    smoke: bool = False,
    short: float | None = None,
    seeds: Sequence[int] | None = None,
    kinds: Sequence[str] = KINDS,
    refs: Sequence[str] = REFS,
    arms: frozenset[str] | None = None,
    maps_doc: FrozenDoc = study2.MAPS,
    art_doc: FrozenDoc = study3.CERTSAFE,
) -> int:
    """Runs shard `part` of the X7 jobs into `out`; returns the number of jobs run.

    Raises:
        ValueError: `short` without `smoke`.
    """
    if short and not smoke:
        raise ValueError("--short is smoke-only")
    art = study3.load_artifact(art_doc.path)
    show = ("seed", "arm", "psvr", "rlf", "st_req", "st_rej", "cpu_s")
    return run_jobs(
        out,
        head(part, kinds, smoke=smoke, short=short, refs=refs, maps_doc=maps_doc, art_doc=art_doc),
        jobs(art, seeds, kinds, smoke=smoke, short=short, refs=refs, arms=arms),
        part,
        smoke,
        show,
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--part", default="0/1")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--short", type=float, default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--kinds", default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: `run` a shard of the X7 jobs."""
    a = _parser().parse_args(argv)
    run(
        Part.parse(a.part),
        Path(a.out),
        smoke=a.smoke,
        short=a.short,
        seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None,
        kinds=tuple(a.kinds.split(",")) if a.kinds else KINDS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
