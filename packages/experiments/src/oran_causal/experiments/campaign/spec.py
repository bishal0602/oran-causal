"""Campaign specs: validation, EVAL authorisation and expansion into units.

A UNIT (one shard record) is (arm, world, regime, lam, n, kappa, seed); its key is the runner's job
key with the arm name as the method. EVAL mode is a spec with any seed >= 3_100_000; it is refused
unless the spec's constant `protocol_sha256` matches a frozen protocol text.
"""

import dataclasses
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oran_causal.experiments.harness.runner import (
    DEV_SEEDS,
    EVAL_FLOOR,
    cell_key,
    job_key,
    kappas,
    seed_range,
)
from oran_causal.experiments.paths import STUDY4_PROTOCOL
from oran_causal.sim.controlled.datasets.truth import REGIMES_OF

ROLES = ("tune", "measure")
DECLARE = ("by", "tau")
DEFAULT_E4_LAMS = {"R1": [1.0], "R2": [1.0], "R3": [0.0, 0.5, 1.0, 1.5], "R4": [0.0, 0.5, 1.0, 1.5]}
FROZEN_MARK = "FROZEN: yes"

type Spec = dict[str, Any]
type DatasetKey = tuple[str, str, float | None, int, int, float]


@dataclasses.dataclass(frozen=True, order=True)
class Unit:
    """One (world, regime, lam, n, kappa, seed, arm) job with its seed role."""

    world: str
    regime: str
    lam: float | None
    n: int
    kappa: float
    seed: int
    arm: str
    role: str

    @property
    def key(self) -> str:
        """The record key."""
        return job_key(self.arm, self.dataset)

    @property
    def dataset(self) -> DatasetKey:
        """The dataset coordinates shared by every arm of the dataset."""
        return (self.world, self.regime, self.lam, self.n, self.seed, self.kappa)

    @property
    def cell(self) -> str:
        """`<arm>|<world>|<regime>|[lam<l>|]k<kappa>|n<n>`."""
        return f"{self.arm}|{cell_key(self.world, self.regime, self.lam, self.n, self.kappa)}"


def sha_lf(data: bytes) -> str:
    """sha256 of `data` with CRLF normalised to LF (the git blob content of a text file)."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def is_frozen(data: bytes) -> bool:
    """Whether a protocol text has a line starting with `FROZEN: yes`."""
    lines = data.replace(b"\r\n", b"\n").decode("utf-8").split("\n")
    return any(line.startswith(FROZEN_MARK) for line in lines)


@dataclasses.dataclass(frozen=True)
class ProtocolTexts:
    """Protocol texts the EVAL guard may verify the spec's constant against.

    Attributes:
        live: The protocol file as shipped, or None if missing.
        at_commit: Returns the protocol text at a freeze commit, or None.
        frozen_copy: A frozen copy bundled by a launcher, or None.
    """

    live: bytes | None
    at_commit: Callable[[str], bytes | None] = lambda _commit: None
    frozen_copy: bytes | None = None


def shipped_protocol(path: Path = STUDY4_PROTOCOL) -> ProtocolTexts:
    """The frozen Study 4 protocol shipped in `packages/experiments/protocols`."""
    return ProtocolTexts(live=path.read_bytes() if path.exists() else None)


def _hex_digest_error(want: str) -> str | None:
    if not want:
        return "spec has no 'protocol_sha256'"
    if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
        return f"spec 'protocol_sha256' is not a sha256 hex digest: {want!r}"
    return None


def _sources(spec: Spec, texts: ProtocolTexts) -> list[tuple[str, bytes | None]]:
    sources: list[tuple[str, bytes | None]] = [("live file", texts.live)]
    if spec.get("freeze_commit"):
        commit = str(spec["freeze_commit"])
        sources.append((f"git blob at {commit}", texts.at_commit(commit)))
    if texts.frozen_copy is not None:
        sources.append(("bundled frozen copy", texts.frozen_copy))
    return sources


def eval_authorised(spec: Spec, texts: ProtocolTexts) -> tuple[bool, str]:
    """EVAL guard: the live protocol is frozen and one frozen text matches the spec's constant.

    Returns:
        (allowed, reason).
    """
    want = str(spec.get("protocol_sha256") or "").lower()
    error = _hex_digest_error(want)
    if error:
        return False, error
    if texts.live is None:
        return False, "protocol file missing"
    if not is_frozen(texts.live):
        return False, "protocol file is not frozen (no line starting 'FROZEN: yes')"
    sources = _sources(spec, texts)
    for name, data in sources:
        if data is not None and sha_lf(data) == want and is_frozen(data):
            return True, f"EVAL authorised: protocol sha256 {want} verified by the {name}"
    checked = ", ".join(n for n, d in sources if d is not None)
    return (
        False,
        f"protocol sha256 constant {want} matches no frozen protocol text (checked: {checked})",
    )


def seed_list(seeds: Any, spec: Spec, texts: ProtocolTexts) -> list[int]:
    """Seeds of a block: DEV always, EVAL (>= 3_100_000) only when authorised."""
    if isinstance(seeds, str) or any(isinstance(x, str) for x in seeds):
        raise ValueError(
            f"seeds must be numbers (a 'TBD' placeholder is filled at the freeze): {seeds!r}"
        )
    out = seed_range(seeds)
    bad = [x for x in out if x not in DEV_SEEDS and x < EVAL_FLOOR]
    if bad:
        raise ValueError(f"seeds outside the DEV block 3000000-3000199 (CONTRACT 6): {bad[:5]}")
    ev = [x for x in out if x >= EVAL_FLOOR]
    if ev:
        ok, why = eval_authorised(spec, texts)
        if not ok:
            raise ValueError(f"EVAL seeds {ev[:3]} refused: {why}")
    return out


def is_eval(spec: Spec) -> bool:
    """EVAL mode: any block seed >= 3_100_000."""
    for b in spec.get("blocks", []):
        try:
            if any(int(x) >= EVAL_FLOOR for x in b.get("seeds", [])):
                return True
        except TypeError, ValueError:
            continue
    return False


def spec_sha256(spec: Any) -> str:
    """sha256 of the canonical JSON of a spec (independent of file formatting)."""
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_arms(arms: dict[str, Any]) -> None:
    for a, d in arms.items():
        if "|" in a or not d.get("ref"):
            raise ValueError(f"arm {a!r}: needs a 'ref' and a name without '|'")
        if d.get("declare") not in DECLARE:
            raise ValueError(f"arm {a!r}: 'declare' must be one of {DECLARE}")


def _validate_blocks(spec: Spec) -> None:
    for b in spec.get("blocks", []):
        if b.get("role") not in ROLES:
            raise ValueError(f"block role must be one of {ROLES}: {b}")
        kappas(b)
        sel = b.get("arms", "all")
        if sel != "all" and set(sel) - set(spec["arms"]):
            raise ValueError(f"block names unknown arms: {sorted(set(sel) - set(spec['arms']))}")


def _validate_eval_roles(spec: Spec, texts: ProtocolTexts) -> None:
    for b in spec["blocks"]:
        sd = seed_list(b["seeds"], spec, texts)
        if b["role"] == "measure" and min(sd) < EVAL_FLOOR:
            raise ValueError("EVAL spec: measure blocks must use EVAL seeds (>= 3100000) only")
        if b["role"] == "tune" and max(sd) >= EVAL_FLOOR:
            raise ValueError("EVAL spec: tune blocks must use DEV seeds only")


def _role_seeds(spec: Spec, role: str, texts: ProtocolTexts) -> set[int]:
    return {
        s for b in spec["blocks"] if b["role"] == role for s in seed_list(b["seeds"], spec, texts)
    }


def validate_spec(spec: Spec, texts: ProtocolTexts) -> None:
    """Refuses malformed specs, unauthorised EVAL seeds and overlapping tune / measure seeds."""
    if not spec.get("arms"):
        raise ValueError("spec needs 'arms'")
    _validate_arms(spec["arms"])
    _validate_blocks(spec)
    if is_eval(spec):
        _validate_eval_roles(spec, texts)
    overlap = _role_seeds(spec, "tune", texts) & _role_seeds(spec, "measure", texts)
    if overlap:
        raise ValueError(f"tune and measurement seeds overlap: {sorted(overlap)[:5]}")


def _arm_applies(arm: dict[str, Any], world: str, n: int) -> bool:
    if "worlds" in arm and world not in arm["worlds"]:
        return False
    return not ("max_n" in arm and int(n) > int(arm["max_n"]))


def _block_units(spec: Spec, block: dict[str, Any], texts: ProtocolTexts) -> list[Unit]:
    arms = spec["arms"]
    names = list(arms) if block.get("arms", "all") == "all" else list(block["arms"])
    lam_map = {**DEFAULT_E4_LAMS, **block.get("e4_lams", {})}
    units = []
    for w in block["worlds"]:
        for r in (r for r in block["regimes"] if r in REGIMES_OF[w]):
            lams = [float(x) for x in lam_map[r]] if w == "E4" else [None]
            cells = [(lam, k, int(n)) for lam in lams for k in kappas(block) for n in block["ns"]]
            seeds = seed_list(block["seeds"], spec, texts)
            units += [
                Unit(w, r, lam, n, float(k), int(s), a, block["role"])
                for lam, k, n in cells
                for s in seeds
                for a in names
                if _arm_applies(arms[a], w, n)
            ]
    return units


def expand(spec: Spec, texts: ProtocolTexts | None = None) -> list[Unit]:
    """All units of the spec, sorted (dataset-major), deduplicated across blocks.

    Args:
        spec: The campaign spec.
        texts: Protocol texts for the EVAL guard; defaults to the shipped Study 4 protocol.

    Returns:
        The sorted units.
    """
    texts = texts if texts is not None else shipped_protocol()
    validate_spec(spec, texts)
    out: dict[str, Unit] = {}
    for block in spec["blocks"]:
        for u in _block_units(spec, block, texts):
            out.setdefault(u.key, u)
    return sorted(out.values())
