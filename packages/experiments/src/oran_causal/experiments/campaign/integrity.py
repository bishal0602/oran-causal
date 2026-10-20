"""Record integrity: run modes, the integrity stamp, EVAL preconditions and merge summaries."""

import platform
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from oran_causal.experiments.campaign.extras import is_extras
from oran_causal.experiments.campaign.provenance import PythonPin
from oran_causal.experiments.campaign.spec import (
    ProtocolTexts,
    eval_authorised,
    is_eval,
    spec_sha256,
)

EVAL_CAP_FACTOR = 2.0
MERGE_SETS = (
    "python",
    "lock_sha256",
    "spec_file_sha256",
    "isolation",
    "budget_cpu_s",
    "cost_table_sha256",
    "parts",
)

type Record = dict[str, Any]
type Spec = dict[str, Any]


def record_status(r: Record) -> str:
    """`status` of a record, or `ok` / `error` from its `error` field for records without one."""
    return r.get("status") or ("ok" if r.get("error") is None else "error")


def run_mode(spec: Spec) -> dict[str, Any]:
    """The `run_mode` block of the spec's records."""
    return {
        "mode": "eval" if is_eval(spec) else "dev",
        "protocol_sha256": spec.get("protocol_sha256"),
    }


def mode_tag(mode: dict[str, Any]) -> tuple[str, str | None]:
    """(mode, protocol sha for EVAL else None): what separates output files."""
    return mode["mode"], mode["protocol_sha256"] if mode["mode"] == "eval" else None


def record_mode_tag(r: Record) -> tuple[str, str | None]:
    """`mode_tag` of a stored record; records without a run mode are DEV."""
    rm = r.get("run_mode") or {}
    mode = rm.get("mode", "dev")
    return mode, rm.get("protocol_sha256") if mode == "eval" else None


def accept_rule(spec: Spec) -> Callable[[Record], bool]:
    """The records a run of `spec` may use.

    EVAL accepts only EVAL records of the spec's protocol and of this spec (canonical sha);
    DEV accepts only DEV records; an extras spec only DEV records of this spec, because its
    designs share unit keys.
    """
    if is_extras(spec):
        sha = spec_sha256(spec)
        return lambda r: (
            (r.get("run_mode") or {}).get("mode", "dev") == "dev"
            and (r.get("integrity") or {}).get("spec_sha256") == sha
        )
    if not is_eval(spec):
        return lambda r: (r.get("run_mode") or {}).get("mode", "dev") == "dev"
    psha, ssha = spec.get("protocol_sha256"), spec_sha256(spec)

    def accept(r: Record) -> bool:
        rm = r.get("run_mode") or {}
        same_protocol = rm.get("mode") == "eval" and rm.get("protocol_sha256") == psha
        return same_protocol and (r.get("integrity") or {}).get("spec_sha256") == ssha

    return accept


def integrity_stamp(
    spec: Spec,
    code: dict[str, Any],
    pins: dict[str, Any],
    *,
    isolate: bool,
    budget: float | None,
    cap_factor: float,
    spec_file_sha256: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The `integrity` block stamped on every record of a run."""
    return {
        "code_commit": code.get("commit"),
        "code_dirty": code.get("dirty"),
        "protocol_sha256": spec.get("protocol_sha256"),
        "spec_sha256": spec_sha256(spec),
        "spec_file_sha256": spec_file_sha256,
        "python": platform.python_version(),
        "lock_sha256": pins.get("lock_sha256"),
        "pin_mismatches": pins.get("mismatches"),
        "torch_build": pins.get("torch_build"),
        "isolation": "fork+rlimit" if isolate else "in-process",
        "budget_cpu_s": budget,
        "cap_factor": cap_factor,
        "cost_table_sha256": None,
        "part": None,
        "parts": None,
        **(extra or {}),
    }


def check_eval_preconditions(
    spec: Spec,
    code: dict[str, Any],
    pins: dict[str, Any],
    texts: ProtocolTexts,
    python: PythonPin | None,
) -> None:
    """EVAL run preconditions.

    Refuses an unauthorised protocol, an interpreter the workspace pin does not admit, unknown
    or dirty code, and installed packages that differ from `uv.lock`.
    """
    ok, why = eval_authorised(spec, texts)
    if not ok:
        raise ValueError(f"EVAL refused: {why}")
    if python is not None and not python.admits(platform.python_version()):
        raise ValueError(
            f"EVAL refused: Python {platform.python_version()}, the workspace pins {python}"
        )
    if not code.get("commit") or code.get("dirty") is not False:
        raise ValueError(f"EVAL refused: code commit unknown or not clean ({code})")
    if pins.get("mismatches") is None:
        raise ValueError("EVAL refused: no uv.lock to check the installed packages against")
    if pins["mismatches"]:
        shown = dict(list(pins["mismatches"].items())[:8])
        raise ValueError(f"EVAL refused: installed packages differ from the lock: {shown}")


def check_one_per_cpu(n_procs: int | None, quota: int) -> None:
    """EVAL allows at most one campaign process per vCPU of the session."""
    if n_procs is not None and n_procs > quota:
        raise ValueError(
            f"EVAL refused: {n_procs} campaign processes on {quota} vCPU(s); "
            "R-55 allows one per vCPU"
        )


class MergeTally:
    """Accumulates the integrity sets of merged records, one record at a time."""

    def __init__(self) -> None:
        self.by_status: dict[str, list[str]] = defaultdict(list)
        self.commits: set[str] = set()
        self.dirty = 0
        self.platforms: set[str] = set()
        self.protocols: set[str] = set()
        self.specs: set[str] = set()
        self.seen: dict[str, set[str]] = defaultdict(set)

    def add(self, r: Record) -> None:
        """Counts one merged record."""
        code = r.get("code") or {}
        integ = r.get("integrity") or {}
        self.by_status[record_status(r)].append(r["key"])
        self.commits.add(str(code.get("commit")))
        self.dirty += code.get("dirty") is not False
        self.platforms.add(str((r.get("host") or {}).get("platform")))
        self.protocols.add(str(integ.get("protocol_sha256")))
        self.specs.add(str(integ.get("spec_sha256")))
        for f in MERGE_SETS:
            self.seen[f].add(str(integ.get(f)))

    def summary(self, want: set[str], stats: dict[str, Any], mode: str) -> dict[str, Any]:
        """The merge summary: completeness, merge statistics and integrity sets."""
        have = {k for keys in self.by_status.values() for k in keys}
        return {
            "mode": mode,
            "n_expected": len(want),
            "n_records": len(have),
            "n_ok": len(self.by_status["ok"]),
            "missing": sorted(want - have),
            "errors": sorted(self.by_status["error"]),
            "infeasible": sorted(self.by_status["infeasible"]),
            "unexpected": sorted(have - want),
            **stats,
            "commits": sorted(self.commits),
            "dirty_records": self.dirty,
            "platforms": sorted(self.platforms),
            "protocol_sha256": sorted(self.protocols),
            "spec_sha256": sorted(self.specs),
            **{f: sorted(self.seen[f]) for f in MERGE_SETS},
        }
