"""Claim-marked test results exchanged between the pytest plugin and the checker."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

FORMAT_VERSION = 1
DEFAULT_PATH = Path("build/trace/tests.json")


@dataclass(frozen=True, slots=True)
class MarkedTest:
    """Outcome of one claim-marked test."""

    nodeid: str
    claims: tuple[str, ...]
    outcome: str

    @property
    def passed(self) -> bool:
        """Whether the test passed."""
        return self.outcome == "passed"


def write_results(path: Path, results: list[MarkedTest]) -> None:
    """Write results as sorted JSON so reruns produce identical files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tests = [asdict(result) for result in sorted(results, key=lambda r: r.nodeid)]
    payload = {"version": FORMAT_VERSION, "tests": tests}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def read_results(path: Path) -> tuple[MarkedTest, ...] | None:
    """Read results written by `write_results`; `None` when the file does not exist."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        MarkedTest(item["nodeid"], tuple(item["claims"]), item["outcome"])
        for item in payload["tests"]
    )
