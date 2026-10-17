"""Pytest plugin: the `claim` marker and the `build/trace/tests.json` writer."""

from pathlib import Path

import pytest

from oran_causal.trace.results import DEFAULT_PATH, MarkedTest, write_results

MARKER = "claim"
USER_PROPERTY = "claim_ids"
MARKER_HELP = "claim(id, *more): links a test to claim ids in spec/claims.yaml"


def _claim_ids(item: pytest.Item) -> tuple[str, ...]:
    return tuple(str(arg) for mark in item.iter_markers(MARKER) for arg in mark.args)


class ClaimCollector:
    """Collects the outcome of every claim-marked test and writes them at session end."""

    def __init__(self, output: Path, *, always_write: bool) -> None:
        self.output = output
        self.always_write = always_write
        self.claims: dict[str, tuple[str, ...]] = {}
        self.outcomes: dict[str, str] = {}

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        """Remember the claim ids of every marked test."""
        for item in items:
            ids = _claim_ids(item)
            if ids:
                self.claims[item.nodeid] = ids

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        """Attach the claim ids to the test's reports, which cross the xdist worker boundary."""
        ids = _claim_ids(item)
        if ids:
            item.user_properties.append((USER_PROPERTY, ids))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record the outcome of a marked test; setup or teardown failures count as errors."""
        for name, value in report.user_properties:
            if name == USER_PROPERTY and isinstance(value, tuple):
                self.claims.setdefault(report.nodeid, value)
        if report.nodeid not in self.claims:
            return
        if report.when == "call":
            self.outcomes[report.nodeid] = report.outcome
        elif report.failed:
            self.outcomes[report.nodeid] = "error"
        elif report.skipped:
            self.outcomes.setdefault(report.nodeid, "skipped")

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Write the results when a marked test ran or `--trace-out` was given."""
        if hasattr(session.config, "workerinput"):
            return
        if not (self.claims or self.always_write):
            return
        results = [
            MarkedTest(nodeid, claims, self.outcomes.get(nodeid, "not run"))
            for nodeid, claims in self.claims.items()
        ]
        write_results(self.output, results)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add `--trace-out`."""
    parser.addoption(
        "--trace-out",
        default=None,
        help=f"where to write claim-marked test results (default: <rootdir>/{DEFAULT_PATH})",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the `claim` marker and the result collector."""
    config.addinivalue_line("markers", MARKER_HELP)
    option = config.getoption("--trace-out")
    output = Path(option) if option else config.rootpath / DEFAULT_PATH
    collector = ClaimCollector(output, always_write=option is not None)
    config.pluginmanager.register(collector, "oran_causal_trace_collector")
