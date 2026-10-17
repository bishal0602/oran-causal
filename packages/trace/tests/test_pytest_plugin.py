import json

import pytest

pytest_plugins = ["pytester"]

TESTS = """
import pytest

@pytest.mark.claim("DEF-A", "DEF-B")
def test_two_claims():
    pass

@pytest.mark.claim("DEF-C")
def test_fails():
    assert False

@pytest.mark.claim("DEF-D")
@pytest.mark.skip(reason="not yet")
def test_skipped():
    pass

@pytest.fixture
def broken():
    raise RuntimeError

@pytest.mark.claim("DEF-E")
def test_setup_error(broken):
    pass

def test_unmarked():
    pass
"""


def outcomes(pytester: pytest.Pytester) -> dict[str, tuple[list[str], str]]:
    payload = json.loads((pytester.path / "build" / "trace" / "tests.json").read_text())
    assert payload["version"] == 1
    return {
        item["nodeid"].split("::")[1]: (item["claims"], item["outcome"])
        for item in payload["tests"]
    }


def test_plugin_records_claim_marked_outcomes(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_sample=TESTS)
    result = pytester.runpytest("--strict-markers", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=2, failed=1, skipped=1, errors=1)
    assert outcomes(pytester) == {
        "test_two_claims": (["DEF-A", "DEF-B"], "passed"),
        "test_fails": (["DEF-C"], "failed"),
        "test_skipped": (["DEF-D"], "skipped"),
        "test_setup_error": (["DEF-E"], "error"),
    }


def test_marker_is_registered(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--markers")
    result.stdout.fnmatch_lines(["@pytest.mark.claim(id, [*]more)*"])


def test_no_file_without_claim_marked_tests(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_plain="def test_plain():\n    pass\n")
    pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1)
    assert not (pytester.path / "build").exists()


def test_trace_out_always_writes(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_plain="def test_plain():\n    pass\n")
    target = pytester.path / "out.json"
    pytester.runpytest("-p", "no:cacheprovider", f"--trace-out={target}")
    assert json.loads(target.read_text()) == {"version": 1, "tests": []}


def test_plugin_records_every_claim_test_under_xdist(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_sample=TESTS)
    pytester.runpytest("-n", "2", "--strict-markers", "-p", "no:cacheprovider")
    assert outcomes(pytester) == {
        "test_two_claims": (["DEF-A", "DEF-B"], "passed"),
        "test_fails": (["DEF-C"], "failed"),
        "test_skipped": (["DEF-D"], "skipped"),
        "test_setup_error": (["DEF-E"], "error"),
    }
