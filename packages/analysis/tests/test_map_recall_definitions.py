"""Every map precision and recall value in `spec/values.toml` equals `core.scoring.confusion`.

The scoring definition (Section 5): precision is the share of declarations on determinate
hypotheses that are true, recall the share of true hypotheses that are declared, both whatever the
sign. Declarations are rebuilt from the stored Study 1 lists and Study 2 edge lists, so a key
expression that reads a sign-restricted count fails here.
"""

import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from oran_causal.analysis.paper_values import Records, load_spec, resolve
from oran_causal.core.scoring import confusion

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "spec" / "values.toml"
S1_CODES = {
    "pmrt": "loadsp_c+wby1s",
    "gby": "granger_by",
    "gr": "granger",
    "qacm": "qacm",
    "int": "int",
    "corr": "corr",
    "tt": "two_tower",
    "shap": "shap_gbdt",
}
S2_ARMS = {
    "pmrt": "MG:PMRT",
    "gby": "MG:granger_by",
    **{
        c: f"MG:{m}@dev"
        for c, m in (
            ("gr", "granger"),
            ("qacm", "qacm"),
            ("int", "int"),
            ("corr", "corr"),
            ("tt", "two_tower"),
            ("shap", "shap_gbdt"),
        )
    },
}
PLC_ARMS = [f"MG:{m}@plc" for m in ("corr", "granger", "shap_gbdt", "two_tower", "int", "qacm")]
KEY = re.compile(r"^s[12]\.map\.(?P<code>[a-z]+)\.(?P<kind>rec|prec)$")

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "records").exists(), reason="frozen record sets not present"
)


@pytest.fixture(scope="module")
def records() -> Records:
    return Records(ROOT)


@pytest.fixture(scope="module")
def aliases() -> Mapping[str, str]:
    return tomllib.loads(SPEC.read_text(encoding="utf-8"))["records"]


@pytest.fixture(scope="module")
def values(records: Records) -> dict[str, float]:
    return {s.key: resolve(s, records).value for s in load_spec(SPEC)}


def reference(cells: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(c["family"], c["relation"], c["kpi"]): c for c in cells}


def s1_score(records: Records, aliases: Mapping[str, str], code: str) -> dict[str, Any]:
    doc = records.document(aliases["v4a"])
    ref = reference(doc["gt"]["cells"])
    declared = doc["slices_result"]["pooled"]["metrics"][code]["declared"]
    decl = {h: {"declared": False, "sign": 0} for h in ref}
    for name, sign in declared:
        decl[tuple(name.split("|"))] = {"declared": True, "sign": sign}
    return confusion(decl, ref, list(ref))


def s2_score(records: Records, aliases: Mapping[str, str], arm: str) -> dict[str, Any]:
    doc = records.document(aliases["cd"])
    ref = reference(doc["gt"]["cells"])
    decl = {h: {"declared": False, "sign": 0} for h in ref}
    for status, edges in doc["edge_quality"][arm]["edges"].items():
        for edge in edges:
            h = tuple(edge)
            sign = ref[h]["sign"] if status == "true_right" else -ref[h].get("sign", 0)
            decl[h] = {"declared": True, "sign": sign}
    return confusion(decl, ref, list(ref))


def test_s1_keys(records: Records, aliases: Mapping[str, str], values: dict[str, float]) -> None:
    for code, method in S1_CODES.items():
        score = s1_score(records, aliases, method)
        assert values[f"s1.map.{code}.prec"] == pytest.approx(score["precision"]), code
        assert values[f"s1.map.{code}.rec"] == pytest.approx(score["recall"]), code
    pmrt = s1_score(records, aliases, S1_CODES["pmrt"])
    assert values["s1.pmrt.precision"] == pytest.approx(pmrt["precision"])
    assert values["s1.pmrt.recall"] == pytest.approx(pmrt["recall"])


def test_s2_keys(records: Records, aliases: Mapping[str, str], values: dict[str, float]) -> None:
    for code, arm in S2_ARMS.items():
        score = s2_score(records, aliases, arm)
        assert values[f"s2.map.{code}.prec"] == pytest.approx(score["precision"]), arm
        assert values[f"s2.map.{code}.rec"] == pytest.approx(score["recall"]), arm


def test_placebo_calibrated_recall(
    records: Records, aliases: Mapping[str, str], values: dict[str, float]
) -> None:
    best = max(s2_score(records, aliases, arm)["recall"] for arm in PLC_ARMS)
    assert values["s2.plc.rec_max"] == pytest.approx(best)


def test_every_precision_recall_key_is_covered(values: dict[str, float]) -> None:
    keys = {k for k in values if KEY.match(k)}
    covered = {f"s{n}.map.{c}.{k}" for n in (1, 2) for c in S1_CODES for k in ("rec", "prec")}
    assert keys == covered
