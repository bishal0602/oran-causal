import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from oran_causal.experiments import manifests
from oran_causal.experiments.campaign.spec import shipped_protocol
from oran_causal.experiments.manifests import Manifest, ManifestError
from oran_causal.experiments.paths import MANIFESTS
from oran_causal.experiments.reproduce import plan

EXPECTED_IDS = [
    "bridging",
    "study1",
    "study2",
    "study3",
    "study4",
    "x1",
    "x2_x3",
    "x4",
    "x5_x6",
    "x7",
    "x8",
    "x9_x16",
]
UNPRICED_IN_TIERS = {"x8": (2.41, 2.41)}
CLOSE = 0.011


@pytest.fixture(scope="module")
def loaded() -> dict[str, Manifest]:
    return manifests.load_all()


@pytest.fixture(scope="module")
def tiers() -> dict[str, Any]:
    return plan.load_manifest()


def raw(item: str) -> dict[str, Any]:
    return yaml.safe_load((MANIFESTS / f"{item}.yaml").read_text(encoding="utf-8"))


def test_every_item_has_a_manifest() -> None:
    assert manifests.ids() == EXPECTED_IDS


@pytest.mark.parametrize("item", EXPECTED_IDS)
def test_manifest_passes_every_check(item: str, loaded: dict[str, Manifest]) -> None:
    assert loaded[item].id == item
    assert manifests.check(loaded[item]) == []


def test_every_empirical_and_protocol_claim_has_a_manifest(loaded: dict[str, Manifest]) -> None:
    doc = yaml.safe_load(manifests.CLAIMS_FILE.read_text(encoding="utf-8"))
    wanted = {c["id"] for c in doc["claims"] if c["kind"] in ("empirical", "protocol")}
    assert wanted <= {c for m in loaded.values() for c in m.claims}


def test_manifest_items_follow_tiers_yaml(
    loaded: dict[str, Manifest], tiers: dict[str, Any]
) -> None:
    priced = {it["name"]: it for it in tiers["items"]}
    assert set(priced) <= set(loaded)
    for name, it in priced.items():
        m = loaded[name]
        assert m.platform == it["platform"]
        assert {t for t in manifests.COST_TIERS if m.tiers[t] is not None} == set(it["tiers"])
    for name in set(loaded) - set(priced):
        assert (
            all(loaded[name].tiers[t] is None for t in manifests.COST_TIERS)
            or name in UNPRICED_IN_TIERS
        )


def test_stage_costs_equal_tiers_yaml(loaded: dict[str, Manifest], tiers: dict[str, Any]) -> None:
    for it in tiers["items"]:
        if "stages" not in it:
            continue
        by_name = {s.name: s for s in loaded[it["name"]].stages}
        priced = [s["name"] for s in it["stages"]]
        assert [n for n in by_name if n in priced] == priced
        assert all(by_name[n].cost.high == 0 for n in by_name if n not in priced)
        for st in it["stages"]:
            got = by_name[st["name"]]
            low, high = plan._range(st["ref_cpu_h"])
            assert (got.cost.low, got.cost.high) == (low, high)
            assert got.cost.basis == st["basis"]
            assert list(got.frozen_in) == st.get("frozen_in", [])


@pytest.mark.parametrize("tier", ["t2", "t3"])
def test_tier_totals_equal_the_dry_run_plan(tier: str, loaded: dict[str, Manifest]) -> None:
    priced = plan.totals(plan.plan(tier, texts=shipped_protocol()))["all"]
    extra = sum(UNPRICED_IN_TIERS[i][0] for i in UNPRICED_IN_TIERS if tier == "t3")
    got = manifests.tier_cost(loaded.values(), tier)
    assert got[0] == pytest.approx(priced[0] + extra, abs=CLOSE)
    assert got[1] == pytest.approx(priced[1] + extra, abs=CLOSE)


def test_study4_splits_the_n24000_cells(loaded: dict[str, Manifest]) -> None:
    s4 = loaded["study4"]
    assert [s.name for s in s4.stages] == ["campaign", "campaign_n24000"]
    assert s4.stages[1].frozen_in == ("t2",)
    assert s4.stage_cost("t2")[0] == pytest.approx(296.53, abs=CLOSE)
    assert s4.stage_cost("t3")[0] == pytest.approx(503.64, abs=CLOSE)


def test_seed_ranges_match_the_registry(loaded: dict[str, Manifest]) -> None:
    registry = json.loads(manifests.SEEDS_FILE.read_text(encoding="utf-8"))
    for m in loaded.values():
        for s in m.seeds:
            assert manifests._inside(s.range, manifests.registered_ranges(registry[s.world]))
    claimed = registry["E4"]["claimed_by"]
    s4 = {(s.block, s.world): s.range for s in loaded["study4"].seeds}
    assert list(s4[("XMETHOD_EVAL", "E4")]) == claimed["XMETHOD_EVAL"][0]
    assert list(s4[("XMETHOD_DEV", "E4")]) == claimed["XMETHOD_DEV"][0]


def test_protocol_hashes_equal_protocol_sums(loaded: dict[str, Manifest]) -> None:
    sums = (manifests.PROTOCOLS / "SHA256SUMS").read_text(encoding="utf-8").split("\n")
    pinned = dict(line.split("  ", 1)[::-1] for line in sums if line)
    for m in loaded.values():
        if m.protocol is not None:
            assert pinned[m.protocol.path] == m.protocol.sha256


def test_unknown_item_is_an_error() -> None:
    with pytest.raises(ManifestError, match="no manifest"):
        manifests.load("nope")


def test_id_must_equal_file_name(tmp_path: Path) -> None:
    doc = raw("x4") | {"id": "x5"}
    (tmp_path / "x4.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ManifestError, match="differs from the file name"):
        manifests.load("x4", tmp_path)


def test_ids_skip_the_tier_plan(tmp_path: Path) -> None:
    for name in ("a.yaml", "tiers.yaml", "b.yaml"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert manifests.ids(tmp_path) == ["a", "b"]


def broken(mutate: Any) -> dict[str, Any]:
    doc = copy.deepcopy(raw("study3"))
    mutate(doc)
    return doc


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.pop("seeds"), "missing seeds"),
        (lambda d: d.update(extra=1), "unknown extra"),
        (lambda d: d["protocol"].update(sha256="abc"), "SHA-256"),
        (lambda d: d["stages"][0]["cost"].update(basis="Z"), "basis"),
        (lambda d: d["stages"][0]["cost"].update(cpu_h=-1), "non-negative"),
        (lambda d: d["stages"][0]["cost"].update(cpu_h=[3, 1]), "reversed"),
        (lambda d: d["stages"][0]["cost"].update(cpu_h=[1]), "low, high"),
        (lambda d: d["stages"][0].update(frozen_in=["t9"]), "unknown tier"),
        (lambda d: d["stages"][0].update(args=[1]), "args"),
        (lambda d: d["seeds"][0].update(range=[5, 1]), "range"),
        (lambda d: d["seeds"][0].update(range=[1.5, 2]), "range"),
        (lambda d: d["tiers"].pop("t3"), "missing t3"),
        (lambda d: d.update(stages=[]), "no stages"),
        (lambda d: d.update(records=[]), "no records"),
        (lambda d: d["records"][0].update(manifest=3), "string"),
        (lambda d: d.update(claims="EMP"), "list"),
        (lambda d: d["equivalence"][0].pop("rule"), "missing rule"),
    ],
)
def test_schema_violations_are_rejected(mutate: Any, message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        manifests.parse(broken(mutate))


def test_costs_accept_a_number_or_a_range() -> None:
    doc = broken(lambda d: d["stages"][0]["cost"].update(cpu_h=[2, 5]))
    cost = manifests.parse(doc).stages[0].cost
    assert (cost.low, cost.high) == (2.0, 5.0)


def test_tier_cost_skips_frozen_stages_and_absent_tiers(loaded: dict[str, Manifest]) -> None:
    s1 = loaded["study1"]
    assert s1.stage_cost("t2")[0] == pytest.approx(22.0)
    assert s1.stage_cost("t3")[0] == pytest.approx(64.0)
    assert loaded["x1"].stage_cost("t3") == (0.0, 0.0)
    with pytest.raises(ValueError, match="costs exist"):
        manifests.tier_cost([s1], "t1")


def problems_of(mutate: Any, **kw: Any) -> list[str]:
    return manifests.check(manifests.parse(broken(mutate)), **kw)


def test_check_flags_unregistered_claim() -> None:
    found = problems_of(lambda d: d["claims"].append("EMP-NOPE"))
    assert found == ["study3: claim EMP-NOPE is not registered"]


def test_check_flags_seed_outside_the_registry() -> None:
    found = problems_of(lambda d: d["seeds"][0].update(range=[191000, 192000]))
    assert len(found) == 1
    assert "not registered for E6" in found[0]


def test_check_flags_unknown_world() -> None:
    found = problems_of(lambda d: d["seeds"][0].update(world="E99"))
    assert "world 'E99' not in seeds.json" in found[0]


def test_check_flags_a_wrong_hash() -> None:
    found = problems_of(lambda d: d["frozen_inputs"][0].update(sha256="0" * 64))
    assert len(found) == 1
    assert "manifest says " + "0" * 64 in found[0]


def test_check_flags_a_wrong_protocol_hash() -> None:
    found = problems_of(lambda d: d["protocol"].update(sha256="1" * 64))
    assert "protocol study3/PROTOCOL.md" in found[0]


def test_check_flags_missing_files() -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["frozen_inputs"][0]["path"] = "artifacts/missing.json"
        d["protocol"]["path"] = "study3/missing.md"
        d["records"][0]["manifest"] = "manifests/missing.sha256"

    found = problems_of(mutate)
    assert any("frozen input artifacts/missing.json is missing" in p for p in found)
    assert any("protocol study3/missing.md is missing" in p for p in found)
    assert any("record manifest manifests/missing.sha256 is missing" in p for p in found)


def test_check_flags_record_set_mismatches() -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["records"][0]["set"] = "no_such_set"
        d["records"][1]["manifest"] = d["records"][0]["manifest"]

    found = problems_of(mutate)
    assert any("no_such_set is not in record_sets.yaml" in p for p in found)
    want = "system_artifacts lives in manifests/system_artifacts.sha256, not manifests/study2_3"
    assert any(want in p for p in found)


def test_check_flags_bad_drivers() -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["stages"][0]["driver"] = "oran_causal.experiments.nothing:run"
        d["stages"][1]["driver"] = "no_colon"

    found = problems_of(mutate)
    assert any("stage eval" in p and "nothing" in p for p in found)
    assert any("expected module:callable" in p for p in found)


def test_check_flags_frozen_in_a_tier_the_item_lacks() -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["tiers"]["t2"] = None
        d["stages"][0]["frozen_in"] = ["t2"]

    found = problems_of(mutate)
    assert found == ["study3: stage eval is frozen in t2 but the item has no t2"]


def test_resolve_driver_returns_the_callable() -> None:
    assert manifests.resolve_driver("oran_causal.experiments.manifests:load") is manifests.load
    with pytest.raises(ManifestError, match="not callable"):
        manifests.resolve_driver("oran_causal.experiments.manifests:TIERS")


def test_lf_sha256_ignores_line_endings(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"x\ny\n")
    (tmp_path / "b").write_bytes(b"x\r\ny\r\n")
    assert manifests.lf_sha256(tmp_path / "a") == manifests.lf_sha256(tmp_path / "b")
