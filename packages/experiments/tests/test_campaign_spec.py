import hashlib

import pytest

from oran_causal.experiments.campaign.spec import (
    ProtocolTexts,
    eval_authorised,
    expand,
    is_eval,
    sha_lf,
    spec_sha256,
)
from oran_causal.experiments.paths import STUDY4_PROTOCOL


def test_expand_counts_restrictions_and_lams(kit, shipped):
    units = expand(kit.make_spec(), shipped)
    assert len(units) == 6 * 2 * 7 * 2
    assert len({u.key for u in units}) == len(units)
    assert {u.lam for u in units if u.world == "E4" and u.regime == "R3"} == {0.0, 0.5, 1.0, 1.5}
    assert {u.lam for u in units if u.world == "E4" and u.regime == "R1"} == {1.0}
    assert {u.lam for u in units if u.world == "E1"} == {None}
    assert not [u for u in units if u.arm == "d_e3"]
    s = kit.make_spec()
    s["blocks"][1]["worlds"] = ["E3"]
    e3 = [u for u in expand(s, shipped) if u.arm == "d_e3"]
    assert e3
    assert {u.n for u in e3} == {300}
    assert {u.role for u in units if u.seed < 3000100} == {"tune"}


def test_unit_key_cell_and_dataset(kit, shipped):
    u = next(u for u in expand(kit.small_spec(), shipped) if u.n == 400)
    assert u.key == f"{u.arm}|E4|R1|lam1|k0.25|n400|s{u.seed}"
    assert u.cell == f"{u.arm}|E4|R1|lam1|k0.25|n400"
    assert u.dataset == ("E4", "R1", 1.0, 400, u.seed, 0.25)


@pytest.mark.claim("PRO-SEEDS")
def test_expand_dedups_across_blocks_and_validates(kit, shipped):
    s = kit.make_spec()
    s["blocks"].append(dict(s["blocks"][1]))
    assert len(expand(s, shipped)) == len(expand(kit.make_spec(), shipped))
    bad = kit.make_spec()
    bad["blocks"][1]["seeds"] = [3000002, 3000003]
    with pytest.raises(ValueError, match="overlap"):
        expand(bad, shipped)
    ev = kit.make_spec()
    ev["blocks"][1]["seeds"] = [3100000]
    with pytest.raises(ValueError, match=r"EVAL seeds .* refused"):
        expand(ev, shipped)
    nk = kit.make_spec()
    del nk["blocks"][0]["kappas"]
    with pytest.raises(ValueError, match="kappas"):
        expand(nk, shipped)
    nd = kit.make_spec()
    nd["arms"]["d_by"].pop("declare")
    with pytest.raises(ValueError, match="declare"):
        expand(nd, shipped)


@pytest.mark.claim("PRO-FROZEN")
def test_eval_guard_accepts_frozen_matching_protocol(kit):
    texts, sha = kit.frozen_texts(crlf=True)
    assert sha_lf(texts.live or b"") == sha
    units = expand(kit.eval_spec(sha), texts)
    assert {u.seed for u in units if u.role == "measure"} == {3100000, 3100001}
    assert eval_authorised(kit.eval_spec(sha.upper()), texts)[0]


@pytest.mark.claim("PRO-FROZEN")
@pytest.mark.parametrize("case", ["no_sha", "wrong_sha", "not_frozen", "missing_file"])
def test_eval_guard_refuses(kit, case):
    text = "# P\n\nFROZEN: no (DRAFT)\n" if case == "not_frozen" else "# P\n\nFROZEN: yes\n"
    texts, sha = kit.frozen_texts(text)
    if case == "missing_file":
        texts = ProtocolTexts(live=None)
    spec = kit.eval_spec(None if case == "no_sha" else "0" * 64 if case == "wrong_sha" else sha)
    with pytest.raises(ValueError, match=r"EVAL seeds .* refused"):
        expand(spec, texts)


@pytest.mark.claim("PRO-FROZEN")
def test_guard_verified_by_commit_blob_or_frozen_copy(kit):
    frozen, sha = kit.frozen_texts()
    amended = frozen.live + b"\nAMENDMENT A-1\n" if frozen.live else b""
    spec = {**kit.eval_spec(sha), "freeze_commit": "93856c2"}
    by_blob = ProtocolTexts(
        live=amended, at_commit=lambda c: frozen.live if c == "93856c2" else None
    )
    ok, why = eval_authorised(spec, by_blob)
    assert ok
    assert "git blob at 93856c2" in why
    ok, why = eval_authorised(
        kit.eval_spec(sha), ProtocolTexts(live=amended, frozen_copy=frozen.live)
    )
    assert ok
    assert "bundled frozen copy" in why
    assert not eval_authorised(kit.eval_spec(sha), ProtocolTexts(live=amended))[0]


@pytest.mark.claim("PRO-FROZEN")
def test_shipped_protocol_is_the_frozen_one_and_dev_is_unguarded(kit, shipped):
    sha = hashlib.sha256(STUDY4_PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert sha == "c5f7a4fe4dff4bfdeef3ccebac2dcc3748dfa9c2d0c3d17f2ac9d6246379fa22"
    assert eval_authorised(kit.eval_spec(sha), shipped)[0]
    assert not eval_authorised(kit.eval_spec("0" * 64), shipped)[0]
    with_sha = {**kit.small_spec(), "protocol_sha256": "0" * 64}
    assert expand(kit.small_spec(), shipped) == expand(with_sha, shipped)
    bad = kit.small_spec()
    bad["blocks"][1]["seeds"] = [3000200]
    with pytest.raises(ValueError, match="DEV block"):
        expand(bad, shipped)


@pytest.mark.claim("PRO-SEEDS")
def test_eval_seed_block_rules(kit):
    texts, sha = kit.frozen_texts()
    spec = kit.eval_spec(sha)
    assert is_eval(spec)
    tune_eval = kit.eval_spec(sha)
    tune_eval["blocks"][0]["seeds"] = [3100100, 3100101]
    with pytest.raises(ValueError, match="tune blocks must use DEV seeds"):
        expand(tune_eval, texts)
    measure_dev = kit.eval_spec(sha)
    measure_dev["blocks"][1]["seeds"] = [3000150, 3100000, 3100001]
    with pytest.raises(ValueError, match="measure blocks must use EVAL seeds"):
        expand(measure_dev, texts)


def test_spec_sha_ignores_formatting():
    assert spec_sha256({"a": 1, "b": [1, 2]}) == spec_sha256({"b": [1, 2], "a": 1})
