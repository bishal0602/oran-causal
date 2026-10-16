import functools
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.sim.controlled.datasets.generate import dataset_hash, generate_dataset

FIXTURE = Path(__file__).parent / "data" / "study4_eval_datasets.json"
LIBM_REFERENCE = "bd958658495fcea8b26353911fb9be74191dcea11c2d72d50dd3a559dacb39fe"
MOMENT_RTOL = 1e-9

pytestmark = pytest.mark.claim("DEF-ENV")


@functools.cache
def libm_signature() -> str:
    digest = hashlib.sha256()
    for k in range(40000):
        digest.update(struct.pack("<dd", math.exp(-k * 0.0049), math.erfc(-6.0 + k * 0.0003)))
    return digest.hexdigest()


def uses_transcendentals(row: dict[str, Any]) -> bool:
    return row["world"] in ("E2", "E5") or (row["world"] == "E4" and row["regime"] == "R3")


def load_rows() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())


def smallest_n_per_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (row["world"], row["regime"], row["lam"], row["kappa"])
        if key not in groups or row["n"] < groups[key]["n"]:
            groups[key] = row
    return list(groups.values())


def row_id(row: dict[str, Any]) -> str:
    return f"{row['world']}-{row['regime']}-lam{row['lam']}-k{row['kappa']}-n{row['n']}"


def check_row(row: dict[str, Any]) -> None:
    ds, _ = generate_dataset(
        row["world"],
        row["regime"],
        row["n"],
        row["seed"],
        lam=1.0 if row["lam"] is None else row["lam"],
        kappa=row["kappa"],
    )
    np.testing.assert_allclose(ds.Y.mean(axis=0), row["y_mean"], rtol=MOMENT_RTOL, atol=1e-12)
    np.testing.assert_allclose(ds.Y.std(axis=0), row["y_std"], rtol=MOMENT_RTOL, atol=1e-12)
    if uses_transcendentals(row) and libm_signature() != LIBM_REFERENCE:
        return
    assert dataset_hash(ds) == row["dataset_sha256"]


ROWS = load_rows()
SMALL = smallest_n_per_group(ROWS)


def test_fixture_covers_every_environment_regime_lambda_and_size() -> None:
    combos = {(r["world"], r["regime"], r["lam"], r["kappa"], r["n"]) for r in ROWS}
    assert len(combos) == len(ROWS) == 100
    assert {r["world"] for r in ROWS} == {"E1", "E2", "E3", "E4", "E5"}
    assert {r["n"] for r in ROWS} == {500, 1000, 4000, 8000, 24000}
    assert {r["lam"] for r in ROWS if r["world"] == "E4"} == {0.0, 0.5, 1.0, 1.5}


@pytest.mark.parametrize("row", SMALL, ids=row_id)
def test_regeneration_matches_frozen_study4_records(row: dict[str, Any]) -> None:
    check_row(row)


@pytest.mark.slow
@pytest.mark.parametrize("row", ROWS, ids=row_id)
def test_full_regeneration_matches_frozen_study4_records(row: dict[str, Any]) -> None:
    check_row(row)
