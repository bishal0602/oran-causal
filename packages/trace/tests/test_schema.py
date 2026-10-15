import pytest

from oran_causal.trace.schema import SchemaError, validate

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id"],
    "properties": {
        "id": {"type": "string", "pattern": "^A-[0-9]+$"},
        "kind": {"enum": ["x", "y"]},
        "n": {"type": "integer", "minimum": 0},
        "tags": {"type": "array", "items": {"type": "string"}},
        "version": {"const": 1},
        "any": {},
    },
}


def test_valid_document_has_no_errors() -> None:
    value = {"id": "A-1", "kind": "x", "n": 0, "tags": ["t"], "version": 1, "any": [1, {}]}
    assert validate(value, SCHEMA) == []


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ({}, "missing required field 'id'"),
        ({"id": "B-1"}, "does not match"),
        ({"id": "A-1", "kind": "z"}, "is not one of"),
        ({"id": "A-1", "n": -1}, "below the minimum"),
        ({"id": "A-1", "n": True}, "expected integer"),
        ({"id": "A-1", "tags": [1]}, "/tags/0: expected string"),
        ({"id": "A-1", "version": 2}, "expected 1"),
        ({"id": "A-1", "extra": 1}, "unknown field 'extra'"),
        ([], "expected object"),
    ],
)
def test_violations_are_reported(value: object, fragment: str) -> None:
    errors = validate(value, SCHEMA)
    assert any(fragment in error for error in errors), errors


def test_unsupported_keyword_is_rejected() -> None:
    with pytest.raises(SchemaError, match="oneOf"):
        validate(1, {"oneOf": []})
