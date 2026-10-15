"""Validation against the JSON Schema subset used by `spec/claims.schema.yaml`."""

import re
from collections.abc import Mapping
from typing import Any

SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "type",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "pattern",
        "minimum",
        "default",
    }
)


class SchemaError(ValueError):
    """The schema uses a keyword this validator does not implement."""


def _is_type(value: Any, name: str) -> bool:
    match name:
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case "string":
            return isinstance(value, str)
        case "boolean":
            return isinstance(value, bool)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
    raise SchemaError(f"unsupported type {name!r}")


def _check_keywords(schema: Mapping[str, Any], path: str) -> None:
    unknown = set(schema) - SUPPORTED_KEYWORDS
    if unknown:
        raise SchemaError(f"{path or '/'}: unsupported schema keywords {sorted(unknown)}")


def _object_errors(value: dict, schema: Mapping[str, Any], path: str) -> list[str]:
    errors = [
        f"{path}: missing required field {key!r}"
        for key in schema.get("required", ())
        if key not in value
    ]
    properties = schema.get("properties", {})
    for key, item in value.items():
        if key in properties:
            errors += validate(item, properties[key], f"{path}/{key}")
        elif schema.get("additionalProperties", True) is False:
            errors.append(f"{path}: unknown field {key!r}")
    return errors


def _scalar_errors(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}, got {value!r}")
    if "pattern" in schema and isinstance(value, str) and not re.search(schema["pattern"], value):
        errors.append(f"{path}: {value!r} does not match {schema['pattern']!r}")
    if "minimum" in schema and _is_type(value, "number") and value < schema["minimum"]:
        errors.append(f"{path}: {value!r} is below the minimum {schema['minimum']!r}")
    return errors


def validate(value: Any, schema: Mapping[str, Any], path: str = "") -> list[str]:
    """Validate a parsed YAML value against a schema.

    Args:
        value: The parsed document or a part of it.
        schema: A JSON Schema restricted to `SUPPORTED_KEYWORDS`.
        path: JSON-pointer-like location used in messages.

    Returns:
        One message per violation; empty when the value is valid.

    Raises:
        SchemaError: The schema uses a keyword outside `SUPPORTED_KEYWORDS`.
    """
    _check_keywords(schema, path)
    expected = schema.get("type")
    if expected is not None and not _is_type(value, expected):
        return [f"{path or '/'}: expected {expected}, got {type(value).__name__}"]
    errors = _scalar_errors(value, schema, path or "/")
    if isinstance(value, dict):
        errors += _object_errors(value, schema, path)
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}/{index}")
    return errors
