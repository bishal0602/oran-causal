"""Import check of the code symbols a claim names (`package.module:Attr.sub`)."""

import importlib
from collections.abc import Callable
from typing import Any

type Importer = Callable[[str], Any]


def resolve(reference: str, importer: Importer = importlib.import_module) -> Any:
    """The object a `module:attribute` reference names.

    Raises:
        ValueError: The reference is malformed, the module does not import, or an attribute is
            missing.
    """
    module_name, colon, attribute = reference.partition(":")
    if not (module_name and colon and attribute):
        raise ValueError("not of the form module:attribute")
    try:
        target = importer(module_name)
    except Exception as error:
        raise ValueError(
            f"module {module_name} does not import ({type(error).__name__}: {error})"
        ) from None
    for name in attribute.split("."):
        try:
            target = getattr(target, name)
        except AttributeError:
            raise ValueError(f"{module_name} has no attribute {attribute}") from None
    return target
