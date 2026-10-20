"""Lazy registry of the controlled-study arms.

`ARMS[name]` imports the arm's module, and with it its heavy dependencies (torch, shap,
causal-learn, xgboost), only on lookup. Every returned class implements
`oran_causal.core.data.Method`.
"""

import importlib
from collections.abc import Iterator, Mapping

_CONTROLLED = "oran_causal.methods.controlled"

PATHS: dict[str, tuple[str, str]] = {
    "dummy": (f"{_CONTROLLED}.dummy", "DummyMethod"),
    "corr": (f"{_CONTROLLED}.corr", "Corr"),
    "granger": (f"{_CONTROLLED}.granger", "Granger"),
    "pcorr": (f"{_CONTROLLED}.pcorr", "PCorrMethod"),
    "pcorr_hac": (f"{_CONTROLLED}.pcorr_hac", "PcorrHac"),
    "pc": (f"{_CONTROLLED}.pc", "PC"),
    "notears": (f"{_CONTROLLED}.notears", "Notears"),
    "shap_dag": (f"{_CONTROLLED}.shap_dag", "ShapDag"),
    "two_tower": (f"{_CONTROLLED}.two_tower", "TwoTowerM"),
    "cdl": (f"{_CONTROLLED}.cdl", "CDLMethod"),
    "rcot2": (f"{_CONTROLLED}.rcot2", "RCoT2Method"),
    "mscr": (f"{_CONTROLLED}.mscr", "MSCRMethod"),
    "pmrt_core": ("oran_causal.core.pmrt.rows", "PmrtCore"),
    "pmrt_nl": ("oran_causal.core.pmrt.nonlinear", "PmrtNl"),
}


class _Lazy(Mapping[str, type]):
    def __getitem__(self, name: str) -> type:
        module, cls = PATHS[name]
        return getattr(importlib.import_module(module), cls)

    def __iter__(self) -> Iterator[str]:
        return iter(PATHS)

    def __len__(self) -> int:
        return len(PATHS)


ARMS: Mapping[str, type] = _Lazy()


def resolve(ref: str) -> type:
    """The arm class of a registry name.

    Raises:
        KeyError: Unknown name.
    """
    if ref not in PATHS:
        raise KeyError(f"unknown arm {ref!r}")
    return ARMS[ref]
