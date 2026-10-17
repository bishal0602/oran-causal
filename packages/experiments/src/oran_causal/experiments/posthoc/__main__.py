"""Command line of the post hoc analyses and drivers.

    python -m oran_causal.experiments.posthoc <item> ...

Re-analyses of the frozen records: `x1-runtime`, `x2x3`, `x4`, `x5x6`, `x7`, `x8`, `x9`, `x10`,
`x11`, `x12`, `x13`, `x14`, `x15`, `x16`. Drivers: `x1-timing`, `x4-run`, `x7-run`, `x8-run`.
"""

import sys
from collections.abc import Callable, Sequence

from oran_causal.analysis.posthoc import (
    x1_runtime,
    x2x3,
    x4,
    x5x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
)
from oran_causal.experiments.posthoc import tools, x1_timing
from oran_causal.experiments.posthoc import x4 as x4_run
from oran_causal.experiments.posthoc import x7 as x7_run
from oran_causal.experiments.posthoc import x8 as x8_run

Main = Callable[[Sequence[str]], int]


def _with_tools(main: Callable[..., int]) -> Main:
    def call(argv: Sequence[str]) -> int:
        return main(argv, tools=tools.referee_tools())

    return call


ITEMS: dict[str, Main] = {
    "x1-runtime": x1_runtime.main,
    "x2x3": x2x3.main,
    "x4": x4.main,
    "x5x6": x5x6.main,
    "x7": x7.main,
    "x8": x8.main,
    "x9": x9.main,
    "x10": _with_tools(x10.main),
    "x11": x11.main,
    "x12": _with_tools(x12.main),
    "x13": x13.main,
    "x14": x14.main,
    "x15": x15.main,
    "x16": _with_tools(x16.main),
    "x1-timing": x1_timing.main,
    "x4-run": x4_run.main,
    "x7-run": x7_run.main,
    "x8-run": x8_run.main,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatches to the item named by the first argument."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ITEMS:
        raise SystemExit(__doc__)
    return ITEMS[args[0]](args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
