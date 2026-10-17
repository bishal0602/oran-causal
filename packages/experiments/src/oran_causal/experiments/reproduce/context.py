"""Locations shared by the t0 items."""

import dataclasses
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from oran_causal.experiments.paths import WORKSPACE_ROOT
from oran_causal.experiments.reproduce.checks import Check, Status

type Log = Callable[[str], None]


@dataclasses.dataclass(frozen=True)
class Context:
    """Locations of one t0 run.

    Attributes:
        records: The frozen record sets (`data/records`).
        out: Scratch directory of the regenerated files.
        paper: The manuscript directory with `figures/` and `tables/`.
        log: Progress sink.
    """

    records: Path
    out: Path
    paper: Path = WORKSPACE_ROOT / "paper"
    log: Log = print

    def item_dir(self, item: str) -> Path:
        """`<out>/<item>`, created."""
        d = self.out / item
        d.mkdir(parents=True, exist_ok=True)
        return d


def missing(item: str, ctx: Context, *rels: str) -> list[Check]:
    """A skip row per record path under `ctx.records` that is absent."""
    return [
        Check(item, rel, Status.SKIP, "record set not present")
        for rel in rels
        if not (ctx.records / rel).exists()
    ]


type Item = Callable[[Context], list[Check]]


def guarded(name: str, item: Item, ctx: Context) -> list[Check]:
    """Runs one item; an exception becomes an error row; the wall time goes to the log."""
    t0 = time.time()
    try:
        checks = item(ctx)
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        checks = [Check(name, "run", Status.ERROR, f"{type(e).__name__}: {tb}")]
    ctx.log(f"[{name}] {len(checks)} checks in {time.time() - t0:.1f} s")
    return checks
