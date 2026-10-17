"""Tier t0, figures: rebuild the paper figures and tables into a scratch directory and compare.

The figures member is an optional extra of experiments, so it runs in a child interpreter.
Tables are compared by bytes (CRLF read as LF); figures by `pdftotext -layout` text and by an
80 dpi `pdftoppm` raster of both PDFs. Figures need Arial: on a host without it they are
skipped, and the tables are still compared.
"""

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from oran_causal.experiments.reproduce.checks import Check, Status, compare_bytes
from oran_causal.experiments.reproduce.context import Context

ITEM = "figures"
OUT_ENV = "ORAN_CAUSAL_FIGURES_OUT"
RASTER_DPI = 80
STUDY4_WORLDS = ("E1", "E2", "E3", "E4", "E5")
TABLES_SCRIPT = """
import importlib, pkgutil, sys
from oran_causal.figures import figs
for info in pkgutil.iter_modules(figs.__path__):
    if info.name.startswith("tab_"):
        mod = importlib.import_module(figs.__name__ + "." + info.name)
        if info.name == "tab_study4_env":
            for world in sys.argv[1:]:
                mod.write(world)
        else:
            mod.write()
"""
FIGURES_SCRIPT = (
    "import sys; from oran_causal.figures.cli import main; "
    "raise SystemExit(main(['build', '--out', sys.argv[1]]))"
)
ARIAL_SCRIPT = "from oran_causal.figures import style; style.arial_files()"
INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+\.pdf)\}")


def _python(script: str, *args: str, out: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg"}
    if out is not None:
        env[OUT_ENV] = str(out)
    cmd = [sys.executable, "-c", script, *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)


def _last_line(p: subprocess.CompletedProcess[str]) -> str:
    lines = (p.stderr or p.stdout).strip().splitlines()
    return lines[-1] if lines else f"exit {p.returncode}"


def tables(ctx: Context, out: Path) -> list[Check]:
    """Every `tab_*` module's files against `paper/tables`."""
    p = _python(TABLES_SCRIPT, *STUDY4_WORLDS, out=out)
    if p.returncode != 0:
        return [Check(ITEM, "tables", Status.ERROR, _last_line(p))]
    stored = sorted((ctx.paper / "tables").iterdir())
    checks = [compare_bytes(ITEM, out / "tables" / f.name, f, eol=True) for f in stored]
    extra = sorted({f.name for f in (out / "tables").iterdir()} - {f.name for f in stored})
    if extra:
        checks.append(Check(ITEM, "tables", Status.DIFF, f"not in paper/tables: {extra}"))
    return checks


def _pdftotext(pdf: Path) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout


def _raster(pdf: Path, prefix: Path) -> list[bytes]:
    subprocess.run(["pdftoppm", "-r", str(RASTER_DPI), "-png", str(pdf), str(prefix)], check=True)
    return [p.read_bytes() for p in sorted(prefix.parent.glob(f"{prefix.name}*.png"))]


def compare_pdf(mine: Path, stored: Path, work: Path) -> Check:
    """Text and raster comparison of two PDFs."""
    if not mine.exists():
        return Check(ITEM, stored.name, Status.ERROR, "not regenerated")
    if _pdftotext(mine) != _pdftotext(stored):
        return Check(ITEM, stored.name, Status.DIFF, "pdftotext differs")
    work.mkdir(parents=True, exist_ok=True)
    a, b = _raster(mine, work / "mine"), _raster(stored, work / "stored")
    if a != b:
        return Check(ITEM, stored.name, Status.DIFF, f"{RASTER_DPI} dpi raster differs")
    return Check(ITEM, stored.name, Status.PASS, f"text and {RASTER_DPI} dpi raster equal")


def missing_tools(tools: Sequence[str] = ("pdftotext", "pdftoppm")) -> list[str]:
    """The PDF tools not on PATH."""
    return [t for t in tools if shutil.which(t) is None]


def included(paper: Path) -> set[str]:
    """File names of the figures that the manuscript's `.tex` sources include."""
    return {m for tex in paper.rglob("*.tex") for m in INCLUDE.findall(tex.read_text("utf-8"))}


def compare_stored(ctx: Context, out: Path) -> list[Check]:
    """Every PDF in `paper/figures`; a stored figure the paper does not include may be absent."""
    used = included(ctx.paper)
    checks = []
    for f in sorted((ctx.paper / "figures").glob("*.pdf")):
        mine = out / "figures" / f.name
        if f.name not in used and not mine.exists():
            checks.append(Check(ITEM, f.name, Status.SKIP, "not built and not included by paper/"))
        else:
            checks.append(compare_pdf(mine, f, out / "raster" / f.stem))
    return checks


def pdfs(ctx: Context, out: Path) -> list[Check]:
    """The paper figures against `paper/figures` (skipped without Arial or the PDF tools)."""
    absent = missing_tools()
    if absent:
        return [Check(ITEM, "figures", Status.SKIP, f"not on PATH: {', '.join(absent)}")]
    arial = _python(ARIAL_SCRIPT)
    if arial.returncode != 0:
        return [Check(ITEM, "figures", Status.SKIP, f"no Arial on this host ({_last_line(arial)})")]
    p = _python(FIGURES_SCRIPT, str(out))
    if p.returncode != 0:
        return [Check(ITEM, "figures", Status.ERROR, _last_line(p))]
    return compare_stored(ctx, out)


def figures(ctx: Context) -> list[Check]:
    """Paper tables and figures rebuilt by the figures member and compared with `paper/`."""
    if _python("import oran_causal.figures").returncode != 0:
        return [Check(ITEM, "figures", Status.SKIP, "figures member not installed")]
    out = ctx.item_dir(ITEM)
    return [*tables(ctx, out), *pdfs(ctx, out)]
