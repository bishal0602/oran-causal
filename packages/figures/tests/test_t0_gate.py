import importlib
import json
import pkgutil
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from oran_causal.figures import figs, paths, plotted, registry, style

GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "plotted_golden.json").read_text(encoding="utf-8")
)
STUDY4_WORLDS = ("E1", "E2", "E3", "E4", "E5")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def built(records):
    style.apply()
    out = {}
    for name, mod in registry.discover().items():
        fig = mod.build()
        fig.draw_without_rendering()
        out[name] = {
            "digest": plotted.digest(fig),
            "rows": len(plotted.rows(fig)),
            "size_in": [round(v, 4) for v in fig.get_size_inches()],
            **plotted.pdf_layout(fig),
        }
        plt.close(fig)
    return out


@pytest.fixture(scope="module")
def tables(records, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    mp.setenv(paths.OUT_ENV, str(out))
    try:
        for info in pkgutil.iter_modules(figs.__path__):
            if not info.name.startswith("tab_"):
                continue
            mod = importlib.import_module(f"{figs.__name__}.{info.name}")
            if info.name == "tab_study4_env":
                for world in STUDY4_WORLDS:
                    mod.write(world)
            else:
                mod.write()
    finally:
        mp.undo()
    return {
        p.name: plotted.table_digest(p.read_text(encoding="utf-8"))
        for p in (out / "tables").iterdir()
    }


def test_the_same_figures_are_registered(built):
    assert sorted(built) == sorted(GOLDEN["figures"])


@pytest.mark.parametrize("name", sorted(GOLDEN["figures"]))
def test_figure_data_equals_the_cddfig_build(built, name):
    assert built[name] == GOLDEN["figures"][name]


PAPER_TABLES = sorted(p.name for p in (paths.PAPER / "tables").iterdir())


def test_the_paper_tables_are_written(tables):
    assert sorted(tables) == PAPER_TABLES


@pytest.mark.parametrize("name", PAPER_TABLES)
def test_table_text_equals_the_paper(tables, name):
    paper = (paths.PAPER / "tables" / name).read_text(encoding="utf-8")
    assert tables[name] == plotted.table_digest(paper)
