from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from oran_causal.figures import registry, style
from oran_causal.figures.paths import PAPER, RECORDS

PLANNED = {
    "fig_study1_discovery": [
        "loadsp_c+wby1s",
        "plain_c+by",
        "max+dagger1s",
        "corr",
        "granger",
        "granger_by",
        "shap_gbdt",
        "int",
        "qacm",
        "two_tower",
    ],
    "fig_study2_maps": [
        "MG:PMRT",
        "MG:GT",
        "MG:rand",
        "MG:corr@dev",
        "MG:corr@plc",
        "MG:granger@dev",
        "MG:granger_by",
        "MG:shap_gbdt@dev",
        "MG:two_tower@plc",
        "MG:int@dev",
        "MG:qacm@plc",
    ],
    "fig_decision_referee": [
        "noarb",
        "never_sleep",
        "sub:SliceGuarantee",
        "sub:ES+PowerES",
        "sub:ES",
        "sub:PowerES",
        "freeze",
        "incumbent",
        "MG:PMRT",
        "MG:GT",
        "MG:rand",
        "MG:granger@dev",
        "CS:PMRT",
        "CS:GT",
        "CS:rand",
        "CS:corr@dev",
        "CS:granger_by",
        "CS:shap_gbdt@dev",
    ],
    "fig_decision_referee_v3": ["MG:PMRT", "MG:GT", "never_sleep", "CS:PMRT", "CS:GT"],
    "fig_study2_deferrals": ["MG:PMRT", "MG:GT"],
    "fig_study3_certification": ["CS:PMRT", "CS:GT"],
    "fig_appC_x2_dose": ["pmrt_nl_eq", "pmrt_eq", "pmrt_r3", "pcorr_eq", "corr", "pcorr_native"],
    "fig_appC_x3_misspec": ["pmrt_nl_eq", "pmrt_eq", "pmrt_r3", "pcorr_eq"],
    "tab_appB_bridge": [
        "pmrt_nl_eq",
        "pmrt_eq",
        "pmrt_r3",
        "pcorr_eq",
        "granger_eq",
        "rcot2_eq",
        "corr",
        "pcorr_native",
        "pcorr_hac_fb",
        "granger_native",
        "rcot2_native",
        "mscr_native",
        "pc_native",
        "notears",
        "shap_dag",
        "two_tower",
        "cdl",
    ],
}
HERMETIC = frozenset(
    {
        "fig_conflict_types",
        "fig_envs",
        "fig_method_pmrt",
        "fig_study4_envs",
        "fig_system_e6p",
        "specimen_bars",
        "specimen_diagram",
        "specimen_entities",
        "specimen_graph",
        "specimen_heatmaps",
        "specimen_kpis",
        "specimen_methods",
        "specimen_panels",
    }
)
DESIGN = Path(__file__).resolve().parents[1] / "DESIGN.md"


@pytest.fixture(autouse=True, scope="session")
def _skip_rendering_without_arial():
    try:
        style.arial_files()
    except RuntimeError as exc:
        reason = str(exc)
    else:
        yield
        return

    def missing():
        pytest.skip(reason)

    mp = pytest.MonkeyPatch()
    mp.setattr(style, "arial_files", missing)
    yield
    mp.undo()


@pytest.fixture(scope="session")
def records():
    if not (RECORDS / "study4_eval" / "eval" / "merged.jsonl.gz").exists():
        pytest.skip("frozen record sets not fetched (data/records)")
    return RECORDS


@pytest.fixture(scope="session")
def journal():
    if not (PAPER / "sections" / "app_c.tex").exists():
        pytest.skip("paper/ not in the workspace")
    return PAPER


@pytest.fixture(scope="session")
def design():
    if not DESIGN.exists():
        pytest.skip("DESIGN.md missing")
    return DESIGN.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def planned():
    return PLANNED


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if {"records", "journal", "design"} & set(item.fixturenames):
            item.add_marker(pytest.mark.slow)


def _figure_params(paper_only: bool) -> list:
    params = []
    for name, mod in registry.discover().items():
        if paper_only and not mod.SPEC.paper:
            continue
        params.append(name if name in HERMETIC else pytest.param(name, marks=pytest.mark.slow))
    return params


def pytest_generate_tests(metafunc):
    if "figure_name" in metafunc.fixturenames:
        metafunc.parametrize("figure_name", _figure_params(paper_only=False))
    if "paper_figure_name" in metafunc.fixturenames:
        metafunc.parametrize("paper_figure_name", _figure_params(paper_only=True))


@pytest.fixture(scope="session")
def build_figure():
    style.apply()
    cache = {}
    mp = pytest.MonkeyPatch()
    mp.setitem(plt.rcParams, "figure.max_open_warning", 0)

    def build(name):
        if name not in cache:
            try:
                cache[name] = registry.discover()[name].build()
            except FileNotFoundError as exc:
                pytest.skip(str(exc))
        return cache[name]

    yield build
    plt.close("all")
    mp.undo()
