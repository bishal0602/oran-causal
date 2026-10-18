"""Global method-colour mapping (oran_causal.figures.methods, DESIGN s.12, [DECIDED 2026-10-08,
user]): table invariants, the mapping rules on every registered figure, one hue per method inside a
figure, the same style for a method in every figure, and each rule firing on a broken figure.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots, registry, style

FIGS = registry.discover()
RULES = ["one hero", "variants", "frozen PMRT", "wording"]


@pytest.fixture(scope="module")
def built(records):
    style.apply()
    out = {}
    for name, mod in FIGS.items():
        fig = mod.build()
        out[name] = (M.audit(fig), _styles(fig))
        plt.close(fig)
    return out


def _styles(fig) -> dict[str, set]:
    """Method key -> {(colour, marker, line style)} of its series in this figure."""
    out = {}
    for kind, key, _, art in M.tagged(fig):
        if kind == "method" and hasattr(art, "get_marker"):
            out.setdefault(key, set()).add(
                (art.get_color(), art.get_marker(), str(art.get_linestyle()))
            )
    return out


# --- table invariants -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fig",
    [
        "fig_study1_discovery",
        "fig_study2_maps",
        "fig_decision_referee",
        "fig_decision_referee_v3",
        "fig_study2_deferrals",
        "fig_study3_certification",
        "fig_appC_x2_dose",
        "fig_appC_x3_misspec",
        "tab_appB_bridge",
    ],
)
def test_planned_arms_resolve(fig, planned):
    for code in planned[fig]:
        M.resolve(
            code, study13=fig.startswith(("fig_study1", "fig_study2", "fig_study3", "fig_decision"))
        )


def test_marker_shape_is_the_base_method():
    """Plain and design-adjusted versions of a test share the marker and differ in hue (they share
    Study 4 figures); RCoT is drawn in ink rows only, so both versions stay neutral.
    """
    for plain, adj in (
        ("pcorr_native", "pcorr_eq"),
        ("granger_native", "granger_eq"),
        ("rcot2_native", "rcot2_eq"),
    ):
        sp, sa = M.style(plain), M.style(adj)
        assert sp["marker"] == sa["marker"]
        assert (sp["ls"], sa["ls"]) == (M.BLIND_DASH, "-")
    assert M.style("pcorr_native")["color"] != M.style("pcorr_eq")["color"]
    assert M.style("granger_native")["color"] != M.style("granger_eq")["color"]
    assert M.color("rcot2_native") == M.color("rcot2_eq") == P.NEUTRAL


def test_family_cues():
    """[DECIDED 2026-10-18, user] Design-based and design-adjusted methods are solid, design-blind ones dashed."""
    for code in ("pmrt_nl_eq", "pcorr_eq", "pmrt_r3"):
        assert M.style(code)["ls"] == "-", code
    for code in ("corr", "pcorr_native", "two_tower", "cdl", "granger_by"):
        assert M.style(code)["ls"] == M.BLIND_DASH, code
    o = M.style("MG:GT")
    assert o["color"] == P.INK
    assert o["mfc"] == "none"
    assert o["alpha"] == 0.6
    assert o["ls"] == M.DASHDOT
    assert M.style("MG:rand")["marker"] == "+"
    assert {M.METHODS[k].family for k in ("cdl", "int")} == {"blind_learned"}
    assert M.METHODS["cdl"].marker not in ("h", "H")  # not a rotation of INT's hexagon


def test_gate_has_no_marker_cue():
    """CS:<map> looks exactly like MG:<map> ([DECIDED 2026-10-08, user]: no gate ring; the banners
    name the gate).
    """
    for m in ("PMRT", "corr@dev", "granger_by"):
        mg, cs = M.style(f"MG:{m}"), M.style(f"CS:{m}")
        assert {k: v for k, v in mg.items() if k != "gid"} == {
            k: v for k, v in cs.items() if k != "gid"
        }
    fig, ax = plots.figure("half")
    M.plot(ax, [1, 2], [1, 2], "CS:PMRT", hero=True)
    M.row_point(ax, 1.5, 1.5, (1, 2), "CS:GT")
    n = len(ax.lines)
    plt.close(fig)
    assert n == 1 + 1 + 1 + 2  # the series, the halo, the glyph and two error-bar caps: no ring


def test_unplotted_methods_refuse_a_marker():
    for code in ("pc_native", "notears", "shap_dag"):
        with pytest.raises(ValueError):
            M.style(code)
    with pytest.raises(KeyError):
        M.resolve("blanket2")


# --- rules on every registered figure -------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", RULES)
def test_mapping_rules(build_figure, figure_name, rule):
    if rule == "wording" and not FIGS[figure_name].SPEC.paper:
        pytest.skip("style specimens use generic 'Proposed / Baseline' names; not in the paper")
    if rule in ("one hero", "frozen PMRT") and figure_name == "specimen_key":
        pytest.skip("the key sheet shows the Studies 1-3 PMRT points beside the Study 4 series")
    problems = M.audit(build_figure(figure_name))[rule]
    assert not problems, f"{figure_name}: " + "; ".join(problems)


@pytest.mark.parametrize("name", [n for n, m in FIGS.items() if m.SPEC.paper])
def test_caption_wording(name):
    hint = FIGS[name].SPEC.caption_hint.lower()
    assert not any(w in hint.split() for w in M.BANNED_WORDS)


NEUTRAL_HUES = {P.INK.lower(), P.NEUTRAL.lower()}
SHARED_TINT = {
    "pmrt_lin",
    "pmrt_lin_r3",
}  # PMRT-Lin and its R3 variant: one tint, told apart by marker


def test_one_hue_per_method_in_a_figure(built):
    """No figure draws two methods in one hue (references and static rules are ink / gray and exempt)."""
    from matplotlib.colors import to_hex

    for name, (_, styles) in built.items():
        hues = {}
        for key, st in styles.items():
            c = to_hex(next(iter(st))[0]).lower()
            if c not in NEUTRAL_HUES:
                hues.setdefault(c, set()).add(key)
        clash = {c: k for c, k in hues.items() if len(k) > 1 and not k <= SHARED_TINT}
        assert not clash, f"{name}: methods share a hue: {clash}"


def test_same_style_in_every_figure(built):
    """A method looks the same (hue, marker, line style) in every figure that draws it as a series."""
    seen = {}
    for name, (_, styles) in built.items():
        for key, st in styles.items():
            assert len(st) == 1, f"{name}: {key} drawn in {len(st)} styles"
            seen.setdefault(key, {})[name] = next(iter(st))
    for key, per_fig in seen.items():
        assert len(set(per_fig.values())) == 1, (
            f"{key} styled differently across figures: {per_fig}"
        )


# --- each rule fires on a deliberately broken figure ----------------------------------------------------------------
def _bad(rule):
    fig, ax = plots.figure("half")
    x = np.arange(3.0)
    if rule == "one hero":
        M.plot(ax, x, x, "pmrt_nl_eq", hero=True)
        M.plot(ax, x, x + 1, "pcorr_eq", hero=True)
    elif rule == "variants":
        M.plot(ax, x, x, "pmrt_nl_eq", hero=True)
        M.plot(ax, x, x + 1, "pcorr_native")
        M.plot(ax, x, x + 2, "pcorr_hac_fb")
    elif rule == "frozen PMRT":
        M.plot(ax, x, x, "pmrt_nl_eq", hero=True)
        M.plot(ax, x, x + 1, "loadsp_c+wby1s")
    elif rule == "wording":
        M.plot(ax, x, x, "pmrt_nl_eq", hero=True, label="Proposed")
        ax.legend()
    return fig


@pytest.mark.parametrize("rule", RULES)
def test_rule_fires(rule):
    style.apply()
    fig = _bad(rule)
    problems = M.audit(fig)[rule]
    plt.close(fig)
    assert problems, f"{rule} did not flag its broken figure"


def test_variants_allowed_as_rows():
    style.apply()
    fig, ax = plots.figure("half")
    M.row_point(ax, 0.05, 0, (0.04, 0.06), "pmrt_nl_eq", hero=True, ink=True)
    for i, code in enumerate(("pcorr_native", "pcorr_hac", "pcorr_hac_fb")):
        M.row_point(ax, 0.2, i + 1, (0.1, 0.3), code, ink=True)
    problems = M.audit(fig)["variants"]
    plt.close(fig)
    assert not problems
