"""Appendix A, E3 page: causal graph, truth-null rate and recall vs n in R1 and R2 (template in
``_env_page``).
"""

from oran_causal.figures.figs import _env_page as E
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study4_env_e3",
    width="full",
    paper=True,
    sources=E.sources("E3"),
    caption_hint=f"E3 ({E.BLURB['E3']}): ground-truth causal graph, truth-null rejection rate and recall "
    "vs n in R1 and R2." + E.COLOUR_NOTE,
)


def build():
    return E.build_page("E3")
