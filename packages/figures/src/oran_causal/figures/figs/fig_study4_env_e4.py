"""Appendix A, E4 page: causal graph, rejection rate and recall vs n in R1 and R2 (template in
``_env_page``).
"""

from oran_causal.figures.figs import _env_page as E
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study4_env_e4",
    width="full",
    paper=True,
    sources=E.sources("E4"),
    caption_hint=f"E4 ({E.BLURB['E4']}): ground-truth causal graph with the confounder Z and the R3 "
    "policy, placebo-edge rejection rate (no truth-null candidate in R1 / R2) and recall vs n "
    "in R1 and R2." + E.COLOUR_NOTE,
)


def build():
    return E.build_page("E4")
