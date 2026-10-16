"""Appendix A, E2 page: causal graph, truth-null rate and recall vs n in R1 and R2 (template in
``_env_page``).
"""

from oran_causal.figures.figs import _env_page as E
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study4_env_e2",
    width="full",
    paper=True,
    sources=E.sources("E2"),
    caption_hint=f"E2 ({E.BLURB['E2']}): ground-truth causal graph, truth-null rejection rate and recall "
    "vs n in R1 and R2." + E.COLOUR_NOTE,
)


def build():
    return E.build_page("E2")
