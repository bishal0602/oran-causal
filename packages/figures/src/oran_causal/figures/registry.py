"""Figure registry. Every module in `oran_causal.figures.figs` that defines SPEC and build() is a
figure.

Module contract:

SPEC = FigSpec(name="fig_results_utility", width="full", paper=True,
sources=["data/records/..."], caption_hint="...")

def build() -> matplotlib.figure.Figure: ...

`paper=True` writes <figures_dir>/<name>.pdf (what LaTeX includes) plus a PNG preview.
`paper=False` writes the preview only (drafts).
"""

import importlib
import pkgutil
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from oran_causal.figures import figs, style
from oran_causal.figures.paths import figures_dir, preview_dir


@dataclass(frozen=True)
class FigSpec:
    """Registry entry: figure name, width class, whether it ships to the journal, and its sources."""

    name: str
    width: str = "full"  # 'full' | 'half'
    paper: bool = True
    sources: list[str] = field(default_factory=list)
    caption_hint: str = ""


def discover() -> dict[str, object]:
    """Import every figure module and return those with a SPEC and build(), by name."""
    mods = {}
    for info in pkgutil.iter_modules(figs.__path__):
        mod = importlib.import_module(f"{figs.__name__}.{info.name}")
        spec = getattr(mod, "SPEC", None)
        if spec is not None and hasattr(mod, "build"):
            if spec.name in mods:
                raise ValueError(f"duplicate figure name {spec.name}")
            mods[spec.name] = mod
    return dict(sorted(mods.items()))


def render(mod, preview_only: bool = False) -> list[str]:
    """Build one figure and write its PNG preview and, if it is a paper figure, its PDF."""
    style.apply()
    spec: FigSpec = mod.SPEC
    fig = mod.build()
    written = []
    preview = preview_dir()
    preview.mkdir(parents=True, exist_ok=True)
    png = preview / f"{spec.name}.png"
    fig.savefig(png)
    written.append(str(png))
    if spec.paper and not preview_only:
        target = figures_dir()
        target.mkdir(parents=True, exist_ok=True)
        pdf = target / f"{spec.name}.pdf"
        fig.savefig(
            pdf, metadata={"Creator": "oran-causal-figures", "Subject": ", ".join(spec.sources)}
        )
        written.append(str(pdf))
    plt.close(fig)
    return written
