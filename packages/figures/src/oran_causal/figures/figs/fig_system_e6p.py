"""Section 3: the system-level simulator, its xApps, the WG3-style ConMit referee and the logging /
discovery loop.

Content from TERMINOLOGY s.2-7 and CDD-ORAN docs/ARCHITECTURE.md s.2-4 (diagram; no numbers beyond
the stated design facts: 24 cells, LL / eMBB / BE slices, the three xApps of the studied scenario).
The hero component is the referee (DESIGN s.9 / s.12.2 rule 3); xApps in their entity slots,
Coverage left out (s.12.2 rule 4); arrows ink.
"""

from oran_causal.figures import diagram as D
from oran_causal.figures import entities as E
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_system_e6p",
    width="full",
    paper=True,
    sources=["CDD-ORAN/docs/ARCHITECTURE.md", "TERMINOLOGY.md s.2-7"],
    caption_hint="The system-level simulator and the conflict-mitigation loop. xApps read E2 KPM reports "
    "and send requests; the referee accepts or rejects each request and logs its mode and "
    "propensity; PMRT turns the logs into a signed causal map that drives the referee. The "
    "knockout ground truth is used for evaluation only and never read by discovery. Colours "
    "identify xApps only; the referee is highlighted",
)

SIM_X, SIM_W, RIC_X, RIC_W, LOG_X, LOG_W = 2.0, 112.0, 140.0, 116.0, 268.0, 96.0  # [G5] SIM_X was 6


def build():
    fig, ax = D.canvas("full", height_pt=192)
    # system-level simulator
    D.container(ax, SIM_X, 6, SIM_W, 180, "System-level simulator")
    sim = [
        ("24 cells", "macro sectors and picos"),
        ("UEs", "LL, eMBB and BE slices"),
        ("Energy", "EARTH power model"),
        ("KPIs per cell", "violations, RLF, energy, load"),
    ]
    for i, (lab, sub) in enumerate(sim):
        D.box(
            ax, SIM_X + 4, 26 + i * 40, SIM_W - 8, 30, lab, sub
        )  # [G5] 8 pt subtitles need 104 pt
    # near-RT RIC: xApps and the referee (hero)
    D.container(ax, RIC_X, 6, RIC_W, 180, "Near-RT RIC")
    bx, bw = RIC_X + 8, RIC_W - 16
    for i, k in enumerate(("ES", "PowerES", "SliceGuarantee")):
        E.xapp_box(ax, bx, 26 + i * 22, bw, 18, k)
    # [2026-10-12] the referee box is 2 pt wider each side than the xApp boxes: its 7 pt subtitle
    # fills 100 pt
    rx, rw = bx - 2, bw + 4
    D.box(
        ax, rx, 122, rw, 36, "Referee", "accepts or rejects requests", role="proposed"
    )  # [G5] 8 pt: "each" cut
    D.arrow(
        ax,
        [(bx + bw / 2, 92), (bx + bw / 2, 121)],
        label="requests",
        label_at=(bx + bw / 2 + 22, 110),
    )
    D.arrow(ax, [(SIM_X + SIM_W - 6, 57), (bx - 1, 57)], label="E2 KPM", label_at=(128, 53))
    D.arrow(
        ax,
        [(rx, 141), (SIM_X + SIM_W - 5, 141)],
        dashed=True,
        label="E2 control",
        label_at=(128, 137),
    )
    # logging and discovery
    ax.text(LOG_X, 16, "Logging and discovery", fontsize=8, fontweight="bold", va="center")
    logs = [
        ("Logged requests", "mode and propensity", 26),
        ("PMRT", "design-based test", 68),
        ("Causal map", "signed edges", 110),
    ]
    for lab, sub, y in logs:
        D.box(ax, LOG_X, y, LOG_W, 30, lab, sub)
    D.box(ax, LOG_X, 152, LOG_W, 30, "Ground truth", "knockout, evaluation only", ls=(0, (3, 2)))
    cx = LOG_X + LOG_W / 2
    D.arrow(ax, [(cx, 56), (cx, 67)])
    D.arrow(ax, [(cx, 98), (cx, 109)])
    D.arrow(
        ax,
        [(rx + rw, 130), (262, 130), (262, 41), (LOG_X - 1, 41)],
        label="logs",
        label_at=(262 - 10, 112),
    )
    # "frozen map": the map is learned offline and fixed before evaluation (team review 2026-10-09);
    # the longer label sits right of the vertical stub, in the gap between Causal map and Ground
    # truth
    D.arrow(
        ax,
        [(LOG_X + 14, 140), (LOG_X + 14, 150), (rx + rw + 1, 150)],
        label="frozen map",
        label_at=(LOG_X + 36, 149.25),
    )
    return fig
