"""Figure 1: the three O-RAN WG3 xApp conflict types, on standard examples from the literature.

Each xApp is a coloured line running down through the parameter it writes to the KPI it targets; the conflict type is
read off where the two lines meet: at the parameter tier (direct), at the KPI tier (indirect), or nowhere, with a
hidden side effect reaching the other xApp's KPI (implicit). Chosen by the user 2026-10-09 (prototype B).

Examples and sources (passages checked against the archive copies; cited in the caption, not in the figure):
  a  ES vs CCO on transmit power: QACM (Wadud et al., IEEE TGCN 2024), Sec. VIII-A: "the ES xApp sets TXP to 6 dBm
     ... from its previous value of 40 dBm set by the CCO xApp".
  b  MRO (CIO) vs MLB (antenna tilt) on the handover boundary: O-RAN WG3 TR ConMit R004 Sec. 4.2.1 ("antenna tilts
     and measurement offsets are different control points, but they both impact the handover boundary"); Wadud et
     al., INFOCOM WKSHPS 2025, Sec. II (MRO on CIO, MLB on tilt, "described as an indirect conflict").
  c  GBR throughput protection degrading non-GBR throughput: O-RAN WG3 TR ConMit R004 Sec. 4.2.1 ("protecting
     throughput metrics for GBR users may degrade non-GBR metrics"). WG3 names no parameter, hence "own".
Colours: ES terracotta as in the rest of the paper; CCO grey, like the paper's (coverage) Coverage xApp; HERO blue is
reserved for PMRT and not used here.
Build: uv run python -m oran_causal.figures.svg.fig_intro_conflict
"""

from oran_causal.figures.svg.figkit import (
    COV,
    ES,
    GREEN,
    GREY,
    INK,
    LIGHT,
    PES,
    PURPLE,
    RED,
    SG,
    Svg,
    tint,
)

# [G5 2026-10-12] Arial, all text >= 8 pt (journal rule; was >= 7 pt) [2026-10-12, review] all text
# >= 7 pt at print size: W just under   extwidth (370.6 bp) so LaTeX never shrinks it, panel
# descriptions on two lines, everything below them 8 pt lower
W, H = 368.5, 175
s = Svg(W, H)

GUT = 40
PW = (W - GUT) / 3
Y_NAME, Y_TOP, Y_PAR, Y_KPI = 54, 61, 99, 143
LW = 2.3  # line width
DX = 25  # half spacing of the two lines
LAB = 8.0
LEAD = 9.0  # line pitch of multi-line labels

# tier labels
for y, t in ((Y_NAME - 2.4, "xApps"), (Y_PAR, "parameter"), (Y_KPI, "KPI")):
    s.text(2, y + 2.6, t, 8.0, 600, GREY)
    s.path(f"M{GUT - 2},{y} L{W - 2},{y}", stroke="#F0F1F4", sw=0.6, z=0)


def vline(x0, y0, x1, y1, color, dash=None, sw=LW):
    if abs(x0 - x1) < 0.01:
        d = f"M{x0:.2f},{y0:.2f} L{x1:.2f},{y1:.2f}"
    else:
        my = (y0 + y1) / 2
        d = f"M{x0:.2f},{y0:.2f} C{x0:.2f},{my:.2f} {x1:.2f},{my:.2f} {x1:.2f},{y1:.2f}"
    s.path(d, stroke=color, sw=sw, dash=dash, z=3, cap="butt")


def station(x, y, r=3.3):
    s.circle(x, y, r, fill="#FFFFFF", stroke=INK, sw=1.0, z=6)


def interchange(x, y):
    s.circle(x, y, 10.5, fill=tint(RED, 0.14), stroke="none", sw=0, z=1)
    s.rect(x - 6.2, y - 3.9, 12.4, 7.8, rx=3.9, fill="#FFFFFF", stroke=RED, sw=1.2, z=6)


def terminal(x, name, color, dot=None):
    for i, line in enumerate(
        reversed(name.split("|"))
    ):  # [G5] a long name stacks upwards on two lines
        s.text(x, Y_NAME - 2 - i * LEAD, line, 8.0, 700, INK, anchor="middle")
    s.circle(x if dot is None else dot, Y_TOP + 1.5, 2.6, fill=color, stroke="none", z=6)


def lab(x, y, lines, anchor="middle", fill=INK, size=LAB, weight=400, italic=False):
    for i, l in enumerate(lines):
        s.text(x, y + i * LEAD, l, size, weight, fill, anchor=anchor, italic=italic, z=9)


def heading(i, name, desc):
    x = GUT + i * PW + 2
    s.text(x, 9.5, name, 9.0, 700, RED)
    for j, line in enumerate(desc):
        s.text(x, 20.0 + j * LEAD, line, 8.0, 400, GREY)


# panel separators
for i in (1, 2):
    x = GUT + i * PW - 3
    s.path(f"M{x:.2f},4 L{x:.2f},{H - 2}", stroke=LIGHT, sw=0.5, z=0)

# ============================================================ a  Direct
cx = GUT + 0 * PW + PW / 2 - 6
a, b = cx - DX, cx + DX
heading(0, "Direct", ["two xApps write", "one parameter"])
terminal(a, "ES", ES)
terminal(b, "CCO", COV)
vline(a, Y_TOP, cx - 1.3, Y_PAR, ES)
vline(b, Y_TOP, cx + 1.3, Y_PAR, COV)
vline(cx - 1.3, Y_PAR, a, Y_KPI, ES)
vline(cx + 1.3, Y_PAR, b, Y_KPI, COV)
interchange(cx, Y_PAR)
lab(cx - 11, Y_PAR - 1.2, ["transmit", "power"], anchor="end", weight=600)
lab(a - 3, Y_TOP + 13, ["lowers"], anchor="end", fill=GREY)
lab(b + 3, Y_TOP + 13, ["sets", "high"], anchor="start", fill=GREY)
station(a, Y_KPI)
station(b, Y_KPI)
lab(a, Y_KPI + 11, ["power", "consumption"])
lab(b, Y_KPI + 11, ["DL", "throughput"])

# ============================================================ b  Indirect
cx = GUT + 1 * PW + PW / 2 - 11  # [G5] was - 4: Arial 8 pt 'antenna' cleared the separator
a, b = cx - 19, cx + 19
heading(1, "Indirect", ["different parameters,", "one shared KPI"])
terminal(a, "MRO", PURPLE)
terminal(b, "MLB", GREEN)
vline(a, Y_TOP, a, Y_PAR, PURPLE)
vline(b, Y_TOP, b, Y_PAR, GREEN)
vline(a, Y_PAR, cx - 1.3, Y_KPI, PURPLE)
vline(b, Y_PAR, cx + 1.3, Y_KPI, GREEN)
station(a, Y_PAR)
station(b, Y_PAR)
lab(a - 6, Y_PAR + 2.2, ["CIO"], anchor="end", weight=600)
lab(b + 6, Y_PAR - 1.2, ["antenna", "tilt"], anchor="start", weight=600)
interchange(cx, Y_KPI)
lab(cx, Y_KPI + 15, ["handover boundary"], weight=600)
lab(cx, Y_KPI + 15 + LEAD, ["and handover KPIs"], fill=GREY)

# ============================================================ c  Implicit
cx = GUT + 2 * PW + PW / 2 - 4
a, b = cx - DX, cx + DX
heading(2, "Implicit", ["no shared parameter", "or declared KPI"])
terminal(a, "GBR|xApp", SG)
terminal(b, "non-GBR|xApp", PES)
for x, col in ((a, SG), (b, PES)):
    vline(x, Y_TOP, x, Y_KPI, col)
    station(x, Y_PAR)
lab(a - 6, Y_PAR + 2.2, ["own"], anchor="end", fill=GREY, italic=True)
lab(b + 6, Y_PAR + 2.2, ["own"], anchor="start", fill=GREY, italic=True)
# degraded KPI of the other xApp
s.circle(b, Y_KPI, 9.5, fill=tint(RED, 0.14), stroke="none", sw=0, z=1)
station(a, Y_KPI)
s.circle(b, Y_KPI, 3.6, fill="#FFFFFF", stroke=RED, sw=1.3, z=6)
# hidden side effect: protecting GBR throughput degrades non-GBR throughput
s.arrow(
    [(a + 3.5, Y_PAR + 2.5), (a + 26, Y_PAR + 12), (b - 18, Y_KPI - 18), (b - 5.2, Y_KPI - 3.6)],
    color=RED,
    dash="2.4,1.6",
    sw=1.0,
    z=4,
    hs=4.2,
)
lab(a + 5, Y_PAR - 5 - LEAD, ["side effect:", "degrades"], anchor="start", fill=RED)
lab(a, Y_KPI + 11, ["GBR", "throughput"])
lab(b, Y_KPI + 11, ["non-GBR", "throughput"])

if __name__ == "__main__":
    s.render("fig_intro_conflict")
