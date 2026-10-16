"""Visible-text guard (TERMINOLOGY s.9-10, user decisions 2026-10-05): internal code names never
appear in figure text, caption hints or the visible part of a generated LaTeX table.
tests/test_wording.py runs ``audit`` on every registered figure, ``check`` on every caption hint and
``tex_visible`` + ``check`` on every table in the journal's ``tables/``. Data files (csv) keep arm
codes on purpose and are not checked.
"""

import re
from pathlib import Path

from matplotlib.text import Text

# pattern -> paper wording (TERMINOLOGY s.10)
BANNED = {
    r"E6[-_ ]?P(?![a-z])": 'the simulator is "the system-level simulator" (never E6-P / E6P)',
    r"\bStudy A\b": '"Study 4"',
    r"\bExp(?:eriment)?\.? B\b": '"the bridging analysis"',
    r"MapGate|CertSafe": '"map-driven referee" / "edge-certified map-driven referee"',
    r"MSCR\+": '"PMRT"',
    r"(?i)\barbiters?\b": '"referee"',
    r"(?i)\bknobs?\b": '"control parameter (NCP)"',
    r"\bCDL\b": '"CDD O-RAN discovery"',
    r"CDD-ORAN": 'repository name only; the conference method is "CDD O-RAN"',
    r"(?i)\boption \(a\)|\bGate A\b|\bK-L0\b": "internal stage names",
    r"\bPowerES\b|\bSliceGuarantee\b": '"Power ES" / "Slice Guarantee"',
    r"\b[a-z0-9]+_(?:nl_)?(?:eq|native)\b|\beq_min\b": "arm codes (use methods.name)",
}
_RX = [(re.compile(p), why) for p, why in BANNED.items()]


def check(text: str) -> list[str]:
    """Banned code names in one string -> ["'<match>' (<use instead>)", ...]."""
    return [f"'{m.group(0)}' ({why})" for rx, why in _RX for m in rx.finditer(text or "")]


def texts(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


def audit(fig) -> list[str]:
    return [f"text '{s}': {hit}" for s in texts(fig) for hit in check(s)]


def tex_visible(path: Path) -> str:
    """The typeset part of a generated .tex table: every line with its comment (an unescaped %) removed."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        out.append(re.split(r"(?<!\\)%", line, maxsplit=1)[0])
    return "\n".join(out)
