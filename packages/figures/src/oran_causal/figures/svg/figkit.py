"""Small SVG kit for hand-laid diagrams (Figure 1), outside the matplotlib pipeline.

Coordinates are in pt; the SVG is rendered at its pt size by the installed Edge (Playwright) to a
vector PDF in the journal's figures/ and a PNG preview in preview/. Text is set in Arial (journal
artwork rule), embedded from the system fonts (oran_causal.figures.style).
"""

from __future__ import annotations

import math
import tempfile
from html import escape
from pathlib import Path

from oran_causal.figures.paths import figures_dir, preview_dir
from oran_causal.figures.style import FONT_FAMILY, arial_files

ARIAL = dict(zip(("Regular", "Bold", "Italic", "BoldItalic"), arial_files(), strict=False))

# Harbor house palette
INK = "#1E2430"
GREY = "#6B7280"
LIGHT = "#C9CDD4"
ES = "#C17154"
PES = "#CBAA56"
SG = "#3FAFA9"
COV = "#8A909C"
RED = "#B94739"
BLUE = "#4273B0"
PURPLE = "#955C98"
GREEN = "#5C9B5E"


def tint(hex_: str, a: float) -> str:
    """Mix colour with white; a=1 full colour."""
    h = hex_.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    m = lambda c: round(255 - (255 - c) * a)
    return f"#{m(r):02X}{m(g):02X}{m(b):02X}"


class Svg:
    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
        self.layers: dict[int, list[str]] = {}

    def add(self, s: str, z: int = 5):
        self.layers.setdefault(z, []).append(s)

    # ---- primitives -------------------------------------------------
    def text(
        self,
        x,
        y,
        s,
        size=8,
        weight=400,
        fill=INK,
        anchor="start",
        italic=False,
        z=8,
        ls=0.0,
        caps=False,
    ):
        st = "font-style:italic;" if italic else ""
        if ls:
            st += f"letter-spacing:{ls}pt;"
        if caps:
            st += "text-transform:uppercase;"
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" style="{st}">{escape(s)}</text>',
            z,
        )

    def rect(self, x, y, w, h, rx=3, fill="none", stroke=INK, sw=0.8, dash=None, z=4, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{op}"/>',
            z,
        )

    def circle(self, cx, cy, r, fill="none", stroke=INK, sw=0.8, dash=None, z=4, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d} opacity="{op}"/>',
            z,
        )

    def path(
        self, d, stroke=INK, sw=0.8, fill="none", dash=None, z=3, cap="round", op=1.0, join="round"
    ):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dd} '
            f'stroke-linecap="{cap}" stroke-linejoin="{join}" opacity="{op}"/>',
            z,
        )

    def head(self, x, y, ang, color=INK, size=4.2, z=6):
        """Filled arrowhead with tip at (x,y) pointing along angle ang (rad)."""
        a = 0.42
        p1 = (x - size * math.cos(ang - a), y - size * math.sin(ang - a))
        p2 = (x - size * math.cos(ang + a), y - size * math.sin(ang + a))
        bx, by = x - size * 0.72 * math.cos(ang), y - size * 0.72 * math.sin(ang)
        self.add(
            f'<path d="M{x:.2f},{y:.2f} L{p1[0]:.2f},{p1[1]:.2f} L{bx:.2f},{by:.2f} '
            f'L{p2[0]:.2f},{p2[1]:.2f} Z" fill="{color}"/>',
            z,
        )

    def arrow(self, pts, color=INK, sw=0.8, dash=None, z=3, hs=4.2, curve=True):
        """Polyline / smooth cubic through control points; head at last point.

        pts with 2 points: straight; 3: quadratic; 4: cubic.
        """
        (x0, y0) = pts[0]
        xe, ye = pts[-1]
        if len(pts) == 2:
            d = f"M{x0:.2f},{y0:.2f} L{xe:.2f},{ye:.2f}"
            px, py = x0, y0
        elif len(pts) == 3:
            d = f"M{x0:.2f},{y0:.2f} Q{pts[1][0]:.2f},{pts[1][1]:.2f} {xe:.2f},{ye:.2f}"
            px, py = pts[1]
        else:
            d = (
                f"M{x0:.2f},{y0:.2f} C{pts[1][0]:.2f},{pts[1][1]:.2f} {pts[2][0]:.2f},{pts[2][1]:.2f} "
                f"{xe:.2f},{ye:.2f}"
            )
            px, py = pts[2]
        ang = math.atan2(ye - py, xe - px)
        # shorten the stroke so the line does not poke through the head tip
        self.path(d, stroke=color, sw=sw, dash=dash, z=z)
        self.head(xe, ye, ang, color=color, size=hs, z=z + 1)

    # ---- output -----------------------------------------------------
    def svg(self) -> str:
        body = "\n".join(s for k in sorted(self.layers) for s in self.layers[k])
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}pt" height="{self.h}pt" font-family="{FONT_FAMILY}">\n'
            f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>\n{body}\n</svg>'
        )

    def render(self, stem: str):
        faces = "".join(
            f"@font-face{{font-family:'{FONT_FAMILY}';src:url('{ARIAL[n].as_uri()}');"
            f"font-weight:{w};font-style:{st};}}"
            for n, w, st in (
                ("Regular", 400, "normal"),
                ("Bold", 500, "normal"),
                ("Bold", 600, "normal"),
                ("Bold", 700, "normal"),
                ("Italic", 400, "italic"),
                ("BoldItalic", 700, "italic"),
            )
        )
        svg = self.svg()
        html = (
            f"<!doctype html><html><head><meta charset='utf-8'><style>{faces}"
            f"@page{{margin:0}}html,body{{margin:0;padding:0}}"
            f"svg{{display:block;width:{self.w}pt;height:{self.h}pt}}</style></head><body>{svg}</body></html>"
        )
        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
            hp = Path(tmp) / f"{stem}.html"
            hp.write_text(html, encoding="utf-8")
            br = pw.chromium.launch(channel="msedge")
            pg = br.new_page(
                device_scale_factor=4,
                viewport={
                    "width": math.ceil(self.w * 96 / 72),
                    "height": math.ceil(self.h * 96 / 72),
                },
            )
            pg.goto(hp.as_uri())
            pg.wait_for_timeout(300)
            pdf, png = figures_dir() / f"{stem}.pdf", preview_dir() / f"{stem}.png"
            pg.pdf(
                path=str(pdf),
                width=f"{self.w / 72:.4f}in",
                height=f"{self.h / 72 + 0.001:.4f}in",
                print_background=True,
                page_ranges="1",
            )
            pg.locator("svg").screenshot(path=str(png))
            br.close()
        print("wrote", pdf, f"({self.w / 72:.2f} in x {self.h / 72:.2f} in)")
        print("wrote", png)
