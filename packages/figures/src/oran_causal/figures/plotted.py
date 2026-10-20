"""The layout of the PDF a figure saves to."""

import io
import re

from matplotlib.figure import Figure


def pdf_layout(fig: Figure) -> dict:
    """Page boxes, page count and font names of the PDF the figure saves to."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    data = buf.getvalue()
    boxes = [m.decode() for m in re.findall(rb"/MediaBox\s*\[([^\]]*)\]", data)]
    names = {m.decode() for m in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-_,]+)", data)}
    fonts = sorted(re.sub(r"^[A-Z]{6}\+", "", n) for n in names)
    return {"mediabox": boxes, "fonts": fonts, "pages": len(re.findall(rb"/Type\s*/Page\b", data))}
