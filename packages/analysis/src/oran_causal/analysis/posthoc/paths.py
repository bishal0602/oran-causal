"""Locations of the frozen record sets and artifacts the post hoc analyses read."""

import hashlib
from pathlib import Path

from oran_causal.analysis import recorded

WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
RECORDS = WORKSPACE_ROOT / "data" / "records"
ARTIFACTS = WORKSPACE_ROOT / "packages" / "experiments" / "artifacts"
PROTOCOLS = WORKSPACE_ROOT / "packages" / "experiments" / "protocols"
SPECS = WORKSPACE_ROOT / "packages" / "experiments" / "specs"
PROTOCOL = PROTOCOLS / "posthoc" / "PROTOCOL.md"
CITED_PROTOCOL = recorded.recorded("protocols/posthoc/PROTOCOL.md")
PROTOCOL_NAME = CITED_PROTOCOL.name
PROTOCOL_BASENAME = PROTOCOL_NAME.rsplit("/", 1)[-1]
PRIMARY = "loadsp_c+wby1s"


def lf_sha256(path: str | Path) -> str:
    """Sha256 of a file with CRLF read as LF, the freeze-commit hash rule."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_text(path: str | Path, text: str) -> None:
    """Writes UTF-8 text with LF endings, creating the parent directory."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
