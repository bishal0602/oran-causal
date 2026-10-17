"""Build the manuscript: pdflatex, bibtex, pdflatex twice, output in build/.

Usage (stdlib only, run from anywhere):
    uv run python paper/build.py [sn-article | supplementary ...] [--out DIR]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ("sn-article", "supplementary")
PDFLATEX = ("pdflatex", "-synctex=1", "-interaction=nonstopmode", "-file-line-error")


def run(cmd: list[str], cwd: Path) -> None:
    """Runs one command and exits with a message when it fails."""
    code = subprocess.run(cmd, cwd=cwd, check=False).returncode
    if code != 0:
        sys.exit(f"{' '.join(cmd)} failed with exit code {code}")


def build(doc: str, root: Path, out: Path) -> Path:
    """Builds one document with the journal recipe and returns its PDF path."""
    out.mkdir(parents=True, exist_ok=True)
    pdflatex = [*PDFLATEX, f"-output-directory={out}", f"{doc}.tex"]
    run(pdflatex, root)
    aux = out / f"{doc}.aux"
    if r"\citation" in aux.read_text(encoding="utf-8", errors="replace"):
        run(["bibtex", str(out / doc)], root)
    run(pdflatex, root)
    run(pdflatex, root)
    return out / f"{doc}.pdf"


def main(argv: list[str] | None = None) -> int:
    """Parses the arguments and builds the requested documents (default: sn-article)."""
    ap = argparse.ArgumentParser(prog="paper/build.py", description=__doc__)
    ap.add_argument("docs", nargs="*", choices=DOCS, metavar="doc")
    ap.add_argument("--root", type=Path, default=ROOT, help="manuscript directory")
    ap.add_argument("--out", type=Path, help="output directory (default: <root>/build)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    out = (args.out or root / "build").resolve()
    for doc in args.docs or DOCS[:1]:
        print(build(doc, root, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
