"""Build the specification: pdflatex twice, output in spec/build/.

Usage (stdlib only, run from anywhere):
    uv run python spec/build.py [--out DIR]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDFLATEX = ("pdflatex", "-interaction=nonstopmode", "-file-line-error")


def build(root: Path, out: Path) -> Path:
    """Builds ``main.tex`` (two passes for references) and returns the PDF path."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [*PDFLATEX, f"-output-directory={out}", "main.tex"]
    for _ in range(2):
        code = subprocess.run(cmd, cwd=root, check=False).returncode
        if code != 0:
            sys.exit(f"{' '.join(cmd)} failed with exit code {code}")
    return out / "main.pdf"


def main(argv: list[str] | None = None) -> int:
    """Parses the arguments and builds the specification."""
    ap = argparse.ArgumentParser(prog="spec/build.py", description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "build", help="output directory")
    args = ap.parse_args(argv)
    print(build(ROOT, args.out.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
