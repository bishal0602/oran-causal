"""Console entry point: `oran-causal-figures list | build [NAME ...] [--preview] [--out DIR]`."""

import argparse
import os
import sys

import matplotlib

from oran_causal.figures.paths import OUT_ENV


def main(argv: list[str] | None = None) -> int:
    """Run the figures CLI; returns the process exit code."""
    ap = argparse.ArgumentParser(
        prog="oran-causal-figures", description="Build the oran-causal journal figures"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list registered figures")
    b = sub.add_parser("build", help="build figures (all if no names given)")
    b.add_argument("names", nargs="*")
    b.add_argument(
        "--preview", action="store_true", help="write PNG previews only, skip the paper PDFs"
    )
    b.add_argument(
        "--out",
        help="write figures/, tables/ and preview/ under this directory instead of paper/",
    )
    args = ap.parse_args(argv)
    matplotlib.use("Agg")
    from oran_causal.figures import registry

    mods = registry.discover()
    if args.cmd == "list":
        for name, mod in mods.items():
            s = mod.SPEC
            print(
                f"{name:40s} {s.width:5s} {'paper' if s.paper else 'preview'}  {', '.join(s.sources) or '-'}"
            )
        return 0

    names = args.names or list(mods)
    missing = [n for n in names if n not in mods]
    if missing:
        print(
            f"unknown figure(s): {', '.join(missing)}; see `uv run oran-causal-figures list`",
            file=sys.stderr,
        )
        return 2
    if args.out:
        os.environ[OUT_ENV] = args.out
    for n in names:
        for path in registry.render(mods[n], preview_only=args.preview):
            print(f"wrote {path}")
    return 0
