"""Reads the LaTeX manuscript and its generated values for the paper rules F8-F11 and F14."""

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOTS = ("sn-article.tex", "supplementary.tex")
PROSE_ROOT = "sn-article.tex"
INPUT = re.compile(r"^\s*\\input\{([^}]+)\}\s*$")
CLAIM = re.compile(r"\\claim\{([^}]+)\}")
VAL = re.compile(r"\\val\{([^}]+)\}")
DEFVAL = re.compile(r"^\\defval\{([^}]+)\}\{(.*)\}$")
HEADING = re.compile(
    r"\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\s*(?:\[[^\]]*\])?\s*\{"
)
MATH = re.compile(
    r"\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]"
    r"|\\begin\{(equation|align|eqnarray|gather|multline)\*?\}.*?\\end\{\1\*?\}",
    re.DOTALL,
)
TABULAR = re.compile(r"\\begin\{(tabular\*?|longtable)\}.*?\\end\{\1\}", re.DOTALL)
REFERENCE_ARGS = re.compile(
    r"\\(?:ref|eqref|autoref|label|cite[a-z]*|includegraphics|input|claim|val|url|pageref)"
    r"(?:\[[^\]]*\])?\{[^}]*\}"
)
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\w])")
SECTION = re.compile(r"\\(?:chapter|section)\*?\s*(?:\[[^\]]*\])?\{")
POSTHOC = re.compile(
    r"post[ -]?hoc|(?:chosen|declared|decided)\s+after\s+the\s+protocols?\s+(?:was|were)\s+fixed",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Place:
    """A position in a manuscript source file."""

    file: str
    line: int


@dataclass(frozen=True, slots=True)
class Unit:
    """The text of one heading, from the sectioning command to the next one."""

    file: str
    line: int
    text: str
    lead: str = ""


@dataclass(frozen=True, slots=True)
class Decimal:
    """A decimal literal printed in prose."""

    place: Place
    literal: str


@dataclass(frozen=True, slots=True)
class Exemption:
    """A decimal literal that is not a result, with the reason."""

    file: str
    literal: str
    reason: str
    count: int | None = None


@dataclass(frozen=True, slots=True)
class PaperScan:
    """Everything the paper rules read from the manuscript sources."""

    claims: tuple[tuple[str, Place], ...] = ()
    vals: tuple[tuple[str, Place], ...] = ()
    claim_units: Mapping[str, tuple[Unit, ...]] = field(default_factory=dict)
    decimals: tuple[Decimal, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedValues:
    """`paper/generated/values.tex` and its JSON sidecar."""

    texts: Mapping[str, str]
    sidecar: Mapping[str, Any] | None


def _source_files(root: Path, name: str, seen: set[Path]) -> Iterator[Path]:
    path = root / name
    if path in seen or not path.is_file():
        return
    seen.add(path)
    yield path
    for line in path.read_text(encoding="utf-8").splitlines():
        match = INPUT.match(line)
        if match and not match[1].startswith("generated/"):
            target = match[1] if match[1].endswith(".tex") else f"{match[1]}.tex"
            yield from _source_files(root, target, seen)


def source_files(paper: Path) -> tuple[Path, ...]:
    r"""The root documents and every file they `\input`, in reading order, each once."""
    seen: set[Path] = set()
    found = [path for name in ROOTS for path in _source_files(paper, name, seen)]
    return tuple(found)


def _uncommented(text: str) -> str:
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _units(name: str, text: str) -> list[Unit]:
    starts = [m.start() for m in HEADING.finditer(text)] or [0]
    if starts[0] != 0:
        starts.insert(0, 0)
    ends = [*starts[1:], len(text)]
    units, lead = [], ""
    for a, b in zip(starts, ends, strict=True):
        body = text[a:b]
        if SECTION.match(body):
            lead = body
        units.append(Unit(name, _line_of(text, a), body, lead))
    return units


def _prose(text: str) -> str:
    blanked = text
    for pattern in (TABULAR, MATH, REFERENCE_ARGS):
        blanked = pattern.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), blanked)
    return blanked


def scan_paper(paper: Path) -> PaperScan | None:
    """Scans the manuscript under `paper/`, or returns None when it is absent."""
    files = source_files(paper)
    if not files:
        return None
    article = set(_source_files(paper, PROSE_ROOT, set()))
    claims, vals, decimals = [], [], []
    units: dict[str, list[Unit]] = {}
    for path in files:
        name = path.relative_to(paper).as_posix()
        text = _uncommented(path.read_text(encoding="utf-8"))
        claims += [(m[1], Place(name, _line_of(text, m.start()))) for m in CLAIM.finditer(text)]
        vals += [(m[1], Place(name, _line_of(text, m.start()))) for m in VAL.finditer(text)]
        for unit in _units(name, text):
            for match in CLAIM.finditer(unit.text):
                units.setdefault(match[1], []).append(unit)
        if path in article:
            prose = _prose(text)
            decimals += [
                Decimal(Place(name, _line_of(prose, m.start())), m[0])
                for m in DECIMAL.finditer(prose)
            ]
    return PaperScan(
        claims=tuple(claims),
        vals=tuple(vals),
        claim_units={key: tuple(value) for key, value in units.items()},
        decimals=tuple(decimals),
    )


def read_generated(paper: Path) -> GeneratedValues | None:
    """Reads `generated/values.tex` and `generated/values.json`, or None without the .tex."""
    tex = paper / "generated" / "values.tex"
    if not tex.is_file():
        return None
    texts = {
        m[1]: m[2]
        for line in tex.read_text(encoding="utf-8").splitlines()
        if (m := DEFVAL.match(line))
    }
    sidecar_path = paper / "generated" / "values.json"
    sidecar = None
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")).get("values", {})
    return GeneratedValues(texts, sidecar)


def read_exemptions(path: Path) -> tuple[Exemption, ...]:
    r"""Reads the decimals that the paper may print without `\val` (`spec/paper_exempt.yaml`)."""
    if not path.is_file():
        return ()
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return tuple(
        Exemption(item["file"], str(item["literal"]), item["reason"], item.get("count"))
        for item in document.get("exempt", ())
    )


def has_posthoc_wording(unit: Unit) -> bool:
    """Whether the heading's text, or the intro of its section, says the analysis was post hoc."""
    return POSTHOC.search(unit.text) is not None or POSTHOC.search(unit.lead) is not None
