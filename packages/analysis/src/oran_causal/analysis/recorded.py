"""The names and sha256 under which the stored records cite the experiment inputs.

`packages/experiments/manifests/recorded.json` maps each protocol, spec, artifact and registry
file of the experiments member, by its member-relative path, to the name and LF sha256 its
records carry (and, for a spec, the sha256 of its canonical JSON); it also lists the names and
sha256 pins of the analysis code the system artifacts record, and the names the stored reports
give the record files they read. Analyses of the stored records cite their inputs
by these names and hashes, and checks on stored records accept them next to the current file's
hash; new runs stamp the current files.
"""

import dataclasses
import functools
import hashlib
import json
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[4] / "experiments"
TABLE = EXPERIMENTS / "manifests" / "recorded.json"


@dataclasses.dataclass(frozen=True)
class Recorded:
    """How the stored records cite one input.

    Attributes:
        path: The input's path relative to the experiments member.
        name: The name the records carry.
        sha256: The LF sha256 the records carry.
        spec_sha256: For a spec, the sha256 of its canonical JSON the records carry.
    """

    path: str
    name: str
    sha256: str
    spec_sha256: str | None = None


def lf_sha256(path: str | Path) -> str:
    """sha256 of a file with CRLF read as LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def spec_sha256(path: str | Path) -> str:
    """sha256 of the canonical JSON of a spec file."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@functools.cache
def _table(path: Path = TABLE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inputs(path: Path = TABLE) -> dict[str, Recorded]:
    """Every recorded input, by member-relative path."""
    return {rel: Recorded(rel, **row) for rel, row in _table(path)["inputs"].items()}


def recorded(rel: str, path: Path = TABLE) -> Recorded:
    """The recorded citation of the input at member-relative path `rel`.

    Raises:
        KeyError: `rel` is not in the table.
    """
    return inputs(path)[rel]


def lookup(file: str | Path, member: Path = EXPERIMENTS, path: Path = TABLE) -> Recorded | None:
    """The recorded citation of `file`, or None if it is not a recorded input."""
    rel = _member_rel(Path(file), member)
    return None if rel is None else inputs(path).get(rel)


def code_sha256(path: Path = TABLE) -> dict[str, str]:
    """The recorded sha256 pins of the analysis code, by neutral name."""
    return {k: row["sha256"] for k, row in _table(path)["code"].items() if "sha256" in row}


def code_name(key: str, path: Path = TABLE) -> str:
    """The name the records give the analysis code with neutral name `key`."""
    return _table(path)["code"][key]["name"]


def record_label(rel: str, path: Path = TABLE) -> str:
    """The name the stored reports give the record file at `data/records`-relative path `rel`.

    Raises:
        KeyError: `rel` is not in the table.
    """
    return _table(path)["records"][rel]


def _member_rel(file: Path, member: Path) -> str | None:
    try:
        return file.resolve().relative_to(member.resolve()).as_posix()
    except ValueError:
        return None


def accepted_sha256(file: str | Path, member: Path = EXPERIMENTS, path: Path = TABLE) -> set[str]:
    """The LF sha256 values a record may carry for `file`: its current one and the recorded one."""
    out = {lf_sha256(file)}
    rel = _member_rel(Path(file), member)
    if rel is not None and rel in (table := inputs(path)):
        out.add(table[rel].sha256)
    return out


def accepted_spec_sha256(
    file: str | Path, member: Path = EXPERIMENTS, path: Path = TABLE
) -> set[str]:
    """The canonical-JSON sha256 values a record may carry for spec `file`."""
    out = {spec_sha256(file)}
    rel = _member_rel(Path(file), member)
    if rel is not None and (row := inputs(path).get(rel)) is not None and row.spec_sha256:
        out.add(row.spec_sha256)
    return out
