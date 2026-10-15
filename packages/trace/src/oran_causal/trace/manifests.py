"""Reader for `data/manifests/<set>.sha256` files (`<sha256>  <path>` per line)."""

from pathlib import Path

type Manifests = dict[str, dict[str, str]]


def _parse_line(line: str) -> tuple[str, str] | None:
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None
    digest, path = parts
    return path.removeprefix("*"), digest.lower()


def read_manifests(directory: Path) -> Manifests | None:
    """Read every manifest in a directory.

    Args:
        directory: The manifests directory.

    Returns:
        A map from set name to a map from record path to sha256, or `None`
        when the directory does not exist.
    """
    if not directory.is_dir():
        return None
    manifests: Manifests = {}
    for file in sorted(directory.glob("*.sha256")):
        lines = file.read_text(encoding="utf-8").splitlines()
        manifests[file.stem] = dict(entry for line in lines if (entry := _parse_line(line)))
    return manifests
