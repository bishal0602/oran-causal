"""Helpers of the Studies 2-3 decision figures: the evaluated arms of a study and the rows of
referee arms grouped by method.
"""

from oran_causal.figures import methods as M
from oran_causal.figures.figs import _studies as ST


def arms(study: str) -> dict:
    return ST.fd()[study]["eval"]["arms"]


def sections_rows(sections):
    """sections: [(banner, [arm codes])] -> method_rows input and the arm codes per block. Codes of
    one method (map thresholds @dev / @plc) form one block: the bold method name, then one sub-row
    per threshold.
    """
    rows, codes = [], []
    for banner, cs in sections:
        rows.append((banner, None))
        groups: dict[str, list[str]] = {}
        for c in cs:
            groups.setdefault(M.resolve(c).key, []).append(c)
        for members in groups.values():
            members = sorted(
                members, key=lambda c: ["dev", "plc", None].index(M.resolve(c).threshold)
            )
            rows.append((M.name(members[0]), members))
            codes.append(members)
    return rows, codes
