import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from oran_causal.trace.cli import Paths, build_context
from oran_causal.trace.paper import read_exemptions, scan_paper
from oran_causal.trace.rules import (
    Context,
    Status,
    bare_decimals,
    claim_tags_registered,
    claims_tagged,
    run_rules,
    values_generated,
)

type Workspace = Callable[..., Path]
type MakeClaim = Callable[..., dict[str, Any]]

ARTICLE = r"""\documentclass{article}
\begin{document}
\input{sections/s01}
\end{document}
"""
SECTION = r"""\section{Results}
\label{sec:x}\claim{EMP-A}
The rate was \val{a.rate} against 0.05, and $0.7$ in math, see \ref{sec:1.5}. % 9.99 comment
\subsection{Extra}
\label{sec:y}\claim{EMP-B}
This analysis is post hoc.
\begin{tabular}{l}
cell 3.14 \\
\end{tabular}
"""


def write_paper(root: Path, section: str = SECTION, values: dict[str, str] | None = None) -> None:
    paper = root / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "generated").mkdir()
    (paper / "sn-article.tex").write_text(ARTICLE, encoding="utf-8")
    (paper / "sections" / "s01.tex").write_text(section, encoding="utf-8")
    values = {"a.rate": "0.9"} if values is None else values
    lines = [rf"\defval{{{key}}}{{{text}}}" for key, text in values.items()]
    (paper / "generated" / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sidecar = {
        key: {
            "text": text,
            "args": {"v": {"record": "set_a/r.json", "pointer": "/a", "value": 0.9}},
        }
        for key, text in values.items()
    }
    (paper / "generated" / "values.json").write_text(
        json.dumps({"values": sidecar}), encoding="utf-8"
    )


def paper_context(root: Path, **changes: Any) -> Context:
    return replace(build_context(Paths.under(root)), **changes)


def claims(new_claim: MakeClaim, **extra: Any) -> list[dict[str, Any]]:
    return [
        new_claim("EMP-A", kind="empirical", **extra),
        new_claim("EMP-B", kind="empirical", posthoc=True),
    ]


def test_scan_finds_tags_values_units_and_prose_decimals(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    root = workspace(claims(new_claim))
    write_paper(root)
    scan = scan_paper(root / "paper")
    assert scan is not None
    assert [claim_id for claim_id, _ in scan.claims] == ["EMP-A", "EMP-B"]
    assert [key for key, _ in scan.vals] == ["a.rate"]
    assert [item.literal for item in scan.decimals] == ["0.05"]
    assert scan.claim_units["EMP-B"][0].text.startswith(r"\subsection{Extra}")


def test_scan_without_a_manuscript_is_none(tmp_path: Path) -> None:
    assert scan_paper(tmp_path / "paper") is None


def test_paper_rules_are_not_checkable_without_a_manuscript(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    results = {r.code: f for r, f in run_rules(paper_context(workspace(claims(new_claim))))}
    for code in ("F8", "F9", "F10", "F11"):
        assert results[code].status is Status.NOT_CHECKABLE


def test_f8_flags_unknown_and_retracted_tags(workspace: Workspace, new_claim: MakeClaim) -> None:
    root = workspace([new_claim("EMP-A", kind="empirical"), new_claim("EMP-B", status="retracted")])
    write_paper(root)
    messages = claim_tags_registered(paper_context(root)).messages
    assert messages == ("sections/s01.tex:5: \\claim names retracted claim EMP-B",)
    write_paper_section = SECTION.replace("EMP-B", "EMP-Z")
    (root / "paper" / "sections" / "s01.tex").write_text(write_paper_section, encoding="utf-8")
    assert "unknown claim EMP-Z" in claim_tags_registered(paper_context(root)).messages[0]


def test_f9_requires_a_tag_for_empirical_and_theorem_claims(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    extra = [new_claim("THM-C", kind="theorem-paper"), new_claim("DEF-D")]
    root = workspace([*claims(new_claim), *extra])
    write_paper(root)
    finding = claims_tagged(paper_context(root))
    assert finding.messages == ("THM-C: not tagged with \\claim in the paper",)


def test_f10_flags_unknown_keys_and_disagreeing_sidecar(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    root = workspace(claims(new_claim))
    write_paper(root, values={})
    assert "{a.rate} is not in values.tex" in values_generated(paper_context(root)).messages[0]
    write_paper_again = root / "paper" / "generated"
    (write_paper_again / "values.tex").write_text(r"\defval{a.rate}{0.8}" + "\n", encoding="utf-8")
    (write_paper_again / "values.json").write_text(
        json.dumps({"values": {"a.rate": {"text": "0.9", "args": {}}}}), encoding="utf-8"
    )
    assert values_generated(paper_context(root)).messages == (
        "a.rate: values.tex and values.json disagree",
    )


def test_f10_checks_stored_values_against_the_record(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    root = workspace(claims(new_claim))
    write_paper(root)
    record = root / "data" / "records" / "set_a" / "r.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"a": 0.9}), encoding="utf-8")
    assert values_generated(paper_context(root)).status is Status.PASS
    record.write_text(json.dumps({"a": 0.8}), encoding="utf-8")
    assert "stored value differs" in values_generated(paper_context(root)).messages[0]


def test_f10_requires_evidence_values_to_be_generated(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    evidence = [{"record": "set_a/r.json", "sha256": "b" * 64, "values": ["a.rate", "a.gone"]}]
    root = workspace(claims(new_claim, evidence=evidence))
    write_paper(root)
    assert values_generated(paper_context(root)).messages == (
        "EMP-A: evidence value a.gone is not generated",
    )


def test_f11_flags_bare_decimals_and_stale_exemptions(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    root = workspace(claims(new_claim))
    write_paper(root)
    assert bare_decimals(paper_context(root)).messages == ("sections/s01.tex:3: bare decimal 0.05",)
    exempt = root / "spec" / "paper_exempt.yaml"
    exempt.write_text(
        "exempt:\n"
        "  - {file: sections/s01.tex, literal: '0.05', count: 1, reason: constant}\n"
        "  - {file: sections/s01.tex, literal: '0.07', count: 1, reason: constant}\n",
        encoding="utf-8",
    )
    assert bare_decimals(paper_context(root)).messages == (
        "sections/s01.tex: exemption for 0.07 is stale",
    )


def test_unreproduced_results_are_read_apart_from_constants(tmp_path: Path) -> None:
    path = tmp_path / "paper_exempt.yaml"
    path.write_text(
        "exempt:\n"
        "  - {file: sections/s01.tex, literal: '0.05', count: 1, reason: constant}\n"
        "unreproduced:\n"
        "  - {file: sections/s01.tex, literal: '0.14', reason: no record}\n",
        encoding="utf-8",
    )
    constant, result = read_exemptions(path)
    assert (constant.unreproduced, result.unreproduced) == (False, True)
    assert read_exemptions(tmp_path / "missing.yaml") == ()


@pytest.mark.parametrize("code", ["F5", "F7", "F12"])
def test_other_phases_rules_stay_pending(
    code: str, workspace: Workspace, new_claim: MakeClaim
) -> None:
    results = {r.code: f for r, f in run_rules(paper_context(workspace(claims(new_claim))))}
    assert results[code].status is Status.NOT_CHECKABLE
