"""Locations of the frozen data files shipped in the experiments member root."""

from pathlib import Path

MEMBER_ROOT = Path(__file__).resolve().parents[3]
PROTOCOLS = MEMBER_ROOT / "protocols"
SPECS = MEMBER_ROOT / "specs"
ARTIFACTS = MEMBER_ROOT / "artifacts"
MANIFESTS = MEMBER_ROOT / "manifests"
WORKSPACE_ROOT = MEMBER_ROOT.parents[1]
STUDY4_PROTOCOL = PROTOCOLS / "study4" / "PROTOCOL_A.md"
EVAL_COST_TABLE = ARTIFACTS / "eval_cost_table.json"
