"""Study 4 report: merged campaign records to `REPORT.md`, `report.json` and figure-ready CSVs.

The report carries every output the protocol pre-registers: the claim and C1, C2a, C2b, C3
with their sensitivity lines, the unfiltered eq-arm table and named-arm sentence, the like-for-
like headline, power among cells not INVALID with the R4 "not applicable" rows, E4, the kappa
sweep, cost and integrity. All numbers come from the analysis modules (same definitions,
nothing tuned); this module drives them arm by arm and lays the outputs out.

Records are split by arm into a temporary directory in one streaming pass and each arm is
screened and summarised on its own; the verdicts, tables and integrity run once on the merged
cells. The result equals `analyse` on the same records.

DEV mode renders DEV runs on the DEV specs with the analysis fields of the EVAL spec and
watermarks every output; it is a rehearsal, never an EVAL result.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis import recorded
from oran_causal.analysis.controlled.cells import Cells, Rows, build_cells, screen
from oran_causal.analysis.controlled.fmax import Dependence, load_dependence
from oran_causal.analysis.controlled.integrity import (
    IntegrityInputs,
    integrity,
    load_amendments,
    protocol_check,
)
from oran_causal.analysis.controlled.protocol import MIN_FLAG_SEEDS, RATE_KEYS
from oran_causal.analysis.controlled.spec import Spec, seeds_of
from oran_causal.analysis.controlled.tables import Tables, assemble, fmt, markdown, to_json
from oran_causal.analysis.records import (
    PathLike,
    Record,
    arm_of,
    file_sha256,
    iter_lines,
    load_records,
)

REPORT_VERSION = "xm-eval-report/1"
DEV_MARK = (
    "DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of "
    "the "
    "report, not the pre-registered result"
)
SLIM_DROP = ("edges", "notes", "scores", "versions", "result_config", "config")
SHARED_FIELDS = ("code", "run_mode", "integrity", "pkgs", "host")
EVAL_MIN_SEED = 3_100_000
CELL = ("arm", "world", "regime", "lam", "n", "kappa")
Pins = Mapping[str, str]
PROTOCOL_NAME = "protocols/study4/PROTOCOL.md"


@dataclasses.dataclass(frozen=True)
class ReportRequest:
    """What to report and where to read and write it.

    Attributes:
        spec: EVAL spec path.
        merged: Merged record files.
        out_dir: Output directory.
        dev: Render a DEV rehearsal; needs `dev_specs`.
        dev_specs: DEV spec paths, united.
        freeze_commit: Commit the protocol was frozen at.
        amendments: Amendments file.
        protocol: Protocol file whose LF sha256 the spec records.
        dependence: Dependence table for `F_max`.
        dev_merged: DEV records for the tune reproducibility check (EVAL mode only).
        tmp_dir: Directory for the per-arm temporary files.
        generated_utc: Timestamp to record; the current time if None. Pin it to reproduce a
            frozen report.
        input_labels: Replacement labels for input paths in the recorded inputs, keyed by path.
        min_flag_seeds: Seeds below which a cell is `few_seeds` (DEV testing only).
        cite_recorded: Cite the spec and the protocol by the names and sha256 the stored records
            carry (`oran_causal.analysis.recorded`), to reproduce a stored report.
    """

    spec: PathLike
    merged: Sequence[PathLike]
    out_dir: PathLike
    dev: bool = False
    dev_specs: Sequence[PathLike] = ()
    freeze_commit: str | None = None
    amendments: PathLike | None = None
    protocol: PathLike | None = None
    dependence: PathLike | None = None
    dev_merged: Sequence[PathLike] = ()
    tmp_dir: PathLike | None = None
    generated_utc: str | None = None
    input_labels: Pins = dataclasses.field(default_factory=dict)
    min_flag_seeds: int = MIN_FLAG_SEEDS
    cite_recorded: bool = False


def split_by_arm(
    paths: Sequence[PathLike], tmp: Path, tag: str
) -> tuple[dict[str, Path], dict[str, int]]:
    """Streams each record line to `<tmp>/<tag>_<arm>.jsonl`; unparseable lines are skipped."""
    files: dict[str, Path] = {}
    handles: dict[str, Any] = {}
    counts: dict[str, int] = defaultdict(int)
    try:
        for path in paths:
            for line in iter_lines(path):
                try:
                    arm = arm_of(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if arm not in handles:
                    digest = hashlib.sha1(arm.encode(), usedforsecurity=False).hexdigest()[:12]
                    files[arm] = tmp / f"{tag}_{digest}.jsonl"
                    handles[arm] = files[arm].open("w", encoding="utf-8", newline="\n")
                handles[arm].write(line if line.endswith("\n") else line + "\n")
                counts[arm] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return files, dict(counts)


class Slimmer:
    """Drops the bulky fields of a record, sharing repeated sub-dicts between records."""

    def __init__(self) -> None:
        self._shared: dict[str, Any] = {}

    def __call__(self, record: Record) -> Record:
        """Returns the record without edges, notes and scores."""
        out = {}
        for key, value in record.items():
            if key not in SLIM_DROP:
                out[key] = self._slim_value(key, value)
        return out

    def _slim_value(self, key: str, value: Any) -> Any:
        if key == "integrity" and isinstance(value, dict):
            return self._share(key, {"spec_file_sha256": value.get("spec_file_sha256")})
        if isinstance(value, dict) and key in SHARED_FIELDS:
            return self._share(key, value)
        return value

    def _share(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        return self._shared.setdefault(f"{key}:{json.dumps(value, sort_keys=True)}", value)


def sub_spec(spec: Spec, arm: str) -> Spec:
    """Restricts a spec to one arm, keeping the focal and analysis fields."""
    blocks = []
    for block in spec["blocks"]:
        selected = block.get("arms", "all")
        if selected == "all" or arm in selected:
            blocks.append({**block, "arms": [arm]})
    return {**spec, "arms": {arm: spec["arms"][arm]}, "blocks": blocks}


def _dev_arms(eval_spec: Spec, dev_specs: list[Spec], notes: list[str]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for s in dev_specs:
        for name, dev in s["arms"].items():
            if name not in eval_spec["arms"]:
                notes.append(f"DEV arm {name} is not in the EVAL spec: left out")
                continue
            ev = eval_spec["arms"][name]
            if (dev.get("ref"), dev.get("config")) != (ev.get("ref"), ev.get("config")):
                notes.append(
                    f"DEV arm {name}: ref / config differ from the EVAL spec (DEV {dev.get('ref')} "
                    f"{dev.get('config')}); EVAL analysis fields used"
                )
            arms[name] = {
                **ev,
                **{k: dev[k] for k in ("ref", "config", "declare", "worlds") if k in dev},
            }
    return arms


def _dev_blocks(dev_specs: list[Spec], arms: dict[str, Any]) -> list[Spec]:
    blocks = []
    for s in dev_specs:
        for block in s["blocks"]:
            selected = block.get("arms", "all")
            names = [a for a in (list(s["arms"]) if selected == "all" else selected) if a in arms]
            if names:
                blocks.append({**block, "arms": names})
    return blocks


def dev_spec(eval_spec: Spec, dev_specs: list[Spec]) -> tuple[Spec, list[str]]:
    """Builds the DEV rendering spec: DEV arms and blocks with the analysis fields of the EVAL spec.

    If the EVAL focal arm has no DEV arm, the protocol's T10 fallback (pmrt_eq focal and
    primary) applies for this rendering only.

    Args:
        eval_spec: The EVAL spec.
        dev_specs: The DEV specs.

    Returns:
        The rendering spec and the notes explaining every deviation.

    Raises:
        SystemExit: If neither the EVAL focal arm nor `pmrt_eq` has DEV records.
    """
    notes: list[str] = []
    arms = _dev_arms(eval_spec, dev_specs, notes)
    focal = eval_spec.get("focal")
    out: Spec = {
        "name": "DEV:" + "+".join(str(s.get("name")) for s in dev_specs),
        "arms": arms,
        "blocks": _dev_blocks(dev_specs, arms),
        "budget_cpu_s": eval_spec.get("budget_cpu_s"),
        "fmax_dependence_sha256": eval_spec.get("fmax_dependence_sha256"),
        "protocol_sha256": eval_spec.get("protocol_sha256"),
        "focal": focal,
        "not_in_grid": (eval_spec.get("not_in_grid") or [])
        + [
            {
                "ruling": "DEV scope",
                "arms": list(arms),
                "regimes": None,
                "ns": None,
                "kappas": None,
                "reason": "the DEV specs ran a subset of the EVAL grid (DEV rendering only)",
            }
        ],
    }
    if focal not in arms:
        if "pmrt_eq" not in arms:
            raise SystemExit(f"DEV specs have neither the EVAL focal arm {focal} nor pmrt_eq")
        out["focal"] = "pmrt_eq"
        arms["pmrt_eq"] = {k: v for k, v in arms["pmrt_eq"].items() if k != "label"} | {
            "analysis": "primary"
        }
        notes.append(
            f"{focal} has no DEV records: PROTOCOL_A T10 fallback applied for this DEV rendering "
            f"only "
            "(pmrt_eq focal and primary, label dropped)"
        )
    missing = sorted(set(eval_spec["arms"]) - set(arms))
    if missing:
        notes.append(
            "EVAL arms without DEV records (absent from this rendering): " + ", ".join(missing)
        )
    return out, notes


def relabel_tune(records: list[Record], seeds: set[int], spec: Spec) -> tuple[list[Record], Spec]:
    """Relabels the records and spec blocks of `seeds` as tune. DEV pilot testing only.

    Raises:
        SystemExit: On an EVAL spec.
    """
    if str(spec.get("name", "")).lower().startswith("eval"):
        raise SystemExit("--as-tune is for DEV pilot testing only; refused on an EVAL spec")
    blocks = []
    for block in spec["blocks"]:
        block_seeds = seeds_of(block["seeds"])
        if block["role"] == "measure" and set(block_seeds) & seeds:
            blocks.append({**block, "role": "tune", "seeds": sorted(set(block_seeds) & seeds)})
            rest = sorted(set(block_seeds) - seeds)
            if rest:
                blocks.append({**block, "seeds": rest})
        else:
            blocks.append(block)
    relabelled = [{**r, "role": "tune"} if int(r["job"]["seed"]) in seeds else r for r in records]
    return relabelled, {**spec, "blocks": blocks}


def merge_integrity(per_arm: list[dict[str, Any]], glob: dict[str, Any]) -> dict[str, Any]:
    """Combines the cross-arm checks with the record-content checks summed over the per-arm runs.

    Args:
        per_arm: Integrity results of the per-arm runs.
        glob: Integrity result on the slim records of all arms.

    Returns:
        The merged V11 with its label recomputed.
    """
    merged = dict(glob)
    incomplete = [k for x in per_arm for k in x["candidates_incomplete"]]
    merged["n_candidates_incomplete"] = sum(x["n_candidates_incomplete"] for x in per_arm)
    merged["candidates_incomplete"] = incomplete[:20]
    disagree = [k for x in per_arm for k in x["by_recheck"]["disagree_first"]]
    merged["by_recheck"] = {
        "records": sum(x["by_recheck"]["records"] for x in per_arm),
        "agree": sum(x["by_recheck"]["agree"] for x in per_arm),
        "disagree_first": disagree[:20],
    }
    repro = [
        x["tune_reproducibility_vs_dev"]
        for x in per_arm
        if x["tune_reproducibility_vs_dev"] is not None
    ]
    keys = ("compared", "dataset_sha_equal", "declarations_equal")
    merged["tune_reproducibility_vs_dev"] = (
        {k: sum(r[k] for r in repro) for k in keys} if repro else None
    )
    checks = dict(merged["checks"])
    checks["candidates_complete"] = merged["n_candidates_incomplete"] == 0
    merged["checks"] = checks
    merged["failed"] = sorted(k for k, ok in checks.items() if not ok)
    merged["label"] = "FINAL" if all(checks.values()) else "PROVISIONAL"
    return merged


@dataclasses.dataclass
class _Accumulator:
    cells: Cells = dataclasses.field(default_factory=dict)
    rows: Rows = dataclasses.field(default_factory=dict)
    slim: list[Record] = dataclasses.field(default_factory=list)
    per_arm: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    screening: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "expected": {},
            "expected_error": None,
            "duplicates": [],
            "unexpected": [],
            "role_mismatch": [],
        }
    )

    def absorb_screening(self, report: dict[str, Any]) -> None:
        """Adds one arm's screening report."""
        expected = self.screening["expected"]
        if report["expected"] is None:
            self.screening["expected"] = None
            self.screening["expected_error"] = report["expected_error"]
        elif expected is not None:
            expected.update(report["expected"])
        for key in ("duplicates", "unexpected", "role_mismatch"):
            self.screening[key] += report[key]


@dataclasses.dataclass(frozen=True)
class ChunkOptions:
    """Options of `chunked_analyse`.

    Attributes:
        dev_paths: DEV record files for the tune reproducibility check.
        tmp_dir: Directory for the per-arm temporary files.
        min_flag_seeds: Seeds below which a cell is `few_seeds`.
        log: Progress sink.
    """

    dev_paths: Sequence[PathLike] = ()
    tmp_dir: PathLike | None = None
    min_flag_seeds: int = MIN_FLAG_SEEDS
    log: Callable[[str], None] = print


@dataclasses.dataclass
class _Run:
    spec: Spec
    inputs: IntegrityInputs
    min_flag_seeds: int
    acc: _Accumulator = dataclasses.field(default_factory=_Accumulator)
    slimmer: Slimmer = dataclasses.field(default_factory=Slimmer)

    def analyse_arm(self, arm: str, path: Path | None, dev_path: Path | None) -> int:
        """Screens, summarises and checks one arm; returns its input record count."""
        sub = sub_spec(self.spec, arm)
        records = load_records(path) if path else []
        used, report = screen(records, sub)
        cells, rows = build_cells(used, sub, min_flag_seeds=self.min_flag_seeds)
        self.acc.cells.update(cells)
        self.acc.rows.update(rows)
        dev_records = None
        if self.inputs.dev_records is not None:
            dev_records = load_records(dev_path) if dev_path else []
        inputs = dataclasses.replace(self.inputs, dev_records=dev_records)
        self.acc.per_arm.append(integrity(used, report, sub, inputs))
        self.acc.absorb_screening(report)
        self.acc.slim += [self.slimmer(r) for r in used]
        return len(records)


def chunked_analyse(
    paths: Sequence[PathLike],
    spec: Spec,
    inputs: IntegrityInputs,
    options: ChunkOptions | None = None,
) -> tuple[Tables, dict[str, int]]:
    """Analyses the records arm by arm; the tables equal `analyse` on the same records.

    Args:
        paths: Merged record files.
        spec: Campaign spec.
        inputs: Integrity inputs; `dev_records` is set to an empty list when DEV files are given.
        options: DEV files, temporary directory, seed threshold and progress sink.

    Returns:
        The tables and the input record counts by arm.
    """
    options = options or ChunkOptions()
    tmp = Path(tempfile.mkdtemp(prefix="xm_eval_report_", dir=options.tmp_dir))
    try:
        files, counts_in = split_by_arm(paths, tmp, "rec")
        dev_files = split_by_arm(options.dev_paths, tmp, "dev")[0] if options.dev_paths else {}
        base = dataclasses.replace(inputs, dev_records=[] if options.dev_paths else None)
        run = _Run(spec, base, options.min_flag_seeds)
        for arm in sorted(set(files) - set(spec["arms"])):
            run.acc.screening["unexpected"] += [r.get("key") for r in load_records(files[arm])]
        for arm in sorted(spec["arms"]):
            started = time.time()
            n_records = run.analyse_arm(arm, files.get(arm), dev_files.get(arm))
            options.log(f"[report] {arm}: {n_records} records ({time.time() - started:.0f} s)")
        acc = run.acc
        for key in ("duplicates", "unexpected", "role_mismatch"):
            acc.screening[key] = sorted(acc.screening[key])
        glob = integrity(
            acc.slim, acc.screening, spec, dataclasses.replace(inputs, dev_records=None)
        )
        v11 = merge_integrity(acc.per_arm, glob)
        return assemble(acc.cells, acc.rows, acc.slim, spec, dep=inputs.dep, v11=v11), counts_in
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def analyse(
    records: list[Record],
    spec: Spec,
    inputs: IntegrityInputs,
    *,
    min_flag_seeds: int = MIN_FLAG_SEEDS,
) -> Tables:
    """Analyses all records at once: screen, cells, integrity, tables.

    Args:
        records: Campaign records.
        spec: Campaign spec.
        inputs: Integrity inputs.
        min_flag_seeds: Seeds below which a cell is `few_seeds`.

    Returns:
        The tables V0 to V11.
    """
    used, screened = screen(records, spec)
    cells, rows = build_cells(used, spec, min_flag_seeds=min_flag_seeds)
    v11 = integrity(used, screened, spec, inputs)
    return assemble(cells, rows, used, spec, dep=inputs.dep, v11=v11)


def _rate_cols(prefix: str, x: dict[str, Any] | None) -> dict[str, Any]:
    x = x or {}
    ci = x.get("ci") or [None, None]
    return {
        f"{prefix}_rate": x.get("rate"),
        f"{prefix}_lo": ci[0],
        f"{prefix}_hi": ci[1],
        f"{prefix}_validity": x.get("validity"),
    }


def _mean_cols(prefix: str, x: dict[str, Any] | None) -> dict[str, Any]:
    x = x or {}
    ci = x.get("ci") or [None, None]
    return {
        f"{prefix}_mean": x.get("mean"),
        f"{prefix}_sd": x.get("sd"),
        f"{prefix}_lo": ci[0],
        f"{prefix}_hi": ci[1],
    }


def _pick(row: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {f: row.get(f) for f in fields}


def _csv_validity(out: Tables) -> list[dict[str, Any]]:
    fields = (*CELL, "block", "declare", "status", "validity", "n_seeds")
    return [
        {
            **_pick(r, fields),
            **{k: v for rk in RATE_KEYS for k, v in _rate_cols(rk, r.get(rk)).items()},
        }
        for r in out["V1_validity"]
    ]


def _csv_recall(out: Tables) -> list[dict[str, Any]]:
    fields = (*CELL[:5], "block", "state", "power_not_applicable")
    return [
        {
            **_pick(r, fields),
            "powered": r["recall"] is not None,
            **_mean_cols("recall", r["recall"]),
            **_mean_cols("fdp", r["fdp"]),
            **_rate_cols("sign_acc", r["sign_acc"]),
        }
        for r in out["V2_recall"]
    ]


def _csv_like(out: Tables) -> list[dict[str, Any]]:
    fields = (
        *CELL,
        "rule",
        "declare",
        "state",
        "n_seeds",
        "power_not_applicable",
        "placebo_is_tuning_column",
    )
    return [
        {
            **_pick(r, fields),
            **_mean_cols("recall", r["recall"]),
            **_rate_cols("null", r["null"]),
            **_rate_cols("placebo", r["placebo"]),
            **_rate_cols("placebo_conf", r["placebo_conf"]),
        }
        for r in out["V0_like_for_like"]
    ]


def _csv_cells(out: Tables) -> dict[str, list[dict[str, Any]]]:
    paired = (*CELL, "rule", "block", "descriptive_only", "n_seeds", "mean_d", "sd_d")
    info = (*CELL, "block", "status", "validity")
    e4 = (*CELL, "block", "status", "validity", "power_not_applicable", "wrong_sign_rate")
    return {
        "paired_diff": [
            {**_pick(r, paired), "lo": r["ci"][0], "hi": r["ci"][1]} for r in out["V3_paired"]
        ],
        "information_levels_R2": [
            {
                **_pick(r, info),
                **_mean_cols("recall", r.get("recall")),
                **_rate_cols("null_raw", r.get("null_raw")),
                **_rate_cols("null_decl", r.get("null_decl")),
            }
            for r in out["V5_information_R2"]
        ],
        "e4_by_lambda": [
            {
                **_pick(r, e4),
                **_mean_cols("recall", r.get("recall")),
                **_rate_cols("plac_raw", r.get("plac_raw")),
                **_rate_cols("conf_raw", r.get("conf_raw")),
                **_rate_cols("conf_decl", r.get("conf_decl")),
            }
            for r in out["V6_E4"]
        ],
        "kappa_sweep": [
            {**_pick(r, (*CELL, "status", "validity")), **_mean_cols("recall", r["recall"])}
            for r in out["V9_kappa_sweep"]
        ],
    }


def _csv_other(out: Tables) -> dict[str, list[dict[str, Any]]]:
    cost = (
        "arm",
        "world",
        "regime",
        "n",
        "budget",
        "n_runs",
        "cpu_s_mean",
        "cpu_s_max",
        "wall_s_mean",
        "wall_s_max",
        "peak_rss_mb_max",
        "gpu_s_mean",
        "n_infeasible",
        "n_errors",
        "t3",
    )
    eq_fields = (
        "membership",
        "native",
        "native_c1",
        "verdict",
        "n_cells",
        "n_invalid",
        "f_max",
        "r2_invalid",
        "r2_planned",
    )
    return {
        "cost": [_pick(r, cost) for r in out["V10_cost"]],
        "not_in_grid": [
            _pick(c, (*CELL, "ruling")) for c in (out.get("not_in_grid") or {}).get("cells", [])
        ],
        "eq_arms": [
            {
                "arm": a,
                **_pick(x, eq_fields),
                **_rate_cols("r1_null_raw", (x.get("pooled_r1") or {}).get("null_raw")),
                **_rate_cols("r2_null_raw", (x.get("pooled_r2") or {}).get("null_raw")),
            }
            for a, x in out["V4_verdicts"].get("eq_arms_table", {}).items()
        ],
    }


def csv_tables(out: Tables) -> dict[str, list[dict[str, Any]]]:
    """Builds the figure-ready long tables: one row per arm x cell (x rule)."""
    return {
        "validity_cells": _csv_validity(out),
        "recall_vs_n": _csv_recall(out),
        "like_for_like": _csv_like(out),
        **_csv_cells(out),
        **_csv_other(out),
    }


def write_csvs(
    tables: dict[str, list[dict[str, Any]]], directory: PathLike, *, dev: bool
) -> dict[str, str]:
    """Writes one CSV per table, the first column `data_mode` marking `DEV-NOT-EVAL` or `EVAL`.

    Returns:
        The LF sha256 of every file by relative path.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    mode = "DEV-NOT-EVAL" if dev else "EVAL"
    shas = {}
    for name, rows in tables.items():
        path = target / f"{name}.csv"
        columns = ["data_mode", *dict.fromkeys(k for r in rows for k in r)]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {"data_mode": mode, **{k: ("" if v is None else v) for k, v in r.items()}}
                )
        shas[f"csv/{name}.csv"] = file_sha256(path)
    return shas


def _rt(x: dict[str, Any] | None) -> str:
    if not x or x.get("rate") is None:
        return "-"
    ci = x.get("ci") or [None, None]
    return (
        f"{fmt(x['rate'], 3)} [{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] {x.get('validity') or ''}".strip()
    )


def _comp_row(name: str, x: dict[str, Any]) -> str:
    pooled = "; ".join(f"{k} {_rt(p)}" for k, p in (x.get("pooled") or {}).items())
    return (
        f"| {name} | {x['verdict']} | {x.get('n_cells', '-')}/{x.get('n_planned', '-')} | "
        f"{x.get('n_invalid', '-')} ({x.get('f_max', '-')}) | {pooled} |"
    )


COMP_HEAD = [
    "| arm | verdict | cells counted / planned | INVALID (F_max) | pooled rates |",
    "|---|---|---|---|---|",
]


def _md_header(out: Tables, meta: dict[str, Any], mark: list[str], protocol_name: str) -> list[str]:
    dev = meta["mode"] == "DEV"
    v11 = out["V11_integrity"]
    protocol = v11["protocol"]
    if protocol.get("ok"):
        ptxt = "frozen, sha matches"
    else:
        ptxt = (
            f"NOT verified: frozen {protocol.get('frozen')}, spec protocol_sha256 "
            f"{str(protocol.get('spec_protocol_sha256'))[:16]} vs file "
            f"{str(protocol.get('file_sha256'))[:16]}"
            + (f" ({protocol['why']})" if protocol.get("why") else "")
        )
    title = "# [DEV, NOT EVAL] Study A report rehearsal" if dev else "# Study A EVAL report"
    failed = f" (failed: {', '.join(v11['failed'])})" if v11["failed"] else ""
    lines = [
        title,
        "",
        *mark,
        f"Generator `{REPORT_VERSION}` on `{out['analysis']}`; spec `{out['spec_name']}` (sha256 "
        f"{(meta['spec_sha256'] or '-')[:16]}); protocol {protocol_name} ({ptxt}); "
        f"integrity **{v11['label']}**{failed}.",
        f"Primary PMRT arm (focal): {out['focal']}; primary kappa {out['primary_kappa']}; "
        f"generated {meta['utc']}.",
        "",
        "Inputs:",
        "",
    ]
    lines += [f"- `{p}`: sha256 {s[:16]}" for p, s in meta["inputs"]]
    by_arm = ", ".join(f"{a} {n}" for a, n in sorted(meta["records_by_arm"].items()))
    lines.append(f"- records by arm: {by_arm}")
    lines += [f"- note: {x}" for x in meta.get("notes", [])]
    return lines


def _md_claim(out: Tables, mark: list[str]) -> list[str]:
    v4 = out["V4_verdicts"]
    comp = v4["components"]
    return [
        "",
        "## 1. Claim (PROTOCOL_A s.10)",
        "",
        *mark,
        f"**Claim: {v4['claim']}**",
        "",
        "| component | verdict | wording |",
        "|---|---|---|",
        f"| C1 design-blind tests invalid on R2 | {comp['C1']} | {v4['C1']['wording']} |",
        f"| C2a design-based test valid (R1 + R2) | {comp['C2a']} | {v4['C2_wording']} |",
        f"| C2b design covariates restore validity | {comp['C2b']} | (C2 wording above) |",
        f"| C3 valid under the logged confounded policy | {comp['C3']} | "
        f"{v4['C3'].get('wording', '-')} |",
    ]


def _md_c1(v4: dict[str, Any], mark: list[str], tag: Callable[[str], str]) -> list[str]:
    c1 = v4["C1"]
    lines = [
        "",
        "## 2. C1: design-blind arms (set D)",
        "",
        *mark,
        f"**{c1['verdict']}**: {c1['n_failures']} of {len(c1['D_counted'])} assessable arms are "
        f"FAILURES "
        f"(|D| = {len(c1['D'])}; >= 3 assessable needed).",
        "",
        "| arm | in D | verdict | R2 INVALID / planned (counted) | pooled R2 truth-null raw | "
        "pooled R2 placebo raw | "
        "R1 INVALID / counted (F_max) |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, x in [*c1["arms"].items(), *c1.get("descriptive", {}).items()]:
        in_d = "yes" if a in c1["arms"] else "no (descriptive)"
        lines.append(
            f"| {a}{tag(a)} | {in_d} | {x['verdict']} | {x['r2_invalid']}/{x['r2_planned']} "
            f"({x['r2_counted']}) | {_rt(x['pooled_r2'].get('null_raw'))} | "
            f"{_rt(x['pooled_r2'].get('plac_raw'))} | {x['r1_invalid']}/{x['r1_counted']} "
            f"({x['r1_f_max']}) |"
        )
    sw = c1.get("sensitivity_without")
    if sw:
        text = (
            f"without {', '.join(sw['dropped'])}: **{sw['verdict']}** ({sw['n_failures']} of "
            f"{len(sw['D_counted'])} assessable arms fail)."
        )
    else:
        text = "no arm flagged `c1_sensitivity_drop` in this spec."
    return [*lines, "", "C1 sensitivity (R-56, no effect): " + text]


def _md_c2a(out: Tables, mark: list[str], tag: Callable[[str], str]) -> list[str]:
    v4 = out["V4_verdicts"]
    lines = [
        "",
        "## 3. C2a: the design-based test (R1 + R2)",
        "",
        *mark,
        *COMP_HEAD,
        _comp_row(f"{out['focal']} (primary)", v4["C2a"]),
    ]
    lines += [
        _comp_row(f"{a}{tag(a)} (secondary PMRT arm, no effect)", x["C2a"])
        for a, x in v4["pmrt_secondary"].items()
    ]
    return lines


def _md_c2b(out: Tables, mark: list[str], tag: Callable[[str], str]) -> list[str]:
    v4 = out["V4_verdicts"]
    c2b = v4["C2b"]
    w2 = c2b.get("with_r1_failing_partners") or {}
    lines = [
        "",
        "## 4. C2b: design-covariate adjustment (eq arms whose native partner is a C1 FAILURE)",
        "",
        *mark,
        f"**{c2b['verdict']}** (membership rule as frozen); with the eq arms whose native partner "
        f"is INVALID "
        f"IN R1 but meets C1 legs (i)-(ii) added (R-56 sensitivity, no effect): "
        f"**{w2.get('verdict', '-')}** "
        f"(added: {', '.join(w2.get('added', [])) or 'none'}).",
        "",
        *COMP_HEAD,
    ]
    lines += [_comp_row(f"{a}{tag(a)} (native {x['native']})", x) for a, x in c2b["arms"].items()]
    lines += [
        _comp_row(f"{a}{tag(a)} EXCLUDED (R-40), no effect", x)
        for a, x in c2b.get("excluded", {}).items()
    ]
    lines += [_comp_row(f"{a} (R-56 sensitivity member)", x) for a, x in w2.get("arms", {}).items()]
    lines += [
        "",
        f"C2 wording: {v4['C2_wording']}.",
        "",
        "Pre-registered disclosure (R-56; named eq arms):",
    ]
    lines += [f"- {x['sentence']}." for x in v4.get("C2_named", [])] or [
        "- none (no eq arm meets the naming rule)."
    ]
    lines += [
        "",
        "Every eq arm, same rule on R1 + R2, unfiltered (R-56):",
        "",
        "| eq arm | C2b membership | native C1 | verdict | INVALID / cells (F_max) | R2 INVALID / "
        "planned | "
        "pooled R1 truth-null raw | pooled R2 truth-null raw |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a, x in v4.get("eq_arms_table", {}).items():
        r2 = "-" if x["r2_invalid"] is None else f"{x['r2_invalid']}/{x['r2_planned']}"
        lines.append(
            f"| {a}{tag(a)} | {x['membership']} | {x['native_c1'] or '-'} | {x['verdict']} | "
            f"{x['n_invalid']}/{x['n_cells']} ({x['f_max']}) | {r2} | "
            f"{_rt((x['pooled_r1'] or {}).get('null_raw'))} | "
            f"{_rt((x['pooled_r2'] or {}).get('null_raw'))} |"
        )
    return lines


def _md_c3(out: Tables, mark: list[str], tag: Callable[[str], str]) -> list[str]:
    v4 = out["V4_verdicts"]
    lines = [
        "",
        "## 5. C3: logged confounded policy (E4 R3; C3 readers at S_E4)",
        "",
        *mark,
        *COMP_HEAD,
        _comp_row(f"{out['focal']} (primary)", v4["C3"]),
    ]
    lines += [
        _comp_row(f"{a}{tag(a)} (eq arm, no claim effect)", x) for a, x in v4["C3_eq_arms"].items()
    ]
    lines += [
        _comp_row(f"{a}{tag(a)} (secondary PMRT arm, no effect)", x["C3"])
        for a, x in v4["pmrt_secondary"].items()
    ]
    return lines


def _md_power(out: Tables, mark: list[str]) -> list[str]:
    not_applicable = [r for r in out["V2_recall"] if r["power_not_applicable"]]
    powered_rows = [r for r in out["V2_recall"] if r["recall"] is not None]
    lines = [
        "",
        "## 6. Power among cells not INVALID (V2) and the R4 'not applicable' rows",
        "",
        *mark,
        f"{len(powered_rows)} (arm, cell) with recall (counted, not INVALID; R-39); the grid is in "
        f"the "
        "appendix (V2) and `csv/recall_vs_n.csv`. PMRT arms in R4 (no known design): power NOT "
        "APPLICABLE, "
        "never recall 0 (R-42); their placebo rates are still read:",
        "",
        "| arm | cell | state | placebo raw (P_placebo) |",
        "|---|---|---|---|",
    ]
    v1 = {r["key"]: r for r in out["V1_validity"]}
    lines += [
        f"| {r['arm']} | {r['key'].split('|', 1)[1].replace('|', ' ')} | NA ({r['state']}) | "
        f"{_rt((v1.get(r['key']) or {}).get('plac_raw'))} |"
        for r in not_applicable
    ]
    return lines


def _md_kappa_and_grid(out: Tables, mark: list[str]) -> list[str]:
    lines = [
        "",
        "## 7. kappa sweep (R2, n 1000; V9)",
        "",
        *mark,
        "| arm | world | kappa | status | validity | recall |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['arm']} | {r['world']} | {r['kappa']:g} | {r['status']} | {r['validity']} | "
        f"{fmt((r['recall'] or {}).get('mean'))} |"
        for r in sorted(out["V9_kappa_sweep"], key=lambda r: (r["arm"], r["world"], r["kappa"]))
    ]
    grid = out.get("not_in_grid") or {}
    lines += [
        "",
        "## 8. Not in grid (pre-registered grid choices, R-54 / R-58)",
        "",
        *mark,
        "Cells outside an arm's grid are not planned: they never count as missing, never cap a "
        "verdict at "
        "PARTIAL and never make a component NOT EVALUABLE; every table labels them `nig`.",
        "",
        "| ruling | arms | where | cells | reason |",
        "|---|---|---|---|---|",
    ]
    for r in grid.get("by_rule", []):
        where = "; ".join(f"{k} {v}" for k, v in r["where"].items() if v) or "all"
        lines.append(
            f"| {r['ruling']} | {', '.join(r['arms'])} | {where} | {r['n_cells']} | {r['reason']} |"
        )
    unexplained = grid.get("unexplained")
    gaps = (
        ", ".join(unexplained[:10]) + (" ..." if len(unexplained) > 10 else "")
        if unexplained
        else "none"
    )
    lines += ["", f"{grid.get('n_cells', 0)} cells in total; unexplained gaps: {gaps}."]
    return lines


def report_md(out: Tables, meta: dict[str, Any], protocol_name: str = PROTOCOL_NAME) -> str:
    """Renders `REPORT.md`: the pre-registered sections and the appendix with every table.

    Args:
        out: Tables from `assemble`.
        meta: Report metadata (`mode`, `utc`, `spec_sha256`, `inputs`, `records_by_arm`,
            `notes`, `csv`).
        protocol_name: The protocol's name in the header.
    """
    dev = meta["mode"] == "DEV"
    mark = [f"> **{DEV_MARK}**", ""] if dev else []
    labels = out["V4_verdicts"].get("arm_labels", {})

    def tag(arm: str) -> str:
        return f" [{labels[arm]}]" if arm in labels else ""

    lines = _md_header(out, meta, mark, protocol_name)
    lines += _md_claim(out, mark)
    lines += _md_c1(out["V4_verdicts"], mark, tag)
    lines += _md_c2a(out, mark, tag)
    lines += _md_c2b(out, mark, tag)
    lines += _md_c3(out, mark, tag)
    lines += _md_power(out, mark)
    lines += _md_kappa_and_grid(out, mark)
    lines += ["", "## 9. Figure-ready CSVs", "", *mark]
    lines += [f"- `{p}` (sha256 {s[:16]})" for p, s in sorted(meta["csv"].items())]
    lines += ["", "## Appendix: every eval_analysis table (V0-V11)", "", *mark]
    text = markdown(out)
    body = text.split("\n", 3)[3] if "\n" in text else ""
    lines.append(body.replace("\n## ", "\n### "))
    if dev:
        lines += ["", f"> **{DEV_MARK}**"]
    return "\n".join(lines).rstrip("\n") + "\n"


def _read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_spec(request: ReportRequest) -> tuple[Spec, str, list[str]]:
    eval_spec = _read_json(request.spec)
    if request.dev:
        if not request.dev_specs:
            raise SystemExit("--dev needs --dev-spec")
        if request.dev_merged:
            raise SystemExit("--dev-merged is the EVAL tune reproducibility check; not in --dev")
        spec, notes = dev_spec(eval_spec, [_read_json(p) for p in request.dev_specs])
        return spec, hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), notes
    if request.dev_specs:
        raise SystemExit("--dev-spec needs --dev")
    if request.min_flag_seeds != MIN_FLAG_SEEDS:
        raise SystemExit("--min-flag-seeds is for DEV testing only; refused on an EVAL spec")
    seeds = [
        int(x)
        for b in eval_spec["blocks"]
        if b["role"] == "measure"
        for x in (b["seeds"] if isinstance(b["seeds"], list) else [])
    ]
    if any(x < EVAL_MIN_SEED for x in seeds):
        raise SystemExit(
            "EVAL mode needs an EVAL spec (measure seeds >= 3_100_000); use --dev for DEV data"
        )
    return eval_spec, file_sha256(request.spec), []


def _input_meta(request: ReportRequest, path: PathLike) -> tuple[str, str]:
    cited = recorded.lookup(path) if request.cite_recorded else None
    if cited is not None:
        return cited.name, cited.sha256
    return request.input_labels.get(str(path), str(path).replace("\\", "/")), file_sha256(path)


def _inputs_meta(request: ReportRequest) -> list[tuple[str, str]]:
    paths = [request.spec, *request.dev_specs, *request.merged]
    return [_input_meta(request, p) for p in paths]


def _cite_recorded(
    request: ReportRequest, spec: Spec, spec_sha: str, protocol: dict[str, Any]
) -> tuple[Spec, str, dict[str, Any], str]:
    cited = recorded.lookup(request.spec)
    if cited is None:
        raise SystemExit(f"{request.spec} is not a recorded input")
    proto = recorded.lookup(request.protocol) if request.protocol else None
    if proto is None:
        return spec, cited.sha256, protocol, PROTOCOL_NAME
    if protocol.get("ok"):
        protocol = {**protocol, "spec_protocol_sha256": proto.sha256, "file_sha256": proto.sha256}
    return {**spec, "protocol_sha256": proto.sha256}, cited.sha256, protocol, proto.name


def run_report(request: ReportRequest, log: Callable[[str], None] = print) -> int:
    """Writes `REPORT.md`, `report.json` and `csv/*.csv` for a merged record set.

    Args:
        request: Inputs, outputs and options.
        log: Progress sink.

    Returns:
        0 for a FINAL EVAL report, 2 otherwise (a DEV rehearsal is never final).
    """
    spec, spec_sha, notes = _resolve_spec(request)
    protocol = protocol_check(spec, request.protocol)
    protocol_name = PROTOCOL_NAME
    if request.cite_recorded:
        spec, spec_sha, protocol, protocol_name = _cite_recorded(request, spec, spec_sha, protocol)
    dep: Dependence | None = load_dependence(request.dependence)
    inputs = IntegrityInputs(
        freeze_commit=request.freeze_commit,
        spec_sha=spec_sha,
        protocol=protocol,
        amendments=load_amendments(request.amendments),
        dep=dep,
    )
    options = ChunkOptions(request.dev_merged, request.tmp_dir, request.min_flag_seeds, log)
    out, counts_in = chunked_analyse(request.merged, spec, inputs, options)
    mode = "DEV" if request.dev else "EVAL"
    target = Path(request.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    csv_sha = write_csvs(csv_tables(out), target / "csv", dev=request.dev)
    meta = {
        "mode": mode,
        "watermark": DEV_MARK if request.dev else None,
        "generator": REPORT_VERSION,
        "utc": request.generated_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec_sha256": spec_sha,
        "inputs": _inputs_meta(request),
        "records_by_arm": counts_in,
        "notes": notes,
        "csv": csv_sha,
    }
    payload = {"meta": meta, "spec": spec if request.dev else None, "tables": out}
    (target / "report.json").write_text(to_json(payload), encoding="utf-8", newline="\n")
    (target / "REPORT.md").write_text(
        report_md(out, meta, protocol_name), encoding="utf-8", newline="\n"
    )
    log(
        f"[report] {mode} {out['V11_integrity']['label']}: claim {out['V4_verdicts']['claim']} "
        f"{out['V4_verdicts']['components']} -> {request.out_dir}"
    )
    return 0 if (not request.dev and out["V11_integrity"]["label"] == "FINAL") else 2


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser of the report."""
    ap = argparse.ArgumentParser(prog="controlled-report")
    ap.add_argument("--spec", required=True, help="EVAL spec")
    ap.add_argument("--merged", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--dev", action="store_true", help="DEV rehearsal (watermarked); needs --dev-spec"
    )
    ap.add_argument("--dev-spec", nargs="+", default=())
    ap.add_argument("--freeze-commit", default=None)
    ap.add_argument("--amendments", default=None)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--dependence", default=None)
    ap.add_argument(
        "--dev-merged",
        nargs="+",
        default=(),
        help="EVAL: DEV records for the tune reproducibility check",
    )
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument(
        "--utc", default=None, help="timestamp to record (pin it to reproduce a frozen report)"
    )
    ap.add_argument(
        "--input-label",
        action="append",
        default=[],
        metavar="PATH=LABEL",
        help="label recorded for an input path instead of the path (repeatable)",
    )
    ap.add_argument("--min-flag-seeds", type=int, default=MIN_FLAG_SEEDS, help="DEV testing only")
    ap.add_argument(
        "--cite-recorded",
        action="store_true",
        help="cite the spec and protocol as the stored records do (reproduces a stored report)",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the report from the command line.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        The exit code of `run_report`.
    """
    a = build_parser().parse_args(argv)
    request = ReportRequest(
        spec=a.spec,
        merged=a.merged,
        out_dir=a.out,
        dev=a.dev,
        dev_specs=a.dev_spec,
        freeze_commit=a.freeze_commit,
        amendments=a.amendments,
        protocol=a.protocol,
        dependence=a.dependence,
        dev_merged=a.dev_merged,
        tmp_dir=a.tmp_dir,
        generated_utc=a.utc,
        input_labels=dict(x.split("=", 1) for x in a.input_label),
        min_flag_seeds=a.min_flag_seeds,
        cite_recorded=a.cite_recorded,
    )
    return run_report(request)
