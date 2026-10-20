"""The `shap_rerun/overlay` record tree: the derived files with the TreeSHAP `shap_gbdt` entries.

Each file of `shap_splice.OVERLAY_FILES` sits at its record-set path under
`data/records/shap_rerun/overlay/`; readers take it from there (`shap_splice.overlaid`:
`oran_causal.figures.paths.record`, `spec/values.toml`) and every other file as recorded. Written:

- `system_artifacts/decision/`: the spliced Study 1 analysis, the Study 2 discovery output, the
  Study 2 / 3 evaluations and verdicts, the Study 3 bounds and the map-quality listing;
- `system_artifacts/report_2026-10-02/figures_data.json` (`shap_splice.splice_fd`);
- the Study 1 PMRT analysis and the Study 2 discovery output at their run paths (X11, X13, X14);
- `posthoc_x9_x16/revision/`: X9 and X11-X14 recomputed on those inputs;
- `posthoc_x7/x7/x7_tables.json`: X7 with the `CS:shap_gbdt@plc` referee.
"""

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.posthoc import x7, x9, x11, x12, x13, x14
from oran_causal.analysis.posthoc.common import shard_files, sub_filter
from oran_causal.analysis.system import report_data as rd
from oran_causal.analysis.system import study2 as astudy2
from oran_causal.analysis.system import study3 as astudy3
from oran_causal.analysis.system.shap_splice import (
    OVERLAY,
    OVERLAY_FILES,
    ShapSources,
    merged_records,
    splice_fd,
    spliced_v4,
)
from oran_causal.analysis.system.stats import N_BOOT, load_jobs
from oran_causal.experiments.posthoc.tools import referee_tools
from oran_causal.experiments.system.shap_rerun import X7_ARMS, X7_REFS

DECISION = Path("system_artifacts", "decision")
FIGURES_DATA = Path("system_artifacts", "report_2026-10-02", "figures_data.json")
S1_PMRT = Path("study1_raw", "e6p-disc-v4an-pmrt-1", "out", "analysis_v4.json")
S2_DISC = Path("study2_raw", "e6p-conf-discan-2", "out", "disc_conf.json")
REVISION = Path("posthoc_x9_x16", "revision")
X7_DIR = Path("posthoc_x7", "x7")
X7_REPLACED = ("X7:sub|shadow", "X7:CS:shap_gbdt@dev|")


def overlay_root(records: Path) -> Path:
    """`<records>/shap_rerun/overlay`."""
    return records / OVERLAY


def _label(rel: Path) -> str:
    return f"{OVERLAY}/{rel.as_posix()}"


def _dump(obj: Any, path: Path, newline: str = "\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline=newline)
    return path


def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def map_quality_text(mq: Mapping[str, Mapping[str, Any]]) -> str:
    """`conf_map_quality.txt` of `report_data.map_quality` output (the recorded layout)."""
    lines = ["arm                  edges TRUE+ WRONG null indet | placebo-map edges"]
    for arm, c in mq.items():
        pl = "-" if c["placebo_edges"] is None else str(c["placebo_edges"])
        lines.append(
            f"{arm:<21}{c['edges']:>5}{c['true_right']:>6}{c['true_wrong']:>6}"
            f"{c['null']:>5}{c['indet']:>6} | {pl}"
        )
    return "\n".join(lines) + "\n"


def decision_files(records: Path, out: Path, fd: Mapping[str, Any]) -> list[Path]:
    """The spliced analyzer outputs under `out` (decision copies and run paths)."""
    src, shap = rd.Sources(records=records), ShapSources(records=records)
    study1 = rd.read_json(shap.study1)
    d = out / DECISION
    (e2, v2), (e3, v3) = shap.report(2), shap.report(3)
    mq = d / "conf_map_quality.txt"
    mq.parent.mkdir(parents=True, exist_ok=True)
    mq.write_text(map_quality_text(fd["option_a"]["map_quality"]), encoding="utf-8", newline="\r\n")
    return [
        _dump(
            spliced_v4(rd.read_json(src.decision / "v4_analysis.json"), study1),
            d / "v4_analysis.json",
        ),
        _dump(spliced_v4(rd.read_json(records / S1_PMRT), study1), out / S1_PMRT),
        _copy(shap.disc, d / "conf_disc_conf.json"),
        _copy(shap.disc, out / S2_DISC),
        _copy(e2, d / "conf_eval.json"),
        _copy(v2, d / "conf_verdict.json"),
        _copy(e3, d / "cs_eval.json"),
        _copy(v3, d / "cs_verdict.json"),
        _copy(shap.bounds, d / "cs_bounds.json"),
        mq,
    ]


def _studies(records: Path, out: Path) -> dict[str, x11.Study]:
    frozen = x11.studies(records)
    return {
        "S1": x11.Study("S1", out / S1_PMRT, _label(S1_PMRT), *_rest(frozen["S1"])),
        "S2": x11.Study("S2", out / S2_DISC, _label(S2_DISC), *_rest(frozen["S2"])),
    }


def _rest(s: x11.Study) -> tuple[Path, tuple[Path, ...]]:
    return s.job_inputs, s.kernels


def _merged(records: Path, study: int) -> tuple[Any, Any, int]:
    src, shap = rd.Sources(records=records), ShapSources(records=records)
    run, n, old, new, filt = (
        ("e6p-conf-eval-1", 12, "study2_maps.json", shap.maps_artifact, astudy2.EVAL_FILTER)
        if study == 2
        else (
            "e6p-cs-eval-1",
            20,
            "study3_certsafe.json",
            shap.certsafe_artifact,
            astudy3.job_filter("eval"),
        )
    )
    frozen = sorted((src.runs / run).glob("*/res_*.jsonl"), key=Path.as_posix)
    if len(frozen) != n:
        raise FileNotFoundError(f"{run}: {len(frozen)} shards, want {n}")
    new_files = shap.eval_files(study)
    art = rd.read_json(new)
    m = merged_records(frozen, new_files, filt, rd.read_json(src.artifact(old)), art)
    return m.R, art, len(frozen) + len(new_files)


def posthoc(records: Path, out: Path, fd: Mapping[str, Any]) -> list[Path]:
    """X9 and X11-X14 on the spliced inputs, written under `out/posthoc_x9_x16/revision`."""
    shap = ShapSources(records=records)
    d = out / REVISION
    study_set = _studies(records, out)
    x11.write(x11.run(records, study_set), d)
    x14.write(x14.run(records, study_set), d)
    x13_in = {k: (s.analysis, s.label) for k, s in study_set.items()}
    x13.write(x13.run(records, x13_in), d)
    x12.write(x12.analyse(rd.read_json(shap.disc), fd, referee_tools()), d)
    x4_tables = rd.read_json(records / "posthoc_x4" / "x4" / "x4_tables.json")
    R3, art3, n3 = _merged(records, 3)
    R2, art2, n2 = _merged(records, 2)
    (e2, _), (e3, _) = shap.report(2), shap.report(3)
    s3 = x9.study3_of(R3, art3, rd.read_json(e3), x4_tables, n3)
    s2 = x9.study2_of(R2, art2, rd.read_json(e2), n2)
    x9.write(x9.assemble(s3, s2), d)
    return sorted(p for p in d.iterdir() if p.is_file())


def x7_tables(records: Path, out: Path, n_boot: int = N_BOOT) -> Path:
    """X7 with the `CS:shap_gbdt@plc` referee: recorded jobs of the kept referees plus the new ones.

    The recorded `X7:sub|shadow` and `CS:shap_gbdt@dev` jobs are replaced by the jobs under
    `shap_rerun/x7`; the Study 3 records are the merged TreeSHAP evaluation records.
    """
    shap = ShapSources(records=records)
    filt = sub_filter(x7.SUB)
    Rx, heads = load_jobs(shard_files(records / X7_DIR / "records"), filt)
    for s in Rx:
        Rx[s] = {a: r for a, r in Rx[s].items() if not a.startswith(X7_REPLACED)}
    Rn, hn = load_jobs(shard_files(shap.root / "x7"), filt)
    for s, by_arm in Rn.items():
        Rx.setdefault(s, {}).update({a: r for a, r in by_arm.items() if a in X7_ARMS})
    Rs, _, _ = _merged(records, 3)
    E = rd.read_json(shap.report(3)[0])
    fac = rd.read_json(records / "posthoc_x1" / "exp_c" / "calib" / "factors.json")
    tables = x7.analyse_records(Rs, Rx, heads + hn, E, fac, refs=X7_REFS, n_boot=n_boot)
    return _dump(tables, out / X7_DIR / "x7_tables.json")


def write(records: Path, *, out: Path | None = None) -> list[Path]:
    """Writes the overlay tree under `out` (default `overlay_root(records)`); returns the files.

    Raises:
        RuntimeError: The written files are not `shap_splice.OVERLAY_FILES`.
    """
    out = overlay_root(records) if out is None else out
    src = rd.Sources(records=records)
    fd = splice_fd(rd.build(src), src, ShapSources(records=records))
    fd_path = out / FIGURES_DATA
    fd_path.parent.mkdir(parents=True, exist_ok=True)
    fd_path.write_text(rd.figures_json(fd), encoding="utf-8", newline=rd.FROZEN_NEWLINE)
    files = [fd_path, *decision_files(records, out, fd), *posthoc(records, out, fd)]
    files.append(x7_tables(records, out))
    written = {p.relative_to(out).as_posix() for p in files}
    if written != OVERLAY_FILES:
        raise RuntimeError(
            f"overlay files differ from OVERLAY_FILES: extra {sorted(written - OVERLAY_FILES)}, "
            f"missing {sorted(OVERLAY_FILES - written)}"
        )
    return files


def main(argv: Sequence[str] | None = None) -> int:
    """Command line (module docstring)."""
    ap = argparse.ArgumentParser(prog="python -m oran_causal.experiments.system.shap_overlay")
    ap.add_argument("--records", type=Path, default=rd.RECORDS)
    a = ap.parse_args(argv)
    for p in write(a.records):
        print(p.relative_to(a.records).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
