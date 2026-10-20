"""Studies 1-3 figure data with the `shap_gbdt` entries of the TreeSHAP baseline.

The recorded Study 1 and 2 `shap_gbdt` numbers came from GBDT permutation importance. The record
set `shap_rerun` holds the same baseline scored with TreeSHAP and the Study 2 / 3 referee jobs on
its maps; `ShapSources` names every file of it that the figures use. `spliced_v4` and `splice_fd`
replace the `shap_gbdt` entries of the recorded Study 1 analysis and of the asserted figure data
(`report_data.build`) and leave every other entry as recorded. `OVERLAY_FILES` lists the recorded
files that carry `shap_gbdt` entries; their TreeSHAP copies sit at the same path under `OVERLAY`.
`overlaid` is the one rule: the figures resolve every record through it, and each
`spec/values.toml` alias must equal `overlaid` of its recorded path (held by a test).
"""

import copy
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.system import report_data as rd
from oran_causal.analysis.system import study1, study2, study3
from oran_causal.analysis.system.stats import JobFilter, Record, Records, alias_fill, load_jobs

METHOD = "shap_gbdt"
POOLED_KEYS = (
    "ind_f1",
    "ind_precision",
    "ind_recall",
    "ov_precision",
    "ov_f1",
    "sign_acc",
    "n_declared",
    "chain_hits",
)

OVERLAY = "shap_rerun/overlay"
OVERLAY_FILES = frozenset(
    {
        "system_artifacts/report_2026-10-02/figures_data.json",
        "system_artifacts/decision/v4_analysis.json",
        "system_artifacts/decision/conf_disc_conf.json",
        "system_artifacts/decision/conf_eval.json",
        "system_artifacts/decision/conf_verdict.json",
        "system_artifacts/decision/cs_eval.json",
        "system_artifacts/decision/cs_verdict.json",
        "system_artifacts/decision/cs_bounds.json",
        "system_artifacts/decision/conf_map_quality.txt",
        "study1_raw/e6p-disc-v4an-pmrt-1/out/analysis_v4.json",
        "study2_raw/e6p-conf-discan-2/out/disc_conf.json",
        "posthoc_x9_x16/revision/x9_decision_e2e.json",
        "posthoc_x9_x16/revision/x9_decision_e2e.csv",
        "posthoc_x9_x16/revision/x11_gt_sensitivity.json",
        "posthoc_x9_x16/revision/x11_gt_sensitivity.csv",
        "posthoc_x9_x16/revision/x11_gt_labels.csv",
        "posthoc_x9_x16/revision/x12_threshold.json",
        "posthoc_x9_x16/revision/x12_threshold.csv",
        "posthoc_x9_x16/revision/x13_attribution.json",
        "posthoc_x9_x16/revision/x13_attribution.csv",
        "posthoc_x9_x16/revision/x14_existence.json",
        "posthoc_x9_x16/revision/x14_existence.csv",
        "posthoc_x7/x7/x7_tables.json",
    }
)

__all__ = [
    "METHOD",
    "OVERLAY",
    "OVERLAY_FILES",
    "Merged",
    "ShapSources",
    "changed_jobs",
    "frozen_twins",
    "merged_records",
    "overlaid",
    "splice_fd",
    "spliced_v4",
]


def overlaid(rel: str) -> str:
    """Records-relative path read for the recorded file `rel`: its `OVERLAY` copy if listed."""
    return f"{OVERLAY}/{rel}" if rel in OVERLAY_FILES else rel


@dataclasses.dataclass(frozen=True)
class ShapSources:
    """The TreeSHAP files the spliced figure data reads (one table for every consumer).

    Attributes:
        records: The record sets root (`data/records`).
        experiments: The experiments member (the SHAP artifacts).
    """

    records: Path = rd.RECORDS
    experiments: Path = rd.EXPERIMENTS

    @property
    def root(self) -> Path:
        """The `shap_rerun` record set."""
        return self.records / "shap_rerun"

    @property
    def study1(self) -> Path:
        """Study 1 DEV tau, placebo and slice declarations."""
        return self.root / "study1" / "study1.json"

    @property
    def disc(self) -> Path:
        """The Study 2 `disc_conf.json` with the TreeSHAP maps."""
        return self.root / "study2" / "disc_conf_shap_gbdt.json"

    @property
    def bounds(self) -> Path:
        """The Study 3 certification bounds on the TreeSHAP maps."""
        return self.root / "study3" / "certsafe_bounds_shap.json"

    @property
    def maps_artifact(self) -> Path:
        """`study2_maps_shap.json`."""
        return self.experiments / "artifacts" / "study2_maps_shap.json"

    @property
    def certsafe_artifact(self) -> Path:
        """`study3_certsafe_shap.json`."""
        return self.experiments / "artifacts" / "study3_certsafe_shap.json"

    def eval_files(self, study: int) -> list[Path]:
        """The referee job records of the changed Study 2 / 3 arms."""
        files = sorted((self.root / f"study{study}_eval").rglob("res_*.jsonl"), key=Path.as_posix)
        if not files:
            raise FileNotFoundError(f"no records under {self.root / f'study{study}_eval'}")
        return files

    def report(self, study: int) -> tuple[Path, Path]:
        """The Study 2 / 3 evaluation and verdict files on the TreeSHAP arms."""
        key = "conf" if study == 2 else "certsafe"
        d = self.root / f"study{study}_report"
        return d / f"eval_{key}.json", d / f"verdict_{key}.json"


def _sig(v: Mapping[str, Any]) -> str:
    return json.dumps(v["signature"])


def frozen_twins(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, str]:
    """{job arm of `new`: job arm of `old`} where only the arm that represents a signature changed.

    The new job's decision signature differs from its own frozen one but equals the signature of
    a frozen job, so the frozen records of that job stand for it.
    """
    by_sig = {_sig(old["arms"][j]): j for j in old["jobs"] if j in old["arms"]}
    out = {}
    for a in new["jobs"]:
        own = old["arms"].get(a)
        if a in old["jobs"] and own is not None and _sig(own) == _sig(new["arms"][a]):
            continue
        twin = by_sig.get(_sig(new["arms"][a]))
        if twin is not None:
            out[a] = twin
    return out


def changed_jobs(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Job arms of `new` whose decision signature no frozen job of `old` simulates."""
    twins = frozen_twins(old, new)
    sig_old = {a: _sig(v) for a, v in old["arms"].items()}
    return [
        a
        for a in new["jobs"]
        if a not in twins and (a not in old["jobs"] or _sig(new["arms"][a]) != sig_old[a])
    ]


@dataclasses.dataclass(frozen=True)
class Merged:
    """Referee records of the recorded run with the changed jobs replaced.

    Attributes:
        R: Records per seed and arm, aliases of `new_art` filled.
        heads: Stage headers of the recorded and the new files.
        redo: Job arms taken from the new files.
        twins: Job arms that take a recorded job's records (`frozen_twins`).
        ledger: `report_data.run_ledger` over the job arms (aliases left out).
    """

    R: Records
    heads: list[Record]
    redo: list[str]
    twins: dict[str, str]
    ledger: dict[str, Any]


def merged_records(
    frozen_files: Sequence[Path],
    new_files: Sequence[Path],
    filt: JobFilter,
    old_art: Mapping[str, Any],
    new_art: Mapping[str, Any],
) -> Merged:
    """The recorded referee run with the jobs of `changed_jobs(old_art, new_art)` from `new_files`.

    Recorded map-arm records whose arm `new_art` no longer simulates with the recorded signature
    are dropped; a frozen twin takes its recorded job's records; `new_art`'s aliases are filled.
    """
    redo = changed_jobs(old_art, new_art)
    twins = frozen_twins(old_art, new_art)
    map_arms = set(new_art["arms"])
    R, heads = load_jobs(frozen_files, filt)
    for s in R:
        kept = {
            a: r
            for a, r in R[s].items()
            if a not in map_arms or (a in new_art["jobs"] and a not in redo and a not in twins)
        }
        R[s] = kept | {a: R[s][j] for a, j in twins.items() if j in R[s]}
    Rn, hn = load_jobs(new_files, filt)
    for s, by_arm in Rn.items():
        R.setdefault(s, {}).update({a: r for a, r in by_arm.items() if a in redo})
    ledger = rd.run_ledger(heads + hn, R)
    alias_fill(R, new_art["alias_of"])
    return Merged(R, heads + hn, redo, twins, ledger)


def _shards(run: Path, n: int) -> list[Path]:
    files = sorted(run.glob("*/res_*.jsonl"), key=Path.as_posix)
    rd.ensure(len(files) == n, f"{run.name}: {len(files)} raw shards, want {n}")
    return files


def spliced_v4(analysis: Mapping[str, Any], rerun: Mapping[str, Any]) -> dict[str, Any]:
    """The Study 1 analysis with the `shap_gbdt` slice scores, placebo check and tau of `rerun`.

    Args:
        analysis: The recorded `analysis_v4.json`.
        rerun: The `study1.json` of the rerun.

    Raises:
        ValueError: If the rerun's slices are not the analysis's `shap_gbdt` slices.
    """
    new = rerun["methods"][METHOD]
    A = copy.deepcopy(dict(analysis))
    have = {s for s, v in A["slices_result"].items() if METHOD in v["metrics"]}
    if have != set(new["slices"]):
        raise ValueError(f"slices differ: recorded {sorted(have)}, rerun {sorted(new['slices'])}")
    for s, m in new["slices"].items():
        A["slices_result"][s]["metrics"][METHOD] = m
    A["placebo_descriptive"][METHOD] = new["placebo"]
    A["desc_tau"][METHOD] = new["tau"]
    return A


def _v4(fd: dict[str, Any], A: Mapping[str, Any], checks: rd.Checks) -> None:
    ref = study1.gt_ref(A["gt"]["cells"])
    st = A["slices_result"]["pooled"]["metrics"][METHOD]
    mine = study1.subset_metrics(rd.decl_from_list(st["declared"]), ref)
    rd.ensure(all(rd.close(mine[k], st[k]) for k in rd.SLICE_KEYS), (mine, st))
    plc = A["placebo_descriptive"][METHOD]
    rd.ensure(plc["n_declared"] == len(plc["declared"]), plc)
    checks.require("shap: Study 1 pooled scores recomputed from the declared list", True)
    fd["pooled"][METHOD] = {k: mine[k] for k in POOLED_KEYS}
    fd["placebo_declarations"][METHOD] = plc["n_declared"]
    fd["desc_tau"][METHOD] = A["desc_tau"][METHOD]


def _option_a(fd: dict[str, Any], src: rd.Sources, shap: ShapSources, checks: rd.Checks) -> None:
    disc = rd.read_json(shap.disc)
    maps_art = rd.read_json(shap.maps_artifact)
    checks.require(
        "shap: maps artifact pins the TreeSHAP disc_conf.json",
        maps_art["sha256"]["disc_conf.json"] == rd.sha_lf(shap.disc),
    )
    fd["maps_sha256"] = src.recorded("artifacts/study2_maps_shap.json").sha256
    fd["disc_json_sha256"] = rd.sha_lf(shap.disc)
    fd["map_quality"] = rd.map_quality(disc, maps_art)
    m = merged_records(
        _shards(src.runs / "e6p-conf-eval-1", 12),
        shap.eval_files(2),
        study2.EVAL_FILTER,
        rd.read_json(src.artifact("study2_maps.json")),
        maps_art,
    )
    E, Vd = (rd.read_json(p) for p in shap.report(2))
    fd["eval"] = rd.opt_a_eval(m.R, m.ledger, maps_art, E, Vd, checks)


def _certsafe(fd: dict[str, Any], src: rd.Sources, shap: ShapSources, checks: rd.Checks) -> None:
    art = rd.read_json(shap.certsafe_artifact)
    old = rd.read_json(src.artifact("study3_certsafe.json"))
    checks.require(
        "shap: certsafe artifact keeps the recorded tau and bounds",
        art["tau"] == old["tau"] and art["bounds"] == old["bounds"],
    )
    tau = {(f, int(d), k): v for f, d, k, v in art["tau"]}
    fd["artifact_sha256"] = src.recorded("artifacts/study3_certsafe_shap.json").sha256
    fd["cert"] = rd.certification(art, tau, checks)
    fd["jobs"] = art["jobs"]
    m = merged_records(
        _shards(src.runs / "e6p-cs-eval-1", 20),
        shap.eval_files(3),
        study3.job_filter("eval"),
        old,
        art,
    )
    E, Vd = (rd.read_json(p) for p in shap.report(3))
    fd["eval"] = rd.cs_eval(m.R, m.ledger, art, E, Vd, checks)


def splice_fd(
    data: Mapping[str, Any], src: rd.Sources | None = None, shap: ShapSources | None = None
) -> dict[str, Any]:
    """`report_data.build` output with the `shap_gbdt` entries recomputed on the TreeSHAP files.

    Replaced: Study 1 `pooled`, `placebo_declarations` and `desc_tau` of `shap_gbdt`; Study 2
    map pins, map quality and evaluation; Study 3 artifact pin, certification, jobs and evaluation.
    Assertion names not yet in `assertions` are appended to it.

    Raises:
        ReportCheckError: A recomputed block differs from the TreeSHAP evaluation files.
    """
    src, shap = src or rd.Sources(), shap or ShapSources()
    checks = rd.Checks()
    fd = copy.deepcopy(dict(data))
    A = spliced_v4(rd.read_json(src.decision / "v4_analysis.json"), rd.read_json(shap.study1))
    _v4(fd["v4"], A, checks)
    _option_a(fd["option_a"], src, shap, checks)
    _certsafe(fd["certsafe"], src, shap, checks)
    fd["assertions"] = list(dict.fromkeys([*data["assertions"], *checks.names]))
    return fd
