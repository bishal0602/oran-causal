# Provenance: per-study manifests

Source commit for every `CDD-ORAN@0c89106` path below is `0c89106` on `feat/v2`. The manifests are new files; nothing is ported line by line. Schema: `../r2/port_map.md` section 4.1, with the changes listed under Schema.

## Files

| Path | Source | Note |
|---|---|---|
| `packages/experiments/src/oran_causal/experiments/manifests.py` | none | Loader (`load`, `load_all`, `ids`, `parse`), validator (`check`), cost sums (`tier_cost`, `Manifest.stage_cost`), driver resolution. Dataclasses and explicit checks; no pydantic |
| `packages/experiments/manifests/{study1,study2,study3,study4,bridging}.yaml` | `manifests/tiers.yaml` (costs, platforms), `manifests/seeds.json` (seed blocks), `protocols/SHA256SUMS` (protocol hashes), `spec/claims.yaml` (claims), `data/zenodo.yaml` (record sets) | Study 4 costs come from `reproduce.plan` on `specs/eval/full.json` and `artifacts/eval_cost_table.json` |
| `packages/experiments/manifests/{x1,x2_x3,x4,x5_x6,x7,x8,x9_x16}.yaml` | as above, plus the record sets `data/manifests/posthoc_*.sha256` | One manifest per record set, as the old code kept them |
| `packages/experiments/tests/test_manifests.py` | none | Validator on every manifest, schema negatives, cost cross-check against `tiers.yaml` and the dry-run plan |

## Schema

| Field | Content |
|---|---|
| `id`, `platform` | Item id (file stem) and the re-run lane of `tiers.yaml` |
| `claims` | Registered IDs. Every empirical and protocol claim of `spec/claims.yaml` appears in at least one manifest |
| `protocol` | `{path, sha256}` under `protocols/`, hash of the LF text |
| `code_origin` | `CDD-ORAN@0c89106:<path>` entries |
| `seeds` | `{block, range, role, world}`; `block` is the `seeds.json` key where one exists; `range` must lie inside a range listed anywhere under `world` |
| `stages` | `{name, driver, args, inputs, outputs, cost, frozen_in}`; `driver` is `module:callable`; `cost` is `{cpu_h, basis}` with `cpu_h` a number or `[low, high]`; `frozen_in` lists tiers that reuse the frozen records |
| `frozen_inputs` | `{path, sha256}` relative to `packages/experiments/`: artifacts, specs, protocol addenda |
| `records` | List of `{set, manifest, zenodo_doi}`; the brief's single entry became a list because Study 1 to 3 draw on two or three sets |
| `tiers` | `t0` to `t3`: text of what the tier does for the item, `null` when the item is not part of it |
| `equivalence` | `{metric, rule}` from `rerun_cost.md` section 4 and the post hoc gate tolerances |

## Choices

- Costs repeat `tiers.yaml` stage by stage; the tests fail if the two differ, so a change goes to both files.
- Study 4 is two stages, `campaign` (the t2 cost) and `campaign_n24000` (the rest, `frozen_in: [t2]`), so a tier sum needs no plan code.
- Analysis stages that `tiers.yaml` does not price carry cost 0.
- X8 is not in `tiers.yaml`. Its stage `variants` is priced at 2.41 CPU-h, the sum of `wall_s` over the 590 stored variants (host not recorded, no reference scaling).
- X1 is priced at 1.49 CPU-h from the `est_cost_s` of the E2 R2 cells (basis `P`); it is in neither t2 nor t3.
- `bridging`, `x5_x6` and `x7` carry no claim: the registry cites none of them.
