# Reproduction standard

Adopted 2026-11-24 (user decision; post hoc amendments A1 and A2). It says what "reproduces" means for every check,
re-run and comparison in this repository.

## Why bit-identity is not the test

| Fact | Consequence |
|---|---|
| The stored Study 2/3 decision records were made with Python 3.12.13, numpy 2.4.2 and glibc 2.35 (legacy driver). Re-runs use the repo lock: Python 3.14.9, numpy 2.5.4, glibc 2.39 | The stored environment cannot be rebuilt from `uv.lock` |
| NumPy gives no bitwise guarantee across versions, CPUs or SIMD paths | Last-bit differences between environments are expected |
| Threshold KPIs (violated UE-seconds, below-floor counts) are discontinuous | A last-bit difference can flip an integer count on some seeds |

Observed: the X17 confirmation differed from the stored records on 2 of 8 seeds by one violated UE-second, with
identical decisions; the X17 step-1 and X19 analysis outputs differed by about 1e-14 to 1e-13.

## Environment identity

An environment is the tuple below. Two runs share an environment only if every field is equal.

| Field | Source |
|---|---|
| Python, numpy, scipy versions | record header, `startup.json` |
| glibc and kernel (`platform`) | record header, `startup.json` |
| CPU SIMD level and CPU model | record header (`simd`, `cpu_model`) |
| `uv.lock` sha256 | `pin_check.json`, provenance stamp |
| Host image / platform (Kaggle, VPS, Colab, instabox) | record header (`host`) |

Every record carries these fields; a record without them cannot claim the E tier.

## Tiers

| Tier | Applies to | Criterion |
|---|---|---|
| E | Any output, same environment | Bit-identical. A mismatch is a bug |
| D | Structural outputs, any environment | Exact: declared edge sets and signs, decision signatures, certsafe signatures, uncertified sets |
| N | Non-simulated numeric outputs, across environments | Relative tolerance 1e-9 per number; missing values must match (amendment A1) |
| S | Simulated outputs, across environments | No per-seed equality. Over K seeds, the paired difference of each reported metric lies within an equivalence margin declared in advance as a fraction of that metric's reported CI half-width |

The S tier follows the ensemble-consistency approach of climate modelling: Baker et al. 2015, Geosci. Model Dev.
8:2829, doi:10.5194/gmd-8-2829-2015.

## Rules

- Never mix environments inside a paired comparison. Re-run the comparators in the new environment, or report the
  S-tier bound beside the contrast.
- A same-environment (E-tier) test runs in one session or kernel: Kaggle may assign different CPUs to different
  kernels.
- The bit-exact outcome is always reported beside any tolerance outcome.
- The tier and, for S, the margin and K are declared before the run, in the protocol entry that uses them.

## Applied so far

| Item | Tier | Use |
|---|---|---|
| A1: X17 step 1, X19 discovery reproduction | N, D | Numbers within relative 1e-9; declarations exact |
| A2: X17 confirmation | E, S | Frozen and strict maps on the 8 confirmation seeds in one Kaggle kernel must be bit-identical (E); drift to the stored records is reported (S) |
| X18 Part C | S | New-environment policies are compared with stored old-environment comparators; the S-tier bound is reported beside its contrasts |

Open team decision: a full single-environment re-baseline of the Study 2/3 decision arms (about 100-150 CPU-h).
