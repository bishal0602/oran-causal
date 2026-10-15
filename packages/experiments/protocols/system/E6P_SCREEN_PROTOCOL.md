# E6-P conflict screen protocol (Gate A v2)

Status: **DRAFT FOR FREEZE, 2026-09-28.** It is frozen together with `E6P_SPEC.md` at one commit (`e6p_spec_commit`);
the harness records that hash and the sha256 of both files and of `SEED_REGISTRY.json`, and asserts that the E6-P
configuration it runs equals the frozen values. **Nothing in either document is tuned after any outcome.** A code
defect found by a mechanism test may be fixed (the mechanics must match the spec); a *value* may not change. If a frozen
value makes a precondition fail, the affected pair is reported "not screenable under E6P-v1". Any value change is a
new version (E6P-v2) with fresh seeds, and the v1 outcome is published as is. This is a DEV screen. It cannot confirm an
edge, it can only kill or pass a pair on to a later, separately preregistered confirmation. Run gating: each stage
below needs the user's explicit go.

Purpose: decide per pair whether E6-P has (1) a material co-deployment loss at matched energy, (2) mostly recoverable
by a budgeted O-RAN WG3 arbiter with perfect knowledge, (3) with state-dependent headroom beyond the best static rule.
Criterion (3) is new relative to E6 Gate A. Same-knob and direct conflicts are expected to fail it
(`PATH2_SCOUT_PY.md`, `PATH2_SCOUT_NIST.md`).

## 1. Common configuration
- Plant: `E6P_SPEC.md` values; `e6p.ptx_on = e6p.prot_on = True` in **every** arm of every pair, so the freeze arm is
  shared across pairs.
- E6 settings: `n_ue` 300, `n_pico` 3, `mobility="ped"`, `kpm="nominal"`, `update=False`, `warmup_s=120`,
  `scored_s=600` (the Gate A episode length), slice mix and traffic as E6-scn-v1.
- WG3 environment: `wg3=True`. Arbiters use only accept / reject / modify / defer / lock / rollback and make no own
  writes (`DECISION.md` item 4).

## 2. Pairs (xApp sets; energy side A, protection side B)
| pair | base mix | `e6p.xapps` | A (energy) | B (protection) | conflict class (ConMit cl. 4.2.1) |
|---|---|---|---|---|---|
| **P1 (primary)** | `ES` (carrier + pico, E6P_SPEC 4.1) | `("SliceGuarantee",)` | ES | SliceGuarantee | implicit / indirect (capacity removed vs guaranteed share) |
| P2 | `none` | `("PowerES", "Coverage")` | PowerES | Coverage | direct (same `ptx` knob) + indirect (neighbour interference) |
| P3 | `ES` | `("PowerES", "SliceGuarantee")` | ES + PowerES | SliceGuarantee | three-way indirect |

P2 is expected to fail criterion (3), because static priority resolves direct conflicts. It is kept as the declared
direct-conflict control.

## 3. Strata and seeds
Strata (4): scenario ∈ {`base`, `surge`} × load ∈ {`L10`, `L40`}. `stratum = 2·scenario_idx + load_idx`: 0 base-L10,
1 base-L40, 2 surge-L10, 3 surge-L40. `surge` = E6-scn-v1 S1 with its frozen DEV defaults (`E6_SCENARIO_CONTRACT.json`).
It is not re-placed, and its location is drawn independently of the protected flags (separate RNG streams).

Fresh E6 DEV ranges. They are disjoint from every E6 range in `SEED_REGISTRY.json` (0-30, 100000-100179, 110000-110179,
120000-120179, 130000-130179, 140000-140059) and below the E6 TEST floor 960000. **They must be registered before
use.**

| use | seeds | mapping |
|---|---|---|
| load calibration, fit | 150000-150009 | base scenario, freeze only |
| load calibration, check | 150010-150019 | same; report realised L only |
| mechanism tests M1-M3, M6 | 150020-150023 | base |
| mechanism test M4 (own KPI vs freeze) | 150020-150059 | `150020 + 10·stratum + j`, j = 0..3 |
| oracle timing pilot | 150060 | wall time only; scores discarded unread |
| **screen** | **150100-150137** | `150100 + 10·stratum + j`, j = 0..7 (8 seeds per stratum); j = 8, 9 reserved unused |
| reserved for a later confirmation of a surviving pair | 150200-150399 | untouched by this protocol |

Paired arms reuse the episode seed, so every arm sees the same plant tape. Bootstrap RNG:
`default_rng([6611, pair_idx, stratum])`, where 6611 is a new stream tag disjoint from the registry tags and from the
E6 plant key 6600.

## 4. Load calibration (Stage 0a, before any xApp arm)
For each of `L10` and `L40`:
1. Bisect `load_factor` in [0.05, 3.0] (at most 12 iterations).
2. Each iteration runs the **freeze** arm on 150000-150009, `base`, E6-P plant, and computes L (E6P_SPEC 3.1).
3. Stop when the fit-seed mean L is within ±1.0 percentage point of the target.
4. Report the realised L (mean and range) on 150010-150019.

The calibration code reads only `prb_used` and `prb_cap` counters. It must not compute, log or print any SLA,
protected or energy quantity (verified by code review). The two `load_factor` values are written into the result
record and are frozen from then on. If `L40` cannot be reached below `load_factor` 3.0, the stratum is reported
infeasible. It is not re-targeted.

## 5. Arms (per pair, per screen seed)
| # | arm | definition |
|---|---|---|
| 1 | freeze | reject every request (3GPP default configuration: ptx 0, prot_min 0, carriers on, picos awake) |
| 2 | each xApp alone | `subset((x,))` for every x in the pair (all stay deployed and proposing) |
| 3 | accept-all (noarb) | no arbitration, last writer wins |
| 4 | static subsets | every other `subset(S)`; P3 uses all 2^3 subsets |
| 5 | per-knob priority | `baselines.priority`, every xApp order (P1, P2: 2 orders; P3: 6) |
| 6 | cell-priority lock | new, `SingleCommandPerNode`-like (NIST ns3-oran CMM). A cell whose knob was changed by a higher-priority xApp in the last 60 s rejects lower-priority requests on any knob of that cell (knob → cell: `carrier`/`ptx`/`prot_min`/`sleep` c → c). 60 s = v1 ES hold = `KnobLock` default lease [I]. Every xApp order is run. |
| 7 | knob-lock | `baselines.KnobLock(lease_s=60)` |
| 8 | QACM | `published.py` QACM behind its declared wrapper, if faithfully representable (E6P_SPEC §6 item 7); otherwise omitted and reported |
| 9 | per-region hindsight static | For each of the 10 regions (7 macro sites + 3 picos, as `wg3_oracle.py`), a fixed xApp subset held for the whole episode. It is chosen per seed by one pass of coordinate descent from accept-all, over regions in index order, minimising PSVR among energy-matched and guardrail-passing candidates. Privileged, not deployable: the strongest static reference. **Stage 3 only.** |
| 10 | budgeted WG3 oracle | `scratchpad/e6_dev/wg3_oracle.py` mechanics, constants unchanged (see below). **Stage 2 only.** |

Oracle details (arm 10):
- Decision epoch D = 20 s, lookahead H = 90 s, n_glob = 6, n_loc = 2.
- A plan is a mode per region × xApp in {accept, reject, half, lock} plus optional rollback. Requests are decided
  individually by their region/xApp plan, so decisions are state- and time-dependent.
- Churn parity: applied knob changes ≤ the accept-all arm's on the same tape. Freeze and accept-all are always
  candidates.
- Objective over H: protected violated UE-s, with all-UE violated UE-s as a lexicographic tie-break.
- Energy constraint: a candidate plan is admissible only if the rollout's cumulative scored energy at the end of the
  horizon ≤ `E_freeze(t) − 0.90·(E_freeze(t) − E_A(t))`. Here `E_freeze(t)` and `E_A(t)` are the per-second
  cumulative energies of the freeze and A-alone arms on the same tape (privileged). If no plan is admissible,
  accept-all is used.
- "Same rights" means the oracle has the same action class, actuator limits and churn budget as every other arbiter.
  Its only privilege is lookahead on the true tape, the same kind of privilege arm 9 has.

## 6. Mechanism preconditions (Stage 0b). ALL must pass before any screen seed is run
| id | test | pass rule |
|---|---|---|
| M1 | **Power step moves SINR as predicted.** Seeds 150020-150023, snapshot at t = 120 s, occupancy frozen for one tick. For each macro c, apply ptx −3 dB and then +3 dB. | (a) Every UE's realised SINR change equals the analytic value from the gain maps and occupancies (`10·log10((I+N)/(I'+N))` plus the own-signal change when served by c) within 0.01 dB. (b) Mean change of neighbour-edge UEs (served by n ≠ c, RSRP from c within 6 dB of serving) is > 0 for −3 dB and < 0 for +3 dB, for ≥ 90 % of macros that have such UEs and non-zero occupancy on the snapshot tick (an idle macro radiates no interference, so its neighbours' change is exactly 0; that case is covered by (a)). (c) Measured L3 input of c shifts by exactly the offset. (d) Sector power change equals `car·Δp·ρ·P_max·(10^(o/10) − 1)` within 1e-9 W. |
| M2 | **Quota is work-conserving.** Scheduler unit tests. | (a) Zero protected demand: allocation bit-identical to `prot_min = 0`. (b) Protected demand < m·cap: protected get their demand and the rest reach the shared pool (Σ alloc = min(Σ demand, cap − dedicated)). (c) Protected demand > m·cap with saturated others: protected alloc ≥ m·cap − 1e-9. (d) `prb_util` and `prb_rsv` unchanged by an idle min share. (e) min share ≤ cap − dedicated. |
| M3 | **Carrier off halves capacity; sleep removes the pico.** | Macro `cap_prb` 106 → 53 on carrier 2 → 1, back to 106 only after `CARRIER_ON_S` = 2 s. Sector power matches the E6 EARTH formula. Pico sleep: cap 0, power `PICO_SLEEP_W`, wake after `PICO_WAKE_S`. |
| M4 | **Each xApp alone improves its own KPI vs freeze.** Seeds per §3, paired. Own KPIs: ES and PowerES → scored energy; SliceGuarantee → PSVR; Coverage → share of UE-seconds with serving SINR < −6 dB. | In every stratum where the xApp made ≥ 1 accepted change per seed on average, the pooled own KPI improves and at least 3 of 4 seeds improve. An xApp that acts in fewer than 2 strata is *inert*. Pairs containing it are not screenable; this is reported, not fixed. |
| M5 | **Bit-identity.** | `E6PConfig()` default gives E6-scn-v1 bit-identical episodes (existing test). With ptx/prot on, no xApps, ptx 0 and prot_min 0, allocation and energy are identical to E6 on the same seed. |
| M6 | **Graceful sleep** (E6P_SPEC §6 item 2). | In ES-alone on 150020-150023 there are 0 RLFs within 2 s after a pico-sleep event among UEs whose best remaining cell has SINR > Q_out. |

Outcome visibility: M4 reveals single-xApp and freeze outcomes on mechanism seeds only. No co-deployment or arbiter
arm is run before Stage 1, and screen seeds are untouched until M1-M6 pass.

## 7. Quantities and Gate A v2 (per pair, per screen stratum; sums over the stratum's 8 seeds)
- `V_a` = PSVR of arm a (E6P_SPEC §5); `AA` = accept-all.
- `V_ref` = min over the single-xApp arms (arm 2) of V (the best single xApp).
- Eligible arm: energy-matched (X = 0.90, E6P_SPEC §5) and passing all guardrails vs AA. A stratum is screenable for
  the pair only if A-alone saves ≥ 1 % of freeze energy (E6P_SPEC §5).
- Co-deployment loss `Λ = (V_AA − V_ref) / V_ref`.
- Recovery of an eligible arm: `R_a = (V_AA − V_a) / (V_AA − V_ref)`.
- `R_static` = max of R over eligible arms 4-9. `R_or` = R of the oracle (arm 10) if it is eligible, else 0.

Criteria, all evaluated **in the same stratum**:
1. **Loss:** AA is energy-matched, and Λ ≥ 0.15, and `V_AA − V_ref ≥ 36` (≥ 1 % of protected UE-time, a materiality
   floor [A] that keeps Λ meaningful when V_ref ≈ 0), and the paired-seed bootstrap (10 000 resamples) one-sided 90 %
   lower bound of `V_AA − V_ref` is > 0.
2. **Recoverable:** `R_or ≥ 0.50`.
3. **State-dependent headroom (new):** `R_or − R_static ≥ 0.10`.
4. **Oracle sign reliability (new, added after the E6 rank diagnosis and request-level pilot, before any E6-P
   outcome):** at every oracle decision point where the oracle deviates from accept-all, re-score its chosen plan and
   accept-all on an independent copy whose post-decision exogenous randomness is re-drawn (`env.copy` with a fresh
   future RNG stream; same state at the decision point). `ρ_sign` = fraction of deviations whose advantage over
   accept-all keeps its sign on the re-drawn copy, pooled over the stratum. Criterion: `ρ_sign ≥ 0.70`. Reason: in E6
   ~44 % of oracle effect variance was tape-specific (reliability 0.56) and neither region-policy nor request-level
   features predicted effect signs (sign AUC ≤ 0.62), so oracle headroom that does not survive a re-draw is
   max-over-noise and no deployable arbiter can capture it.

Verdict per pair:
- **PASS** if at least one stratum satisfies 1-4.
- **NOISE STOP** if 1-3 hold somewhere but 4 fails in every such stratum.
- **DEAD (no loss)** if no screenable stratum satisfies 1.
- **DEAD (not recoverable)** if every stratum satisfying 1 fails 2.
- **NO-EDGE STOP** if 1 and 2 hold somewhere but 3 fails in every such stratum.

If AA is not energy-matched in a stratum, that stratum cannot satisfy 1. Its energy shortfall (1 − retention) is
reported as an energy-side co-deployment loss. Criteria 2 and 3 are decided on point estimates (DEV screen). Bootstrap
intervals are reported for Λ, R_or and R_or − R_static. All four strata and all arms are reported for every pair,
whatever the verdict.

## 8. Stages and cost (each stage needs an explicit go; times are estimates to be replaced by measured values)
| stage | content | estimate |
|---|---|---|
| 0a | load calibration (§4) | ≤ 2 × 12 × 10 freeze episodes |
| 0b | M1-M6 | minutes (M1-M3, M5 unit-level) + M4: 16 seeds × (1 + 5 xApps) episodes |
| 0c | oracle timing pilot on 150060 (scores discarded) | 1 oracle episode |
| 1 | arms 1-8, all pairs, all 32 seed-strata | ≈ 30 episodes per seed-stratum; ~22 s per E6 Gate-A-length episode in v1 → a few CPU-h (measure) |
| 2 | oracle, only in strata where criterion 1 passed | per the 0c measurement |
| 3 | arm 9, only in strata where 1 and 2 passed | P1/P2: 10 regions × 3 = 30 episodes per seed; P3: 70 |

The stages are gated sequentially because the criteria are conjunctive. Skipping the oracle where criterion 1 failed
cannot change a verdict. Drivers use timeouts, and background processes are cleaned up after every stage.

## 8a. Amendments before any protocol-seed run (2026-09-28)
- M1(b) restricted to macros with non-zero occupancy on the snapshot tick. Reason: physics identity found on a DEV
  smoke run (3/21 macros idle at the snapshot → 0 dB neighbour change); no protocol seed had been run.
- QACM (arm 7) is fitted per (pair, stratum) on accept-all logs of E6 DEV seeds 20-23 (the protocol named none);
  QoS thresholds from `sg_viol_hi` and `cov_sinr_low_db`.
- Oracle "half" = slew rule with 1 dB / 0.05 quanta (legacy halving rounds carrier/sleep/prot_min to no-op or full step).

- **M4 amended after Stage 0b (user decision 2026-09-28).** M4 is evaluated per xApp only in strata where the freeze
  arm's own KPI is material: PSVR ≥ 36 UE-s per UE-h, low-SINR share ≥ 0.01 (the same 1 %-of-UE-time floor as
  criterion 1); energy is always material. An xApp with fewer than 2 acting, material strata is inert. A pair
  containing a failing or inert xApp is not screenable (reported, not fixed); the screen proceeds for the remaining
  pairs. **Disclosure:** this was amended after single-xApp vs freeze results on the mechanism seeds (150020-150059)
  were seen: SliceGuarantee left PSVR unchanged at L10 where freeze PSVR was 10.8-13.4 (< 36; min share non-binding)
  and cut it 255→58 / 384→125 at L40; Coverage worsened its KPI. No co-deployment, arbiter or screen-seed outcome had
  been run. Gate A v2 quantities and thresholds are unchanged.

## 9. Frozen at doc commit
All of `E6P_SPEC.md` §2-§5 (values, semantics, xApp rules, metric, X = 0.90, guardrail margin, screenability floor);
this document's pairs, strata, seed ranges, arm definitions, oracle constants and energy constraint, precondition pass
rules, Gate A v2 thresholds (15 %, 36 UE-s per UE-h, 50 %, 10 pp, ρ_sign 0.70), bootstrap settings and verdict rules. The two
`load_factor` values are produced by the frozen §4 procedure and recorded, never chosen. Not frozen and not
outcome-relevant: code structure, logging and plotting.
