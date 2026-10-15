# E6-P: transmit-power and protected-slice extension of E6 (parameter specification)

Status: **DRAFT FOR FREEZE, 2026-09-28.** Every value below was fixed from external sources, from frozen pre-E6-P E6
values, or as a declared assumption, **before any E6-P episode with an xApp or arbiter was run or seen by the author**.
The screen that uses this plant is `E6P_SCREEN_PROTOCOL.md`. Both documents are frozen together at one commit
(`e6p_spec_commit`). From that commit on, no value in either document is changed in response to any outcome. A change
for any reason creates a new version (`E6P-v2`) with fresh seeds, and the v1 result is published unchanged.

Design (given, not chosen here): per-cell Tx-power offset knob; protected eMBB slice with a work-conserving minimum PRB
share; four xApps (power-ES, coverage, carrier/pico ES, slice guarantee); screen pairs P1-P3. Design rationale:
`scratchpad/e6_dev/decision/PATH2_SCOUT_PY.md`. Implementation interface: `cdd_oran/envs/e6/config.py::E6PConfig`
(field names below are that interface).

Provenance tags: **[S]** value stated in the cited source; **[S→]** derived from a sourced quantity by the rule given;
**[I]** inherited unchanged from a frozen pre-E6-P E6 value (E6-scn-v1 / v1 xApps / `E6_METRIC.md`), so it is not a new
degree of freedom; **[A]** declared modelling assumption (no source, fixed before outcomes).

## 1. Sources

| id | source | used for |
|---|---|---|
| TR38864 | 3GPP TR 38.864 V18.1.0 (2023-03), *Study on network energy savings for NR* | cl. 1 (scope: idle/low/medium load prioritised); cl. 5.1 Table 5.1-1 (Set 2 FR1 FDD 20 MHz, total DL power 49 dBm); cl. 6.2.1 B-1 (SCell/carrier off at low load, "UPT degrades by 14 % if one SCell goes to dormant state"); cl. 6.4.1 D-1, Table 6.4.1.2-1 (Tx-power reduction evaluated at -3/-6/-9 dB (MTK, Nokia), up to -12/-18 dB; semi-static reduction 3-18 dB gives 2.3-51.5 % ES gain, UPT loss 2.03-19.49 %); Annex A Table A-1 (load L = % of resources used for UE-specific PDSCH; low 0<L≤15, light 15<L≤30, medium 30<L≤50); Annex A Table A-2 (FTP model 3, 0.5 MB) |
| EARTH | Auer et al., "How much energy is needed to run a wireless network?", IEEE Wireless Commun. 18(5), 2011, eq. (1) and Table II | P_in = N_TRX (P0 + Δp·P_out) for 0 < P_out ≤ P_max, N_TRX·P_sleep at P_out = 0; macro N_TRX 6 (3 sectors), P_max 20 W, P0 130 W, Δp 4.7, P_sleep 75 W; pico N_TRX 2, P_max 0.13 W, P0 6.8 W, Δp 4.0, P_sleep 4.3 W; cl. III-B: load = P_out/P_max ∝ utilised resources; "the load dependency of pico and femto BSs is negligible" |
| TS28541 | 3GPP TS 28.541 V18.15.0 (2026-06), 5G NRM | cl. 4.3.36 RRMPolicyRatio (dedicated / prioritized / shared resources; min ratio = "guaranteed for use ... when it needs to use them. When not used, these resources may be used by other" members); cl. 4.4.1 attribute table (rRMPolicyMinRatio, rRMPolicyDedicatedRatio: integer 0..100, **defaultValue 0**; sum of min ratios ≤ 100); cl. 4.3.58 DESManagementFunction (intraRatEsActivationOriginalCellLoadParameters / ...CandidateCellsLoadParameters / intraRatEsDeactivationCandidateCellsLoadParameters = LoadTimeThreshold {loadThreshold = % PRB usage, timeDuration s}; no default values) |
| TS23501 | 3GPP TS 23.501, cl. 5.7.4 Table 5.7.4-1 | 5QI 4 = GBR, "Non-Conversational Video (Buffered Streaming)", PDB 300 ms, PER 1e-6 **[V: default priority and 2000 ms averaging window to be checked against the table]** |
| TR26925 | 3GPP TR 26.925 V16.0.0 (ETSI TR 126 925, 2020-11), cl. 5.1 | typical streaming video bitrates: "720p HD: 2 - 5 Mbps" |
| M2410 | ITU-R M.2410-0 (11/2017), cl. 4.4, Table 1 | 5th-percentile user = the edge-user statistic (Dense Urban DL 0.225 bit/s/Hz) |
| TS38331 | 3GPP TS 38.331 (NR RRC) | ss-PBCH-BlockPower and powerControlOffset are integer-dB parameters (1 dB actuator grid) |
| ConMit | O-RAN WG3 TR *Conflict Mitigation* R004 v01.00 (2024), cl. 4.2.1, 6.1.1 | direct / indirect / implicit conflicts ("protecting throughput metrics for GBR users may degrade non-GBR metrics"); resolution by accept/reject/agreement enforced by ConMit |
| E6 | `cdd_oran/envs/e6/config.py` (E6-scn-v1), `xapps.py` (v1), `docs/benchmark/E6_METRIC.md` | inherited [I] values |

## 2. Plant changes (all gated by `cfg.e6p`; with `E6PConfig()` E6-scn-v1 stays bit-identical)

### 2.1 Tx-power knob `("ptx", c)`
- **Semantics: total carrier Tx power** (data and SSB/common channels together), applied as a dB offset to cell c's
  whole gain column: serving RSRP, SINR, the interference c causes, and the measured L3 RSRP all follow, so the A3
  handover border moves. This matches TR38864 Table 6.4.1.2-1 entries that reduce total DL Tx power (Nokia "Reduced DL
  transmit power by 3/6/9 dB" from 49 dBm; Qualcomm 55→52 dBm). The PDSCH-only variant (D-1 powerControlOffset, SSB
  unchanged, no border move) is **not modelled**; stated as a limitation.
- **Energy (EARTH eq. 1):** per active macro carrier P = P0 + Δp·ρ·P_max·10^(o/10), where o is the offset in dB. P0 does
  not scale (EARTH: "P0 is the power consumption at the minimum non-zero output power"). For o > 0 the linear model is
  **extrapolated** beyond EARTH's fitted range (P_out ≤ P_max = 20 W per TRX). This is flagged, not corrected.
- **Scope: macro sectors only** [S→]. EARTH cl. III-B says pico load dependency is negligible: Δp·P_max = 4.0 × 0.13 W =
  0.52 W against P0 = 6.8 W. A pico power reduction therefore saves nothing, and picos save energy only by sleeping.

### 2.2 Protected slice and min share `("prot_min", c)`
- Protected UEs are a fixed subset of eMBB UEs drawn from their own RNG stream (`"e6p"`). They keep eMBB traffic
  (FTP model 3, 0.5 MB files = TR38864 Table A-2) and still count in every eMBB statistic.
- Scheduler pool order per cell: (1) the LL **dedicated** pool `ll_ratio` (TS28541 rRMPolicyDedicatedRatio: not shared
  even when idle; E6 semantics unchanged). (2) The protected **min share** `min(cap·prot_min, cap − dedicated)`:
  protected UEs are served from it first with equal share, and whatever they leave unused returns to the shared pool
  in the same tick (TS28541 prioritized resources = rRMPolicyMinRatio − rRMPolicyDedicatedRatio, **work-conserving**).
  (3) The shared pool: LL remainder, eMBB (protected residual included) and BE, equal share.
- TS28541 constraint: the sum of min ratios must be ≤ 100 %. This holds because `ll_ratio ≤ 0.5` and `prot_min ≤ 0.5`,
  and in the screen `ll_ratio` stays at 0.
- **KPM:** `prb_util` must count only PRBs actually used plus idle **dedicated** PRBs. Idle min-share PRBs are borrowable
  and must not appear as used or reserved. Mechanism test M2(d) checks this.

### 2.3 Per-UE floor (primary SLA of the protected slice)
A protected UE-second is violated if the UE was backlogged ≥ 0.2 s in that second and its user-perceived throughput was
below the floor, or if it was in RLF outage or out of coverage. This is the same activity gate and outage rule as the
E6 eMBB SLA [I]. The 1 s evaluation deviates from the TS23501 GBR averaging window (2000 ms [V]); it is kept for
comparability with `E6_METRIC.md` and stated as a deviation.

## 3. Parameter values

| field | value | tag | justification |
|---|---|---|---|
| `ptx_scope` | `"macro"` | [S→] | EARTH cl. III-B (see 2.1) |
| `ptx_range_db` | **(-9.0, +3.0)** | [S] | Lower bound: TR38864 Table 6.4.1.2-1, the -3/-6/-9 dB grid evaluated by MTK and Nokia. ES gain there is flat or falling beyond -6 dB while UPT loss keeps rising (MTK Cat 1 light: 8.7 / 11.1 / 9.0 % gain, 2.0 / 5.7 / 10.8 % UPT loss). Upper bound: TR38864 Table 5.1-1 Set 2 (FR1 FDD, 20 MHz, 15 kHz, the E6 carrier) total DL power 49 dBm = E6 46 dBm + 3 dB. |
| `ptx_init_db` | 0.0 | [I] | E6 nominal 46 dBm (TR 36.814) |
| `ptx_grid_db` | 1.0 | [S] | TS38331 integer-dB power parameters; lets the WG3 MODIFY (half-step) action be represented |
| `ptx_max_step_db` | 3.0 | [S→] | one TR38864 evaluation step per write |
| `ptx_min_interval_s` | 10.0 | [A] | equals the power xApps' cadence; no source for a power-change dwell |
| `prot_on` | True (all pairs) | design | one plant for P1-P3; the freeze arm is shared |
| `prot_frac` | **0.20 of eMBB UEs (≈ 11 % of all UEs)** | [A] | No source. Chosen from spec constants only, not from outcomes, so that the floor is feasible under a same-rights subset (the precondition for conflict-limited harm, PATH2_SCOUT_PY). About 33 protected UEs over 24 cells gives ~1.4 per cell, so the aggregate floor demand of ~3 Mb/s is well below one 53-PRB carrier (≈ 53 × 135 kHz × 1.5 bit/s/Hz ≈ 10.7 Mb/s). A single edge UE, however, needs 14.8/SE PRBs: 25 at 0 dB, 42 at -3 dB, 76 at -6 dB (E6 SE = 0.6·log2(1+SINR)). So edge floors become fragile when a carrier is off or power is lowered, and that fragility is the mechanism under test. |
| `prot_floor_bps` | **2.0e6** | [S] + [I] | TR26925 cl. 5.1 "720p HD: 2 - 5 Mbps" (lower end), carried as 5QI 4 GBR (TS23501 Table 5.7.4-1). This is also the E6 eMBB UPT target `EMBB_THP_TARGET_BPS`, so no new threshold is introduced. It **replaces the implementer's provisional 3e6**. |
| `prot_min_backlog_s` | 0.2 | [I] | E6 eMBB SLA activity gate |
| `prot_min_init` | **0.0** | [S] | TS28541 cl. 4.4.1 rRMPolicyMinRatio defaultValue 0. The freeze arm is the 3GPP default configuration (as in E6). |
| `prot_min_range` | (0.0, 0.5) | [I] | E6 `ll_ratio` actuator range (ric.LIMITS); keeps the sum of min ratios ≤ 100 % |
| `prot_min_grid` | 0.05 | [I] | E6 `ll_ratio` quantum (TS28541 allows integer %) |
| `prot_min_max_step` | 0.10 | [I] | E6 `ll_ratio` max step |
| `prot_min_interval_s` | 5.0 | [A] | equals the SliceGuarantee cadence (the v1 `ll_ratio` dwell of 1 s matched the v1 SLICE cadence of 1 s) |
| `edge_pct` | 5.0 | [S] | M2410 cl. 4.4: the 5th-percentile user is the edge-user statistic |
| `ll_ratio` (not actuated in the screen) | 0.0 | [S] | TS28541 rRMPolicyDedicatedRatio defaultValue 0. The v1 LL SLICE xApp is not deployed. The dedicated semantics stay documented (2.2) and unchanged. |

### 3.1 Load levels (TR38864 Annex A Table A-1 mapped onto the E6 load knob)
E6 load is set by `load_factor` (offered traffic scale; `load="medium"` → 1.8 in E6-scn-v1). E6-P does **not** reuse
the E6 names. It declares two strata by their TR38864 load L:

| E6-P stratum | TR38864 class | target L | evidence that the value is a TR38864 evaluation point |
|---|---|---|---|
| `L10` | low (0 < L ≤ 15) | **10 %** | RU = 10 % (OPPO, ZTE rows, Tables 6.1.1.2-x / 6.4.1.2-1) |
| `L40` | medium (30 < L ≤ 50) | **40 %** | class midpoint; cf. RU 38.4 % (ZTE) and 42 % (Samsung) rows |

L is measured in E6 as follows: the network mean, over all 24 cells (equal weight) and all scored seconds, of used PRBs
divided by `N_PRB` (full band). It is measured in the **freeze** arm of the E6-P plant (ptx 0, prot_min 0, all carriers
on, no sleep), scenario `base`, with the E6 ramp (0.4→1.0) and hidden log-OU load unchanged. This matches the TR
definition (resources used for UE-specific PDSCH; E6 has no common-channel PRBs). `load_factor` per stratum is found
by the calibration procedure in `E6P_SCREEN_PROTOCOL.md` §4. Only PRB counters are read, so no SLA or protected
quantity is computed. The found values are then frozen into the result record. The surge scenario uses the same
`load_factor` as base.

## 4. xApps (all v1 conventions kept: act on delivered KPM + applied config only; per-seed implementation variant, i.e.
threshold scale U(0.75, 1.25) on **dimensionless** thresholds only, blocked-action behaviour, cadence phase)

### 4.1 ES (carrier + pico sleep), xApp name `"ES"`
- **Macro capacity-carrier part: the v1 `xapps.ES` class unchanged [I].** It switches a carrier off when full-band
  load < 0.30 and the macro-neighbour mean < 0.70 have held for 60 s, and back on when active-capacity util > 0.80;
  carrier dwell 120 s; reactivation 2 s. Anchors: 0.30 = TR38864 Table A-1 light/medium boundary; "network may turn
  off SCell for energy saving when traffic load is low" (TR38864 cl. 6.2.1.2); the structure is the TS28541
  LoadTimeThreshold (loadThreshold %, timeDuration s).
- **Pico part (new; same constants, TS28541 cl. 4.3.58 roles)** [I values / S structure]. The pico is the *original
  cell*. Its *candidate cell* is the macro cell with the strongest gain at the pico site. The pico sleeps when
  pico full-band load < 0.30 (intraRatEsActivationOriginalCellLoadParameters) **and** candidate full-band load < 0.70
  (…CandidateCellsLoadParameters) have both held for 60 s (timeDuration). It wakes when candidate active-capacity util
  > 0.80 (intraRatEsDeactivationCandidateCellsLoadParameters). `sleep` dwell 120 s and wake delay `PICO_WAKE_S` 5 s
  are unchanged [I]. Note: TR38864 Table 5.1-4 lists a 10 s deep-sleep transition for BS Category 2. E6 keeps 5 s,
  and the discrepancy is reported.

### 4.2 PowerES (`"PowerES"`, macro Tx-power back-off; TR38864 D-1)
| field | value | tag | note |
|---|---|---|---|
| `pes_cadence_s` | 10 | [I] | v1 ES cadence |
| `pes_u_low` | 0.30 full-band | [I]/[S→] | same trigger as carrier ES (TR38864 light/medium boundary) |
| `pes_hold_s` | 60 | [I] | v1 ES low-load spell |
| `pes_step_db` | 3 | [S→] | one TR38864 grid step; floor -9 dB |
| `pes_u_high` | **0.80 of ACTIVE capacity** | [I] | same as v1 ES `u_on`: restore by +3 dB (never above 0 dB). **The implementer's code compares full-band load: change it (see §6).** |

### 4.3 Coverage (`"Coverage"`, CCO-style Tx-power raise; TS 28.313 CCO, TR 36.902 cl. 4.3)
| field | value | tag | note |
|---|---|---|---|
| `cov_cadence_s` | 10 | [A] | same clock as PowerES (a direct same-knob conflict needs comparable rates) |
| `cov_sinr_low_db` | **-6.0** | [S→] | Q_in (E6 `QIN_DB`, TR 36.839): a 5th-percentile edge below the in-sync level is a coverage problem |
| `cov_sinr_ok_db` | **-3.0** | [S→] | Q_in + one 3 dB power step: release a positive offset only if the edge would stay above Q_in after one step down |
| `cov_margin_db` | **3.0** | [S→] | raise only if the cell's edge is at least one power step below its neighbours' mean edge (so a uniform raise, which is a no-op when interference-limited, is not taken) |
| `cov_floor_frac` | **0.05** | [S→] | protected below-floor share of evaluated protected UE-s that also triggers a raise: 5th-percentile rule (M2410), i.e. the floor must hold for ≥ 95 % of protected UE-time |
| `cov_step_db` | 3.0 | [S→] | TR38864 grid; ceiling +3 dB. It may raise from any value (including PowerES-lowered values) and releases only positive offsets. |

### 4.4 SliceGuarantee (`"SliceGuarantee"`, protected min-share adaptation; ConMit cl. 4.2.1 GBR example)
| field | value | tag | note |
|---|---|---|---|
| `sg_cadence_s` | 5 | [I] | v1 SLICE decision window = last 5 s of `fast` reports |
| `sg_viol_hi` | **0.05** | [S→] | same 95 % availability rule as `cov_floor_frac` |
| `sg_viol_lo` | **0.01** | [A] | release only when violations are ≤ 1/5 of the trigger (hysteresis) |
| `sg_slack` | 0.5 | [I] | v1 SLICE "reservation mostly wasted" ratio (idle > 0.5) |
| `sg_hold_s` | **10** | [I] | v1 SLICE good-spell before a step down (implementer provisional 30 s is **replaced**) |
| `sg_step` | 0.05 | [I] | v1 SLICE step |

## 5. Metrics
- **Primary: PSVR** = protected-floor violation UE-seconds per protected UE-hour over the scored window
  (`3600 · sla["prot_viol"] / sla["prot_ue_s"]`). Every protected UE is in the denominator.
- **Energy:** scored-window kWh (EARTH, as in E6). **Matched-energy rule:** for a pair with energy-side xApp set A
  (P1: ES; P2: PowerES; P3: ES + PowerES), an arm a is *energy-matched* in a stratum iff
  `E_freeze − E_a ≥ X · (E_freeze − E_A-alone)` with energies summed over the stratum's seeds, and **X = 0.90**.
  - Why 0.90: it is the goal-retention fraction proposed in `E6_METRIC.md` §2 (2026-09-27, before E6-P existed), so it
    adds no new degree of freedom.
  - It makes "never save energy" (retention 0) and any arm that vetoes most shutdowns fail, which is the degenerate
    solution PATH2_SCOUT_PY warns about.
  - The give-back it allows is small in absolute terms. TR38864 reports ES gains of a few % to ~25 % for these
    techniques, so 10 % of that is ≤ ~2.5 % of network energy.
  - A stratum is screenable for a pair only if `E_freeze − E_A-alone ≥ 0.01 · E_freeze` [A]. With a smaller saving
    the energy side does nothing to protect against.
- **Guardrails** (non-inferiority vs accept-all, pooled per stratum, margin 1.10 [I] `E6_METRIC.md` §2): all-UE SVR,
  non-protected eMBB violated UE-s, LL violated UE-s and RLF each ≤ 1.10 × accept-all.
- **Secondaries (descriptive):** all-UE SVR; per-slice violations; energy retention; carrier-off, pico-sleep and ptx
  time profiles; HO / RLF / ping-pong; applied knob changes (churn); split of protected violations into SINR-limited
  (the UE alone on the full band would miss the floor) and capacity-limited.

## 6. Items the current E6 / E6-P code cannot represent or must change (implementer attention)
1. **Pico sleep has no xApp.** v1 `ES` proposes only `("carrier", c)`, and no xApp writes `("sleep", pico)`. Add the
   pico part of 4.1 under the `"ES"` name without changing v1 `ES` behaviour for macros.
2. **Pico sleep drops served UEs into RLF.** The gain column goes to -300 dB, UEs fall through T310, and RLF plus 1 s
   outage follows. Real NES hands UEs over first (TR38864 cl. 6.5.2, CHO). Required: at sleep, hand every served UE
   over to its best remaining cell with the normal `HO_EXEC_S` interruption. Otherwise ES×SLICE harm is inflated by an
   artefact. This is checked by M6. Gate it under `cfg.e6p` so E6-scn-v1 stays bit-identical.
3. **PowerES restore trigger** must use active-capacity util (`prb_util`) > 0.80, not full-band load > u_high (§4.2).
4. **Provisional values to replace:** `ptx_range_db` (-6, 3) → (-9, 3); `prot_floor_bps` 3e6 → 2e6; `pes_step_db`
   1 → 3; `pes_u_high` 0.70 → 0.80 (active); `cov_*`, `sg_*` per §4; `sg_hold_s` 30 → 10.
5. **EARTH extrapolation for o > 0** (2.1): keep the formula and flag it in results.
6. **EARTH sleep-power mapping (pre-existing E6 inconsistency).** E6 charges an off carrier `MACRO_SLEEP_W/NTRX` =
   37.5 W while an on carrier draws P0 = 130 W. EARTH eq. (1) gives P_sleep = 75 W **per TRX**, and E6 maps EARTH's two
   TRX per sector onto two carriers. The E6 carrier-off saving is therefore larger than a per-TRX EARTH reading. E6-P
   keeps E6 constants for comparability. Matched-energy is a ratio, so it is insensitive to a common scale, but in P3
   the carrier-vs-ptx saving mix depends on this choice. Report it.
7. **QACM and other published arbiters** (`published.py`) need KPI and knob-type entries for the new xApps: ES →
   energy; PowerES → energy; SliceGuarantee → PSVR; Coverage → 5th-percentile edge SINR; knob types `ptx`, `prot_min`,
   `sleep`. If this is not possible faithfully, the arm is omitted and the omission reported.
8. **New arbiters:** cell-priority lock and per-region hindsight static (`E6P_SCREEN_PROTOCOL.md` §5). The oracle also
   needs the per-second cumulative energy traces of the freeze and energy-side-alone arms on the same tape.
9. **Load naming:** E6-P strata are `L10` / `L40` set through `load_factor`. Do not pass `load="low"`, which E6 maps to
   the "high" default via `lf()`.
10. **Seed registry:** the ranges in `E6P_SCREEN_PROTOCOL.md` §3 must be added to `SEED_REGISTRY.json` (E6 world) by
    the owner of that file before any run.
11. **KPM prb_util** must exclude idle min-share PRBs (2.2). The current `ric.py` adds only `prb_rsv` (dedicated), so
    this is compliant as long as the min share is never added to `prb_rsv`.

## 7. Not modelled / declared limitations
PDSCH-only power offset (no border move); BS power states and transition energies of TR38864 Tables 5.1-2..5.1-5 (E6
keeps EARTH); per-carrier SINR (E6 uses a band-averaged approximation); GBR 2 s averaging window; UL. The two load
strata do not cover TR38864 "light" (15-30 %) or idle. That is a declared choice made before outcomes, not a finding.
