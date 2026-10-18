# DESIGN.md: Figure Design Guide

How every figure and diagram in the journal paper looks (design "soft modern"). The code in
`packages/figures/src/oran_causal/figures/` implements this guide; if they disagree, fix the code or change this
guide deliberately. This file states current rules only.

## 1. Principles

1. **One message per figure.** Before drawing, write the one sentence the figure proves. Anything that does not serve
   it goes to a table, the appendix or the ESM. A figure never tries to show everything.
2. **Lean, but complete elsewhere.** Dropping a result from a figure is fine only if a table or the ESM prints it.
3. **One hero.** The proposed method is the only filled, heavy series. Everything else is quieter context.
4. **One palette.** Colour is never the only cue: every series also has a marker, a line style or a label, and
   figures survive greyscale and colour-vision deficiency.
5. **Built at print size, never rescaled.** 8 pt in the code is 8 pt on the page.
6. **No titles inside figures.** The caption names the figure.
7. **No invented data.** Paper figures read `data/records` read-only (`oran_causal.figures.data`).
8. **No internal labels.** No study codes, regime codes, claim codes or protocol tags (G1, R2, E4, K0, CS:, MG: ...)
   in any visible text. Use plain names.

## 2. Sizes

The `sn-jnl` text block is 5.15 in wide.

| Use | Width | LaTeX | Code |
|---|---|---|---|
| Full width | 5.15 in | `width=\textwidth` | `plots.figure("full")` |
| Half width | 2.5 in | `width=0.49\textwidth` | `plots.figure("half")` |

- Default height 0.55 × width; maximum 4.5 in.
- One PDF per figure, panels (a, b) drawn inside it. At most three panels in a body figure.
- A data panel is at least 1.0 in wide. If facets do not fit, restructure; never shrink them.

## 3. Type

The journal requires Helvetica or Arial at 8-12 pt. Every figure uses **Arial**; no text is below **8 pt**.

| Element | Size |
|---|---|
| Tick labels, legend, annotations, callouts, cell values, node and diagram labels | 8 pt |
| Panel tags (short words beside the letter) | 8 pt bold |
| Axis labels (sentence case, units in parentheses) | 9.5 pt |
| Panel letters (bold lowercase, top-left, outside the axes) | 11 pt bold |

- All text is ink `#262B33`; never colour text with a series colour. Use a true minus sign.
- Panel header: letter and optional tag on one baseline above the axes (`layout.panel_header`).
- Arial comes from the system fonts (`style.arial_files()`), embedded as TrueType (`pdf.fonttype = 42`).

## 4. Axes and lines

| Property | Value |
|---|---|
| Panel | Field gray `#EFF1F4` behind the data, no spines, no tick marks (`plots.finish(ax)`) |
| Grid | White, 0.9 pt, on both axes, behind data |
| Tick labels | Ink 8 pt; axis colour `#5A606B` for the few remaining axis lines (colour bars) |
| Hero line / other method line / entity line | 2.0 / 1.2 / 1.4 pt |
| Design-blind method line | Same width, dashed `(0,(4,1.8))` |
| Reference or threshold line | Ink at 80 %, dash `(0,(4,2))`, 0.9 pt, labelled directly, not in the legend |
| Whiskers | Ink 0.8 pt, cap 2 pt |
| Marker | 4.0 pt (hero +0.4); non-hero markers open (white fill) |

- Call `plots.finish(ax)` on every Cartesian axes and `layout.tidy_grid(axs)` on shared axes: every facet keeps its
  field and grid; only the outer panels show tick labels.
- Rates near 0: symmetric-log axis (`layout.rate_axis`), never a clipped floor.
- Direct labels sit within 6 pt of the point they name (`layout.label_line`).
- **Direction cue** only where the better direction is not obvious and matters: "(higher is better)" in the axis
  label, or one ink "better" arrow towards the good corner of a two-objective plane. Never on an axis whose target
  is a value (e.g. a nominal error rate: the reference line says it).
- Never: dual y-axes, 3-D, shadows, gradients, pie charts, `sns.set_theme()` after `style.apply()`.

## 5. Colour

Defined once in `palette.py`; every figure reads it. Contrast is the WCAG ratio against white and against the field
gray; L* is CIELAB lightness (the greyscale value).

| Hue | Hex | On white | On field | L* |
|---|---|---|---|---|
| blue (hero) | `#3767A8` | 5.73 | 5.06 | 43 |
| terracotta | `#C2553A` | 4.51 | 3.99 | 50 |
| teal | `#1A7F8E` | 4.70 | 4.15 | 49 |
| gold | `#9A7414` | 4.30 | 3.80 | 51 |
| purple | `#8B4F9F` | 5.65 | 4.99 | 44 |
| green | `#55803A` | 4.63 | 4.09 | 49 |
| PMRT-Lin (light blue) | `#6E95CC` | 3.07 | 2.72 | 61 |
| QACM (slate) | `#6F8DB8` | 3.40 | 3.00 | 58 |
| INT (lilac) | `#A07AB2` | 3.55 | 3.13 | 57 |

Hues of similar lightness never carry a distinction alone: the marker, the line style (solid or dashed) or a direct
label separates them in greyscale and under protan, deutan and tritan simulation.

| Neutral role | Hex |
|---|---|
| Ink (text, whiskers, oracle, correct edges, arrows) | `#262B33` |
| Axis | `#5A606B` |
| Neutral (diagram outlines, methods without a slot, Coverage) | `#707782` |
| Neutral light (hairlines) | `#9AA0AA` |
| Missed edge | `#8A909C` |
| Spurious edge | terracotta `#C2553A` |
| Field (panel background, null map tiles, diagram containers) | `#EFF1F4` |
| Legend box face / edge | `#F6F7F9` / `#D5D9DF` |

**Continuous maps** (only where a figure has no plainer form): `harbor_seq` for magnitudes (light to blue);
`harbor_div` for signed values, symmetric about 0; rates centred at the nominal 0.05. No jet, rainbow or hot.

**Semantic colour (paper-wide meaning; `palette.OURS`, `GOOD`, `BAD`, `DESIGN_ROLE`, `CONFOUNDER`).** Colour carries
meaning, the same in every figure:

| Meaning | Hue |
|---|---|
| Our method, our components | blue |
| Valid, safe, correct, right sign | teal |
| Invalid, unsafe, wrong sign, false edge | terracotta |
| The logged randomisation / logging design | gold |
| Network state, confounding | purple |

Use it on zones, tiles, fills, highlighted rows, arrows and key annotations, not only on lines. Tag these artists
with gid `sem:<role>`. In a method figure, method series keep their own colours; semantic colour goes on zones and
marks of status.

**Zones** (`plots.zone`): a 10 % tint of the role hue, bordered on its limit by a full-strength 0.9 pt line.

**Two encodings, never mixed in one figure:** *methods* (Section 6) or *entities* (Section 7). A figure uses one.

## 6. Methods

- **One fixed colour, marker and line style per method, paper-wide** (`methods.py` is the only place that styles a
  method; figures pass arm codes). No figure draws two methods in one hue.
- **Hero** (PMRT; PMRT-GBM in the controlled environments): filled marker with white edge, 2.0 pt, drawn last.
  Exactly one hero per figure. No legend or caption calls it "proposed" or "best".
- **Every other method:** open marker, 1.2 pt line. Design-based and design-adjusted methods are solid,
  design-blind methods dashed.
- **Marker = base method**; plain and design-adjusted versions of a test share a marker and differ in hue and dash.
- **Bars:** the hero bar full-strength; other bars a 28 % tint with a 0.7 pt edge in their hue.
- **CI bands:** fill only, 0.18 hero, 0.14 others.
- **References:** the oracle (ground-truth map) is an open ink star at 60 %, dash-dot; the random map and static
  rules are neutral. Static rules are rows with their CI, never lines; a constant is a reference line.

| Method (paper name) | Colour | Marker | Line |
|---|---|---|---|
| PMRT | blue `#3767A8` | `o`, filled | solid |
| PMRT-GBM | blue `#3767A8` | `D`, filled | solid |
| PMRT-Lin | light blue `#6E95CC` | `D` (R3 covariates: `H`, filled) | solid |
| Design-adjusted partial correlation | purple `#8B4F9F` | `^` | solid |
| Design-adjusted Granger | terracotta `#C2553A` | `s` | solid |
| Design-adjusted RCoT | neutral `#707782` | `v` | solid |
| Correlation | gold `#9A7414` | `p` | dashed |
| Plain partial correlation | green `#55803A` | `^` | dashed |
| Granger (plain and tuned) | teal `#1A7F8E` | `s` | dashed |
| Granger + BY | terracotta `#C2553A` | `P` | dashed |
| SHAP-GBDT | purple `#8B4F9F` | `>` | dashed |
| Two-tower | green `#55803A` | `<` | dashed |
| QACM | slate `#6F8DB8` | `X` | dashed |
| INT | lilac `#A07AB2` | `h` | dashed |
| RCoT, MSCR, CDL, PC, NOTEARS, SHAP-DAG | neutral (tables or ink rows only) | `v`, `8`, `d` | dashed |
| Oracle map | ink 60 % | `*`, open | dash-dot |
| Random map, static rules | neutral | `+`, `\|` | |

Hues shared between two methods (terracotta, purple, green) are used only by methods that never share a figure.
The frozen PMRT beside PMRT-GBM is an ink reference line, never blue.

## 7. Entities

Entities (xApps, their parameters, KPIs, graph node roles) are peers: matched chroma, filled markers, 1.4 pt lines,
and an ink label on every box. Entity hues never appear in a method figure, and one figure colours one entity type.
`entities.py` is the only place that styles an entity.

### Entities: xApps and KPIs

| xApp (paper name) | Slot | Hex | Marker | Writes |
|---|---|---|---|---|
| ES | 2 | `#C2553A` terracotta | `s` | carrier, sleep |
| Slice Guarantee | 3 | `#1A7F8E` teal | `^` | prot_min |
| Power ES | 4 | `#9A7414` gold | `D` | ptx |
| Coverage | neutral | `#707782` neutral, dashed outline | none | ptx |

| KPI (paper name) | Slot | Hex | Marker | Code |
|---|---|---|---|---|
| radio-link failures | 1 | `#3767A8` blue | `o` | rlf |
| served load | 2 | `#C2553A` terracotta | `s` | load |
| protected-slice violations | 3 | `#1A7F8E` teal | `^` | pv |
| service violations | 4 | `#9A7414` gold | `D` | v |
| energy | 5 | `#8B4F9F` purple | `v` | e |

- Figures use the paper names, never code names (`PowerES`, `SliceGuarantee`). Parameter labels name the writer:
  "sleep (ES)".
- When energy and RLF share a plot, both get a direct label and their marker.
- Causal-graph node roles: action blue circle, KPI teal rounded square, confounder purple dashed circle, placebo
  neutral.

## 8. Legends and signs

**Legends**

- One framed legend row above the panels, centred over the figure (`layout.fig_legend`, `methods.legend`;
  location `outside upper center`). Face `#F6F7F9`, edge `#D5D9DF`, 8 pt, handle length 2.0.
- Order: hero first, then design-based, design-adjusted, design-blind, then reference maps and static rules.
- Columns: as many as fit in one row; when they do not, balanced rows (no row more than one entry longer than
  another). A right-hand column only where a figure asks for it.
- Never inside a data area, never below the panels, never one legend per panel.
- Prefer direct labels and labelled rows to legends; a single series needs no legend.

**Signs**

- Method colour, marker and dash: Section 6, the same in every figure.
- Reference and threshold lines: ink 80 % dashed, labelled at their end.
- Zones: Section 5 (tint plus limit line).
- INVALID condition: an INVALID strip above the recall scale, the method's marker drawn there
  (`layout.invalid_marks`).
- Callouts (`layout.callout`): 8 pt ink, placed automatically where they collide with no data, text or legend; a
  leader line when the text sits more than 6 pt from its anchor.
- Coinciding values: dodged horizontally (`layout.dodge`), said in the caption.

**Causal maps** (tile grids, `fig_maps`): ground-truth edge = 62 % blue tile, white triangle in the sign;
right sign = teal tile, white triangle; wrong sign = terracotta tile, white triangle; false edge = 30 % terracotta
tile, terracotta triangle; missed = white tile, outlined gray triangle in the true sign; not scored = diagonal
hatch on a 22 % neutral tile; true null = the field gray. The hero's facet is framed in blue, 1.1 pt. Row labels and
KPI labels on the left facet only. The legend row shows the tiles themselves.

The gallery sheet `specimen_key` draws this whole key in one figure.

## 9. Mark types

| Chart | Rule |
|---|---|
| Line | Mean over seeds; `markevery` on dense series |
| CI band | Fill only (Section 6) |
| Error bars, whiskers | Ink 0.8 pt, cap 2 pt |
| Few seeds | Box plus every seed as a jittered dot |
| Forest (row-wise) | One labelled row per method, no legend; family headers are banner rows; the whisker breaks at the glyph |
| Heatmap | 1 pt white gaps, values in ink (white on dark cells, WCAG 4.5:1); a colour bar unless values are printed |

## 10. Graphs and diagrams

- **Causal graph:** actions in a left column, KPIs in a right column, edges left to right. Correct edges ink solid;
  spurious terracotta dashed; missed `#8A909C` dotted.
- **Environment truth graph** (appendix): `oran_causal.figures.envgraph`, layered, crossing-minimised, bundled edges.
- **Diagrams** (`oran_causal.figures.diagram`, units in points). One corner radius, 5 pt, for every box.
  - Neutral components: white, 0.8 pt neutral outline.
  - xApp boxes: 12 % tint of their hue, 1.0 pt outline in the hue.
  - The one proposed component (the referee): 18 % hero tint, 1.2 pt outline, bold label.
  - Containers (near-RT RIC, SMO): field-gray cards, no outline, radius 7 pt, bold 8 pt title.
  - Arrows: plain ink 0.7 pt, straight or with a single elbow, dashed for control or feedback; labels as plain 8 pt
    text beside the line, no badges.
- Use O-RAN interface names exactly (E2, A1, O1); do not imply interfaces the work does not implement.

## 11. Springer Nature artwork rules

| Rule | How the figures meet it |
|---|---|
| Fonts Helvetica/Arial (Times/Symbol for maths) | Arial for all text and maths (`mathtext.fontset = custom`) |
| 8-12 pt at final size | 8 to 11 pt, built at print width; checked on every figure |
| Consistent sizing across figures | One type scale (Section 3) and one set of line weights (Section 4) |
| Lines at least 0.25 pt | Thinnest line 0.3 pt (hairline separators); checked on every figure |
| Vector line art | PDF with embedded TrueType fonts |
| No colour-only distinctions | Marker, dash or direct label beside every hue (Section 5) |
| Column widths | 5.15 in (full) and 2.5 in (half), the `sn-jnl` text block |
| No titles inside figures | The caption names the figure |

## 12. Captions

- **A few words, like a title** ("O-RAN conflict types.", "Research questions."). Add n or the interval
  type only when uncertainty is drawn, as a short clause. The figure speaks for itself through
  its data, labels, legend and direct annotation, not through the caption.
- No findings, mechanism, definitions of encodings or re-description of what is shown.
- If a reader needs explanatory text to read the figure, the figure is wrong: redesign it (clearer labels, direct
  annotation, fewer marks) rather than adding words.

## 13. Build and checks

- `uv run oran-causal-figures build <name>` writes `paper/figures/<name>.pdf` (vector) and
  `packages/figures/preview/<name>.png`.
- Name files `fig_<topic>`; the LaTeX label is `fig:<topic>`.
- `uv run pytest packages/figures/tests` must pass. It builds every registered figure and checks text size and
  font, palette, method and entity rules, overlaps, legend placement, anchored labels, tick hygiene, facet width,
  clipped or coinciding markers, and that everything lies on the canvas (`checks.audit`).

**New figure:** a module in `oran_causal/figures/figs/` with a `SPEC` (name, width, sources) and `build()`; data only
through `oran_causal.figures.data`; methods through `methods`, entities through `entities`, layout through `layout`
ending with `layout.finalize(fig)`.
