# DESIGN.md: Figure Design Guide

How every figure and diagram in the journal paper looks. The code in `figures/src/cddfig/` implements this guide;
if they disagree, fix the code or change this guide deliberately. This file states current rules only. The history
of past rulings is in `archive/guides-history/DESIGN_to-2026-10-13.md`; do not add dated rulings here.

## 1. Principles

1. **One message per figure.** Before drawing, write the one sentence the figure proves. Anything that does not serve
   it goes to a table, the appendix or the ESM. A figure never tries to show everything.
2. **Lean, but complete elsewhere.** Dropping a result from a figure is fine only if a table or the ESM prints it.
3. **One hero.** The proposed method is the only filled, heavy series. Everything else is quieter context.
4. **One palette (Harbor).** Colour is never the only cue: every series also has a marker or a label, and figures
   survive greyscale and colour-vision deficiency.
5. **Built at print size, never rescaled.** 8 pt in the code is 8 pt on the page.
6. **No titles inside figures.** The caption names the figure.
7. **No invented data.** Paper figures read CDD-ORAN artifacts read-only.
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

The journal requires Helvetica or Arial at 8-12 pt. Every figure uses **Arial**; no text is below **8 pt**
(`test_text_floor`).

| Element | Size |
|---|---|
| Tick labels, legend, annotations, cell values, node and diagram labels | 8 pt |
| Axis labels (sentence case, units in parentheses) | 9 pt |
| Panel letters (bold lowercase, top-left, outside the axes) | 10 pt |

- All text is ink `#1E2430`; never colour text with a series colour. Use a true minus sign.
- Panel header: letter and optional short tag on one baseline above the axes (`layout.panel_header`).
- Arial comes from the system fonts (`style.arial_files()`), embedded as TrueType (`pdf.fonttype = 42`).

## 4. Axes and lines

| Property | Value |
|---|---|
| Spines | Left and bottom only, offset 3 pt, trimmed to the ticks (`plots.finish(ax)`); bar axes not trimmed |
| Axis and ticks | `#3A4150`, 0.6 pt; ticks outward 2.5 pt, no minor ticks |
| Grid | Horizontal hairline `#E8EAEE`, 0.4 pt, behind data |
| Background | White |
| Hero line / other method line / entity line | 1.9 / 1.1 / 1.3 pt, all solid |
| Reference or threshold line | Ink at 60 %, dash `(0,(4,2))`, 0.7 pt, labelled directly, not in the legend |
| Marker | 3.6 pt (hero +0.4) |

- Call `plots.finish(ax)` on every Cartesian axes and `layout.tidy_grid(axs)` on shared axes (only the outer panels
  show tick labels).
- Rates near 0: symmetric-log axis (`layout.rate_axis`), never a clipped floor.
- Direct labels sit within 6 pt of the point they name (`layout.label_line`).
- **Direction cue** only where the better direction is not obvious and matters: "(higher is better)" in the axis
  label, or one ink "better" arrow towards the good corner of a two-objective plane. Never on an axis whose target
  is a value (e.g. a nominal error rate: the reference line says it).
- Never: dual y-axes, 3-D, shadows, gradients, pie charts, `sns.set_theme()` after `style.apply()`.

## 5. Colour: Harbor

Defined once in `figures/src/cddfig/palette.py`; every figure reads it (`tests/test_palette.py`).

**Six hues:** blue `#4273B0`, terracotta `#C17154`, teal `#3FAFA9`, gold `#CBAA56`, purple `#955C98`,
green `#5C9B5E`. Three tints: PMRT-Lin `#7196C4`, QACM `#8CA7CF`, INT `#B58DB7`.

| Neutral role | Hex |
|---|---|
| Ink (text, whiskers, oracle) | `#1E2430` |
| Axis | `#3A4150` |
| Grid | `#E8EAEE` |
| Gray (static rules, random map, "Other", placebo node) | `#8A909C` |
| Container outline (dashed) | `#A9AFBA` |
| Spurious edge / missed edge | `#B94739` / `#9AA0AA` |

**Continuous maps** (only where a figure has no plainer form): `harbor_seq` for magnitudes (light to blue);
`harbor_div` for signed values, symmetric about 0; rates centred at the nominal 0.05. No jet, rainbow or hot.

**Semantic colour (paper-wide meaning; `palette.OURS`, `GOOD`, `BAD`, `DESIGN_ROLE`, `CONFOUNDER`).** Lean must not
mean grey. Colour carries meaning, the same in every figure:

| Meaning | Hue |
|---|---|
| Our method, our components | blue |
| Valid, safe, correct, right sign | teal |
| Invalid, unsafe, wrong sign, false edge | terracotta |
| The logged randomisation / logging design | gold |
| Network state, confounding | purple |

Use it on zones (tint 0.12), tiles, fills, highlighted rows, arrows and key annotations, not only on lines. Tag
these artists with gid `sem:<role>`. Ink and gray stay for text, axes and references. In a method figure, method
series keep their own colours; semantic colour goes on backgrounds, zones and marks of status.

**Two encodings, never mixed in one figure:** *methods* (Section 6) or *entities* (Section 7). A figure uses one.

## 6. Methods

- **One fixed colour per method, paper-wide** (`methods.py` is the only place that styles a method; figures pass arm
  codes). No figure draws two methods in one hue.
- **Hero** (PMRT; PMRT-GBM in the controlled environments): filled marker with white edge, 1.9 pt, drawn last.
  Exactly one hero per figure. No legend or caption calls it "proposed" or "best".
- **Every other method:** open marker, 1.1 pt line.
- **Marker = base method**; plain and design-adjusted versions of a test share a marker and differ in hue.
- **Bars:** full-strength fill, no edge.
- **References:** the oracle (ground-truth map) is an open ink star, dash-dot; the random map and static rules are gray.
  Static rules are rows with their CI, never lines; a constant is a reference line.
- **Legend order:** design-based, design-adjusted, design-blind, then reference maps and static rules.

| Method (paper name) | Colour | Marker |
|---|---|---|
| PMRT | blue `#4273B0` | `o` |
| PMRT-GBM | blue `#4273B0` | `D` |
| PMRT-Lin | tint `#7196C4` | `D`, open |
| Design-adjusted partial correlation | purple `#955C98` | `^` |
| Design-adjusted Granger | terracotta `#C17154` | `s` |
| Correlation | gold `#CBAA56` | `p` |
| Plain partial correlation | green `#5C9B5E` | `^` |
| Granger (plain and tuned) | teal `#3FAFA9` | `s` |
| Granger + BY | terracotta `#C17154` | `P` |
| SHAP-GBDT | purple `#955C98` | `>` |
| Two-tower | green `#5C9B5E` | `<` |
| QACM | slate `#8CA7CF` | `X` |
| INT | lilac `#B58DB7` | `h` |
| RCoT, MSCR, CDD O-RAN discovery, PC, NOTEARS, SHAP-DAG | gray (tables or ink rows only) | |
| Oracle map | ink 60 % | `*`, open |
| Random map, static rules | gray `#8A909C` | `+`, `\|` |

Hues shared between two methods (terracotta, purple, green) are used only by methods that never share a figure.
The frozen PMRT beside PMRT-GBM is an ink reference line, never blue.

## 7. Entities

Entities (xApps, their parameters, KPIs, graph node roles) are peers: matched chroma, filled markers, 1.3 pt lines,
and an ink label on every box. Entity hues never appear in a method figure, and one figure colours one entity type.
`entities.py` is the only place that styles an entity (`tests/test_entities.py`).

### Entities: xApps and KPIs

| xApp (paper name) | Slot | Hex | Marker | Writes |
|---|---|---|---|---|
| ES | 2 | `#C17154` terracotta | `s` | carrier, sleep |
| Slice Guarantee | 3 | `#3FAFA9` teal | `^` | prot_min |
| Power ES | 4 | `#CBAA56` gold | `D` | ptx |
| Coverage | neutral | `#8A909C` gray, dashed outline | none | ptx |

| KPI (paper name) | Slot | Hex | Marker | Code |
|---|---|---|---|---|
| radio-link failures | 1 | `#4273B0` blue | `o` | rlf |
| served load | 2 | `#C17154` terracotta | `s` | load |
| protected-slice violations | 3 | `#3FAFA9` teal | `^` | pv |
| service violations | 4 | `#CBAA56` gold | `D` | v |
| energy | 5 | `#955C98` purple | `v` | e |

- Figures use the paper names, never code names (`PowerES`, `SliceGuarantee`). Parameter labels name the writer:
  "sleep (ES)".
- When energy and RLF share a plot, both get a direct label and their marker.
- Causal-graph node roles: action blue circle, KPI teal rounded square, confounder purple dashed circle, placebo gray.

## 8. Legends

- Full-width single-panel figure: frameless column to the right. Otherwise a frameless row on top, centred over the
  figure (`layout.fig_legend`; `methods.legend` with `where="auto"` decides).
- Never below the panels or inside a data area.
- Prefer direct labels and labelled rows to legends; a single series needs no legend.

## 9. Mark types

| Chart | Rule |
|---|---|
| Line | Mean over seeds; `markevery` on dense series |
| CI band | Fill only: 0.14 hero, 0.08 others |
| Error bars, whiskers | Ink 0.6 pt, cap 2 pt |
| Few seeds | Box plus every seed as a jittered dot |
| Forest (row-wise) | One labelled row per method, no legend; family headers are light gray banner rows; the whisker breaks at the glyph |
| Heatmap | 1 pt white gaps, values in ink; a colour bar unless values are printed |
| Coinciding values | Dodge horizontally (`layout.dodge`) and say so in the caption |

## 10. Graphs and diagrams

- **Causal graph:** actions in a left column, KPIs in a right column, edges left to right. Correct edges ink solid;
  spurious `#B94739` dashed; missed `#9AA0AA` dotted. Edges are always ink otherwise.
- **Environment truth graph** (appendix): `cddfig.envgraph`, layered, crossing-minimised, bundled edges.
- **Diagrams** (`cddfig.diagram`, units in points): neutral components white with gray outline; xApp boxes 10 % tint
  of their hue with a 1.0 pt outline; the one proposed component (the referee) 16 % hero tint, bold label;
  containers (near-RT RIC, SMO) dashed; arrows ink 0.7 pt, dashed for control or feedback.
- Use O-RAN interface names exactly (E2, A1, O1); do not imply interfaces the work does not implement.
- Figure 1 (conflict types) is a matplotlib module: `cddfig/figs/fig_conflict_types.py` (the old SVG build `cddfig/svg/fig_intro_conflict.py` is retired).

## 11. Captions

- **A few words, like a title** ("O-RAN conflict types.", "Research questions."). Add n or the interval
  type only when uncertainty is drawn, as a short clause. The figure speaks for itself through
  its data, labels, legend and direct annotation, not through the caption.
- No findings, mechanism, definitions of encodings or re-description of what is shown.
- If a reader needs explanatory text to read the figure, the figure is wrong: redesign it (clearer labels, direct
  annotation, fewer marks) rather than adding words.

## 12. Build and checks

- `uv run cddfig build <name>` (in `figures/`) writes the journal's `figures/<name>.pdf` (vector) and
  `figures/preview/<name>.png`.
- Name files `fig_<topic>`; the LaTeX label is `fig:<topic>`.
- `uv run pytest -q` must pass. It builds every registered figure and checks text size and font, palette, method and
  entity rules, overlaps, legend placement, anchored labels, tick hygiene, facet width, clipped or coinciding markers,
  and that everything lies on the canvas.

**New figure:** a module in `figures/src/cddfig/figs/` with a `SPEC` (name, width, sources) and `build()`; data only
through `cddfig.data`; methods through `cddfig.methods`, entities through `cddfig.entities`, layout through
`cddfig.layout` ending with `layout.finalize(fig)`. Check the preview by eye against the approved figures before
calling it done.
