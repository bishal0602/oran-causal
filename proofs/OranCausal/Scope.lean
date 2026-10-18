module

public import OranCausal.StepUp
public import OranCausal.FDRAsymptotic
public import OranCausal.Exact
import all OranCausal.StepUp
import all OranCausal.FDRAsymptotic
import all OranCausal.Exact

public section

open MeasureTheory Finset Filter
open scoped BigOperators

namespace OranCausal.Scope

/-- The relation of a hypothesis `(f, r, k)`. -/
inductive Relation
  | own
  | nbr
  | far
  deriving DecidableEq

/-- One family `f` and one KPI `k`. Cells `C`, episodes `E`, seconds `T`, assignments `A` of `f`'s modes.
`nbrs c` is the observable neighbour list of cell `c`; `pre t` and `post t` are the windows around an opening at
second `t`; `kpi e c s a` is the potential KPI of cell `c` at second `s` of episode `e` under assignment `a`. -/
structure Model (E C T A : Type*) where
  nbrs : C → Finset C
  pre : T → Finset T
  post : T → Finset T
  kpi : E → C → T → A → ℝ

/-- Key of a possible unit of `f` (the xApp is fixed: one writer per family). -/
structure Key (E C T : Type*) where
  episode : E
  cell : C
  opening : T

variable {E C T A : Type*} [Fintype C] [DecidableEq C] [DecidableEq T]

namespace Model

variable (M : Model E C T A)

/-- Cells that relation `r` covers for a unit on cell `c`: the cell itself, its neighbours other than itself, and
the remaining cells. -/
def cells : Relation → C → Finset C
  | .own, c => {c}
  | .nbr, c => M.nbrs c \ {c}
  | .far, c => univ \ insert c (M.nbrs c)

/-- Pre- and post-window of an opening at second `t`. -/
def window (t : T) : Finset T := M.pre t ∪ M.post t

/-- Per-cell target (`app:proof_setting`): KPI `k` of every cell of relation `r` at every second of the window. -/
def target (r : Relation) (ξ : Key E C T) (a : A) : M.cells r ξ.cell → M.window ξ.opening → ℝ :=
  fun c s => M.kpi ξ.episode c s a

/-- Summed outcome (`sec:estimand`, `y_u`): KPI `k` summed over the cells of `r`, post-window minus pre-window. -/
noncomputable def summed (r : Relation) (ξ : Key E C T) (a : A) : ℝ :=
  ∑ c ∈ M.cells r ξ.cell,
    (∑ s ∈ M.post ξ.opening, M.kpi ξ.episode c s a - ∑ s ∈ M.pre ξ.opening, M.kpi ξ.episode c s a)

/-- Sharp null `H₀(f, r, k)`: every key's per-cell target is the same under any assignment of `f`'s modes. -/
def SharpNull (r : Relation) : Prop :=
  ∀ (ξ : Key E C T) (a a' : A), M.target r ξ a = M.target r ξ a'

/-- Sharp null on the summed outcome of `sec:estimand`. -/
def SummedNull (r : Relation) : Prop :=
  ∀ (ξ : Key E C T) (a a' : A), M.summed r ξ a = M.summed r ξ a'

omit [DecidableEq T] in
lemma mem_cells_nbr {c c' : C} : c ∈ M.cells .nbr c' ↔ c ∈ M.nbrs c' ∧ c ≠ c' := by
  simp [cells]

omit [DecidableEq T] in
lemma mem_cells_far {c c' : C} : c ∈ M.cells .far c' ↔ c ≠ c' ∧ c ∉ M.nbrs c' := by
  simp [cells]

/-- The per-cell sharp null fails iff some cell of `r` changes at some second of some key's window. -/
lemma not_sharpNull_iff (r : Relation) :
    ¬ M.SharpNull r ↔ ∃ (ξ : Key E C T) (a a' : A) (c : C) (s : T),
      c ∈ M.cells r ξ.cell ∧ s ∈ M.window ξ.opening ∧ M.kpi ξ.episode c s a ≠ M.kpi ξ.episode c s a' := by
  constructor
  · intro h
    simp only [SharpNull, not_forall] at h
    obtain ⟨ξ, a, a', h⟩ := h
    obtain ⟨c, hc⟩ := Function.ne_iff.mp h
    obtain ⟨s, hs⟩ := Function.ne_iff.mp hc
    exact ⟨ξ, a, a', c, s, c.2, s.2, hs⟩
  · rintro ⟨ξ, a, a', c, s, hc, hs, hne⟩ h
    exact hne (congrFun (congrFun (h ξ a a') ⟨c, hc⟩) ⟨s, hs⟩)

/-! ### Propagation of an own-cell effect (`sec:reading`) -/

/-- The key of a possible unit of `f` on cell `c'`, in the same episode and with the same opening second as `ξ`.
It has the same pre- and post-windows as `ξ`. -/
@[expose] def _root_.OranCausal.Scope.Key.atCell (ξ : Key E C T) (c' : C) : Key E C T := ⟨ξ.episode, c', ξ.opening⟩

omit [Fintype C] [DecidableEq C] [DecidableEq T] in
@[simp] lemma _root_.OranCausal.Scope.Key.atCell_cell (ξ : Key E C T) (c' : C) : (ξ.atCell c').cell = c' := rfl

/-- Neighbour, per witness: if `f`'s modes change KPI `k` of cell `ξ.cell` at a second of `ξ`'s window, and that
cell is on the neighbour list of another cell `c'`, then `H₀(f, nbr, k)` is false. -/
theorem not_sharpNull_nbr_of_witness {ξ : Key E C T} {a a' : A} {s : T} (hs : s ∈ M.window ξ.opening)
    (hne : M.kpi ξ.episode ξ.cell s a ≠ M.kpi ξ.episode ξ.cell s a') {c' : C} (hc' : ξ.cell ∈ M.nbrs c')
    (hcc' : ξ.cell ≠ c') : ¬ M.SharpNull .nbr :=
  (M.not_sharpNull_iff .nbr).mpr
    ⟨ξ.atCell c', a, a', ξ.cell, s, (M.mem_cells_nbr).mpr ⟨hc', hcc'⟩, hs, hne⟩

/-- The own sharp null fails exactly when some cell's own KPI changes at some second of its window. -/
lemma not_sharpNull_own_iff : ¬ M.SharpNull .own ↔ ∃ (ξ : Key E C T) (a a' : A) (s : T),
    s ∈ M.window ξ.opening ∧ M.kpi ξ.episode ξ.cell s a ≠ M.kpi ξ.episode ξ.cell s a' := by
  rw [M.not_sharpNull_iff .own]
  constructor
  · rintro ⟨ξ, a, a', c, s, hc, hs, hne⟩
    have hc : c = ξ.cell := by simpa [cells] using hc
    subst hc
    exact ⟨ξ, a, a', s, hs, hne⟩
  · rintro ⟨ξ, a, a', s, hs, hne⟩
    exact ⟨ξ, a, a', ξ.cell, s, by simp [cells], hs, hne⟩

/-- Neighbour, per affected cell: if every cell whose own KPI `f` changes is on another cell's neighbour list,
then a false own null makes `H₀(f, nbr, k)` false. -/
theorem not_sharpNull_nbr_of_own (hown : ¬ M.SharpNull .own)
    (hN : ∀ (ξ : Key E C T) (a a' : A) (s : T), s ∈ M.window ξ.opening →
      M.kpi ξ.episode ξ.cell s a ≠ M.kpi ξ.episode ξ.cell s a' → ∃ c', ξ.cell ∈ M.nbrs c' ∧ ξ.cell ≠ c') :
    ¬ M.SharpNull .nbr := by
  obtain ⟨ξ, a, a', s, hs, hne⟩ := (M.not_sharpNull_own_iff).mp hown
  obtain ⟨c', hc', hcc'⟩ := hN ξ a a' s hs hne
  exact M.not_sharpNull_nbr_of_witness hs hne hc' hcc'

/-- Neighbour, uniform network condition: if every cell is on the neighbour list of some other cell, a false own
null makes `H₀(f, nbr, k)` false. -/
theorem not_sharpNull_nbr_of_own_uniform (hown : ¬ M.SharpNull .own)
    (hN : ∀ c : C, ∃ c', c ∈ M.nbrs c' ∧ c ≠ c') : ¬ M.SharpNull .nbr :=
  M.not_sharpNull_nbr_of_own hown fun ξ _ _ _ _ _ => hN ξ.cell

/-! ### Summed outcome (`sec:estimand`, `y_u`) -/

/-- Post-window minus pre-window total of KPI `k` on one cell. -/
noncomputable def cellChange (e : E) (c : C) (t : T) (a : A) : ℝ :=
  ∑ s ∈ M.post t, M.kpi e c s a - ∑ s ∈ M.pre t, M.kpi e c s a

omit [DecidableEq T] in
lemma summed_eq (r : Relation) (ξ : Key E C T) (a : A) :
    M.summed r ξ a = ∑ c ∈ M.cells r ξ.cell, M.cellChange ξ.episode c ξ.opening a := by
  unfold summed cellChange; rfl

omit [DecidableEq T] in
lemma summed_own (ξ : Key E C T) (a : A) :
    M.summed .own ξ a = M.cellChange ξ.episode ξ.cell ξ.opening a := by
  simp [summed_eq, cells]

omit [Fintype C] [DecidableEq C] [DecidableEq T] in
/-- Non-cancellation: if a nonzero change `x` at one cell is accompanied by changes of the same sign (or zero)
at the other cells, their sum is nonzero. -/
lemma sum_ne_zero_of_sameSign {S : Finset C} {Δ : C → ℝ} {c : C} (hc : c ∈ S) (hΔ : Δ c ≠ 0)
    (hsign : ∀ d ∈ S, 0 ≤ Δ d * Δ c) : ∑ d ∈ S, Δ d ≠ 0 := by
  intro h0
  have hle : Δ c * Δ c ≤ ∑ d ∈ S, Δ d * Δ c :=
    single_le_sum (f := fun d => Δ d * Δ c) hsign hc
  rw [← sum_mul, h0, zero_mul] at hle
  exact hΔ (mul_self_eq_zero.mp (le_antisymm hle (mul_self_nonneg _)))

omit [DecidableEq T] in
/-- Neighbour, summed outcome, per witness: if `f`'s modes change the own outcome of `ξ`, its cell is on the
neighbour list of another cell `c'`, and no cell of `c'`'s neighbour sum moves against it (non-cancellation),
then the summed `H₀(f, nbr, k)` is false. -/
theorem not_summedNull_nbr_of_witness {ξ : Key E C T} {a a' : A}
    (hne : M.summed .own ξ a ≠ M.summed .own ξ a') {c' : C} (hc' : ξ.cell ∈ M.nbrs c') (hcc' : ξ.cell ≠ c')
    (hsign : ∀ d ∈ M.cells .nbr c',
      0 ≤ (M.cellChange ξ.episode d ξ.opening a - M.cellChange ξ.episode d ξ.opening a') *
        (M.cellChange ξ.episode ξ.cell ξ.opening a - M.cellChange ξ.episode ξ.cell ξ.opening a')) :
    ¬ M.SummedNull .nbr := by
  intro h
  have hsum := h (ξ.atCell c') a a'
  rw [summed_eq, summed_eq, ← sub_eq_zero, ← sum_sub_distrib] at hsum
  rw [summed_own, summed_own, ne_eq, ← sub_eq_zero] at hne
  exact sum_ne_zero_of_sameSign ((M.mem_cells_nbr).mpr ⟨hc', hcc'⟩) hne hsign hsum

/-- Far, per witness: if `f`'s modes change KPI `k` of cell `ξ.cell` at a second of `ξ`'s window, and that cell is
neither `c'` nor on `c'`'s neighbour list, then `H₀(f, far, k)` is false. -/
theorem not_sharpNull_far_of_witness {ξ : Key E C T} {a a' : A} {s : T} (hs : s ∈ M.window ξ.opening)
    (hne : M.kpi ξ.episode ξ.cell s a ≠ M.kpi ξ.episode ξ.cell s a') {c' : C} (hcc' : ξ.cell ≠ c')
    (hc' : ξ.cell ∉ M.nbrs c') : ¬ M.SharpNull .far :=
  (M.not_sharpNull_iff .far).mpr
    ⟨ξ.atCell c', a, a', ξ.cell, s, (M.mem_cells_far).mpr ⟨hcc', hc'⟩, hs, hne⟩

/-- Far, per affected cell: if every cell whose own KPI `f` changes is far from some cell, then a false own null
makes `H₀(f, far, k)` false. -/
theorem not_sharpNull_far_of_own (hown : ¬ M.SharpNull .own)
    (hF : ∀ (ξ : Key E C T) (a a' : A) (s : T), s ∈ M.window ξ.opening →
      M.kpi ξ.episode ξ.cell s a ≠ M.kpi ξ.episode ξ.cell s a' → ∃ c', ξ.cell ≠ c' ∧ ξ.cell ∉ M.nbrs c') :
    ¬ M.SharpNull .far := by
  obtain ⟨ξ, a, a', s, hs, hne⟩ := (M.not_sharpNull_own_iff).mp hown
  obtain ⟨c', hcc', hc'⟩ := hF ξ a a' s hs hne
  exact M.not_sharpNull_far_of_witness hs hne hcc' hc'

/-- Far, uniform network condition: if every cell is far from some cell, a false own null makes
`H₀(f, far, k)` false. -/
theorem not_sharpNull_far_of_own_uniform (hown : ¬ M.SharpNull .own)
    (hF : ∀ c : C, ∃ c', c ≠ c' ∧ c ∉ M.nbrs c') : ¬ M.SharpNull .far :=
  M.not_sharpNull_far_of_own hown fun ξ _ _ _ _ _ => hF ξ.cell

omit [DecidableEq T] in
/-- Far, summed outcome, per witness: as `not_summedNull_nbr_of_witness`, with `ξ.cell` far from `c'` and
non-cancellation over `c'`'s far sum. -/
theorem not_summedNull_far_of_witness {ξ : Key E C T} {a a' : A}
    (hne : M.summed .own ξ a ≠ M.summed .own ξ a') {c' : C} (hcc' : ξ.cell ≠ c') (hc' : ξ.cell ∉ M.nbrs c')
    (hsign : ∀ d ∈ M.cells .far c',
      0 ≤ (M.cellChange ξ.episode d ξ.opening a - M.cellChange ξ.episode d ξ.opening a') *
        (M.cellChange ξ.episode ξ.cell ξ.opening a - M.cellChange ξ.episode ξ.cell ξ.opening a')) :
    ¬ M.SummedNull .far := by
  intro h
  have hsum := h (ξ.atCell c') a a'
  rw [summed_eq, summed_eq, ← sub_eq_zero, ← sum_sub_distrib] at hsum
  rw [summed_own, summed_own, ne_eq, ← sub_eq_zero] at hne
  exact sum_ne_zero_of_sameSign ((M.mem_cells_far).mpr ⟨hcc', hc'⟩) hne hsign hsum

end Model

/-! ### Necessity of the hypotheses (`sec:reading`) -/

namespace Counterexamples

open Model

/-- Every cell's KPI is 1 when the assignment is `true` and 0 otherwise; one-second post-window, empty pre-window. -/
noncomputable def bumpModel (C : Type*) (nbrs : C → Finset C) : Model Unit C ℕ Bool where
  nbrs := nbrs
  pre := fun _ => ∅
  post := fun t => {t}
  kpi := fun _ _ _ a => if a then 1 else 0

lemma bump_not_own (C : Type*) [Fintype C] [DecidableEq C] (nbrs : C → Finset C) [Nonempty C] :
    ¬ (bumpModel C nbrs).SharpNull .own :=
  (not_sharpNull_own_iff _).mpr
    ⟨⟨(), Classical.arbitrary C, 0⟩, true, false, 0, by simp [window, bumpModel], by simp [bumpModel]⟩

lemma sharpNull_of_cells_empty {E C T A : Type*} [Fintype C] [DecidableEq C] [DecidableEq T]
    (M : Model E C T A) (r : Relation) (h : ∀ c, M.cells r c = ∅) : M.SharpNull r := by
  intro ξ a a'
  funext c
  exact absurd c.2 (by simp [h])

/-- A single cell: `f` changes its own cell, yet the neighbour and far sharp nulls hold, because there are no
neighbour or far cells. The topology hypotheses of steps 3 and 4 cannot be dropped. -/
theorem single_cell :
    ¬ (bumpModel Unit fun _ => ∅).SharpNull .own ∧ (bumpModel Unit fun _ => ∅).SharpNull .nbr ∧
      (bumpModel Unit fun _ => ∅).SharpNull .far :=
  ⟨bump_not_own Unit _,
    sharpNull_of_cells_empty _ .nbr fun c => by simp [cells, bumpModel],
    sharpNull_of_cells_empty _ .far fun c => by simp [cells, bumpModel]⟩

/-- Two cells on each other's neighbour lists: the own and neighbour sharp nulls are false, but the far sharp null
holds because no cell is far from another. "The far hypotheses are then false" needs a far cell. -/
theorem two_cells :
    ¬ (bumpModel (Fin 2) fun _ => univ).SharpNull .own ∧
      ¬ (bumpModel (Fin 2) fun _ => univ).SharpNull .nbr ∧
      (bumpModel (Fin 2) fun _ => univ).SharpNull .far :=
  ⟨bump_not_own (Fin 2) _,
    (bumpModel (Fin 2) fun _ => univ).not_sharpNull_nbr_of_witness (ξ := ⟨(), 0, 0⟩) (a := true)
      (a' := false) (s := 0) (by simp [window, bumpModel]) (by simp [bumpModel]) (c' := 1)
      (by simp [bumpModel]) (by decide),
    sharpNull_of_cells_empty _ .far fun c => by simp [cells, bumpModel]⟩

/-- Three cells; cell 2 lists cells 0 and 1 as neighbours. Accepting raises cell 0's KPI by 1 and lowers cell 1's
by 1. -/
noncomputable def cancelModel : Model Unit (Fin 3) ℕ Bool where
  nbrs := fun c => if c = 2 then {0, 1} else ∅
  pre := fun _ => ∅
  post := fun t => {t}
  kpi := fun _ c _ a => if a then (if c = 0 then 1 else if c = 1 then -1 else 0) else 0

/-- Cancellation in the summed outcome: `f` changes its own cell (summed own null false) and the per-cell neighbour
sharp null is false, yet the summed neighbour null holds, because the effects on cells 0 and 1 cancel in cell 2's
neighbour sum. "Also moves the neighbour and far sums" needs non-cancellation. -/
theorem summed_cancellation :
    ¬ cancelModel.SummedNull .own ∧ cancelModel.SummedNull .nbr ∧ ¬ cancelModel.SharpNull .nbr := by
  refine ⟨fun h => ?_, ?_, ?_⟩
  · have := h ⟨(), 0, 0⟩ true false
    simp [summed_own, cellChange, cancelModel] at this
  · rintro ⟨e, c, t⟩ a a'
    rw [summed_eq, summed_eq]
    have h2 : ({0, 1} : Finset (Fin 3)) \ {2} = {0, 1} := by decide
    fin_cases c <;> cases a <;> cases a' <;> simp [cellChange, cells, cancelModel, h2]
  · exact cancelModel.not_sharpNull_nbr_of_witness (ξ := ⟨(), 0, 0⟩) (a := true) (a' := false) (s := 0)
      (by simp [window, cancelModel]) (by simp [cancelModel]) (c' := 2) (by simp [cancelModel]) (by decide)

end Counterexamples

/-! ### Declarations of false nulls (`sec:reading`: "declaring them is correct") -/

namespace Declarations

open OranCausal.StepUp OranCausal.FDRAsymptotic

variable {m : ℕ}

/-- The true nulls among `m` hypotheses: those whose sharp null holds. -/
noncomputable def nullSet (null : Fin m → Prop) : Finset (Fin m) := by
  classical exact univ.filter null

lemma mem_nullSet {null : Fin m → Prop} {j : Fin m} : j ∈ nullSet null ↔ null j := by
  classical simp [nullSet]

/-- False discovery proportion of a declaration set `s`, the integrand of `StepUp.fdr`. -/
noncomputable def fdp (H0 s : Finset (Fin m)) : ℝ := ((H0 ∩ s).card : ℝ) / (max s.card 1 : ℕ)

lemma fdr_eq_integral_fdp {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (S : Ω → Finset (Fin m))
    (H0 : Finset (Fin m)) : fdr μ S H0 = ∫ ω, fdp H0 (S ω) ∂μ := by
  unfold fdr fdp; rfl

lemma fdp_nonneg (H0 s : Finset (Fin m)) : 0 ≤ fdp H0 s := by
  unfold fdp; positivity

lemma fdp_le_one (H0 s : Finset (Fin m)) : fdp H0 s ≤ 1 := by
  unfold fdp
  have h : (H0 ∩ s).card ≤ (max s.card 1 : ℕ) := (card_le_card inter_subset_right).trans (le_max_left _ _)
  exact div_le_one_of_le₀ (by exact_mod_cast h) (Nat.cast_nonneg _)

/-- Declaring a hypothesis whose sharp null is false adds no false discovery. -/
theorem inter_insert_of_not_mem {H0 s : Finset (Fin m)} {j : Fin m} (hj : j ∉ H0) :
    H0 ∩ insert j s = H0 ∩ s := by
  ext i; by_cases h : i = j <;> simp_all

/-- If `s'` contains `s` and has the same false discoveries, its false discovery proportion is not larger. -/
theorem fdp_le_of_extra {H0 s s' : Finset (Fin m)} (hsub : s ⊆ s') (hsame : H0 ∩ s' = H0 ∩ s) :
    fdp H0 s' ≤ fdp H0 s := by
  unfold fdp
  rw [hsame]
  refine div_le_div_of_nonneg_left (Nat.cast_nonneg _) ?_ ?_
  · exact_mod_cast (Nat.lt_of_lt_of_le zero_lt_one (le_max_right _ _))
  · exact_mod_cast max_le_max (card_le_card hsub) le_rfl

/-- Declaring a false null never raises the false discovery proportion. -/
theorem fdp_insert_le {H0 s : Finset (Fin m)} {j : Fin m} (hj : j ∉ H0) :
    fdp H0 (insert j s) ≤ fdp H0 s :=
  fdp_le_of_extra (subset_insert j s) (inter_insert_of_not_mem hj)

/-- Adding declarations `D` of false nulls never raises the false discovery proportion. -/
theorem fdp_union_le {H0 s D : Finset (Fin m)} (hD : Disjoint D H0) :
    fdp H0 (s ∪ D) ≤ fdp H0 s := by
  refine fdp_le_of_extra subset_union_left ?_
  rw [inter_union_distrib_left, (disjoint_iff_inter_eq_empty.mp hD.symm), union_empty]

section Probability

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]

lemma integrable_fdp {S : Ω → Finset (Fin m)} (hS : Measurable S) (H0 : Finset (Fin m)) :
    Integrable (fun ω => fdp H0 (S ω)) μ :=
  integrable_of_le_of_le ((measurable_of_countable (fdp H0)).comp hS).aestronglyMeasurable
    (Eventually.of_forall fun ω => fdp_nonneg H0 (S ω))
    (Eventually.of_forall fun ω => fdp_le_one H0 (S ω)) (integrable_const (0 : ℝ))
    (integrable_const (1 : ℝ))

/-- Declarations that add only false nulls to `S` do not raise the FDR. -/
theorem fdr_union_le {S D : Ω → Finset (Fin m)} (hS : Measurable S)
    (hSD : Measurable fun ω => S ω ∪ D ω) (H0 : Finset (Fin m)) (hD : ∀ ω, Disjoint (D ω) H0) :
    fdr μ (fun ω => S ω ∪ D ω) H0 ≤ fdr μ S H0 := by
  rw [fdr_eq_integral_fdp, fdr_eq_integral_fdp]
  exact integral_mono (integrable_fdp μ hSD H0) (integrable_fdp μ hS H0) fun ω => fdp_union_le (hD ω)

/-- THM-FDR-ASY with the true nulls taken as the hypotheses whose sharp null holds: the weighted BY step-up,
together with any further declarations `D n` of hypotheses whose sharp null is false, has asymptotic FDR at
most `q`. A declared edge is a false discovery only if its sharp null holds; its sign, relation and magnitude
do not enter. -/
theorem limsup_fdr_le_with_false_nulls {m : ℕ} (hm : 1 ≤ m) {q : ℝ} (hq : 0 ≤ q)
    (w : ℕ → Fin m → ℝ) (hw : ∀ n j, 0 ≤ w n j) (hsum : ∀ n, ∑ j, w n j = m)
    (null : Fin m → Prop) (p : Fin m → ℕ → Ω → ℝ) (hp : ∀ j n, Measurable (p j n))
    (T : ℕ → Ω → Finset (Fin m)) (hT : ∀ n, Measurable (T n))
    (hnull : ∀ j, null j → NullLimit μ (p j)) (D : ℕ → Ω → Finset (Fin m))
    (hD : ∀ n ω, ∀ j ∈ D n ω, ¬ null j)
    (hDm : ∀ n, Measurable (D n)) :
    limsup (fun n => fdr μ (fun ω => rejections q w p T n ω ∪ D n ω) (nullSet null)) atTop ≤ q := by
  have hR : ∀ n, Measurable (rejections q w p T n) := fun n => by
    unfold rejections
    exact measurable_stepUp q (w n) (p := fun ω j => p j n ω) (fun j => hp j n) (hT n)
  have hRD : ∀ n, Measurable fun ω => rejections q w p T n ω ∪ D n ω := fun n =>
    (measurable_of_countable fun x : Finset (Fin m) × Finset (Fin m) => x.1 ∪ x.2).comp
      ((hR n).prodMk (hDm n))
  obtain ⟨h1, h2⟩ := limsup_fdr_le μ hm hq w hw hsum (nullSet null) p hp T hT
    (fun j hj => hnull j (mem_nullSet.mp hj))
  refine le_trans ?_ (h1.trans h2)
  refine limsup_le_limsup (Eventually.of_forall fun n => ?_)
    (isCoboundedUnder_le_of_le atTop fun n => fdr_nonneg μ _ _)
    (isBoundedUnder_of ⟨1, fun n => fdr_le_one μ _ _⟩)
  refine fdr_union_le μ (hR n) (hRD n) _ fun ω => disjoint_left.mpr fun j hj hj0 => ?_
  exact hD n ω j hj (mem_nullSet.mp hj0)

end Probability

end Declarations

/-! ### What the FDR bound does not cover (`sec:reading`, `tab:guarantees`: sign, relation, magnitude) -/

namespace NonCoverage

open OranCausal.Exact Declarations Model

variable {m : ℕ}

/-- Directional false discovery proportion: a declaration is an error if its sharp null holds or its reported
sign is wrong (`wrong` = declared hypotheses whose sign is wrong). -/
noncomputable def dirFdp (H0 wrong s : Finset (Fin m)) : ℝ :=
  ((s ∩ (H0 ∪ wrong)).card : ℝ) / (max s.card 1 : ℕ)

/-- The FDR's integrand never exceeds the directional one. -/
lemma fdp_le_dirFdp (H0 wrong s : Finset (Fin m)) : fdp H0 s ≤ dirFdp H0 wrong s := by
  unfold fdp dirFdp
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  rw [inter_comm]
  exact_mod_cast card_le_card (inter_subset_inter_left subset_union_left)

/-- Sign is not covered: one declared hypothesis whose sharp null is false and whose sign is wrong has false
discovery proportion 0 and directional false discovery proportion 1. -/
theorem sign_not_covered (j : Fin m) : fdp ∅ {j} = 0 ∧ dirFdp ∅ {j} {j} = 1 := by
  simp [fdp, dirFdp]

/-- Score indicator `L(accept) = 1`, `L(reject) = 0`. -/
def L (b : Bool) : ℝ := if b then 1 else 0

variable {U : Type*} [Fintype U] [DecidableEq U]

/-- Marginal of one unit's mode under independent draws with propensities `π`. -/
lemma sum_mass_mul_coord (π : U → ℝ) (u : U) (f : Bool → ℝ) :
    ∑ x : U → Bool, assignmentMass π x * f (x u) = ∑ b, bitMass (π u) b * f b := by
  have key : ∀ x : U → Bool, assignmentMass π x * f (x u) =
      ∏ i, (bitMass (π i) (x i) * (if i = u then f (x i) else 1)) := by
    intro x
    rw [prod_mul_distrib, prod_ite_eq' univ u (fun i => f (x i))]
    simp [assignmentMass]
  simp_rw [key]
  rw [← Fintype.prod_sum (fun i b => bitMass (π i) b * (if i = u then f b else 1))]
  have hfac : ∀ i, ∑ b, bitMass (π i) b * (if i = u then f b else 1) =
      if i = u then ∑ b, bitMass (π u) b * f b else 1 := by
    intro i
    by_cases h : i = u
    · subst h; simp
    · simp only [h, ite_false, mul_one]
      exact bitMass_sum _
  simp_rw [hfac]
  rw [prod_ite_eq' univ u]
  simp

/-- Linear part of the PMRT statistic `S = Σ v_u w_u` (`eq:score`, `eq:stat`) with weight
`w_u = g_u + h_u y_u`: a predictable offset `g_u` (prediction and centring) plus a predictable filter gain `h_u`
times the outcome `y_u = y0_u + τ_u L(W_u)`, whose unit effect is `τ_u`. -/
noncomputable def stat (sgn π g h y0 τ : U → ℝ) (x : U → Bool) : ℝ :=
  ∑ u, sgn u * (L (x u) - π u) * (g u + h u * (y0 u + τ u * L (x u)))

/-- The drift of `S` under independent draws is `Σ_u sgn_u h_u τ_u π_u (1 - π_u)`: it weighs each unit's effect by
its filter gain, not equally. -/
theorem expected_stat (sgn π g h y0 τ : U → ℝ) :
    ∑ x, assignmentMass π x * stat sgn π g h y0 τ x = ∑ u, sgn u * h u * τ u * (π u * (1 - π u)) := by
  simp_rw [stat, mul_sum]
  rw [sum_comm]
  refine sum_congr rfl fun u _ => ?_
  rw [sum_mass_mul_coord π u
    (fun b => sgn u * (L b - π u) * (g u + h u * (y0 u + τ u * L b)))]
  simp [bitMass, L]
  ring

/-- Sign is not covered (drift): two units with positive filter gains `1/10` and `1`, propensity `1/2`, effects
`2` and `-1`. The average effect is `1/2 > 0`, yet the expected statistic is `-1/5 < 0`, so the reported sign,
that of `S` times a fixed orientation, points the wrong way on average. -/
theorem sign_reversal :
    (∑ u : Fin 2, (1 : ℝ) * ![2, -1] u) / 2 = 1 / 2 ∧
      ∑ x, assignmentMass (fun _ : Fin 2 => (1 / 2 : ℝ)) x *
        stat (fun _ => 1) (fun _ => 1 / 2) (fun _ => 0) ![1 / 10, 1] (fun _ => 0) ![2, -1] x = -1 / 5 := by
  refine ⟨by simp [Fin.sum_univ_two]; norm_num, ?_⟩
  rw [expected_stat]
  simp [Fin.sum_univ_two]
  norm_num

/-- Two units on cells 0 and 1, outcome own-cell KPI; accepting raises cell 0 by 1 and lowers cell 1 by 1. -/
noncomputable def magnitudeModel : Model Unit (Fin 2) ℕ (Fin 2 → Bool) where
  nbrs := fun _ => ∅
  pre := fun _ => ∅
  post := fun t => {t}
  kpi := fun _ c _ a => ![1, -1] c * L (a c)

/-- Magnitude is not covered: the own sharp null is false, yet for every assignment of the other unit the unit
effects (own outcome, accept minus reject) are `1` and `-1`, so their average, and any average over a logging
policy, is 0. A correct declaration can carry a zero average effect. -/
theorem magnitude_not_covered :
    ¬ magnitudeModel.SharpNull .own ∧
      ∀ (a : Fin 2 → Bool) (t : ℕ),
        ∑ c : Fin 2, (magnitudeModel.summed .own ⟨(), c, t⟩ (Function.update a c true) -
          magnitudeModel.summed .own ⟨(), c, t⟩ (Function.update a c false)) = 0 := by
  refine ⟨(not_sharpNull_own_iff _).mpr ⟨⟨(), 0, 0⟩, fun _ => true, fun _ => false, 0, ?_, ?_⟩, ?_⟩
  · simp [window, magnitudeModel]
  · simp [magnitudeModel, L]
  · intro a t
    simp [summed_own, cellChange, magnitudeModel, L, Fin.sum_univ_two]

end NonCoverage

end OranCausal.Scope

#print axioms OranCausal.Scope.Declarations.fdp_insert_le
#print axioms OranCausal.Scope.Declarations.fdr_union_le
#print axioms OranCausal.Scope.Declarations.limsup_fdr_le_with_false_nulls
#print axioms OranCausal.Scope.Model.not_sharpNull_nbr_of_witness
#print axioms OranCausal.Scope.Model.not_sharpNull_nbr_of_own
#print axioms OranCausal.Scope.Model.not_sharpNull_nbr_of_own_uniform
#print axioms OranCausal.Scope.Model.not_summedNull_nbr_of_witness
#print axioms OranCausal.Scope.Model.not_sharpNull_far_of_witness
#print axioms OranCausal.Scope.Model.not_sharpNull_far_of_own
#print axioms OranCausal.Scope.Model.not_sharpNull_far_of_own_uniform
#print axioms OranCausal.Scope.Model.not_summedNull_far_of_witness
#print axioms OranCausal.Scope.Counterexamples.single_cell
#print axioms OranCausal.Scope.Counterexamples.two_cells
#print axioms OranCausal.Scope.Counterexamples.summed_cancellation
#print axioms OranCausal.Scope.NonCoverage.fdp_le_dirFdp
#print axioms OranCausal.Scope.NonCoverage.sign_not_covered
#print axioms OranCausal.Scope.NonCoverage.expected_stat
#print axioms OranCausal.Scope.NonCoverage.sign_reversal
#print axioms OranCausal.Scope.NonCoverage.magnitude_not_covered
