module

public import OranCausal.StepUp
public import Mathlib.Probability.ConditionalExpectation
public import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

public section

open MeasureTheory ProbabilityTheory MeasurableSpace Set
open scoped BigOperators ENNReal Classical

namespace OranCausal.Fresh

variable {Ω K : Type*} [Countable K] [MeasurableSpace K] [MeasurableSingletonClass K]

/-- Information before draw `n`; indices start at zero. -/
@[expose, instance_reducible] def history (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (key : ℕ → Ω → K) :
    ℕ → MeasurableSpace Ω
  | 0 => G0
  | n + 1 => history G0 U key n ⊔ (borel ℝ).comap (fun ω => U (key n ω) ω)

def avoids (key : ℕ → Ω → K) (n : ℕ) (k : K) : Set Ω :=
  {ω | ∀ i < n, key i ω ≠ k}

/-- The sigma-algebra of events whose intersection with `E` is `H`-measurable. -/
@[instance_reducible] def localized (H : MeasurableSpace Ω) (E : Set Ω) (hE : MeasurableSet[H] E) :
    MeasurableSpace Ω where
  MeasurableSet' A := MeasurableSet[H] (A ∩ E)
  measurableSet_empty := by simp
  measurableSet_compl A hA := by
    have heq : Aᶜ ∩ E = E \ (A ∩ E) := by ext ω; simp; tauto
    rw [heq]
    exact hE.diff hA
  measurableSet_iUnion A hA := by
    rw [iUnion_inter]
    exact MeasurableSet.iUnion hA

lemma measurable_select {G : MeasurableSpace Ω} {k : Ω → K} {U : K → Ω → ℝ}
    (hk : Measurable[G] k) (hU : ∀ i, Measurable[G] (U i)) :
    Measurable[G] (fun ω => U (k ω) ω) := by
  intro C hC
  have heq : {ω | U (k ω) ω ∈ C} = ⋃ i : K, k ⁻¹' {i} ∩ U i ⁻¹' C := by
    ext ω
    simp
  change MeasurableSet[G] {ω | U (k ω) ω ∈ C}
  rw [heq]
  exact MeasurableSet.iUnion fun i => (hk (measurableSet_singleton i)).inter (hU i hC)

omit [Countable K] [MeasurableSpace K] [MeasurableSingletonClass K] in
lemma avoids_succ (key : ℕ → Ω → K) (n : ℕ) (k : K) :
    avoids key (n + 1) k = avoids key n k ∩ {ω | key n ω ≠ k} := by
  ext ω
  simp only [avoids, mem_ofPred_eq, mem_inter_iff]
  constructor
  · intro h
    exact ⟨fun i hi => h i (by omega), h n (by omega)⟩
  · rintro ⟨h, hn⟩ i hi
    by_cases he : i = n
    · simpa [he] using hn
    · exact h i (by omega)

/-- Recursion cannot reveal a coordinate on a branch that has never read it. -/
lemma history_localized (G0 H : MeasurableSpace Ω) (U : K → Ω → ℝ)
    (key : ℕ → Ω → K) (target : K) (hG0 : G0 ≤ H)
    (hU : ∀ k, k ≠ target → Measurable[H] (U k))
    (hkey : ∀ n, Measurable[history G0 U key n] (key n)) :
    ∀ n A, MeasurableSet[history G0 U key n] A →
      MeasurableSet[H] (A ∩ avoids key n target) := by
  intro n
  induction n with
  | zero =>
    intro A hA
    simpa [avoids] using hG0 A hA
  | succ n ih =>
    have havoid : MeasurableSet[H] (avoids key n target) := by
      simpa using ih univ MeasurableSet.univ
    have hnext : MeasurableSet[H] (avoids key (n + 1) target) := by
      rw [avoids_succ, inter_comm]
      exact ih _ ((hkey n (measurableSet_singleton target)).compl)
    have hold : history G0 U key n ≤ localized H (avoids key (n + 1) target) hnext := by
      intro A hA
      change MeasurableSet[H] (A ∩ avoids key (n + 1) target)
      have heq : A ∩ avoids key (n + 1) target =
          (A ∩ avoids key n target) ∩ avoids key (n + 1) target := by
        rw [avoids_succ]; ext ω; simp only [mem_inter_iff]; tauto
      rw [heq]
      exact (ih A hA).inter hnext
    have hdraw : Measurable[localized H (avoids key (n + 1) target) hnext]
        (fun ω => U (key n ω) ω) := by
      intro C hC
      change MeasurableSet[H] ((fun ω => U (key n ω) ω) ⁻¹' C ∩ avoids key (n + 1) target)
      have heq : (fun ω => U (key n ω) ω) ⁻¹' C ∩ avoids key (n + 1) target =
          ⋃ k : {k : K // k ≠ target},
            (key n ⁻¹' {k.val} ∩ avoids key n target) ∩ U k.val ⁻¹' C := by
        ext ω
        simp only [mem_inter_iff, mem_preimage, mem_singleton_iff,
          mem_iUnion, Subtype.exists, avoids_succ, mem_ofPred_eq]
        constructor
        · rintro ⟨hC, hpast, hne⟩
          exact ⟨key n ω, hne, ⟨rfl, hpast⟩, hC⟩
        · rintro ⟨k, hk, ⟨he, hpast⟩, hc⟩
          subst k
          exact ⟨hc, hpast, hk⟩
      rw [heq]
      exact MeasurableSet.iUnion fun k =>
        (ih _ (hkey n (measurableSet_singleton k.val))).inter (hU k.val k.property hC)
    intro A hA
    exact (sup_le hold hdraw.comap_le) A hA

@[instance_reducible] def fieldInfo (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) : Option K → MeasurableSpace Ω
  | none => G0
  | some k => (borel ℝ).comap (U k)

@[instance_reducible] def otherInfo (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (k : K) : MeasurableSpace Ω :=
  ⨆ i, ⨆ (_ : i ≠ some k), fieldInfo G0 U i

omit [Countable K] [MeasurableSpace K] [MeasurableSingletonClass K] in
lemma initial_le_other (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (k : K) :
    G0 ≤ otherInfo G0 U k :=
  le_iSup_of_le none (le_iSup_of_le (by simp) le_rfl)

omit [Countable K] [MeasurableSpace K] [MeasurableSingletonClass K] in
lemma measurable_other (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) {j k : K} (h : j ≠ k) :
    Measurable[otherInfo G0 U k] (U j) := by
  apply Measurable.of_comap_le
  exact le_iSup_of_le (some j) (le_iSup_of_le (by simpa using h) le_rfl)

lemma history_le (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (key : ℕ → Ω → K)
    [mΩ : MeasurableSpace Ω]
    (hG0 : G0 ≤ mΩ) (hU : ∀ k, Measurable (U k))
    (hkey : ∀ n, Measurable[history G0 U key n] (key n)) : ∀ n, history G0 U key n ≤ mΩ := by
  intro n
  induction n with
  | zero => exact hG0
  | succ n ih => exact sup_le ih (measurable_select ((hkey n).mono ih le_rfl) hU).comap_le

omit [Countable K] [MeasurableSpace K] [MeasurableSingletonClass K] in
lemma otherInfo_le (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ)
    [mΩ : MeasurableSpace Ω]
    (hG0 : G0 ≤ mΩ) (hU : ∀ k, Measurable (U k)) (k : K) : otherInfo G0 U k ≤ mΩ := by
  apply iSup_le
  intro i
  apply iSup_le
  intro _
  cases i with
  | none => exact hG0
  | some j => exact (hU j).comap_le

omit [Countable K] [MeasurableSpace K] [MeasurableSingletonClass K] in
lemma coordinate_indep_other (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω)
    (hG0 : G0 ≤ mΩ) (hU : ∀ k, Measurable (U k))
    (hind : iIndep (fieldInfo G0 U) μ) (k : K) :
    Indep ((borel ℝ).comap (U k)) (otherInfo G0 U k) μ := by
  have hle : ∀ i, fieldInfo G0 U i ≤ mΩ := by
    intro i; cases i with
    | none => exact hG0
    | some j => exact (hU j).comap_le
  have h := indep_iSup_of_disjoint hle hind
    (S := {some k}) (T := {i | i ≠ some k}) (by simp)
  simpa [otherInfo, fieldInfo] using h

lemma selection_slice_measurable (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ)
    (key : ℕ → Ω → K)
    (hkey : ∀ n, Measurable[history G0 U key n] (key n))
    (n : ℕ) (hfresh : ∀ ω i, i < n → key i ω ≠ key n ω)
    (A : Set Ω) (hA : MeasurableSet[history G0 U key n] A) (k : K) :
    MeasurableSet[otherInfo G0 U k] (A ∩ key n ⁻¹' {k}) := by
  have h := history_localized G0 (otherInfo G0 U k) U key k
    (initial_le_other G0 U k) (fun j hj => measurable_other G0 U hj) hkey n
    (A ∩ key n ⁻¹' {k}) (hA.inter (hkey n (measurableSet_singleton k)))
  have heq : (A ∩ key n ⁻¹' {k}) ∩ avoids key n k = A ∩ key n ⁻¹' {k} := by
    apply inter_eq_left.mpr
    rintro ω ⟨_, hk⟩ i hi
    have he : key n ω = k := hk
    simpa [he] using hfresh ω i hi
  simpa only [heq] using h

/-- The selected fresh coordinate has the common field law and factors against
every event in the recursively generated past. -/
theorem fresh_factorization (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (key : ℕ → Ω → K)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) (ν : Measure ℝ)
    (hG0 : G0 ≤ mΩ) (hU : ∀ k, Measurable (U k))
    (hind : iIndep (fieldInfo G0 U) μ) (hlaw : ∀ k, μ.map (U k) = ν)
    (hkey : ∀ n, Measurable[history G0 U key n] (key n))
    (n : ℕ) (hfresh : ∀ ω i, i < n → key i ω ≠ key n ω)
    (A : Set Ω) (hA : MeasurableSet[history G0 U key n] A)
    (C : Set ℝ) (hC : MeasurableSet C) :
    μ (A ∩ (fun ω => U (key n ω) ω) ⁻¹' C) = μ A * ν C := by
  let E : K → Set Ω := fun k => A ∩ key n ⁻¹' {k}
  have hE : ∀ k, MeasurableSet (E k) := fun k =>
    otherInfo_le G0 U hG0 hU k _ (selection_slice_measurable G0 U key hkey n hfresh A hA k)
  have hdis : Pairwise (fun j k => Disjoint (E j) (E k)) := by
    intro j k hjk
    apply Set.disjoint_left.mpr
    rintro ω ⟨_, hj⟩ ⟨_, hk⟩
    exact hjk (hj.symm.trans hk)
  have hunion : (⋃ k, E k) = A := by
    ext ω
    simp [E]
  have hsplit : A ∩ (fun ω => U (key n ω) ω) ⁻¹' C =
      ⋃ k, E k ∩ U k ⁻¹' C := by
    ext ω
    simp [E, and_assoc]
  rw [hsplit, measure_iUnion
    (fun j k h => (hdis h).mono inter_subset_left inter_subset_left)
    (fun k => (hE k).inter (hU k hC))]
  have hfactor : ∀ k, μ (E k ∩ U k ⁻¹' C) = μ (E k) * ν C := by
    intro k
    have hi := coordinate_indep_other G0 U μ hG0 hU hind k
    have he := ((Indep_iff _ _ μ).mp hi.symm) (E k) (U k ⁻¹' C)
      (selection_slice_measurable G0 U key hkey n hfresh A hA k)
      (Measurable.of_comap_le le_rfl hC)
    rw [he, ← Measure.map_apply (hU k) hC, hlaw k]
  simp_rw [hfactor]
  rw [ENNReal.tsum_mul_right, ← measure_iUnion hdis hE, hunion]

/-- Uniform probability law on the open unit interval. -/
noncomputable def unitUniform : Measure ℝ := volume.restrict (Ioo 0 1)

instance : IsProbabilityMeasure unitUniform := by
  constructor
  simp [unitUniform]

theorem selected_law (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (key : ℕ → Ω → K)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG0 : G0 ≤ mΩ) (hU : ∀ k, Measurable (U k))
    (hind : iIndep (fieldInfo G0 U) μ) (hlaw : ∀ k, μ.map (U k) = unitUniform)
    (hkey : ∀ n, Measurable[history G0 U key n] (key n))
    (n : ℕ) (hfresh : ∀ ω i, i < n → key i ω ≠ key n ω) :
    μ.map (fun ω => U (key n ω) ω) = unitUniform := by
  have hV := measurable_select ((hkey n).mono (history_le G0 U key hG0 hU hkey n) le_rfl) hU
  apply Measure.ext
  intro C hC
  rw [Measure.map_apply hV hC]
  simpa using fresh_factorization G0 U key μ unitUniform hG0 hU hind hlaw hkey n hfresh
    univ MeasurableSet.univ C hC

theorem selected_independent (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (key : ℕ → Ω → K)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG0 : G0 ≤ mΩ) (hU : ∀ k, Measurable (U k))
    (hind : iIndep (fieldInfo G0 U) μ) (hlaw : ∀ k, μ.map (U k) = unitUniform)
    (hkey : ∀ n, Measurable[history G0 U key n] (key n))
    (n : ℕ) (hfresh : ∀ ω i, i < n → key i ω ≠ key n ω) :
    Indep ((borel ℝ).comap (fun ω => U (key n ω) ω)) (history G0 U key n) μ := by
  have hmap := selected_law G0 U key μ hG0 hU hind hlaw hkey n hfresh
  have hV := measurable_select ((hkey n).mono (history_le G0 U key hG0 hU hkey n) le_rfl) hU
  apply (Indep_iff _ _ μ).mpr
  intro D A hD hA
  obtain ⟨C, hC, rfl⟩ := hD
  rw [inter_comm, fresh_factorization G0 U key μ unitUniform hG0 hU hind hlaw hkey n hfresh A hA C hC]
  rw [← Measure.map_apply hV hC, hmap, mul_comm]

lemma unitUniform_Iio {x : ℝ} (_hx : 0 ≤ x) (hx1 : x ≤ 1) :
    unitUniform (Iio x) = ENNReal.ofReal x := by
  rw [unitUniform, Measure.restrict_apply measurableSet_Iio]
  have heq : Iio x ∩ Ioo (0 : ℝ) 1 = Ioo 0 x := by
    ext y
    simp only [mem_inter_iff, mem_Iio, mem_Ioo]
    constructor
    · rintro ⟨h, h0, _⟩; exact ⟨h0, h⟩
    · rintro ⟨h0, h⟩; exact ⟨h, h0, h.trans_le hx1⟩
  rw [heq, Real.volume_Ioo, sub_zero]

lemma independent_uniform_threshold [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (V X : Ω → ℝ) (hV : Measurable V) (hX : Measurable X)
    (hlaw : μ.map V = unitUniform) (hind : IndepFun X V μ)
    (hbound : ∀ ω, X ω ∈ Icc (0 : ℝ) 1) :
    μ {ω | V ω < X ω} = ∫⁻ ω, ENNReal.ofReal (X ω) ∂μ := by
  let D : Set (ℝ × ℝ) := {z | z.2 < z.1}
  have hD : MeasurableSet D := measurableSet_lt measurable_snd measurable_fst
  have heq : {ω | V ω < X ω} = (fun ω => (X ω, V ω)) ⁻¹' D := rfl
  rw [heq, ← Measure.map_apply (hX.prodMk hV) hD,
    hind.map_prod_eq_prod_map_map hX.aemeasurable hV.aemeasurable, hlaw, Measure.prod_apply hD,
    lintegral_map (measurable_measure_prodMk_left hD) hX]
  apply lintegral_congr
  intro ω
  exact unitUniform_Iio (hbound ω).1 (hbound ω).2

/-- The defining conditional-probability identity on all events in `G`.
The right-hand side integrates the candidate conditional probability. -/
def ConditionalAcceptance (G : MeasurableSpace Ω) [MeasurableSpace Ω] (μ : Measure Ω)
    (V π : Ω → ℝ) : Prop :=
  ∀ A, MeasurableSet[G] A →
    μ (A ∩ {ω | V ω < π ω}) = ∫⁻ ω in A, ENNReal.ofReal (π ω) ∂μ

lemma conditional_acceptance_of_uniform (G : MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : G ≤ mΩ) (V π : Ω → ℝ)
    (hV : Measurable V) (hπ : Measurable[G] π)
    (hbound : ∀ ω, π ω ∈ Icc (0 : ℝ) 1)
    (hlaw : μ.map V = unitUniform) (hind : Indep ((borel ℝ).comap V) G μ) :
    ConditionalAcceptance G μ V π := by
  intro A hA
  let X : Ω → ℝ := A.indicator π
  have hXG : Measurable[G] X := hπ.indicator hA
  have hX : Measurable X := hXG.mono hG le_rfl
  have hXV : IndepFun X V μ := (IndepFun_iff_Indep X V μ).mpr
    (indep_of_indep_of_le_left hind.symm hXG.comap_le)
  have hb : ∀ ω, X ω ∈ Icc (0 : ℝ) 1 := by
    intro ω
    by_cases h : ω ∈ A
    · simpa [X, h] using hbound ω
    · simp [X, h]
  have hzero : μ {ω | V ω < 0} = 0 := by
    change μ (V ⁻¹' Iio 0) = 0
    rw [← Measure.map_apply hV measurableSet_Iio, hlaw,
      unitUniform_Iio le_rfl (by norm_num), ENNReal.ofReal_zero]
  have hnonneg : ∀ᵐ ω ∂μ, 0 ≤ V ω := by
    rw [ae_iff]
    simpa only [not_le] using hzero
  calc
    μ (A ∩ {ω | V ω < π ω}) = μ {ω | V ω < X ω} := by
      apply measure_congr
      filter_upwards [hnonneg] with ω hω
      by_cases h : ω ∈ A
      · simp [X, h]
      · simp [X, h, not_lt_of_ge hω]
    _ = ∫⁻ ω, ENNReal.ofReal (X ω) ∂μ :=
      independent_uniform_threshold μ V X hV hX hlaw hXV hb
    _ = ∫⁻ ω in A, ENNReal.ofReal (π ω) ∂μ := by
      rw [← lintegral_indicator (hG A hA)]
      apply lintegral_congr
      intro ω
      by_cases h : ω ∈ A <;> simp [X, h]

lemma ConditionalAcceptance.condExp_eq (G : MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : G ≤ mΩ) (V π : Ω → ℝ)
    (hV : Measurable V) (hπ : Measurable[G] π)
    (hbound : ∀ ω, π ω ∈ Icc (0 : ℝ) 1)
    (h : ConditionalAcceptance G μ V π) :
    μ[(fun ω => if V ω < π ω then (1 : ℝ) else 0) | G] =ᵐ[μ] π := by
  have hπm : Measurable π := hπ.mono hG le_rfl
  have hπi : Integrable π μ := integrable_of_le_of_le hπm.aestronglyMeasurable
    (Filter.Eventually.of_forall fun ω => (hbound ω).1)
    (Filter.Eventually.of_forall fun ω => (hbound ω).2)
    (integrable_const (0 : ℝ)) (integrable_const (1 : ℝ))
  have he : MeasurableSet {ω | V ω < π ω} := measurableSet_lt hV hπm
  apply Filter.EventuallyEq.symm
  apply ae_eq_condExp_of_forall_setIntegral_eq hG ((integrable_const (1 : ℝ)).indicator he)
    (fun _ _ _ => hπi.integrableOn) _ hπ.aestronglyMeasurable
  intro A hA _
  have hi := congrArg ENNReal.toReal (h A hA)
  rw [← integral_eq_lintegral_of_nonneg_ae
    (Filter.Eventually.of_forall fun ω => (hbound ω).1)
    hπm.aestronglyMeasurable] at hi
  rw [integral_indicator he, Measure.restrict_restrict he]
  simpa [Measure.real, inter_comm] using hi.symm

/-- A recursively selected unused draw is uniform and independent of the past;
thresholding it at a predictable propensity has that conditional probability. -/
theorem fresh_draw (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (key : ℕ → Ω → K)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG0 : G0 ≤ mΩ) (hU : ∀ k, Measurable (U k))
    (hind : iIndep (fieldInfo G0 U) μ) (hlaw : ∀ k, μ.map (U k) = unitUniform)
    (hkey : ∀ n, Measurable[history G0 U key n] (key n))
    (n : ℕ) (hfresh : ∀ ω i, i < n → key i ω ≠ key n ω)
    (π : Ω → ℝ) (hπ : Measurable[history G0 U key n] π)
    (hbound : ∀ ω, π ω ∈ Icc (0 : ℝ) 1) :
    μ.map (fun ω => U (key n ω) ω) = unitUniform ∧
    Indep ((borel ℝ).comap (fun ω => U (key n ω) ω)) (history G0 U key n) μ ∧
    μ[(fun ω => if U (key n ω) ω < π ω then (1 : ℝ) else 0) | history G0 U key n]
      =ᵐ[μ] π := by
  have hG := history_le G0 U key hG0 hU hkey n
  have hV := measurable_select ((hkey n).mono hG le_rfl) hU
  have hL := selected_law G0 U key μ hG0 hU hind hlaw hkey n hfresh
  have hI := selected_independent G0 U key μ hG0 hU hind hlaw hkey n hfresh
  have hC := conditional_acceptance_of_uniform _ μ hG _ π hV hπ hbound hL hI
  exact ⟨hL, hI, ConditionalAcceptance.condExp_eq _ μ hG _ π hV hπ hbound hC⟩

omit [Countable K] [MeasurableSpace K] [MeasurableSingletonClass K] in
lemma history_zero (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (key : ℕ → Ω → K) :
    history G0 U key 0 = G0 := rfl

omit [Countable K] [MeasurableSpace K] [MeasurableSingletonClass K] in
lemma history_succ (G0 : MeasurableSpace Ω) (U : K → Ω → ℝ) (key : ℕ → Ω → K) (n : ℕ) :
    history G0 U key (n + 1) =
      history G0 U key n ⊔ (borel ℝ).comap (fun ω => U (key n ω) ω) := rfl

end OranCausal.Fresh

#print axioms OranCausal.Fresh.fresh_draw
