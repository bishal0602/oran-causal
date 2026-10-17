module

public import OranCausal.Fresh
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.Indicator
public import Mathlib.Analysis.SpecialFunctions.Sqrt

public section

open Finset MeasureTheory ProbabilityTheory MeasurableSpace
open scoped BigOperators Classical

namespace OranCausal.Exact

/-- Upper rank, including the distinguished observation and every tie. -/
noncomputable def upperRank {J : Type*} [Fintype J] (s : J → ℝ) (j : J) : ℕ :=
  (univ.filter fun k => s j ≤ s k).card

noncomputable def rankP {J : Type*} [Fintype J] (s : J → ℝ) (j : J) : ℝ :=
  upperRank s j / Fintype.card J

lemma rank_count_bound {J : Type*} [Fintype J] [Nonempty J] (s : J → ℝ)
    (t : ℝ) (ht : 0 ≤ t) :
    (∑ j : J, if rankP s j ≤ t then (1 : ℝ) else 0) ≤ Fintype.card J * t := by
  let S : Finset J := univ.filter fun j => rankP s j ≤ t
  have heq : (∑ j : J, if rankP s j ≤ t then (1 : ℝ) else 0) = S.card := by
    simp [S]
  rw [heq]
  by_cases hS : S.Nonempty
  · obtain ⟨j, hj, hmin⟩ := S.exists_min_image s hS
    have hsub : S ⊆ univ.filter (fun k => s j ≤ s k) := by
      intro k hk
      exact mem_filter.mpr ⟨mem_univ _, hmin k hk⟩
    have hc : (S.card : ℝ) ≤ upperRank s j := by exact_mod_cast card_le_card hsub
    have hjt : rankP s j ≤ t := (mem_filter.mp hj).2
    have hpos : (0 : ℝ) < Fintype.card J := by exact_mod_cast Fintype.card_pos
    have hr := (div_le_iff₀ hpos).mp hjt
    nlinarith
  · simp only [not_nonempty_iff_eq_empty.mp hS, card_empty, Nat.cast_zero]
    positivity

lemma upperRank_perm {J : Type*} [Fintype J] (s : J → ℝ) (e : Equiv.Perm J) (j : J) :
    upperRank (s ∘ e) j = upperRank s (e j) := by
  unfold upperRank
  apply Finset.card_bij (fun k _ => e k)
  · intro k hk
    simpa only [Function.comp_apply, mem_filter, mem_univ, true_and] using hk
  · intro k _ l _ h
    exact e.injective h
  · intro k hk
    refine ⟨e.symm k, ?_, by simp⟩
    simpa only [Function.comp_apply, mem_filter, mem_univ, true_and, e.apply_symm_apply] using hk

lemma rankP_perm {J : Type*} [Fintype J] (s : J → ℝ) (e : Equiv.Perm J) (j : J) :
    rankP (s ∘ e) j = rankP s (e j) := by
  simp only [rankP, upperRank_perm]

/-- Independent sampling from a finite mass function. -/
noncomputable def iidMass {J A : Type*} [Fintype J] (p : A → ℝ) (x : J → A) : ℝ :=
  ∏ j, p (x j)

lemma iidMass_nonneg {J A : Type*} [Fintype J] {p : A → ℝ} (hp : ∀ a, 0 ≤ p a)
    (x : J → A) : 0 ≤ iidMass p x := prod_nonneg fun _ _ => hp _

lemma iidMass_sum {J A : Type*} [Fintype J] [Fintype A]
    (p : A → ℝ) (hp : ∑ a, p a = 1) : ∑ x : J → A, iidMass p x = 1 := by
  simp only [iidMass, ← Fintype.prod_sum, hp, Finset.prod_const_one]

lemma iidMass_perm {J A : Type*} [Fintype J] (p : A → ℝ) (x : J → A) (e : Equiv.Perm J) :
    iidMass p (x ∘ e) = iidMass p x := by
  exact Equiv.prod_comp e (fun j => p (x j))

/-- Rejection probability at a specified slot in the iid experiment. -/
noncomputable def rejectionMass {J A : Type*} [Fintype J] [Fintype A]
    (p : A → ℝ) (score : A → ℝ) (j : J) (t : ℝ) : ℝ :=
  ∑ x : J → A, iidMass p x * if rankP (score ∘ x) j ≤ t then 1 else 0

lemma rejectionMass_same_slot {J A : Type*} [Fintype J] [Fintype A]
    (p : A → ℝ) (score : A → ℝ) (j k : J) (t : ℝ) :
    rejectionMass p score j t = rejectionMass p score k t := by
  let e := Equiv.swap j k
  let E : (J → A) ≃ (J → A) := Equiv.arrowCongr e (Equiv.refl A)
  have he : ∀ x : J → A, E x = x ∘ e := by
    intro x; ext l; simp [E, e, Equiv.arrowCongr_apply]
  unfold rejectionMass
  rw [← E.sum_comp]
  apply sum_congr rfl
  intro x _
  rw [he, iidMass_perm]
  have hs : score ∘ (x ∘ e) = (score ∘ x) ∘ e := rfl
  rw [hs, rankP_perm]
  simp [e]

theorem iid_rank_valid {J A : Type*} [Fintype J] [Nonempty J] [Fintype A]
    (p : A → ℝ) (hp : ∀ a, 0 ≤ p a) (hsum : ∑ a, p a = 1)
    (score : A → ℝ) (j : J) (t : ℝ) (ht : 0 ≤ t) :
    rejectionMass p score j t ≤ t := by
  have hpos : (0 : ℝ) < Fintype.card J := by exact_mod_cast Fintype.card_pos
  have h : (Fintype.card J : ℝ) * rejectionMass p score j t ≤ Fintype.card J * t := by
    calc
      _ = ∑ k : J, rejectionMass p score k t := by
        simp_rw [← rejectionMass_same_slot p score j]
        simp
      _ = ∑ x : J → A, iidMass p x *
          (∑ k : J, if rankP (score ∘ x) k ≤ t then (1 : ℝ) else 0) := by
        unfold rejectionMass
        rw [sum_comm]
        simp only [mul_sum]
      _ ≤ ∑ x : J → A, iidMass p x * (Fintype.card J * t) :=
        sum_le_sum fun x _ => mul_le_mul_of_nonneg_left
          (rank_count_bound (score ∘ x) t ht) (iidMass_nonneg hp x)
      _ = _ := by rw [← sum_mul, iidMass_sum p hsum, one_mul]
  exact (mul_le_mul_iff_right₀ hpos).mp h

/-- Real indicator used in conditional probabilities. -/
noncomputable def ind {Ω : Type*} (E : Set Ω) : Ω → ℝ := E.indicator (fun _ => 1)

lemma ind_inter {Ω : Type*} (A C : Set Ω) :
    ind (A ∩ C) = A.indicator (ind C) := by
  simp [ind, Set.indicator_indicator]

lemma ind_mul {Ω : Type*} (A : Set Ω) (f : Ω → ℝ) :
    f * ind A = A.indicator f := by
  ext ω
  by_cases h : ω ∈ A <;> simp [ind, h]

section Conditional

variable {Ω : Type*} (G : MeasurableSpace Ω) [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]

lemma integrable_ind {A : Set Ω} (hA : MeasurableSet A) : Integrable (ind A) μ :=
  (integrable_const (1 : ℝ)).indicator hA

lemma integrable_unit_interval {f : Ω → ℝ} (hf : Measurable f)
    (hb : ∀ ω, f ω ∈ Set.Icc (0 : ℝ) 1) : Integrable f μ :=
  integrable_of_le_of_le hf.aestronglyMeasurable
    (Filter.Eventually.of_forall fun ω => (hb ω).1)
    (Filter.Eventually.of_forall fun ω => (hb ω).2)
    (integrable_const (0 : ℝ)) (integrable_const (1 : ℝ))

lemma conditional_finite_event {A : Type*} [Fintype A] [MeasurableSpace A]
    [MeasurableSingletonClass A] (hG : G ≤ mΩ) (X : Ω → A) (hX : Measurable X)
    (mass : A → Ω → ℝ)
    (hlaw : ∀ a, μ[ind (X ⁻¹' {a}) | G] =ᵐ[μ] mass a)
    (E : A → Set Ω) (hE : ∀ a, MeasurableSet[G] (E a)) :
    μ[ind {ω | ω ∈ E (X ω)} | G] =ᵐ[μ]
      (fun ω => ∑ a, mass a ω * ind (E a) ω) := by
  have heq : ind {ω | ω ∈ E (X ω)} =
      ∑ a : A, (E a).indicator (ind (X ⁻¹' {a})) := by
    ext ω
    rw [Finset.sum_apply]
    rw [sum_eq_single (X ω)]
    · by_cases h : ω ∈ E (X ω) <;> simp [ind, h]
    · intro a _ ha
      by_cases h : ω ∈ E a <;> simp [ind, h, Ne.symm ha]
    · simp
  rw [heq]
  have hi : ∀ a : A, Integrable ((E a).indicator (ind (X ⁻¹' {a}))) μ := fun a =>
    (integrable_ind μ (hX (measurableSet_singleton a))).indicator (hG _ (hE a))
  have hc := condExp_finsetSum (s := univ) (fun a _ => hi a) G
  have ht : ∀ a : A, μ[(E a).indicator (ind (X ⁻¹' {a})) | G] =ᵐ[μ]
      (fun ω => mass a ω * ind (E a) ω) := by
    intro a
    have h := condExp_indicator (integrable_ind μ (hX (measurableSet_singleton a))) (hE a)
    filter_upwards [h, hlaw a] with ω hc hl
    rw [hc, Set.indicator_apply, hl]
    by_cases he : ω ∈ E a <;> simp [ind, he]
  have hall := (Filter.eventually_all).mpr ht
  filter_upwards [hc, hall] with ω hc hall
  simp only [Finset.sum_apply] at hc
  rw [hc]
  exact sum_congr rfl fun a _ => hall a

end Conditional

noncomputable def bitMass (p : ℝ) (b : Bool) : ℝ := if b then p else 1 - p

lemma bitMass_nonneg {p : ℝ} (hp : p ∈ Set.Icc (0 : ℝ) 1) (b : Bool) :
    0 ≤ bitMass p b := by cases b <;> simp [bitMass] <;> linarith [hp.1, hp.2]

lemma bitMass_sum (p : ℝ) : ∑ b : Bool, bitMass p b = 1 := by
  simp [bitMass]

/-- Probability of one complete assignment, conditional on the fixed design. -/
noncomputable def assignmentMass {I : Type*} [Fintype I] (π : I → ℝ) (x : I → Bool) : ℝ :=
  ∏ i, bitMass (π i) (x i)

lemma assignmentMass_sum {I : Type*} [Fintype I] (π : I → ℝ) :
    ∑ x : I → Bool, assignmentMass π x = 1 := by
  simp only [assignmentMass, ← Fintype.prod_sum, bitMass_sum, prod_const_one]

lemma assignmentMass_nonneg {I : Type*} [Fintype I] (π : I → ℝ)
    (hp : ∀ i, π i ∈ Set.Icc (0 : ℝ) 1) (x : I → Bool) :
    0 ≤ assignmentMass π x := prod_nonneg fun i _ => bitMass_nonneg (hp i) _

def prefixEvent {Ω : Type*} (X : ℕ → Ω → Bool) (v : ℕ → Bool) : ℕ → Set Ω
  | 0 => Set.univ
  | n + 1 => prefixEvent X v n ∩ X n ⁻¹' {v n}

lemma prefixEvent_iff {Ω : Type*} (X : ℕ → Ω → Bool) (v : ℕ → Bool) (n : ℕ) (ω : Ω) :
    ω ∈ prefixEvent X v n ↔ ∀ i < n, X i ω = v i := by
  induction n with
  | zero => simp [prefixEvent]
  | succ n ih =>
    simp only [prefixEvent, Set.mem_inter_iff, Set.mem_preimage, Set.mem_singleton_iff, ih]
    constructor
    · rintro ⟨h, hn⟩ i hi
      by_cases he : i = n
      · simpa [he] using hn
      · exact h i (by omega)
    · intro h
      exact ⟨fun i hi => h i (by omega), h n (by omega)⟩

section Chain

variable {Ω : Type*} (G : MeasurableSpace Ω) (H : ℕ → MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]

omit mΩ in
lemma prefix_measurable (X : ℕ → Ω → Bool) (v : ℕ → Bool)
    (hH : Monotone H) (hX : ∀ n, Measurable[H (n + 1)] (X n)) (n : ℕ) :
    MeasurableSet[H n] (prefixEvent X v n) := by
  induction n with
  | zero => exact MeasurableSet.univ
  | succ n ih => exact (hH (by omega) _ ih).inter (hX n (measurableSet_singleton _))

lemma conditional_bit (hH : ∀ n, H n ≤ mΩ) (X : ℕ → Ω → Bool)
    (hX : ∀ n, Measurable (X n)) (π : ℕ → Ω → ℝ)
    (n : ℕ) (htrue : μ[ind (X n ⁻¹' {true}) | H n] =ᵐ[μ] π n) (b : Bool) :
    μ[ind (X n ⁻¹' {b}) | H n] =ᵐ[μ] (fun ω => bitMass (π n ω) b) := by
  cases b
  · have heq : ind (X n ⁻¹' {false}) = (fun _ => (1 : ℝ)) - ind (X n ⁻¹' {true}) := by
      ext ω
      cases hx : X n ω <;> simp [ind, hx]
    rw [heq]
    have h := condExp_sub (integrable_const (1 : ℝ))
      (integrable_ind μ (hX n (measurableSet_singleton true))) (H n)
    filter_upwards [h, htrue] with ω hc ht
    simp only [hc, condExp_const (hH n), Pi.sub_apply, ht, bitMass, Bool.false_eq_true, ite_false]
  · simpa only [bitMass, Bool.true_eq, ite_true] using htrue

/-- Chain rule derived from conditional Bernoulli draws, not from an assumed
joint assignment law. All propensities are measurable in the initial history. -/
theorem conditional_prefix_mass
    (hH : ∀ n, H n ≤ mΩ) (hGH : ∀ n, G ≤ H n) (hmono : Monotone H)
    (X : ℕ → Ω → Bool) (hX : ∀ n, Measurable[H (n + 1)] (X n))
    (π : ℕ → Ω → ℝ) (hπ : ∀ n, Measurable[G] (π n))
    (hbound : ∀ n ω, π n ω ∈ Set.Icc (0 : ℝ) 1)
    (limit : ℕ) (htrue : ∀ n, n < limit → μ[ind (X n ⁻¹' {true}) | H n] =ᵐ[μ] π n)
    (v : ℕ → Bool) (n : ℕ) (hn : n ≤ limit) :
    μ[ind (prefixEvent X v n) | G] =ᵐ[μ]
      (fun ω => ∏ i ∈ range n, bitMass (π i ω) (v i)) := by
  have hG : G ≤ mΩ := (hGH 0).trans (hH 0)
  have hXm : ∀ i, Measurable (X i) := fun i => (hX i).mono (hH _) le_rfl
  have hpre : ∀ i, MeasurableSet[H i] (prefixEvent X v i) := prefix_measurable H X v hmono hX
  induction n with
  | zero => simp [prefixEvent, ind, condExp_const hG]
  | succ n ih =>
    let f : Ω → ℝ := fun ω => bitMass (π n ω) (v n)
    have hfG : Measurable[G] f := by
      cases hv : v n <;> simp [f, bitMass, hv] <;> fun_prop
    have hfb : ∀ ω, f ω ∈ Set.Icc (0 : ℝ) 1 := by
      intro ω
      have hb := hbound n ω
      cases hv : v n <;> simp [f, bitMass, hv] <;> constructor <;> linarith [hb.1, hb.2]
    have hfi : Integrable f μ := integrable_unit_interval μ (hfG.mono hG le_rfl) hfb
    have hpi : Integrable (ind (prefixEvent X v n)) μ := integrable_ind μ (hH n _ (hpre n))
    have hprod : Integrable (f * ind (prefixEvent X v n)) μ := by
      rw [ind_mul]
      exact hfi.indicator (hH n _ (hpre n))
    have hinner : μ[ind (prefixEvent X v (n + 1)) | H n] =ᵐ[μ]
        f * ind (prefixEvent X v n) := by
      rw [prefixEvent, ind_inter]
      have hc := condExp_indicator (integrable_ind μ (hXm n (measurableSet_singleton (v n)))) (hpre n)
      have hb := conditional_bit H μ hH X hXm π n (htrue n (by omega)) (v n)
      filter_upwards [hc, hb] with ω hc hb
      rw [hc, Set.indicator_apply, hb]
      by_cases hw : ω ∈ prefixEvent X v n <;> simp [ind, f, hw]
    calc
      _ =ᵐ[μ] μ[μ[ind (prefixEvent X v (n + 1)) | H n] | G] :=
        (condExp_condExp_of_le (hGH n) (hH n)).symm
      _ =ᵐ[μ] μ[f * ind (prefixEvent X v n) | G] := condExp_congr_ae hinner
      _ =ᵐ[μ] f * μ[ind (prefixEvent X v n) | G] :=
        condExp_mul_of_aestronglyMeasurable_left hfG.aestronglyMeasurable hprod hpi
      _ =ᵐ[μ] _ := by
        filter_upwards [ih (by omega)] with ω hω
        simp [Pi.mul_apply, hω, prod_range_succ, f, mul_comm]

end Chain

lemma measurable_rankP {Ω J : Type*} [MeasurableSpace Ω] [Fintype J]
    (s : J → Ω → ℝ) (hs : ∀ j, Measurable (s j)) (j : J) :
    Measurable (fun ω => rankP (fun k => s k ω) j) := by
  have heq : (fun ω => rankP (fun k => s k ω) j) =
      (fun ω => (∑ k : J, if s j ω ≤ s k ω then (1 : ℝ) else 0) / Fintype.card J) := by
    ext ω
    simp [rankP, upperRank]
  rw [heq]
  apply Measurable.div_const
  apply Finset.measurable_sum
  intro k _
  exact Measurable.ite (measurableSet_le (hs j) (hs k)) measurable_const measurable_const

theorem conditional_iid_rank_valid {Ω J A : Type*} [Fintype J] [Nonempty J]
    [Fintype A] [MeasurableSpace A] [MeasurableSingletonClass A]
    (G : MeasurableSpace Ω) [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : G ≤ mΩ) (X : Ω → J → A) (hX : Measurable X)
    (p : A → Ω → ℝ) (hp : ∀ a ω, 0 ≤ p a ω) (hsum : ∀ ω, ∑ a, p a ω = 1)
    (hlaw : ∀ x : J → A, μ[ind (X ⁻¹' {x}) | G] =ᵐ[μ]
      (fun ω => iidMass (fun a => p a ω) x))
    (score : A → Ω → ℝ) (hscore : ∀ a, Measurable[G] (score a))
    (j : J) (t : ℝ) (ht : 0 ≤ t) :
    μ[ind {ω | rankP (fun k => score (X ω k) ω) j ≤ t} | G] ≤ᵐ[μ] (fun _ => t) := by
  let E : (J → A) → Set Ω := fun x => {ω | rankP (fun k => score (x k) ω) j ≤ t}
  have hE : ∀ x, MeasurableSet[G] (E x) := by
    let : MeasurableSpace Ω := G
    intro x
    exact measurableSet_le
      (measurable_rankP (fun k => score (x k)) (fun k => hscore (x k)) j) measurable_const
  have h := conditional_finite_event G μ hG X hX
    (fun x ω => iidMass (fun a => p a ω) x) hlaw E hE
  filter_upwards [h] with ω hω
  change μ[ind {ω | ω ∈ E (X ω)} | G] ω ≤ t
  rw [hω]
  exact iid_rank_valid (fun a => p a ω) (fun a => hp a ω) (hsum ω)
    (fun a => score a ω) j t ht

/-- Replicate-major order of the observed assignment (replicate zero) and redraws. -/
def experiment {Ω : Type*} (B M : ℕ) (X : ℕ → Ω → Bool) (ω : Ω)
    (r : Fin (B + 1)) (i : Fin M) : Bool := X (finProdFinEquiv (r, i)).val ω

noncomputable def repeatedPropensity {Ω : Type*} (B M : ℕ) (π : Fin M → Ω → ℝ)
    (n : ℕ) (ω : Ω) : ℝ :=
  if hn : n < (B + 1) * M then π (finProdFinEquiv.symm ⟨n, hn⟩).2 ω else 1 / 2

theorem conditional_experiment_mass {Ω : Type*}
    (G : MeasurableSpace Ω) (H : ℕ → MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hH : ∀ n, H n ≤ mΩ) (hGH : ∀ n, G ≤ H n) (hmono : Monotone H)
    (X : ℕ → Ω → Bool) (hX : ∀ n, Measurable[H (n + 1)] (X n))
    (B M : ℕ) (π : Fin M → Ω → ℝ) (hπ : ∀ i, Measurable[G] (π i))
    (hb : ∀ i ω, π i ω ∈ Set.Icc (0 : ℝ) 1)
    (htrue : ∀ n, n < (B + 1) * M →
      μ[ind (X n ⁻¹' {true}) | H n] =ᵐ[μ] repeatedPropensity B M π n)
    (x : Fin (B + 1) → Fin M → Bool) :
    μ[ind (experiment B M X ⁻¹' {x}) | G] =ᵐ[μ]
      (fun ω => iidMass (assignmentMass (fun i => π i ω)) x) := by
  let L := (B + 1) * M
  let v : ℕ → Bool := fun n => if hn : n < L then
    x (finProdFinEquiv.symm ⟨n, hn⟩).1 (finProdFinEquiv.symm ⟨n, hn⟩).2 else false
  have heq : experiment B M X ⁻¹' {x} = prefixEvent X v L := by
    ext ω
    simp only [Set.mem_preimage, Set.mem_singleton_iff, prefixEvent_iff]
    constructor
    · intro hx n hn
      have h := congrFun (congrFun hx (finProdFinEquiv.symm ⟨n, hn⟩).1)
        (finProdFinEquiv.symm ⟨n, hn⟩).2
      change X (finProdFinEquiv (finProdFinEquiv.symm ⟨n, hn⟩)).val ω = _ at h
      simpa only [Equiv.apply_symm_apply, v, dite_eq_left hn] using h
    · intro h
      funext r i
      have he := h (finProdFinEquiv (r, i)).val (finProdFinEquiv (r, i)).isLt
      simpa only [v, L, dite_eq_left (finProdFinEquiv (r, i)).isLt, Fin.eta,
        Equiv.symm_apply_apply, experiment] using he
  rw [heq]
  have hm : ∀ n, Measurable[G] (repeatedPropensity B M π n) := by
    intro n
    unfold repeatedPropensity
    split_ifs <;> first | exact hπ _ | exact measurable_const
  have hbounds : ∀ n ω, repeatedPropensity B M π n ω ∈ Set.Icc (0 : ℝ) 1 := by
    intro n ω
    unfold repeatedPropensity
    split_ifs
    · exact hb _ _
    · norm_num
  have h := conditional_prefix_mass G H μ hH hGH hmono X hX
    (repeatedPropensity B M π) hm hbounds L htrue v L le_rfl
  filter_upwards [h] with ω hω
  rw [hω]
  rw [← Fin.prod_univ_eq_prod_range]
  change (∏ n : Fin L, bitMass (repeatedPropensity B M π n ω) (v n)) = _
  rw [← finProdFinEquiv.prod_comp]
  simp only [Fintype.prod_prod_type, iidMass, assignmentMass]
  apply prod_congr rfl
  intro r _
  apply prod_congr rfl
  intro i _
  simp only [repeatedPropensity, v, L, dite_eq_left (finProdFinEquiv (r, i)).isLt,
    Fin.eta, Equiv.symm_apply_apply]

/-- The source's plus-one redraw p-value; the finite sum counts exceedances. -/
noncomputable def monteCarloP {B : ℕ} (s : Fin (B + 1) → ℝ) : ℝ :=
  (1 + ∑ b : Fin B, if s 0 ≤ s b.succ then (1 : ℝ) else 0) / (B + 1)

lemma monteCarloP_eq_rankP {B : ℕ} (s : Fin (B + 1) → ℝ) :
    monteCarloP s = rankP s 0 := by
  have h : (upperRank s 0 : ℝ) = ∑ j, if s 0 ≤ s j then (1 : ℝ) else 0 := by
    simp [upperRank]
  rw [rankP, h, Fin.sum_univ_succ]
  simp [monteCarloP]

/-- The PMRT statistic with all design quantities held fixed across assignments.
Division by zero returns zero, so a zero-variance design gives p-value one. -/
noncomputable def pmrtZ {Ω : Type*} {M : ℕ} (π sgn w : Fin M → Ω → ℝ)
    (a : Fin M → Bool) (ω : Ω) : ℝ :=
  (∑ i, sgn i ω * ((if a i then (1 : ℝ) else 0) - π i ω) * w i ω) /
    Real.sqrt (∑ i, π i ω * (1 - π i ω) * (w i ω) ^ 2)

lemma measurable_pmrtZ {Ω : Type*} [MeasurableSpace Ω] {M : ℕ}
    (π sgn w : Fin M → Ω → ℝ) (hπ : ∀ i, Measurable (π i))
    (hs : ∀ i, Measurable (sgn i)) (hw : ∀ i, Measurable (w i)) (a : Fin M → Bool) :
    Measurable (pmrtZ π sgn w a) := by
  unfold pmrtZ
  fun_prop

noncomputable def twoSidedP {Ω : Type*} (B M : ℕ) (π sgn w : Fin M → Ω → ℝ)
    (X : ℕ → Ω → Bool) (ω : Ω) : ℝ :=
  monteCarloP fun r => |pmrtZ π sgn w (experiment B M X ω r) ω|

noncomputable def oneSidedP {Ω : Type*} (B M : ℕ) (π sgn w : Fin M → Ω → ℝ)
    (d : ℝ) (o : Ω → ℝ) (X : ℕ → Ω → Bool) (ω : Ω) : ℝ :=
  monteCarloP fun r => d * o ω * pmrtZ π sgn w (experiment B M X ω r) ω

/-- Exact PMRT validity from the design law for the observed modes and the
independent redraws. The joint iid assignment law is derived by the chain rule. -/
theorem conditional_design_exact {Ω : Type*}
    (G : MeasurableSpace Ω) (H : ℕ → MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hH : ∀ n, H n ≤ mΩ) (hGH : ∀ n, G ≤ H n) (hmono : Monotone H)
    (X : ℕ → Ω → Bool) (hX : ∀ n, Measurable[H (n + 1)] (X n))
    (B M : ℕ) (π sgn w : Fin M → Ω → ℝ)
    (hπ : ∀ i, Measurable[G] (π i)) (hs : ∀ i, Measurable[G] (sgn i))
    (hw : ∀ i, Measurable[G] (w i)) (hb : ∀ i ω, π i ω ∈ Set.Icc (0 : ℝ) 1)
    (htrue : ∀ n, n < (B + 1) * M →
      μ[ind (X n ⁻¹' {true}) | H n] =ᵐ[μ] repeatedPropensity B M π n)
    (d : ℝ) (o : Ω → ℝ) (ho : Measurable[G] o) (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    (μ[ind {ω | twoSidedP B M π sgn w X ω ≤ t} | G] ≤ᵐ[μ] (fun _ => t)) ∧
    (μ[ind {ω | oneSidedP B M π sgn w d o X ω ≤ t} | G] ≤ᵐ[μ] (fun _ => t)) := by
  have hG : G ≤ mΩ := (hGH 0).trans (hH 0)
  have hXm : Measurable (experiment B M X) := Measurable.of_eval fun r =>
    Measurable.of_eval fun i => (hX _).mono (hH _) le_rfl
  have hmass := conditional_experiment_mass G H μ hH hGH hmono X hX B M π hπ hb htrue
  have hscore : ∀ a, Measurable[G] (pmrtZ π sgn w a) := by
    let : MeasurableSpace Ω := G
    exact measurable_pmrtZ π sgn w hπ hs hw
  have habs : ∀ a, Measurable[G] (fun ω => |pmrtZ π sgn w a ω|) := by
    let : MeasurableSpace Ω := G
    intro a
    simpa only [Real.norm_eq_abs] using (hscore a).norm
  have horient : ∀ a, Measurable[G] (fun ω => d * o ω * pmrtZ π sgn w a ω) := by
    let : MeasurableSpace Ω := G
    intro a
    exact (measurable_const.mul ho).mul (hscore a)
  have htwo := conditional_iid_rank_valid G μ hG (experiment B M X) hXm
    (fun a ω => assignmentMass (fun i => π i ω) a)
    (fun a ω => assignmentMass_nonneg _ (fun i => hb i ω) a)
    (by intro ω; simp only [assignmentMass, ← Fintype.prod_sum, bitMass_sum, prod_const_one]) hmass
    (fun a ω => |pmrtZ π sgn w a ω|) habs
    (0 : Fin (B + 1)) t ht.1
  have hone := conditional_iid_rank_valid G μ hG (experiment B M X) hXm
    (fun a ω => assignmentMass (fun i => π i ω) a)
    (fun a ω => assignmentMass_nonneg _ (fun i => hb i ω) a)
    (by intro ω; simp only [assignmentMass, ← Fintype.prod_sum, bitMass_sum, prod_const_one]) hmass
    (fun a ω => d * o ω * pmrtZ π sgn w a ω) horient (0 : Fin (B + 1)) t ht.1
  simpa only [twoSidedP, oneSidedP, monteCarloP_eq_rankP] using And.intro htwo hone

lemma initial_le_history {Ω K : Type*} (G : MeasurableSpace Ω)
    (U : K → Ω → ℝ) (key : ℕ → Ω → K) (n : ℕ) : G ≤ Fresh.history G U key n := by
  induction n with
  | zero => rw [Fresh.history_zero]
  | succ n ih => rw [Fresh.history_succ]; exact ih.trans le_sup_left

lemma history_monotone {Ω K : Type*} (G : MeasurableSpace Ω)
    (U : K → Ω → ℝ) (key : ℕ → Ω → K) : Monotone (Fresh.history G U key) := by
  apply monotone_nat_of_le_succ
  intro n
  rw [Fresh.history_succ]
  exact le_sup_left

/-- Observed and redrawn modes, with replicate-major enumeration. -/
noncomputable def modes {Ω K : Type*} (U : K → Ω → ℝ) (key : ℕ → Ω → K)
    (q : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : Bool :=
  if U (key n ω) ω < q n ω then true else false

/-- Exact validity under the full fixed-design source model. The uniform field
includes the observed draws and fresh, independent Monte Carlo redraw draws. -/
theorem fixed_design_exact {Ω K : Type*} [Countable K] [MeasurableSpace K]
    [MeasurableSingletonClass K] (G : MeasurableSpace Ω)
    (U : K → Ω → ℝ) (key : ℕ → Ω → K)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : G ≤ mΩ) (hU : ∀ k, Measurable (U k))
    (hind : iIndep (Fresh.fieldInfo G U) μ) (hlaw : ∀ k, μ.map (U k) = Fresh.unitUniform)
    (hkey : ∀ n, Measurable[G] (key n)) (B M : ℕ)
    (hfresh : ∀ ω n, n < (B + 1) * M → ∀ i < n, key i ω ≠ key n ω)
    (π sgn w : Fin M → Ω → ℝ)
    (hπ : ∀ i, Measurable[G] (π i)) (hs : ∀ i, Measurable[G] (sgn i))
    (hw : ∀ i, Measurable[G] (w i)) (hb : ∀ i ω, π i ω ∈ Set.Icc (0 : ℝ) 1)
    (d : ℝ) (o : Ω → ℝ) (ho : Measurable[G] o) (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    let X := modes U key (repeatedPropensity B M π)
    (μ[ind {ω | twoSidedP B M π sgn w X ω ≤ t} | G] ≤ᵐ[μ] (fun _ => t)) ∧
    (μ[ind {ω | oneSidedP B M π sgn w d o X ω ≤ t} | G] ≤ᵐ[μ] (fun _ => t)) := by
  let H := Fresh.history G U key
  let q := repeatedPropensity B M π
  have hk : ∀ n, Measurable[H n] (key n) := fun n =>
    (hkey n).mono (initial_le_history G U key n) le_rfl
  have hH : ∀ n, H n ≤ mΩ := Fresh.history_le G U key hG hU hk
  have hq : ∀ n, Measurable[G] (q n) := by
    intro n
    unfold q repeatedPropensity
    split_ifs <;> first | exact hπ _ | exact measurable_const
  have hqb : ∀ n ω, q n ω ∈ Set.Icc (0 : ℝ) 1 := by
    intro n ω
    unfold q repeatedPropensity
    split_ifs
    · exact hb _ _
    · norm_num
  have hX : ∀ n, Measurable[H (n + 1)] (modes U key q n) := by
    intro n
    have hV : Measurable[H (n + 1)] (fun ω => U (key n ω) ω) := by
      apply Measurable.of_comap_le
      change (borel ℝ).comap (fun ω => U (key n ω) ω) ≤ Fresh.history G U key (n + 1)
      rw [Fresh.history_succ]
      exact le_sup_right
    exact Measurable.ite
      (measurableSet_lt hV ((hq n).mono (initial_le_history G U key _) le_rfl))
      measurable_const measurable_const
  have htrue : ∀ n, n < (B + 1) * M →
      μ[ind (modes U key q n ⁻¹' {true}) | H n] =ᵐ[μ] q n := by
    intro n hn
    have h := Fresh.fresh_draw G U key μ hG hU hind hlaw hk n
      (fun ω i hi => hfresh ω n hn i hi) (q n)
      ((hq n).mono (initial_le_history G U key n) le_rfl) (hqb n)
    have heq : ind (modes U key q n ⁻¹' {true}) =
        (fun ω => if U (key n ω) ω < q n ω then (1 : ℝ) else 0) := by
      ext ω
      by_cases hh : U (key n ω) ω < q n ω <;> simp [ind, modes, hh]
    rw [heq]
    exact h.2.2
  exact conditional_design_exact G H μ hH (initial_le_history G U key)
    (history_monotone G U key) (modes U key q) hX B M π sgn w hπ hs hw hb htrue d o ho t ht

noncomputable def maskedWeight {Ω : Type*} {M : ℕ} (active : Fin M → Set Ω)
    (w : Fin M → Ω → ℝ) : Fin M → Ω → ℝ := fun i => (active i).indicator (w i)

/-- Padding absent units with zero weights is exactly summation over the
realised units, for both the score and its variance. -/
lemma pmrtZ_masked {Ω : Type*} {M : ℕ} (active : Fin M → Set Ω)
    (π sgn w : Fin M → Ω → ℝ) (a : Fin M → Bool) (ω : Ω) :
    pmrtZ π sgn (maskedWeight active w) a ω =
      (∑ i ∈ univ.filter (fun i => ω ∈ active i),
        sgn i ω * ((if a i then (1 : ℝ) else 0) - π i ω) * w i ω) /
      Real.sqrt (∑ i ∈ univ.filter (fun i => ω ∈ active i),
        π i ω * (1 - π i ω) * (w i ω) ^ 2) := by
  unfold pmrtZ
  congr 1
  · rw [sum_filter]
    apply sum_congr rfl
    intro i _
    by_cases hi : ω ∈ active i <;> simp [maskedWeight, hi]
  · congr 1
    rw [sum_filter]
    apply sum_congr rfl
    intro i _
    by_cases hi : ω ∈ active i <;> simp [maskedWeight, hi]

/-- Lemma 2 for arbitrary initial-measurable realised unit sets, including
two-sided and fixed-direction one-sided Monte Carlo p-values. -/
theorem exact_redraw_pvalues {Ω K : Type*} [Countable K] [MeasurableSpace K]
    [MeasurableSingletonClass K] (G : MeasurableSpace Ω)
    (U : K → Ω → ℝ) (key : ℕ → Ω → K)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : G ≤ mΩ) (hU : ∀ k, Measurable (U k))
    (hind : iIndep (Fresh.fieldInfo G U) μ) (hlaw : ∀ k, μ.map (U k) = Fresh.unitUniform)
    (hkey : ∀ n, Measurable[G] (key n)) (B M : ℕ)
    (hfresh : ∀ ω n, n < (B + 1) * M → ∀ i < n, key i ω ≠ key n ω)
    (active : Fin M → Set Ω) (hactive : ∀ i, MeasurableSet[G] (active i))
    (π sgn w : Fin M → Ω → ℝ)
    (hπ : ∀ i, Measurable[G] (π i)) (hs : ∀ i, Measurable[G] (sgn i))
    (hw : ∀ i, Measurable[G] (w i)) (hb : ∀ i ω, π i ω ∈ Set.Icc (0 : ℝ) 1)
    (d : ℝ) (o : Ω → ℝ) (ho : Measurable[G] o) (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    let X := modes U key (repeatedPropensity B M π)
    (μ[ind {ω | twoSidedP B M π sgn (maskedWeight active w) X ω ≤ t} | G]
      ≤ᵐ[μ] (fun _ => t)) ∧
    (μ[ind {ω | oneSidedP B M π sgn (maskedWeight active w) d o X ω ≤ t} | G]
      ≤ᵐ[μ] (fun _ => t)) := by
  have hm : ∀ i, Measurable[G] (maskedWeight active w i) := by
    let : MeasurableSpace Ω := G
    exact fun i => (hw i).indicator (hactive i)
  exact fixed_design_exact G U key μ hG hU hind hlaw hkey B M hfresh
    π sgn (maskedWeight active w) hπ hs hm hb d o ho t ht

end OranCausal.Exact

#print axioms OranCausal.Exact.fixed_design_exact
#print axioms OranCausal.Exact.exact_redraw_pvalues
