module

public import Mathlib.MeasureTheory.Integral.Bochner.Set
public import Mathlib.MeasureTheory.MeasurableSpace.Constructions
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.NormNum
public import Mathlib.Tactic.Positivity
public import Mathlib.Tactic.Ring
public import Mathlib.Tactic.SplitIfs

public section

open scoped BigOperators
open MeasureTheory Finset

namespace OranCausal.StepUp

noncomputable def harmonic (m : ℕ) : ℝ :=
  ∑ i ∈ range m, (1 : ℝ) / (i + 1)

noncomputable def threshold (m : ℕ) (q : ℝ) (w : Fin m → ℝ)
    (j : Fin m) (k : ℕ) : ℝ := q * w j * k / (m * harmonic m)

noncomputable def envelope (n : ℕ) (f : ℕ → ℝ) : ℝ :=
  f n / n + ∑ i ∈ range (n - 1),
    f (i + 1) * ((1 : ℝ) / (i + 1) - 1 / (i + 2))

lemma reciprocal_diff_nonneg (i : ℕ) :
    0 ≤ (1 : ℝ) / (i + 1) - 1 / (i + 2) := by
  have h : (0 : ℝ) < i + 1 := by positivity
  have h' : (0 : ℝ) < i + 2 := by positivity
  apply sub_nonneg.mpr
  apply (div_le_div_iff₀ h' h).mpr
  linarith

lemma envelope_nonneg {n : ℕ} {f : ℕ → ℝ} (hf : ∀ i, 0 ≤ f i) :
    0 ≤ envelope n f := by
  unfold envelope
  exact add_nonneg (div_nonneg (hf n) (Nat.cast_nonneg n))
    (sum_nonneg fun i _ => mul_nonneg (hf _) (reciprocal_diff_nonneg i))

lemma envelope_step {n : ℕ} (hn : 1 ≤ n) (f : ℕ → ℝ) :
    envelope (n + 1) f = envelope n f + (f (n + 1) - f n) / (n + 1) := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : n ≠ 0)
  simp only [envelope, Nat.succ_eq_add_one, Nat.add_sub_cancel, sum_range_succ, Nat.cast_add,
    Nat.cast_one]
  ring

lemma ratio_le_envelope {n r : ℕ} (hn : 1 ≤ n) (hr : 1 ≤ r) (hrn : r ≤ n)
    {f : ℕ → ℝ} (hf : ∀ i, 0 ≤ f i) (hm : Monotone f) :
    f r / r ≤ envelope n f := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases h : r = n + 1
    · subst r
      unfold envelope
      exact le_add_of_nonneg_right
        (sum_nonneg fun i _ => mul_nonneg (hf _) (reciprocal_diff_nonneg i))
    · have hn' : 1 ≤ n := by omega
      rw [envelope_step hn']
      exact (ih hn' (by omega)).trans (le_add_of_nonneg_right
        (div_nonneg (sub_nonneg.mpr (hm (by omega))) (by positivity)))

lemma envelope_linear (n : ℕ) (a δ : ℝ) (hn : 1 ≤ n) :
    envelope n (fun i => a * i + δ) = a * harmonic n + δ := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases h : n = 0
    · subst n
      simp [envelope, harmonic]
    · have hn' : 1 ≤ n := by omega
      rw [envelope_step hn', ih hn']
      simp only [harmonic, sum_range_succ, Nat.cast_add, Nat.cast_one]
      ring

lemma envelope_mono {n : ℕ} {f g : ℕ → ℝ}
    (h : ∀ i, 1 ≤ i → i ≤ n → f i ≤ g i) (hn : 1 ≤ n) :
    envelope n f ≤ envelope n g := by
  unfold envelope
  apply add_le_add
  · exact div_le_div_of_nonneg_right (h n hn le_rfl) (Nat.cast_nonneg n)
  · apply sum_le_sum
    intro i hi
    exact mul_le_mul_of_nonneg_right (h (i + 1) (by omega)
      (by have := mem_range.mp hi; omega)) (reciprocal_diff_nonneg i)

lemma harmonic_pos {m : ℕ} (hm : 1 ≤ m) : 0 < harmonic m := by
  unfold harmonic
  apply sum_pos'
  · intro i _
    positivity
  · exact ⟨0, mem_range.mpr (by omega), by norm_num⟩

lemma threshold_eq {m : ℕ} (q : ℝ) (w : Fin m → ℝ) (j : Fin m) (k : ℕ) :
    threshold m q w j k = (q * w j / (m * harmonic m)) * k := by
  unfold threshold
  ring

noncomputable def gridIndicator (a : ℝ) (p : ℝ) (i : ℕ) : ℝ :=
  if p ≤ a * i then 1 else 0

lemma gridIndicator_nonneg (a p : ℝ) (i : ℕ) : 0 ≤ gridIndicator a p i := by
  unfold gridIndicator
  split_ifs <;> norm_num

lemma gridIndicator_monotone {a : ℝ} (ha : 0 ≤ a) (p : ℝ) :
    Monotone (gridIndicator a p) := by
  intro i k hik
  have h : a * (i : ℝ) ≤ a * (k : ℝ) := mul_le_mul_of_nonneg_left
    (Nat.cast_le.mpr hik) ha
  by_cases hi : p ≤ a * i
  · simp [gridIndicator, hi, hi.trans h]
  · simp only [gridIndicator, ite_eq_right hi]
    split_ifs <;> norm_num

noncomputable def contribution {m : ℕ} (s : Finset (Fin m)) (j : Fin m) : ℝ :=
  if j ∈ s then 1 / (max s.card 1 : ℕ) else 0

noncomputable def fdr {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (μ : Measure Ω) (S : Ω → Finset (Fin m)) (H0 : Finset (Fin m)) : ℝ :=
  ∫ ω, ((H0 ∩ S ω).card : ℝ) / (max (S ω).card 1 : ℕ) ∂μ

lemma contribution_bounds {m : ℕ} (s : Finset (Fin m)) (j : Fin m) :
    0 ≤ contribution s j ∧ contribution s j ≤ 1 := by
  unfold contribution
  split_ifs
  · constructor
    · positivity
    · apply (div_le_one (by positivity : (0 : ℝ) < (max s.card 1 : ℕ))).mpr
      exact_mod_cast (le_max_right s.card 1)
  · norm_num

lemma sum_contribution {m : ℕ} (s H0 : Finset (Fin m)) :
    ∑ j ∈ H0, contribution s j =
      ((H0 ∩ s).card : ℝ) / (max s.card 1 : ℕ) := by
  classical
  simp only [contribution, ← sum_filter]
  simp [filter_mem_eq_inter, sum_const, nsmul_eq_mul, div_eq_mul_inv]

section Probability

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]

lemma integrable_contribution {m : ℕ} {S : Ω → Finset (Fin m)}
    (hS : Measurable S) (j : Fin m) : Integrable (fun ω => contribution (S ω) j) μ := by
  have hmeas : Measurable (fun ω => contribution (S ω) j) :=
    (measurable_of_countable (fun s : Finset (Fin m) => contribution s j)).comp hS
  refine integrable_of_le_of_le hmeas.aestronglyMeasurable
    ?_ ?_ (integrable_const (0 : ℝ)) (integrable_const (1 : ℝ))
  · exact Filter.Eventually.of_forall fun ω => (contribution_bounds (S ω) j).1
  · exact Filter.Eventually.of_forall fun ω => (contribution_bounds (S ω) j).2

lemma integrable_grid {p : Ω → ℝ} (hp : Measurable p) (a : ℝ) (i : ℕ) :
    Integrable (fun ω => gridIndicator a (p ω) i) μ := by
  have hs : MeasurableSet {ω | p ω ≤ a * i} := measurableSet_le hp measurable_const
  exact (integrable_const (1 : ℝ)).indicator hs

omit [IsProbabilityMeasure μ] in
lemma integral_grid {p : Ω → ℝ} (hp : Measurable p) (a : ℝ) (i : ℕ) :
    (∫ ω, gridIndicator a (p ω) i ∂μ) = μ.real {ω | p ω ≤ a * i} := by
  exact integral_indicator_one (measurableSet_le hp measurable_const)

lemma integrable_envelope {p : Ω → ℝ} (hp : Measurable p) (a : ℝ) (n : ℕ) :
    Integrable (fun ω => envelope n (gridIndicator a (p ω))) μ := by
  exact ((integrable_grid μ hp a n).div_const n).add
    (integrable_finsetSum _ fun i _ => (integrable_grid μ hp a (i + 1)).mul_const _)

lemma integral_envelope {p : Ω → ℝ} (hp : Measurable p) (a : ℝ) (n : ℕ) :
    (∫ ω, envelope n (gridIndicator a (p ω)) ∂μ) =
      envelope n (fun i => μ.real {ω | p ω ≤ a * i}) := by
  unfold envelope
  rw [integral_add ((integrable_grid μ hp a n).div_const n)
    (integrable_finsetSum _ fun i _ => (integrable_grid μ hp a (i + 1)).mul_const _)]
  rw [integral_div, integral_grid μ hp,
    integral_finsetSum _ (fun i _ => (integrable_grid μ hp a (i + 1)).mul_const _)]
  simp_rw [integral_mul_const, integral_grid μ hp]

lemma contribution_le_envelope {m : ℕ} (hm : 1 ≤ m) {a : ℝ} (ha : 0 ≤ a)
    (p : ℝ) (s : Finset (Fin m)) (j : Fin m)
    (hsc : j ∈ s → p ≤ a * s.card) :
    contribution s j ≤ envelope m (gridIndicator a p) := by
  classical
  by_cases hj : j ∈ s
  · have hr : 1 ≤ s.card := card_pos.mpr ⟨j, hj⟩
    have hrm : s.card ≤ m := by simpa using card_le_univ s
    have hg : gridIndicator a p s.card = 1 := ite_eq_left (hsc hj)
    have h := ratio_le_envelope hm hr hrm (gridIndicator_nonneg a p)
      (gridIndicator_monotone ha p)
    simpa [contribution, hj, max_eq_left hr, hg] using h
  · simpa [contribution, hj] using envelope_nonneg (gridIndicator_nonneg a p) (n := m)

lemma null_contribution_bound {m : ℕ} (hm : 1 ≤ m) {a δ : ℝ} (ha : 0 ≤ a)
    {p : Ω → ℝ} (hp : Measurable p) {S : Ω → Finset (Fin m)} (hS : Measurable S)
    (j : Fin m) (hsc : ∀ ω, j ∈ S ω → p ω ≤ a * (S ω).card)
    (hgrid : ∀ i, 1 ≤ i → i ≤ m → μ.real {ω | p ω ≤ a * i} ≤ a * i + δ) :
    (∫ ω, contribution (S ω) j ∂μ) ≤ a * harmonic m + δ := by
  calc
    _ ≤ ∫ ω, envelope m (gridIndicator a (p ω)) ∂μ :=
      integral_mono (integrable_contribution μ hS j) (integrable_envelope μ hp a m)
        (fun ω => contribution_le_envelope hm ha (p ω) (S ω) j (hsc ω))
    _ = envelope m (fun i => μ.real {ω | p ω ≤ a * i}) := integral_envelope μ hp a m
    _ ≤ envelope m (fun i => a * i + δ) := envelope_mono hgrid hm
    _ = a * harmonic m + δ := envelope_linear m a δ hm

/-- Finite weighted BY control from self-consistency and marginal grid bounds. -/
theorem fdr_le_of_selfConsistent {m : ℕ} (hm : 1 ≤ m)
    (q δ : ℝ) (hq : 0 ≤ q) (w : Fin m → ℝ) (hw : ∀ j, 0 ≤ w j)
    (H0 : Finset (Fin m)) (p : Fin m → Ω → ℝ) (hp : ∀ j, Measurable (p j))
    (S : Ω → Finset (Fin m)) (hS : Measurable S)
    (hsc : ∀ ω j, j ∈ S ω → p j ω ≤ threshold m q w j (S ω).card)
    (hgrid : ∀ j ∈ H0, ∀ i, 1 ≤ i → i ≤ m →
      μ.real {ω | p j ω ≤ threshold m q w j i} ≤ threshold m q w j i + δ) :
    fdr μ S H0 ≤ (q / m) * (∑ j ∈ H0, w j) + H0.card * δ := by
  classical
  have hm0 : (m : ℝ) ≠ 0 := by exact_mod_cast (by omega : m ≠ 0)
  have hc : harmonic m ≠ 0 := ne_of_gt (harmonic_pos hm)
  have hj : ∀ j ∈ H0, (∫ ω, contribution (S ω) j ∂μ) ≤ (q / m) * w j + δ := by
    intro j hj
    have ha : 0 ≤ q * w j / (m * harmonic m) :=
      div_nonneg (mul_nonneg hq (hw j))
        (mul_nonneg (Nat.cast_nonneg m) (harmonic_pos hm).le)
    have h := null_contribution_bound μ hm ha (hp j) hS j
      (by intro ω hj; simpa only [threshold_eq] using hsc ω j hj)
      (by intro i hi him; simpa only [threshold_eq] using hgrid j hj i hi him)
    have he : q * w j / (m * harmonic m) * harmonic m = (q / m) * w j := by
      field_simp [hc]
    simpa only [he] using h
  calc
    fdr μ S H0 = ∫ ω, ∑ j ∈ H0, contribution (S ω) j ∂μ := by
      unfold fdr
      apply integral_congr_ae
      exact Filter.Eventually.of_forall fun ω => (sum_contribution (S ω) H0).symm
    _ = ∑ j ∈ H0, ∫ ω, contribution (S ω) j ∂μ :=
      integral_finsetSum H0 (fun j _ => integrable_contribution μ hS j)
    _ ≤ ∑ j ∈ H0, ((q / m) * w j + δ) := sum_le_sum hj
    _ = _ := by rw [sum_add_distrib, ← mul_sum]; simp [nsmul_eq_mul]

end Probability

end OranCausal.StepUp

#print axioms OranCausal.StepUp.fdr_le_of_selfConsistent
