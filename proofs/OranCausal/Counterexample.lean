module

public import OranCausal.StepUp
public import Mathlib.Data.Nat.Choose.Central
public import Mathlib.Data.Nat.Choose.Sum
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Analysis.SpecificLimits.Basic

public section

open Finset Filter
open scoped BigOperators Topology Classical

namespace OranCausal.Counterexample

/-- Two independent fair assignment bits; `true` means acceptance.
The second bit is unused if the first unit was accepted. -/
abbrev Outcome := Bool × Bool

def hasSecond (ω : Outcome) : Bool := !ω.1

def signed (b : Bool) : ℝ := if b then 1 else -1

/-- Twice the PMRT score, holding the realised existence of unit 2 fixed. -/
def doubledScore (a : ℝ) (second : Bool) (b : Outcome) : ℝ :=
  signed b.1 + if second then a * signed b.2 else 0

/-- Four times the design variance for the realised unit set. -/
def varianceFour (a : ℝ) (second : Bool) : ℝ :=
  1 + if second then a ^ 2 else 0

noncomputable def z (a : ℝ) (second : Bool) (b : Outcome) : ℝ :=
  doubledScore a second b / Real.sqrt (varianceFour a second)

lemma varianceFour_pos (a : ℝ) (second : Bool) : 0 < varianceFour a second := by
  cases second
  · norm_num [varianceFour]
  · dsimp [varianceFour]; positivity

lemma z_le_iff (a : ℝ) (second : Bool) (b c : Outcome) :
    z a second b ≤ z a second c ↔ doubledScore a second b ≤ doubledScore a second c := by
  exact div_le_div_iff_of_pos_right (Real.sqrt_pos.mpr (varianceFour_pos a second))

/-- Probability under the uniform law on the four assignment bit pairs. -/
noncomputable def assignmentProbability (E : Outcome → Prop) : ℝ :=
  (∑ ω : Outcome, if E ω then (1 : ℝ) else 0) / 4

/-- Exact one-sided redraw tail, including ties, on the observed unit set. -/
noncomputable def pInfinity (a : ℝ) (ω : Outcome) : ℝ :=
  assignmentProbability fun b => z a (hasSecond ω) ω ≤ z a (hasSecond ω) b

lemma pInfinity_values (a : ℝ) (ha : 1 < a) (ω : Outcome) :
    pInfinity a ω = if ω.1 || ω.2 then (1 : ℝ) / 2 else 1 := by
  rcases ω with ⟨x, y⟩
  simp only [pInfinity, assignmentProbability, Fintype.sum_prod_type, Fintype.sum_bool]
  cases x <;> cases y <;>
    simp [hasSecond, z_le_iff, doubledScore, signed] <;>
    split_ifs <;> norm_num at * <;> linarith

/-- The ideal redraw-tail p-value rejects with probability three quarters. -/
theorem exact_tail_rejection (a : ℝ) (ha : 1 < a) :
    assignmentProbability (fun ω => pInfinity a ω ≤ 1 / 2) = 3 / 4 := by
  simp only [assignmentProbability, pInfinity_values a ha,
    Fintype.sum_prod_type, Fintype.sum_bool]
  norm_num

/-- The plus-one Monte Carlo p-value for `B` redraws and `k` exceedances. -/
noncomputable def monteCarloP (B k : ℕ) : ℝ := (1 + (k : ℝ)) / (B + 1)

lemma monteCarloP_le_half (B k : ℕ) :
    monteCarloP B k ≤ 1 / 2 ↔ 2 * k < B := by
  rw [monteCarloP, div_le_iff₀ (by positivity)]
  constructor
  · intro h
    have : (2 : ℝ) * k < B := by linarith
    exact_mod_cast this
  · intro h
    have : 2 * k + 1 ≤ B := h
    have : (2 : ℝ) * k + 1 ≤ B := by exact_mod_cast this
    linarith

/-- Binomial law of the exceedance count for conditionally iid redraws. -/
noncomputable def binomialMass (B k : ℕ) (p : ℝ) : ℝ :=
  (B.choose k : ℝ) * p ^ k * (1 - p) ^ (B - k)

lemma binomialMass_sum (B : ℕ) (p : ℝ) :
    ∑ k ∈ range (B + 1), binomialMass B k p = 1 := by
  have h := add_pow p (1 - p) B
  rw [show p + (1 - p) = 1 by ring, one_pow] at h
  simpa [binomialMass, mul_comm, mul_left_comm, mul_assoc] using h.symm

lemma binomialMass_nonneg (B k : ℕ) {p : ℝ} (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ binomialMass B k p := by
  exact mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg hp _))
    (pow_nonneg (sub_nonneg.mpr hp1) _)

/-- Probability of rejecting at one half, averaged over the observed assignment
and its conditional binomial redraw-count law. -/
noncomputable def monteCarloRejection (a : ℝ) (B : ℕ) : ℝ :=
  (∑ ω : Outcome, ∑ k ∈ range (B + 1),
    if monteCarloP B k ≤ 1 / 2 then binomialMass B k (pInfinity a ω) else 0) / 4

noncomputable def halfTail (B : ℕ) : ℝ :=
  (∑ k ∈ range (B + 1), if 2 * k < B then (B.choose k : ℝ) else 0) / 2 ^ B

noncomputable def centralMass (n : ℕ) : ℝ := (n.centralBinom : ℝ) / 4 ^ n

lemma centralMass_nonneg (n : ℕ) : 0 ≤ centralMass n := by
  unfold centralMass
  positivity

lemma centralMass_succ (n : ℕ) :
    centralMass (n + 1) = ((2 * n + 1) / (2 * (n + 1) : ℝ)) * centralMass n := by
  have h : ((n : ℝ) + 1) * ((n + 1).centralBinom : ℝ) =
      2 * (2 * n + 1) * (n.centralBinom : ℝ) := by
    exact_mod_cast Nat.succ_mul_centralBinom_succ n
  unfold centralMass
  rw [pow_succ]
  field_simp
  nlinarith

lemma centralMass_sq_bound (n : ℕ) : centralMass n ^ 2 ≤ 1 / (n + 1 : ℝ) := by
  induction n with
  | zero => norm_num [centralMass]
  | succ n ih =>
    rw [centralMass_succ, mul_pow]
    calc
      _ ≤ ((2 * n + 1) / (2 * (n + 1) : ℝ)) ^ 2 * (1 / (n + 1 : ℝ)) :=
        mul_le_mul_of_nonneg_left ih (sq_nonneg _)
      _ ≤ 1 / ((n + 1 : ℕ) + 1 : ℝ) := by
        push_cast
        field_simp
        nlinarith [Nat.cast_nonneg (α := ℝ) n]

lemma tendsto_centralMass : Tendsto centralMass atTop (𝓝 0) := by
  have h : Tendsto (fun n => centralMass n ^ 2) atTop (𝓝 0) :=
    squeeze_zero (fun n => sq_nonneg _) centralMass_sq_bound
      tendsto_one_div_add_atTop_nhds_zero_nat
  have hs := Real.continuous_sqrt.continuousAt.tendsto.comp h
  simpa only [Function.comp_def, Real.sqrt_sq (centralMass_nonneg _), Real.sqrt_zero] using hs

noncomputable def tieMass (B : ℕ) : ℝ :=
  (∑ k ∈ range (B + 1), if 2 * k = B then (B.choose k : ℝ) else 0) / 2 ^ B

lemma tieMass_eq (B : ℕ) :
    tieMass B = if 2 * (B / 2) = B then centralMass (B / 2) else 0 := by
  unfold tieMass
  rw [sum_eq_single (B / 2)]
  · split_ifs with h
    · conv_lhs => rw [← h]
      norm_num [centralMass, Nat.centralBinom, pow_mul]
    · simp
  · intro k hk hne
    have h : ¬ 2 * k = B := by omega
    simp [h]
  · intro h
    have : B / 2 ∈ range (B + 1) := mem_range.mpr (by omega)
    exact (h this).elim

lemma tendsto_tieMass : Tendsto tieMass atTop (𝓝 0) := by
  apply squeeze_zero (fun B => by
      rw [tieMass_eq]; split_ifs; exact centralMass_nonneg _; exact le_rfl)
    (fun B => show tieMass B ≤ centralMass (B / 2) by
      rw [tieMass_eq]; split_ifs <;> first | exact le_rfl | exact centralMass_nonneg _)
  exact tendsto_centralMass.comp (Nat.tendsto_div_const_atTop (by decide : 2 ≠ 0))

lemma halfTail_eq (B : ℕ) : halfTail B = (1 - tieMass B) / 2 := by
  let L : ℝ := ∑ k ∈ range (B + 1), if 2 * k < B then (B.choose k : ℝ) else 0
  let U : ℝ := ∑ k ∈ range (B + 1), if B < 2 * k then (B.choose k : ℝ) else 0
  let M : ℝ := ∑ k ∈ range (B + 1), if 2 * k = B then (B.choose k : ℝ) else 0
  have hu : U = L := by
    dsimp [U, L]
    rw [← sum_range_reflect (fun k => if B < 2 * k then (B.choose k : ℝ) else 0)]
    apply sum_congr rfl
    intro k hk
    have hkB : k ≤ B := by have := mem_range.mp hk; omega
    simp only [Nat.add_sub_cancel]
    rw [Nat.choose_symm hkB]
    have hh : B < 2 * (B - k) ↔ 2 * k < B := by omega
    simp only [hh]
  have ht : L + U + M = (2 : ℝ) ^ B := by
    calc
      _ = ∑ k ∈ range (B + 1), (B.choose k : ℝ) := by
        dsimp [L, U, M]
        rw [← sum_add_distrib, ← sum_add_distrib]
        apply sum_congr rfl
        intro k _
        rcases lt_trichotomy (2 * k) B with h | h | h
        · simp [h, ne_of_lt h, not_lt_of_ge h.le]
        · simp [h]
        · simp [h, ne_of_gt h, not_lt_of_ge h.le]
      _ = _ := by exact_mod_cast Nat.sum_range_choose B
  change L / 2 ^ B = (1 - M / 2 ^ B) / 2
  rw [hu] at ht
  field_simp
  linarith

lemma tendsto_halfTail : Tendsto halfTail atTop (𝓝 (1 / 2)) := by
  simpa only [sub_zero] using
    ((tendsto_const_nhds.sub tendsto_tieMass).div_const 2).congr
      (fun B => (halfTail_eq B).symm)

lemma binomialMass_half (B k : ℕ) (hk : k ≤ B) :
    binomialMass B k (1 / 2) = (B.choose k : ℝ) / 2 ^ B := by
  unfold binomialMass
  norm_num only [show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num]
  rw [mul_assoc, ← pow_add, Nat.add_sub_of_le hk, one_div_pow, mul_one_div]

lemma certain_tail_never_rejects (B : ℕ) :
    (∑ k ∈ range (B + 1),
      if monteCarloP B k ≤ 1 / 2 then binomialMass B k 1 else 0) = 0 := by
  apply sum_eq_zero
  intro k hk
  simp only [monteCarloP_le_half]
  split_ifs with h
  · have hsub : B - k ≠ 0 := by omega
    simp [binomialMass, hsub]
  · rfl

lemma half_tail_rejection (B : ℕ) :
    (∑ k ∈ range (B + 1),
      if monteCarloP B k ≤ 1 / 2 then binomialMass B k (1 / 2) else 0) = halfTail B := by
  unfold halfTail
  rw [sum_div]
  apply sum_congr rfl
  intro k hk
  simp only [monteCarloP_le_half,
    binomialMass_half B k (by have := mem_range.mp hk; omega)]
  split_ifs <;> simp

theorem monteCarloRejection_eq (a : ℝ) (ha : 1 < a) (B : ℕ) :
    monteCarloRejection a B = (3 / 4 : ℝ) * halfTail B := by
  simp only [monteCarloRejection, pInfinity_values a ha,
    Fintype.sum_prod_type, Fintype.sum_bool, Bool.false_or, Bool.true_or,
    Bool.or_false, Bool.false_eq_true, ite_true, ite_false]
  rw [certain_tail_never_rejects, half_tail_rejection]
  ring

/-- At the discontinuity one half, the finite-redraw rejection probabilities
converge to three eighths, not the exact-tail rejection probability. -/
theorem finite_redraw_rejection_limit (a : ℝ) (ha : 1 < a) :
    Tendsto (monteCarloRejection a) atTop (𝓝 (3 / 8)) := by
  have h := tendsto_const_nhds.mul tendsto_halfTail (a := (3 / 4 : ℝ))
  norm_num at h
  exact h.congr (fun B => (monteCarloRejection_eq a ha B).symm)

theorem finite_redraw_rejection_formula (a : ℝ) (ha : 1 < a) (B : ℕ) :
    monteCarloRejection a B = (3 / 8 : ℝ) *
      (1 - if 2 * (B / 2) = B then centralMass (B / 2) else 0) := by
  rw [monteCarloRejection_eq a ha, halfTail_eq, tieMass_eq]
  ring

theorem one_redraw_rejection (a : ℝ) (ha : 1 < a) :
    monteCarloRejection a 1 = 3 / 8 := by
  norm_num [finite_redraw_rejection_formula a ha]

theorem exact_tail_violates_validity (a : ℝ) (ha : 1 < a) :
    (1 / 2 : ℝ) < assignmentProbability (fun ω => pInfinity a ω ≤ 1 / 2) := by
  rw [exact_tail_rejection a ha]
  norm_num

/-- Binomial second moment in doubled, centred count coordinates. -/
lemma choose_centered_second_moment (B : ℕ) :
    (∑ k ∈ range (B + 1), (B.choose k : ℝ) * (2 * k - B) ^ 2) = (B : ℝ) * 2 ^ B := by
  induction B with
  | zero => norm_num
  | succ B ih =>
    have hr := sum_choose_succ_mul
      (fun k _ : ℕ => (2 * (k : ℝ) - ((B : ℝ) + 1)) ^ 2) B
    have hs : (∑ k ∈ range (B + 1), (B.choose k : ℝ)) = 2 ^ B := by
      exact_mod_cast Nat.sum_range_choose B
    push_cast at hr ⊢
    rw [show B + 1 + 1 = B + 2 by omega]
    rw [hr, ← sum_add_distrib]
    calc
      _ = 2 * (∑ k ∈ range (B + 1), (B.choose k : ℝ) * (2 * k - B) ^ 2) +
          2 * ∑ k ∈ range (B + 1), (B.choose k : ℝ) := by
        rw [mul_sum, mul_sum, ← sum_add_distrib]
        apply sum_congr rfl
        intro k _
        ring
      _ = _ := by rw [ih, hs, pow_succ]; ring

lemma fair_count_second_moment (B : ℕ) :
    (∑ k ∈ range (B + 1), binomialMass B k (1 / 2) * (2 * k - B) ^ 2) = (B : ℝ) := by
  have he : (∑ k ∈ range (B + 1), binomialMass B k (1 / 2) * (2 * k - B) ^ 2) =
      (∑ k ∈ range (B + 1), (B.choose k : ℝ) * (2 * k - B) ^ 2) / 2 ^ B := by
    rw [sum_div]
    apply sum_congr rfl
    intro k hk
    rw [binomialMass_half B k (by simpa using mem_range.mp hk)]
    ring
  rw [he, choose_centered_second_moment]
  field_simp

/-- A second-moment bound for the actual plus-one p-value, valid also at B=0. -/
lemma plus_one_second_moment_bound (B : ℕ) :
    (∑ k ∈ range (B + 1), binomialMass B k (1 / 2) * (monteCarloP B k - 1 / 2) ^ 2)
      ≤ 1 / (2 * ((B : ℝ) + 1)) := by
  have hB : 0 < (B : ℝ) + 1 := by positivity
  calc
    _ ≤ ∑ k ∈ range (B + 1), binomialMass B k (1 / 2) *
        ((2 * (2 * k - B) ^ 2 + 2) / (4 * ((B : ℝ) + 1) ^ 2)) := by
      apply sum_le_sum
      intro k _
      apply mul_le_mul_of_nonneg_left _ (binomialMass_nonneg B k (by norm_num) (by norm_num))
      have he : monteCarloP B k - 1 / 2 = (2 * k - B + 1) / (2 * ((B : ℝ) + 1)) := by
        unfold monteCarloP
        field_simp
        ring
      rw [he, div_pow, mul_pow]
      norm_num only [show (2 : ℝ) ^ 2 = 4 by norm_num]
      apply (div_le_div_iff_of_pos_right (by positivity : 0 < 4 * ((B : ℝ) + 1) ^ 2)).mpr
      nlinarith [sq_nonneg (2 * (k : ℝ) - B - 1)]
    _ = (2 * (B : ℝ) + 2) / (4 * ((B : ℝ) + 1) ^ 2) := by
      simp_rw [← mul_div_assoc, mul_add]
      rw [← sum_div, sum_add_distrib]
      have h1 := fair_count_second_moment B
      have h0 := binomialMass_sum B (1 / 2)
      have he : (∑ k ∈ range (B + 1), binomialMass B k (1 / 2) *
          (2 * (2 * k - B) ^ 2)) = 2 * (B : ℝ) := by
        calc
          _ = 2 * ∑ k ∈ range (B + 1), binomialMass B k (1 / 2) * (2 * k - B) ^ 2 := by
            rw [mul_sum]; apply sum_congr rfl; intros; ring
          _ = _ := by rw [h1]
      rw [he, ← sum_mul, h0]
      ring
    _ = _ := by field_simp; ring

noncomputable def binomialRejectionAt (B : ℕ) (p t : ℝ) : ℝ :=
  ∑ k ∈ range (B + 1), if monteCarloP B k ≤ t then binomialMass B k p else 0

noncomputable def monteCarloRejectionAt (a t : ℝ) (B : ℕ) : ℝ :=
  (∑ ω : Outcome, binomialRejectionAt B (pInfinity a ω) t) / 4

theorem monteCarloRejectionAt_half (a : ℝ) (B : ℕ) :
    monteCarloRejectionAt a (1 / 2) B = monteCarloRejection a B := by
  unfold monteCarloRejectionAt binomialRejectionAt monteCarloRejection
  rfl

lemma fair_nonrejection_bound (B : ℕ) (t : ℝ) (ht : 1 / 2 < t) :
    0 ≤ 1 - binomialRejectionAt B (1 / 2) t ∧
    1 - binomialRejectionAt B (1 / 2) t ≤
      (1 / (2 * ((B : ℝ) + 1))) / (t - 1 / 2) ^ 2 := by
  have hε : 0 < t - 1 / 2 := sub_pos.mpr ht
  have he : 1 - binomialRejectionAt B (1 / 2) t =
      ∑ k ∈ range (B + 1), if t < monteCarloP B k then binomialMass B k (1 / 2) else 0 := by
    conv_lhs => lhs; rw [← binomialMass_sum B (1 / 2)]
    unfold binomialRejectionAt
    rw [← sum_sub_distrib]
    apply sum_congr rfl
    intro k _
    by_cases h : monteCarloP B k ≤ t
    · simp [h, not_lt_of_ge h]
    · simp [h, lt_of_not_ge h]
  rw [he]
  constructor
  · apply sum_nonneg
    intro k _
    split_ifs
    · exact binomialMass_nonneg B k (by norm_num) (by norm_num)
    · exact le_rfl
  · calc
      _ ≤ (∑ k ∈ range (B + 1), binomialMass B k (1 / 2) *
          (monteCarloP B k - 1 / 2) ^ 2) / (t - 1 / 2) ^ 2 := by
        rw [sum_div]
        apply sum_le_sum
        intro k _
        have hm := binomialMass_nonneg B k (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num)
        split_ifs with hk
        · apply (le_div_iff₀ (sq_pos_of_pos hε)).mpr
          apply mul_le_mul_of_nonneg_left _ hm
          nlinarith
        · positivity
      _ ≤ _ := div_le_div_of_nonneg_right (plus_one_second_moment_bound B) (sq_nonneg _)

lemma fair_rejection_above_half (t : ℝ) (ht : 1 / 2 < t) :
    Tendsto (fun B => binomialRejectionAt B (1 / 2) t) atTop (𝓝 1) := by
  have hz : Tendsto (fun B => 1 - binomialRejectionAt B (1 / 2) t) atTop (𝓝 0) := by
    apply squeeze_zero (fun B => (fair_nonrejection_bound B t ht).1)
      (fun B => (fair_nonrejection_bound B t ht).2)
    have h := (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).div_const
      (2 * (t - 1 / 2) ^ 2)
    simpa only [zero_div] using h.congr (fun B => by
      rw [div_div, div_div]
      congr 1
      ring)
  have h := tendsto_const_nhds.sub hz (a := (1 : ℝ))
  simpa only [sub_sub_cancel, sub_zero] using h

lemma certain_nonrejection_below_one (B : ℕ) (t : ℝ) (ht : t < 1) :
    binomialRejectionAt B 1 t = 0 := by
  unfold binomialRejectionAt
  apply sum_eq_zero
  intro k hk
  by_cases he : k = B
  · subst k
    have hp : monteCarloP B B = 1 := by unfold monteCarloP; field_simp; ring
    simp [hp, not_le.mpr ht]
  · have hsub : B - k ≠ 0 := by have := mem_range.mp hk; omega
    simp [binomialMass, hsub]

theorem monteCarloRejectionAt_eq (a t : ℝ) (ha : 1 < a) (ht : t < 1) (B : ℕ) :
    monteCarloRejectionAt a t B = (3 / 4 : ℝ) * binomialRejectionAt B (1 / 2) t := by
  simp only [monteCarloRejectionAt, pInfinity_values a ha,
    Fintype.sum_prod_type, Fintype.sum_bool, Bool.false_or, Bool.true_or,
    Bool.or_false, Bool.false_eq_true, ite_true, ite_false]
  rw [certain_nonrejection_below_one B t ht]
  ring

/-- Away from the atom at one half, the actual finite-redraw test has rejection limit 3/4. -/
theorem finite_redraw_rejection_limit_above_half (a t : ℝ) (ha : 1 < a)
    (ht : 1 / 2 < t) (ht1 : t < 1) :
    Tendsto (monteCarloRejectionAt a t) atTop (𝓝 (3 / 4)) := by
  have h := tendsto_const_nhds.mul (fair_rejection_above_half t ht) (a := (3 / 4 : ℝ))
  simpa only [mul_one] using h.congr (fun B => (monteCarloRejectionAt_eq a t ha ht1 B).symm)

theorem finite_redraw_rejection_limit_between_half_and_three_quarters (a t : ℝ)
    (ha : 1 < a) (ht : 1 / 2 < t) (htq : t < 3 / 4) :
    Tendsto (monteCarloRejectionAt a t) atTop (𝓝 (3 / 4)) :=
  finite_redraw_rejection_limit_above_half a t ha ht (by linarith)

end OranCausal.Counterexample

#print axioms OranCausal.Counterexample.exact_tail_rejection
#print axioms OranCausal.Counterexample.finite_redraw_rejection_limit
#print axioms OranCausal.Counterexample.finite_redraw_rejection_limit_above_half
#print axioms OranCausal.Counterexample.finite_redraw_rejection_limit_between_half_and_three_quarters
