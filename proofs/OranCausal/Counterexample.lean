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

end OranCausal.Counterexample

#print axioms OranCausal.Counterexample.exact_tail_rejection
#print axioms OranCausal.Counterexample.finite_redraw_rejection_limit
