module

public import OranCausal.StepUp

public section

open Finset
open scoped BigOperators

namespace OranCausal.GateBudget

/-- Development-data tolerance for one class and one guard KPI. -/
noncomputable def tolerance (N G : ℝ) : ℝ := (1 / 20 : ℝ) * G / (3 * max N 1)

theorem tolerance_nonneg (N G : ℝ) (hG : 0 < G) : 0 ≤ tolerance N G := by
  unfold tolerance
  have hm : (0 : ℝ) < max N 1 := lt_of_lt_of_le zero_lt_one (le_max_right _ _)
  positivity

/-- The full first-order development budget is exactly five percentage points. -/
theorem budget_identity (N G : ℝ) (hG : 0 < G) :
    3 * max N 1 * tolerance N G / G = (1 / 20 : ℝ) := by
  have hm : (0 : ℝ) < max N 1 := lt_of_lt_of_le zero_lt_one (le_max_right _ _)
  unfold tolerance
  field_simp

theorem budget_bound (N G : ℝ) (hG : 0 < G) :
    3 * max N 1 * tolerance N G / G ≤ (1 / 20 : ℝ) :=
  le_of_eq (budget_identity N G hG)

/-- Additive movement with a possibly nonintegral mean count within the budget. -/
theorem mean_count_budget (N G count movement : ℝ) (hG : 0 < G)
    (hcount : count ≤ max N 1)
    (hadd : |movement| ≤ 3 * count * tolerance N G) :
    |movement / G| ≤ (1 / 20 : ℝ) := by
  rw [abs_div, abs_of_pos hG]
  calc
    _ ≤ (3 * count * tolerance N G) / G := div_le_div_of_nonneg_right hadd hG.le
    _ ≤ (3 * max N 1 * tolerance N G) / G := by
      apply div_le_div_of_nonneg_right _ hG.le
      exact mul_le_mul_of_nonneg_right (by linarith) (tolerance_nonneg N G hG)
    _ = _ := budget_identity N G hG

/-- First-order ratio movement over rejected units and the three relations. -/
theorem additive_guard_budget {I : Type*} (units : Finset I) (N G ratioMovement : ℝ)
    (effect : I → Fin 3 → ℝ) (hG : 0 < G)
    (hcount : (units.card : ℝ) ≤ max N 1)
    (hedge : ∀ i ∈ units, ∀ r, |effect i r| ≤ tolerance N G)
    (hadd : ratioMovement = (∑ i ∈ units, ∑ r, effect i r) / G) :
    |ratioMovement| ≤ (1 / 20 : ℝ) := by
  rw [hadd]
  apply mean_count_budget N G (units.card : ℝ) _ hG hcount
  calc
    |∑ i ∈ units, ∑ r, effect i r| ≤ ∑ i ∈ units, |∑ r, effect i r| :=
      abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ units, ∑ r : Fin 3, |effect i r| :=
      sum_le_sum fun i _ => abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i ∈ units, ∑ _r : Fin 3, tolerance N G :=
      sum_le_sum fun i hi => sum_le_sum fun r _ => hedge i hi r
    _ = _ := by simp; ring

/-- Scope note: two units exceed the count budget when the development mean is one. -/
theorem count_overrun_example :
    tolerance 1 100 = (5 / 3 : ℝ) ∧
    (2 : ℝ) > max 1 1 ∧
    ((∑ _i : Fin 2, ∑ _r : Fin 3, tolerance 1 100) / 100 = (1 / 10 : ℝ)) ∧
    (∑ _i : Fin 2, ∑ _r : Fin 3, tolerance 1 100) / 100 > (1 / 20 : ℝ) := by
  norm_num [tolerance]

end OranCausal.GateBudget

#print axioms OranCausal.GateBudget.budget_bound
#print axioms OranCausal.GateBudget.mean_count_budget
#print axioms OranCausal.GateBudget.additive_guard_budget
#print axioms OranCausal.GateBudget.count_overrun_example
