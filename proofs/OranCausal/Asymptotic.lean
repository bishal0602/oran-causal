module

public import OranCausal.StepUp
public import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.IntegrationByParts
public import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
public import Mathlib.Probability.Martingale.Basic
public import Mathlib.MeasureTheory.Function.L2Space
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.Indicator
public import Mathlib.MeasureTheory.Measure.Portmanteau
public import Mathlib.Probability.Moments.Variance
public import Mathlib.Probability.Distributions.Gaussian.Real
public import Mathlib.Probability.CDF
public import Mathlib.Data.Finset.Powerset

public section

open MeasureTheory ProbabilityTheory Finset Filter BoundedContinuousFunction
open scoped BigOperators Topology Classical

namespace OranCausal.Asymptotic

noncomputable def binomialMass (B j : ℕ) (p : ℝ) : ℝ :=
  (B.choose j : ℝ) * p ^ j * (1 - p) ^ (B - j)

noncomputable def betaNat (a b : ℕ) : ℝ :=
  ∫ x in (0 : ℝ)..1, x ^ a * (1 - x) ^ b

lemma betaNat_recurrence (a b : ℕ) :
    (a + 1 : ℝ) * betaNat a (b + 1) = (b + 1 : ℝ) * betaNat (a + 1) b := by
  have h := intervalIntegral.integral_deriv_mul_eq_sub_of_hasDerivAt
    (u := fun x : ℝ => x ^ (a + 1)) (v := fun x : ℝ => (1 - x) ^ (b + 1))
    (u' := fun x => (a + 1 : ℝ) * x ^ a)
    (v' := fun x => -((b + 1 : ℝ) * (1 - x) ^ b))
    (a := 0) (b := 1)
    (by fun_prop) (by fun_prop)
    (by intro x _; convert! (hasDerivAt_id x).pow (a + 1) using 1; simp)
    (by intro x _; convert! ((hasDerivAt_id x).const_sub 1).pow (b + 1) using 1; simp)
    (by apply Continuous.intervalIntegrable; fun_prop)
    (by apply Continuous.intervalIntegrable; fun_prop)
  have he : (fun x : ℝ => (a + 1 : ℝ) * x ^ a * (1 - x) ^ (b + 1) +
      x ^ (a + 1) * -((b + 1 : ℝ) * (1 - x) ^ b)) =
      (fun x => (a + 1 : ℝ) * (x ^ a * (1 - x) ^ (b + 1)) -
        (b + 1 : ℝ) * (x ^ (a + 1) * (1 - x) ^ b)) := by funext x; ring
  rw [he, intervalIntegral.integral_sub
    (by apply Continuous.intervalIntegrable; fun_prop)
    (by apply Continuous.intervalIntegrable; fun_prop)] at h
  simp only [intervalIntegral.integral_const_mul] at h
  simp only [one_pow, sub_self, zero_pow (Nat.succ_ne_zero _), mul_zero, zero_mul, sub_zero] at h
  change (a + 1 : ℝ) * betaNat a (b + 1) - (b + 1 : ℝ) * betaNat (a + 1) b = 0 at h
  linarith

theorem betaNat_factorial (a b : ℕ) :
    betaNat a b = (a.factorial : ℝ) * b.factorial / (a + b + 1).factorial := by
  induction b generalizing a with
  | zero =>
    simp [betaNat, integral_pow, Nat.factorial_succ]
    field_simp
  | succ b ih =>
    have h := betaNat_recurrence a b
    rw [ih] at h
    rw [Nat.factorial_succ a] at h
    rw [Nat.factorial_succ b]
    push_cast at h ⊢
    have hd : (a + 1 + b + 1) = (a + (b + 1) + 1) := by omega
    rw [hd] at h
    apply (eq_div_iff (by positivity)).mpr
    have ha : (a + 1 : ℝ) ≠ 0 := by positivity
    field_simp at h
    nlinarith

theorem binomial_uniform_integral (B j : ℕ) (hj : j ≤ B) :
    (∫ x in (0 : ℝ)..1, binomialMass B j x) = 1 / (B + 1 : ℝ) := by
  simp only [binomialMass, mul_assoc, intervalIntegral.integral_const_mul]
  change (B.choose j : ℝ) * betaNat j (B - j) = _
  rw [betaNat_factorial, Nat.add_sub_of_le hj]
  have h := Nat.choose_mul_factorial_mul_factorial hj
  have hc : (B.choose j : ℝ) * j.factorial * (B - j).factorial = B.factorial := by
    exact_mod_cast h
  rw [Nat.factorial_succ]
  push_cast
  field_simp
  nlinarith

lemma binomialMass_nonneg (B j : ℕ) {p : ℝ} (hp : p ∈ Set.Icc (0 : ℝ) 1) :
    0 ≤ binomialMass B j p := by
  unfold binomialMass
  exact mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg hp.1 _))
    (pow_nonneg (sub_nonneg.mpr hp.2) _)

lemma binomialMass_le (B j : ℕ) {p : ℝ} (hp : p ∈ Set.Icc (0 : ℝ) 1) :
    binomialMass B j p ≤ B.choose j := by
  have h1 : p ^ j ≤ 1 := pow_le_one₀ hp.1 hp.2
  have h2 : (1 - p) ^ (B - j) ≤ 1 := pow_le_one₀ (by linarith [hp.2]) (by linarith [hp.1])
  have h0 : 0 ≤ (1 - p) ^ (B - j) := pow_nonneg (sub_nonneg.mpr hp.2) _
  calc
    _ ≤ (B.choose j : ℝ) * 1 * 1 := by
      unfold binomialMass
      gcongr
    _ = _ := by ring

noncomputable def clip (p : ℝ) : ℝ := max 0 (min p 1)

lemma clip_mem (p : ℝ) : clip p ∈ Set.Icc (0 : ℝ) 1 := by
  exact ⟨le_max_left _ _, max_le (by norm_num) (min_le_right _ _)⟩

lemma clip_eq {p : ℝ} (hp : p ∈ Set.Icc (0 : ℝ) 1) : clip p = p := by
  simp [clip, min_eq_left hp.2, max_eq_right hp.1]

noncomputable def binomialTest (B j : ℕ) : ℝ →ᵇ ℝ :=
  BoundedContinuousFunction.mkOfBound
    ⟨fun p => binomialMass B j (clip p), by unfold binomialMass clip; fun_prop⟩
    (2 * B.choose j) (by
      intro x y
      rw [Real.dist_eq]
      apply abs_le.mpr
      have hx := binomialMass_le B j (clip_mem x)
      have hy := binomialMass_le B j (clip_mem y)
      have hx0 := binomialMass_nonneg B j (clip_mem x)
      have hy0 := binomialMass_nonneg B j (clip_mem y)
      constructor <;> dsimp at * <;> linarith)

noncomputable def uniformLaw : Measure ℝ := volume.restrict (Set.Ioo 0 1)

instance : IsProbabilityMeasure uniformLaw := by
  constructor
  simp [uniformLaw]

theorem binomialTest_integral (B j : ℕ) (hj : j ≤ B) :
    (∫ x, binomialTest B j x ∂uniformLaw) = 1 / (B + 1 : ℝ) := by
  rw [uniformLaw, restrict_Ioo_eq_restrict_Ioc,
    ← intervalIntegral.integral_of_le (by norm_num : (0 : ℝ) ≤ 1)]
  rw [← binomial_uniform_integral B j hj]
  apply intervalIntegral.integral_congr
  intro x hx
  have hx' : x ∈ Set.Icc (0 : ℝ) 1 := by simpa using hx
  change binomialMass B j (clip x) = binomialMass B j x
  rw [clip_eq hx']

/-- The binomial mixture step uses weak convergence, not an assumed atom limit. -/
theorem binomial_mixture_limit {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (P : ℕ → Ω → ℝ)
    (hP : ∀ n ω, P n ω ∈ Set.Icc (0 : ℝ) 1)
    (hweak : TendstoInDistribution P atTop id (fun _ => μ) uniformLaw)
    (B j : ℕ) (hj : j ≤ B) :
    Tendsto (fun n => ∫ ω, binomialMass B j (P n ω) ∂μ) atTop (𝓝 (1 / (B + 1 : ℝ))) := by
  have h := (tendstoInDistribution_iff_forall_integral_rclike_tendsto ℝ
    hweak.forall_aemeasurable hweak.aemeasurable_limit).mp hweak (binomialTest B j)
  have heq : ∀ n ω, binomialTest B j (P n ω) = binomialMass B j (P n ω) := by
    intro n ω
    change binomialMass B j (clip (P n ω)) = _
    rw [clip_eq (hP n ω)]
  simpa only [heq, id_eq, binomialTest_integral B j hj] using h

noncomputable def gridCDF (B : ℕ) (t : ℝ) : ℝ :=
  (⌊t * (B + 1)⌋₊ : ℝ) / (B + 1)

lemma grid_count (B : ℕ) (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    (∑ j ∈ range (B + 1), if (1 + (j : ℝ)) / (B + 1) ≤ t then
      (1 / (B + 1 : ℝ)) else 0) = gridCDF B t := by
  have hp : (0 : ℝ) < B + 1 := by positivity
  have hn : 0 ≤ t * (B + 1) := mul_nonneg ht.1 hp.le
  have hf : ⌊t * (B + 1)⌋₊ ≤ B + 1 := by
    apply Nat.floor_le_of_le
    push_cast
    nlinarith [ht.2]
  have he : (range (B + 1)).filter (fun j : ℕ => (1 + (j : ℝ)) / (B + 1) ≤ t) =
      range ⌊t * (B + 1)⌋₊ := by
    ext j
    simp only [mem_filter, mem_range, div_le_iff₀ hp]
    have hh : (1 + (j : ℝ) ≤ t * (B + 1)) ↔ j < ⌊t * (B + 1)⌋₊ := by
      rw [← Nat.succ_le_iff, Nat.le_floor_iff hn]
      norm_num [Nat.cast_succ, add_comm]
    rw [hh]
    exact ⟨fun h => h.2, fun h => ⟨lt_of_lt_of_le h hf, h⟩⟩
  rw [← sum_filter, he]
  simp [gridCDF, div_eq_mul_inv]

theorem gridCDF_le (B : ℕ) (t : ℝ) (ht : 0 ≤ t) : gridCDF B t ≤ t := by
  unfold gridCDF
  apply (div_le_iff₀ (by positivity)).mpr
  exact Nat.floor_le (by positivity)

/-- Fixed-grid rejection probabilities converge, even at grid discontinuities. -/
theorem finite_redraw_limit {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (P : ℕ → Ω → ℝ)
    (hP : ∀ n ω, P n ω ∈ Set.Icc (0 : ℝ) 1)
    (hweak : TendstoInDistribution P atTop id (fun _ => μ) uniformLaw)
    (B : ℕ) (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    Tendsto (fun n => ∑ j ∈ range (B + 1),
      if (1 + (j : ℝ)) / (B + 1) ≤ t then ∫ ω, binomialMass B j (P n ω) ∂μ else 0)
      atTop (𝓝 (gridCDF B t)) := by
  rw [← grid_count B t ht]
  apply tendsto_finsetSum
  intro j hj
  by_cases h : (1 + (j : ℝ)) / (B + 1) ≤ t
  · simp only [h, ite_true]
    exact binomial_mixture_limit μ P hP hweak B j (by have := mem_range.mp hj; omega)
  · simp only [h, ite_false]
    exact tendsto_const_nhds

section Moments

variable {Ω : Type*} (F : MeasurableSpace Ω) [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]

lemma weighted_bernoulli (_hF : F ≤ mΩ) (L p c : Ω → ℝ)
    (hL : Measurable L) (hLb : ∀ ω, L ω = 0 ∨ L ω = 1)
    (hc : StronglyMeasurable[F] c) (hci : Integrable c μ)
    (hmean : μ[L | F] =ᵐ[μ] p) :
    μ[(fun ω => c ω * L ω) | F] =ᵐ[μ] (fun ω => c ω * p ω) := by
  have hb : ∀ᵐ ω ∂μ, ‖L ω‖ ≤ 1 := Eventually.of_forall fun ω => by
    rcases hLb ω with h | h <;> simp [h]
  have hi : Integrable L μ := (integrable_const (1 : ℝ)).mono' hL.aestronglyMeasurable hb
  have hprod := hci.mul_bdd hL.aestronglyMeasurable hb
  exact (condExp_mul_of_stronglyMeasurable_left hc hprod hi).trans
    (EventuallyEq.mul (EventuallyEq.refl _ c) hmean)

/-- Conditional first and second moments of a predictably weighted Bernoulli draw. -/
theorem bernoulli_increment_moments (hF : F ≤ mΩ) (L p c : Ω → ℝ)
    (hL : Measurable L) (hLb : ∀ ω, L ω = 0 ∨ L ω = 1)
    (hp : Measurable[F] p) (hpb : ∀ ω, p ω ∈ Set.Icc (0 : ℝ) 1)
    (hc : Measurable[F] c) (hci : Integrable c μ)
    (hc2 : Integrable (fun ω => c ω ^ 2) μ)
    (hmean : μ[L | F] =ᵐ[μ] p) :
    (μ[(fun ω => c ω * (L ω - p ω)) | F] =ᵐ[μ] 0) ∧
    (μ[(fun ω => (c ω * (L ω - p ω)) ^ 2) | F] =ᵐ[μ]
      (fun ω => c ω ^ 2 * p ω * (1 - p ω))) := by
  have hpM := hp.mono hF le_rfl
  have hcM := hc.mono hF le_rfl
  have hbL : ∀ᵐ ω ∂μ, ‖L ω‖ ≤ 1 := Eventually.of_forall fun ω => by
    rcases hLb ω with h | h <;> simp [h]
  have hbp : ∀ᵐ ω ∂μ, ‖p ω‖ ≤ 1 := Eventually.of_forall fun ω => by
    simpa [Real.norm_eq_abs, abs_of_nonneg (hpb ω).1] using (hpb ω).2
  have hiL := hci.mul_bdd hL.aestronglyMeasurable hbL
  have hip := hci.mul_bdd hpM.aestronglyMeasurable hbp
  have hcp : StronglyMeasurable[F] (fun ω => c ω * p ω) := by
    let : MeasurableSpace Ω := F
    exact (hc.mul hp).stronglyMeasurable
  constructor
  · have he : (fun ω => c ω * (L ω - p ω)) =
        (fun ω => c ω * L ω) - (fun ω => c ω * p ω) := by funext ω; simp; ring
    rw [he]
    have hs := condExp_sub hiL hip F
    have hw := weighted_bernoulli F μ hF L p c hL hLb hc.stronglyMeasurable hci hmean
    filter_upwards [hs, hw] with ω hs hw
    simp only [hs, Pi.sub_apply, hw, condExp_of_stronglyMeasurable hF hcp hip, sub_self, Pi.zero_apply]
  · let a : Ω → ℝ := fun ω => c ω ^ 2 * (1 - 2 * p ω)
    let b : Ω → ℝ := fun ω => c ω ^ 2 * p ω ^ 2
    have ha : Measurable[F] a := by let : MeasurableSpace Ω := F; unfold a; fun_prop
    have hb : Measurable[F] b := by let : MeasurableSpace Ω := F; unfold b; fun_prop
    have hai : Integrable a μ := hc2.mul_bdd (c := 1)
      (by fun_prop) (Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_le]; constructor <;> linarith [(hpb ω).1, (hpb ω).2])
    have hbi : Integrable b μ := hc2.mul_bdd
      (hpM.pow_const 2).aestronglyMeasurable (Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        exact pow_le_one₀ (hpb ω).1 (hpb ω).2)
    have haL := hai.mul_bdd hL.aestronglyMeasurable hbL
    have he : (fun ω => (c ω * (L ω - p ω)) ^ 2) =
        (fun ω => a ω * L ω) + b := by
      funext ω
      rcases hLb ω with h | h <;> simp [a, b, h] <;> ring
    rw [he]
    have hs := condExp_add haL hbi F
    have hw := weighted_bernoulli F μ hF L p a hL hLb ha.stronglyMeasurable hai hmean
    filter_upwards [hs, hw] with ω hs hw
    simp only [hs, Pi.add_apply, hw, condExp_of_stronglyMeasurable hF hb.stronglyMeasurable hbi]
    dsimp [a, b]
    ring

end Moments

/-- Partial sums of integrable adapted centred increments form a martingale. -/
theorem partial_sum_martingale {Ω : Type*} [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (F : Filtration ℕ mΩ)
    (D : ℕ → Ω → ℝ) (hD : ∀ i, StronglyMeasurable[F (i + 1)] (D i))
    (hi : ∀ i, Integrable (D i) μ) (hzero : ∀ i, μ[D i | F i] =ᵐ[μ] 0) :
    Martingale (fun n ω => ∑ i ∈ range n, D i ω) F μ := by
  have ha : StronglyAdapted F (fun n ω => ∑ i ∈ range n, D i ω) := by
    intro n
    exact Finset.stronglyMeasurable_fun_sum _ fun i hi =>
      (hD i).mono (F.mono (by have := mem_range.mp hi; omega))
  apply martingale_of_condExp_sub_eq_zero_nat ha
    (fun n => integrable_finsetSum _ fun i _ => hi i)
  intro n
  have he : (fun ω => ∑ i ∈ range (n + 1), D i ω) -
      (fun ω => ∑ i ∈ range n, D i ω) = D n := by
    funext ω
    simp [sum_range_succ]
  rw [he]
  exact hzero n

/-- A1--A3 plus square-integrable weights give the martingale and the
predictable sum of conditional squared increments. -/
theorem pmrt_martingale {Ω : Type*} [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (F : Filtration ℕ mΩ)
    (L p s w : ℕ → Ω → ℝ)
    (hL : ∀ i, Measurable[F (i + 1)] (L i))
    (hLb : ∀ i ω, L i ω = 0 ∨ L i ω = 1)
    (hp : ∀ i, Measurable[F i] (p i)) (hpb : ∀ i ω, p i ω ∈ Set.Icc (0 : ℝ) 1)
    (hs : ∀ i, Measurable[F i] (s i)) (hsq : ∀ i ω, s i ω ^ 2 = 1)
    (hw : ∀ i, Measurable[F i] (w i)) (hw2 : ∀ i, MemLp (w i) 2 μ)
    (hmean : ∀ i, μ[L i | F i] =ᵐ[μ] p i) :
    let D := fun i ω => (s i ω * w i ω) * (L i ω - p i ω)
    Martingale (fun n ω => ∑ i ∈ range n, D i ω) F μ ∧
    (∀ n, (fun ω => ∑ i ∈ range n, μ[(fun ω => D i ω ^ 2) | F i] ω) =ᵐ[μ]
      (fun ω => ∑ i ∈ range n, p i ω * (1 - p i ω) * w i ω ^ 2)) := by
  intro D
  have hsm : ∀ i ω, |s i ω| = 1 := fun i ω => by
    have := hsq i ω
    nlinarith [sq_abs (s i ω), abs_nonneg (s i ω)]
  have hc : ∀ i, Measurable[F i] (fun ω => s i ω * w i ω) := by
    intro i
    let : MeasurableSpace Ω := F i
    exact (hs i).mul (hw i)
  have hci : ∀ i, Integrable (fun ω => s i ω * w i ω) μ := fun i =>
    (hw2 i).integrable (by norm_num) |>.bdd_mul (c := 1)
      ((hs i).mono (F.le _) le_rfl).aestronglyMeasurable
      (Eventually.of_forall fun ω => by simp [Real.norm_eq_abs, hsm i ω])
  have hci2 : ∀ i, Integrable (fun ω => (s i ω * w i ω) ^ 2) μ := by
    intro i
    simpa only [mul_pow, hsq, one_mul] using (hw2 i).integrable_sq
  have hm := fun i => bernoulli_increment_moments (F i) μ (F.le _) (L i) (p i)
    (fun ω => s i ω * w i ω) ((hL i).mono (F.le _) le_rfl) (hLb i)
    (hp i) (hpb i) (hc i) (hci i) (hci2 i) (hmean i)
  have hDm : ∀ i, Measurable[F (i + 1)] (D i) := by
    intro i
    let : MeasurableSpace Ω := F (i + 1)
    exact ((hc i).mono (F.mono (Nat.le_succ i)) le_rfl).mul
      ((hL i).sub ((hp i).mono (F.mono (Nat.le_succ i)) le_rfl))
  have hDi : ∀ i, Integrable (D i) μ := by
    intro i
    apply (hci i).mul_bdd (c := 1)
      (((hL i).mono (F.le _) le_rfl).sub ((hp i).mono (F.le _) le_rfl)).aestronglyMeasurable
    apply Eventually.of_forall
    intro ω
    rw [Real.norm_eq_abs, abs_le]
    simp only [Pi.sub_apply]
    rcases hLb i ω with h | h <;> rw [h] <;> constructor <;>
      linarith [(hpb i ω).1, (hpb i ω).2]
  constructor
  · exact partial_sum_martingale μ F D (fun i => (hDm i).stronglyMeasurable) hDi (fun i => (hm i).1)
  · intro n
    have hall := ae_all_iff.mpr (fun i => (hm i).2)
    filter_upwards [hall] with ω hω
    apply sum_congr rfl
    intro i _
    rw [hω i]
    rw [mul_pow, hsq, one_mul]
    ring

/-- A positive probability limit makes the zero-variance event negligible. -/
theorem zero_event_vanishes {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (V : ℕ → Ω → ℝ) (η : Ω → ℝ)
    (hV : ∀ n, AEMeasurable (V n) μ) (hη : AEMeasurable η μ)
    (hpos : ∀ᵐ ω ∂μ, 0 < η ω)
    (hlim : TendstoInMeasure μ V atTop η) :
    Tendsto (fun n => μ {ω | V n ω = 0}) atTop (𝓝 0) := by
  have hw := hlim.tendstoInDistribution hV
  have hz : μ.map η ({0} : Set ℝ) = 0 := by
    rw [Measure.map_apply_of_aemeasurable hη (measurableSet_singleton _)]
    apply measure_eq_zero_iff_ae_notMem.mpr
    filter_upwards [hpos] with ω hω
    simpa using ne_of_gt hω
  have hf : μ.map η (frontier ({0} : Set ℝ)) = 0 :=
    measure_mono_null (frontier_subset_iff_isClosed.mpr isClosed_singleton) hz
  have h := ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto' hw.tendsto hf
  change Tendsto (fun n => μ.map (V n) {0}) atTop (𝓝 (μ.map η {0})) at h
  have he : ∀ n, V n ⁻¹' {0} = {ω | V n ω = 0} := by intro n; ext ω; simp
  simpa only [Measure.map_apply_of_aemeasurable (hV _) (measurableSet_singleton _), he, hz] using h

/-- At each fixed data set, independent centred redraw terms standardize to
mean zero and variance one whenever their total variance is positive. -/
theorem standardized_redraw_moments {Ω I : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (units : Finset I) (X : I → Ω → ℝ)
    (hX : ∀ i ∈ units, MemLp (X i) 2 μ)
    (hind : Set.Pairwise (units : Set I) fun i j => IndepFun (X i) (X j) μ)
    (hzero : ∀ i ∈ units, ∫ ω, X i ω ∂μ = 0) (σ : ℝ)
    (hσ : 0 < σ) (hvar : ∑ i ∈ units, variance (X i) μ = σ ^ 2) :
    (∫ ω, (∑ i ∈ units, X i ω) / σ ∂μ) = 0 ∧
    variance (fun ω => (∑ i ∈ units, X i ω) / σ) μ = 1 := by
  constructor
  · rw [integral_div, integral_finsetSum _ (fun i hi => (hX i hi).integrable (by norm_num))]
    rw [sum_eq_zero (fun i hi => hzero i hi), zero_div]
  · have he : (fun ω => (∑ i ∈ units, X i ω) / σ) =
        fun ω => σ⁻¹ * (∑ i ∈ units, X i) ω := by funext ω; simp [div_eq_mul_inv, mul_comm]
    rw [he, variance_const_mul, IndepFun.variance_sum hX hind, hvar]
    field_simp

noncomputable def reportedP (σ raw : ℝ) : ℝ := if σ = 0 then 1 else raw

theorem zero_variance_convention {I : Type*} (S raw : ℝ) (redraw : I → ℝ) :
    S / (0 : ℝ) = 0 ∧ (∀ i, redraw i / (0 : ℝ) = 0) ∧ reportedP 0 raw = 1 := by
  simp [reportedP]

/-- Conditional binomial masses imply unconditional mixture masses by the tower law. -/
theorem conditional_binomial_atom {Ω : Type*} (G : MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : G ≤ mΩ) (Q : Ω → ℕ) (hQ : Measurable Q) (P : Ω → ℝ) (B j : ℕ)
    (hlaw : μ[(Q ⁻¹' {j}).indicator (fun _ => (1 : ℝ)) | G] =ᵐ[μ]
      (fun ω => binomialMass B j (P ω))) :
    μ.real (Q ⁻¹' {j}) = ∫ ω, binomialMass B j (P ω) ∂μ := by
  have hi := integral_congr_ae hlaw
  rw [integral_condExp hG, integral_indicator (hQ (measurableSet_singleton _)),
    setIntegral_const, smul_eq_mul, mul_one] at hi
  exact hi

theorem rejection_probability_formula {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (Q : Ω → ℕ) (hQ : Measurable Q)
    (B : ℕ) (hbound : ∀ ω, Q ω ≤ B) (t : ℝ) :
    μ.real {ω | (1 + (Q ω : ℝ)) / (B + 1) ≤ t} =
      ∑ j ∈ range (B + 1), if (1 + (j : ℝ)) / (B + 1) ≤ t then
        μ.real (Q ⁻¹' {j}) else 0 := by
  rw [← sum_filter]
  rw [sum_measureReal_preimage_singleton _ (fun j _ => hQ (measurableSet_singleton j))]
  congr 1
  ext ω
  simp only [Set.mem_preimage, mem_coe, mem_filter, mem_range, Set.mem_ofPred_eq]
  exact ⟨fun h => ⟨Nat.lt_succ_of_le (hbound ω), h⟩, fun h => h.2⟩

/-- Part 3 on the exact finite Monte Carlo grid, conditional on a uniform
limit for the redraw-tail probability. -/
theorem conditional_binomial_rejection_limit {Ω : Type*} (G : ℕ → MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : ∀ n, G n ≤ mΩ) (Q : ℕ → Ω → ℕ) (hQ : ∀ n, Measurable (Q n))
    (P : ℕ → Ω → ℝ) (hP : ∀ n ω, P n ω ∈ Set.Icc (0 : ℝ) 1)
    (hweak : TendstoInDistribution P atTop id (fun _ => μ) uniformLaw)
    (B : ℕ) (hbound : ∀ n ω, Q n ω ≤ B)
    (hlaw : ∀ n j, j ≤ B →
      μ[(Q n ⁻¹' {j}).indicator (fun _ => (1 : ℝ)) | G n] =ᵐ[μ]
        (fun ω => binomialMass B j (P n ω)))
    (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    Tendsto (fun n => μ.real {ω | (1 + (Q n ω : ℝ)) / (B + 1) ≤ t}) atTop
      (𝓝 (gridCDF B t)) ∧ gridCDF B t ≤ t := by
  constructor
  · have he : ∀ n, μ.real {ω | (1 + (Q n ω : ℝ)) / (B + 1) ≤ t} =
        ∑ j ∈ range (B + 1), if (1 + (j : ℝ)) / (B + 1) ≤ t then
          ∫ ω, binomialMass B j (P n ω) ∂μ else 0 := by
      intro n
      rw [rejection_probability_formula μ (Q n) (hQ n) B (hbound n) t]
      apply sum_congr rfl
      intro j hj
      rw [conditional_binomial_atom (G n) μ (hG n) (Q n) (hQ n) (P n) B j
        (hlaw n j (by have := mem_range.mp hj; omega))]
    simp_rw [he]
    exact finite_redraw_limit μ P hP hweak B t ht
  · exact gridCDF_le B t ht.1

noncomputable def standardNormal : Measure ℝ := gaussianReal 0 1

instance : IsProbabilityMeasure standardNormal := by unfold standardNormal; infer_instance

instance : NullSingletonClass standardNormal := by
  change NullSingletonClass (gaussianReal 0 1)
  exact nullSingletonClass_gaussianReal (by norm_num)

noncomputable def normalCDF : ℝ → ℝ := cdf standardNormal

noncomputable def cdfDistanceTo (ν ρ : Measure ℝ) : ℝ :=
  sSup (Set.range fun x => |cdf ν x - cdf ρ x|)

noncomputable def cdfDistance (ν : Measure ℝ) : ℝ :=
  sSup (Set.range fun x => |cdf ν x - normalCDF x|)

/-- Stable convergence to a conditionally Gaussian limit: bounded measurable
test variables may depend on the original probability space. -/
def StableGaussianLimit {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (X : ℕ → Ω → ℝ) (η : Ω → ℝ) : Prop :=
  ∀ (H : Ω → ℝ), Measurable H → (∃ C : ℝ, ∀ ω, |H ω| ≤ C) →
    ∀ f : ℝ →ᵇ ℝ, Tendsto (fun n => ∫ ω, H ω * f (X n ω) ∂μ) atTop
      (𝓝 (∫ ω, ∫ z, H ω * f (η ω * z) ∂standardNormal ∂μ))

noncomputable def predictableVariance {Ω : Type*} [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) (F : Filtration ℕ mΩ) (D : ℕ → Ω → ℝ) (m : ℕ) (ω : Ω) : ℝ :=
  ∑ i ∈ range m, μ[(fun ω => D i ω ^ 2) | F i] ω

noncomputable def conditionalLindeberg {Ω : Type*} [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) (F : Filtration ℕ mΩ) (D : ℕ → Ω → ℝ)
    (m : ℕ) (scale γ : ℝ) (ω : Ω) : ℝ :=
  ∑ i ∈ range m, μ[(fun ω => if γ < |D i ω / scale| then
    (D i ω / scale) ^ 2 else 0) | F i] ω

/-- Named external theorem obligation: Hall--Heyde, Corollaries 3.1/3.2,
specialized to initial segments of one filtration and deterministic scaling.
This is a proposition supplied as a hypothesis, not an axiom declaration. -/
def HallHeyde {Ω : Type*} [mΩ : MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] : Prop :=
  ∀ (F : Filtration ℕ mΩ) (D : ℕ → Ω → ℝ) (m : ℕ → ℕ) (scale : ℕ → ℝ) (η : Ω → ℝ),
    (∀ i, StronglyMeasurable[F (i + 1)] (D i)) →
    (∀ i, MemLp (D i) 2 μ) → (∀ i, μ[D i | F i] =ᵐ[μ] 0) →
    Monotone m → Tendsto m atTop atTop →
    (∀ n, 0 < scale n) → Tendsto scale atTop atTop →
    Measurable η → (∀ᵐ ω ∂μ, 0 < η ω) →
    TendstoInMeasure μ
      (fun n ω => predictableVariance μ F D (m n) ω / scale n ^ 2)
      atTop (fun ω => η ω ^ 2) →
    (∀ γ > 0, TendstoInMeasure μ
      (fun n => conditionalLindeberg μ F D (m n) (scale n) γ) atTop (fun _ => 0)) →
    StableGaussianLimit μ (fun n ω => (∑ i ∈ range (m n), D i ω) / scale n) η ∧
    StableGaussianLimit μ (fun n ω => (∑ i ∈ range (m n), D i ω) /
      Real.sqrt (predictableVariance μ F D (m n) ω)) (fun _ => 1)

/-- Named external theorem obligation: Lindeberg--Feller (Durrett, Theorem
3.4.10), followed by Pólya (Exercise 3.2.9). Rows are independent and centred;
total variance tends to one and every Lindeberg sum tends to zero. -/
def LindebergFellerPolya {Ω : Type*} [MeasurableSpace Ω] (ν : Measure Ω)
    [IsProbabilityMeasure ν] : Prop :=
  ∀ (X : ℕ → ℕ → Ω → ℝ) (m : ℕ → ℕ),
    (∀ n i, MemLp (X n i) 2 ν) →
    (∀ n, iIndepFun (X n) ν) →
    (∀ n i, ∫ ω, X n i ω ∂ν = 0) →
    Tendsto (fun n => ∑ i ∈ range (m n), variance (X n i) ν) atTop (𝓝 1) →
    (∀ γ > 0, Tendsto (fun n => ∑ i ∈ range (m n),
      ∫ ω, if γ < |X n i ω| then X n i ω ^ 2 else 0 ∂ν) atTop (𝓝 0)) →
    TendstoInDistribution (fun n ω => ∑ i ∈ range (m n), X n i ω)
      atTop id (fun _ => ν) standardNormal ∧
    Tendsto (fun n => cdfDistance (ν.map (fun ω => ∑ i ∈ range (m n), X n i ω))) atTop (𝓝 0)

/-- Pólya's theorem in its general continuous-limit form (Durrett,
Exercise 3.2.9), used after a continuous score transformation. -/
def Polya : Prop :=
  ∀ (ν : ℕ → ProbabilityMeasure ℝ) (ρ : ProbabilityMeasure ℝ),
    Continuous (cdf (ρ : Measure ℝ)) → Tendsto ν atTop (𝓝 ρ) →
    Tendsto (fun n => cdfDistanceTo (ν n : Measure ℝ) (ρ : Measure ℝ)) atTop (𝓝 0)

theorem StableGaussianLimit.tendstoInDistribution {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ℕ → Ω → ℝ)
    (hX : ∀ n, AEMeasurable (X n) μ)
    (h : StableGaussianLimit μ X (fun _ => 1)) :
    TendstoInDistribution X atTop id (fun _ => μ) standardNormal := by
  apply (tendstoInDistribution_iff_forall_integral_rclike_tendsto ℝ hX measurable_id.aemeasurable).mpr
  intro f
  have hf := h (fun _ => 1) measurable_const ⟨1, by simp⟩ f
  simpa using hf

theorem conservative_pvalue_limsup {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (raw reported : ℕ → Ω → ℝ)
    (hconservative : ∀ n ω, raw n ω ≤ reported n ω) (t c : ℝ)
    (hlim : Tendsto (fun n => μ.real {ω | raw n ω ≤ t}) atTop (𝓝 c)) :
    Filter.limsup (fun n => μ.real {ω | reported n ω ≤ t}) atTop ≤ c := by
  rw [← hlim.limsup_eq]
  apply limsup_le_limsup
  · exact Eventually.of_forall fun n => measureReal_mono
      (fun ω hω => le_trans (hconservative n ω) hω) (measure_ne_top _ _)
  · exact isCoboundedUnder_le_of_le atTop fun _ => measureReal_nonneg
  · refine ⟨1, ?_⟩
    change ∀ᶠ n in atTop, μ.real {ω | raw n ω ≤ t} ≤ 1
    exact Eventually.of_forall fun _ => measureReal_le_one

/-- Conditional Lindeberg--Feller is applied pathwise on subsequences; the
subsequence criterion then gives uniform CDF convergence in probability. -/
theorem redraw_cdf_limit {Ω Ω' : Type*} [MeasurableSpace Ω] [MeasurableSpace Ω']
    (μ : Measure Ω) [IsProbabilityMeasure μ] (ν : Ω → Measure Ω')
    [∀ ω, IsProbabilityMeasure (ν ω)]
    (lf : ∀ ω, LindebergFellerPolya (ν ω))
    (X : ℕ → ℕ → Ω → Ω' → ℝ) (m : ℕ → ℕ) (bound : ℕ → Ω → ℝ)
    (hX : ∀ n i ω, MemLp (X n i ω) 2 (ν ω))
    (hind : ∀ n ω, iIndepFun (fun i => X n i ω) (ν ω))
    (hzero : ∀ n i ω, ∫ x, X n i ω x ∂ν ω = 0)
    (hbound : ∀ n i ω x, |X n i ω x| ≤ bound n ω)
    (hb : TendstoInMeasure μ bound atTop (fun _ => 0))
    (hv : TendstoInMeasure μ
      (fun n ω => ∑ i ∈ range (m n), variance (X n i ω) (ν ω)) atTop (fun _ => 1))
    (hmeas : ∀ n, AEStronglyMeasurable
      (fun ω => cdfDistance ((ν ω).map (fun x => ∑ i ∈ range (m n), X n i ω x))) μ) :
    TendstoInMeasure μ
      (fun n ω => cdfDistance ((ν ω).map (fun x => ∑ i ∈ range (m n), X n i ω x)))
      atTop (fun _ => 0) := by
  apply (exists_seq_tendstoInMeasure_atTop_iff hmeas).mpr
  intro ns hns
  obtain ⟨bs, hbs, hbpath⟩ := (hb.comp hns.tendsto_atTop).exists_seq_tendsto_ae
  obtain ⟨cs, hcs, hvpath⟩ :=
    (hv.comp (hns.tendsto_atTop.comp hbs.tendsto_atTop)).exists_seq_tendsto_ae
  refine ⟨bs ∘ cs, hbs.comp hcs, ?_⟩
  filter_upwards [hbpath, hvpath] with ω hbω hvω
  have hbω' := hbω.comp hcs.tendsto_atTop
  apply And.right (lf ω (fun n i => X (ns (bs (cs n))) i ω) (fun n => m (ns (bs (cs n))))
    (fun n i => hX _ _ _) (fun n => hind _ _) (fun n i => hzero _ _ _) hvω ?_)
  intro γ hγ
  have he : ∀ᶠ n in atTop, ∑ i ∈ range (m (ns (bs (cs n)))),
      ∫ x, if γ < |X (ns (bs (cs n))) i ω x| then X (ns (bs (cs n))) i ω x ^ 2 else 0
        ∂ν ω = 0 := by
    filter_upwards [hbω'.eventually (gt_mem_nhds hγ)] with n hn
    apply sum_eq_zero
    intro i _
    have hh : ∀ x, ¬ γ < |X (ns (bs (cs n))) i ω x| := fun x =>
      not_lt_of_ge (le_trans (hbound _ _ _ _) hn.le)
    simp only [hh, ite_false, integral_zero]
  apply tendsto_const_nhds.congr'
  filter_upwards [he] with n hn
  exact hn.symm

lemma condExp_zero_on {Ω : Type*} (G : MeasurableSpace Ω) [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) (f : Ω → ℝ) (hf : Integrable f μ) (E : Set Ω)
    (hE : MeasurableSet[G] E) (hzero : ∀ ω ∈ E, f ω = 0) :
    ∀ᵐ ω ∂μ, ω ∈ E → μ[f | G] ω = 0 := by
  have he : E.indicator f = 0 := by
    funext ω
    by_cases h : ω ∈ E <;> simp [h, hzero]
  have h := condExp_indicator hf hE
  rw [he, condExp_zero] at h
  filter_upwards [h] with ω hω hmem
  simpa [hmem] using hω.symm

/-- Predictability makes a truncated conditional second moment zero wherever
the predictable bound is below the truncation threshold. -/
lemma conditional_truncation_zero {Ω : Type*} (G : MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) (D w : Ω → ℝ)
    (hD : MemLp D 2 μ) (hDm : Measurable D) (hw : Measurable[G] w)
    (hbound : ∀ ω, |D ω| ≤ |w ω|) (scale γ : ℝ) (hs : 0 < scale) :
    ∀ᵐ ω ∂μ, |w ω| / scale ≤ γ →
      μ[(fun ω => if γ < |D ω / scale| then (D ω / scale) ^ 2 else 0) | G] ω = 0 := by
  let f : Ω → ℝ := fun ω => if γ < |D ω / scale| then (D ω / scale) ^ 2 else 0
  have hi : Integrable f μ := by
    have hbase : Integrable (fun ω => (D ω / scale) ^ 2) μ := by
      simpa only [div_eq_mul_inv] using (hD.mul_const scale⁻¹).integrable_sq
    have hm : Measurable (fun ω => |D ω / scale|) := by
      simpa only [Real.norm_eq_abs] using (hDm.div_const scale).norm
    exact hbase.mono' ((Measurable.ite (measurableSet_lt measurable_const hm)
      ((hDm.div_const scale).pow_const 2) measurable_const).aestronglyMeasurable)
      (Eventually.of_forall fun ω => by
        unfold f
        split_ifs
        · rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg (D ω / scale))]
        · simpa only [norm_zero] using sq_nonneg (D ω / scale))
  have hE : MeasurableSet[G] {ω | |w ω| / scale ≤ γ} := by
    let : MeasurableSpace Ω := G
    apply measurableSet_le _ measurable_const
    simpa only [Real.norm_eq_abs] using hw.norm.div_const scale
  apply condExp_zero_on G μ f hi _ hE
  intro ω hω
  have h : |D ω / scale| ≤ γ := by
    rw [abs_div, abs_of_pos hs]
    exact le_trans (div_le_div_of_nonneg_right (hbound ω) hs.le) hω
  simp [f, not_lt_of_ge h]

/-- Functions supported only where a bound exceeds a fixed positive threshold
vanish in probability when that bound converges to zero. -/
theorem vanishes_off_bad_event {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (f bound : ℕ → Ω → ℝ)
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) (γ : ℝ) (hγ : 0 < γ)
    (hsupport : ∀ n, ∀ᵐ ω ∂μ, bound n ω ≤ γ → f n ω = 0)
    (hb : TendstoInMeasure μ bound atTop (fun _ => 0)) :
    TendstoInMeasure μ f atTop (fun _ => 0) := by
  apply (exists_seq_tendstoInMeasure_atTop_iff hf).mpr
  intro ns hns
  obtain ⟨ks, hks, hpath⟩ := (hb.comp hns.tendsto_atTop).exists_seq_tendsto_ae
  refine ⟨ks, hks, ?_⟩
  filter_upwards [hpath, ae_all_iff.mpr hsupport] with ω hp hs
  apply tendsto_const_nhds.congr'
  filter_upwards [hp.eventually (gt_mem_nhds hγ)] with n hn
  exact (hs _ hn.le).symm

theorem conditional_lindeberg_from_max {Ω : Type*} [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (F : Filtration ℕ mΩ)
    (D w : ℕ → Ω → ℝ) (hD : ∀ i, MemLp (D i) 2 μ) (hDm : ∀ i, Measurable (D i))
    (hw : ∀ i, Measurable[F i] (w i)) (hDw : ∀ i ω, |D i ω| ≤ |w i ω|)
    (m : ℕ → ℕ) (scale : ℕ → ℝ) (hs : ∀ n, 0 < scale n)
    (bound : ℕ → Ω → ℝ)
    (hmax : ∀ n i, i < m n → ∀ ω, |w i ω| / scale n ≤ bound n ω)
    (hb : TendstoInMeasure μ bound atTop (fun _ => 0)) (γ : ℝ) (hγ : 0 < γ) :
    TendstoInMeasure μ
      (fun n => conditionalLindeberg μ F D (m n) (scale n) γ) atTop (fun _ => 0) := by
  apply vanishes_off_bad_event μ _ bound _ γ hγ _ hb
  · intro n
    exact (Finset.stronglyMeasurable_fun_sum _ fun i _ =>
      (stronglyMeasurable_condExp.mono (F.le i))).aestronglyMeasurable
  · intro n
    have hall := ae_all_iff.mpr fun i =>
      conditional_truncation_zero (F i) μ (D i) (w i) (hD i) (hDm i) (hw i) (hDw i) (scale n) γ (hs n)
    filter_upwards [hall] with ω hω hgood
    apply sum_eq_zero
    intro i hi
    exact hω i (le_trans (hmax n i (mem_range.mp hi) ω) hgood)

/-- A4's maximal-weight condition supplies conditional Lindeberg; the named
Hall--Heyde obligation then yields the self-normalized normal limit. -/
theorem self_normalized_clt {Ω : Type*} [mΩ : MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (hh : HallHeyde μ)
    (F : Filtration ℕ mΩ) (D w : ℕ → Ω → ℝ)
    (hadapt : ∀ i, StronglyMeasurable[F (i + 1)] (D i))
    (hD : ∀ i, MemLp (D i) 2 μ) (hzero : ∀ i, μ[D i | F i] =ᵐ[μ] 0)
    (hw : ∀ i, Measurable[F i] (w i)) (hDw : ∀ i ω, |D i ω| ≤ |w i ω|)
    (m : ℕ → ℕ) (hm : Monotone m) (hmTop : Tendsto m atTop atTop)
    (scale : ℕ → ℝ) (hs : ∀ n, 0 < scale n) (hsTop : Tendsto scale atTop atTop)
    (η : Ω → ℝ) (hη : Measurable η) (hηpos : ∀ᵐ ω ∂μ, 0 < η ω)
    (hv : TendstoInMeasure μ
      (fun n ω => predictableVariance μ F D (m n) ω / scale n ^ 2)
      atTop (fun ω => η ω ^ 2))
    (bound : ℕ → Ω → ℝ)
    (hmax : ∀ n i, i < m n → ∀ ω, |w i ω| / scale n ≤ bound n ω)
    (hb : TendstoInMeasure μ bound atTop (fun _ => 0)) :
    TendstoInDistribution
      (fun n ω => (∑ i ∈ range (m n), D i ω) /
        Real.sqrt (predictableVariance μ F D (m n) ω))
      atTop id (fun _ => μ) standardNormal := by
  have hstable := (hh F D m scale η hadapt hD hzero hm hmTop hs hsTop hη hηpos hv
    (fun γ hγ => conditional_lindeberg_from_max μ F D w hD
      (fun i => ((hadapt i).mono (F.le _)).measurable) hw hDw m scale hs bound hmax hb γ hγ)).2
  apply StableGaussianLimit.tendstoInDistribution μ _ _ hstable
  intro n
  have hsum : AEStronglyMeasurable (fun ω => ∑ i ∈ range (m n), D i ω) μ :=
    Finset.aestronglyMeasurable_fun_sum _ fun i _ => (hD i).aestronglyMeasurable
  have hvar : Measurable (predictableVariance μ F D (m n)) := by
    unfold predictableVariance
    exact Finset.measurable_sum _ fun i _ =>
      (stronglyMeasurable_condExp.mono (F.le i)).measurable
  exact hsum.aemeasurable.div (Real.continuous_sqrt.measurable.comp hvar).aemeasurable

lemma continuous_cdf_of_no_atoms (ν : Measure ℝ) [IsProbabilityMeasure ν]
    [NullSingletonClass ν] : Continuous (cdf ν) := by
  have he : (fun b : ℝ => ∫ _x in Set.Iic b, (1 : ℝ) ∂ν) = cdf ν := by
    funext b
    simp [cdf_eq_real]
  rw [← he]
  apply continuous_iff_continuousAt.mpr
  intro x
  exact ((integrable_const (1 : ℝ)).integrableOn.continuousOn_Iic_primitive_Iic
    (a₀ := x + 1)).continuousAt (Iic_mem_nhds (by linarith))

lemma cdf_range_point (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hc : Continuous (cdf ν)) (t : ℝ) (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    ∃ x, cdf ν x = t := by
  obtain ⟨a, ha⟩ := ((tendsto_cdf_atBot ν).eventually (gt_mem_nhds ht.1)).exists
  obtain ⟨b, hb⟩ := ((tendsto_cdf_atTop ν).eventually (lt_mem_nhds ht.2)).exists
  have hab : a ≤ b := by
    by_contra h
    have hh := monotone_cdf ν (le_of_lt (lt_of_not_ge h))
    linarith
  obtain ⟨x, _, hx⟩ := intermediate_value_Icc hab hc.continuousOn ⟨ha.le, hb.le⟩
  exact ⟨x, hx⟩

lemma uniformLaw_Iic (t : ℝ) : uniformLaw (Set.Iic t) = ENNReal.ofReal (max 0 (min t 1)) := by
  rw [uniformLaw, restrict_Ioo_eq_restrict_Ioc, Measure.restrict_apply measurableSet_Iic]
  have he : Set.Iic t ∩ Set.Ioc (0 : ℝ) 1 = Set.Ioc 0 (min t 1) := by ext x; simp; tauto
  rw [he, Real.volume_Ioc]
  simp [ENNReal.ofReal_max]

/-- Probability integral transform for any continuous CDF, including those
with flat portions. -/
theorem cdf_uniform (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hc : Continuous (cdf ν)) :
    ν.map (cdf ν) = uniformLaw := by
  apply Measure.ext_of_Iic
  intro t
  rw [Measure.map_apply hc.measurable measurableSet_Iic, uniformLaw_Iic]
  by_cases ht0 : t < 0
  · have he : cdf ν ⁻¹' Set.Iic t = ∅ := by
      ext x
      simp only [Set.mem_preimage, Set.mem_Iic, Set.mem_empty_iff_false, iff_false]
      exact not_le_of_gt (lt_of_lt_of_le ht0 (cdf_nonneg ν x))
    rw [he, measure_empty]
    simp [max_eq_left (le_trans (min_le_left _ _) ht0.le)]
  · by_cases ht1 : 1 ≤ t
    · have he : cdf ν ⁻¹' Set.Iic t = Set.univ := by
        ext x
        simp only [Set.mem_preimage, Set.mem_Iic, Set.mem_univ, iff_true]
        exact le_trans (cdf_le_one ν x) ht1
      rw [he, measure_univ]
      simp [min_eq_right ht1]
    · have ht : t ∈ Set.Ico (0 : ℝ) 1 := ⟨le_of_not_gt ht0, lt_of_not_ge ht1⟩
      rw [min_eq_left ht.2.le, max_eq_right ht.1]
      apply (ENNReal.toReal_eq_toReal_iff' (measure_ne_top _ _) ENNReal.ofReal_ne_top).mp
      rw [ENNReal.toReal_ofReal ht.1]
      change ν.real (cdf ν ⁻¹' Set.Iic t) = t
      apply le_antisymm
      · apply le_of_forall_pos_le_add
        intro ε hε
        by_cases he : t + ε < 1
        · obtain ⟨b, hb⟩ := cdf_range_point ν hc (t + ε) ⟨by linarith, he⟩
          have hsub : cdf ν ⁻¹' Set.Iic t ⊆ Set.Iic b := by
            intro x hx
            by_contra hn
            have hh := monotone_cdf ν (le_of_lt (lt_of_not_ge hn))
            change cdf ν x ≤ t at hx
            linarith
          calc
            _ ≤ ν.real (Set.Iic b) := measureReal_mono hsub (measure_ne_top _ _)
            _ = _ := by rw [← cdf_eq_real, hb]
        · exact le_trans measureReal_le_one (le_of_not_gt he)
      · by_cases hz : t = 0
        · simp [hz]
        · obtain ⟨a, ha⟩ := cdf_range_point ν hc t ⟨lt_of_le_of_ne ht.1 (Ne.symm hz), ht.2⟩
          calc
            t = ν.real (Set.Iic a) := by rw [← cdf_eq_real, ha]
            _ ≤ _ := measureReal_mono
              (fun x hx => by change cdf ν x ≤ t; rw [← ha]; exact monotone_cdf ν hx)
              (measure_ne_top _ _)

theorem tendstoInMeasure_of_abs_le {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (f e : ℕ → Ω → ℝ)
    (hf : ∀ n, AEStronglyMeasurable (f n) μ)
    (hle : ∀ n, ∀ᵐ ω ∂μ, |f n ω| ≤ |e n ω|)
    (he : TendstoInMeasure μ e atTop (fun _ => 0)) :
    TendstoInMeasure μ f atTop (fun _ => 0) := by
  apply (exists_seq_tendstoInMeasure_atTop_iff hf).mpr
  intro ns hns
  obtain ⟨ks, hks, hp⟩ := (he.comp hns.tendsto_atTop).exists_seq_tendsto_ae
  refine ⟨ks, hks, ?_⟩
  filter_upwards [hp, ae_all_iff.mpr hle] with ω hp hle
  apply squeeze_zero_norm (fun n => by simpa only [Real.norm_eq_abs] using hle (ns (ks n)))
  simpa using hp.abs

/-- A uniform conditional-CDF approximation and a continuous limiting score
law imply a uniform limiting redraw-tail probability. -/
theorem redraw_tail_uniform {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (ρ : Measure ℝ) [IsProbabilityMeasure ρ]
    (hc : Continuous (cdf ρ)) (score P error : ℕ → Ω → ℝ)
    (hscore : TendstoInDistribution score atTop id (fun _ => μ) ρ)
    (hP : ∀ n, AEMeasurable (P n) μ)
    (hclose : ∀ n, ∀ᵐ ω ∂μ, |P n ω - cdf ρ (score n ω)| ≤ |error n ω|)
    (he : TendstoInMeasure μ error atTop (fun _ => 0)) :
    TendstoInDistribution P atTop id (fun _ => μ) uniformLaw := by
  have hbase := hscore.continuous_comp hc
  have hd : TendstoInMeasure μ
      (fun n ω => P n ω - cdf ρ (score n ω)) atTop (fun _ => 0) := by
    apply tendstoInMeasure_of_abs_le μ _ error _ hclose he
    intro n
    exact ((hP n).sub (hc.measurable.comp_aemeasurable (hscore.forall_aemeasurable n))).aestronglyMeasurable
  have h := tendstoInDistribution_of_tendstoInMeasure_sub P (cdf ρ ∘ id) hbase hd hP
  refine ⟨hP, measurable_id.aemeasurable, ?_⟩
  simpa only [Function.comp_id, cdf_uniform ρ hc, Measure.map_id] using h.tendsto

lemma standardNormal_neg : standardNormal.map (fun x => -x) = standardNormal := by
  simp [standardNormal, gaussianReal_map_neg]

theorem stable_fixed_orientation {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ℕ → Ω → ℝ)
    (hX : ∀ n, AEMeasurable (X n) μ)
    (hstable : StableGaussianLimit μ X (fun _ => 1))
    (o : Ω → ℝ) (ho : Measurable o) (hos : ∀ ω, o ω = 1 ∨ o ω = -1) :
    TendstoInDistribution (fun n ω => o ω * X n ω) atTop id (fun _ => μ) standardNormal := by
  apply (tendstoInDistribution_iff_forall_integral_rclike_tendsto ℝ
    (fun n => ho.aemeasurable.mul (hX n)) measurable_id.aemeasurable).mpr
  intro f
  let E : Set Ω := {ω | o ω = 1}
  let H : Ω → ℝ := E.indicator (fun _ => 1)
  let K : Ω → ℝ := Eᶜ.indicator (fun _ => 1)
  have hE : MeasurableSet E := measurableSet_eq_fun ho measurable_const
  have hH : Measurable H := measurable_const.indicator hE
  have hK : Measurable K := measurable_const.indicator hE.compl
  have hHi : Integrable H μ := (integrable_const (1 : ℝ)).indicator hE
  have hKi : Integrable K μ := (integrable_const (1 : ℝ)).indicator hE.compl
  have hHK : ∀ ω, H ω + K ω = 1 := by intro ω; by_cases h : ω ∈ E <;> simp [H, K, h]
  have hHb : ∃ C : ℝ, ∀ ω, |H ω| ≤ C := ⟨1, fun ω => by
    by_cases h : ω ∈ E <;> simp [H, h]⟩
  have hKb : ∃ C : ℝ, ∀ ω, |K ω| ≤ C := ⟨1, fun ω => by
    by_cases h : ω ∈ E <;> simp [K, h]⟩
  let g : ℝ →ᵇ ℝ := f.compContinuous ⟨Neg.neg, continuous_neg⟩
  have hg : (∫ z, g z ∂standardNormal) = ∫ z, f z ∂standardNormal := by
    have h := integral_map (μ := standardNormal) measurable_neg.aemeasurable
      f.continuous.measurable.aestronglyMeasurable
    rw [standardNormal_neg] at h
    exact h.symm
  have hp := hstable H hH hHb f
  have hm := hstable K hK hKb g
  simp only [one_mul, integral_const_mul, integral_mul_const, hg] at hp hm
  have hi : (∫ ω, H ω ∂μ) + (∫ ω, K ω ∂μ) = 1 := by
    rw [← integral_add hHi hKi]
    simp [hHK]
  have hlim := hp.add hm
  have hr : (∫ ω, H ω ∂μ) * (∫ z, f z ∂standardNormal) +
      (∫ ω, K ω ∂μ) * (∫ z, f z ∂standardNormal) = ∫ z, f z ∂standardNormal := by
    rw [← add_mul, hi, one_mul]
  rw [hr] at hlim
  have he : ∀ n, (∫ ω, H ω * f (X n ω) ∂μ) + (∫ ω, K ω * g (X n ω) ∂μ) =
      ∫ ω, f (o ω * X n ω) ∂μ := by
    intro n
    have hfi : Integrable (fun ω => H ω * f (X n ω)) μ := by
      convert! hHi.mul_bdd (f.continuous.measurable.comp_aemeasurable (hX n)).aestronglyMeasurable
        (Eventually.of_forall fun ω => f.norm_coe_le_norm _) using 1
    have hgi : Integrable (fun ω => K ω * g (X n ω)) μ := by
      convert! hKi.mul_bdd (g.continuous.measurable.comp_aemeasurable (hX n)).aestronglyMeasurable
        (Eventually.of_forall fun ω => g.norm_coe_le_norm _) using 1
    rw [← integral_add hfi hgi]
    apply integral_congr_ae
    apply Eventually.of_forall
    intro ω
    rcases hos ω with h | h
    · have he : ω ∈ E := h
      simp [H, K, he, h]
    · have he : ω ∉ E := by change o ω ≠ 1; rw [h]; norm_num
      simp [H, K, he, h, g]
  simpa only [he, id_eq, Pi.mul_apply] using hlim

lemma lower_absolute_atomless (ν : Measure ℝ) [NullSingletonClass ν] :
    NullSingletonClass (ν.map (fun x : ℝ => -‖x‖)) := by
  constructor
  intro a
  rw [Measure.map_apply (by fun_prop) (measurableSet_singleton _)]
  apply measure_mono_null (t := ({a, -a} : Set ℝ))
  · intro x hx
    change -‖x‖ = a at hx
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
    rcases le_total 0 x with h | h
    · rw [Real.norm_eq_abs, abs_of_nonneg h] at hx
      right; linarith
    · rw [Real.norm_eq_abs, abs_of_nonpos h] at hx
      left; linarith
  · exact (Set.toFinite _).measure_zero ν

lemma cdf_error_bound (ν ρ : Measure ℝ) (x : ℝ) :
    |cdf ν x - cdf ρ x| ≤ cdfDistanceTo ν ρ := by
  apply le_csSup
  · refine ⟨1, ?_⟩
    rintro y ⟨z, rfl⟩
    apply abs_le.mpr
    constructor <;> linarith [cdf_nonneg ν z, cdf_le_one ν z, cdf_nonneg ρ z, cdf_le_one ρ z]
  · exact ⟨x, rfl⟩

/-- Part 3 with the probability-integral-transform and binomial-mixture steps
proved. Scores are negated so inclusive exceedances are lower CDF events. -/
theorem asymptotic_rejection_bound {Ω : Type*} (G : ℕ → MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : ∀ n, G n ≤ mΩ) (ρ : Measure ℝ) [IsProbabilityMeasure ρ]
    (hc : Continuous (cdf ρ)) (score : ℕ → Ω → ℝ)
    (hscore : TendstoInDistribution score atTop id (fun _ => μ) ρ)
    (redrawLaw : ℕ → Ω → Measure ℝ)
    (herror : TendstoInMeasure μ (fun n ω => cdfDistanceTo (redrawLaw n ω) ρ) atTop (fun _ => 0))
    (hP : ∀ n, AEMeasurable (fun ω => cdf (redrawLaw n ω) (score n ω)) μ)
    (Q : ℕ → Ω → ℕ) (hQ : ∀ n, Measurable (Q n)) (B : ℕ)
    (hbound : ∀ n ω, Q n ω ≤ B)
    (hlaw : ∀ n j, j ≤ B →
      μ[(Q n ⁻¹' {j}).indicator (fun _ => (1 : ℝ)) | G n] =ᵐ[μ]
        (fun ω => binomialMass B j (cdf (redrawLaw n ω) (score n ω))))
    (reported : ℕ → Ω → ℝ)
    (hconservative : ∀ n ω, (1 + (Q n ω : ℝ)) / (B + 1) ≤ reported n ω)
    (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    Filter.limsup (fun n => μ.real {ω | reported n ω ≤ t}) atTop ≤ gridCDF B t ∧
      gridCDF B t ≤ t := by
  let P : ℕ → Ω → ℝ := fun n ω => cdf (redrawLaw n ω) (score n ω)
  have hweak := redraw_tail_uniform μ ρ hc score P _ hscore hP
    (fun n => Eventually.of_forall fun ω =>
      le_trans (cdf_error_bound _ _ _) (le_abs_self _)) herror
  have hlim := conditional_binomial_rejection_limit G μ hG Q hQ P
    (fun n ω => ⟨cdf_nonneg _ _, cdf_le_one _ _⟩) hweak B hbound hlaw t ht
  exact ⟨conservative_pvalue_limsup μ _ reported hconservative t _ hlim.1, hlim.2⟩

theorem bernoulli_redraw_moments {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (L : Ω → ℝ)
    (hL : Measurable L) (hLb : ∀ ω, L ω = 0 ∨ L ω = 1)
    (p c : ℝ) (hp : p ∈ Set.Icc (0 : ℝ) 1) (hmean : ∫ ω, L ω ∂μ = p) :
    (∫ ω, c * (L ω - p) ∂μ) = 0 ∧
      variance (fun ω => c * (L ω - p)) μ = c ^ 2 * p * (1 - p) := by
  have hm := bernoulli_increment_moments ⊥ μ bot_le L (fun _ => p) (fun _ => c)
    hL hLb measurable_const (fun _ => hp) measurable_const (integrable_const c)
    (integrable_const (c ^ 2)) (by simp [condExp_bot, hmean])
  have hzero : (∫ ω, c * (L ω - p) ∂μ) = 0 := by
    have hi := integral_congr_ae hm.1
    simpa only [integral_condExp bot_le, Pi.zero_apply, integral_zero] using hi
  refine ⟨hzero, ?_⟩
  rw [variance_eq_integral (by fun_prop), hzero]
  simp only [sub_zero]
  have hi := integral_congr_ae hm.2
  simpa only [integral_condExp bot_le, integral_const, probReal_univ, one_smul] using hi

theorem redraw_score_cdf_limit {Ω Ω' : Type*} [MeasurableSpace Ω] [MeasurableSpace Ω']
    (μ : Measure Ω) [IsProbabilityMeasure μ] (ν : Ω → Measure Ω')
    [∀ ω, IsProbabilityMeasure (ν ω)]
    (lf : ∀ ω, LindebergFellerPolya (ν ω)) (polya : Polya)
    (X : ℕ → ℕ → Ω → Ω' → ℝ) (m : ℕ → ℕ) (bound : ℕ → Ω → ℝ)
    (hX : ∀ n i ω, MemLp (X n i ω) 2 (ν ω))
    (hind : ∀ n ω, iIndepFun (fun i => X n i ω) (ν ω))
    (hzero : ∀ n i ω, ∫ x, X n i ω x ∂ν ω = 0)
    (hbound : ∀ n i ω x, |X n i ω x| ≤ bound n ω)
    (hb : TendstoInMeasure μ bound atTop (fun _ => 0))
    (hv : TendstoInMeasure μ
      (fun n ω => ∑ i ∈ range (m n), variance (X n i ω) (ν ω)) atTop (fun _ => 1))
    (g : ℝ → ℝ) (hg : Continuous g) [NullSingletonClass (standardNormal.map g)]
    (hmeas : ∀ n, AEStronglyMeasurable
      (fun ω => cdfDistanceTo ((ν ω).map (fun x => g (∑ i ∈ range (m n), X n i ω x)))
        (standardNormal.map g)) μ) :
    TendstoInMeasure μ
      (fun n ω => cdfDistanceTo ((ν ω).map (fun x => g (∑ i ∈ range (m n), X n i ω x)))
        (standardNormal.map g)) atTop (fun _ => 0) := by
  apply (exists_seq_tendstoInMeasure_atTop_iff hmeas).mpr
  intro ns hns
  obtain ⟨bs, hbs, hbpath⟩ := (hb.comp hns.tendsto_atTop).exists_seq_tendsto_ae
  obtain ⟨cs, hcs, hvpath⟩ :=
    (hv.comp (hns.tendsto_atTop.comp hbs.tendsto_atTop)).exists_seq_tendsto_ae
  refine ⟨bs ∘ cs, hbs.comp hcs, ?_⟩
  filter_upwards [hbpath, hvpath] with ω hbω hvω
  have hbω' := hbω.comp hcs.tendsto_atTop
  have hl : ∀ γ > 0, Tendsto (fun n => ∑ i ∈ range (m (ns (bs (cs n)))),
      ∫ x, if γ < |X (ns (bs (cs n))) i ω x| then X (ns (bs (cs n))) i ω x ^ 2 else 0
        ∂ν ω) atTop (𝓝 0) := by
    intro γ hγ
    apply tendsto_const_nhds.congr'
    filter_upwards [hbω'.eventually (gt_mem_nhds hγ)] with n hn
    symm
    apply sum_eq_zero
    intro i _
    have hh : ∀ x, ¬ γ < |X (ns (bs (cs n))) i ω x| := fun x =>
      not_lt_of_ge (le_trans (hbound _ _ _ _) hn.le)
    simp only [hh, ite_false, integral_zero]
  have hweak := (lf ω (fun n i => X (ns (bs (cs n))) i ω) (fun n => m (ns (bs (cs n))))
    (fun n i => hX _ _ _) (fun n => hind _ _) (fun n i => hzero _ _ _) hvω hl).1
  have hscore := hweak.continuous_comp hg
  have hc : Continuous (cdf (standardNormal.map (g ∘ id))) := by
    simpa only [Function.comp_id] using continuous_cdf_of_no_atoms (standardNormal.map g)
  have h := polya
    (fun n => ⟨(ν ω).map (g ∘ fun x => ∑ i ∈ range (m (ns (bs (cs n)))), X (ns (bs (cs n))) i ω x), inferInstance⟩)
    ⟨standardNormal.map (g ∘ id), inferInstance⟩ hc hscore.tendsto
  change Tendsto (fun n => cdfDistanceTo
    ((ν ω).map (g ∘ fun x => ∑ i ∈ range (m (ns (bs (cs n)))), X (ns (bs (cs n))) i ω x))
    (standardNormal.map (g ∘ id))) atTop (𝓝 0) at h
  simp only [Function.comp_id] at h
  convert! h using 1

theorem normalized_max_vanishes {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (a v : ℕ → Ω → ℝ) (η : Ω → ℝ)
    (haM : ∀ n, AEMeasurable (a n) μ) (hvM : ∀ n, AEMeasurable (v n) μ)
    (ha : TendstoInMeasure μ a atTop (fun _ => 0))
    (hv : TendstoInMeasure μ v atTop η) (hη : ∀ᵐ ω ∂μ, 0 < η ω) :
    TendstoInMeasure μ (fun n ω => a n ω / Real.sqrt (v n ω)) atTop (fun _ => 0) := by
  apply (exists_seq_tendstoInMeasure_atTop_iff (fun n =>
    ((haM n).div (Real.continuous_sqrt.measurable.comp_aemeasurable (hvM n))).aestronglyMeasurable)).mpr
  intro ns hns
  obtain ⟨bs, hbs, hapath⟩ := (ha.comp hns.tendsto_atTop).exists_seq_tendsto_ae
  obtain ⟨cs, hcs, hvpath⟩ :=
    (hv.comp (hns.tendsto_atTop.comp hbs.tendsto_atTop)).exists_seq_tendsto_ae
  refine ⟨bs ∘ cs, hbs.comp hcs, ?_⟩
  filter_upwards [hapath, hvpath, hη] with ω haω hvω hηω
  have h := (haω.comp hcs.tendsto_atTop).div
    (Real.continuous_sqrt.continuousAt.tendsto.comp hvω) (ne_of_gt (Real.sqrt_pos.mpr hηω))
  simp only [zero_div] at h
  convert! h using 1

lemma normalized_max_identity (a σ scale : ℝ) (hσ : 0 ≤ σ) (hs : 0 < scale) :
    (a / scale) / Real.sqrt (σ ^ 2 / scale ^ 2) = a / σ := by
  rw [Real.sqrt_div (sq_nonneg _), Real.sqrt_sq hσ, Real.sqrt_sq hs.le]
  by_cases h : σ = 0
  · simp [h]
  · field_simp

theorem redraw_variance_tends_one {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (σ : ℕ → Ω → ℝ)
    (hz : Tendsto (fun n => μ {ω | σ n ω = 0}) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => if σ n ω = 0 then (0 : ℝ) else 1) atTop (fun _ => 1) := by
  apply tendstoInMeasure_iff_dist.mpr
  intro ε hε
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hz (fun _ => zero_le)
  intro n
  apply measure_mono
  intro ω hω
  by_contra hn
  change σ n ω ≠ 0 at hn
  simp [hn] at hω
  linarith

lemma success_set_mass (B j : ℕ) (p : ℝ) :
    (∑ s : Finset (Fin B), if s.card = j then p ^ s.card * (1 - p) ^ (B - s.card) else 0) =
      binomialMass B j p := by
  have he : (univ : Finset (Finset (Fin B))).filter (fun s => s.card = j) =
      (univ : Finset (Fin B)).powersetCard j := by
    ext s
    simp
  rw [← sum_filter, he]
  have hh : (∑ s ∈ (univ : Finset (Fin B)).powersetCard j,
      p ^ s.card * (1 - p) ^ (B - s.card)) =
      ∑ _s ∈ (univ : Finset (Fin B)).powersetCard j, p ^ j * (1 - p) ^ (B - j) := by
    apply sum_congr rfl
    intro s hs
    rw [(mem_powersetCard.mp hs).2]
  rw [hh]
  simp [card_powersetCard, binomialMass, mul_assoc]

/-- A conditionally independent success-set law yields the binomial count
law. Each subset records which of the B iid redraws exceeds the observation. -/
theorem conditional_binomial_from_success_sets {Ω : Type*} (G : MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (B : ℕ) (A : Ω → Finset (Fin B)) (P : Ω → ℝ)
    (hA : ∀ s, MeasurableSet {ω | A ω = s})
    (hlaw : ∀ s, μ[({ω | A ω = s}).indicator (fun _ => (1 : ℝ)) | G] =ᵐ[μ]
      (fun ω => P ω ^ s.card * (1 - P ω) ^ (B - s.card))) (j : ℕ) :
    μ[({ω | (A ω).card = j}).indicator (fun _ => (1 : ℝ)) | G] =ᵐ[μ]
      (fun ω => binomialMass B j (P ω)) := by
  let f : Finset (Fin B) → Ω → ℝ := fun s =>
    if s.card = j then ({ω | A ω = s}).indicator (fun _ => 1) else 0
  have he : ({ω | (A ω).card = j}).indicator (fun _ => (1 : ℝ)) = ∑ s, f s := by
    funext ω
    rw [Finset.sum_apply, sum_eq_single (A ω)]
    · by_cases h : (A ω).card = j <;> simp [f, h]
    · intro s _ hs
      have hne : A ω ≠ s := Ne.symm hs
      have hmem : ω ∉ {ω | A ω = s} := hne
      by_cases hc : s.card = j <;> simp [f, hc, hmem]
    · simp
  rw [he]
  have hi : ∀ s, Integrable (f s) μ := by
    intro s
    unfold f
    split_ifs
    · exact (integrable_const (1 : ℝ)).indicator (hA s)
    · exact integrable_zero _ _ _
  have hc := condExp_finsetSum (s := univ) (fun s _ => hi s) G
  have hall : ∀ s, μ[f s | G] =ᵐ[μ] (fun ω => if s.card = j then
      P ω ^ s.card * (1 - P ω) ^ (B - s.card) else 0) := by
    intro s
    unfold f
    split_ifs
    · exact hlaw s
    · rw [condExp_zero]
      exact Eventually.of_forall fun _ => rfl
  filter_upwards [hc, Filter.eventually_all.mpr hall] with ω hc hall
  simp only [Finset.sum_apply] at hc
  rw [hc, ← success_set_mass B j (P ω)]
  exact sum_congr rfl fun s _ => hall s

theorem continuous_score_limit {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (Z : ℕ → Ω → ℝ)
    (hZ : TendstoInDistribution Z atTop id (fun _ => μ) standardNormal)
    (g : ℝ → ℝ) (hg : Continuous g) :
    TendstoInDistribution (fun n ω => g (Z n ω)) atTop id (fun _ => μ) (standardNormal.map g) := by
  have h := hZ.continuous_comp hg
  refine ⟨h.forall_aemeasurable, measurable_id.aemeasurable, ?_⟩
  simp only [Measure.map_id]
  have ht := h.tendsto
  simp only [Function.comp_id] at ht
  convert! ht using 1

theorem lower_absolute_normal_continuous :
    Continuous (cdf (standardNormal.map (fun x : ℝ => -‖x‖))) := by
  have : NullSingletonClass (standardNormal.map (fun x : ℝ => -‖x‖)) :=
    lower_absolute_atomless standardNormal
  exact continuous_cdf_of_no_atoms _

theorem fixed_direction_score_limit {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (Z : ℕ → Ω → ℝ)
    (hZ : ∀ n, AEMeasurable (Z n) μ)
    (hstable : StableGaussianLimit μ Z (fun _ => 1))
    (d : ℝ) (hd : d = 1 ∨ d = -1) (o : Ω → ℝ)
    (ho : Measurable o) (hos : ∀ ω, o ω = 1 ∨ o ω = -1) :
    TendstoInDistribution (fun n ω => (-d * o ω) * Z n ω) atTop id (fun _ => μ) standardNormal := by
  apply stable_fixed_orientation μ Z hZ hstable (fun ω => -d * o ω) (by fun_prop)
  intro ω
  rcases hd with h | h <;> rcases hos ω with h' | h' <;> simp [h, h']

theorem zero_variance_vanishes {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (σ : ℕ → Ω → ℝ) (scale : ℕ → ℝ)
    (η : Ω → ℝ) (hσ : ∀ n, Measurable (σ n)) (hscale : ∀ n, 0 < scale n)
    (hη : Measurable η) (hpos : ∀ᵐ ω ∂μ, 0 < η ω)
    (hv : TendstoInMeasure μ (fun n ω => σ n ω ^ 2 / scale n ^ 2)
      atTop (fun ω => η ω ^ 2)) :
    Tendsto (fun n => μ {ω | σ n ω = 0}) atTop (𝓝 0) := by
  have h := zero_event_vanishes μ _ (fun ω => η ω ^ 2)
    (fun n => by fun_prop) (by fun_prop)
    (hpos.mono fun ω hω => sq_pos_of_pos hω) hv
  have he : ∀ n, {ω | σ n ω ^ 2 / scale n ^ 2 = 0} = {ω | σ n ω = 0} := by
    intro n
    ext ω
    simp [div_eq_zero_iff, ne_of_gt (hscale n)]
  simpa only [he] using h

lemma cdf_mapped_exceedance (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (g : ℝ → ℝ) (hg : Measurable g) (z : ℝ) :
    cdf (ν.map (fun x => -g x)) (-g z) = ν.real {x | g z ≤ g x} := by
  have hm : Measurable (fun x => -g x) := by fun_prop
  rw [cdf_eq_real, measureReal_def, Measure.map_apply hm measurableSet_Iic]
  have he : (fun x => -g x) ⁻¹' Set.Iic (-g z) = {x | g z ≤ g x} := by ext x; simp
  rw [he]
  rfl

lemma lower_absolute_exceedance (z redraw : ℝ) :
    |z| ≤ |redraw| ↔ -‖redraw‖ ≤ -‖z‖ := by simp [Real.norm_eq_abs]

lemma reportedP_conservative (B Q : ℕ) (hQ : Q ≤ B) (σ : ℝ) :
    (1 + (Q : ℝ)) / (B + 1) ≤ reportedP σ ((1 + (Q : ℝ)) / (B + 1)) := by
  unfold reportedP
  split_ifs
  · apply (div_le_iff₀ (by positivity)).mpr
    have h : (Q : ℝ) ≤ B := by exact_mod_cast hQ
    linarith
  · exact le_rfl

theorem zero_redraw_moments {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (S : Ω → ℝ) : (∫ ω, S ω / (0 : ℝ) ∂μ) = 0 ∧
      variance (fun ω => S ω / (0 : ℝ)) μ = 0 := by
  simp only [div_zero, integral_zero, true_and]
  change variance (0 : Ω → ℝ) μ = 0
  exact variance_zero μ

end OranCausal.Asymptotic

#print axioms OranCausal.Asymptotic.binomial_uniform_integral
#print axioms OranCausal.Asymptotic.pmrt_martingale
#print axioms OranCausal.Asymptotic.self_normalized_clt
#print axioms OranCausal.Asymptotic.redraw_cdf_limit
#print axioms OranCausal.Asymptotic.redraw_score_cdf_limit
#print axioms OranCausal.Asymptotic.asymptotic_rejection_bound
#print axioms OranCausal.Asymptotic.conditional_binomial_from_success_sets
#print axioms OranCausal.Asymptotic.fixed_direction_score_limit
#print axioms OranCausal.Asymptotic.zero_variance_vanishes
#print axioms OranCausal.Asymptotic.standardized_redraw_moments
#print axioms OranCausal.Asymptotic.bernoulli_redraw_moments
#print axioms OranCausal.Asymptotic.cdf_uniform
#print axioms OranCausal.Asymptotic.conditional_binomial_rejection_limit
#print axioms OranCausal.Asymptotic.zero_redraw_moments
