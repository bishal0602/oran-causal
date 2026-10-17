module

public import OranCausal.StepUp
public import Mathlib.Analysis.SpecificLimits.Basic
public import Mathlib.Probability.Martingale.Basic
public import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.Real
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut
public import Mathlib.Probability.Distributions.Bernoulli
public import Mathlib.Probability.Independence.InfinitePi

public section

open Finset Filter
open scoped BigOperators Topology Classical ENNReal NNReal

namespace OranCausal.Slope

def label (b : Bool) : ℝ := if b then 1 else 0

def signed (b : Bool) : ℝ := 2 * label b - 1

noncomputable def score (s p : ℝ) (b : Bool) : ℝ := s * (label b - p)

noncomputable def slope (N : ℕ) (s p y c : ℕ → ℝ) (b : ℕ → Bool) : ℝ :=
  (∑ u ∈ range N, score (s u) (p u) (b u) * (y u - c u)) /
    (∑ u ∈ range N, score (s u) (p u) (b u) ^ 2)

noncomputable def ht (N : ℕ) (s p y c : ℕ → ℝ) (b : ℕ → Bool) : ℝ :=
  (∑ u ∈ range N, s u * (label (b u) * (y u - c u) / p u -
    (1 - label (b u)) * (y u - c u) / (1 - p u))) / N

/-- The identity is purely algebraic; neither predictability nor boundedness is needed. -/
theorem half_propensity_identity (N : ℕ) (s y c : ℕ → ℝ) (b : ℕ → Bool)
    (hs : ∀ u < N, s u ^ 2 = 1) :
    slope N s (fun _ => 1 / 2) y c b = ht N s (fun _ => 1 / 2) y c b := by
  have hd : (∑ u ∈ range N, score (s u) (1 / 2) (b u) ^ 2) = (N : ℝ) / 4 := by
    calc
      _ = ∑ _u ∈ range N, (1 / 4 : ℝ) := by
        apply sum_congr rfl
        intro u hu
        have h := hs u (mem_range.mp hu)
        cases b u <;> dsimp [score, label] <;> nlinarith
      _ = _ := by simp; ring
  have hn : (∑ u ∈ range N, s u *
      (label (b u) * (y u - c u) / (1 / 2) -
      (1 - label (b u)) * (y u - c u) / (1 - 1 / 2))) =
      4 * ∑ u ∈ range N, score (s u) (1 / 2) (b u) * (y u - c u) := by
    rw [mul_sum]
    apply sum_congr rfl
    intro u _
    unfold score
    ring
  unfold slope ht
  rw [hd, hn]
  ring

theorem half_propensity_target (N : ℕ) (s tau : ℕ → ℝ) :
    (∑ u ∈ range N, (1 / 2 : ℝ) * (1 - 1 / 2) * s u * tau u) /
      (∑ _u ∈ range N, (1 / 2 : ℝ) * (1 - 1 / 2)) =
    (∑ u ∈ range N, s u * tau u) / N := by
  have he : (∑ u ∈ range N, (1 / 2 : ℝ) * (1 - 1 / 2) * s u * tau u) =
      (1 / 4 : ℝ) * ∑ u ∈ range N, s u * tau u := by
    rw [mul_sum]
    apply sum_congr rfl
    intro u _
    ring
  rw [he]
  simp only [sum_const, card_range, nsmul_eq_mul]
  ring

lemma abs_signed (b : Bool) : |signed b| = 1 := by
  cases b <;> norm_num [signed, label]

noncomputable def geometricSum (N : ℕ) (b : ℕ → Bool) : ℝ :=
  ∑ u ∈ range N, signed (b u) * 3 ^ (u + 1)

lemma geometricSum_upper (N : ℕ) (b : ℕ → Bool) :
    |geometricSum N b| ≤ (3 ^ (N + 1) - 3) / 2 := by
  induction N with
  | zero => norm_num [geometricSum]
  | succ N ih =>
    have he : geometricSum (N + 1) b = geometricSum N b + signed (b N) * 3 ^ (N + 1) := by
      simp [geometricSum, sum_range_succ]
    rw [he]
    calc
      _ ≤ |geometricSum N b| + |signed (b N) * 3 ^ (N + 1)| := abs_add_le _ _
      _ ≤ (3 ^ (N + 1) - 3) / 2 + 3 ^ (N + 1) := by
        rw [abs_mul, abs_signed, one_mul,
          abs_of_nonneg (by positivity : (0 : ℝ) ≤ 3 ^ (N + 1))]
        linarith
      _ = _ := by rw [pow_succ]; ring

lemma geometricSum_lower (N : ℕ) (b : ℕ → Bool) :
    (3 ^ (N + 1) + 3) / 2 ≤ |geometricSum (N + 1) b| := by
  have hu := geometricSum_upper N b
  have he : geometricSum (N + 1) b = geometricSum N b + signed (b N) * 3 ^ (N + 1) := by
    simp [geometricSum, sum_range_succ]
  have ht := abs_add_le (geometricSum (N + 1) b) (-geometricSum N b)
  rw [abs_neg] at ht
  have hv : geometricSum (N + 1) b + -geometricSum N b = signed (b N) * 3 ^ (N + 1) := by
    rw [he]; ring
  rw [hv, abs_mul, abs_signed, one_mul, abs_of_nonneg (by positivity)] at ht
  linarith

noncomputable def badSlope (N : ℕ) (b : ℕ → Bool) : ℝ :=
  slope N (fun _ => 1) (fun _ => 1 / 2) (fun _ => 0) (fun u => 3 ^ (u + 1)) b

lemma badSlope_eq (N : ℕ) (b : ℕ → Bool) :
    badSlope N b = -2 * geometricSum N b / N := by
  unfold badSlope
  rw [half_propensity_identity N _ _ _ _ (by intros; norm_num)]
  unfold ht geometricSum
  rw [mul_sum]
  congr 1
  apply sum_congr rfl
  intro u _
  dsimp [signed]
  ring

/-- Every assignment path violates convergence, including independent fair assignments.
The zero-based centre `3^(u+1)` is the manuscript's one-based `3^u`. -/
theorem unbounded_centre_counterexample (N : ℕ) (hN : 0 < N) (b : ℕ → Bool) :
    (3 ^ N + 3) / (N : ℝ) ≤ |badSlope N b| := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hN)
  have hl := geometricSum_lower k b
  rw [badSlope_eq, abs_div, abs_mul]
  norm_num only [abs_neg, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
  rw [abs_of_nonneg (by positivity : (0 : ℝ) ≤ ↑(k + 1))]
  apply div_le_div_of_nonneg_right _ (by positivity)
  change 3 ^ (k + 1) + 3 ≤ 2 * |geometricSum (k + 1) b|
  linarith

open MeasureTheory ProbabilityTheory

noncomputable def fairBit : Measure Bool :=
  bernoulliMeasure true false ⟨1 / 2, by constructor <;> norm_num⟩

instance : IsProbabilityMeasure fairBit := by unfold fairBit; infer_instance

noncomputable def fairAssignments : Measure (ℕ → Bool) :=
  Measure.infinitePi (fun _ => fairBit)

instance : IsProbabilityMeasure fairAssignments := by unfold fairAssignments; infer_instance

theorem fairAssignments_independent :
    iIndepFun (fun u (b : ℕ → Bool) => b u) fairAssignments :=
  iIndepFun_infinitePi (X := fun _ => id) (fun _ => measurable_id)

theorem fairAssignments_marginal (u : ℕ) :
    fairAssignments {b | b u = true} = (1 / 2 : ℝ≥0∞) := by
  have h := Measure.infinitePi_map_eval (fun _ : ℕ => fairBit) u
  have hm : MeasurableSet ({true} : Set Bool) := measurableSet_singleton _
  have he := congrArg (fun μ : Measure Bool => μ {true}) h
  rw [Measure.map_apply (measurable_pi_apply u) hm] at he
  change fairAssignments {b | b u = true} = _ at he
  rw [he]
  norm_num [fairBit, bernoulliMeasure_apply, unitInterval.toNNReal]
  change (↑((1 : ℝ≥0) / 2) : ℝ≥0∞) = 2⁻¹
  norm_num

lemma nat_le_three_pow (N : ℕ) : (N : ℝ) ≤ 3 ^ N := by
  induction N with
  | zero => norm_num
  | succ N ih =>
    have hp : (1 : ℝ) ≤ 3 ^ N := one_le_pow₀ (by norm_num)
    rw [Nat.cast_succ, pow_succ]
    linarith

theorem counterexample_probability_one (N : ℕ) (hN : 0 < N) :
    fairAssignments {b | 1 ≤ |badSlope N b|} = 1 := by
  have hb : ∀ b, 1 ≤ |badSlope N b| := by
    intro b
    apply le_trans _ (unbounded_centre_counterexample N hN b)
    apply (le_div_iff₀ (by exact_mod_cast hN)).mpr
    have := nat_le_three_pow N
    linarith
  simp [hb]

theorem counterexample_not_convergent :
    ¬ TendstoInMeasure fairAssignments (fun N b => badSlope N b) atTop (fun _ => 0) := by
  intro h
  have ht := (tendstoInMeasure_iff_dist.mp h) 1 (by norm_num)
  have he : (fun N => fairAssignments {b | 1 ≤ dist (badSlope N b) 0}) =ᶠ[atTop]
      (fun _ => (1 : ℝ≥0∞)) := by
    filter_upwards [eventually_gt_atTop 0] with N hN
    simpa [Real.dist_eq] using counterexample_probability_one N hN
  have hz : Tendsto (fun _ : ℕ => (1 : ℝ≥0∞)) atTop (𝓝 0) := ht.congr' he
  have : (1 : ℝ≥0∞) = 0 := tendsto_nhds_unique tendsto_const_nhds hz
  exact one_ne_zero this

variable {Ω : Type*} [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]

/-- Named library bridge: the bounded martingale-difference weak law.
This is a universally quantified theorem obligation, not a modelling assumption. -/
def BoundedMartingaleWeakLaw : Prop :=
  ∀ (F : Filtration ℕ mΩ) (D : ℕ → Ω → ℝ) (K : ℝ),
    (∀ e, StronglyMeasurable[F (e + 1)] (D e)) →
    (∀ e, Integrable (D e) μ) →
    (∀ e, μ[D e | F e] =ᵐ[μ] 0) →
    (∀ e, ∀ᵐ ω ∂μ, |D e ω| ≤ K) →
    TendstoInMeasure μ (fun n ω => (∑ e ∈ range n, D e ω) / n) atTop (fun _ => 0)

lemma integrable_bounded {f : Ω → ℝ} {K : ℝ}
    (hf : AEStronglyMeasurable f μ) (hb : ∀ᵐ ω ∂μ, |f ω| ≤ K) : Integrable f μ := by
  apply (integrable_const K).mono' hf
  simpa only [Real.norm_eq_abs] using hb

/-- H1--H3 turn unit-centred errors into bounded episode martingale differences.
Nonexistent slots are zero; the finite slot type implements the H2 cap. -/
theorem episode_blocking {K : ℕ}
    (weakLaw : BoundedMartingaleWeakLaw μ)
    (F : Filtration ℕ mΩ)
    (H : ℕ → Fin K → MeasurableSpace Ω) (X : ℕ → Fin K → Ω → ℝ) (C : ℝ)
    (hbefore : ∀ e j, F e ≤ H e j)
    (hafter : ∀ e j, H e j ≤ F (e + 1))
    (hobs : ∀ e j, StronglyMeasurable[F (e + 1)] (X e j))
    (hzero : ∀ e j, μ[X e j | H e j] =ᵐ[μ] 0)
    (hbound : ∀ e j, ∀ᵐ ω ∂μ, |X e j ω| ≤ C) :
    TendstoInMeasure μ
      (fun n ω => (∑ e ∈ range n, ∑ j : Fin K, X e j ω) / n) atTop (fun _ => 0) := by
  have hi : ∀ e j, Integrable (X e j) μ := fun e j =>
    integrable_bounded μ ((hobs e j).mono (F.le _)).aestronglyMeasurable (hbound e j)
  apply weakLaw F (fun e ω => ∑ j : Fin K, X e j ω) (K * C)
  · intro e
    exact Finset.stronglyMeasurable_fun_sum _ (fun j _ => hobs e j)
  · intro e
    exact integrable_finsetSum _ (fun j _ => hi e j)
  · intro e
    have hz : ∀ j, μ[X e j | F e] =ᵐ[μ] 0 := by
      intro j
      have ht := condExp_condExp_of_le (μ := μ) (f := X e j)
        (hbefore e j) ((hafter e j).trans (F.le _))
      exact ht.symm.trans ((condExp_congr_ae (hzero e j)).trans (by simp))
    have hs := condExp_finsetSum (μ := μ) (m := F e) (s := Finset.univ)
      (f := X e) (fun j _ => hi e j)
    have he : (∑ j : Fin K, X e j) = (fun ω => ∑ j : Fin K, X e j ω) := by
      funext ω; simp
    rw [he] at hs
    filter_upwards [hs, ae_all_iff.mpr hz] with ω hsω hzω
    simpa [Finset.sum_apply, hzω] using hsω
  · intro e
    filter_upwards [ae_all_iff.mpr (hbound e)] with ω hω
    calc
      _ ≤ ∑ j : Fin K, |X e j ω| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ _j : Fin K, C := sum_le_sum (fun j _ => hω j)
      _ = _ := by simp

theorem ratio_error_bound {A B T W d M : ℝ} (hd : 0 < d)
    (hB : d ≤ B) (hW : d ≤ W) (hT : |T| ≤ M) :
    |A / B - T / W| ≤ |A - T| / d + M * |B - W| / d ^ 2 := by
  have hBp : 0 < B := lt_of_lt_of_le hd hB
  have hWp : 0 < W := lt_of_lt_of_le hd hW
  have he : A / B - T / W = (A - T) / B - T * (B - W) / (B * W) := by
    field_simp; ring
  rw [he]
  calc
    _ ≤ |(A - T) / B| + |T * (B - W) / (B * W)| := abs_sub _ _
    _ = |A - T| / B + |T| * |B - W| / (B * W) := by
      rw [abs_div, abs_div, abs_mul, abs_of_pos hBp, abs_of_pos (mul_pos hBp hWp)]
    _ ≤ |A - T| / d + M * |B - W| / d ^ 2 := by
      apply add_le_add
      · exact div_le_div_of_nonneg_left (abs_nonneg _) hd hB
      · apply le_trans (div_le_div_of_nonneg_right
          (mul_le_mul_of_nonneg_right hT (abs_nonneg _)) (mul_pos hBp hWp).le)
        apply div_le_div_of_nonneg_left (mul_nonneg ((abs_nonneg _).trans hT) (abs_nonneg _))
          (sq_pos_of_pos hd)
        nlinarith [mul_le_mul hB hW hd.le hBp.le]

/-- Convergence is preserved under a fixed linear bound holding with probability tending to one. -/
theorem convergence_on_good_event (f x y : ℕ → Ω → ℝ) (good : ℕ → Set Ω)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hx : TendstoInMeasure μ x atTop (fun _ => 0))
    (hy : TendstoInMeasure μ y atTop (fun _ => 0))
    (hg : Tendsto (fun n => μ.real (good n)ᶜ) atTop (𝓝 0))
    (hbound : ∀ n, ∀ᵐ ω ∂μ, ω ∈ good n → |f n ω| ≤ a * |x n ω| + b * |y n ω|) :
    TendstoInMeasure μ f atTop (fun _ => 0) := by
  rw [tendstoInMeasure_iff_measureReal_norm] at hx hy ⊢
  simp only [sub_zero, Real.norm_eq_abs] at hx hy ⊢
  intro ε hε
  let δ := ε / (2 * (a + b + 1))
  have hδ : 0 < δ := div_pos hε (by linarith)
  have hsum : Tendsto (fun n => μ.real (good n)ᶜ +
      μ.real {ω | δ ≤ |x n ω|} + μ.real {ω | δ ≤ |y n ω|}) atTop (𝓝 0) := by
    simpa using (hg.add (hx δ hδ)).add (hy δ hδ)
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hsum
    (fun _ => measureReal_nonneg)
  intro n
  have hs : {ω | ε ≤ |f n ω|} ≤ᵐ[μ]
      (good n)ᶜ ∪ {ω | δ ≤ |x n ω|} ∪ {ω | δ ≤ |y n ω|} := by
    filter_upwards [hbound n] with ω hω
    intro hf
    by_contra hn
    have hgω : ω ∈ good n := by simpa only [Set.mem_union, Set.mem_compl_iff,
      Set.mem_ofPred_eq, not_or, not_not] using (show ω ∉ (good n)ᶜ from fun h => hn (Or.inl (Or.inl h)))
    have hxx : |x n ω| < δ := lt_of_not_ge (fun h => hn (Or.inl (Or.inr h)))
    have hyy : |y n ω| < δ := lt_of_not_ge (fun h => hn (Or.inr h))
    have h1 := mul_le_mul_of_nonneg_left hxx.le ha
    have h2 := mul_le_mul_of_nonneg_left hyy.le hb
    have hδeq : δ * (2 * (a + b + 1)) = ε := by
      dsimp [δ]; field_simp
    have hbd := hω hgω
    change ε ≤ |f n ω| at hf
    nlinarith
  calc
    _ ≤ μ.real ((good n)ᶜ ∪ {ω | δ ≤ |x n ω|} ∪ {ω | δ ≤ |y n ω|}) :=
      ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae hs)
    _ ≤ _ := (measureReal_union_le _ _).trans
      (add_le_add (measureReal_union_le _ _) le_rfl)

lemma ce_affine {H : MeasurableSpace Ω} (hH : H ≤ mΩ)
    {a b L p : Ω → ℝ} (ha : StronglyMeasurable[H] a)
    (hb : StronglyMeasurable[H] b)
    (ia : MemLp a ∞ μ) (ib : MemLp b ∞ μ) (iL : MemLp L ∞ μ)
    (hL : μ[L | H] =ᵐ[μ] p) :
    μ[(fun ω => a ω * L ω + b ω) | H] =ᵐ[μ] (fun ω => a ω * p ω + b ω) := by
  let : MeasurableSpace Ω := mΩ
  have iap : MemLp (fun ω => a ω * L ω) ∞ μ := ia.mul iL
  have h1 := condExp_add (iap.integrable (by simp)) (ib.integrable (by simp)) H
  have h2 := condExp_mul_of_stronglyMeasurable_left ha
    (iap.integrable (by simp)) (iL.integrable (by simp))
  have h3 := condExp_of_stronglyMeasurable (μ := μ) hH hb (ib.integrable (by simp))
  filter_upwards [h1, h2, hL] with ω h1ω h2ω hLω
  change μ[(fun ω => a ω * L ω + b ω) | H] ω =
    μ[(fun ω => a ω * L ω) | H] ω + μ[b | H] ω at h1ω
  change μ[(fun ω => a ω * L ω) | H] ω = a ω * μ[L | H] ω at h2ω
  rw [h1ω, h2ω, h3, hLω]

/-- A conditional assignment effect written using joint conditional moments. -/
noncomputable def assignmentEffect (H : MeasurableSpace Ω) (L p y : Ω → ℝ) : Ω → ℝ :=
  fun ω => μ[(fun z => L z * y z) | H] ω / p ω -
    (μ[y | H] ω - μ[(fun z => L z * y z) | H] ω) / (1 - p ω)

omit [IsProbabilityMeasure μ] in
theorem zero_outcome_effect (H : MeasurableSpace Ω) (L p : Ω → ℝ) :
    assignmentEffect (mΩ := mΩ) μ H L p (fun _ => 0) = 0 := by
  funext ω
  simp [assignmentEffect]

omit [IsProbabilityMeasure μ] in
/-- The rejection term in the effect is the joint rejection-outcome conditional moment. -/
theorem assignmentEffect_joint_moments (H : MeasurableSpace Ω) (L p y : Ω → ℝ)
    (iy : Integrable y μ) (iLy : Integrable (fun ω => L ω * y ω) μ) :
    assignmentEffect (mΩ := mΩ) μ H L p y =ᵐ[μ]
      (fun ω => μ[(fun z => L z * y z) | H] ω / p ω -
        μ[(fun z => (1 - L z) * y z) | H] ω / (1 - p ω)) := by
  let : MeasurableSpace Ω := mΩ
  have hs := condExp_sub iy iLy H
  have he : y - (fun ω => L ω * y ω) = (fun ω => (1 - L ω) * y ω) := by
    funext ω; dsimp; ring
  rw [he] at hs
  filter_upwards [hs] with ω hω
  simp only [Pi.sub_apply] at hω
  simp only [assignmentEffect, hω]

/-- Bernoulli moments with a predictable masked direction `a` and centre `c`. -/
theorem binary_design_moments {H : MeasurableSpace Ω} (hH : H ≤ mΩ)
    (a p c L y : Ω → ℝ)
    (ha : StronglyMeasurable[H] a) (hp : StronglyMeasurable[H] p)
    (hc : StronglyMeasurable[H] c)
    (ia : MemLp a ∞ μ) (ip : MemLp p ∞ μ) (ic : MemLp c ∞ μ)
    (iL : MemLp L ∞ μ) (iy : MemLp y ∞ μ)
    (hL : ∀ ω, L ω = 0 ∨ L ω = 1)
    (hoverlap : ∀ ω, p ω ≠ 0 ∧ 1 - p ω ≠ 0)
    (hdesign : μ[L | H] =ᵐ[μ] p) :
    let q := fun ω => a ω * (L ω - p ω)
    (μ[q | H] =ᵐ[μ] 0) ∧
    (μ[(fun ω => q ω ^ 2) | H] =ᵐ[μ] (fun ω => a ω ^ 2 * p ω * (1 - p ω))) ∧
    (μ[(fun ω => q ω * (y ω - c ω)) | H] =ᵐ[μ]
      (fun ω => a ω * p ω * (1 - p ω) * assignmentEffect (mΩ := mΩ) μ H L p y ω)) ∧
    (μ[(fun ω => q ω * (y ω - c ω)) | H] =ᵐ[μ] μ[(fun ω => q ω * y ω) | H]) := by
  let : MeasurableSpace Ω := mΩ
  let q := fun ω => a ω * (L ω - p ω)
  have iq : MemLp q ∞ μ := ia.mul (iL.sub ip)
  have hap : MemLp (fun ω => a ω * p ω) ∞ μ := ia.mul ip
  have hq : μ[q | H] =ᵐ[μ] 0 := by
    have he : q = fun ω => a ω * L ω + -(a ω * p ω) := by funext ω; dsimp [q]; ring
    rw [he]
    have hh := ce_affine μ hH ha (ha.mul hp).neg ia hap.neg iL hdesign
    filter_upwards [hh] with ω hω
    simpa using hω
  refine ⟨hq, ?_, ?_, ?_⟩
  · have he : (fun ω => q ω ^ 2) =
        (fun ω => (a ω * a ω * (1 - 2 * p ω)) * L ω + (a ω * a ω * (p ω * p ω))) := by
      funext ω
      rcases hL ω with hh | hh <;> dsimp [q] <;> rw [hh] <;> ring
    rw [he]
    have ih1 : MemLp (fun ω => a ω * a ω * (1 - 2 * p ω)) ∞ μ :=
      (ia.mul ia : MemLp _ ∞ μ).mul ((memLp_const 1).sub (ip.const_mul 2))
    have ih2 : MemLp (fun ω => a ω * a ω * (p ω * p ω)) ∞ μ :=
      (ia.mul ia : MemLp _ ∞ μ).mul (ip.mul ip : MemLp _ ∞ μ)
    have hh := ce_affine μ hH ((ha.mul ha).mul (stronglyMeasurable_const.sub (hp.const_mul 2)))
      ((ha.mul ha).mul (hp.mul hp)) ih1 ih2 iL hdesign
    filter_upwards [hh] with ω hω
    simp only [Pi.mul_apply, Pi.sub_apply] at hω
    rw [hω]; ring
  · have iqy : MemLp (fun ω => q ω * y ω) ∞ μ := iq.mul iy
    have iqc : MemLp (fun ω => q ω * c ω) ∞ μ := iq.mul ic
    have hsub := condExp_sub (iqy.integrable (by simp)) (iqc.integrable (by simp)) H
    have hqc := condExp_mul_of_stronglyMeasurable_right hc
      (iqc.integrable (by simp)) (iq.integrable (by simp))
    have ialy : MemLp (fun ω => a ω * (L ω * y ω)) ∞ μ := ia.mul (iL.mul iy : MemLp _ ∞ μ)
    have iapy : MemLp (fun ω => (a ω * p ω) * y ω) ∞ μ := hap.mul iy
    have hy1 := condExp_sub (ialy.integrable (by simp)) (iapy.integrable (by simp)) H
    have hy2 := condExp_mul_of_stronglyMeasurable_left ha
      (ialy.integrable (by simp)) ((iL.mul iy : MemLp _ ∞ μ).integrable (by simp))
    have hy3 := condExp_mul_of_stronglyMeasurable_left (ha.mul hp)
      (iapy.integrable (by simp)) (iy.integrable (by simp))
    have he1 : (fun ω => q ω * (y ω - c ω)) =
        (fun ω => q ω * y ω) - (fun ω => q ω * c ω) := by funext ω; dsimp; ring
    have he2 : (fun ω => q ω * y ω) =
        (fun ω => a ω * (L ω * y ω)) - (fun ω => (a ω * p ω) * y ω) := by
      funext ω; dsimp [q]; ring
    rw [← he2] at hy1
    rw [← he1] at hsub
    filter_upwards [hsub, hqc, hq, hy1, hy2, hy3] with ω hsubω hqcω hqω hy1ω hy2ω hy3ω
    simp only [Pi.sub_apply, Pi.mul_apply, Pi.zero_apply] at *
    change μ[(fun ω => q ω * c ω) | H] ω = μ[q | H] ω * c ω at hqcω
    change μ[(fun ω => a ω * (L ω * y ω)) | H] ω =
      a ω * μ[(fun ω => L ω * y ω) | H] ω at hy2ω
    change μ[(fun ω => (a ω * p ω) * y ω) | H] ω =
      (a ω * p ω) * μ[y | H] ω at hy3ω
    rw [hsubω, hqcω, hqω, zero_mul, sub_zero, hy1ω, hy2ω, hy3ω]
    dsimp [assignmentEffect]
    obtain ⟨hp0, hp1⟩ := hoverlap ω
    field_simp
    <;> ring
  · have iqy : MemLp (fun ω => q ω * y ω) ∞ μ := iq.mul iy
    have iqc : MemLp (fun ω => q ω * c ω) ∞ μ := iq.mul ic
    have hsub := condExp_sub (iqy.integrable (by simp)) (iqc.integrable (by simp)) H
    have hqc := condExp_mul_of_stronglyMeasurable_right hc
      (iqc.integrable (by simp)) (iq.integrable (by simp))
    have he : (fun ω => q ω * (y ω - c ω)) =
        (fun ω => q ω * y ω) - (fun ω => q ω * c ω) := by funext ω; dsimp; ring
    rw [← he] at hsub
    filter_upwards [hsub, hqc, hq] with ω hsω hcω hqω
    simp only [Pi.sub_apply, Pi.mul_apply, Pi.zero_apply] at *
    change μ[(fun ω => q ω * c ω) | H] ω = μ[q | H] ω * c ω at hcω
    rw [hsω, hcω, hqω, zero_mul, sub_zero]

/-- Zero-padded episode slots. `a` is the direction on existing units and zero otherwise.
`L` is the acceptance indicator. `H` is the natural pre-opening history. -/
structure EpisodeDesign (K : ℕ) (F : Filtration ℕ mΩ)
    (H : ℕ → Fin K → MeasurableSpace Ω) (M C η : ℝ) where
  a : ℕ → Fin K → Ω → ℝ
  p : ℕ → Fin K → Ω → ℝ
  c : ℕ → Fin K → Ω → ℝ
  L : ℕ → Fin K → Ω → ℝ
  y : ℕ → Fin K → Ω → ℝ
  before : ∀ e j, F e ≤ H e j
  after : ∀ e j, H e j ≤ F (e + 1)
  a_pre : ∀ e j, StronglyMeasurable[H e j] (a e j)
  p_pre : ∀ e j, StronglyMeasurable[H e j] (p e j)
  c_pre : ∀ e j, StronglyMeasurable[H e j] (c e j)
  L_end : ∀ e j, StronglyMeasurable[F (e + 1)] (L e j)
  y_end : ∀ e j, StronglyMeasurable[F (e + 1)] (y e j)
  a_values : ∀ e j ω, a e j ω = 0 ∨ a e j ω = 1 ∨ a e j ω = -1
  L_values : ∀ e j ω, L e j ω = 0 ∨ L e j ω = 1
  overlap : ∀ e j ω, η ≤ p e j ω ∧ p e j ω ≤ 1 - η
  y_bound : ∀ e j ω, |y e j ω| ≤ M
  c_bound : ∀ e j ω, |c e j ω| ≤ C
  design : ∀ e j, μ[L e j | H e j] =ᵐ[μ] p e j

namespace EpisodeDesign

variable {μ} {K : ℕ} {F : Filtration ℕ mΩ}
  {H : ℕ → Fin K → MeasurableSpace Ω} {M C η : ℝ}
  (d : EpisodeDesign μ K F H M C η)

noncomputable def q (e : ℕ) (j : Fin K) (ω : Ω) : ℝ :=
  d.a e j ω * (d.L e j ω - d.p e j ω)

noncomputable def w (e : ℕ) (j : Fin K) (ω : Ω) : ℝ :=
  d.a e j ω ^ 2 * d.p e j ω * (1 - d.p e j ω)

noncomputable def t (e : ℕ) (j : Fin K) (ω : Ω) : ℝ :=
  d.a e j ω * d.p e j ω * (1 - d.p e j ω) *
    assignmentEffect (mΩ := mΩ) μ (H e j) (d.L e j) (d.p e j) (d.y e j) ω

noncomputable def z (e : ℕ) (j : Fin K) (ω : Ω) : ℝ :=
  d.q e j ω * (d.y e j ω - d.c e j ω)

noncomputable def total (f : ℕ → Fin K → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ∑ e ∈ range n, ∑ j : Fin K, f e j ω

noncomputable def count (n : ℕ) (ω : Ω) : ℝ :=
  total (fun e j ω => d.a e j ω ^ 2) n ω

noncomputable def beta (n : ℕ) (ω : Ω) : ℝ :=
  total d.z n ω / total (fun e j ω => d.q e j ω ^ 2) n ω

noncomputable def target (n : ℕ) (ω : Ω) : ℝ :=
  total d.t n ω / total d.w n ω

omit [IsProbabilityMeasure μ] in
lemma a_bound (e : ℕ) (j : Fin K) (ω : Ω) : |d.a e j ω| ≤ 1 := by
  rcases d.a_values e j ω with h | h | h <;> rw [h] <;> norm_num

omit [IsProbabilityMeasure μ] in
lemma p_bound (hη : 0 < η) (e : ℕ) (j : Fin K) (ω : Ω) : |d.p e j ω| ≤ 1 := by
  have h := d.overlap e j ω
  rw [abs_of_nonneg (by linarith)]
  linarith

omit [IsProbabilityMeasure μ] in
lemma L_bound (e : ℕ) (j : Fin K) (ω : Ω) : |d.L e j ω| ≤ 1 := by
  rcases d.L_values e j ω with h | h <;> rw [h] <;> norm_num

omit [IsProbabilityMeasure μ] in
lemma q_end (e : ℕ) (j : Fin K) : StronglyMeasurable[F (e + 1)] (d.q e j) :=
  ((d.a_pre e j).mono (d.after e j)).mul
    ((d.L_end e j).sub ((d.p_pre e j).mono (d.after e j)))

omit [IsProbabilityMeasure μ] in
lemma w_pre (e : ℕ) (j : Fin K) : StronglyMeasurable[H e j] (d.w e j) :=
  (((d.a_pre e j).pow 2).mul (d.p_pre e j)).mul
    (stronglyMeasurable_const.sub (d.p_pre e j))

omit [IsProbabilityMeasure μ] in
lemma t_pre (e : ℕ) (j : Fin K) : StronglyMeasurable[H e j] (d.t e j) := by
  unfold t assignmentEffect
  exact (((d.a_pre e j).mul (d.p_pre e j)).mul
    (stronglyMeasurable_const.sub (d.p_pre e j))).mul
      ((stronglyMeasurable_condExp.div (d.p_pre e j)).sub
        ((stronglyMeasurable_condExp.sub stronglyMeasurable_condExp).div
          (stronglyMeasurable_const.sub (d.p_pre e j))))

omit [IsProbabilityMeasure μ] in
lemma z_end (e : ℕ) (j : Fin K) : StronglyMeasurable[F (e + 1)] (d.z e j) :=
  (d.q_end e j).mul ((d.y_end e j).sub ((d.c_pre e j).mono (d.after e j)))

omit [IsProbabilityMeasure μ] in
lemma scalar_bounds (hη : 0 < η) (e : ℕ) (j : Fin K) (ω : Ω) :
    |d.q e j ω| ≤ 1 ∧ |d.w e j ω| ≤ 1 ∧
    η ^ 2 * d.a e j ω ^ 2 ≤ d.q e j ω ^ 2 ∧
    η ^ 2 * d.a e j ω ^ 2 ≤ d.w e j ω := by
  have hp := d.overlap e j ω
  have hp0 : 0 ≤ d.p e j ω := le_trans hη.le hp.1
  have hp1 : d.p e j ω ≤ 1 := by linarith
  have hb : η * η ≤ d.p e j ω * (1 - d.p e j ω) :=
    mul_le_mul hp.1 (by linarith) hη.le hp0
  have hsq1 : η ^ 2 ≤ d.p e j ω ^ 2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hp.1) (add_nonneg hp0 hη.le)]
  have hsq2 : η ^ 2 ≤ (1 - d.p e j ω) ^ 2 := by
    have h1 : 0 ≤ 1 - d.p e j ω - η := by linarith
    have h2 : 0 ≤ 1 - d.p e j ω + η := by linarith
    nlinarith [mul_nonneg h1 h2]
  rcases d.a_values e j ω with ha | ha | ha <;>
    rcases d.L_values e j ω with hL | hL
  all_goals norm_num [q, w, ha, hL, abs_le, abs_of_nonneg hp0,
    abs_of_nonneg (sub_nonneg.mpr hp1)]
  all_goals repeat' apply And.intro
  all_goals nlinarith [sq_nonneg (d.p e j ω)]

lemma moments (hη : 0 < η) (e : ℕ) (j : Fin K) :
    (μ[(fun ω => d.q e j ω ^ 2) | H e j] =ᵐ[μ] d.w e j) ∧
    (μ[d.z e j | H e j] =ᵐ[μ] d.t e j) ∧
    (d.t e j =ᵐ[μ] μ[(fun ω => d.q e j ω * d.y e j ω) | H e j]) := by
  have hH := (d.after e j).trans (F.le _)
  have ia := memLp_top_of_bound (μ := μ) ((d.a_pre e j).mono hH).aestronglyMeasurable 1
    (Filter.Eventually.of_forall (fun ω => by simpa using d.a_bound e j ω))
  have ip := memLp_top_of_bound (μ := μ) ((d.p_pre e j).mono hH).aestronglyMeasurable 1
    (Filter.Eventually.of_forall (fun ω => by simpa using d.p_bound hη e j ω))
  have ic := memLp_top_of_bound (μ := μ) ((d.c_pre e j).mono hH).aestronglyMeasurable C
    (Filter.Eventually.of_forall (fun ω => by simpa using d.c_bound e j ω))
  have iL := memLp_top_of_bound (μ := μ) ((d.L_end e j).mono (F.le _)).aestronglyMeasurable 1
    (Filter.Eventually.of_forall (fun ω => by simpa using d.L_bound e j ω))
  have iy := memLp_top_of_bound (μ := μ) ((d.y_end e j).mono (F.le _)).aestronglyMeasurable M
    (Filter.Eventually.of_forall (fun ω => by simpa using d.y_bound e j ω))
  have hm := binary_design_moments μ hH (d.a e j) (d.p e j) (d.c e j) (d.L e j) (d.y e j)
    (d.a_pre e j) (d.p_pre e j) (d.c_pre e j) ia ip ic iL iy (d.L_values e j)
    (by intro ω; have hp := d.overlap e j ω; constructor <;> linarith) (d.design e j)
  exact ⟨hm.2.1, hm.2.2.1, hm.2.2.1.symm.trans hm.2.2.2⟩

lemma t_bound (hη : 0 < η) (_hM : 0 ≤ M) (e : ℕ) (j : Fin K) :
    ∀ᵐ ω ∂μ, |d.t e j ω| ≤ M := by
  have hb : ∀ᵐ ω ∂μ, ‖d.q e j ω * d.y e j ω‖ ≤ M := by
    apply Filter.Eventually.of_forall
    intro ω
    rw [Real.norm_eq_abs, abs_mul]
    have hq := (d.scalar_bounds hη e j ω).1
    calc
      _ ≤ 1 * M := mul_le_mul hq (d.y_bound e j ω) (abs_nonneg _) (by norm_num)
      _ = _ := one_mul M
  have ht := ae_bdd_norm_condExp_of_ae_bdd_norm (m := H e j) hb
  filter_upwards [ht, (d.moments hη e j).2.2] with ω htω heω
  simpa [heω] using htω

lemma centered_zero {G : MeasurableSpace Ω} (hG : G ≤ mΩ)
    {f t : Ω → ℝ} (hi : Integrable f μ) (hit : Integrable t μ)
    (ht : StronglyMeasurable[G] t) (hm : μ[f | G] =ᵐ[μ] t) :
    μ[(fun ω => f ω - t ω) | G] =ᵐ[μ] 0 := by
  let : MeasurableSpace Ω := mΩ
  have hs := condExp_sub hi hit G
  have hself := condExp_of_stronglyMeasurable (μ := μ) hG ht hit
  filter_upwards [hs, hm] with ω hsω hmω
  change μ[(fun ω => f ω - t ω) | G] ω = μ[f | G] ω - μ[t | G] ω at hsω
  simpa [hself, hmω] using hsω

omit [IsProbabilityMeasure μ] in
lemma z_bound (hη : 0 < η) (_hM : 0 ≤ M) (_hC : 0 ≤ C)
    (e : ℕ) (j : Fin K) (ω : Ω) : |d.z e j ω| ≤ M + C := by
  rw [z, abs_mul]
  calc
    _ ≤ 1 * (M + C) := mul_le_mul (d.scalar_bounds hη e j ω).1
      ((abs_sub _ _).trans (add_le_add (d.y_bound e j ω) (d.c_bound e j ω)))
      (abs_nonneg _) (by norm_num)
    _ = _ := one_mul _

lemma errors_weak_law (weakLaw : BoundedMartingaleWeakLaw μ)
    (hη : 0 < η) (hM : 0 ≤ M) (hC : 0 ≤ C) :
    TendstoInMeasure μ (fun n ω =>
      (total d.z n ω - total d.t n ω) / n) atTop (fun _ => 0) ∧
    TendstoInMeasure μ (fun n ω =>
      (total (fun e j ω => d.q e j ω ^ 2) n ω - total d.w n ω) / n)
      atTop (fun _ => 0) := by
  have hiZ : ∀ e j, Integrable (d.z e j) μ := fun e j =>
    integrable_bounded μ ((d.z_end e j).mono (F.le _)).aestronglyMeasurable
      (Filter.Eventually.of_forall (d.z_bound hη hM hC e j))
  have hiT : ∀ e j, Integrable (d.t e j) μ := fun e j =>
    integrable_bounded μ ((d.t_pre e j).mono ((d.after e j).trans (F.le _))).aestronglyMeasurable
      (d.t_bound hη hM e j)
  have hiQ : ∀ e j, Integrable (fun ω => d.q e j ω ^ 2) μ := by
    intro e j
    apply integrable_bounded μ (((d.q_end e j).pow 2).mono (F.le _)).aestronglyMeasurable
    apply Filter.Eventually.of_forall
    intro ω
    have h := (d.scalar_bounds hη e j ω).1
    have h' := abs_le.mp h
    rw [abs_of_nonneg (sq_nonneg _)]
    exact (sq_le_one_iff_abs_le_one _).mpr h
  have hiW : ∀ e j, Integrable (d.w e j) μ := fun e j =>
    integrable_bounded μ ((d.w_pre e j).mono ((d.after e j).trans (F.le _))).aestronglyMeasurable
      (Filter.Eventually.of_forall (fun ω => (d.scalar_bounds hη e j ω).2.1))
  constructor
  · have hx := episode_blocking μ weakLaw F H (fun e j ω => d.z e j ω - d.t e j ω)
      (2 * M + C) d.before d.after
      (fun e j => (d.z_end e j).sub ((d.t_pre e j).mono (d.after e j)))
      (fun e j => centered_zero ((d.after e j).trans (F.le _))
        (hiZ e j) (hiT e j) (d.t_pre e j) (d.moments hη e j).2.1)
      (by
        intro e j
        filter_upwards [d.t_bound hη hM e j] with ω ht
        have hz := d.z_bound hη hM hC e j ω
        exact (abs_sub _ _).trans (by linarith))
    simpa only [total, sum_sub_distrib] using hx
  · have hx := episode_blocking μ weakLaw F H (fun e j ω => d.q e j ω ^ 2 - d.w e j ω)
      2 d.before d.after
      (fun e j => ((d.q_end e j).pow 2).sub ((d.w_pre e j).mono (d.after e j)))
      (fun e j => centered_zero ((d.after e j).trans (F.le _))
        (hiQ e j) (hiW e j) (d.w_pre e j) (d.moments hη e j).1)
      (by
        intro e j
        apply Filter.Eventually.of_forall
        intro ω
        have hb := d.scalar_bounds hη e j ω
        have hsq := (sq_le_one_iff_abs_le_one _).mpr hb.1
        have hsq' : |d.q e j ω ^ 2| ≤ 1 := by rwa [abs_of_nonneg (sq_nonneg _)]
        exact (abs_sub _ _).trans (by linarith [hb.2.1]))
    simpa only [total, sum_sub_distrib] using hx

omit [IsProbabilityMeasure μ] in
lemma total_lower_bounds (hη : 0 < η) (n : ℕ) (ω : Ω) :
    η ^ 2 * d.count n ω ≤ total (fun e j ω => d.q e j ω ^ 2) n ω ∧
    η ^ 2 * d.count n ω ≤ total d.w n ω := by
  unfold count total
  rw [mul_sum]
  constructor
  all_goals apply sum_le_sum; intro e he; rw [mul_sum]; apply sum_le_sum; intro j hj
  · exact (d.scalar_bounds hη e j ω).2.2.1
  · exact (d.scalar_bounds hη e j ω).2.2.2

omit [IsProbabilityMeasure μ] in
lemma count_cap (n : ℕ) (ω : Ω) : 0 ≤ d.count n ω ∧ d.count n ω ≤ (n : ℝ) * K := by
  unfold count total
  constructor
  · exact sum_nonneg (fun e _ => sum_nonneg (fun j _ => sq_nonneg _))
  · calc
      _ ≤ ∑ _e ∈ range n, ∑ _j : Fin K, (1 : ℝ) :=
        sum_le_sum (fun e _ => sum_le_sum (fun j _ =>
          (sq_le_one_iff_abs_le_one _).mpr (d.a_bound e j ω)))
      _ = _ := by simp

lemma total_target_bound (hη : 0 < η) (hM : 0 ≤ M) (n : ℕ) :
    ∀ᵐ ω ∂μ, |total d.t n ω| ≤ (n : ℝ) * (K * M) := by
  have hb : ∀ᵐ ω ∂μ, ∀ e j, |d.t e j ω| ≤ M :=
    ae_all_iff.mpr (fun e => ae_all_iff.mpr (d.t_bound hη hM e))
  filter_upwards [hb] with ω hω
  unfold total
  calc
    _ ≤ ∑ e ∈ range n, ∑ j : Fin K, |d.t e j ω| :=
      (abs_sum_le_sum_abs _ _).trans (sum_le_sum (fun e _ => abs_sum_le_sum_abs _ _))
    _ ≤ ∑ _e ∈ range n, ∑ _j : Fin K, M :=
      sum_le_sum (fun e _ => sum_le_sum (fun j _ => hω e j))
    _ = _ := by simp

/-- Corrected slope convergence, grouping by episode and using only the named MDS weak law.
The count hypothesis is the lower half of `N/n` being bounded away from zero and infinity
with probability tending to one; H2 already gives the upper bound. -/
theorem corrected_convergence (weakLaw : BoundedMartingaleWeakLaw μ)
    (hη : 0 < η) (hM : 0 ≤ M) (hC : 0 ≤ C)
    (ρ : ℝ) (hρ : 0 < ρ)
    (hcount : Tendsto (fun n => μ.real {ω | d.count n ω < ρ * n}) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => d.beta n ω - d.target n ω) atTop (fun _ => 0) := by
  let δ := η ^ 2 * ρ
  have hδ : 0 < δ := mul_pos (sq_pos_of_pos hη) hρ
  let good : ℕ → Set Ω := fun n => {ω | 0 < n ∧ ρ * n ≤ d.count n ω}
  have hg : Tendsto (fun n => μ.real (good n)ᶜ) atTop (𝓝 0) := by
    apply hcount.congr'
    filter_upwards [eventually_gt_atTop 0] with n hn
    congr 1
    ext ω
    simp [good, hn, not_le]
  obtain ⟨hx, hy⟩ := d.errors_weak_law weakLaw hη hM hC
  apply convergence_on_good_event μ _ _ _ good (1 / δ) ((K * M) / δ ^ 2)
    (by positivity) (by positivity) hx hy hg
  intro n
  filter_upwards [d.total_target_bound hη hM n] with ω hT hgood
  have hn : 0 < (n : ℝ) := by exact_mod_cast hgood.1
  have hlow := d.total_lower_bounds hη n ω
  have hB : δ ≤ total (fun e j ω => d.q e j ω ^ 2) n ω / n := by
    apply (le_div_iff₀ hn).mpr
    have := mul_le_mul_of_nonneg_left hgood.2 (sq_nonneg η)
    dsimp [δ]; nlinarith [hlow.1]
  have hW : δ ≤ total d.w n ω / n := by
    apply (le_div_iff₀ hn).mpr
    have := mul_le_mul_of_nonneg_left hgood.2 (sq_nonneg η)
    dsimp [δ]; nlinarith [hlow.2]
  have hT' : |total d.t n ω / n| ≤ (K : ℝ) * M := by
    rw [abs_div, abs_of_pos hn]
    exact (div_le_iff₀ hn).mpr (by nlinarith)
  have he := ratio_error_bound hδ hB hW hT'
    (A := total d.z n ω / n)
  have hcancel (a b : ℝ) : (a / n) / (b / n) = a / b := by
    exact div_div_div_cancel_right₀ (ne_of_gt hn) a b
  rw [hcancel, hcancel, ← sub_div, ← sub_div] at he
  simpa only [beta, target, div_eq_mul_inv, one_mul, mul_assoc, mul_comm, mul_left_comm] using he

end EpisodeDesign

end OranCausal.Slope

#print axioms OranCausal.Slope.half_propensity_identity
#print axioms OranCausal.Slope.half_propensity_target
#print axioms OranCausal.Slope.unbounded_centre_counterexample
#print axioms OranCausal.Slope.fairAssignments_independent
#print axioms OranCausal.Slope.fairAssignments_marginal
#print axioms OranCausal.Slope.counterexample_not_convergent
#print axioms OranCausal.Slope.zero_outcome_effect
#print axioms OranCausal.Slope.assignmentEffect_joint_moments
#print axioms OranCausal.Slope.binary_design_moments
#print axioms OranCausal.Slope.episode_blocking
#print axioms OranCausal.Slope.EpisodeDesign.corrected_convergence
