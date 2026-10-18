module

public import OranCausal.StepUp
public import OranCausal.Asymptotic

public section

open MeasureTheory Finset Filter
open scoped BigOperators Classical Topology

namespace OranCausal.FDRAsymptotic

open OranCausal.StepUp

/-- Tested hypotheses whose p-value meets the level-`k` threshold `t_j(k) = q w_j k / (m c_m)`.
An untested hypothesis carries the p-value `+∞`, so it is never a candidate. -/
noncomputable def candidates {m : ℕ} (q : ℝ) (w : Fin m → ℝ) (p : Fin m → ℝ)
    (T : Finset (Fin m)) (k : ℕ) : Finset (Fin m) :=
  T.filter fun j => p j ≤ threshold m q w j k

/-- Number of rejections `ρ = max {k ≤ m : #{j : p_j ≤ t_j(k)} ≥ k}` of the weighted BY step-up. -/
noncomputable def stepUpCount {m : ℕ} (q : ℝ) (w : Fin m → ℝ) (p : Fin m → ℝ)
    (T : Finset (Fin m)) : ℕ :=
  Nat.findGreatest (fun k => k ≤ (candidates q w p T k).card) m

/-- Rejection set of the weighted BY step-up: the tested `j` with `p_j ≤ t_j(ρ)`. -/
noncomputable def stepUp {m : ℕ} (q : ℝ) (w : Fin m → ℝ) (p : Fin m → ℝ)
    (T : Finset (Fin m)) : Finset (Fin m) :=
  candidates q w p T (stepUpCount q w p T)

/-- Step-up rejections at sample size `n`, with weights `w n` that may move with `n`. -/
noncomputable def rejections {Ω : Type*} {m : ℕ} (q : ℝ) (w : ℕ → Fin m → ℝ)
    (p : Fin m → ℕ → Ω → ℝ) (T : ℕ → Ω → Finset (Fin m)) (n : ℕ) (ω : Ω) :
    Finset (Fin m) :=
  stepUp q (w n) (fun j => p j n ω) (T n ω)

/-- Excess `F(t) - t` of the distribution function of `X` over the uniform one. -/
noncomputable def excess {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (X : Ω → ℝ)
    (t : ℝ) : ℝ :=
  μ.real {ω | X ω ≤ t} - t

/-- Slack `δ_n`: the largest positive excess of a true null over the grid `t_{j,i}`, `1 ≤ i ≤ m`. -/
noncomputable def slack {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {m : ℕ} (q : ℝ)
    (w : ℕ → Fin m → ℝ) (p : Fin m → ℕ → Ω → ℝ) (H0 : Finset (Fin m)) (n : ℕ) : ℝ :=
  ∑ j ∈ H0, ∑ i ∈ Icc 1 m, max 0 (excess μ (p j n) (threshold m q (w n) j i))

lemma stepUpCount_le {m : ℕ} (q : ℝ) (w : Fin m → ℝ) (p : Fin m → ℝ) (T : Finset (Fin m)) :
    stepUpCount q w p T ≤ m :=
  Nat.findGreatest_le m

lemma threshold_mono {m : ℕ} {q : ℝ} (hq : 0 ≤ q) {w : Fin m → ℝ} (hw : ∀ j, 0 ≤ w j)
    (j : Fin m) {k l : ℕ} (hkl : k ≤ l) : threshold m q w j k ≤ threshold m q w j l := by
  have hm : 1 ≤ m := Nat.one_le_iff_ne_zero.mpr fun h => (h ▸ j).elim0
  rw [threshold_eq, threshold_eq]
  exact mul_le_mul_of_nonneg_left (Nat.cast_le.mpr hkl)
    (div_nonneg (mul_nonneg hq (hw j)) (mul_nonneg (Nat.cast_nonneg m) (harmonic_pos hm).le))

lemma candidates_mono {m : ℕ} {q : ℝ} (hq : 0 ≤ q) {w : Fin m → ℝ} (hw : ∀ j, 0 ≤ w j)
    (p : Fin m → ℝ) (T : Finset (Fin m)) {k l : ℕ} (hkl : k ≤ l) :
    candidates q w p T k ⊆ candidates q w p T l := by
  intro j hj
  simp only [candidates, mem_filter] at hj ⊢
  exact ⟨hj.1, hj.2.trans (threshold_mono hq hw j hkl)⟩

/-- The step-up rejects exactly the tested `j` with `p_j ≤ t_j(ρ)`. -/
lemma mem_stepUp {m : ℕ} (q : ℝ) (w : Fin m → ℝ) (p : Fin m → ℝ) (T : Finset (Fin m))
    (j : Fin m) :
    j ∈ stepUp q w p T ↔ j ∈ T ∧ p j ≤ threshold m q w j (stepUpCount q w p T) := by
  simp [stepUp, candidates]

/-- Every level `k ≤ m` with at least `k` candidates is at most `ρ`. -/
lemma le_stepUpCount {m : ℕ} (q : ℝ) (w : Fin m → ℝ) (p : Fin m → ℝ) (T : Finset (Fin m))
    {k : ℕ} (hkm : k ≤ m) (hk : k ≤ (candidates q w p T k).card) :
    k ≤ stepUpCount q w p T :=
  Nat.le_findGreatest hkm hk

lemma stepUpCount_le_card {m : ℕ} (q : ℝ) (w : Fin m → ℝ) (p : Fin m → ℝ)
    (T : Finset (Fin m)) : stepUpCount q w p T ≤ (stepUp q w p T).card :=
  Nat.findGreatest_spec (P := fun k => k ≤ (candidates q w p T k).card) (Nat.zero_le m)
    (Nat.zero_le _)

/-- The step-up rejects exactly `ρ` hypotheses. -/
theorem card_stepUp {m : ℕ} {q : ℝ} (hq : 0 ≤ q) {w : Fin m → ℝ} (hw : ∀ j, 0 ≤ w j)
    (p : Fin m → ℝ) (T : Finset (Fin m)) :
    (stepUp q w p T).card = stepUpCount q w p T := by
  set ρ := stepUpCount q w p T with hρ
  set c := (stepUp q w p T).card with hc
  have hρc : ρ ≤ c := stepUpCount_le_card q w p T
  have hcm : c ≤ m := by simpa using card_le_univ (stepUp q w p T)
  by_contra hne
  have hlt : ρ < c := lt_of_le_of_ne hρc (Ne.symm hne)
  have hsub : stepUp q w p T ⊆ candidates q w p T c :=
    candidates_mono hq hw p T hlt.le
  have hcand : c ≤ (candidates q w p T c).card := card_le_card hsub
  have := le_stepUpCount q w p T hcm hcand
  omega

/-- Self-consistency of the step-up: every rejection is tested and meets `t_j(|S|)`. -/
theorem stepUp_selfConsistent {m : ℕ} {q : ℝ} (hq : 0 ≤ q) {w : Fin m → ℝ}
    (hw : ∀ j, 0 ≤ w j) (p : Fin m → ℝ) (T : Finset (Fin m)) (j : Fin m)
    (hj : j ∈ stepUp q w p T) :
    j ∈ T ∧ p j ≤ threshold m q w j (stepUp q w p T).card := by
  rw [card_stepUp hq hw p T]
  exact (mem_stepUp q w p T j).mp hj

lemma findGreatest_congr {P Q : ℕ → Prop} [DecidablePred P] [DecidablePred Q] (N : ℕ)
    (h : ∀ k ≤ N, (P k ↔ Q k)) : Nat.findGreatest P N = Nat.findGreatest Q N := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Nat.findGreatest_succ, Nat.findGreatest_succ, ih fun k hk => h k (by omega)]
    simp only [h (N + 1) le_rfl]

/-- The step-up depends on the p-values only through the events `p_j ≤ t_j(k)`, `k ≤ m`. -/
lemma stepUp_congr {m : ℕ} (q : ℝ) (w : Fin m → ℝ) {p p' : Fin m → ℝ} (T : Finset (Fin m))
    (h : ∀ j, ∀ k ≤ m, (p j ≤ threshold m q w j k ↔ p' j ≤ threshold m q w j k)) :
    stepUp q w p T = stepUp q w p' T := by
  have hc : ∀ k ≤ m, candidates q w p T k = candidates q w p' T k := fun k hk =>
    filter_congr fun j _ => h j k hk
  have hρ : stepUpCount q w p T = stepUpCount q w p' T :=
    findGreatest_congr m fun k hk => by rw [hc k hk]
  rw [stepUp, stepUp, hρ, hc _ (stepUpCount_le q w p' T)]

lemma measurable_stepUp {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (q : ℝ) (w : Fin m → ℝ)
    {p : Ω → Fin m → ℝ} (hp : ∀ j, Measurable fun ω => p ω j) {T : Ω → Finset (Fin m)}
    (hT : Measurable T) : Measurable fun ω => stepUp q w (p ω) (T ω) := by
  let sig : (Fin m → ℝ) → Fin m → Fin (m + 1) → Prop := fun x j k =>
    x j ≤ threshold m q w j k
  let Ψ : (Fin m → Fin (m + 1) → Prop) × Finset (Fin m) → Finset (Fin m) := fun x =>
    if h : ∃ x', sig x' = x.1 then stepUp q w h.choose x.2 else ∅
  have hsig : Measurable fun ω => sig (p ω) :=
    Measurable.of_eval fun j => Measurable.of_eval fun k =>
      measurableSet_setOfPred.mp (measurableSet_le (hp j) measurable_const)
  have heq : (fun ω => stepUp q w (p ω) (T ω)) = Ψ ∘ fun ω => (sig (p ω), T ω) := by
    funext ω
    have h : ∃ x', sig x' = sig (p ω) := ⟨p ω, rfl⟩
    simp only [Function.comp_apply, Ψ, h, dite_true]
    refine (stepUp_congr q w (T ω) fun j k hk => ?_).symm
    have := congrFun (congrFun h.choose_spec j) ⟨k, Nat.lt_succ_of_le hk⟩
    exact Iff.of_eq this
  rw [heq]
  exact (measurable_of_countable Ψ).comp (hsig.prodMk hT)

lemma slack_nonneg {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {m : ℕ} (q : ℝ)
    (w : ℕ → Fin m → ℝ) (p : Fin m → ℕ → Ω → ℝ) (H0 : Finset (Fin m)) (n : ℕ) :
    0 ≤ slack μ q w p H0 n :=
  sum_nonneg fun _ _ => sum_nonneg fun _ _ => le_max_left _ _

lemma excess_le_slack {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {m : ℕ} (q : ℝ)
    (w : ℕ → Fin m → ℝ) (p : Fin m → ℕ → Ω → ℝ) {H0 : Finset (Fin m)} (n : ℕ) {j : Fin m}
    (hj : j ∈ H0) {i : ℕ} (hi : 1 ≤ i) (him : i ≤ m) :
    excess μ (p j n) (threshold m q (w n) j i) ≤ slack μ q w p H0 n := by
  have hin : i ∈ Icc 1 m := mem_Icc.mpr ⟨hi, him⟩
  calc
    _ ≤ max 0 (excess μ (p j n) (threshold m q (w n) j i)) := le_max_right _ _
    _ ≤ ∑ i ∈ Icc 1 m, max 0 (excess μ (p j n) (threshold m q (w n) j i)) :=
      single_le_sum (f := fun i => max 0 (excess μ (p j n) (threshold m q (w n) j i)))
        (fun _ _ => le_max_left _ _) hin
    _ ≤ slack μ q w p H0 n :=
      single_le_sum (f := fun j => ∑ i ∈ Icc 1 m,
          max 0 (excess μ (p j n) (threshold m q (w n) j i)))
        (fun _ _ => sum_nonneg fun _ _ => le_max_left _ _) hj

/-- Finite weighted BY bound at every sample size, whatever the dependence: the FDR of the
step-up is at most `(q/m) Σ_{H0} w_{n,j} + |H0| δ_n`. -/
theorem fdr_le_slack {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {m : ℕ} (hm : 1 ≤ m) {q : ℝ} (hq : 0 ≤ q) (w : ℕ → Fin m → ℝ) (hw : ∀ n j, 0 ≤ w n j)
    (H0 : Finset (Fin m)) (p : Fin m → ℕ → Ω → ℝ) (hp : ∀ j n, Measurable (p j n))
    (T : ℕ → Ω → Finset (Fin m)) (hT : ∀ n, Measurable (T n)) (n : ℕ) :
    fdr μ (rejections q w p T n) H0 ≤
      (q / m) * (∑ j ∈ H0, w n j) + H0.card * slack μ q w p H0 n := by
  refine fdr_le_of_selfConsistent μ hm q _ hq (w n) (hw n) H0 (fun j => p j n) (hp · n)
    (rejections q w p T n) (measurable_stepUp q (w n) (fun j => hp j n) (hT n)) ?_ ?_
  · intro ω j hj
    exact (stepUp_selfConsistent hq (hw n) _ _ j hj).2
  · intro j hj i hi him
    have := excess_le_slack μ q w p n hj hi him
    simp only [excess] at this
    linarith

/-- Uniform slack of a plus-one p-value on the lattice `k / (B + 1)`, `0 ≤ k ≤ B + 1`. -/
noncomputable def latticeSlack {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (raw : ℕ → Ω → ℝ) (B n : ℕ) : ℝ :=
  ∑ k ∈ range (B + 2), max 0 (μ.real {ω | raw n ω ≤ k / (B + 1)} - k / (B + 1))

lemma latticeSlack_nonneg {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (raw : ℕ → Ω → ℝ) (B n : ℕ) : 0 ≤ latticeSlack μ raw B n :=
  sum_nonneg fun _ _ => le_max_left _ _

/-- A plus-one p-value below `t` is below the lattice point `⌊t (B + 1)⌋ / (B + 1)`. -/
lemma plusOne_le_lattice (B Q : ℕ) {t : ℝ} (h : (1 + (Q : ℝ)) / (B + 1) ≤ t) :
    (1 + (Q : ℝ)) / (B + 1) ≤ (⌊t * (B + 1)⌋₊ : ℝ) / (B + 1) := by
  have hp : (0 : ℝ) < B + 1 := by positivity
  rw [div_le_iff₀ hp] at h
  have hk : 1 + Q ≤ ⌊t * (B + 1)⌋₊ := Nat.le_floor (by push_cast; linarith)
  exact div_le_div_of_nonneg_right (by exact_mod_cast hk) hp.le

/-- Uniformity in `t ≥ 0`: a p-value at least the plus-one value `(1 + Q) / (B + 1)` has excess
`F(t) - t` at most the lattice slack, for every `t ≥ 0` at once. -/
theorem excess_le_latticeSlack {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (B : ℕ) (Q : ℕ → Ω → ℕ) (rep : ℕ → Ω → ℝ)
    (hrep : ∀ n ω, (1 + (Q n ω : ℝ)) / (B + 1) ≤ rep n ω) (n : ℕ) {t : ℝ} (ht : 0 ≤ t) :
    excess μ (rep n) t ≤
      latticeSlack μ (fun n ω => (1 + (Q n ω : ℝ)) / (B + 1)) B n := by
  have hp : (0 : ℝ) < B + 1 := by positivity
  have hS := latticeSlack_nonneg μ (fun n ω => (1 + (Q n ω : ℝ)) / (B + 1)) B n
  set k := ⌊t * (B + 1)⌋₊ with hk
  unfold excess
  by_cases hkB : k ≤ B + 1
  · have hsub : {ω | rep n ω ≤ t} ⊆ {ω | (1 + (Q n ω : ℝ)) / (B + 1) ≤ k / (B + 1)} :=
      fun ω hω => plusOne_le_lattice B (Q n ω) ((hrep n ω).trans hω)
    have hkt : (k : ℝ) / (B + 1) ≤ t :=
      (div_le_iff₀ hp).mpr (Nat.floor_le (mul_nonneg ht hp.le))
    have hterm : μ.real {ω | (1 + (Q n ω : ℝ)) / (B + 1) ≤ k / (B + 1)} - k / (B + 1) ≤
        latticeSlack μ (fun n ω => (1 + (Q n ω : ℝ)) / (B + 1)) B n :=
      (le_max_right _ _).trans (single_le_sum
        (f := fun k : ℕ => max 0 (μ.real {ω | (1 + (Q n ω : ℝ)) / (B + 1) ≤ k / (B + 1)} -
          k / (B + 1))) (fun _ _ => le_max_left _ _) (mem_range.mpr (by omega)))
    have := measureReal_mono (μ := μ) hsub
    linarith
  · have hk2 : ((B + 2 : ℕ) : ℝ) ≤ t * (B + 1) :=
      (Nat.cast_le.mpr (by omega)).trans (Nat.floor_le (mul_nonneg ht hp.le))
    push_cast at hk2
    have h1 : 1 ≤ t := by nlinarith
    have := measureReal_le_one (μ := μ) (s := {ω | rep n ω ≤ t})
    linarith

lemma tendsto_posPart_of_limsup_le {a : ℕ → ℝ} (hb : ∀ n, a n ≤ 1) {c : ℝ}
    (h : limsup a atTop ≤ c) : Tendsto (fun n => max 0 (a n - c)) atTop (𝓝 0) := by
  refine tendsto_order.2 ⟨fun b hb0 => Eventually.of_forall fun n =>
    lt_of_lt_of_le hb0 (le_max_left _ _), fun b hb0 => ?_⟩
  filter_upwards [eventually_lt_add_pos_of_limsup_le (isBoundedUnder_of ⟨1, hb⟩) h hb0]
    with n hn
  exact max_lt hb0 (by linarith)

/-- Task 07's pointwise bound `limsup_n Pr(raw_n ≤ t) ≤ gridCDF B t`, for fixed `B`, makes the
lattice slack vanish, hence `limsup_n sup_{t ≥ 0} (F_n(t) - t) ≤ 0`. -/
theorem tendsto_latticeSlack {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (raw : ℕ → Ω → ℝ) (B : ℕ)
    (h : ∀ t ∈ Set.Icc (0 : ℝ) 1,
      limsup (fun n => μ.real {ω | raw n ω ≤ t}) atTop ≤ Asymptotic.gridCDF B t) :
    Tendsto (latticeSlack μ raw B) atTop (𝓝 0) := by
  have hz : (0 : ℝ) = ∑ k ∈ range (B + 2), (0 : ℝ) := by simp
  rw [hz]
  refine tendsto_finsetSum _ fun k hk => ?_
  have hp : (0 : ℝ) < B + 1 := by positivity
  have hkB : (k : ℝ) ≤ B + 1 := by
    have := mem_range.mp hk
    exact_mod_cast (by omega : k ≤ B + 1)
  have hI : (k : ℝ) / (B + 1) ∈ Set.Icc (0 : ℝ) 1 :=
    ⟨by positivity, (div_le_one hp).mpr hkB⟩
  have hl := h _ hI
  exact tendsto_posPart_of_limsup_le (fun n => measureReal_le_one)
    (hl.trans (Asymptotic.gridCDF_le B _ hI.1))

section TaskSeven

open ProbabilityTheory OranCausal.Asymptotic

/-- Per-null input from task 07: a plus-one p-value `(1 + Q_n) / (B + 1)` for a fixed `B` whose
rejection probabilities converge to `gridCDF B t` for every `t ∈ [0, 1]` (the first conclusion of
`conditional_binomial_rejection_limit`), and a used p-value `p_n` at least that value. -/
def NullLimit {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (p : ℕ → Ω → ℝ) : Prop :=
  ∃ (B : ℕ) (Q : ℕ → Ω → ℕ),
    (∀ t ∈ Set.Icc (0 : ℝ) 1, Tendsto (fun n => μ.real {ω | (1 + (Q n ω : ℝ)) / (B + 1) ≤ t})
      atTop (𝓝 (gridCDF B t))) ∧
    ∀ n ω, (1 + (Q n ω : ℝ)) / (B + 1) ≤ p n ω

/-- A true null with the task 07 input has `limsup_n sup_{t ≥ 0} (F_n(t) - t) ≤ 0`, in the form
of a vanishing nonnegative bound `E_n` on every excess at `t ≥ 0`. -/
theorem uniform_excess_of_nullLimit {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {p : ℕ → Ω → ℝ} (h : NullLimit μ p) :
    ∃ E : ℕ → ℝ, Tendsto E atTop (𝓝 0) ∧
      ∀ n, 0 ≤ E n ∧ ∀ t, 0 ≤ t → excess μ (p n) t ≤ E n := by
  obtain ⟨B, Q, hlim, hrep⟩ := h
  exact ⟨latticeSlack μ (fun n ω => (1 + (Q n ω : ℝ)) / (B + 1)) B,
    tendsto_latticeSlack μ _ B fun t ht => (hlim t ht).limsup_eq.le,
    fun n => ⟨latticeSlack_nonneg _ _ _ _, fun _ ht => excess_le_latticeSlack μ B Q p hrep n ht⟩⟩

/-- The task 07 input from the hypotheses of `conditional_binomial_rejection_limit`. -/
theorem nullLimit_of_binomial {Ω : Type*} (G : ℕ → MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : ∀ n, G n ≤ mΩ) (Q : ℕ → Ω → ℕ) (hQ : ∀ n, Measurable (Q n))
    (P : ℕ → Ω → ℝ) (hP : ∀ n ω, P n ω ∈ Set.Icc (0 : ℝ) 1)
    (hweak : TendstoInDistribution P atTop id (fun _ => μ) uniformLaw)
    (B : ℕ) (hbound : ∀ n ω, Q n ω ≤ B)
    (hlaw : ∀ n j, j ≤ B →
      μ[(Q n ⁻¹' {j}).indicator (fun _ => (1 : ℝ)) | G n] =ᵐ[μ]
        (fun ω => binomialMass B j (P n ω)))
    (p : ℕ → Ω → ℝ) (hp : ∀ n ω, (1 + (Q n ω : ℝ)) / (B + 1) ≤ p n ω) :
    NullLimit μ p :=
  ⟨B, Q, fun t ht =>
    (conditional_binomial_rejection_limit G μ hG Q hQ P hP hweak B hbound hlaw t ht).1, hp⟩

/-- The task 07 input from the hypotheses of `asymptotic_rejection_bound` (score, redraw laws,
conditional binomial count, conservative reporting). -/
theorem nullLimit_of_pmrt {Ω : Type*} (G : ℕ → MeasurableSpace Ω)
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
    (hconservative : ∀ n ω, (1 + (Q n ω : ℝ)) / (B + 1) ≤ reported n ω) :
    NullLimit μ reported := by
  have hweak := redraw_tail_uniform μ ρ hc score (fun n ω => cdf (redrawLaw n ω) (score n ω)) _
    hscore hP (fun n => Eventually.of_forall fun ω =>
      le_trans (cdf_error_bound _ _ _) (le_abs_self _)) herror
  exact nullLimit_of_binomial G μ hG Q hQ _ (fun n ω => ⟨cdf_nonneg _ _, cdf_le_one _ _⟩) hweak
    B hbound hlaw reported hconservative

/-- Two-sided p-value `p`: score `-|z|`, with `z` asymptotically standard normal. -/
theorem nullLimit_of_twoSided_pmrt {Ω : Type*} (G : ℕ → MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : ∀ n, G n ≤ mΩ) (Z : ℕ → Ω → ℝ)
    (hZ : TendstoInDistribution Z atTop id (fun _ => μ) standardNormal)
    (redrawLaw : ℕ → Ω → Measure ℝ)
    (herror : TendstoInMeasure μ (fun n ω => cdfDistanceTo (redrawLaw n ω)
      (standardNormal.map (fun x : ℝ => -‖x‖))) atTop (fun _ => 0))
    (hP : ∀ n, AEMeasurable (fun ω => cdf (redrawLaw n ω) (-‖Z n ω‖)) μ)
    (Q : ℕ → Ω → ℕ) (hQ : ∀ n, Measurable (Q n)) (B : ℕ)
    (hbound : ∀ n ω, Q n ω ≤ B)
    (hlaw : ∀ n j, j ≤ B →
      μ[(Q n ⁻¹' {j}).indicator (fun _ => (1 : ℝ)) | G n] =ᵐ[μ]
        (fun ω => binomialMass B j (cdf (redrawLaw n ω) (-‖Z n ω‖))))
    (reported : ℕ → Ω → ℝ)
    (hconservative : ∀ n ω, (1 + (Q n ω : ℝ)) / (B + 1) ≤ reported n ω) :
    NullLimit μ reported :=
  nullLimit_of_pmrt G μ hG _ lower_absolute_normal_continuous (fun n ω => -‖Z n ω‖)
    (continuous_score_limit μ Z hZ (fun x : ℝ => -‖x‖) continuous_norm.neg) redrawLaw herror hP
    Q hQ B hbound hlaw
    reported hconservative

/-- One-sided p-value `p_d` for a direction `d` fixed in advance and an orientation `o` fixed
across `n`: score `-d o z`, with the stable normal limit of `z`. -/
theorem nullLimit_of_directed_pmrt {Ω : Type*} (G : ℕ → MeasurableSpace Ω)
    [mΩ : MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (hG : ∀ n, G n ≤ mΩ) (Z : ℕ → Ω → ℝ) (hZ : ∀ n, AEMeasurable (Z n) μ)
    (hstable : StableGaussianLimit μ Z (fun _ => 1))
    (d : ℝ) (hd : d = 1 ∨ d = -1) (o : Ω → ℝ) (ho : Measurable o)
    (hos : ∀ ω, o ω = 1 ∨ o ω = -1)
    (redrawLaw : ℕ → Ω → Measure ℝ)
    (herror : TendstoInMeasure μ (fun n ω => cdfDistanceTo (redrawLaw n ω) standardNormal)
      atTop (fun _ => 0))
    (hP : ∀ n, AEMeasurable (fun ω => cdf (redrawLaw n ω) ((-d * o ω) * Z n ω)) μ)
    (Q : ℕ → Ω → ℕ) (hQ : ∀ n, Measurable (Q n)) (B : ℕ)
    (hbound : ∀ n ω, Q n ω ≤ B)
    (hlaw : ∀ n j, j ≤ B →
      μ[(Q n ⁻¹' {j}).indicator (fun _ => (1 : ℝ)) | G n] =ᵐ[μ]
        (fun ω => binomialMass B j (cdf (redrawLaw n ω) ((-d * o ω) * Z n ω))))
    (reported : ℕ → Ω → ℝ)
    (hconservative : ∀ n ω, (1 + (Q n ω : ℝ)) / (B + 1) ≤ reported n ω) :
    NullLimit μ reported :=
  nullLimit_of_pmrt G μ hG standardNormal (continuous_cdf_of_no_atoms _)
    (fun n ω => (-d * o ω) * Z n ω) (fixed_direction_score_limit μ Z hZ hstable d hd o ho hos)
    redrawLaw herror hP Q hQ B hbound hlaw reported hconservative

end TaskSeven

lemma threshold_nonneg {m : ℕ} {q : ℝ} (hq : 0 ≤ q) {w : Fin m → ℝ} (hw : ∀ j, 0 ≤ w j)
    (j : Fin m) (i : ℕ) : 0 ≤ threshold m q w j i := by
  have h0 : threshold m q w j 0 = 0 := by rw [threshold_eq]; simp
  exact h0 ▸ threshold_mono hq hw j (Nat.zero_le i)

/-- The slack vanishes even though the thresholds `t_{j,i} = q w_{n,j} i / (m c_m)` move with
`n`: each true null's excess is bounded uniformly in `t ≥ 0`. -/
theorem tendsto_slack {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {m : ℕ} {q : ℝ} (hq : 0 ≤ q) (w : ℕ → Fin m → ℝ) (hw : ∀ n j, 0 ≤ w n j)
    (p : Fin m → ℕ → Ω → ℝ) (H0 : Finset (Fin m)) (hnull : ∀ j ∈ H0, NullLimit μ (p j)) :
    Tendsto (slack μ q w p H0) atTop (𝓝 0) := by
  have hE : ∀ j ∈ H0, ∃ E : ℕ → ℝ, Tendsto E atTop (𝓝 0) ∧
      ∀ n, 0 ≤ E n ∧ ∀ t, 0 ≤ t → excess μ (p j n) t ≤ E n :=
    fun j hj => uniform_excess_of_nullLimit (hnull j hj)
  choose! E hElim hEb using hE
  have hup : Tendsto (fun n => ∑ j ∈ H0, ∑ _i ∈ Icc 1 m, E j n) atTop (𝓝 0) := by
    have hz : (0 : ℝ) = ∑ j ∈ H0, ∑ _i ∈ Icc 1 m, (0 : ℝ) := by simp
    rw [hz]
    exact tendsto_finsetSum _ fun j hj => tendsto_finsetSum _ fun _ _ => hElim j hj
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hup
    (slack_nonneg μ q w p H0) fun n => ?_
  refine sum_le_sum fun j hj => sum_le_sum fun i _ => max_le (hEb j hj n).1 ?_
  exact (hEb j hj n).2 _ (threshold_nonneg hq (hw n) j i)

/-- Limsup substitution: under any dependence, `FDR_n ≤ (q/m) Σ_{H0} w_{n,j} + o(1)` for the
step-up with weights that may move with `n`. -/
theorem fdr_le_add_vanishing {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] {m : ℕ} (hm : 1 ≤ m) {q : ℝ} (hq : 0 ≤ q) (w : ℕ → Fin m → ℝ)
    (hw : ∀ n j, 0 ≤ w n j) (H0 : Finset (Fin m)) (p : Fin m → ℕ → Ω → ℝ)
    (hp : ∀ j n, Measurable (p j n)) (T : ℕ → Ω → Finset (Fin m))
    (hT : ∀ n, Measurable (T n)) (hnull : ∀ j ∈ H0, NullLimit μ (p j)) :
    ∃ δ : ℕ → ℝ, Tendsto δ atTop (𝓝 0) ∧ (∀ n, 0 ≤ δ n) ∧
      ∀ n, fdr μ (rejections q w p T n) H0 ≤ (q / m) * (∑ j ∈ H0, w n j) + δ n := by
  refine ⟨fun n => H0.card * slack μ q w p H0 n, ?_, fun n =>
    mul_nonneg (Nat.cast_nonneg _) (slack_nonneg μ q w p H0 n),
    fdr_le_slack μ hm hq w hw H0 p hp T hT⟩
  simpa using (tendsto_slack μ hq w hw p H0 hnull).const_mul (H0.card : ℝ)

/-- An untested hypothesis (p-value `+∞`) is never rejected. -/
lemma not_mem_rejections_of_untested {Ω : Type*} {m : ℕ} (q : ℝ) (w : ℕ → Fin m → ℝ)
    (p : Fin m → ℕ → Ω → ℝ) (T : ℕ → Ω → Finset (Fin m)) (n : ℕ) (ω : Ω) {j : Fin m}
    (hj : j ∉ T n ω) : j ∉ rejections q w p T n ω :=
  fun h => hj ((mem_stepUp q (w n) _ (T n ω) j).mp h).1

/-- With `w ≥ 0` and `Σ_j w_j = m`, the level `(q/m) Σ_{H0} w_j` lies in `[0, q]`. -/
lemma nullLevel_bounds {m : ℕ} (hm : 1 ≤ m) {q : ℝ} (hq : 0 ≤ q) {w : Fin m → ℝ}
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = m) (H0 : Finset (Fin m)) :
    0 ≤ (q / m) * (∑ j ∈ H0, w j) ∧ (q / m) * (∑ j ∈ H0, w j) ≤ q := by
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have hqm : 0 ≤ q / m := div_nonneg hq hm0.le
  refine ⟨mul_nonneg hqm (sum_nonneg fun j _ => hw j), ?_⟩
  have hsub : ∑ j ∈ H0, w j ≤ ∑ j, w j :=
    sum_le_sum_of_subset_of_nonneg (subset_univ H0) fun j _ _ => hw j
  calc
    _ ≤ (q / m) * m := mul_le_mul_of_nonneg_left (hsum ▸ hsub) hqm
    _ = q := by field_simp

/-- Corollary `cor:fdr` with weights that move with `n` (the projected weights): under any
dependence, `limsup FDR_n ≤ limsup_n (q/m) Σ_{H0} w_{n,j} ≤ q`. -/
theorem limsup_fdr_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] {m : ℕ} (hm : 1 ≤ m) {q : ℝ} (hq : 0 ≤ q) (w : ℕ → Fin m → ℝ)
    (hw : ∀ n j, 0 ≤ w n j) (hsum : ∀ n, ∑ j, w n j = m) (H0 : Finset (Fin m))
    (p : Fin m → ℕ → Ω → ℝ) (hp : ∀ j n, Measurable (p j n)) (T : ℕ → Ω → Finset (Fin m))
    (hT : ∀ n, Measurable (T n)) (hnull : ∀ j ∈ H0, NullLimit μ (p j)) :
    limsup (fun n => fdr μ (rejections q w p T n) H0) atTop ≤
        limsup (fun n => (q / m) * (∑ j ∈ H0, w n j)) atTop ∧
      limsup (fun n => (q / m) * (∑ j ∈ H0, w n j)) atTop ≤ q := by
  obtain ⟨δ, hδ, -, hle⟩ := fdr_le_add_vanishing μ hm hq w hw H0 p hp T hT hnull
  have ha : ∀ n, 0 ≤ (q / m) * (∑ j ∈ H0, w n j) ∧ (q / m) * (∑ j ∈ H0, w n j) ≤ q :=
    fun n => nullLevel_bounds hm hq (hw n) (hsum n) H0
  have haB : IsBoundedUnder (· ≤ ·) atTop fun n => (q / m) * (∑ j ∈ H0, w n j) :=
    isBoundedUnder_of ⟨q, fun n => (ha n).2⟩
  refine ⟨le_of_forall_pos_le_add fun ε hε => ?_,
    limsup_le_of_le (isCoboundedUnder_le_of_le atTop fun n => (ha n).1)
      (Eventually.of_forall fun n => (ha n).2)⟩
  have h1 := eventually_lt_add_pos_of_limsup_le haB le_rfl (half_pos hε)
  have h2 : ∀ᶠ n in atTop, δ n < ε / 2 := (tendsto_order.1 hδ).2 _ (half_pos hε)
  refine limsup_le_of_le (isCoboundedUnder_le_of_le atTop fun n => fdr_nonneg μ _ H0) ?_
  filter_upwards [h1, h2] with n h1 h2
  linarith [hle n]

/-- Convergent weights `w_n → w`: `limsup FDR_n ≤ (q/m) Σ_{H0} w_j ≤ q`, the paper's middle
term exactly. -/
theorem limsup_fdr_le_of_tendsto {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] {m : ℕ} (hm : 1 ≤ m) {q : ℝ} (hq : 0 ≤ q) (w : ℕ → Fin m → ℝ)
    (hw : ∀ n j, 0 ≤ w n j) (hsum : ∀ n, ∑ j, w n j = m) (w₀ : Fin m → ℝ)
    (hconv : ∀ j, Tendsto (fun n => w n j) atTop (𝓝 (w₀ j))) (H0 : Finset (Fin m))
    (p : Fin m → ℕ → Ω → ℝ) (hp : ∀ j n, Measurable (p j n)) (T : ℕ → Ω → Finset (Fin m))
    (hT : ∀ n, Measurable (T n)) (hnull : ∀ j ∈ H0, NullLimit μ (p j)) :
    limsup (fun n => fdr μ (rejections q w p T n) H0) atTop ≤ (q / m) * (∑ j ∈ H0, w₀ j) ∧
      (q / m) * (∑ j ∈ H0, w₀ j) ≤ q := by
  have hlim : Tendsto (fun n => (q / m) * (∑ j ∈ H0, w n j)) atTop
      (𝓝 ((q / m) * (∑ j ∈ H0, w₀ j))) :=
    (tendsto_finsetSum H0 fun j _ => hconv j).const_mul _
  have hw₀ : ∀ j, 0 ≤ w₀ j := fun j => ge_of_tendsto' (hconv j) fun n => hw n j
  have hsum₀ : ∑ j, w₀ j = m :=
    tendsto_nhds_unique (tendsto_finsetSum univ fun j _ => hconv j)
      (tendsto_const_nhds.congr fun n => (hsum n).symm)
  have h := (limsup_fdr_le μ hm hq w hw hsum H0 p hp T hT hnull).1
  rw [hlim.limsup_eq] at h
  exact ⟨h, (nullLevel_bounds hm hq hw₀ hsum₀ H0).2⟩

/-- Corollary `cor:fdr` with weights fixed in `n`, as stated in the paper:
`limsup FDR_n ≤ (q/m) Σ_{H0} w_j ≤ q` whatever the dependence between the p-values. -/
theorem limsup_fdr_le_const {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] {m : ℕ} (hm : 1 ≤ m) {q : ℝ} (hq : 0 ≤ q) (w : Fin m → ℝ)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = m) (H0 : Finset (Fin m))
    (p : Fin m → ℕ → Ω → ℝ) (hp : ∀ j n, Measurable (p j n)) (T : ℕ → Ω → Finset (Fin m))
    (hT : ∀ n, Measurable (T n)) (hnull : ∀ j ∈ H0, NullLimit μ (p j)) :
    limsup (fun n => fdr μ (rejections q (fun _ => w) p T n) H0) atTop ≤
        (q / m) * (∑ j ∈ H0, w j) ∧
      (q / m) * (∑ j ∈ H0, w j) ≤ q :=
  limsup_fdr_le_of_tendsto μ hm hq (fun _ => w) (fun _ => hw) (fun _ => hsum) w
    (fun _ => tendsto_const_nhds) H0 p hp T hT hnull

end OranCausal.FDRAsymptotic

#print axioms OranCausal.FDRAsymptotic.card_stepUp
#print axioms OranCausal.FDRAsymptotic.stepUp_selfConsistent
#print axioms OranCausal.FDRAsymptotic.fdr_le_slack
#print axioms OranCausal.FDRAsymptotic.excess_le_latticeSlack
#print axioms OranCausal.FDRAsymptotic.tendsto_latticeSlack
#print axioms OranCausal.FDRAsymptotic.uniform_excess_of_nullLimit
#print axioms OranCausal.FDRAsymptotic.nullLimit_of_pmrt
#print axioms OranCausal.FDRAsymptotic.nullLimit_of_twoSided_pmrt
#print axioms OranCausal.FDRAsymptotic.nullLimit_of_directed_pmrt
#print axioms OranCausal.FDRAsymptotic.tendsto_slack
#print axioms OranCausal.FDRAsymptotic.fdr_le_add_vanishing
#print axioms OranCausal.FDRAsymptotic.limsup_fdr_le
#print axioms OranCausal.FDRAsymptotic.limsup_fdr_le_of_tendsto
#print axioms OranCausal.FDRAsymptotic.limsup_fdr_le_const
