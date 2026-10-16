"""Constants fixed by the Study 4 protocol and by the frozen analysis definition."""

ANALYSIS_VERSION = "xm-eval-analysis/5"
ALPHA = 0.05
VALID_UB = 0.075
Q_BY = 0.05
CONFORMAL_LEVEL = 0.05
BOOT_REPS = 2000
BOOT_SEED = 20261002
MIN_FLAG_SEEDS = 10
MIN_COVERAGE = 0.9
PRIMARY_KAPPA = 0.25
SWEEP_KAPPAS = (0.125, 0.25, 0.5)
SWEEP_N = 1000
FOCAL = "pmrt_eq"
NO_DESIGN_REGIMES = ("R4",)
FMAX_NSIM = 2000
FMAX_SEED = 20261003
FMAX_Q = 0.95
FROZEN_MARK = "FROZEN: yes"
DEFAULT_E4_LAMS = {
    "R1": [1.0],
    "R2": [1.0],
    "R3": [0.0, 0.5, 1.0, 1.5],
    "R4": [0.0, 0.5, 1.0, 1.5],
}
RATE_KEYS = ("null_raw", "null_decl", "plac_raw", "plac_decl", "conf_raw", "conf_decl")
LIKE_RAW_KEYS = ("null_decl", "plac_decl", "conf_decl")
COUNTED = ("ok", "under_seeded")
P_KEYS = ("null_raw", "plac_raw")
C3_P_KEYS = ("plac_raw", "conf_raw")
C3_TAU_KEYS = ("conf_decl",)
EQ_TAU_KEYS = ("null_decl",)
S_FLOOR = 40
S_CAP = 60
T1_NS = (500, 1000, 4000)
