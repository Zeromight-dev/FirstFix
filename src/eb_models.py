"""
FixFirst - Phase 2: noise-aware models trained from scratch
Run from the FirstFix folder:   python src/eb_models.py

Trains, on the PROVIDED data only (no APIs, no pretrained models):
  M-A  Poisson GLM via IRLS (written from scratch)  - do location attributes explain incident counts?
  M-B  Empirical-Bayes Gamma-Poisson (prior fitted by maximum likelihood, written from scratch)
       -> shrunken incident rate per location, 90% credible interval, rank interval, P(top-50)
Also runs: Poisson dispersion test, time-based validation (fit on Jan-Jun, test on Jul-Dec).

Writes: outputs/models/eb_location_table.csv, outputs/models/eb_summary.json,
        models/eb_model.joblib, outputs/models/eb_report.txt   (paste eb_report.txt back)
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTM = BASE / "outputs" / "models"
MOD = BASE / "models"
OUTM.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s); self.f.write(s)

    def flush(self):
        self.o.flush(); self.f.flush()


sys.stdout = Tee(OUTM / "eb_report.txt")
RNG = np.random.default_rng(42)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ------------------------------------------------------------------ data
loc = pd.read_csv(DATA / "locations.csv")
inc = pd.read_csv(DATA / "incidents.csv")
inc["ts"] = pd.to_datetime(inc["timestamp"])
SPLIT = pd.Timestamp("2025-07-01")
T0, T1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
E1 = (SPLIT - T0).days / 365.0       # exposure (years) of first half
E2 = (T1 - SPLIT).days / 365.0       # exposure of second half
EF = (T1 - T0).days / 365.0

ids = loc.location_id.values
y_h1 = inc[inc.ts < SPLIT].groupby("location_id").size().reindex(ids, fill_value=0).values.astype(float)
y_h2 = inc[inc.ts >= SPLIT].groupby("location_id").size().reindex(ids, fill_value=0).values.astype(float)
y_all = y_h1 + y_h2
print(f"Locations: {len(ids)}   incidents: {int(y_all.sum())}   mean/location/yr: {y_all.mean() / EF:.2f}")
print(f"Exposure years  H1={E1:.3f}  H2={E2:.3f}  full={EF:.3f}")


# ------------------------------------------------------------------ shared metrics
def pois_dev(y, mu):
    mu = np.clip(mu, 1e-6, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * np.sum(t - (y - mu))


def mean_dev(y, mu):
    return pois_dev(y, mu) / len(y)


# ------------------------------------------------------------------ 0. dispersion test
section("0. IS THERE ANY LOCATION-TO-LOCATION VARIATION BEYOND PURE CHANCE?")
m = y_all.mean()
pearson = np.sum((y_all - m) ** 2 / m)
dfree = len(y_all) - 1
p_disp = stats.chi2.sf(pearson, dfree)
print(f"Pearson dispersion statistic = {pearson:.1f} on {dfree} df  (ratio {pearson / dfree:.3f}; Poisson null = 1.0)")
print(f"p-value (H0: all locations share one rate, counts are Poisson) = {p_disp:.4f}")
print("Interpretation:", "evidence of real heterogeneity between locations" if p_disp < 0.01 else
      "no convincing evidence that locations differ beyond random chance")


# ------------------------------------------------------------------ M-A: Poisson GLM from scratch (IRLS)
def fit_poisson_irls(X, y, offset, l2=1e-4, iters=100, tol=1e-9):
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(y.sum() / np.exp(offset).sum())
    pen = np.eye(p) * l2
    pen[0, 0] = 0.0
    for _ in range(iters):
        eta = X @ beta + offset
        mu = np.exp(np.clip(eta, -20, 20))
        z = eta - offset + (y - mu) / mu
        A = X.T @ (X * mu[:, None]) + pen
        new = np.linalg.solve(A, X.T @ (mu * z))
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def design(df, ref=None):
    cat = ["zone", "road_type", "road_condition", "pedestrian_activity"]
    d = pd.get_dummies(df[cat].astype(str), drop_first=True, dtype=float)
    d["near_school"] = df.near_school.astype(float)
    d["near_hospital"] = df.near_hospital.astype(float)
    d["log_density"] = np.log(df.population_density_per_km2.astype(float))
    d["lat"], d["lon"] = df.latitude, df.longitude
    if ref is None:
        ref = (d.mean(), d.std().replace(0, 1))
    d = (d - ref[0]) / ref[1]
    d.insert(0, "intercept", 1.0)
    return d, ref


section("M-A. POISSON GLM (IRLS, from scratch): do location attributes explain incident counts?")
Xtr, ref = design(loc)
X = Xtr.values
off1, off2, offF = np.full(len(ids), np.log(E1)), np.full(len(ids), np.log(E2)), np.full(len(ids), np.log(EF))

# likelihood-ratio test on full year
b_full = fit_poisson_irls(X, y_all, offF)
mu_full = np.exp(X @ b_full + offF)
mu_null = np.full(len(ids), y_all.mean())
lrt = pois_dev(y_all, mu_null) - pois_dev(y_all, mu_full)
k = X.shape[1] - 1
p_lrt = stats.chi2.sf(lrt, k)
print(f"Predictors in model: {k}   LRT statistic vs intercept-only = {lrt:.1f}   p = {p_lrt:.4f}")
print("Expected LRT under pure noise ~", k, "(its degrees of freedom)")

# time-based validation: fit on H1 counts, predict H2 counts
b_h1 = fit_poisson_irls(X, y_h1, off1, l2=1.0)
mu_glm = np.exp(X @ b_h1 + off2)
print("\nGLM top coefficients (standardized, fitted on H1):")
coef = pd.Series(b_h1[1:], index=Xtr.columns[1:]).sort_values(key=np.abs, ascending=False)
print(coef.head(8).round(3).to_string())

# ------------------------------------------------------------------ M-B: Empirical-Bayes Gamma-Poisson
def nb_negloglik(theta, y, e):
    a, b = np.exp(theta[0]), np.exp(theta[1])
    ll = (gammaln(y + a) - gammaln(a) - gammaln(y + 1) + a * np.log(b / (b + e)) + y * np.log(e / (b + e)))
    return -ll.sum()


def fit_eb(y, e):
    """Maximum-likelihood Gamma(a, b) prior on location rates (rate = incidents per year)."""
    mean_rate = y.sum() / (e * len(y)) if np.isscalar(e) else y.sum() / e.sum()
    best = None
    for la in [0.0, 2.0, 4.0, 6.0]:
        th0 = [la, la - np.log(mean_rate)]
        r = optimize.minimize(nb_negloglik, th0, args=(y, e), method="L-BFGS-B",
                              bounds=[(-3, 9), (-6, 15)])
        if best is None or r.fun < best.fun:
            best = r
    a, b = np.exp(best.x)
    return a, b, best


section("M-B. EMPIRICAL-BAYES GAMMA-POISSON (prior fitted by ML, from scratch)")
a1, b1, r1 = fit_eb(y_h1, E1)
print(f"Prior fitted on H1 only: shape a={a1:.1f}, rate b={b1:.1f}  -> prior mean {a1 / b1:.2f}/yr, "
      f"prior coefficient of variation {1 / np.sqrt(a1):.3f}")
print("(CV near 0 means the data says: all locations have essentially the same underlying rate)")

# validation on H2
post1 = (a1 + y_h1) / (b1 + E1)                 # posterior mean rate (per year)
mu_eb = post1 * E2
mu_global = np.full(len(ids), y_h1.sum() / len(ids) * (E2 / E1))
mu_raw = np.clip(y_h1 / E1 * E2, 0.05, None)    # raw history, floor to avoid log(0)

section("VALIDATION: fit on Jan-Jun, predict Jul-Dec incident counts per location")
vt = pd.DataFrame({
    "model": ["Global mean (baseline 1)", "Raw H1 history (baseline 2)", "Poisson GLM on attributes (M-A)", "Empirical-Bayes (M-B)"],
    "mean_poisson_deviance": [mean_dev(y_h2, mu_global), mean_dev(y_h2, mu_raw), mean_dev(y_h2, mu_glm), mean_dev(y_h2, mu_eb)],
    "MAE": [np.abs(y_h2 - m_).mean() for m_ in (mu_global, mu_raw, mu_glm, mu_eb)],
}).round(4)
print(vt.to_string(index=False))
print("Lower is better. If EB ~ global mean << raw history, shrinkage has correctly learned that history is mostly noise.")

# ranking coverage with random band
def coverage(score, y, k=50):
    return y[np.argsort(-score)[:k]].sum() / y.sum()


cov = {"Empirical-Bayes rank": coverage(post1, y_h2), "Raw H1 count rank": coverage(y_h1, y_h2),
       "GLM rank": coverage(mu_glm, y_h2)}
rand = np.array([coverage(RNG.random(len(ids)), y_h2) for _ in range(5000)])
lo, hi = np.percentile(rand, [2.5, 97.5])
print(f"\nShare of Jul-Dec incidents located in the top-50 locations ranked from Jan-Jun data (random expectation 5.0%):")
for k_, v in cov.items():
    flag = "inside random band" if lo <= v <= hi else "OUTSIDE random band"
    print(f"  {k_:22s} {v:.2%}   [{flag}; 95% of random picks fall in {lo:.2%} - {hi:.2%}]")

# ------------------------------------------------------------------ final fit on full year + rank uncertainty
section("FINAL MODEL (fit on full year) + RANK UNCERTAINTY")
aF, bF, rF = fit_eb(y_all, EF)
print(f"Prior on full year: a={aF:.1f}, b={bF:.1f}, prior mean {aF / bF:.2f}/yr, prior CV {1 / np.sqrt(aF):.3f}")
shape, rate = aF + y_all, bF + EF
post_mean = shape / rate
NS = 4000
draws = RNG.gamma(shape, 1.0 / rate, size=(NS, len(ids)))
ranks = (-draws).argsort(1).argsort(1) + 1                     # 1 = highest rate
ci_lo, ci_hi = np.percentile(draws, [5, 95], axis=0)
r_lo, r_med, r_hi = np.percentile(ranks, [5, 50, 95], axis=0)
p_top50 = (ranks <= 50).mean(0)
p_above = (draws > np.median(post_mean)).mean(0)
city_mean = np.mean(post_mean)
p_above_mean = (draws > aF / bF).mean(0)

tbl = pd.DataFrame({
    "location_id": ids, "zone": loc.zone.values, "incidents_2025": y_all.astype(int),
    "raw_rate_per_yr": (y_all / EF).round(2), "eb_rate_per_yr": post_mean.round(2),
    "rate_ci5": ci_lo.round(2), "rate_ci95": ci_hi.round(2),
    "rank_5pct": r_lo.astype(int), "rank_median": r_med.astype(int), "rank_95pct": r_hi.astype(int),
    "p_top50": p_top50.round(3), "p_above_citywide_mean": p_above_mean.round(3),
}).sort_values("rank_median")
tbl.to_csv(OUTM / "eb_location_table.csv", index=False)

print("Top 10 by EB rate with uncertainty:")
print(tbl.head(10).to_string(index=False))
widths = (r_hi - r_lo)
print(f"\nMedian rank-interval width (5th-95th pct) across all locations: {np.median(widths):.0f} out of 1000 places")
print(f"Locations with P(in top-50) >= 0.5 : {(p_top50 >= 0.5).sum()}")
print(f"Locations with P(in top-50) >= 0.8 : {(p_top50 >= 0.8).sum()}")
print(f"Locations with P(rate above citywide mean) >= 0.95 : {(p_above_mean >= 0.95).sum()}   (chance alone gives ~0)")
print("Interpretation: if few/no locations are confidently distinguishable, the honest recommendation is to treat most "
      "places as a statistical tie and break ties with exposure, open backlog and travel bundling, not past counts.")

joblib.dump({"prior_full": (aF, bF), "prior_h1": (a1, b1), "glm_beta_h1": b_h1, "glm_columns": list(Xtr.columns),
             "glm_ref_mean": ref[0].to_dict(), "glm_ref_std": ref[1].to_dict(), "exposure_years": EF}, MOD / "eb_model.joblib")

summary = {
    "dispersion_ratio": float(pearson / dfree), "dispersion_p": float(p_disp),
    "glm_lrt": float(lrt), "glm_lrt_df": int(k), "glm_lrt_p": float(p_lrt),
    "eb_prior_cv_h1": float(1 / np.sqrt(a1)), "eb_prior_cv_full": float(1 / np.sqrt(aF)),
    "validation": vt.to_dict(orient="records"),
    "top50_coverage": {k_: float(v) for k_, v in cov.items()}, "random_band_95": [float(lo), float(hi)],
    "median_rank_interval_width": float(np.median(widths)),
    "n_p_top50_ge_0.5": int((p_top50 >= 0.5).sum()), "n_above_mean_ge_0.95": int((p_above_mean >= 0.95).sum()),
}
(OUTM / "eb_summary.json").write_text(json.dumps(summary, indent=2))
print("\nDONE. Saved eb_report.txt, eb_summary.json, eb_location_table.csv, eb_model.joblib")
