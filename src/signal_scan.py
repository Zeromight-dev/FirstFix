"""
FixFirst - Phase 1b: Signal scan
Run from the FirstFix folder:   python src/signal_scan.py
Finds WHERE the synthetic data has real structure, so we model the right targets.
Writes: outputs/eda/signal_scan.txt   (paste the whole file back to your assistant)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, r2_score, roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
DATA, OUT = BASE / "data", BASE / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s); self.f.write(s)

    def flush(self):
        self.o.flush(); self.f.flush()


sys.stdout = Tee(OUT / "signal_scan.txt")


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ------------------------------------------------------------------ load + prep
loc = pd.read_csv(DATA / "locations.csv")
inc = pd.read_csv(DATA / "incidents.csv")
com = pd.read_csv(DATA / "public_complaints.csv")
for df in (loc, inc, com):
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype(str).str.strip()

inc["ts"] = pd.to_datetime(inc["timestamp"])
com["ts"] = pd.to_datetime(com["created_at"])
for df in (inc, com):
    df["hour_bucket"] = pd.cut(df.ts.dt.hour, [-1, 5, 11, 17, 23], labels=["night", "morning", "afternoon", "evening"]).astype(str)
    df["weekday"] = df.ts.dt.dayofweek.astype(str)
    df["month"] = df.ts.dt.month.astype(str)

inc["injured"] = (inc.injuries > 0).astype(int).astype(str)
inc["resolved_c"] = inc.resolved.astype(str)
inc["log_res_h"] = np.log1p(inc.resolution_hours)
com["log_days"] = np.log1p(com.days_to_resolution)

LOC_CAT = ["zone", "road_type", "road_condition", "pedestrian_activity", "near_school", "near_hospital"]
LOC_NUM = ["population_density_per_km2", "latitude", "longitude"]
loc[["near_school", "near_hospital"]] = loc[["near_school", "near_hospital"]].astype(str)
I = inc.merge(loc, on="location_id")
C = com.merge(loc, on="location_id")

# location-level counts
L = loc.set_index("location_id").copy()
L["inc_n"] = inc.groupby("location_id").size().reindex(L.index, fill_value=0)
L["com_n"] = com.groupby("location_id").size().reindex(L.index, fill_value=0)
L["inj_sum"] = inc.groupby("location_id").injuries.sum().reindex(L.index, fill_value=0)
L["unresolved_n"] = inc[inc.resolved == 0].groupby("location_id").size().reindex(L.index, fill_value=0)
L["mean_log_res_h"] = I.groupby("location_id").log_res_h.mean().reindex(L.index)


# ------------------------------------------------------------------ generic association test
def test_pair(df, y, x, y_kind, x_kind):
    d = df[[y, x]].dropna()
    if d[y].nunique() < 2 or d[x].nunique() < 2:
        return None
    if y_kind == "cat" and x_kind == "cat":
        ct = pd.crosstab(d[y], d[x])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        eff = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        return "chi2 (Cramers V)", eff, p
    if y_kind == "num" and x_kind == "num":
        r, p = stats.spearmanr(d[y], d[x])
        return "spearman r", r, p
    num, cat = (y, x) if y_kind == "num" else (x, y)
    groups = [g[num].values for _, g in d.groupby(cat) if len(g) >= 5]
    if len(groups) < 2:
        return None
    h, p = stats.kruskal(*groups)
    return "kruskal (eps^2)", h / max(len(d) - 1, 1), p


def scan(df, table, targets, predictors):
    rows = []
    for y, yk in targets.items():
        for x, xk in predictors.items():
            if x == y:
                continue
            r = test_pair(df, y, x, yk, xk)
            if r:
                rows.append((table, y, x, r[0], round(r[1], 4), r[2]))
    return rows


# ------------------------------------------------------------------ A. association scan
section("A. ASSOCIATION SCAN (every target vs every candidate predictor)")
loc_pred = {**{c: "cat" for c in LOC_CAT}, **{c: "num" for c in LOC_NUM}}
rows = []

# location-level counts
rows += scan(L.reset_index(), "location", {c: "num" for c in ["inc_n", "com_n", "inj_sum", "unresolved_n", "mean_log_res_h"]},
             {**loc_pred, "com_n": "num"})

# incident rows
inc_targets = {"incident_type": "cat", "severity": "cat", "reported_by": "cat", "resolved_c": "cat", "injured": "cat",
               "log_res_h": "num", "injuries": "num"}
inc_pred = {**loc_pred, "hour_bucket": "cat", "weekday": "cat", "month": "cat",
            "incident_type": "cat", "severity": "cat", "reported_by": "cat"}
rows += scan(I, "incident", inc_targets, inc_pred)

# complaint rows
com_targets = {"category": "cat", "priority_reported": "cat", "status": "cat", "log_days": "num"}
com_pred = {**loc_pred, "hour_bucket": "cat", "weekday": "cat", "month": "cat",
            "category": "cat", "priority_reported": "cat", "status": "cat"}
rows += scan(C, "complaint", com_targets, com_pred)

res = pd.DataFrame(rows, columns=["table", "target", "predictor", "test", "effect", "p"]).sort_values("p")
n_tests = len(res)
bonf = 0.05 / n_tests
print(f"Tests run: {n_tests}.  Bonferroni threshold for 'real' = p < {bonf:.2e}.  Expect ~{0.05*n_tests:.0f} false hits at p<0.05 by chance.\n")
sig = res[res.p < bonf]
print(f"Associations passing Bonferroni: {len(sig)}")
print(sig.to_string(index=False) if len(sig) else "NONE - the data looks like noise at this level.")
print("\nNext 20 strongest (may be chance):")
print(res[res.p >= bonf].head(20).to_string(index=False))

# ------------------------------------------------------------------ B. ML probes
section("B. ML PROBES (time-based: train Jan-Aug, test Sep-Dec; HistGradientBoosting vs naive baseline)")
SPLIT = pd.Timestamp("2025-09-01")


def make_X(df, drop):
    cols = [c for c in LOC_CAT + LOC_NUM + ["hour_bucket", "weekday", "month", "incident_type", "severity", "reported_by"]
            if c in df.columns and c not in drop]
    return pd.get_dummies(df[cols], dtype=float)


def probe_class(df, tcol, drop, label):
    y = df[tcol].astype(str)
    X = make_X(df, drop | {tcol})
    tr, te = df.ts < SPLIT, df.ts >= SPLIT
    classes = sorted(y.unique())
    if len(classes) == 2:
        pos = classes[1]
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, early_stopping=False, random_state=0)
        m.fit(X[tr], (y[tr] == pos).astype(int))
        p = m.predict_proba(X[te])[:, 1]
        yt = (y[te] == pos).astype(int)
        base = np.full(len(yt), (y[tr] == pos).mean())
        return (label, "AUC", round(roc_auc_score(yt, p), 3), 0.5, round(log_loss(yt, p), 4), round(log_loss(yt, base), 4), f"pos rate {yt.mean():.2f}")
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, early_stopping=False, random_state=0)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])
    prior = y[tr].value_counts(normalize=True).reindex(m.classes_).values
    base = np.tile(prior, (te.sum(), 1))
    acc = accuracy_score(y[te], m.classes_[p.argmax(1)])
    maj = (y[te] == y[tr].mode()[0]).mean()
    return (label, "accuracy", round(acc, 3), round(maj, 3), round(log_loss(y[te], p, labels=m.classes_), 4),
            round(log_loss(y[te], base, labels=m.classes_), 4), f"{len(classes)} classes")


def probe_reg(df, tcol, drop, label):
    y = df[tcol]
    X = make_X(df, drop | {tcol})
    tr, te = df.ts < SPLIT, df.ts >= SPLIT
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, early_stopping=False, random_state=0)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    mae_b = mean_absolute_error(y[te], np.full(te.sum(), y[tr].median()))
    return (label, "R2 (MAE)", f"{r2_score(y[te], p):.3f} ({mean_absolute_error(y[te], p):.3f})", f"0 ({mae_b:.3f})", "", "", "lower MAE than baseline = signal")


probes = [
    probe_class(I, "resolved_c", set(), "incident: resolved (open vs closed)"),
    probe_class(I, "injured", set(), "incident: injured (injuries>0)"),
    probe_class(I, "severity", {"injuries"}, "incident: severity"),
    probe_class(I, "incident_type", set(), "incident: incident_type"),
    probe_class(I, "reported_by", set(), "incident: reported_by"),
    probe_reg(I, "log_res_h", set(), "incident: log resolution hours"),
    probe_class(C, "status", set(), "complaint: status"),
    probe_class(C, "priority_reported", set(), "complaint: priority_reported"),
    probe_reg(C, "log_days", set(), "complaint: log days_to_resolution"),
]
# complaint probe uses 'category' etc. columns that exist only in C, so make_X handles absent cols
tbl = pd.DataFrame(probes, columns=["target", "metric", "model", "baseline", "logloss_model", "logloss_base", "note"])
print(tbl.to_string(index=False))
print("\nHow to read: AUC ~0.50 / accuracy ~ majority / R2 ~ 0 / model logloss ~ baseline logloss = NO usable signal.")

# ------------------------------------------------------------------ C. location-level forecasting probe
section("C. LOCATION-LEVEL FORECAST PROBE (28-day counts, train cutoffs <= Jun, test cutoffs >= Aug)")
cut_all = pd.date_range("2025-03-01", "2025-12-03", freq="14D")
feats = []
for t in cut_all:
    past = inc[inc.ts < t]
    past90 = inc[(inc.ts < t) & (inc.ts >= t - pd.Timedelta(days=90))]
    pastc90 = com[(com.ts < t) & (com.ts >= t - pd.Timedelta(days=90))]
    fut = inc[(inc.ts >= t) & (inc.ts < t + pd.Timedelta(days=28))]
    d = loc[["location_id"]].copy()
    d["cutoff"] = t
    d["n_all"] = d.location_id.map(past.groupby("location_id").size()).fillna(0)
    d["n_90"] = d.location_id.map(past90.groupby("location_id").size()).fillna(0)
    d["c_90"] = d.location_id.map(pastc90.groupby("location_id").size()).fillna(0)
    d["y"] = d.location_id.map(fut.groupby("location_id").size()).fillna(0)
    feats.append(d)
F = pd.concat(feats).merge(loc, on="location_id")
Xf = pd.get_dummies(F[["n_all", "n_90", "c_90"] + LOC_CAT + LOC_NUM], dtype=float)
tr = F.cutoff <= pd.Timestamp("2025-06-30")
te = F.cutoff >= pd.Timestamp("2025-08-01")
print(f"train rows {tr.sum()}, test rows {te.sum()} (28-day purge gap respected: train targets end before test cutoffs)")
m = HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05, max_iter=150, early_stopping=False, random_state=0)
m.fit(Xf[tr], F.y[tr])
pred = m.predict(Xf[te])
base_mean = np.full(te.sum(), F.y[tr].mean())
base_90 = F.n_90[te] / 90 * 28
from sklearn.metrics import mean_poisson_deviance
print("Mean Poisson deviance  model :", round(mean_poisson_deviance(F.y[te], np.clip(pred, 1e-6, None)), 4))
print("Mean Poisson deviance  global mean baseline:", round(mean_poisson_deviance(F.y[te], base_mean), 4))
print("Mean Poisson deviance  trailing-90d rate baseline:", round(mean_poisson_deviance(F.y[te], np.clip(base_90, 1e-6, None)), 4))
print("MAE model / global mean / trailing-90d:", round(mean_absolute_error(F.y[te], pred), 4),
      round(mean_absolute_error(F.y[te], base_mean), 4), round(mean_absolute_error(F.y[te], base_90), 4))

# top-50 coverage per test cutoff
cov_model, cov_hist, cov_rand = [], [], []
rng = np.random.default_rng(0)
Ft = F[te].copy()
Ft["pred"] = pred
for t, g in Ft.groupby("cutoff"):
    tot = g.y.sum()
    if tot == 0:
        continue
    cov_model.append(g.nlargest(50, "pred").y.sum() / tot)
    cov_hist.append(g.nlargest(50, "n_all").y.sum() / tot)
    cov_rand.append(g.sample(50, random_state=int(rng.integers(1e9))).y.sum() / tot)
print(f"\nShare of next-28-day incidents captured by top-50 locations (random pick would be 5.0%):")
print(f"  model        : {np.mean(cov_model):.3%}")
print(f"  most past inc: {np.mean(cov_hist):.3%}")
print(f"  random       : {np.mean(cov_rand):.3%}")

print("\nDONE. Saved:", OUT / "signal_scan.txt")
