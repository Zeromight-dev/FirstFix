"""
FixFirst - Phase 1: Exploratory Data Analysis
Run from the FirstFix folder:   python src/eda.py
Reads:  data/locations.csv, data/incidents.csv, data/public_complaints.csv
Writes: outputs/eda/eda_report.txt  +  outputs/eda/*.png
Paste eda_report.txt (or the key sections) back to your assistant.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)


class Tee:
    """Print to console and to the report file at the same time."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


sys.stdout = Tee(OUT / "eda_report.txt")


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- load
loc = pd.read_csv(DATA / "locations.csv")
inc = pd.read_csv(DATA / "incidents.csv")
com = pd.read_csv(DATA / "public_complaints.csv")

for df in (loc, inc, com):
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype(str).str.strip()

inc["ts"] = pd.to_datetime(inc["timestamp"], errors="coerce")
com["ts"] = pd.to_datetime(com["created_at"], errors="coerce")
print("Unparseable timestamps  incidents:", inc.ts.isna().sum(), " complaints:", com.ts.isna().sum())

# ---------------------------------------------------------------- 1. time range
section("1. TIME RANGE")
for name, df in [("incidents", inc), ("complaints", com)]:
    print(f"{name:11s} min={df.ts.min()}  max={df.ts.max()}  span_days={(df.ts.max() - df.ts.min()).days}")
t_min = min(inc.ts.min(), com.ts.min())
t_max = max(inc.ts.max(), com.ts.max())
print("Overall window:", t_min, "->", t_max)

print("\nIncidents per month:")
inc_m = inc.groupby(inc.ts.dt.to_period("M")).size()
print(inc_m.to_string())
print("\nComplaints per month:")
com_m = com.groupby(com.ts.dt.to_period("M")).size()
print(com_m.to_string())

# ---------------------------------------------------------------- 2. categories
section("2. CATEGORY VALUE COUNTS")
for df, name, cols in [
    (inc, "incidents", ["incident_type", "severity", "reported_by", "resolved", "injuries"]),
    (com, "complaints", ["category", "priority_reported", "status"]),
    (loc, "locations", ["zone", "road_type", "road_condition", "pedestrian_activity", "near_school", "near_hospital"]),
]:
    for c in cols:
        print(f"\n[{name}.{c}]")
        print(df[c].value_counts(dropna=False).to_string())

# ---------------------------------------------------------------- 3. resolution fields
section("3. RESOLUTION FIELDS (what do resolved / resolution_hours really mean?)")
print("incidents: resolved vs resolution_hours present")
print(pd.crosstab(inc.resolved, inc.resolution_hours.notna(), rownames=["resolved"], colnames=["hours_present"]))
print("\nresolution_hours describe by resolved flag:")
print(inc.groupby("resolved").resolution_hours.describe().to_string())

# Censoring test: for unresolved rows, is resolution_hours ~ elapsed time up to dataset end?
elapsed_h = (t_max - inc.ts).dt.total_seconds() / 3600
unres = inc[inc.resolved == 0].copy()
unres["elapsed_h"] = elapsed_h[unres.index]
unres["rh_over_elapsed"] = unres.resolution_hours / unres.elapsed_h
print("\nUnresolved incidents: resolution_hours / hours-until-dataset-end")
print(unres.rh_over_elapsed.describe().to_string())
print("Share of unresolved with resolution_hours > elapsed time (impossible if real):",
      round((unres.resolution_hours > unres.elapsed_h).mean(), 3))
print("Correlation(resolution_hours, elapsed) among unresolved:",
      round(unres[["resolution_hours", "elapsed_h"]].corr().iloc[0, 1], 3))

print("\nResolved incidents: resolution_hours distribution")
r = inc[inc.resolved == 1].resolution_hours
print(r.describe(percentiles=[.1, .25, .5, .75, .9, .99]).to_string())
print("\nMedian resolution_hours by incident_type (resolved only):")
print(inc[inc.resolved == 1].groupby("incident_type").resolution_hours.median().sort_values().to_string())
print("\nMedian resolution_hours by severity (resolved only):")
print(inc[inc.resolved == 1].groupby("severity").resolution_hours.median().to_string())
print("\nResolved share by severity:")
print(inc.groupby("severity").resolved.mean().round(3).to_string())

print("\ncomplaints: status vs days_to_resolution present")
print(pd.crosstab(com.status, com.days_to_resolution.notna()))
print("\ndays_to_resolution by status:")
print(com.groupby("status").days_to_resolution.describe().to_string())
print("\nMedian days_to_resolution by category:")
print(com.groupby("category").days_to_resolution.median().sort_values().to_string())
print("\nMedian days_to_resolution by priority_reported:")
print(com.groupby("priority_reported").days_to_resolution.median().to_string())

# ---------------------------------------------------------------- 4. sparsity
section("4. SPARSITY (events per location)")
n_loc = len(loc)
inc_per_loc = inc.groupby("location_id").size().reindex(loc.location_id, fill_value=0)
com_per_loc = com.groupby("location_id").size().reindex(loc.location_id, fill_value=0)
print("Incidents per location:")
print(inc_per_loc.describe(percentiles=[.1, .25, .5, .75, .9, .99]).to_string())
print("Locations with 0 incidents:", int((inc_per_loc == 0).sum()), f"({(inc_per_loc == 0).mean():.1%})")
print("\nComplaints per location:")
print(com_per_loc.describe(percentiles=[.1, .25, .5, .75, .9, .99]).to_string())
print("Locations with 0 complaints:", int((com_per_loc == 0).sum()), f"({(com_per_loc == 0).mean():.1%})")

# 28-day window sparsity
win = 28
starts = pd.date_range(t_min.normalize(), t_max, freq=f"{win}D")
cnt = np.zeros((n_loc, len(starts)))
idx = {l: i for i, l in enumerate(loc.location_id)}
w_of = ((inc.ts - starts[0]).dt.days // win).clip(0, len(starts) - 1)
for l, w in zip(inc.location_id, w_of):
    cnt[idx[l], int(w)] += 1
print(f"\n{win}-day windows: {len(starts)} windows x {n_loc} locations = {cnt.size} cells")
print("Share of cells with >=1 incident:", round((cnt > 0).mean(), 3))
print("Mean count per cell:", round(cnt.mean(), 3), " Variance:", round(cnt.var(), 3),
      " Var/Mean (1 = Poisson, >1 = overdispersed):", round(cnt.var() / max(cnt.mean(), 1e-9), 2))
print("Distribution of counts per cell:")
print(pd.Series(cnt.ravel()).astype(int).value_counts().sort_index().head(10).to_string())

# ---------------------------------------------------------------- 5. signal checks
section("5. SIGNAL CHECKS (does anything predict incident counts?)")
L = loc.set_index("location_id").copy()
L["inc_n"] = inc_per_loc
L["com_n"] = com_per_loc
L["inj_sum"] = inc.groupby("location_id").injuries.sum().reindex(L.index, fill_value=0)
L["high_sev_n"] = inc[inc.severity.str.lower().isin(["high", "critical"])].groupby("location_id").size().reindex(L.index, fill_value=0)

for c in ["road_condition", "road_type", "pedestrian_activity", "near_school", "near_hospital"]:
    print(f"\nMean incidents / complaints per location by {c}:")
    print(L.groupby(c)[["inc_n", "com_n", "inj_sum", "high_sev_n"]].mean().round(2).to_string())

print("\nZone: mean incidents per location (top 8 / bottom 4):")
z = L.groupby("zone").inc_n.agg(["mean", "count"]).sort_values("mean", ascending=False)
print(z.head(8).round(2).to_string())
print("...")
print(z.tail(4).round(2).to_string())
print("Zone spread (max-min of means):", round(z["mean"].max() - z["mean"].min(), 2))

print("\nSpearman correlations with incident count per location:")
num = L[["population_density_per_km2", "near_school", "near_hospital", "latitude", "longitude", "com_n"]]
print(num.corrwith(L.inc_n, method="spearman").round(3).to_string())

# Persistence: does the first half predict the second half?
mid = t_min + (t_max - t_min) / 2
h1 = inc[inc.ts < mid].groupby("location_id").size().reindex(L.index, fill_value=0)
h2 = inc[inc.ts >= mid].groupby("location_id").size().reindex(L.index, fill_value=0)
print(f"\nPersistence (split at {mid.date()}):")
print("Spearman corr (incidents H1 vs H2) per location:", round(h1.corr(h2, method="spearman"), 3))
c1 = com[com.ts < mid].groupby("location_id").size().reindex(L.index, fill_value=0)
print("Spearman corr (complaints H1 vs incidents H2):", round(c1.corr(h2, method="spearman"), 3))
top50_h1 = set(h1.sort_values(ascending=False).head(50).index)
top50_h2 = set(h2.sort_values(ascending=False).head(50).index)
print("Overlap of top-50 locations H1 vs H2:", len(top50_h1 & top50_h2), "/ 50   (random expectation ~2.5)")

# By incident type and incident-complaint pairing
print("\nIncident type by road_condition (row %):")
m = inc.merge(loc[["location_id", "road_condition", "road_type"]], on="location_id")
print((pd.crosstab(m.incident_type, m.road_condition, normalize="columns") * 100).round(1).to_string())

print("\nSeverity vs injuries (mean injuries):")
print(inc.groupby("severity").injuries.agg(["mean", "max", "count"]).round(2).to_string())
print("\nReported_by vs resolved share / median hours:")
print(inc.groupby("reported_by").agg(n=("resolved", "size"), resolved=("resolved", "mean"),
                                     med_hours=("resolution_hours", "median")).round(2).to_string())

print("\nIncidents by hour of day:")
print(inc.groupby(inc.ts.dt.hour).size().to_string())
print("\nIncidents by weekday (0=Mon):")
print(inc.groupby(inc.ts.dt.dayofweek).size().to_string())
print("\nIncidents by month number (seasonality check):")
print(inc.groupby(inc.ts.dt.month).size().to_string())

# Is the mean incident rate trending over time?
months_idx = np.arange(len(inc_m))
if len(inc_m) > 2:
    slope = np.polyfit(months_idx, inc_m.values, 1)[0]
    print(f"\nMonthly incident trend slope: {slope:.1f} incidents/month per month (mean {inc_m.mean():.0f})")

# Zone-level time variation (is there location-specific time structure to exploit?)
print("\nTop 10 locations by incident count:")
print(L.sort_values("inc_n", ascending=False)[["zone", "road_type", "road_condition", "inc_n", "com_n", "inj_sum"]].head(10).to_string())

# ---------------------------------------------------------------- 6. charts
section("6. CHARTS SAVED TO outputs/eda/")
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)
    print("saved", name)

fig, ax = plt.subplots(figsize=(9, 4))
inc_m.plot(ax=ax, marker="o", label="Incidents")
com_m.plot(ax=ax, marker="s", label="Complaints")
ax.set_title("Monthly incidents and complaints"); ax.set_xlabel(""); ax.legend()
save(fig, "01_monthly_counts.png")

fig, ax = plt.subplots(figsize=(8, 4))
inc.incident_type.value_counts().plot.barh(ax=ax)
ax.set_title("Incident types"); ax.invert_yaxis()
save(fig, "02_incident_types.png")

fig, ax = plt.subplots(figsize=(8, 4))
r.clip(upper=r.quantile(0.99)).hist(bins=40, ax=ax)
ax.set_title("Resolution hours (resolved incidents, clipped at P99)")
save(fig, "03_resolution_hours.png")

fig, ax = plt.subplots(figsize=(8, 4))
L.groupby("road_condition").inc_n.mean().plot.bar(ax=ax)
ax.set_title("Mean incidents per location by road condition"); ax.set_xlabel("")
save(fig, "04_incidents_by_road_condition.png")

fig, ax = plt.subplots(figsize=(7, 4))
inc_per_loc.hist(bins=range(0, int(inc_per_loc.max()) + 2), ax=ax)
ax.set_title("Incidents per location (sparsity)"); ax.set_xlabel("incidents")
save(fig, "05_incidents_per_location.png")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(h1, h2, alpha=0.3, s=12)
ax.set_xlabel("Incidents first half"); ax.set_ylabel("Incidents second half")
ax.set_title("Persistence: first half vs second half")
save(fig, "06_persistence.png")

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(L.longitude, L.latitude, c=L.inc_n, s=14, cmap="viridis")
plt.colorbar(sc, ax=ax, label="incidents")
ax.set_title("Simulated coordinates coloured by incident count")
save(fig, "07_map_incidents.png")

print("\nDONE. Full text report saved to:", OUT / "eda_report.txt")
