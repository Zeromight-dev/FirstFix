"""
FixFirst - Phase 3: current-state triage engine
Ranks locations on FACTS AS OF A DATE (open incidents, open complaints, recent harm, exposure),
not on forecasts. Uses only records dated before the as-of date (leakage-safe).

Run:  python src/priority.py            -> prints top locations, saves outputs/triage_table.csv
Import in the app:  from priority import load_all, build_features, score, rank_stability, add_receipts, LENSES
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA, OUT = BASE / "data", BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---- assumptions (state these in the README) -------------------------------------------------
SEV_W = {"low": 1, "minor": 1, "medium": 2, "moderate": 2, "high": 3, "major": 3,
         "critical": 4, "severe": 4, "fatal": 4}
PRI_W = {"low": 1, "normal": 1, "medium": 2, "high": 2, "urgent": 3, "critical": 3, "emergency": 3}
PED_W = {"low": 1, "medium": 2, "high": 3}
CLOSED_STATUS = {"resolved", "closed"}
RECENT_DAYS = 180

COMPONENTS = ["open_incident_load", "injury_severity", "open_complaint_load",
              "complaint_age", "exposure", "shrunk_history"]
LABELS = {"open_incident_load": "Open incidents (severity-weighted)",
          "injury_severity": f"Recent harm (last {RECENT_DAYS}d: severity + injuries)",
          "open_complaint_load": "Open complaints (priority-weighted)",
          "complaint_age": "Age of oldest open complaint",
          "exposure": "Exposure (school / hospital / pedestrians / density)",
          "shrunk_history": "Long-run incident rate (heavily shrunk, low weight)"}
LENSES = {
    "Safety-first":          dict(open_incident_load=.20, injury_severity=.35, open_complaint_load=.10,
                                  complaint_age=.05, exposure=.25, shrunk_history=.05),
    "Backlog-first":         dict(open_incident_load=.30, injury_severity=.10, open_complaint_load=.30,
                                  complaint_age=.20, exposure=.05, shrunk_history=.05),
    "Vulnerable-areas-first": dict(open_incident_load=.15, injury_severity=.15, open_complaint_load=.10,
                                   complaint_age=.05, exposure=.50, shrunk_history=.05),
}


def _strip(df):
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype(str).str.strip()
    return df


def load_all(verbose=False):
    loc = _strip(pd.read_csv(DATA / "locations.csv"))
    inc = _strip(pd.read_csv(DATA / "incidents.csv"))
    com = _strip(pd.read_csv(DATA / "public_complaints.csv"))
    inc["ts"] = pd.to_datetime(inc["timestamp"])
    com["ts"] = pd.to_datetime(com["created_at"])
    inc["sev_w"] = inc.severity.str.lower().map(SEV_W)
    com["pri_w"] = com.priority_reported.str.lower().map(PRI_W)
    loc["ped_w"] = loc.pedestrian_activity.str.lower().map(PED_W)
    if verbose:
        for name, s in [("severity", inc.severity[inc.sev_w.isna()]), ("priority_reported", com.priority_reported[com.pri_w.isna()]),
                        ("pedestrian_activity", loc.pedestrian_activity[loc.ped_w.isna()])]:
            if len(s):
                print(f"WARNING unmapped {name} values -> default weight used: {sorted(s.unique())}")
    inc["sev_w"] = inc.sev_w.fillna(2)
    com["pri_w"] = com.pri_w.fillna(1)
    loc["ped_w"] = loc.ped_w.fillna(2)
    eb_path = OUT / "models" / "eb_location_table.csv"
    eb = pd.read_csv(eb_path)[["location_id", "eb_rate_per_yr", "p_top50"]] if eb_path.exists() else None
    return loc, inc, com, eb


def pct(s):
    return s.rank(pct=True, method="average") * 100


def build_features(loc, inc, com, eb, asof):
    """All fields use only records with timestamp < asof."""
    asof = pd.Timestamp(asof)
    i = inc[inc.ts < asof].copy()
    closed_at = i.ts + pd.to_timedelta(i.resolution_hours, unit="h")
    i["is_open"] = ~((i.resolved == 1) & (closed_at <= asof))
    c = com[com.ts < asof].copy()
    c_closed_at = c.ts + pd.to_timedelta(c.days_to_resolution, unit="D")
    c["is_open"] = ~(c.status.str.lower().isin(CLOSED_STATUS) & (c_closed_at <= asof))

    F = loc.set_index("location_id").copy()
    g = lambda s: s.reindex(F.index).fillna(0)
    oi = i[i.is_open]
    F["open_inc_n"] = g(oi.groupby("location_id").size())
    F["open_incident_load"] = g(oi.groupby("location_id").sev_w.sum())
    rec = i[i.ts >= asof - pd.Timedelta(days=RECENT_DAYS)]
    F["recent_inc_n"] = g(rec.groupby("location_id").size())
    F["recent_injuries"] = g(rec.groupby("location_id").injuries.sum())
    F["injury_severity"] = g(rec.groupby("location_id").sev_w.sum()) + 2 * F.recent_injuries
    oc = c[c.is_open]
    F["open_com_n"] = g(oc.groupby("location_id").size())
    F["open_complaint_load"] = g(oc.groupby("location_id").pri_w.sum())
    F["complaint_age"] = g(((asof - oc.ts).dt.days).groupby(oc.location_id).max())
    F["inc_total"] = g(i.groupby("location_id").size())
    F["com_total"] = g(c.groupby("location_id").size())
    raw_exp = (2.0 * F.near_school + 1.5 * F.near_hospital + F.ped_w
               + 1.5 * F.population_density_per_km2.rank(pct=True))
    F["exposure"] = raw_exp
    F["shrunk_history"] = F.index.map(eb.set_index("location_id").eb_rate_per_yr) if eb is not None else 0.0
    F["shrunk_history"] = F.shrunk_history.astype(float).fillna(0.0)
    F["p_top50_history"] = F.index.map(eb.set_index("location_id").p_top50) if eb is not None else np.nan
    for k in COMPONENTS:
        F[k + "_pts"] = pct(F[k]) if F[k].nunique() > 1 else 0.0
    F["voice_record_gap"] = pct(F.com_total) - pct(F.inc_total)     # >0: citizens complain more than records show
    return F


def score(F, weights):
    w = pd.Series(weights)
    out = F.copy()
    for k in COMPONENTS:
        out[k + "_contrib"] = out[k + "_pts"] * w[k]
    out["priority_score"] = sum(out[k + "_contrib"] for k in COMPONENTS)
    out["rank"] = out.priority_score.rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank")


def rank_stability(F, weights, n=500, conc=40, seed=0):
    """Perturb lens weights (Dirichlet) and see how stable each location's rank is."""
    rng = np.random.default_rng(seed)
    w0 = np.array([weights[k] for k in COMPONENTS])
    P = F[[k + "_pts" for k in COMPONENTS]].values
    W = rng.dirichlet(w0 * conc + 1e-3, size=n)
    ranks = (-(P @ W.T)).argsort(0).argsort(0) + 1          # locations x draws
    r = pd.DataFrame({"rank_p5": np.percentile(ranks, 5, axis=1).round().astype(int),
                      "rank_p95": np.percentile(ranks, 95, axis=1).round().astype(int),
                      "p_top10": (ranks <= 10).mean(1), "p_top50": (ranks <= 50).mean(1)}, index=F.index)
    return r


def add_receipts(S, asof):
    """Plain-language evidence, confidence and missing-information flags (template-based, no LLM)."""
    asof = pd.Timestamp(asof)
    reasons, flags, conf = [], [], []
    for lid, r in S.iterrows():
        top = sorted(COMPONENTS, key=lambda k: -r[k + "_contrib"])[:3]
        parts = []
        for k in top:
            if k == "open_incident_load":
                parts.append(f"{int(r.open_inc_n)} open incident(s), severity load {r.open_incident_load:.0f}")
            elif k == "injury_severity":
                parts.append(f"{int(r.recent_inc_n)} incident(s) in last {RECENT_DAYS}d, {int(r.recent_injuries)} injuries")
            elif k == "open_complaint_load":
                parts.append(f"{int(r.open_com_n)} open complaint(s)")
            elif k == "complaint_age":
                parts.append(f"oldest open complaint {int(r.complaint_age)}d old")
            elif k == "exposure":
                bits = [b for b, v in [("school nearby", r.near_school), ("hospital nearby", r.near_hospital)] if v]
                parts.append(f"{r.pedestrian_activity} pedestrian activity" + (", " + ", ".join(bits) if bits else ""))
            else:
                parts.append("long-run incident rate (weak evidence)")
        reasons.append("; ".join(parts))
        f = []
        if r.inc_total < 5:
            f.append("few historical incidents (<5)")
        if r.recent_inc_n == 0 and r.open_com_n == 0:
            f.append("no recent activity in data")
        if r.complaint_age > 90:
            f.append("complaint open >90 days")
        if r.voice_record_gap > 40:
            f.append("complaints exceed official incidents: verify on site")
        elif r.voice_record_gap < -40:
            f.append("incidents but few complaints: possible under-reporting area")
        if pd.notna(r.p_top50_history) and r.p_top50_history < 0.5:
            f.append("long-run history is a statistical tie (cannot distinguish from average)")
        flags.append(" | ".join(f) if f else "none")
        conf.append("High" if r.p_top10 >= 0.8 else ("Medium" if r.p_top50 >= 0.8 else "Low"))
    S = S.copy()
    S["why"], S["evidence_gaps"], S["rank_confidence"] = reasons, flags, conf
    return S


def run(asof="2025-12-31 23:59", lens="Safety-first", verbose=True):
    loc, inc, com, eb = load_all(verbose)
    F = build_features(loc, inc, com, eb, asof)
    S = score(F, LENSES[lens])
    S = S.join(rank_stability(F, LENSES[lens]))
    return add_receipts(S, asof)


if __name__ == "__main__":
    loc, inc, com, eb = load_all(verbose=True)
    print("severity values :", sorted(inc.severity.unique()))
    print("complaint status:", sorted(com.status.unique()))
    print("priority values :", sorted(com.priority_reported.unique()))
    S = run()
    cols = ["rank", "zone", "open_inc_n", "open_com_n", "recent_injuries", "complaint_age", "priority_score", "rank_p5", "rank_p95",
            "p_top10", "rank_confidence"]
    print("\nTOP 15 (Safety-first, as of 2025-12-31):")
    print(S[cols].head(15).round(2).to_string())
    print("\nWhy #1:", S.iloc[0].why)
    print("Evidence gaps #1:", S.iloc[0].evidence_gaps)
    print("\nRank confidence counts:\n", S.rank_confidence.value_counts().to_string())
    print(f"Median rank interval width (5-95%): {(S.rank_p95 - S.rank_p5).median():.0f} of {len(S)}")
    print("Open incidents total:", int(S.open_inc_n.sum()), " Open complaints total:", int(S.open_com_n.sum()))
    S.to_csv(OUT / "triage_table.csv")
    print("\nSaved", OUT / "triage_table.csv")
