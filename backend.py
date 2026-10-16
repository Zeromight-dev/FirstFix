"""
FixFirst API (FastAPI) - wraps src/priority.py and src/planner.py. No new model logic here.

Run from the FirstFix folder:
    pip install fastapi uvicorn python-multipart
    uvicorn backend:app --reload --port 8000
Docs (try every endpoint): http://localhost:8000/docs

Notes
- Single-user in-memory session: uploaded/demo rows live in `STATE` until /api/reset or restart.
- "rank_confidence" from priority.py is exposed as "rank_stability" (robustness to weight changes).
- `asof` is a YYYY-MM-DD date; omitted = latest date in the data (so new uploads show up immediately).
"""
import json
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.cluster import KMeans

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from planner import COST_NOTE, add_cost, compare, plan  # noqa: E402
from priority import (CLOSED_STATUS, COMPONENTS, LABELS, LENSES, PRI_W, RECENT_DAYS, SEV_W,  # noqa: E402
                      add_receipts, build_features, load_all, rank_stability, score)

SYNTH = ("Synthetic educational data. Coordinates and records are simulated and must not be read as real civic "
         "reports or used for real public-safety decisions.")
MIN_ASOF = pd.Timestamp("2025-02-01")

app = FastAPI(title="FixFirst API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

loc, INC0, COM0, EB = load_all()
EB_TABLE = pd.read_csv(BASE / "outputs" / "models" / "eb_location_table.csv").set_index("location_id") \
    if (BASE / "outputs" / "models" / "eb_location_table.csv").exists() else None
STATE = {"inc": pd.DataFrame(), "com": pd.DataFrame(), "version": 0}
_CACHE: dict = {}


# ------------------------------------------------------------------ helpers
def inc_all():
    return pd.concat([INC0, STATE["inc"]], ignore_index=True) if len(STATE["inc"]) else INC0


def com_all():
    return pd.concat([COM0, STATE["com"]], ignore_index=True) if len(STATE["com"]) else COM0


def max_date():
    return max(inc_all().ts.max(), com_all().ts.max()).normalize()


def parse_asof(asof: Optional[str]) -> pd.Timestamp:
    md = max_date()
    if not asof:
        d = md
    else:
        try:
            d = pd.Timestamp(asof).normalize()
        except Exception:
            raise HTTPException(422, f"Bad asof date: {asof!r} (use YYYY-MM-DD)")
    d = min(max(d, MIN_ASOF), md)
    return d + pd.Timedelta(hours=23, minutes=59)


def resolve_weights(lens: str, weights: Optional[str] = None) -> dict:
    if lens == "Custom":
        try:
            raw = json.loads(weights or "{}")
            raw = {k: max(float(raw.get(k, 0)), 0) for k in COMPONENTS}
        except Exception:
            raise HTTPException(422, "weights must be JSON like {\"open_incident_load\":20,...}")
        tot = sum(raw.values())
        if tot <= 0:
            raise HTTPException(422, "Custom weights must not all be zero")
        return {k: v / tot for k, v in raw.items()}
    if lens not in LENSES:
        raise HTTPException(404, f"Unknown lens {lens!r}. Use one of {list(LENSES)} or 'Custom'.")
    return dict(LENSES[lens])


def scored(asof_ts: pd.Timestamp, w: dict) -> pd.DataFrame:
    key = (asof_ts, tuple(sorted(w.items())), STATE["version"])
    if key not in _CACHE:
        F = build_features(loc, inc_all(), com_all(), EB, asof_ts)
        S = score(F, w).join(rank_stability(F, w))
        S = add_receipts(S, asof_ts).rename(columns={"rank_confidence": "rank_stability"})
        if len(_CACHE) >= 24:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = S
    return _CACHE[key]


def records(df: pd.DataFrame, decimals=3):
    return json.loads(df.to_json(orient="records", date_format="iso", double_precision=decimals))


def get_S(asof, lens, weights):
    return scored(parse_asof(asof), resolve_weights(lens, weights)), parse_asof(asof)


def item_tables(lid, asof_ts):
    i = inc_all()[(inc_all().location_id == lid) & (inc_all().ts < asof_ts)].copy()
    closed = i.ts + pd.to_timedelta(i.resolution_hours, unit="h")
    i["open"] = ~((i.resolved == 1) & (closed <= asof_ts))
    c = com_all()[(com_all().location_id == lid) & (com_all().ts < asof_ts)].copy()
    cc = c.ts + pd.to_timedelta(c.days_to_resolution, unit="D")
    c["open"] = ~(c.status.astype(str).str.lower().isin(CLOSED_STATUS) & (cc <= asof_ts))
    return i, c


# ---- ingestion (same rules as the Streamlit app) -------------------------------------------------
def prep_incidents(df):
    df = df.copy()
    need = {"location_id", "timestamp", "severity"}
    if not need <= set(df.columns):
        return None, f"Incident file needs columns {sorted(need)}"
    if "incident_id" not in df:
        df["incident_id"] = [f"UP{STATE['version']}-{i:05d}" for i in range(len(df))]
    for c, d in [("incident_type", "Unknown"), ("injuries", 0), ("reported_by", "Upload"), ("resolved", 0)]:
        if c not in df:
            df[c] = d
    if "resolution_hours" not in df:
        df["resolution_hours"] = np.nan
    df["injuries"] = pd.to_numeric(df["injuries"], errors="coerce").fillna(0)
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sev_w"] = df.severity.astype(str).str.strip().str.lower().map(SEV_W).fillna(2)
    return df, ""


def prep_complaints(df):
    df = df.copy()
    need = {"location_id", "created_at"}
    if not need <= set(df.columns):
        return None, f"Complaint file needs columns {sorted(need)}"
    if "complaint_id" not in df:
        df["complaint_id"] = [f"UC{STATE['version']}-{i:05d}" for i in range(len(df))]
    for c, d in [("category", "Unknown"), ("priority_reported", "Normal"), ("status", "Open")]:
        if c not in df:
            df[c] = d
    if "days_to_resolution" not in df:
        df["days_to_resolution"] = np.nan
    df["ts"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["pri_w"] = df.priority_reported.astype(str).str.strip().str.lower().map(PRI_W).fillna(1)
    return df, ""


def _store(kind, prep):
    key, idc = ("inc", "incident_id") if kind == "incident" else ("com", "complaint_id")
    prep = prep[prep.location_id.isin(loc.location_id) & prep.ts.notna()]
    merged = pd.concat([STATE[key], prep], ignore_index=True).drop_duplicates(subset=idc, keep="last")
    STATE[key] = merged
    STATE["version"] += 1
    return len(prep)


def ingest_frame(raw: pd.DataFrame, name: str) -> str:
    kind = "incident" if "timestamp" in raw.columns else ("complaint" if "created_at" in raw.columns else None)
    if kind is None:
        return f"{name}: not recognised (need a 'timestamp' column for incidents or 'created_at' for complaints)."
    prep, err = (prep_incidents if kind == "incident" else prep_complaints)(raw)
    if prep is None:
        return f"{name}: {err}"
    before = len(prep)
    kept = _store(kind, prep)
    return f"{name}: added {kept} {kind} rows ({before - kept} dropped: unknown location_id or bad date)."


def state_info():
    return {"max_date": str(max_date().date()), "incidents_total": len(inc_all()), "complaints_total": len(com_all()),
            "extra_incidents": len(STATE["inc"]), "extra_complaints": len(STATE["com"])}


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health():
    return {"ok": True, **state_info()}


@app.get("/api/summary")
def summary(asof: Optional[str] = None, lens: str = "Safety-first", weights: Optional[str] = None):
    S, ts = get_S(asof, lens, weights)
    inc, com = inc_all(), com_all()
    return {
        "asof": str(ts.date()), **state_info(),
        "records_in_use": {"incidents": int((inc.ts < ts).sum()), "complaints": int((com.ts < ts).sum())},
        "kpis": {"open_incidents": int(S.open_inc_n.sum()), "open_complaints": int(S.open_com_n.sum()),
                 "locations_scored": len(S),
                 "stability_counts": S.rank_stability.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int).to_dict(),
                 "median_rank_interval_width": float((S.rank_p95 - S.rank_p5).median())},
        "lens": lens, "weights": resolve_weights(lens, weights),
        "lenses": {k: v for k, v in LENSES.items()}, "components": LABELS,
        "zones": sorted(S.zone.unique().tolist()), "synthetic_notice": SYNTH,
        "stability_help": ("Weights are perturbed 500 times. High = stays in the top 10 in at least 80% of runs; "
                           "Medium = stays in the top 50 in at least 80%. This measures robustness to policy weights, "
                           "not statistical certainty."),
    }


@app.get("/api/queue")
def queue(asof: Optional[str] = None, lens: str = "Safety-first", weights: Optional[str] = None,
          topN: int = Query(50, ge=1, le=1000), zones: Optional[str] = Query(None, description="comma-separated"),
          stability: Optional[str] = Query(None, description="comma-separated: High,Medium,Low"),
          lite: bool = Query(False, description="only id/lat/lon/rank/score (for map background)")):
    S, ts = get_S(asof, lens, weights)
    V = S
    if stability:
        V = V[V.rank_stability.isin([s.strip() for s in stability.split(",")])]
    if zones:
        V = V[V.zone.isin([z.strip() for z in zones.split(",")])]
    V = V.head(topN).reset_index()
    cols = ["rank", "location_id", "zone", "latitude", "longitude", "priority_score"]
    if not lite:
        cols += ["road_type", "road_condition", "rank_p5", "rank_p95", "p_top10", "p_top50", "rank_stability",
                 "open_inc_n", "open_com_n", "recent_injuries", "complaint_age", "voice_record_gap", "why", "evidence_gaps"]
        cols += [k + "_contrib" for k in COMPONENTS]
    return {"asof": str(ts.date()), "lens": lens, "count": len(V), "rows": records(V[cols], 2)}


@app.get("/api/location/{location_id}")
def location(location_id: str, asof: Optional[str] = None, lens: str = "Safety-first", weights: Optional[str] = None):
    S, ts = get_S(asof, lens, weights)
    if location_id not in S.index:
        raise HTTPException(404, f"Unknown location_id {location_id}")
    r = S.loc[location_id]
    i, c = item_tables(location_id, ts)
    m = pd.concat([i.set_index("ts").resample("MS").size().rename("incidents"),
                   c.set_index("ts").resample("MS").size().rename("complaints")], axis=1).fillna(0)
    m.index = m.index.strftime("%Y-%m")
    m.index.name = "month"
    summ = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in
            ["rank", "zone", "road_type", "road_condition", "pedestrian_activity", "population_density_per_km2",
             "near_school", "near_hospital", "latitude", "longitude", "priority_score", "rank_p5", "rank_p95", "p_top10",
             "p_top50", "rank_stability", "open_inc_n", "open_com_n", "recent_inc_n", "recent_injuries", "complaint_age",
             "inc_total", "com_total", "voice_record_gap", "why", "evidence_gaps"]}
    eb = None
    if EB_TABLE is not None and location_id in EB_TABLE.index:
        eb = {k: (v.item() if hasattr(v, "item") else v) for k, v in EB_TABLE.loc[location_id].items()}
    brief = (f"WORK ORDER BRIEF: {location_id} ({summ['zone']}, {summ['road_type']})\n"
             f"As of {ts.date()}, policy lens: {lens}\n\n"
             f"Priority rank #{summ['rank']} of {len(S)} (score {summ['priority_score']:.1f}; rank range "
             f"#{summ['rank_p5']}-#{summ['rank_p95']}; stability {summ['rank_stability']}).\n"
             f"Why: {summ['why']}.\n"
             f"Open items: {int(summ['open_inc_n'])} incident(s), {int(summ['open_com_n'])} complaint(s); "
             f"oldest open complaint {int(summ['complaint_age'])} days.\n"
             f"Context: {summ['road_type']} road, {summ['road_condition']} condition, {summ['pedestrian_activity']} pedestrian activity"
             f"{', school nearby' if summ['near_school'] else ''}{', hospital nearby' if summ['near_hospital'] else ''}.\n"
             f"Verify on site / missing information: {summ['evidence_gaps']}.\n"
             f"Statistical note: past incident counts alone do not distinguish this location from the citywide average; "
             f"justify the visit on the open items and exposure above.\n\n{SYNTH}\n")
    oi = i[i.open].sort_values("ts", ascending=False)[["incident_id", "ts", "incident_type", "severity", "injuries", "reported_by"]]
    oc = c[c.open].sort_values("ts", ascending=False)[["complaint_id", "ts", "category", "priority_reported", "status"]]
    return {"asof": str(ts.date()), "location_id": location_id, "summary": summ,
            "receipt": [{"component": k, "label": LABELS[k], "points": round(float(r[k + "_contrib"]), 2),
                         "percentile": round(float(r[k + "_pts"]), 1)} for k in COMPONENTS],
            "history": records(m.reset_index(), 0), "open_incidents": records(oi), "open_complaints": records(oc),
            "eb": eb, "brief": brief}


@app.get("/api/gap")
def gap(asof: Optional[str] = None, lens: str = "Safety-first", weights: Optional[str] = None):
    S, ts = get_S(asof, lens, weights)
    g = S.reset_index()
    g["incident_pct"] = g.inc_total.rank(pct=True) * 100
    g["complaint_pct"] = g.com_total.rank(pct=True) * 100
    cols = ["location_id", "zone", "inc_total", "com_total", "incident_pct", "complaint_pct", "voice_record_gap"]
    return {"asof": str(ts.date()), "points": records(g[cols], 1),
            "complaints_exceed_records": records(g.sort_values("voice_record_gap", ascending=False)[cols].head(15), 1),
            "records_exceed_complaints": records(g.sort_values("voice_record_gap")[cols].head(15), 1),
            "note": "Inspection leads, not proven problems."}


class PlanReq(BaseModel):
    asof: Optional[str] = None
    lens: str = "Safety-first"
    weights: Optional[dict] = None
    budget: float = 60


@app.post("/api/plan")
def make_plan(req: PlanReq):
    if not 1 <= req.budget <= 1000:
        raise HTTPException(422, "budget must be between 1 and 1000 crew-days")
    w = json.dumps(req.weights) if req.weights else None
    S, ts = get_S(req.asof, req.lens, w)
    key = ("compare", ts, tuple(sorted(resolve_weights(req.lens, w).items())), STATE["version"], req.budget)
    if key not in _CACHE:
        _CACHE[key] = compare(S, req.budget)
    comp = _CACHE[key]
    chosen, used = plan(S, req.budget, "FixFirst")
    P = add_cost(S).loc[chosen].reset_index()
    bundles = []
    if len(P):
        k = max(1, int(np.ceil(len(P) / 6)))
        P["bundle"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(P[["latitude", "longitude"]]) + 1
        B = P.groupby("bundle").agg(locations=("location_id", list), n=("location_id", "size"), crew_days=("cost_days", "sum"),
                                    open_incidents=("open_inc_n", "sum"), open_complaints=("open_com_n", "sum"),
                                    lat=("latitude", "mean"), lon=("longitude", "mean")).round(2).reset_index()
        bundles = records(B, 2)
    cols = ["bundle", "rank", "location_id", "zone", "latitude", "longitude", "priority_score", "cost_days",
            "open_inc_n", "open_com_n", "why"] if len(P) else []
    return {"asof": str(ts.date()), "budget": req.budget, "crew_days_used": round(float(used), 1), "cost_note": COST_NOTE,
            "comparison": records(comp, 1), "locations": records(P[cols], 2) if len(P) else [], "bundles": bundles,
            "note": "Accounting of EXISTING open items, not a forecast of future incidents."}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    msgs = []
    for f in files:
        try:
            msgs.append(ingest_frame(pd.read_csv(f.file), f.filename))
        except Exception as e:  # unreadable CSV
            msgs.append(f"{f.filename}: could not read CSV ({e})")
    return {"messages": msgs, **state_info()}


class Report(BaseModel):
    kind: Literal["incident", "complaint"]
    location_id: str
    timestamp: str
    severity: str = "Medium"
    injuries: int = 0
    incident_type: str = "Unknown"
    priority: str = "Normal"
    category: str = "Unknown"
    status: str = "Open"


@app.post("/api/report")
def report(r: Report):
    """Log ONE new incident or complaint (form input)."""
    if r.location_id not in set(loc.location_id):
        raise HTTPException(404, f"Unknown location_id {r.location_id}")
    if r.kind == "incident":
        raw = pd.DataFrame([dict(location_id=r.location_id, timestamp=r.timestamp, severity=r.severity,
                                 injuries=r.injuries, incident_type=r.incident_type, reported_by="Form", resolved=0)])
    else:
        raw = pd.DataFrame([dict(location_id=r.location_id, created_at=r.timestamp, priority_reported=r.priority,
                                 category=r.category, status=r.status)])
    msg = ingest_frame(raw, f"{r.kind} form")
    if " added 0 " in msg:
        raise HTTPException(422, "Could not parse the date. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS.")
    return {"message": msg, **state_info()}


@app.post("/api/demo-batch")
def demo_batch():
    """Simulated burst at 3 locations in Jan 2026. Front end should then set asof = returned max_date."""
    rng = np.random.default_rng(7)
    ids = [str(x) for x in rng.choice(loc.location_id.values, 3, replace=False)]
    top_sev = max(INC0.severity.unique(), key=lambda s: SEV_W.get(str(s).lower(), 2))
    top_pri = max(COM0.priority_reported.unique(), key=lambda s: PRI_W.get(str(s).lower(), 1))
    ri, rc = [], []
    for lid in ids:
        for k in range(6):
            ri.append(dict(incident_id=f"DEMO-{lid}-{k}", location_id=lid, timestamp=f"2026-01-{2 + k % 7:02d} {8 + k:02d}:15:00",
                           incident_type=INC0.incident_type.iloc[0], severity=top_sev, injuries=int(rng.integers(1, 4)),
                           reported_by="Demo", resolved=0))
        for k in range(4):
            rc.append(dict(complaint_id=f"DEMOC-{lid}-{k}", location_id=lid, created_at=f"2026-01-{3 + k:02d} {10 + k:02d}:00:00",
                           category=COM0.category.iloc[0], priority_reported=top_pri, status="Open"))
    msgs = [ingest_frame(pd.DataFrame(ri), "demo incidents"), ingest_frame(pd.DataFrame(rc), "demo complaints")]
    return {"locations": ids, "messages": msgs, **state_info()}


@app.post("/api/reset")
def reset():
    STATE.update(inc=pd.DataFrame(), com=pd.DataFrame(), version=STATE["version"] + 1)
    return {"message": "Reset to original data", **state_info()}


@app.get("/api/evidence")
def evidence():
    def j(name):
        p = BASE / "outputs" / "models" / name
        return json.loads(p.read_text()) if p.exists() else None
    probes = [
        ["Incident resolved", "AUC", 0.495, 0.500], ["Incident injured", "AUC", 0.516, 0.500],
        ["Incident severity (4 classes)", "Accuracy", 0.361, 0.382], ["Incident type (6 classes)", "Accuracy", 0.173, 0.163],
        ["Incident reported_by (4 classes)", "Accuracy", 0.499, 0.501], ["Incident log resolution hours", "R2", -0.003, 0.0],
        ["Complaint status (5 classes)", "Accuracy", 0.402, 0.410], ["Complaint priority (3 classes)", "Accuracy", 0.564, 0.567],
        ["Complaint log days to resolution", "R2", -0.015, 0.0]]
    return {
        "synthetic_notice": SYNTH,
        "association_scan": {"tests": 208, "passed_bonferroni": 0, "threshold": 2.4e-4, "strongest_raw_p": 0.008},
        "probes": {"setup": "HistGradientBoosting, train Jan-Aug, test Sep-Dec",
                   "rows": [dict(target=a, metric=b, model=c, baseline=d) for a, b, c, d in probes]},
        "location_probe": {"model_deviance": 1.135, "global_mean_deviance": 1.098, "top50_model": 0.0547, "top50_random": 0.0557},
        "eb_summary": j("eb_summary.json"), "power_check": j("power_check.json"),
        "rank_intervals": (records(EB_TABLE.sort_values("rank_median").head(30).reset_index()
                                   [["location_id", "rank_5pct", "rank_median", "rank_95pct", "p_top50"]], 1)
                           if EB_TABLE is not None else []),
        "triage_method": {"lenses": LENSES, "components": LABELS, "severity_weights": SEV_W, "priority_weights": PRI_W,
                          "recent_days": RECENT_DAYS, "cost_model": COST_NOTE},
        "disclosure": ["Data: only the provided locations, incidents and complaints CSVs. No extra data.",
                       "From scratch: Poisson GLM (IRLS) and empirical-Bayes Gamma-Poisson.",
                       "scikit-learn: HistGradientBoosting baseline probes and KMeans route bundling.",
                       "No API keys, no pretrained models, no LLM or external services. Briefs are template-generated."],
        "limitations": ["Synthetic data: nothing here applies to a real city.",
                        "About 10 incidents per location per year: history is too noisy to rank by.",
                        "Triage ranks existing backlog and exposure; it does not predict or prove prevention.",
                        "Cost model and severity/priority weights are illustrative assumptions.",
                        "Decision support only; not for real public-safety decisions."],
    }


# ================================================================== case-level work plan
# Officer inputs crews / days / budget -> open incident CASES that fit, in priority order, plus very-high-priority
# cases that do NOT fit (with the exact shortfall). Accounting of EXISTING open cases, not a forecast.
import math  # noqa: E402
from pydantic import Field  # noqa: E402

CASE_W = {"severity": .35, "harm": .20, "age": .15, "location": .30}
CASE_W_LABELS = {"severity": "Case severity", "harm": "Injuries", "age": "Days open (capped at 90)",
                 "location": "Location priority score (current lens)"}
SEV_LEVEL = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
DEFAULT_COST_DAYS = {"Low": 0.25, "Medium": 0.5, "High": 0.75, "Critical": 1.0}
DEFAULT_COST_MONEY = {"Low": 3000.0, "Medium": 6000.0, "High": 12000.0, "Critical": 20000.0}
DEFAULT_OVERHEAD_DAYS, DEFAULT_OVERHEAD_MONEY = 0.25, 2000.0
VERY_HIGH_PCTL = 0.95   # top 5% of open cases by case priority = "very high priority"
WORK_NOTE = ("Variable cost model: each incident's cost is computed dynamically based on its priority score, "
             "severity tier, and reported injuries, plus mobilisation overhead when moving to a new location.")


class WorkReq(BaseModel):
    asof: Optional[str] = None
    lens: str = "Safety-first"
    weights: Optional[dict] = None
    crews: int = Field(4, ge=1, le=50)
    days: float = Field(5, ge=0.5, le=60)
    budget: float = Field(300000, ge=0, le=1e9)
    overhead_days: float = Field(DEFAULT_OVERHEAD_DAYS, ge=0, le=5)
    overhead_money: float = Field(DEFAULT_OVERHEAD_MONEY, ge=0, le=1e7)
    cost_days: Optional[dict] = None
    cost_money: Optional[dict] = None


def _costs(over, default, name):
    out = dict(default)
    for k, v in (over or {}).items():
        k = str(k).strip().title()
        if k not in out:
            raise HTTPException(422, f"{name}: unknown severity {k!r}; use {list(out)}")
        try:
            v = float(v)
        except Exception:
            raise HTTPException(422, f"{name}[{k}] must be a number")
        if v <= 0:
            raise HTTPException(422, f"{name}[{k}] must be > 0")
        out[k] = v
    return out


@app.get("/api/workplan/defaults")
def workplan_defaults():
    return {"inputs": {"crews": 4, "days": 5, "budget": 300000}, "currency": "INR",
            "cost_days": DEFAULT_COST_DAYS, "cost_money": DEFAULT_COST_MONEY,
            "overhead_days": DEFAULT_OVERHEAD_DAYS, "overhead_money": DEFAULT_OVERHEAD_MONEY,
            "case_priority_weights": CASE_W, "case_priority_labels": CASE_W_LABELS,
            "very_high_rule": f"Top {int((1 - VERY_HIGH_PCTL) * 100)}% of open cases by case priority", "note": WORK_NOTE}


@app.post("/api/workplan")
def workplan(req: WorkReq):
    w = json.dumps(req.weights) if req.weights else None
    S, ts = get_S(req.asof, req.lens, w)
    cd = _costs(req.cost_days, DEFAULT_COST_DAYS, "cost_days")
    cm = _costs(req.cost_money, DEFAULT_COST_MONEY, "cost_money")
    od, om = req.overhead_days, req.overhead_money

    i = inc_all()
    i = i[i.ts < ts].copy()
    i = i[~((i.resolved == 1) & ((i.ts + pd.to_timedelta(i.resolution_hours, unit="h")) <= ts))]
    i = i[i.location_id.isin(S.index)].copy()
    if i.empty:
        raise HTTPException(404, "No open incident cases at this as-of date")
    i["level"] = i.sev_w.round().clip(1, 4).astype(int).map(SEV_LEVEL)
    i["age_days"] = (ts - i.ts).dt.total_seconds() / 86400
    pts = {"severity": (i.sev_w.clip(1, 4) - 1) / 3 * 100, "harm": i.injuries.clip(0, 3) / 3 * 100,
           "age": i.age_days.clip(0, 90) / 90 * 100, "location": S.priority_score.reindex(i.location_id).values}
    for k, v in pts.items():
        i["pt_" + k] = v * CASE_W[k]
    i["case_priority"] = sum(i["pt_" + k] for k in CASE_W)
    i = i.sort_values(["case_priority", "incident_id"], ascending=[False, True]).reset_index(drop=True)
    cut = float(i.case_priority.quantile(VERY_HIGH_PCTL))
    i["very_high"] = i.case_priority >= cut

    cap = [float(req.days)] * req.crews
    money_left = float(req.budget)
    crew_locs = [set() for _ in range(req.crews)]
    sl, sg, cnt = [0.0] * req.crews, [0.0] * req.crews, [0] * req.crews
    sched, deferred = [], []
    lat_of, lon_of, rank_of = S.latitude, S.longitude, S["rank"]
    zone_of = S.zone

    for r in i.itertuples():
        base_d, base_m = cd[r.level], cm[r.level]
        # Variable cost model: priority-dependent scaling + injury complexity
        pri_factor = 0.85 + (float(r.case_priority) / 100.0) * 0.35
        injury_m = int(r.injuries) * 2000.0
        injury_d = int(r.injuries) * 0.10
        m0 = round(base_m * pri_factor + injury_m, -2)
        d0 = round(base_d * pri_factor + injury_d, 2)
        lid = r.location_id
        lat, lon = float(lat_of[lid]), float(lon_of[lid])
        opts = []
        for c in range(req.crews):
            mob = lid not in crew_locs[c]
            opts.append((c, mob, d0 + (od if mob else 0.0), m0 + (om if mob else 0.0)))
        feas = [o for o in opts if o[2] <= cap[o[0]] + 1e-9 and o[3] <= money_left + 1e-9]
        base = dict(case_id=r.incident_id, location_id=lid, zone=zone_of[lid], latitude=lat, longitude=lon,
                    severity=r.level, injuries=int(r.injuries), incident_type=r.incident_type, age_days=int(r.age_days),
                    case_priority=round(float(r.case_priority), 1), location_rank=int(rank_of[lid]),
                    very_high=bool(r.very_high),
                    points={k: round(float(getattr(r, "pt_" + k)), 1) for k in CASE_W},
                    why=(f"{r.level} {r.incident_type}; {int(r.injuries)} injur{'y' if int(r.injuries) == 1 else 'ies'}; "
                         f"open {int(r.age_days)} days; location ranked #{int(rank_of[lid])}"))
        if feas:
            def key(o):
                c, mob = o[0], o[1]
                dist = 0.0 if (not mob or cnt[c] == 0) else math.hypot(lat - sl[c] / cnt[c], lon - sg[c] / cnt[c])
                return (mob, dist, -cap[c])
            c, mob, dn, mn = min(feas, key=key)
            sched.append({**base, "order": len(sched) + 1, "crew": c + 1, "day": min(int(req.days - cap[c]) + 1, math.ceil(req.days)),
                          "cost_days": round(dn, 2), "cost_money": round(mn), "mobilisation": mob})
            cap[c] -= dn
            money_left -= mn
            if mob:
                crew_locs[c].add(lid)
                sl[c] += lat
                sg[c] += lon
                cnt[c] += 1
        else:
            money_ok = any(o[3] <= money_left + 1e-9 for o in opts)
            crew_ok = any(o[2] <= cap[o[0]] + 1e-9 for o in opts)
            short_m = max(0.0, min(o[3] for o in opts) - money_left)
            short_d = min(max(0.0, o[2] - cap[o[0]]) for o in opts)
            reason = "Budget and crew capacity" if (not money_ok and not crew_ok) else ("Budget" if not money_ok else "Crew capacity")
            parts = []
            if not money_ok or (short_m > 0 and reason != "Crew capacity"):
                parts.append(f"INR {short_m:,.0f} more budget")
            if not crew_ok or reason == "Crew capacity":
                parts.append(f"{short_d:.2f} more crew-days on one crew")
            deferred.append({**base, "reason": reason, "shortfall_money": round(short_m), "shortfall_crew_days": round(short_d, 2),
                             "unlock": "Add " + " and ".join(parts) if parts else "Add capacity",
                             "cost_days": round(d0, 2), "cost_money": round(m0)})

    dvh = [d for d in deferred if d["very_high"]]
    dother = [d for d in deferred if not d["very_high"]]
    seen_locs = set().union(*crew_locs)
    new_locs = {d["location_id"] for d in dvh} - seen_locs
    ask_days = sum(d["cost_days"] for d in dvh) + od * len(new_locs)
    ask_money = sum(d["cost_money"] for d in dvh) + om * len(new_locs)
    nxt = dvh[:10]
    nxt_new = {d["location_id"] for d in nxt} - seen_locs
    nxt_days = sum(d["cost_days"] for d in nxt) + od * len(nxt_new)
    nxt_money = sum(d["cost_money"] for d in nxt) + om * len(nxt_new)
    used_days = sum(req.days - c for c in cap)
    crews_out = []
    for c in range(req.crews):
        mine = [s for s in sched if s["crew"] == c + 1]
        crews_out.append({"crew": c + 1, "days_capacity": req.days, "days_used": round(req.days - cap[c], 2),
                          "money": sum(s["cost_money"] for s in mine), "cases": len(mine),
                          "locations": sorted({s["location_id"] for s in mine})})
    vh_total = int(i.very_high.sum())
    return {
        "asof": str(ts.date()), "lens": req.lens, "currency": "INR",
        "inputs": {"crews": req.crews, "days": req.days, "budget": req.budget, "overhead_days": od, "overhead_money": om},
        "cost_model": {"cost_days": cd, "cost_money": cm, "note": WORK_NOTE},
        "case_priority_weights": CASE_W, "case_priority_labels": CASE_W_LABELS,
        "very_high_cut": round(cut, 1), "very_high_rule": f"Top {int((1 - VERY_HIGH_PCTL) * 100)}% of open cases by case priority",
        "capacity": {"crew_days_total": req.crews * req.days, "crew_days_used": round(used_days, 2),
                     "crew_days_remaining": round(sum(cap), 2), "budget_used": round(req.budget - money_left),
                     "budget_remaining": round(money_left),
                     "binding": ("both" if money_left < min(cm.values()) and max(cap) < min(cd.values()) else
                                 "budget" if money_left < min(cm.values()) else
                                 "crew capacity" if max(cap) < min(cd.values()) else "neither")},
        "summary": {"open_cases": len(i), "scheduled": len(sched), "deferred": len(deferred),
                    "share_scheduled": f"{len(sched) / len(i):.1%}", "locations_visited": len(seen_locs),
                    "very_high_total": vh_total, "very_high_scheduled": vh_total - len(dvh), "very_high_deferred": len(dvh),
                    "severity_load_scheduled": float(sum({"Low": 1, "Medium": 2, "High": 3, "Critical": 4}[s["severity"]] for s in sched)),
                    "injuries_in_scheduled_cases": int(sum(s["injuries"] for s in sched))},
        "ask_to_clear_very_high": {"extra_budget": round(ask_money), "extra_crew_days": round(ask_days, 1),
                                   "extra_crews_for_same_days": math.ceil(ask_days / req.days) if dvh else 0,
                                   "next_10": {"extra_budget": round(nxt_money), "extra_crew_days": round(nxt_days, 1),
                                               "extra_crews_for_same_days": math.ceil(nxt_days / req.days) if nxt else 0}},
        "scheduled": sched, "crews": crews_out,
        "deferred_very_high": dvh[:100], "deferred_very_high_total": len(dvh),
        "deferred_other": dother[:50], "deferred_other_total": len(dother),
        "note": "Plans EXISTING open cases only; it does not predict future incidents. Costs are illustrative.",
    }
