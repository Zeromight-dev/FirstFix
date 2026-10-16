"""
FixFirst - Streamlit app.   Run from the FirstFix folder:   streamlit run app.py
Pages: Priority map & queue | Location receipt | Budget planner | Evidence & limits
Input workflow: pick an as-of date + policy lens, optionally upload a new batch of incidents/complaints
(or press 'Add demo batch'), and the ranking is rebuilt from scratch using only records before that date.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.cluster import KMeans

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from planner import COST_NOTE, add_cost, compare, plan  # noqa: E402
from priority import (CLOSED_STATUS, COMPONENTS, LABELS, LENSES, PRI_W, RECENT_DAYS, SEV_W,  # noqa: E402
                      add_receipts, build_features, load_all, rank_stability, score)

st.set_page_config(page_title="FixFirst", page_icon="🛠️", layout="wide")
SYNTH = ("Synthetic educational data. Coordinates and records are simulated and must not be read as real civic reports "
         "or used for real public-safety decisions.")


# ------------------------------------------------------------------ data
@st.cache_data
def get_base():
    return load_all()


@st.cache_data
def get_eb_table():
    p = BASE / "outputs" / "models" / "eb_location_table.csv"
    return pd.read_csv(p).set_index("location_id") if p.exists() else None


@st.cache_data
def get_json(name):
    p = BASE / "outputs" / "models" / name
    return json.loads(p.read_text()) if p.exists() else None


loc, inc0, com0, eb = get_base()
if "extra_inc" not in st.session_state:
    st.session_state.extra_inc = pd.DataFrame()
    st.session_state.extra_com = pd.DataFrame()
    st.session_state.notice = ""


def prep_incidents(df):
    df = df.copy()
    need = {"location_id", "timestamp", "severity"}
    if not need <= set(df.columns):
        return None, f"Incident file needs columns {sorted(need)}"
    df["incident_id"] = df.get("incident_id", pd.Series([f"UP{i:05d}" for i in range(len(df))], index=df.index))
    for c, d in [("incident_type", "Unknown"), ("injuries", 0), ("reported_by", "Upload"), ("resolved", 0)]:
        if c not in df:
            df[c] = d
    if "resolution_hours" not in df:
        df["resolution_hours"] = np.nan
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sev_w"] = df.severity.astype(str).str.strip().str.lower().map(SEV_W).fillna(2)
    return df, ""


def prep_complaints(df):
    df = df.copy()
    need = {"location_id", "created_at"}
    if not need <= set(df.columns):
        return None, f"Complaint file needs columns {sorted(need)}"
    df["complaint_id"] = df.get("complaint_id", pd.Series([f"UC{i:05d}" for i in range(len(df))], index=df.index))
    for c, d in [("category", "Unknown"), ("priority_reported", "Normal"), ("status", "Open")]:
        if c not in df:
            df[c] = d
    if "days_to_resolution" not in df:
        df["days_to_resolution"] = np.nan
    df["ts"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["pri_w"] = df.priority_reported.astype(str).str.strip().str.lower().map(PRI_W).fillna(1)
    return df, ""


def add_upload(file):
    raw = pd.read_csv(file)
    kind = "incident" if "timestamp" in raw.columns else ("complaint" if "created_at" in raw.columns else None)
    if kind is None:
        return "Could not recognise the file: need a 'timestamp' (incidents) or 'created_at' (complaints) column."
    prep, err = (prep_incidents if kind == "incident" else prep_complaints)(raw)
    if prep is None:
        return err
    before = len(prep)
    prep = prep[prep.location_id.isin(loc.location_id) & prep.ts.notna()]
    key = "extra_inc" if kind == "incident" else "extra_com"
    st.session_state[key] = pd.concat([st.session_state[key], prep], ignore_index=True)
    return f"Added {len(prep)} {kind} rows from {file.name} ({before - len(prep)} dropped: unknown location_id or bad date)."


def demo_batch():
    rng = np.random.default_rng(7)
    ids = rng.choice(loc.location_id.values, 3, replace=False)
    top_sev = max(inc0.severity.unique(), key=lambda s: SEV_W.get(str(s).lower(), 2))
    top_pri = max(com0.priority_reported.unique(), key=lambda s: PRI_W.get(str(s).lower(), 1))
    ri, rc = [], []
    for lid in ids:
        for k in range(6):
            ri.append(dict(incident_id=f"DEMO{lid}{k}", location_id=lid, timestamp=f"2026-01-0{2 + k % 7} 0{8 + k}:15:00",
                           incident_type=inc0.incident_type.iloc[0], severity=top_sev, injuries=int(rng.integers(1, 4)),
                           reported_by="Demo", resolved=0, resolution_hours=np.nan))
        for k in range(4):
            rc.append(dict(complaint_id=f"DEMOC{lid}{k}", location_id=lid, created_at=f"2026-01-0{3 + k} 1{k}:00:00",
                           category=com0.category.iloc[0], priority_reported=top_pri, status="Open", days_to_resolution=np.nan))
    di, _ = prep_incidents(pd.DataFrame(ri))
    dc, _ = prep_complaints(pd.DataFrame(rc))
    st.session_state.extra_inc = pd.concat([st.session_state.extra_inc, di], ignore_index=True)
    st.session_state.extra_com = pd.concat([st.session_state.extra_com, dc], ignore_index=True)
    return ids


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛠️ FixFirst")
st.sidebar.caption("Civic triage with receipts: which locations to review first, on what evidence, with what confidence.")
page = st.sidebar.radio("Page", ["Priority map & queue", "Location receipt", "Budget planner", "Evidence & limits"])

st.sidebar.markdown("### 1. Input")
inc_all = pd.concat([inc0, st.session_state.extra_inc], ignore_index=True) if len(st.session_state.extra_inc) else inc0
com_all = pd.concat([com0, st.session_state.extra_com], ignore_index=True) if len(st.session_state.extra_com) else com0
max_d = max(inc_all.ts.max(), com_all.ts.max()).normalize()
asof_date = st.sidebar.date_input("As-of date (only earlier records are used)", value=max_d.date(),
                                  min_value=pd.Timestamp("2025-02-01").date(), max_value=max_d.date())
ASOF = pd.Timestamp(asof_date) + pd.Timedelta(hours=23, minutes=59)

lens_choice = st.sidebar.selectbox("Policy lens", list(LENSES) + ["Custom"])
if lens_choice == "Custom":
    with st.sidebar.expander("Custom weights", expanded=True):
        raw_w = {k: st.slider(LABELS[k], 0, 100, int(LENSES["Safety-first"][k] * 100)) for k in COMPONENTS}
    tot = max(sum(raw_w.values()), 1)
    WEIGHTS = {k: v / tot for k, v in raw_w.items()}
else:
    WEIGHTS = LENSES[lens_choice]

with st.sidebar.expander("New data batch (optional)"):
    ups = st.file_uploader("Upload incidents.csv or public_complaints.csv rows (same columns)", type="csv", accept_multiple_files=True)
    if st.button("Add uploaded files") and ups:
        st.session_state.notice = " ".join(add_upload(f) for f in ups)
        st.rerun()
    if st.button("Add demo batch (simulated burst at 3 locations)"):
        ids_demo = demo_batch()
        st.session_state.notice = f"Demo batch added for {', '.join(ids_demo)}: 6 high-severity open incidents and 4 urgent complaints each, dated Jan 2026."
        st.rerun()
    if st.button("Reset to original data"):
        st.session_state.extra_inc, st.session_state.extra_com, st.session_state.notice = pd.DataFrame(), pd.DataFrame(), ""
        st.rerun()
if st.session_state.notice:
    st.sidebar.success(st.session_state.notice)
st.sidebar.caption(f"Records in use: {len(inc_all[inc_all.ts < ASOF]):,} incidents, {len(com_all[com_all.ts < ASOF]):,} complaints.")
st.sidebar.warning(SYNTH)


# ------------------------------------------------------------------ compute
@st.cache_data(show_spinner="Rebuilding features, scores and rank stability...")
def compute(asof, weights_items, n_inc, n_com, _inc, _com):
    w = dict(weights_items)
    F = build_features(loc, _inc, _com, eb, asof)
    S = score(F, w).join(rank_stability(F, w))
    return add_receipts(S, asof)


S = compute(ASOF, tuple(sorted(WEIGHTS.items())), len(inc_all), len(com_all), inc_all, com_all)
ebt = get_eb_table()


def open_items(lid):
    i = inc_all[(inc_all.location_id == lid) & (inc_all.ts < ASOF)].copy()
    closed = i.ts + pd.to_timedelta(i.resolution_hours, unit="h")
    i["open"] = ~((i.resolved == 1) & (closed <= ASOF))
    c = com_all[(com_all.location_id == lid) & (com_all.ts < ASOF)].copy()
    cc = c.ts + pd.to_timedelta(c.days_to_resolution, unit="D")
    c["open"] = ~(c.status.astype(str).str.lower().isin(CLOSED_STATUS) & (cc <= ASOF))
    return i, c


# ================================================================== PAGE 1
if page == "Priority map & queue":
    st.title("Where should we fix first?")
    st.caption(f"As of **{asof_date}** · lens **{lens_choice}** · ranking built from open items, recent harm and exposure, "
               "not from forecasts.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Open incidents", f"{int(S.open_inc_n.sum()):,}")
    k2.metric("Open complaints", f"{int(S.open_com_n.sum()):,}")
    k3.metric("Locations with High/Medium rank confidence", int((S.rank_confidence != "Low").sum()))
    k4.metric("Locations scored", len(S))
    st.info("**How to read the confidence column:** weights are perturbed 500 times; *High* means the location stays in the top 10 in "
            "at least 80% of runs. Past incident counts are NOT used to justify rank: long-run history is statistically indistinguishable "
            "from chance (see 'Evidence & limits').")

    f1, f2, f3 = st.columns([1, 2, 2])
    topn = f1.slider("Show top N", 10, 200, 50, step=10)
    zones = f2.multiselect("Zone filter", sorted(S.zone.unique()))
    confs = f3.multiselect("Rank confidence", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    V = S[S.rank_confidence.isin(confs)]
    if zones:
        V = V[V.zone.isin(zones)]
    V = V.head(topn).copy()

    tab_map, tab_gap = st.tabs(["Map & ranked queue", "Voice vs record gap"])
    with tab_map:
        mp = S.reset_index()
        mp["color"] = [[150, 150, 150, 50]] * len(mp)
        t = V.reset_index()
        t["score_txt"] = t.priority_score.round(1).astype(str)
        tt = (t.priority_score - t.priority_score.min()) / max(t.priority_score.max() - t.priority_score.min(), 1e-9)
        t["color"] = [[255, int(210 * (1 - v)), 40, 220] for v in tt]
        view = pdk.ViewState(latitude=float(loc.latitude.mean()), longitude=float(loc.longitude.mean()), zoom=9.6)
        deck = pdk.Deck(layers=[
            pdk.Layer("ScatterplotLayer", data=mp[["longitude", "latitude", "color"]], get_position="[longitude, latitude]",
                      get_fill_color="color", get_radius=90, radius_min_pixels=2),
            pdk.Layer("ScatterplotLayer", data=t[["longitude", "latitude", "color", "rank", "location_id", "zone", "score_txt"]],
                      get_position="[longitude, latitude]", get_fill_color="color", get_radius=250, radius_min_pixels=6, pickable=True)],
            initial_view_state=view, tooltip={"text": "#{rank} {location_id} ({zone})\nScore {score_txt}"})
        st.pydeck_chart(deck)
        st.caption("Red/orange = highest priority in the current view; grey = all other locations. Coordinates are simulated.")
        show = V.reset_index()[["rank", "location_id", "zone", "priority_score", "rank_p5", "rank_p95", "rank_confidence",
                                "open_inc_n", "open_com_n", "recent_injuries", "complaint_age", "why", "evidence_gaps"]]
        show.columns = ["Rank", "Location", "Zone", "Score", "Rank p5", "Rank p95", "Confidence", "Open incidents",
                        "Open complaints", f"Injuries ({RECENT_DAYS}d)", "Oldest complaint (d)", "Why", "Evidence gaps / what to verify"]
        show["Score"] = show["Score"].round(1)
        st.dataframe(show, hide_index=True)
        st.download_button("Download this queue (CSV)", show.to_csv(index=False), "fixfirst_queue.csv", "text/csv")
    with tab_gap:
        st.markdown("Locations where citizens complain much more (or less) than official incident records show. "
                    "These are **inspection leads**, not proven problems.")
        g = S.reset_index()
        fig = px.scatter(g, x=g.inc_total.rank(pct=True) * 100, y=g.com_total.rank(pct=True) * 100, color="voice_record_gap",
                         hover_name="location_id", labels={"x": "Incident record percentile", "y": "Complaint percentile",
                                                           "color": "Gap"}, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig)
        c1, c2 = st.columns(2)
        cols = ["location_id", "zone", "inc_total", "com_total", "voice_record_gap"]
        c1.markdown("**Complaints exceed records (verify on site)**")
        c1.dataframe(S.reset_index().sort_values("voice_record_gap", ascending=False)[cols].head(15).round(0), hide_index=True)
        c2.markdown("**Records exceed complaints (possible under-reporting area)**")
        c2.dataframe(S.reset_index().sort_values("voice_record_gap")[cols].head(15).round(0), hide_index=True)

# ================================================================== PAGE 2
elif page == "Location receipt":
    st.title("Location receipt")
    default = S.index[0]
    lid = st.selectbox("Location", S.index, index=0, format_func=lambda x: f"#{S.loc[x, 'rank']}  {x}  ({S.loc[x, 'zone']})")
    r = S.loc[lid]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Priority rank", f"#{int(r['rank'])}", f"range #{int(r.rank_p5)}-#{int(r.rank_p95)} (90%)", delta_color="off")
    c2.metric("Score", f"{r.priority_score:.1f} / 100")
    c3.metric("Rank confidence", r.rank_confidence)
    c4.metric("Open items", f"{int(r.open_inc_n)} inc · {int(r.open_com_n)} complaints")
    st.markdown(f"**Why:** {r.why}")
    st.markdown(f"**Evidence gaps / verify:** {r.evidence_gaps}")
    st.caption(f"{r.road_type} road, condition {r.road_condition}, {r.pedestrian_activity} pedestrian activity, "
               f"density {int(r.population_density_per_km2):,}/km²")

    left, right = st.columns(2)
    contrib = pd.DataFrame({"Component": [LABELS[k] for k in COMPONENTS], "Points": [r[k + "_contrib"] for k in COMPONENTS]})
    fig = px.bar(contrib.sort_values("Points"), x="Points", y="Component", orientation="h", title="Score receipt (points per component)")
    left.plotly_chart(fig)
    i, c = open_items(lid)
    m = pd.concat([i.set_index("ts").resample("MS").size().rename("Incidents"),
                   c.set_index("ts").resample("MS").size().rename("Complaints")], axis=1).fillna(0).reset_index()
    fig2 = px.bar(m, x="ts", y=["Incidents", "Complaints"], barmode="group", title="Monthly history at this location",
                  labels={"ts": "", "value": "count"})
    right.plotly_chart(fig2)

    if ebt is not None and lid in ebt.index:
        e = ebt.loc[lid]
        st.markdown(f"**Long-run statistics (empirical-Bayes model):** {int(e.incidents_2025)} incidents in 2025. Shrunk rate "
                    f"{e.eb_rate_per_yr:.1f}/yr (90% interval {e.rate_ci5:.1f}-{e.rate_ci95:.1f}); rank on history alone "
                    f"#{int(e.rank_5pct)}-#{int(e.rank_95pct)}; P(top-50) = {e.p_top50:.0%}. "
                    "The citywide mean is ~10/yr, so history alone does not make this location stand out.")
    st.markdown("**Open incidents**")
    oi = i[i.open].sort_values("ts", ascending=False)[["incident_id", "ts", "incident_type", "severity", "injuries", "reported_by"]]
    st.dataframe(oi, hide_index=True)
    st.markdown("**Open complaints**")
    oc = c[c.open].sort_values("ts", ascending=False)[["complaint_id", "ts", "category", "priority_reported", "status"]]
    st.dataframe(oc, hide_index=True)

    brief = (f"WORK ORDER BRIEF: {lid} ({r.zone}, {r.road_type})\n"
             f"As of {asof_date}, policy lens: {lens_choice}\n\n"
             f"Priority rank #{int(r['rank'])} of {len(S)} (score {r.priority_score:.1f}; rank range #{int(r.rank_p5)}-#{int(r.rank_p95)}; "
             f"confidence {r.rank_confidence}).\n"
             f"Why: {r.why}.\n"
             f"Open items: {int(r.open_inc_n)} incident(s), {int(r.open_com_n)} complaint(s); oldest open complaint {int(r.complaint_age)} days.\n"
             f"Context: {r.road_type} road, {r.road_condition} condition, {r.pedestrian_activity} pedestrian activity"
             f"{', school nearby' if r.near_school else ''}{', hospital nearby' if r.near_hospital else ''}.\n"
             f"Verify on site / missing information: {r.evidence_gaps}.\n"
             f"Statistical note: past incident counts alone do not distinguish this location from the citywide average; "
             f"justify the visit on the open items and exposure above.\n\n{SYNTH}\n")
    st.download_button("Download work-order brief (.txt)", brief, f"brief_{lid}.txt")
    with st.expander("Preview brief"):
        st.code(brief)

# ================================================================== PAGE 3
elif page == "Budget planner":
    st.title("Budget planner")
    st.caption("Choose how many crew-days you have. FixFirst picks locations; we compare against simpler strategies by how much "
               "**existing backlog** each clears. This is accounting of known open items, not a forecast of future incidents.")
    st.caption(COST_NOTE)
    budget = st.slider("Crew-days available", 10, 300, 60, step=5)


    @st.cache_data(show_spinner="Comparing strategies...")
    def cached_compare(_S, key, b):
        return compare(_S, b)


    comp = cached_compare(S, (ASOF, tuple(sorted(WEIGHTS.items())), len(inc_all), len(com_all)), budget)
    st.dataframe(comp, hide_index=True)
    metric = st.selectbox("Chart metric", ["severity_backlog_cleared", "open_incidents_cleared", "open_complaints_cleared",
                                           "exposure_weighted_backlog"])
    fig = px.bar(comp, x="strategy", y=metric, color="strategy", title=f"{metric.replace('_', ' ')} at {budget} crew-days")
    st.plotly_chart(fig)
    st.caption("Strategies optimise different goals, so none wins every column. FixFirst optimises the selected policy lens; "
               "'Most past incidents first' ranks on history, which our tests show is no better than chance for the future.")

    chosen, used = plan(S, budget, "FixFirst")
    P = add_cost(S).loc[chosen].reset_index()
    k = max(1, int(np.ceil(len(P) / 6)))
    P["bundle"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(P[["latitude", "longitude"]]) + 1 if len(P) >= k else 1
    st.subheader(f"FixFirst plan: {len(P)} locations, {used:.1f} crew-days, {k} geographic bundle(s)")
    fig3 = px.scatter(P, x="longitude", y="latitude", color=P.bundle.astype(str), hover_name="location_id",
                      size="cost_days", labels={"color": "Bundle"}, title="Crew bundles (nearby locations grouped into trips)")
    st.plotly_chart(fig3)
    B = P.groupby("bundle").agg(locations=("location_id", lambda s: ", ".join(s)), n=("location_id", "size"),
                                crew_days=("cost_days", "sum"), open_incidents=("open_inc_n", "sum"),
                                open_complaints=("open_com_n", "sum")).round(1).reset_index()
    st.dataframe(B, hide_index=True)
    out = P[["bundle", "rank", "location_id", "zone", "priority_score", "cost_days", "open_inc_n", "open_com_n", "why"]]
    st.download_button("Download plan (CSV)", out.to_csv(index=False), "fixfirst_plan.csv", "text/csv")

# ================================================================== PAGE 4
else:
    st.title("Evidence & limits")
    st.warning(SYNTH)
    st.markdown("### 1. What we tested (and what the data allows us to claim)")
    st.markdown("We trained models from scratch on the provided CSVs only, with time-based splits and naive baselines. "
                "**Result: past incident counts and location attributes do not predict future incidents in this dataset.** "
                "A product that claimed otherwise would fail its own backtest, so FixFirst ranks on current-state facts and reports uncertainty.")
    st.markdown("**Association scan (`signal_scan.py`):** 208 tests across incidents, complaints and location counts; "
                "**0 passed** the Bonferroni threshold (p < 2.4e-4). Strongest raw p-value: 0.008 (about 10 hits at p<0.05 are expected by chance).")
    probes = pd.DataFrame([
        ["Incident resolved (open vs closed)", "AUC", "0.495", "0.500"],
        ["Incident injured (injuries>0)", "AUC", "0.516", "0.500"],
        ["Incident severity (4 classes)", "Accuracy", "0.361", "0.382 (majority)"],
        ["Incident type (6 classes)", "Accuracy", "0.173", "0.163 (majority)"],
        ["Incident reported_by (4 classes)", "Accuracy", "0.499", "0.501 (majority)"],
        ["Incident log resolution hours", "R2", "-0.003", "0"],
        ["Complaint status (5 classes)", "Accuracy", "0.402", "0.410 (majority)"],
        ["Complaint priority (3 classes)", "Accuracy", "0.564", "0.567 (majority)"],
        ["Complaint log days to resolution", "R2", "-0.015", "0"]],
        columns=["Target (HistGradientBoosting, train Jan-Aug, test Sep-Dec)", "Metric", "Model", "Baseline"])
    st.dataframe(probes, hide_index=True)
    st.markdown("**28-day location forecast probe:** model Poisson deviance 1.135 vs 1.098 for the global mean; "
                "top-50 locations captured 5.47% of next-28-day incidents vs 5.57% for a random pick.")

    st.markdown("### 2. Models trained from scratch (`eb_models.py`)")
    sm = get_json("eb_summary.json")
    if sm:
        a, b, c = st.columns(3)
        a.metric("Dispersion ratio (Poisson = 1)", f"{sm['dispersion_ratio']:.3f}", f"p = {sm['dispersion_p']:.3f}", delta_color="off")
        b.metric("Poisson GLM LRT", f"{sm['glm_lrt']:.1f} on {sm['glm_lrt_df']} df", f"p = {sm['glm_lrt_p']:.3f}", delta_color="off")
        c.metric("Locations with P(top-50) ≥ 0.5", sm["n_p_top50_ge_0.5"])
        st.markdown("**Validation: fit on Jan-Jun, predict Jul-Dec incident counts per location**")
        st.dataframe(pd.DataFrame(sm["validation"]), hide_index=True)
        lo, hi = sm["random_band_95"]
        st.markdown(f"Top-50 coverage of Jul-Dec incidents (random expectation 5.0%, 95% random band {lo:.2%}-{hi:.2%}): " +
                    ", ".join(f"{k} {v:.2%}" for k, v in sm["top50_coverage"].items()))
        st.markdown(f"Median 90% rank interval spans **{sm['median_rank_interval_width']:.0f} of 1000** places on history alone: "
                    "most locations are a statistical tie.")
    else:
        st.info("Run `python src/eb_models.py` to populate this section.")
    if ebt is not None:
        top = ebt.sort_values("rank_median").head(30).reset_index()
        fig = go.Figure(go.Scatter(x=top.location_id, y=top.rank_median, mode="markers",
                                   error_y=dict(type="data", symmetric=False, array=top.rank_95pct - top.rank_median,
                                                arrayminus=top.rank_median - top.rank_5pct)))
        fig.update_layout(title="Rank on long-run history alone: 90% intervals for the 'top 30' (wide = not distinguishable)",
                          yaxis_title="rank (1 = highest rate)", yaxis_autorange="reversed")
        st.plotly_chart(fig)

    pc = get_json("power_check.json")
    st.markdown("### 3. Power check: would our pipeline detect real structure if it existed?")
    if pc:
        st.dataframe(pd.DataFrame(pc["rows"]), hide_index=True)
        st.markdown(pc["summary"])
    else:
        st.info("Run `python src/power_check.py` to populate this section.")

    st.markdown("### 4. How the triage score works")
    st.markdown("Each component is converted to a 0-100 percentile among locations and combined with lens weights. "
                "Only records dated before the as-of date are used (open = not yet resolved at that date).")
    st.dataframe(pd.DataFrame(LENSES).T.rename(columns=LABELS), )
    st.markdown("Assumptions (editable in `src/priority.py`): severity weights " + str(SEV_W) + "; complaint priority weights "
                + str(PRI_W) + "; crew-day cost model: " + COST_NOTE)

    st.markdown("### 5. Tools and data disclosure")
    st.markdown("- **Data:** only the provided `locations.csv`, `incidents.csv`, `public_complaints.csv`. No extra data.\n"
                "- **Models:** Poisson GLM (IRLS) and empirical-Bayes Gamma-Poisson, written from scratch; scikit-learn used for baseline "
                "probes (HistGradientBoosting), KMeans route bundling.\n"
                "- **No** API keys, **no** pretrained models, **no** LLM or external services. Briefs are template-generated.\n"
                "- Libraries: pandas, numpy, scipy, scikit-learn, plotly, pydeck, streamlit.")
    st.markdown("### 6. Limitations")
    st.markdown("- Synthetic data: coordinates, timestamps and outcomes are simulated; nothing here applies to a real city.\n"
                "- With ~10 incidents per location per year, individual rates are too noisy to rank by history.\n"
                "- Triage ranks existing backlog and exposure; it does not predict future incidents or prove prevention.\n"
                "- Cost model and severity/priority weights are illustrative assumptions.\n"
                "- Decision support for officials; not for real public-safety decisions.")
