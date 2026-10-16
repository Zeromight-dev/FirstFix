"""Smoke test: python test_api.py   (run from FirstFix, no server needed)"""
import io

from fastapi.testclient import TestClient

from backend import app

c = TestClient(app)


def ok(r):
    assert r.status_code == 200, (r.status_code, r.text[:300])
    return r.json()


s = ok(c.get("/api/summary"))
print("summary:", s["asof"], s["kpis"]["open_incidents"], "open inc |", s["kpis"]["open_complaints"], "open com |",
      s["records_in_use"], s["kpis"]["stability_counts"])
q = ok(c.get("/api/queue?topN=5&stability=High,Medium,Low"))
top = q["rows"][0]["location_id"]
print("top5:", [(r["location_id"], r["priority_score"], r["rank_stability"]) for r in q["rows"]])
assert "rank_confidence" not in q["rows"][0]
print("lite:", ok(c.get("/api/queue?topN=3&lite=true"))["rows"][0])
loc_ = ok(c.get(f"/api/location/{top}"))
print("location:", top, "receipt parts", len(loc_["receipt"]), "open inc", len(loc_["open_incidents"]), "brief chars", len(loc_["brief"]))
print("gap:", len(ok(c.get("/api/gap"))["points"]), "points")
p = ok(c.post("/api/plan", json={"budget": 60}))
print("plan:", len(p["locations"]), "locs,", p["crew_days_used"], "days,", len(p["bundles"]), "bundles;",
      [(r["strategy"], r["open_incidents_cleared"]) for r in p["comparison"]])
ok(c.get("/api/evidence"))

# earlier as-of removes later records
early = ok(c.get("/api/summary?asof=2025-06-30"))
print("as-of 2025-06-30:", early["records_in_use"], "open", early["kpis"]["open_incidents"])
assert early["records_in_use"]["incidents"] < s["records_in_use"]["incidents"]

# demo batch: max_date jumps, the 3 locations rise
before = {r["location_id"]: r["rank"] for r in ok(c.get("/api/queue?topN=1000&lite=true"))["rows"]}
d = ok(c.post("/api/demo-batch"))
print("demo:", d["locations"], d["messages"], "max_date ->", d["max_date"])
q2 = ok(c.get("/api/queue?topN=1000&lite=true"))
print("asof after demo:", q2["asof"], "| rank before -> after:",
      [(l, before[l], next(r["rank"] for r in q2["rows"] if r["location_id"] == l)) for l in d["locations"]])

# single form entry + CSV upload
print(ok(c.post("/api/report", json={"kind": "incident", "location_id": top, "timestamp": "2026-01-10 09:00", "severity": "Critical", "injuries": 2}))["message"])
bad = c.post("/api/report", json={"kind": "incident", "location_id": "NOPE", "timestamp": "2026-01-10"})
assert bad.status_code == 404
csv = "location_id,created_at,priority_reported,status\n" + f"{top},2026-01-11 10:00:00,High,Open\nNOPE,2026-01-11,High,Open\n"
print(ok(c.post("/api/upload", files=[("files", ("new_complaints.csv", io.BytesIO(csv.encode()), "text/csv"))]))["messages"])

# ---- case-level work plan
wd = ok(c.get("/api/workplan/defaults"))
wp = ok(c.post("/api/workplan", json={"crews": 4, "days": 5, "budget": 300000}))
sch = wp["scheduled"]
assert [s["case_priority"] for s in sch] == sorted((s["case_priority"] for s in sch), reverse=True)
assert wp["capacity"]["crew_days_used"] <= wp["capacity"]["crew_days_total"] + 1e-6
assert wp["capacity"]["budget_used"] <= 300000
assert all(d["very_high"] and d["reason"] and d["unlock"] for d in wp["deferred_very_high"])
print("workplan:", wp["summary"]["scheduled"], "scheduled |", wp["summary"]["very_high_deferred"], "very-high deferred | limiting:",
      wp["capacity"]["binding"], "| next10:", wp["ask_to_clear_very_high"]["next_10"])
big = ok(c.post("/api/workplan", json={"crews": 12, "days": 10, "budget": 3000000}))
assert big["summary"]["scheduled"] > wp["summary"]["scheduled"] and big["summary"]["very_high_deferred"] < wp["summary"]["very_high_deferred"]
assert c.post("/api/workplan", json={"crews": 0}).status_code == 422
print("workplan more resources -> more cases:", big["summary"]["scheduled"], "scheduled,", big["summary"]["very_high_deferred"], "very-high deferred")
print("custom lens:", ok(c.get('/api/queue?topN=1&lens=Custom&weights={"exposure":100}'))["rows"][0]["location_id"])
print("reset:", ok(c.post("/api/reset"))["max_date"])
print("ALL OK")
