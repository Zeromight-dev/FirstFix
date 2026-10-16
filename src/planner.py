"""
FixFirst - Phase 4: crew-day budget planner
Given a crew-day budget, choose which locations to visit and compare strategies by how much
EXISTING backlog they clear. This is accounting of known open items, NOT a prediction of future incidents.

Run:  python src/planner.py
Import in the app:  from planner import plan, compare, COST_NOTE
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from priority import LENSES, OUT, build_features, load_all, score  # noqa: E402

# ---- cost assumptions (state these in the README) ----------------------------------------------
BASE_COST = 0.5          # crew-days to visit/inspect any location
PER_OPEN_INCIDENT = 0.25  # crew-days per open incident
PER_OPEN_COMPLAINT = 0.10  # crew-days per open complaint
COST_NOTE = (f"Assumed cost per location = {BASE_COST} + {PER_OPEN_INCIDENT} x open incidents + "
             f"{PER_OPEN_COMPLAINT} x open complaints crew-days (illustrative, not from the data).")


def add_cost(S):
    S = S.copy()
    S["cost_days"] = BASE_COST + PER_OPEN_INCIDENT * S.open_inc_n + PER_OPEN_COMPLAINT * S.open_com_n
    return S


def greedy(order, cost, budget):
    chosen, used = [], 0.0
    for lid in order:
        c = cost[lid]
        if used + c <= budget:
            chosen.append(lid)
            used += c
    return chosen, used


def plan(S, budget, strategy="FixFirst", seed=0):
    S = add_cost(S)
    if strategy == "FixFirst":
        value = S.priority_score / 100 * (S.open_incident_load + 0.5 * S.open_complaint_load)
        order = (value / S.cost_days).sort_values(ascending=False).index
    elif strategy == "Worst severity first":
        order = S.sort_values(["open_incident_load", "open_com_n"], ascending=False).index
    elif strategy == "Most open complaints first":
        order = S.sort_values(["open_com_n", "open_incident_load"], ascending=False).index
    elif strategy == "Most past incidents first":
        order = S.sort_values("inc_total", ascending=False).index
    else:  # Random
        order = np.random.default_rng(seed).permutation(S.index)
    chosen, used = greedy(order, S.cost_days, budget)
    return chosen, used


def outcome(S, chosen, used):
    S = add_cost(S)
    c = S.loc[chosen]
    tot = S
    return {
        "locations_visited": len(chosen),
        "crew_days_used": round(used, 1),
        "open_incidents_cleared": int(c.open_inc_n.sum()),
        "severity_backlog_cleared": round(c.open_incident_load.sum(), 1),
        "open_complaints_cleared": int(c.open_com_n.sum()),
        "exposure_weighted_backlog": round((c.open_incident_load * (1 + c.exposure_pts / 100)).sum(), 1),
        "share_of_all_open_incidents": f"{c.open_inc_n.sum() / max(tot.open_inc_n.sum(), 1):.1%}",
    }


def compare(S, budget, n_random=200):
    rows = []
    for strat in ["FixFirst", "Worst severity first", "Most open complaints first", "Most past incidents first"]:
        ch, used = plan(S, budget, strat)
        rows.append({"strategy": strat, **outcome(S, ch, used)})
    rr = [outcome(S, *plan(S, budget, "Random", seed=s)) for s in range(n_random)]
    avg = {k: (np.mean([r[k] for r in rr]) if not isinstance(rr[0][k], str) else "") for k in rr[0]}
    avg = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in avg.items()}
    avg["share_of_all_open_incidents"] = f"{np.mean([float(r['share_of_all_open_incidents'].strip('%')) for r in rr]):.1f}%"
    rows.append({"strategy": f"Random (mean of {n_random})", **avg})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    loc, inc, com, eb = load_all()
    F = build_features(loc, inc, com, eb, "2025-12-31 23:59")
    S = score(F, LENSES["Safety-first"])
    print(COST_NOTE)
    print(f"Backlog at as-of date: {int(S.open_inc_n.sum())} open incidents, {int(S.open_com_n.sum())} open complaints")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    results = []
    for b in (30, 60, 120):
        t = compare(S, b)
        t.insert(0, "budget_crew_days", b)
        print(f"\n=== Budget {b} crew-days ===")
        print(t.drop(columns="budget_crew_days").to_string(index=False))
        results.append(t)
    pd.concat(results).to_csv(OUT / "planner_comparison.csv", index=False)
    print("\nNote: this measures how much KNOWN backlog each plan clears. It does not claim to prevent future incidents.")
    print("Saved", OUT / "planner_comparison.csv")
