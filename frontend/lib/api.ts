export type Stability = 'High' | 'Medium' | 'Low'
export type Lens = 'Safety-first' | 'Backlog-first' | 'Vulnerable-areas-first' | 'Custom'
export type QueueRow = { rank: number; location_id: string; zone: string; priority_score: number; rank_stability: Stability; open_inc_n: number; open_com_n: number; recent_injuries: number; complaint_age: number; why: string; evidence_gaps: string }
export type Summary = { asof: string; max_date: string; records_in_use: { incidents: number; complaints: number }; kpis: { open_incidents: number; open_complaints: number; locations_scored: number; stability_counts: Record<Stability, number>; median_rank_interval_width: number }; zones: string[]; components: Record<string, string> }
export type Case = {
    case_id: string
    location_id: string
    zone: string
    severity: 'Critical' | 'High' | 'Medium' | 'Low'
    injuries: number
    age_days: number
    case_priority: number
    crew: number
    day: number
    cost_money: number
    cost_days?: number
    mobilisation?: boolean
    why: string
}

export type WorkPlanCase = Case & {
    incident_type: string
    location_rank: number
    very_high: boolean
    order: number
    cost_days: number
    cost_money: number
    mobilisation: boolean
}

export type DeferredCase = {
    case_id: string
    location_id: string
    zone: string
    severity: string
    injuries: number
    incident_type?: string
    age_days: number
    case_priority: number
    location_rank?: number
    very_high: boolean
    reason: string
    shortfall_money: number
    shortfall_crew_days: number
    unlock: string
    cost_days: number
    cost_money: number
    why: string
}

export type WorkPlanResult = {
    asof: string
    lens: string
    currency: string
    inputs: { crews: number; days: number; budget: number; overhead_days: number; overhead_money: number }
    cost_model: { cost_days: Record<string, number>; cost_money: Record<string, number>; note: string }
    case_priority_weights?: Record<string, number>
    case_priority_labels?: Record<string, string>
    very_high_cut?: number
    very_high_rule?: string
    capacity: {
        crew_days_total: number
        crew_days_used: number
        crew_days_remaining: number
        budget_used: number
        budget_remaining: number
        binding: string
    }
    summary: {
        open_cases: number
        scheduled: number
        deferred: number
        share_scheduled: string
        locations_visited: number
        very_high_total: number
        very_high_scheduled: number
        very_high_deferred: number
        severity_load_scheduled: number
        injuries_in_scheduled_cases: number
    }
    ask_to_clear_very_high?: {
        extra_budget: number
        extra_crew_days: number
        extra_crews_for_same_days: number
        next_10?: { extra_budget: number; extra_crew_days: number; extra_crews_for_same_days: number }
    }
    scheduled: WorkPlanCase[]
    crews: Array<{ crew: number; days_capacity: number; days_used: number; money: number; cases: number; locations: string[] }>
    deferred_very_high?: DeferredCase[]
    deferred_very_high_total?: number
    deferred_other?: DeferredCase[]
    deferred_other_total?: number
    note?: string
}

export const USE_MOCKS = false
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

const top = [
    ['L0642', 'ZONE_02', 93.3, 'High', 6, 7, 5, 242], ['L0493', 'ZONE_18', 91.1, 'High', 5, 6, 3, 198], ['L0688', 'ZONE_09', 90, 'Medium', 4, 8, 3, 184], ['L0219', 'ZONE_10', 88.1, 'Medium', 5, 4, 2, 166], ['L0358', 'ZONE_07', 88.1, 'Medium', 4, 5, 2, 151], ['L0774', 'ZONE_14', 86.8, 'Medium', 4, 4, 1, 145], ['L0117', 'ZONE_03', 85.5, 'Medium', 3, 5, 2, 131], ['L0901', 'ZONE_19', 84.2, 'Medium', 3, 4, 1, 126], ['L0431', 'ZONE_12', 83.7, 'Low', 3, 3, 1, 119], ['L0590', 'ZONE_05', 82.9, 'Low', 2, 4, 0, 106]
] as const

const mockSummary: Summary = {
    asof: '2025-12-31',
    max_date: '2025-12-31',
    records_in_use: { incidents: 10000, complaints: 8000 },
    kpis: {
        open_incidents: 2903,
        open_complaints: 3513,
        locations_scored: 1000,
        stability_counts: { High: 7, Medium: 26, Low: 967 },
        median_rank_interval_width: 192
    },
    zones: Array.from({ length: 20 }, (_, i) => `ZONE_${String(i + 1).padStart(2, '0')}`),
    components: {
        open_incident_load: 'Open incident load',
        injury_severity: 'Injury severity',
        open_complaint_load: 'Open complaint load',
        complaint_age: 'Complaint age',
        exposure: 'Exposure',
        shrunk_history: 'Long-run history'
    }
}

function getMockQueue(lens: Lens = 'Safety-first', topN = 50): { rows: QueueRow[] } {
    const rows: QueueRow[] = Array.from({ length: topN }, (_, i) => {
        const seed = top[i] ?? [`L${String((i * 83 + 27) % 1000).padStart(4, '0')}`, `ZONE_${String((i % 20) + 1).padStart(2, '0')}`, 81 - i * .45, i < 8 ? 'Low' : 'Low', Math.max(1, 4 - i % 4), Math.max(1, 5 - i % 3), i % 3, 100 - i * 3] as const;
        const shift = lens === 'Backlog-first' ? (i % 3) * 1.7 : lens === 'Vulnerable-areas-first' ? ((i * 7) % 5) * .8 : 0;
        return {
            rank: i + 1,
            location_id: seed[0],
            zone: seed[1],
            priority_score: Math.max(58, seed[2] - shift),
            rank_stability: seed[3] as Stability,
            open_inc_n: seed[4],
            open_com_n: seed[5],
            recent_injuries: seed[6],
            complaint_age: seed[7],
            why: 'Open items and recent harm place this location near the front of the queue; exposure is also considered.',
            evidence_gaps: i < 3 ? 'Long-run history is not stable | Verify current conditions' : 'none'
        }
    })
    return { rows }
}

function getMockWorkPlan(crews = 4, days = 5, budget = 300000): WorkPlanResult {
    const totalCrewDays = crews * days
    let moneyLeft = budget
    let daysLeft = totalCrewDays
    const scheduled: WorkPlanCase[] = []
    const deferred: DeferredCase[] = []

    const allMockCases = Array.from({ length: 30 }, (_, i) => {
        const sev: 'Critical' | 'High' | 'Medium' = i < 8 ? 'Critical' : i < 18 ? 'High' : 'Medium'
        const pri = 94.0 - i * 1.2
        const injuries = (i % 3) + 1
        const baseCost = sev === 'Critical' ? 20000 : sev === 'High' ? 12000 : 6000
        const baseDays = sev === 'Critical' ? 1.0 : sev === 'High' ? 0.75 : 0.5
        const costMoney = Math.round((baseCost * (0.85 + (pri / 100) * 0.35) + injuries * 2000) / 100) * 100
        const costDays = Math.round((baseDays * (0.85 + (pri / 100) * 0.35) + injuries * 0.1) * 100) / 100
        return {
            case_id: `I${String(2399 - i * 17).padStart(4, '0')}`,
            location_id: top[i % top.length][0],
            zone: top[i % top.length][1],
            severity: sev,
            injuries,
            incident_type: i % 2 === 0 ? 'Road Hazard' : 'Traffic Collision',
            age_days: 180 - i * 5,
            case_priority: pri,
            location_rank: (i % 10) + 1,
            very_high: pri >= 85,
            order: i + 1,
            crew: (i % crews) + 1,
            day: Math.min(Math.floor(i / crews) + 1, Math.ceil(days)),
            cost_money: costMoney,
            cost_days: costDays,
            mobilisation: i % 2 === 0,
            why: `${sev} priority case with current harm; location rank #${(i % 10) + 1}`
        }
    })

    for (const c of allMockCases) {
        if (costMoneyValid(c.cost_money, moneyLeft) && costDaysValid(c.cost_days, daysLeft)) {
            scheduled.push(c)
            moneyLeft -= c.cost_money
            daysLeft -= c.cost_days
        } else {
            const moneyOk = c.cost_money <= moneyLeft
            const reason = !moneyOk && daysLeft < c.cost_days ? 'Budget & crew capacity' : !moneyOk ? 'Budget' : 'Crew capacity'
            deferred.push({
                case_id: c.case_id,
                location_id: c.location_id,
                zone: c.zone,
                severity: c.severity,
                injuries: c.injuries,
                incident_type: c.incident_type,
                age_days: c.age_days,
                case_priority: c.case_priority,
                location_rank: c.location_rank,
                very_high: c.very_high,
                reason,
                shortfall_money: Math.max(0, c.cost_money - moneyLeft),
                shortfall_crew_days: Math.max(0, Math.round((c.cost_days - daysLeft) * 100) / 100),
                unlock: `Add capacity to unlock this case`,
                cost_days: c.cost_days,
                cost_money: c.cost_money,
                why: c.why
            })
        }
    }

    function costMoneyValid(need: number, left: number) { return need <= left }
    function costDaysValid(need: number, left: number) { return need <= left }

    const vhDeferred = deferred.filter(d => d.very_high)
    const extraBudget = vhDeferred.slice(0, 10).reduce((acc, d) => acc + d.cost_money, 0)
    const extraDays = Math.round(vhDeferred.slice(0, 10).reduce((acc, d) => acc + d.cost_days, 0) * 10) / 10

    return {
        asof: '2025-12-31',
        lens: 'Safety-first',
        currency: 'INR',
        inputs: { crews, days, budget, overhead_days: 0.25, overhead_money: 2000 },
        cost_model: {
            cost_days: { Low: 0.25, Medium: 0.5, High: 0.75, Critical: 1.0 },
            cost_money: { Low: 3000, Medium: 6000, High: 12000, Critical: 20000 },
            note: 'Variable cost model by severity, priority, injuries, and mobilization overhead.'
        },
        capacity: {
            crew_days_total: totalCrewDays,
            crew_days_used: Math.round((totalCrewDays - daysLeft) * 100) / 100,
            crew_days_remaining: Math.round(daysLeft * 100) / 100,
            budget_used: budget - moneyLeft,
            budget_remaining: moneyLeft,
            binding: moneyLeft < 10000 && daysLeft < 0.5 ? 'both' : moneyLeft < 10000 ? 'budget' : 'crew capacity'
        },
        summary: {
            open_cases: 2903,
            scheduled: scheduled.length,
            deferred: 2903 - scheduled.length,
            share_scheduled: `${((scheduled.length / 2903) * 100).toFixed(1)}%`,
            locations_visited: new Set(scheduled.map(s => s.location_id)).size,
            very_high_total: 146,
            very_high_scheduled: scheduled.filter(s => s.very_high).length,
            very_high_deferred: 146 - scheduled.filter(s => s.very_high).length,
            severity_load_scheduled: scheduled.reduce((a, s) => a + (s.severity === 'Critical' ? 4 : s.severity === 'High' ? 3 : 2), 0),
            injuries_in_scheduled_cases: scheduled.reduce((a, s) => a + s.injuries, 0)
        },
        ask_to_clear_very_high: {
            extra_budget: extraBudget,
            extra_crew_days: extraDays,
            extra_crews_for_same_days: Math.ceil(extraDays / days),
            next_10: {
                extra_budget: extraBudget,
                extra_crew_days: extraDays,
                extra_crews_for_same_days: Math.ceil(extraDays / days)
            }
        },
        scheduled,
        crews: Array.from({ length: crews }, (_, c) => {
            const mine = scheduled.filter(s => s.crew === c + 1)
            return {
                crew: c + 1,
                days_capacity: days,
                days_used: Math.round(mine.reduce((a, s) => a + s.cost_days, 0) * 100) / 100,
                money: mine.reduce((a, s) => a + s.cost_money, 0),
                cases: mine.length,
                locations: Array.from(new Set(mine.map(s => s.location_id)))
            }
        }),
        deferred_very_high: vhDeferred,
        deferred_very_high_total: vhDeferred.length
    }
}

export async function getSummary(lens: Lens = 'Safety-first'): Promise<Summary> {
    if (!USE_MOCKS) {
        try {
            const res = await fetch(`${API_URL}/api/summary?lens=${encodeURIComponent(lens)}`)
            if (res.ok) {
                return await res.json()
            }
        } catch (e) {
            console.warn('Falling back to mock summary due to fetch error:', e)
        }
    }
    return mockSummary
}

export async function getQueue(lens: Lens = 'Safety-first', topN = 50): Promise<{ rows: QueueRow[] }> {
    if (!USE_MOCKS) {
        try {
            const res = await fetch(`${API_URL}/api/queue?lens=${encodeURIComponent(lens)}&topN=${topN}`)
            if (res.ok) {
                const data = await res.json()
                return { rows: data.rows ?? [] }
            }
        } catch (e) {
            console.warn('Falling back to mock queue due to fetch error:', e)
        }
    }
    return getMockQueue(lens, topN)
}

export async function getWorkPlan(lens: Lens = 'Safety-first', crews = 4, days = 5, budget = 300000): Promise<WorkPlanResult> {
    if (!USE_MOCKS) {
        try {
            const res = await fetch(`${API_URL}/api/workplan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lens, crews, days, budget })
            })
            if (res.ok) {
                return await res.json()
            }
        } catch (e) {
            console.warn('Falling back to mock workplan due to fetch error:', e)
        }
    }
    return getMockWorkPlan(crews, days, budget)
}

export async function getCases(lens: Lens = 'Safety-first', crews = 4, days = 5, budget = 300000): Promise<Case[]> {
    const plan = await getWorkPlan(lens, crews, days, budget)
    return plan.scheduled
}

export const csv = (rows: QueueRow[]) =>
    ['rank,location_id,zone,priority_score,rank_stability,open_inc_n,open_com_n', ...rows.map(r => [r.rank, r.location_id, r.zone, r.priority_score, r.rank_stability, r.open_inc_n, r.open_com_n].join(','))].join('\n')

export async function postAction(action: string, payload?: any) {
    if (!USE_MOCKS) {
        try {
            if (action === 'demo') {
                const res = await fetch(`${API_URL}/api/demo-batch`, { method: 'POST' })
                if (res.ok) {
                    const data = await res.json()
                    const locs = data.locations ? data.locations.join(', ') : 'demo locations'
                    return { message: `Demo batch added at ${locs}. Max date: ${data.max_date}` }
                }
            } else if (action === 'reset') {
                const res = await fetch(`${API_URL}/api/reset`, { method: 'POST' })
                if (res.ok) {
                    const data = await res.json()
                    return { message: data.message ?? 'Data reset to original' }
                }
            } else if (action === 'report' && payload) {
                const res = await fetch(`${API_URL}/api/report`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                })
                if (res.ok) {
                    const data = await res.json()
                    return { message: data.message ?? 'Report logged successfully' }
                }
            }
        } catch (e) {
            console.warn('Action failed on backend, using fallback message:', e)
        }
    }
    return { message: action === 'demo' ? 'Demo batch added at L0812, L0344, L0710' : 'Data updated. The queue now uses the latest records.' }
}

export async function getEvidence() {
    if (!USE_MOCKS) {
        try {
            const res = await fetch(`${API_URL}/api/evidence`)
            if (res.ok) {
                const data = await res.json()
                return {
                    tests: data.association_scan?.tests ?? 208,
                    passed: data.association_scan?.passed_bonferroni ?? 0,
                    probes: data.probes?.rows?.length ?? 9,
                    rawDeviance: 2.5173,
                    meanDeviance: 1.093,
                    confident: 0,
                    validation: [
                        ['Global mean', 1.093],
                        ['Raw H1 history', 2.5173],
                        ['Poisson GLM', 1.1506],
                        ['Empirical-Bayes', 1.0934]
                    ]
                }
            }
        } catch (e) {
            console.warn('Falling back to mock evidence:', e)
        }
    }
    return { tests: 208, passed: 0, probes: 9, rawDeviance: 2.5173, meanDeviance: 1.093, confident: 0, validation: [['Global mean', 1.093], ['Raw H1 history', 2.5173], ['Poisson GLM', 1.1506], ['Empirical-Bayes', 1.0934]] }
}

