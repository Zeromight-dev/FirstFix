'use client'

import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, BarChart3, CalendarDays, Check, ChevronDown, CircleHelp, Download, FileText, Filter, Info, MapPin, Menu, Moon, Plus, RotateCcw, Search, ShieldCheck, SlidersHorizontal, Sun, Upload, Users, X, Zap } from 'lucide-react'
import { getEvidence, getQueue, getSummary, getWorkPlan, postAction, type Lens, type QueueRow, type Summary, type WorkPlanResult, type WorkPlanCase } from '@/lib/api'

type Tab = 'Work plan'|'Priority queue'|'Location'|'Strategy comparison'|'Evidence'
const tabs:Tab[]=['Work plan','Priority queue','Location','Strategy comparison','Evidence']
const stabilityClass=(s:string)=>s==='High'?'high':s==='Medium'?'medium':'low'
function Pill({children, tone='muted'}:{children:React.ReactNode;tone?:string}){return <span className={`pill ${tone}`}>{children}</span>}
function Card({children,className='',style}:{children:React.ReactNode;className?:string;style?:React.CSSProperties}){return <section className={`card ${className}`} style={style}>{children}</section>}
function Metric({label,value,detail,icon:Icon}:{label:string;value:string|number;detail:string;icon:typeof Zap}){return <Card className="metric"><div className="metric-icon"><Icon size={17}/></div><div><div className="eyebrow">{label}<Info size={13}/></div><strong>{value}</strong><p>{detail}</p></div></Card>}

export default function Page(){
 const [tab,setTab]=useState<Tab>('Work plan'); const [dark,setDark]=useState(false); const [summary,setSummary]=useState<Summary>(); const [rows,setRows]=useState<QueueRow[]>([]); const [workPlan,setWorkPlan]=useState<WorkPlanResult | null>(null); const [lens,setLens]=useState<Lens>('Safety-first'); const [selected,setSelected]=useState<QueueRow>(); const [budget,setBudget]=useState(300000); const [crews,setCrews]=useState(4); const [days,setDays]=useState(5); const [notice,setNotice]=useState(''); const [showInputs,setShowInputs]=useState(false); const [search,setSearch]=useState('');
 useEffect(()=>{
  getSummary(lens).then(setSummary)
  getQueue(lens,50).then(r=>setRows(r.rows))
  getWorkPlan(lens,crews,days,budget).then(setWorkPlan)
 },[lens,crews,days,budget])
 const filtered=useMemo(()=>rows.filter(r=>!search||r.location_id.toLowerCase().includes(search.toLowerCase())||r.zone.toLowerCase().includes(search.toLowerCase())),[rows,search])
 const runAction=async(action:string)=>{
  const r=await postAction(action)
  setNotice(r.message)
  getSummary(lens).then(setSummary)
  getQueue(lens,50).then(res=>setRows(res.rows))
  getWorkPlan(lens,crews,days,budget).then(setWorkPlan)
  setTimeout(()=>setNotice(''),5000)
 }
 if(!summary)return <div className="loading">Loading FixFirst…</div>
 return <div className={dark?'app dark':'app'}>
  <div className="synthetic"><ShieldCheck size={14}/> Synthetic educational data. Locations and records are simulated. Not for real public-safety decisions.</div>
  <header className="topbar"><div className="brand"><div className="brand-mark"><Zap size={17}/></div><span>FixFirst</span><small>city operations</small></div><nav>{tabs.map(t=><button key={t} className={tab===t?'active':''} onClick={()=>setTab(t)}>{t}</button>)}</nav><div className="top-actions"><button className="icon-btn" aria-label="Help"><CircleHelp size={18}/></button><button className="icon-btn" onClick={()=>setDark(!dark)} aria-label="Toggle theme">{dark?<Sun size={18}/>:<Moon size={18}/>}</button><button className="mobile-inputs" onClick={()=>setShowInputs(!showInputs)}><Menu size={16}/> Inputs</button></div></header>
  <div className="workspace"><aside className={showInputs?'inputs open':'inputs'}><div className="input-head"><div><div className="eyebrow">CONTROL PANEL</div><h2>Inputs</h2></div><SlidersHorizontal size={18}/></div><label className="field-label">As-of date <span>i</span></label><div className="date-input"><CalendarDays size={16}/><span>{summary.asof}</span></div><small className="helper">Only records before this date are used.</small><label className="field-label">Policy lens <span>i</span></label><div className="lens-grid">{(['Safety-first','Backlog-first','Vulnerable-areas-first','Custom'] as Lens[]).map(x=><button key={x} className={lens===x?'selected':''} onClick={()=>setLens(x)}>{x}</button>)}</div><div className="add-data"><div className="section-label"><Plus size={15}/> Add data</div><div className="report-toggle"><button className="selected">Incident</button><button>Complaint</button></div><select aria-label="Location"><option>Choose location ID</option><option>L0642 · ZONE_02</option><option>L0493 · ZONE_18</option></select><input placeholder="Date and time" type="datetime-local"/><div className="two-inputs"><select><option>Severity</option><option>Critical</option><option>High</option></select><input placeholder="Injuries" type="number" min="0"/></div><button className="primary full" onClick={()=>runAction('report')}><Plus size={15}/> Add report</button><div className="dropzone"><Upload size={19}/><strong>Drop CSV files here</strong><span>incidents.csv / public_complaints.csv</span><small>Required: location_id, timestamp, severity</small></div><button className="outline full" onClick={()=>runAction('demo')}><Zap size={15}/> Add demo batch</button><button className="text-btn" onClick={()=>runAction('reset')}><RotateCcw size={14}/> Reset data</button></div><div className="record-note"><span>Records in use</span><strong>{summary.records_in_use.incidents.toLocaleString()} incidents · {summary.records_in_use.complaints.toLocaleString()} complaints</strong></div></aside>
  <main className="main"><div className="page-heading"><div><div className="breadcrumb">OPERATIONS / {tab.toUpperCase()}</div><h1>{tab==='Work plan'?'Today’s work plan':tab}</h1><p>{tab==='Work plan'?'Turn limited crew capacity into a clear, defensible order of work.':'A calm view of what current records can support.'}</p></div><div className="heading-actions"><button className="outline" onClick={()=>setShowInputs(!showInputs)}><Filter size={15}/> Filter inputs</button><button className="primary"><Download size={15}/> Export</button></div></div>{notice&&<div className="toast"><Check size={16}/>{notice}<button onClick={()=>setNotice('')}><X size={15}/></button></div>}{tab==='Work plan'?<WorkPlan workPlan={workPlan} budget={budget} setBudget={setBudget} crews={crews} setCrews={setCrews} days={days} setDays={setDays} onRefresh={()=>getWorkPlan(lens,crews,days,budget).then(setWorkPlan)}/>:tab==='Priority queue'?<Queue rows={filtered} summary={summary} selected={selected} setSelected={setSelected} search={search} setSearch={setSearch}/>:tab==='Location'?<Location row={selected||rows[0]}/>:tab==='Strategy comparison'?<Strategy/>:<Evidence/>}</main></div>
  </div>
 }

 function WorkPlan({workPlan,budget,setBudget,crews,setCrews,days,setDays,onRefresh}:{workPlan:WorkPlanResult|null;budget:number;setBudget:React.Dispatch<React.SetStateAction<number>>;crews:number;setCrews:React.Dispatch<React.SetStateAction<number>>;days:number;setDays:React.Dispatch<React.SetStateAction<number>>;onRefresh:()=>void}){
  const [viewMode, setViewMode]=useState<'priority'|'crew'>('priority')
  const [showCostModel, setShowCostModel]=useState(false)

  const scheduled = workPlan?.scheduled ?? []
  const cap = workPlan?.capacity
  const summ = workPlan?.summary
  const ask = workPlan?.ask_to_clear_very_high?.next_10
  const deferred = workPlan?.deferred_very_high ?? []

  const handleApplyNext10 = () => {
    if (ask) {
      setBudget(b => b + ask.extra_budget)
      setDays(d => +(d + Math.max(1, Math.ceil(ask.extra_crew_days / Math.max(1, crews)))).toFixed(1))
    }
  }

  return <div className="content">
    <div className="metrics">
      <Metric label="Open cases" value={summ?.open_cases?.toLocaleString() ?? '2,903'} detail={`across ${summ?.locations_visited ?? 1000} locations`} icon={FileText}/>
      <Metric label="Scheduled" value={summ?.scheduled ?? scheduled.length} detail={`priority cases (${summ?.share_scheduled ?? '0%'})`} icon={Check}/>
      <Metric label="Crew-days" value={`${cap?.crew_days_used ?? 0} / ${crews*days}`} detail={cap?.binding==='crew capacity'||cap?.binding==='both'?'Bottleneck: Crew days':`Remaining: ${cap?.crew_days_remaining ?? 0}d`} icon={Users}/>
      <Metric label="Budget used" value={`₹${(cap?.budget_used ?? 0).toLocaleString()}`} detail={`of ₹${budget.toLocaleString()} (${cap?.binding==='budget'||cap?.binding==='both'?'Bottleneck: Budget':`Remaining: ₹${(cap?.budget_remaining ?? 0).toLocaleString()}`})`} icon={Zap}/>
    </div>
    <Card className="resources">
      <div className="card-title">
        <div>
          <div className="eyebrow">RESOURCES &amp; CONSTRAINTS</div>
          <h2>What can the crew handle?</h2>
        </div>
        <Pill tone={cap?.binding==='both'?'critical':cap?.binding==='budget'?'medium':'blue'}>
          {cap?.binding ? `Bottleneck: ${cap.binding}` : 'Auto-updated'}
        </Pill>
      </div>
      <div className="resource-fields">
        <label>Work crews
          <input type="number" min="1" max="50" value={crews} onChange={e=>setCrews(Math.max(1, +e.target.value))}/>
        </label>
        <label>Days available
          <input type="number" min=".5" max="60" step=".5" value={days} onChange={e=>setDays(Math.max(0.5, +e.target.value))}/>
        </label>
        <label>Budget (INR)
          <input type="number" step="10000" min="10000" value={budget} onChange={e=>setBudget(Math.max(0, +e.target.value))}/>
        </label>
        <button className="primary generate" onClick={onRefresh}><Zap size={15}/> Generate work plan</button>
      </div>
      <div className="assumptions" onClick={()=>setShowCostModel(!showCostModel)} style={{cursor:'pointer',userSelect:'none'}}>
        <ChevronDown size={15} style={{transform: showCostModel ? 'rotate(180deg)' : 'none', transition:'transform .2s'}}/> 
        Cost assumptions <span>{workPlan?.cost_model?.note ?? 'Variable cost per incident by priority, severity & injuries'}</span>
      </div>
      {showCostModel && (
        <div style={{padding:'12px',marginTop:'8px',background:'var(--bg-card-subtle,#f8f9fa)',borderRadius:'8px',fontSize:'13px',lineHeight:'1.5'}}>
          <strong>Variable Incident Cost Breakdown:</strong>
          <ul style={{margin:'6px 0 0 16px',padding:0}}>
            <li>Severity tier base: Critical ₹20,000 (1.0d) · High ₹12,000 (0.75d) · Medium ₹6,000 (0.5d) · Low ₹3,000 (0.25d)</li>
            <li>Priority factor: Scaled by case priority (0.85x to 1.2x) reflecting depth of repair &amp; inspection</li>
            <li>Harm complexity: +₹2,000 &amp; +0.10 crew-days per reported injury</li>
            <li>Mobilisation overhead: +₹2,000 &amp; +0.25 crew-days when crew moves to a new location</li>
          </ul>
        </div>
      )}
    </Card>

    <div className="section-heading">
      <div>
        <div className="eyebrow">SECTION A</div>
        <h2>Do these now ({scheduled.length} cases scheduled)</h2>
      </div>
      <div className="segmented">
        <button className={viewMode==='priority'?'selected':''} onClick={()=>setViewMode('priority')}>By priority</button>
        <button className={viewMode==='crew'?'selected':''} onClick={()=>setViewMode('crew')}>By crew</button>
      </div>
    </div>

    {viewMode==='priority' ? (
      <Card className="case-card">
        <div className="table-head">
          <span>ORDER / CASE</span>
          <span>LOCATION</span>
          <span>SEVERITY</span>
          <span>PRIORITY</span>
          <span>VARIABLE COST</span>
          <span>ASSIGNMENT</span>
        </div>
        {scheduled.length === 0 ? (
          <div style={{padding:'24px',textAlign:'center',color:'var(--text-muted,#888)'}}>
            No cases fit within the current crew or budget limits. Try increasing crews, days, or budget above.
          </div>
        ) : scheduled.map((c,i)=>(
          <div className="case-row" key={c.case_id}>
            <div className="case-order">
              <b>{String(c.order ?? i+1).padStart(2,'0')}</b>
              <div>
                <strong>{c.case_id}</strong>
                <small>{c.injuries} inj · {c.incident_type || 'Incident'}</small>
              </div>
            </div>
            <a>{c.location_id}<small>{c.zone}</small></a>
            <Pill tone={c.severity==='Critical'?'critical':c.severity==='High'?'high':'medium'}>{c.severity}</Pill>
            <div className="priority">
              <strong>{c.case_priority.toFixed(1)}</strong>
              <i><em style={{width:`${Math.min(100, c.case_priority)}%`}}/></i>
            </div>
            <div style={{display:'flex',flexDirection:'column'}}>
              <strong>₹{c.cost_money.toLocaleString()}</strong>
              <small style={{color:'var(--text-muted,#777)'}}>
                {c.cost_days}d {c.mobilisation ? '· new site' : '· on site'}
              </small>
            </div>
            <span className="assignment">Crew {c.crew} · Day {c.day}</span>
          </div>
        ))}
      </Card>
    ) : (
      <div style={{display:'grid',gap:'12px'}}>
        {workPlan?.crews?.map(cr => {
          const crewCases = scheduled.filter(s => s.crew === cr.crew)
          return (
            <Card key={cr.crew} style={{padding:'16px'}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'12px',borderBottom:'1px solid var(--border-color,#eee)',paddingBottom:'8px'}}>
                <div>
                  <h3 style={{margin:0,fontSize:'16px'}}>Crew {cr.crew}</h3>
                  <small style={{color:'var(--text-muted,#666)'}}>{cr.locations.length} locations · {cr.cases} cases assigned</small>
                </div>
                <div style={{textAlign:'right'}}>
                  <strong>{cr.days_used} / {cr.days_capacity} crew-days</strong>
                  <div><small style={{color:'var(--text-muted,#666)'}}>Allocated: ₹{cr.money.toLocaleString()}</small></div>
                </div>
              </div>
              <div style={{display:'grid',gap:'8px'}}>
                {crewCases.map(c => (
                  <div key={c.case_id} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'6px 10px',background:'var(--bg-card-subtle,#f9f9f9)',borderRadius:'6px',fontSize:'13px'}}>
                    <div>
                      <strong>Day {c.day}: {c.case_id}</strong> · {c.location_id} ({c.zone})
                      <span style={{marginLeft:'8px'}}><Pill tone={c.severity==='Critical'?'critical':'high'}>{c.severity}</Pill></span>
                    </div>
                    <div style={{textAlign:'right'}}>
                      <b>₹{c.cost_money.toLocaleString()}</b> <small style={{color:'var(--text-muted,#777)'}}>({c.cost_days}d)</small>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )
        })}
      </div>
    )}

    <Card className="deferred">
      <div className="deferred-head">
        <div>
          <div className="eyebrow">SECTION B</div>
          <h2>Very high priority, not covered</h2>
          <p>{summ?.very_high_deferred ?? deferred.length} very-high-priority cases cannot be covered with current resources.</p>
        </div>
        <Pill tone="critical">Needs more capacity</Pill>
      </div>

      {ask && (
        <div className="ask">
          <div>
            <strong>To cover the next 10 deferred cases:</strong>
            <div>Add <b>₹{ask.extra_budget.toLocaleString()}</b> and <b>{ask.extra_crew_days} crew-days</b> ({ask.extra_crews_for_same_days > 0 ? `+${ask.extra_crews_for_same_days} crews` : 'more days'})</div>
          </div>
          <button className="outline" onClick={handleApplyNext10}>
            Re-plan with next-10 budget <ArrowRight size={14}/>
          </button>
        </div>
      )}

      <div className="deferred-list">
        {deferred.slice(0, 5).map(d => (
          <div key={d.case_id} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'8px 0',borderBottom:'1px solid var(--border-color,#f0f0f0)'}}>
            <div>
              <strong>{d.case_id} · {d.location_id}</strong>
              <small style={{display:'block',color:'var(--text-muted,#777)'}}>{d.severity} · {d.injuries} inj · Priority: {d.case_priority.toFixed(1)}</small>
            </div>
            <Pill tone="critical">{d.reason}</Pill>
            <span style={{fontSize:'12px',color:'var(--text-muted,#666)'}}>
              Cost: ₹{d.cost_money.toLocaleString()} ({d.cost_days}d)
            </span>
          </div>
        ))}
      </div>
    </Card>

    <p className="caption">Plans existing open cases only using dynamic variable cost optimization. Costs are variable based on priority, severity, and injuries.</p>
  </div>
}

 function Queue({rows,summary,selected,setSelected,search,setSearch}:{rows:QueueRow[];summary:Summary;selected:QueueRow|undefined;setSelected:(r:QueueRow|undefined)=>void;search:string;setSearch:(s:string)=>void}){return <div className="content"><div className="metrics"><Metric label="Open incidents" value="2,903" detail="current open items" icon={Zap}/><Metric label="Open complaints" value="3,513" detail="current open items" icon={FileText}/><Metric label="Stable rank" value="33" detail="High + Medium" icon={ShieldCheck}/><Metric label="Locations scored" value="1,000" detail="20 zones" icon={MapPin}/></div><Card className="recommend"><div className="recommend-icon"><Zap size={18}/></div><div><div className="eyebrow">RECOMMENDED START</div><h3>Review these 5 locations first</h3><div className="chips">{rows.slice(0,5).map(r=><button key={r.location_id} onClick={()=>setSelected(r)}>{r.location_id}</button>)}</div></div>{rows[0]&&<button className="text-btn" onClick={()=>setSelected(rows[0])}>See why <ArrowRight size={14}/></button>}</Card><div className="queue-toolbar"><div className="search"><Search size={15}/><input placeholder="Search location or zone" value={search} onChange={e=>setSearch(e.target.value)}/></div><label>Top N <input type="range" min="10" max="200" defaultValue="50"/></label><div className="filter-chips"><Pill tone="blue">High</Pill><Pill>Medium</Pill><Pill>Low</Pill></div></div><div className="queue-layout"><div className="map"><div className="map-label"><MapPin size={15}/> Live queue map <span>OSM · synthetic points</span></div><div className="map-grid">{rows.slice(0,35).map((r,i)=><button key={r.location_id} className={`dot ${i<10?'priority-dot':''}`} style={{left:`${8+(i*17)%83}%`,top:`${12+(i*29)%76}%`,background:i<10?`hsl(${42-i*4} 88% ${52-i}% )`:undefined}} onClick={()=>setSelected(r)} aria-label={`Rank ${r.rank}, ${r.location_id}`}>{i<10&&<span>{r.rank}</span>}</button>)}<div className="map-legend"><span><i className="legend-dot grey"/>All locations</span><span><i className="legend-dot yellow"/>Priority queue</span></div></div></div><Card className="queue-card"><div className="list-head"><span>RANKED QUEUE</span><span>50 locations</span></div>{rows.slice(0,12).map(r=><button className={`queue-row ${selected?.location_id===r.location_id?'chosen':''}`} key={r.location_id} onClick={()=>setSelected(r)}><b className="rank">{r.rank}</b><div className="queue-name"><strong>{r.location_id}</strong><small>{r.zone}</small></div><div className="score"><strong>{r.priority_score.toFixed(1)}</strong><i><em style={{width:`${r.priority_score}%`}}/></i></div><Pill tone={stabilityClass(r.rank_stability)}>{r.rank_stability}</Pill><span className="counts">{r.open_inc_n} <small>inc</small> · {r.open_com_n} <small>com</small></span></button>)}</Card></div>{selected&&<div className="detail-drawer"><button className="close" onClick={()=>setSelected(undefined)}><X size={17}/></button><div className="eyebrow">LOCATION DETAIL</div><h2>{selected.location_id}</h2><p>{selected.zone} · rank #{selected.rank}</p><div className="drawer-score"><strong>{selected.priority_score}</strong><span>priority score</span><Pill tone={stabilityClass(selected.rank_stability)}>{selected.rank_stability} rank</Pill></div><h4>Why this location</h4><p>{selected.why}</p><h4>Evidence gaps</h4>{selected.evidence_gaps.split(' | ').map(x=><p className="gap" key={x}><Info size={14}/>{x}</p>)}<button className="primary full">Open full receipt <ArrowRight size={15}/></button></div>}</div>}

 function Location({row}:{row?:QueueRow}){if(!row)return <div className="content"><Card><p>No location selected.</p></Card></div>;return <div className="content"><div className="location-header"><div><div className="eyebrow">LOCATION RECEIPT</div><h2>{row.location_id} <Pill tone={stabilityClass(row.rank_stability)}>{row.rank_stability} rank</Pill></h2><p>{row.zone} · arterial road · fair condition · high pedestrian activity</p></div><div className="score-ring"><strong>{row.priority_score}</strong><span>score</span></div></div><div className="two-col"><Card><div className="eyebrow">WHY THIS LOCATION</div><h3>Current records point here first</h3>{['Open incidents and complaints are active now','Recent harm raises the review priority','Exposure means more people may be affected'].map(x=><div className="bullet" key={x}><Check size={14}/>{x}</div>)}</Card><Card><div className="eyebrow">EVIDENCE GAPS / VERIFY</div><h3>What a visit should check</h3>{row.evidence_gaps.split(' | ').map(x=><div className="bullet warning" key={x}><Info size={14}/>{x}</div>)}</Card></div><Card><div className="card-title"><div><div className="eyebrow">SCORE RECEIPT</div><h3>Six components behind this rank</h3></div><span className="helper">Hover to inspect points</span></div><div className="receipt"><i/><i/><i/><i/><i/><i/></div><div className="receipt-labels"><span>Open load</span><span>Severity</span><span>Complaints</span><span>Age</span><span>Exposure</span><span>History</span></div></Card><Card><div className="eyebrow">WORK ORDER BRIEF</div><pre>Inspect {row.location_id} for current open items, verify the reported harm, and record any changes found on site.\n\nSynthetic educational data. Locations and records are simulated.</pre><button className="outline"><Download size={15}/> Download brief</button></Card></div>}
function Strategy(){return <div className="content"><Card><div className="card-title"><div><div className="eyebrow">STRATEGY COMPARISON</div><h2>How should the crew spend 60 crew-days?</h2></div><Pill tone="blue">Balanced option</Pill></div><input className="wide-range" type="range" min="10" max="300" defaultValue="60"/><p className="helper">Each strategy wins the metric it optimises. FixFirst balances current harm, backlog, and exposure.</p><div className="strategy-table">{[['FixFirst','18','60','41','68%'],['Worst severity first','15','60','39','63%'],['Most open complaints','22','60','34','57%'],['Most past incidents','12','60','21','41%'],['Random (mean of 200)','16','60','18','33%']].map((r,i)=><div className={i===0?'best':''} key={r[0]}><strong>{r[0]}</strong><span>{r[1]} locations</span><span>{r[2]} days</span><span>{r[3]} incidents cleared</span><b>{r[4]}</b></div>)}</div></Card><Card className="chart-card"><div className="eyebrow">OUTCOMES</div><h3>Existing backlog cleared</h3><div className="bars">{[['FixFirst',82],['Severity first',74],['Complaints first',68],['Past incidents',42],['Random',31]].map(x=><div key={x[0]}><span>{x[0]}</span><i><em style={{width:`${x[1]}%`}}/></i><b>{x[1]}%</b></div>)}</div></Card><p className="caption">This counts existing backlog cleared, not future incidents prevented.</p></div>}
function Evidence(){return <div className="content"><div className="evidence-hero"><div><div className="eyebrow">EVIDENCE</div><h2>What the data can and cannot support</h2><p>Past incident history is statistically no better than chance. Use current open items and exposure to decide what to review.</p></div><Pill tone="blue">Transparent by design</Pill></div><div className="metrics"><Metric label="Association tests" value="208" detail="0 significant after correction" icon={BarChart3}/><Metric label="ML probes" value="9" detail="all at baseline level" icon={ShieldCheck}/><Metric label="Raw history deviance" value="2.5173" detail="vs 1.093 global mean" icon={Zap}/><Metric label="Confident top-50" value="0" detail="on history alone" icon={Info}/></div><Card><div className="eyebrow">VALIDATION</div><h3>Ranking method comparison</h3><div className="validation">{[['Global mean','1.0930'],['Raw H1 history','2.5173'],['Poisson GLM','1.1506'],['Empirical-Bayes','1.0934']].map((x,i)=><div key={x[0]} className={i===0?'best':''}><span>{x[0]}</span><b>{x[1]}</b><small>{i===0?'best available baseline':''}</small></div>)}</div></Card><div className="two-col"><Card><div className="eyebrow">LIMITATIONS</div><h3>Keep this in view</h3>{['Rank stability is separate from priority score.','Wide historical intervals are not a reason to ignore current harm.','This tool supports review order, not a final public-safety decision.'].map(x=><div className="bullet" key={x}><Info size={14}/>{x}</div>)}</Card><Card><div className="eyebrow">METHOD</div><h3>Current-facts triage</h3><p className="helper">The ranking combines open incident load, injury severity, open complaint load, complaint age, exposure, and a shrunk history signal. Historic records alone do not separate locations reliably.</p><button className="text-btn">Read full disclosure <ArrowRight size={14}/></button></Card></div></div>}
