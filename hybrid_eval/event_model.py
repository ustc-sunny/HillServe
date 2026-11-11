"""Dependency/resource event model. Virtual time is never Python wall time.
No paper-policy or FPGA timing defaults. Buffer ownership spans producer/consumer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import heapq, math
@dataclass
class Task:
    id: str
    duration_ms: float
    resources: dict = field(default_factory=dict)
    deps: tuple = ()
    acquire: dict = field(default_factory=dict)
    release: dict = field(default_factory=dict)
    profile_id: str | None = None

def schedule(tasks, capacities, buffers=None):
    buffers = dict(buffers or {})
    tasks = list(tasks); by_id={t.id:t for t in tasks}
    if len(by_id)!=len(tasks): raise ValueError('duplicate event')
    for t in tasks:
        if not math.isfinite(t.duration_ms) or t.duration_ms<0:raise ValueError('invalid duration')
        if any(d not in by_id or d==t.id for d in t.deps):raise ValueError('invalid producer dependency')
        for requests,limits in [(t.resources,capacities),(t.acquire,buffers),(t.release,buffers)]:
            if any(k not in limits or not isinstance(v,int) or v<=0 or v>limits[k] for k,v in requests.items()):raise ValueError('invalid capacity request')
    free=dict(capacities); slots=dict(buffers); pending=dict(by_id);done=set();running=[];timeline=[];now=0.;sequence=0
    while pending or running:
        for name,t in list(pending.items()):
            if not all(d in done for d in t.deps):continue
            if any(free[k]<v for k,v in t.resources.items()) or any(slots[k]<v for k,v in t.acquire.items()):continue
            for k,v in t.resources.items():free[k]-=v
            for k,v in t.acquire.items():slots[k]-=v
            heapq.heappush(running,(now+t.duration_ms,sequence,t,now));sequence+=1;del pending[name]
        if not running:raise ValueError('dependency cycle or buffer deadlock')
        now=running[0][0]
        while running and running[0][0]==now:
            _,_,t,start=heapq.heappop(running)
            for k,v in t.resources.items():free[k]+=v
            for k,v in t.release.items():
                slots[k]+=v
                if slots[k]>buffers[k]:raise ValueError('release without ownership')
            done.add(t.id);timeline.append({'event':t.id,'start_ms':start,'end_ms':now,'profile_id':t.profile_id})
    if slots!=buffers:raise ValueError('unreleased buffer')
    return {'evidence_mode':'S_UNVALIDATED','validation_scope':'synthetic_scheduler_only','target_time_ms':now,'timeline':timeline,'paper_eligible':False}

def from_profile(event_id,profile,**kwargs):
    if profile.get('evidence_mode')!='M_COMPONENT' or profile.get('status')!='passed':raise ValueError('real measured component required')
    if profile.get('includes_queue_wait') is not False:raise ValueError('queue wait would be double-counted')
    value=profile.get('service_ms')
    if value is None or not math.isfinite(value) or value<0:raise ValueError('missing service time; never default to zero')
    return Task(event_id,value,profile_id=profile['profile_id'],**kwargs)

def require_publication_validation(manifest):
    if not manifest.get('independent_holdout_passed') or not manifest.get('measurement_ids') or not manifest.get('holdout_ids'):raise ValueError('no independent measured validation')
    if set(manifest['measurement_ids']) & set(manifest['holdout_ids']):raise ValueError('calibration/holdout leakage')
    if manifest.get('scope') not in ['component','retrieval_chain','closed_loop_decode','full_request']:raise ValueError('scope unspecified')
