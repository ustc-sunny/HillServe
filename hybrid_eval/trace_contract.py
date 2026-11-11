"""Validation contract only; does not authorize policy execution or synthesize traces."""
from pathlib import PurePosixPath

def validate(manifest, events, mode):
    if mode not in ('recompute_scores','fixed_request_ids'):raise ValueError('unsupported mode')
    if mode=='recompute_scores':
        if not manifest.get('method_is_confirmed') or not manifest.get('policy_hash'):raise ValueError('unbound policy')
        if manifest.get('oracle_ids_available_to_policy'):raise ValueError('oracle IDs leak')
    if manifest.get('future_query_visibility')!='dependency_gated':raise ValueError('future query release missing')
    cap=manifest.get('payload_bytes',0)
    if not isinstance(cap,int) or not 0<cap<=256*1024**2:raise ValueError('trace size budget')
    ids=set();completed=set();append_positions=set()
    for e in events:
        key=(e['request'],e['layer'],e['original_position'])
        if e['id'] in ids:raise ValueError('duplicate event')
        if not set(e['deps'])<=completed:raise ValueError('producer not released')
        if e.get('query') and not e['deps']:raise ValueError('query requires actual producer')
        for ref in e.get('data_refs',[]):
            path=PurePosixPath(ref['file'])
            if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe payload path')
            if ref['offset']<0 or ref['length']<=0 or ref['offset']+ref['length']>cap:raise ValueError('invalid payload slice')
        if e.get('append'):
            if key in append_positions:raise ValueError('nonincremental duplicate append')
            append_positions.add(key)
        ids.add(e['id']);completed.add(e['id'])
    return {'status':'passed','scope':'schema_and_dependency_only','method_execution_validated':False}
