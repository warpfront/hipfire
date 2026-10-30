#!/usr/bin/env python3
"""Posthoc only: strict MTP/LRU receipts, never launches GPU or retries a model."""
import argparse, hashlib, json
from pathlib import Path

class Invalid(ValueError):
    pass

def need(ok, msg):
    if not ok:
        raise Invalid(msg)

def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_cpu(r):
    need(r.get('status')=='PASS' and r.get('complete') is True and r.get('phase')=='cpu','CPU controls incomplete')
    e=r['evidence']
    need(e.get('gpu_opened') is False and e.get('model_loaded') is False and e.get('gpu_or_model_qualification') is False,'CPU controls mislabel GPU proof')
    need(len(e.get('cases',[]))>=14 and all(c.get('status')=='PASS' for c in e['cases']),'CPU negative controls missing')

def verify_collector(r):
    need(r.get('schema')=='hipfire-fn-mtp-lru-oracle-v1' and r.get('phase')=='collector' and r.get('status')=='PASS' and r.get('complete') is True,'collector incomplete')
    e=r['evidence']
    need(e.get('no_model') is True and e.get('model_or_native_head_exercised') is False,'collector mislabels model proof')
    cells=e.get('collector_cells',[])
    need([c.get('rows') for c in cells]==list(range(1,9)),'collector rows1..8 incomplete/duplicated')
    for c in cells:
        rows=c['rows']
        want=[(71*l+43*r+17*k)%512 for l in range(48) for r in range(rows) for k in range(10)]
        need(c.get('actual_ids')==want,f'collector missing/transposed/aliased route R{rows}')
        raw=b''.join(i.to_bytes(4,'little',signed=True) for i in want)
        need(c.get('actual_ids_sha256')==hashlib.sha256(raw).hexdigest(),'collector hash')
        need(c.get('layers')==48 and c.get('topk')==10 and c.get('guard_rows')==rows,'collector/guard geometry')
        need(c.get('epoch_before')==c.get('epoch_after_readback'),'publication changed epoch inside guarded batch')
        for field in ['publication_refused_while_guarded','wrong_width_refused','guard_active_through_readback']:
            need(c.get(field) is True,f'{field} absent')
    need(e.get('missing_layer_negative',{}).get('detected') is True,'unwritten layer negative missing')
    ps=e.get('publications',[])
    need(len(ps)==2 and [p['expert'] for p in ps]==[0,1] and [p['generation'] for p in ps]==[1,2],'publication turnover missing')
    need(ps[0]['slot']==ps[1]['slot'] and ps[0]['gate_up']==ps[1]['gate_up'] and ps[0]['down']==ps[1]['down'] and ps[1]['epoch']>ps[0]['epoch'],'physical slot pair/epoch turnover')
    for p in ps:
        for f in ['gu_complete_dn_not_enqueued_refused','full_payload_readback_exact','table_pair_readback_exact','batch8_publication_refused']:
            need(p.get(f) is True,f'publication {f} missing')

def verify_publication(r):
    need(r.get('status')=='PASS' and r.get('all_checks_complete') is True,'async publication incomplete')
    need(r.get('production_mutable_residency_contract_exercised') is True and r.get('gpu_quiescence_proven_at_cleanup') is True and r.get('cleanup_errors')==[],'publication lifecycle proof missing')
    checks=r.get('checks',[])
    need(len(checks)==15 and all(c.get('status')=='PASS' and c.get('production_token_guard') is True and c.get('full_outputs_and_guards_match_canonical_host') is True for c in checks),'publication eager/recorded/PM4 cells incomplete')
    need(len({c['case'] for c in checks})==5,'publication mappings missing')
    copies=r.get('copies',[])
    need(len(copies)==2,'paired async fills missing')
    for c in copies:
        need(c.get('both_completion_events_ready_before_publish') is True and c.get('full_payload_copy_exact') is True and c.get('explicit_compute_to_copy_event') is True,'async pair ordering/payload proof missing')
    negatives={x['case'] for x in r.get('negative_controls',[]) if x.get('status')=='PASS'}
    need({'busy_token_blocks_publication','gu_complete_dn_not_enqueued','stale_slot_generation','actual_partial_table_upload_poisoned'}<=negatives,'publication negatives incomplete')
    need(len(r.get('typed_id_copy',[]))==3,'opaque ID-copy eager/recorded/PM4 proof missing')

def published_slots(s):
    need(s.get('schema')=='hipfire-expert-cache-snapshot-v1' and s.get('poisoned') is None and s.get('outstanding') is False and s.get('rows_cap')==8,'cache snapshot authority')
    layers=s.get('layers',[]);need(len(layers)==48 and {l['layer'] for l in layers}==set(range(48)),'cache layer set')
    slots={}
    for l in layers:
        for key in ['mapping','gate_up','down','canonical_gate_up','canonical_down']:need(len(l.get(key,[]))==512,'cache table extent '+key)
        for expert,m in enumerate(l['mapping']):
            gu=l['gate_up'][expert];dn=l['down'][expert]
            need(isinstance(gu,int) and gu>0 and isinstance(dn,int) and dn>0,'null/invalid paired address')
            if m is None:need(gu==l['canonical_gate_up'][expert] and dn==l['canonical_down'][expert],'canonical GU/DN mismatch');continue
            need(m['expert']==expert and m['generation']>0,'wrong slot expert/generation')
            need(gu!=l['canonical_gate_up'][expert] and dn!=l['canonical_down'][expert],'half published GU/DN')
            key=(l['layer'],m['slot']);need(key not in slots,'slot aliases two experts')
            slots[key]={**m,'gate_up':gu,'down':dn,'epoch':l['epoch']}
    return slots

def verify_model(r,cache):
    need(r.get('phase')=='model' and r.get('status')=='PASS' and r.get('complete') is True,'model oracle incomplete')
    e=r['evidence']
    need(e.get('native_mtp_attached') is True and e.get('cache_enabled') is cache,'native head/cache arm mismatch')
    need(e.get('performance_claim') is False,'instrumented oracle must not claim speed')
    need(e.get('head_reserve_k')==7 and e.get('target_rows')==[1,2,4,8],'matched K7 reserve / R8 target qualification missing')
    cases=e.get('cases',[]);need(cases,'no model cases')
    for c in cases:
        n=c['native'];ids=n['ids']
        need(ids and ids==c['ar_ids'],'native IDs != sequential AR')
        need(n.get('native_target_state_sha256')==c.get('ar_final_state_sha256') and n.get('native_target_state_sha256'),'native full target state != AR')
        for field in ['native_target_all_state_exact']:
            need(n.get(field) is True,f'{field} missing')
        need(n.get('native_head_teacher_forced_comparison')=='diagnostic_cross_head_schedule','cross-head schedule differences must be diagnostic')
        need(n.get('native_mtp_state_sha256') and len(n.get('native_pending_hidden_sha256',''))==64,'missing same-native-schedule head/pending evidence')
        target=c.get('target_transactions',{})
        need(target.get('full_logits_bitwise_claim') is True and target.get('abandon_and_ticket_reuse_checked') is True,'target logits/abandon coverage missing')
        cells=target.get('cells',[])
        want={(rows,a,path) for rows in [1,2,4,8] for a in range(rows) for path in ['native_retain','generic_commit_prefix']}
        got={(x['rows'],x['accepted_drafts'],x['path']) for x in cells}
        need(got==want and len(cells)==len(want),'R1/2/4/8 accepted-prefix cells missing/duplicated')
        need(all(x.get('all_state_exact') is True and x.get('subsequent_ar_checked') is True for x in cells),'target fullstate/continuation failure')
        accepted=n.get('actual_acceptance_cells',{})
        need(all(accepted.get(k) is True for k in ['zero','interior','full']),'actual native acceptance coverage missing')
        native=[]
        for frame in n['windows']:
            m=frame['measurement']
            if m['source']=='Mtp':
                need(m['actual_drafts']>0 and m['actual_drafts']==m['requested_drafts'] and m['outcome_mismatch'] is False,'requested K passed as actual MTP')
                need(m['route'] in ['Batched','Interleaved'] and 0<=m['accepted_prefix']<=m['actual_drafts'] and m['committed']>0,'ARfallback/impossible native transaction')
                native.append(frame)
        emitted=[ids[0]]
        previous=None
        for frame in n['windows']:
            m=frame['measurement']
            need(len(frame.get('committed_ids',[]))==m['committed'],'missing actual window token IDs')
            if previous is not None:need(m['position_before']==previous,'window target positions discontinuous')
            previous=m['position_after'];emitted.extend(frame['committed_ids'])
        need(emitted==ids,'committed native/AR-tail windows do not reconstruct exact transcript')
        need(native,'no actual native window')
        need(any(f['measurement']['route']=='Batched' for f in native),'native batch guard was never exercised (incremental-only is separate qualification)')
        observed={
            'zero':any(f['measurement']['accepted_prefix']==0 for f in native),
            'interior':any(0<f['measurement']['accepted_prefix']<f['measurement']['actual_drafts'] for f in native),
            'full':any(f['measurement']['accepted_prefix']==f['measurement']['actual_drafts'] for f in native),
        }
        need(all(observed.values()) and observed==accepted,'acceptance booleans do not match actual native windows')
        for f in native:
            m=f['measurement']
            need(m['committed']==m['accepted_prefix']+1 and m['position_after']-m['position_before']==m['committed'],'native pending-seed/commit counts inconsistent')
        if cache:
            changes=n.get('completed_victim_turnover',[]);need(changes,'counter-only eviction is not turnover')
            before=n['cache_before_native'];after=n['cache_after_native']
            old_slots=published_slots(before);new_slots=published_slots(after)
            need(after['stats']['prefill_restores']==before['stats']['prefill_restores'],'bulk restore inside native interval')
            need(after['stats']['fills_completed']>before['stats']['fills_completed'] and after['stats']['cache_hits']>before['stats']['cache_hits'],'native fills/hits absent')
            queued=[p for frame in native for p in (frame.get('pending_after') or [])]
            initial=before.get('pending',[])
            for t in changes:
                ticket={'layer':t['layer'],'slot':t['slot'],'expert':t['to_expert'],'generation':t['generation_after']}
                need(t['from_expert']!=t['to_expert'] and t['generation_after']>t['generation_before'] and t['epoch_after']>t['epoch_before'],'non-turnover ticket')
                need(ticket in queued and ticket not in initial,'AR warmup fill mislabeled native enqueue')
                key=(t['layer'],t['slot']);need(key in old_slots and key in new_slots,'turnover slot absent from actual published tables')
                old=old_slots[key];new=new_slots[key]
                need(old['expert']==t['from_expert'] and new['expert']==t['to_expert'] and old['generation']==t['generation_before'] and new['generation']==t['generation_after'],'turnover witness != actual published tickets')
                need(old['gate_up']==new['gate_up']==t['gate_up'] and old['down']==new['down']==t['down'],'turnover GU/DN pair moved')
            batch=[f['measurement']['actual_drafts']+1 for f in native if f['measurement']['route']=='Batched']
            if batch:
                need(after['stats']['verify_batches']>before['stats']['verify_batches'] and after['stats']['max_guard_rows']>=max(batch),'actual production verify guard missing')
        t=n.get('terminal',{})
        need(t.get('source')=='native_mtp' and t.get('target_state_exact') is True,'terminal consumed-prefix target repair missing')
        need(t.get('native_schedule_exact') is True,'terminal was rebuilt under a different prefill/head schedule')
        replay=t.get('replay_frames',[])
        need(replay and len(replay)<=len(n['windows']),'terminal native replay missing')
        need(all(f['measurement'].get('outcome_mismatch') is False for f in replay),'terminal replay did not execute its requested native windows')
        keys=['source','route','requested_drafts','actual_drafts','accepted_prefix','committed','position_before','position_after']
        signature=lambda frame:([frame['measurement'][key] for key in keys],frame['committed_ids'])
        need([signature(f) for f in replay]==[signature(f) for f in n['windows'][:len(replay)]],'terminal replay differs from the original native schedule')
        last=replay[-1]['measurement']
        need(last['source']=='Mtp' and last['actual_drafts']>0 and last['committed']>1,'terminal did not stop inside an actual native window')
        need(t.get('verified_drafts')==last['actual_drafts'] and t.get('consumed_outputs')==1,'terminal consumed-prefix extent is not the actual native window')
        need(t.get('retained_prefix_repaired') is True and t.get('pending_seed_flushed') is True,'terminal repair/flush was not completed')
        target=t.get('repaired_target_state_sha256')
        need(target and target==t.get('repaired_target_ar_reference_sha256'),'terminal target differs from sequential AR consuming the same history')
        need(t.get('mtp_repaired_state_sha256'),'repaired terminal head/target state hashes missing')
        need(len(t.get('native_pending_hidden_sha256',''))==64 and isinstance(t.get('live_next_turn_seed'),int),'terminal pending/live-seed receipts missing')
        for key in ['live_next_turn_in_place','live_plan_state_unchanged','live_next_turn_positions_exact']:
            need(t.get(key) is True,'terminal live continuation evidence missing '+key)
        need(t.get('live_next_turn_exact') is False,'cross-head-schedule next-seed exactness must not be inferred')
        pr=c.get('prefix_restore',{})
        need(len(pr.get('cells',[]))==2 and all(x['reused']>=x['chunk'] and x['all_mtp_state_exact'] is True and x['seed_exact'] is True for x in pr['cells']),'cold/hit branch state proof missing')
    mapped={(c['prompt_sha256'],c['prompt_ids_sha256']):c for c in cases}
    need(len(mapped)==len(cases),'duplicate prompt/token identities hide missing model cases')
    return mapped

def _check_run(root,name,manifest):
    run=load(root/(name+'.run.json'))
    key='publication' if name=='publication' else 'oracle'
    need(run.get('status')=='PASS' and run.get('oracle_sha256')==manifest['binaries'][key]['sha256'],'run/binary evidence mismatch '+name)
    need(run.get('report_sha256')==sha(root/(name+'.json')),'report changed after run '+name)
    need(run.get('stdout_sha256')==sha(root/(name+'.stdout.log')) and run.get('stderr_sha256')==sha(root/(name+'.stderr.log')),'raw logs changed after run '+name)

def verify_directory(root,partial=False):
    root=Path(root);manifest=load(root/'RUN.json');missing=[]
    for key,fn in [('cpu',verify_cpu),('collector',verify_collector),('publication',verify_publication)]:
        p=root/(key+'.json')
        if not p.exists():missing.append(key);continue
        fn(load(p))
        _check_run(root,key,manifest)
    models={}
    for label,cache in [('cache-on',True),('cache-off',False)]:
        p=root/(label+'.json')
        if not p.exists():missing.append(label);continue
        models[label]=verify_model(load(p),cache)
        _check_run(root,label,manifest)
    if len(models)==2:
        off=load(root/'cache-off.json')['evidence'];on=load(root/'cache-on.json')['evidence']
        for key in ['native_k','head_reserve_k','max_seq','state_format','kv_backend','replay_backend']:
            need(off.get(key) is not None and off.get(key)==on.get(key),'matched model configuration differs '+key)
        a=models['cache-off'];b=models['cache-on'];need(a.keys()==b.keys(),'arm prompt/token identities differ')
        for k in a:
            need(a[k]['ar_ids']==b[k]['ar_ids'],'cache off/on AR IDs differ')
            need(a[k]['native']['ids']==b[k]['native']['ids'],'cache off/on MTP IDs differ')
            need(a[k]['ar_final_state_sha256']==b[k]['ar_final_state_sha256'],'cache off/on target state differs')
            left=a[k]['native'];right=b[k]['native']
            keys=['source','route','requested_drafts','actual_drafts','accepted_prefix','committed','position_before','position_after']
            schedule=lambda n:[([f['measurement'][key] for key in keys],f['committed_ids']) for f in n['windows']]
            need(schedule(left)==schedule(right),'actual native schedule differs; no cross-schedule head exactness claim')
            need(left['native_mtp_state_sha256']==right['native_mtp_state_sha256'],'cache off/on head state differs under identical native schedule')
            need(left['native_pending_hidden_sha256']==right['native_pending_hidden_sha256'],'cache off/on pending hidden differs under identical native schedule')
            for key in ['repaired_target_state_sha256','repaired_target_ar_reference_sha256','mtp_repaired_state_sha256','native_pending_hidden_sha256','live_next_turn_seed']:
                need(left['terminal'].get(key)==right['terminal'].get(key),'cache off/on terminal state differs under identical native replay '+key)
    if missing and not partial:raise Invalid('missing phases: '+','.join(missing))
    return {'status':'PARTIAL' if missing else 'PASS','complete':not missing,'missing':missing,'performance_claim':False,'model_qualified':not missing,'run_manifest_sha256':sha(root/'RUN.json'),'limitations':['No live Q8Ef, sampled MTP, ngram-copy, SSE terminal or uninstrumented speed proof.']}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory');ap.add_argument('--partial',action='store_true');ap.add_argument('--out');a=ap.parse_args()
    try:r=verify_directory(a.directory,a.partial)
    except Exception as e:r={'status':'FAIL','complete':False,'error':repr(e),'performance_claim':False}
    out=json.dumps(r,indent=2)
    if a.out:
        with open(a.out,'x') as f:f.write(out+'\n')
    print(out)
    return 0 if r['status'] in ('PASS','PARTIAL') else 1
if __name__=='__main__':raise SystemExit(main())
