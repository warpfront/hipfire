#!/usr/bin/env python3
"""Synthetic corruption controls only; no GPU/model is opened."""
import copy,hashlib,importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('v',Path(__file__).with_name('verify-mtp-lru.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def collector():
    cells=[]
    for rows in range(1,9):
        ids=[(71*l+43*r+17*k)%512 for l in range(48) for r in range(rows) for k in range(10)]
        raw=b''.join(i.to_bytes(4,'little',signed=True) for i in ids)
        cells.append(dict(rows=rows,layers=48,topk=10,actual_ids=ids,actual_ids_sha256=hashlib.sha256(raw).hexdigest(),guard_rows=rows,epoch_before=1,epoch_after_readback=1,publication_refused_while_guarded=True,wrong_width_refused=True,guard_active_through_readback=True))
    publications=[dict(expert=e,generation=e+1,slot=0,gate_up=1234,down=5678,epoch=e+1,gu_complete_dn_not_enqueued_refused=True,full_payload_readback_exact=True,table_pair_readback_exact=True,batch8_publication_refused=True) for e in range(2)]
    return dict(schema='hipfire-fn-mtp-lru-oracle-v1',phase='collector',status='PASS',complete=True,evidence=dict(no_model=True,model_or_native_head_exercised=False,collector_cells=cells,publications=publications,missing_layer_negative={'detected':True}))

def snapshot(expert,generation,count):
    layers=[]
    for layer in range(48):
        cg=[100000+layer*10000+e*8 for e in range(512)];cd=[x+10000000 for x in cg]
        gu=cg.copy();dn=cd.copy();mapping=[None]*512
        gu[expert]=50000000+layer*100;dn[expert]=60000000+layer*100
        mapping[expert]=dict(slot=0,expert=expert,generation=generation)
        layers.append(dict(layer=layer,epoch=generation,gate_up=gu,down=dn,canonical_gate_up=cg,canonical_down=cd,mapping=mapping))
    return dict(schema='hipfire-expert-cache-snapshot-v1',rows_cap=8,poisoned=None,outstanding=False,layers=layers,pending=[],stats=dict(selected=count,cache_hits=count,fills_completed=count,prefill_restores=0,verify_batches=count,max_guard_rows=4))

def model():
    before=snapshot(0,1,0);after=snapshot(1,2,5)
    ticket=dict(layer=0,slot=0,expert=1,generation=2)
    frames=[];pos=100;cursor=1
    for accepted in [0,1,3,3,3]:
        count=accepted+1
        m=dict(source='Mtp',route='Batched',actual_drafts=3,requested_drafts=3,outcome_mismatch=False,accepted_prefix=accepted,committed=count,position_before=pos,position_after=pos+count)
        frames.append(dict(measurement=m,committed_ids=list(range(cursor,cursor+count)),pending_after=[ticket]))
        pos+=count;cursor+=count
    turnover=dict(layer=0,slot=0,from_expert=0,to_expert=1,generation_before=1,generation_after=2,epoch_before=1,epoch_after=2,gate_up=50000000,down=60000000)
    state=['a'*64,'b'*64]
    n=dict(ids=list(range(16)),native_target_state_sha256=state,native_target_all_state_exact=True,native_head_teacher_forced_comparison='diagnostic_cross_head_schedule',native_mtp_state_sha256=state,native_pending_hidden_sha256='e'*64,teacher_forced_different_parts=[0,1],teacher_forced_pending_exact=False,actual_acceptance_cells=dict(zero=True,interior=True,full=True),windows=frames,completed_victim_turnover=[turnover],cache_before_native=before,cache_after_native=after,terminal=dict(source='native_mtp',verified_drafts=3,consumed_outputs=1,retained_prefix_repaired=True,pending_seed_flushed=True,native_pending_hidden_sha256='e'*64,live_next_turn_seed=7,target_state_exact=True,native_schedule_exact=True,replay_frames=copy.deepcopy(frames[:2]),repaired_target_state_sha256=state,repaired_target_ar_reference_sha256=state,mtp_repaired_state_sha256=state,live_next_turn_in_place=True,live_plan_state_unchanged=True,live_next_turn_positions_exact=True,live_next_turn_exact=False))
    c=dict(prompt_sha256='c'*64,prompt_ids_sha256='d'*64,ar_ids=list(range(16)),ar_final_state_sha256=state,native=n,target_transactions=dict(full_logits_bitwise_claim=True,abandon_and_ticket_reuse_checked=True,cells=[dict(rows=rows,accepted_drafts=a,path=path,all_state_exact=True,subsequent_ar_checked=True) for rows in [1,2,4,8] for a in range(rows) for path in ['native_retain','generic_commit_prefix']]),prefix_restore=dict(cells=[dict(reused=4096,chunk=4096,all_mtp_state_exact=True,seed_exact=True)]*2))
    return dict(phase='model',status='PASS',complete=True,evidence=dict(native_mtp_attached=True,cache_enabled=True,head_reserve_k=7,target_rows=[1,2,4,8],performance_claim=False,cases=[c]))

class Controls(unittest.TestCase):
    def test_valid_synthetic_references(self):
        v.verify_collector(collector());v.verify_model(model(),True)
    def test_collector_rejects_missing_transposed_or_guarded_mutation(self):
        for name in ['missing','transpose','epoch','busy']:
            r=collector();c=r['evidence']['collector_cells'][3]
            if name=='missing':c['actual_ids']=c['actual_ids'][:-10]
            elif name=='transpose':c['actual_ids'][10:20]=c['actual_ids'][:10]
            elif name=='epoch':c['epoch_after_readback']+=1
            else:c['publication_refused_while_guarded']=False
            with self.subTest(name=name),self.assertRaises(v.Invalid):v.verify_collector(r)
    def test_model_rejects_ar_fallback_or_fake_acceptance(self):
        for key,value in [('source','ArRouted'),('actual_drafts',0),('outcome_mismatch',True),('committed',9)]:
            r=model();r['evidence']['cases'][0]['native']['windows'][0]['measurement'][key]=value
            with self.subTest(key=key),self.assertRaises(v.Invalid):v.verify_model(r,True)
        r=model();r['evidence']['cases'][0]['native']['windows']=r['evidence']['cases'][0]['native']['windows'][2:]
        with self.assertRaises(v.Invalid):v.verify_model(r,True)
    def test_eviction_counter_and_ar_pending_cannot_prove_turnover(self):
        for mode in ['counter','ar_pending','unpublished','generation','half_pair','guard','transcript']:
            r=model();n=r['evidence']['cases'][0]['native']
            if mode=='counter':n['completed_victim_turnover']=[];n['cache_after_native']['stats']['evictions']=9999
            elif mode=='ar_pending':n['cache_before_native']['pending']=n['windows'][0]['pending_after']
            elif mode=='unpublished':n['cache_after_native']['layers'][0]['mapping'][1]=None
            elif mode=='generation':n['completed_victim_turnover'][0]['generation_after']=1
            elif mode=='half_pair':n['cache_after_native']['layers'][0]['down'][1]=n['cache_after_native']['layers'][0]['canonical_down'][1]
            elif mode=='guard':n['cache_after_native']['outstanding']=True
            else:n['windows'][0]['committed_ids']=[999]
            with self.subTest(mode=mode),self.assertRaises(v.Invalid):v.verify_model(r,True)
    def test_terminal_rejects_other_schedule_or_unsupported_exactness(self):
        for mode in ['missing_replay','other_schedule','outcome','ar_reference','repair','pending','live_state','seed_claim']:
            r=model();t=r['evidence']['cases'][0]['native']['terminal']
            if mode=='missing_replay':t['replay_frames']=[]
            elif mode=='other_schedule':t['replay_frames'][0]['measurement']['actual_drafts']=2
            elif mode=='outcome':t['replay_frames'][0]['measurement']['outcome_mismatch']=True
            elif mode=='ar_reference':t['repaired_target_ar_reference_sha256']=['f'*64]
            elif mode=='repair':t['retained_prefix_repaired']=False
            elif mode=='pending':t['native_pending_hidden_sha256']=''
            elif mode=='live_state':t['live_plan_state_unchanged']=False
            else:t['live_next_turn_exact']=True
            with self.subTest(mode=mode),self.assertRaises(v.Invalid):v.verify_model(r,True)
if __name__=='__main__':unittest.main()
