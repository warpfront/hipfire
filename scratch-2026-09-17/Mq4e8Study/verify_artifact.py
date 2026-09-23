#!/usr/bin/env python3
"""Consumer validation of completed artifacts against the independent CPU oracle."""
from measure_constraints import *
def verify(arm,path):
    p=Path(path);ents=repack.read_index(str(p))[-1];other={e['name']:e for e in ents};assert set(other)==set(emap)
    art=np.memmap(p,np.uint8,mode='r');unchanged=0;unchanged_bytes=0;t=time.time()
    for name,e in emap.items():
        o=other[name];assert e['shape']==o['shape'] and e['qt']==o['qt'] and e['ds']==o['ds'],name
        if e['qt']!=44:
            for start in range(0,e['ds'],8*1024*1024):
                n=min(8*1024*1024,e['ds']-start)
                assert np.array_equal(mm[e['off']+start:e['off']+start+n],art[o['off']+start:o['off']+start+n]),name
            unchanged+=1;unchanged_bytes+=e['ds']
    oracle_name='model.language_model.layers.0.linear_attn.in_proj_a.weight';o=other[oracle_name]
    got=art[o['off']:o['off']+o['ds']];want=np.frombuffer((OUT/f'oracle_L0_in_proj_a_{arm}.qt44').read_bytes(),np.uint8)
    bad=np.flatnonzero(got!=want)
    r={'arm':arm,'artifact':str(p),'tensor_count':len(ents),'qt44_tensors':sum(e['qt']==44 for e in ents),'unchanged_non_qt44_tensors':unchanged,'unchanged_non_qt44_bytes':unchanged_bytes,'oracle_tensor':oracle_name,'oracle_bytes':int(got.size),'oracle_mismatched_bytes':len(bad),'first_oracle_mismatch_offsets':bad[:20].tolist()}
    (OUT/f'artifact_{arm}_verification.json').write_text(json.dumps(r,indent=2)+'\n');assert len(bad)==0,r
    groups=0;zeros_checked=0
    for e in qentries:
        o=other[e['name']]
        old=np.ndarray((e['ds']//136,4),dtype='<f2',buffer=mm,offset=e['off'],strides=(136,2)).astype(np.float32)
        new=np.ndarray((e['ds']//136,4),dtype='<f2',buffer=art,offset=o['off'],strides=(136,2)).astype(np.float32)
        s=old[:,[0,2]];gm=s.max(1)[:,None];actual=new[:,[0,2]]
        if arm=='C0':expected=s
        elif arm=='C1':
            expected=np.broadcast_to(gm,s.shape).copy()
            for d in range(1,5):
                cand=gm*np.float32(2.**-d);valid=cand==cand.astype(np.float16).astype(np.float32)
                expected=np.where((abs(s-cand)<abs(s-expected))&valid,cand,expected)
        elif arm=='C2':
            bits=trunc16(gm/16).astype(np.float16).view(np.uint16)&np.uint16(0xfff0)
            b=np.maximum(bits,np.uint16(0x10)).view(np.float16).astype(np.float32)
            expected=b*np.clip(np.floor(s/b+np.float32(.5)),1,16)
        elif arm=='C3':expected=np.broadcast_to(gm,s.shape)
        else:
            m,k=e['shape'];b=np.maximum(trunc16(s.reshape(m,-1).max(1)/256),np.float32(2.**-24))
            if arm=='C4z':b=np.maximum(b.astype(np.float16).view(np.uint16)&np.uint16(0xfff0),np.uint16(0x10)).view(np.float16).astype(np.float32)
            expected=nearest_pow(s,np.repeat(b,k//256)[:,None],0,8)
        assert np.array_equal(actual,expected),(arm,e['name'],'scale mismatch')
        assert np.isfinite(new).all(),(arm,e['name'],'nonfinite header')
        if arm=='C4z':
            k=new[:,[1,3]]/actual
            assert np.array_equal(k,np.rint(k)) and (k>=-15).all() and (k<=0).all()
            zeros_checked+=k.size
        if arm=='C4s':
            assert np.array_equal(new[:,[1,3]],-8*actual);zeros_checked+=actual.size
        groups+=len(new)
    hr={'artifact':str(p),'arm':arm,'tensors_checked':len(qentries),'groups_checked':groups,'half_scales_checked':groups*2,'scale_mismatches':0,'nonfinite_headers':0,'integer_zero_constraints_checked':zeros_checked,'wall_seconds':time.time()-t}
    (OUT/f'artifact_{arm}_header_verification.json').write_text(json.dumps(hr,indent=2)+'\n')
    return r,hr
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['C0','C1','C2','C3','C4','C4z','C4s'],required=True);ap.add_argument('--model',required=True);a=ap.parse_args()
    print(json.dumps(verify(a.arm,a.model),indent=2))
