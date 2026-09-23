#!/usr/bin/env python3
"""Added C4s arm: unchanged C4 scales, exactly fixed z=-8*s, no free zero."""
from measure_constraints import *
@njit(parallel=True)
def errors_sym(G,steps,thr):
    out=np.zeros((len(G),3),np.float64)
    for b in prange(len(G)):
        maxi=0
        for j in range(1,256):
            if abs(G[b,j])>abs(G[b,maxi]):maxi=j
        for h in range(2):
            s=steps[b,h];z=np.float32(-8*s)
            for j in range(h*128,(h+1)*128):
                q=assign(G[b,j],z,s)
                v=np.float32(np.float32(s*np.float32(q))+z)
                err=float(v)-float(G[b,j])
                if abs(G[b,j])>=thr:out[b,0]+=err*err;out[b,1]+=1.
                if j==maxi:out[b,2]=abs(err)/max(abs(float(G[b,j])),1e-30)
    return out

def measure_sym(e,maxrows):
    t=time.time();nr,k=e['shape'];rows=np.arange(nr) if not maxrows or nr<=maxrows else np.linspace(0,nr-1,maxrows,dtype=np.int64)
    u=parent_tensor(e)[rows];w=(u.astype(np.uint32)<<16).view(np.float32);G=fwht(w,awq(e),S1,S2);del w,u
    raw=np.ndarray((nr,k//256,136),np.uint8,buffer=mm,offset=e['off'])[rows].reshape(-1,136)
    hdr=raw[:,:8].copy().view('<f2').astype(np.float32);s=hdr[:,[0,2]]
    half=G.reshape(-1,2,128);assert np.array_equal(trunc16((half.max(2)-half.min(2))/np.float32(15)),s)
    master=np.maximum(trunc16(s.reshape(len(rows),-1).max(1)/np.float32(256)),np.float32(2.**-24))
    c4=nearest_pow(s,np.repeat(master,k//256)[:,None],0,8)
    assert np.array_equal(c4,c4.astype(np.float16).astype(np.float32))
    assert np.array_equal(-8*c4,(-8*c4).astype(np.float16).astype(np.float32))
    thr=float(np.quantile(np.abs(G),.99));out=errors_sym(G,c4,thr);n=float(out[:,1].sum())
    return {'tensor':e['name'],'family':family(e['name']),'shape':e['shape'],'rows_measured':len(rows),'groups':len(G),'coefficients':int(G.size),'tail_threshold':thr,'seconds':time.time()-t,'metrics':{'C4s':{'tail_sse':float(out[:,0].sum()),'tail_n':int(n),'tail_mse':float(out[:,0].sum()/n),'maxcoef_rel_sum':float(out[:,2].sum()),'maxcoef_rel_mean':float(out[:,2].mean()),'maxcoef_rel_worst':float(out[:,2].max())}}}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['pilot','all'],required=True);ap.add_argument('--max-rows',type=int,default=0);a=ap.parse_args()
    selected=qentries
    if a.scope=='pilot':selected=[e for e in selected if re.search(r'layers\.(0|3|31|32|60|63)\.',e['name']) or e['name']=='lm_head.weight']
    path=OUT/f'{a.scope}_rows{a.max_rows}_symmetry.jsonl'
    with path.open('w',buffering=1) as log:
        meta={'kind':'metadata','scope':a.scope,'max_rows':a.max_rows,'fixture':str(FIX),'fixture_sha256':'80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'argv':os.sys.argv,'n_tensors':len(selected),'contract':'C4s exactC4scale, zero=-8scale, noLloydzeroupdate, f32assignsameDAG'}
        log.write(json.dumps(meta)+'\n');print(json.dumps(meta),flush=True)
        for e in selected:
            r=measure_sym(e,a.max_rows);log.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
    print('COMPLETE',str(path),flush=True)
