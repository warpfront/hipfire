#!/usr/bin/env python3
"""Offline measurement only: original AWQ+FWHT coefficients, exact fixture baseline."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','24')
import argparse, collections, hashlib, importlib.util, json, math, re, struct, time
from pathlib import Path
import numpy as np
from numba import njit, prange
from gguf import GGUFReader
ROOT=Path('/home/kaden/ClaudeCode/warpfront/wt-mq4e8')
OUT=ROOT/'scratch-2026-09-17/Mq4e8Study'
FIX=Path('/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.ctrl.hfq')
PARENT=Path('/home/kaden/qcal/parents/qwen3.8-27b')
IMATRIX=Path('/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf')
spec=importlib.util.spec_from_file_location('repack',ROOT/'tools/quant-design/mq4c_repack.py')
repack=importlib.util.module_from_spec(spec);spec.loader.exec_module(repack)
prefix,magic,version,arch,moff,doff,entries=repack.read_index(str(FIX))
emap={e['name']:e for e in entries};qentries=[e for e in entries if e['qt']==44]
mm=np.memmap(FIX,dtype=np.uint8,mode='r')
parent_index=json.loads((PARENT/'model.safetensors.index.json').read_text())['weight_map']
imreader=GGUFReader(str(IMATRIX));imt={t.name:t for t in imreader.tensors}
shards={}
NAMES=['fixture','C0','C1','C2','C3','C4','C4z']
FAMILIES={'mlp.gate_proj':'ffn_gate','mlp.up_proj':'ffn_up','mlp.down_proj':'ffn_down','self_attn.q_proj':'attn_q','self_attn.k_proj':'attn_k','self_attn.v_proj':'attn_v','self_attn.o_proj':'attn_output','linear_attn.in_proj_qkv':'attn_qkv','linear_attn.in_proj_z':'attn_gate','linear_attn.in_proj_a':'ssm_alpha','linear_attn.in_proj_b':'ssm_beta','linear_attn.out_proj':'ssm_out'}
def family(name):return re.sub(r'^model\.language_model\.layers\.\d+\.', '', name).removesuffix('.weight')
def trunc16(x):
    x=np.asarray(x,dtype=np.float32);y=x.astype(np.float16)
    return np.where(np.abs(y.astype(np.float32))>np.abs(x),np.nextafter(y,np.float16(0)),y).astype(np.float32)
def signs(seed):
    out=[]
    for _ in range(256):
        seed=(seed*1103515245+12345)&0x7fffffff;out.append(1. if (seed>>16)&1 else -1.)
    return np.array(out,np.float32)
S1=signs(42);S2=signs(1042)
def parent_tensor(e):
    path=PARENT/parent_index[e['name']]
    if path not in shards:
        with path.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
        shards[path]=(np.memmap(path,dtype=np.uint8,mode='r'),h,8+n)
    b,h,off=shards[path];t=h[e['name']];assert t['dtype']=='BF16'
    return np.ndarray(t['shape'],dtype='<u2',buffer=b,offset=off+t['data_offsets'][0])
def awq(e):
    layer=re.search(r'layers\.(\d+)\.',e['name'])
    if not layer:return np.ones(e['shape'][1],np.float32)
    v=imt[f"blk.{layer[1]}.{FAMILIES[family(e['name'])]}.weight.in_sum2"].data.reshape(-1).astype(np.float64)
    logs=np.log(np.fmin(np.fmax(v,1e-12),1e30))*(float(np.float32(.55))*.5)
    avg=sum(float(x) for x in logs)/len(logs)
    s=np.clip(np.exp(logs-avg).astype(np.float32),np.float32(.01),np.float32(100))
    side=emap[e['name'].removesuffix('.weight')+'.awq_scale.weight']
    stored=np.ndarray((len(s),),dtype='<f2',buffer=mm,offset=side['off'])
    assert np.array_equal(trunc16(s),stored),e['name']
    return s
@njit(parallel=True)
def fwht(w,s,s1,s2):
    k=w.shape[1];out=np.empty((w.shape[0]*(k//256),256),np.float32)
    for b in prange(len(out)):
        r=b//(k//256);g=b%(k//256)
        for j in range(256):out[b,j]=np.float32(np.float32(w[r,g*256+j]*s[g*256+j])*s1[j])
        stride=1
        while stride<256:
            for i in range(0,256,2*stride):
                for j in range(stride):
                    a=out[b,i+j];c=out[b,i+j+stride]
                    out[b,i+j]=np.float32(a+c);out[b,i+j+stride]=np.float32(a-c)
            stride*=2
        for j in range(256):out[b,j]=np.float32(out[b,j]*np.float32(.0625*s2[j]))
    return out
@njit
def trunc_scalar(x):
    a=abs(float(x))
    if a==0:return np.float32(x)
    if a<2.0**-14:step=2.0**-24
    else:step=math.ldexp(1.,math.frexp(a)[1]-11)
    y=math.floor(a/step)*step
    return np.float32(-y if x<0 else y)
@njit
def assign(w,z,s):
    if s==0:return 0
    a=np.float32(np.float32(w-z)*np.float32(1./s))
    return min(15,max(0,int(math.floor(np.float32(a+np.float32(.5))))))
@njit(parallel=True)
def errors(G,hdr,payload,scales,thr):
    # Per config and group: tail squared sum, tail count, max-coefficient relative error.
    out=np.zeros((7,len(G),3),np.float64)
    for b in prange(len(G)):
        maxi=0
        for j in range(1,256):
            if abs(G[b,j])>abs(G[b,maxi]):maxi=j
        for cfg in range(7):
            for h in range(2):
                if cfg==0:s=hdr[b,h*2];z=hdr[b,h*2+1]
                else:
                    s=scales[cfg-1,b,h];z=hdr[b,h*2+1]
                    if cfg==6:
                        iz=min(0,max(-15,int(math.floor(float(z)/float(s)+.5))))
                        z=np.float32(s*np.float32(iz))
                    for it in range(8):
                        acc=0.
                        for j in range(h*128,(h+1)*128):
                            q=assign(G[b,j],z,s)
                            acc+=float(np.float32(G[b,j]-np.float32(s*np.float32(q))))
                        nz=trunc_scalar(np.float32(acc/128.))
                        if cfg==6:
                            iz=min(0,max(-15,int(math.floor((acc/128.)/float(s)+.5))))
                            nz=np.float32(s*np.float32(iz))
                        if nz==z:break
                        z=nz
                for j in range(h*128,(h+1)*128):
                    if cfg==0:q=(int(payload[b,j//2])>>(4*(j%2)))&15
                    else:q=assign(G[b,j],z,s)
                    v=np.float32(np.float32(s*np.float32(q))+z)
                    err=float(v)-float(G[b,j])
                    if abs(G[b,j])>=thr:out[cfg,b,0]+=err*err;out[cfg,b,1]+=1.
                    if j==maxi:out[cfg,b,2]=abs(err)/max(abs(float(G[b,j])),1e-30)
    return out

def nearest_pow(s,base,emin,emax):
    exp=np.clip(np.floor(np.log2(np.maximum(s/base,1e-30))).astype(np.int32),emin,emax)
    low=np.ldexp(base,exp);nex=np.ldexp(base,np.minimum(exp+1,emax))
    return np.where(abs(s-nex)<=abs(s-low),nex,low).astype(np.float32)
def run_tensor(e,maxrows):
    t=time.time();nr,k=e['shape'];rows=np.arange(nr) if not maxrows or nr<=maxrows else np.linspace(0,nr-1,maxrows,dtype=np.int64)
    u=parent_tensor(e)[rows];w=(u.astype(np.uint32)<<16).view(np.float32);G=fwht(w,awq(e),S1,S2);del w,u
    raw=np.ndarray((nr,k//256,136),np.uint8,buffer=mm,offset=e['off'])[rows].reshape(-1,136)
    hdr=raw[:,:8].copy().view('<f2').astype(np.float32);s=hdr[:,[0,2]];Gmax=s.max(1)[:,None]
    half=G.reshape(-1,2,128);lo=half.min(2);hi=half.max(2)
    assert np.array_equal(trunc16((hi-lo)/np.float32(15)),s),f"scale oracle mismatch {e['name']}"
    assert np.array_equal(trunc16(lo),hdr[:,[1,3]]),f"zero oracle mismatch {e['name']}"
    assert (s>0).all()
    c1=np.broadcast_to(Gmax,s.shape).copy()
    c1bad=0
    for d in range(1,5):
        candidate=(Gmax*np.float32(2.**-d)).astype(np.float32)
        valid=candidate==candidate.astype(np.float16).astype(np.float32)
        better=abs(s-candidate)<abs(s-c1)
        c1bad+=int((better & ~valid).sum())
        c1=np.where(better & valid,candidate,c1)
    bb=trunc16(Gmax/np.float32(16)).astype(np.float16).view(np.uint16)&np.uint16(0xfff0)
    clamps=int((bb==0).sum());bb=np.maximum(bb,np.uint16(0x10))
    B=bb.view(np.float16).astype(np.float32)
    c2=B*np.clip(np.floor(s/B+np.float32(.5)),1,16)
    c3=np.broadcast_to(Gmax,s.shape)
    master=trunc16(s.reshape(len(rows),-1).max(1)/np.float32(256))
    c4clamps=int((master==0).sum())
    master=np.maximum(master,np.float32(2.**-24))
    c4=nearest_pow(s,np.repeat(master,k//256)[:,None],0,8)
    masterz_bits=master.astype(np.float16).view(np.uint16)&np.uint16(0xfff0)
    c4zclamps=int((masterz_bits==0).sum())
    masterz=np.maximum(masterz_bits,np.uint16(0x10)).view(np.float16).astype(np.float32)
    c4z=nearest_pow(s,np.repeat(masterz,k//256)[:,None],0,8)
    scales=np.stack([s,c1,c2,c3,c4,c4z]).astype(np.float32)
    assert np.array_equal(scales,scales.astype(np.float16).astype(np.float32))
    thr=float(np.quantile(np.abs(G),.99));out=errors(G,hdr,raw[:,8:].copy(),scales,thr)
    res={'tensor':e['name'],'family':family(e['name']),'shape':e['shape'],'rows_measured':len(rows),'groups':len(G),'coefficients':int(G.size),'tail_threshold':thr,'c1_nonrepresentable_candidates':c1bad,'c2_base_clamps':clamps,'c4_master_clamps':c4clamps,'c4z_master_clamps':c4zclamps,'seconds':time.time()-t,'metrics':{}}
    for i,name in enumerate(NAMES):
        a=out[i];n=float(a[:,1].sum());res['metrics'][name]={'tail_sse':float(a[:,0].sum()),'tail_n':int(n),'tail_mse':float(a[:,0].sum()/n),'maxcoef_rel_sum':float(a[:,2].sum()),'maxcoef_rel_mean':float(a[:,2].mean()),'maxcoef_rel_worst':float(a[:,2].max())}
    return res

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['pilot','all'],required=True);ap.add_argument('--max-rows',type=int,default=0);ap.add_argument('--only');a=ap.parse_args()
    selected=qentries
    if a.scope=='pilot':selected=[e for e in selected if re.search(r'layers\.(0|3|31|32|60|63)\.',e['name']) or e['name']=='lm_head.weight']
    if a.only:selected=[e for e in selected if a.only in e['name']]
    path=OUT/f'{a.scope}_rows{a.max_rows}'+Path('') if False else OUT/f'{a.scope}_rows{a.max_rows}.jsonl'
    with path.open('w',buffering=1) as log:
        meta={'kind':'metadata','scope':a.scope,'max_rows':a.max_rows,'fixture':str(FIX),'fixture_sha256':'80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3','original_parent':str(PARENT),'imatrix':str(IMATRIX),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'argv':os.sys.argv,'numba_threads':os.environ['NUMBA_NUM_THREADS'],'n_tensors':len(selected),'constraints':'composer-contract.txt','tail_definition':'per-tensor exact99th percentile on measured original AWQ+FWHT coefficients; same mask for all variants'}
        log.write(json.dumps(meta)+'\n');print(json.dumps(meta),flush=True)
        for e in selected:
            r=run_tensor(e,a.max_rows);log.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
    print('COMPLETE',str(path),flush=True)
if __name__=='__main__':main()
