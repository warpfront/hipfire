#!/usr/bin/env python3
import hashlib, json, time
import numpy as np
from measure_constraints import FIX,OUT,entries,mm,family
bins_ratio=np.array([0,.25,.5,.625,.75,.875,1,1.125,1.25,1.5,2,4,np.inf])
bins_norm=np.array([0,.25,.5,.625,.75,.875,.9375,.96875,1,1.00001])
stats={};t=time.time()
for e in entries:
    if e['qt']!=44:continue
    hdr=np.ndarray((e['ds']//136,4),dtype='<f2',buffer=mm,offset=e['off'],strides=(136,2)).astype(np.float32)
    s=hdr[:,[0,2]];good=(s>0).all(1)&np.isfinite(s).all(1);s=s[good]
    r=s[:,1]/s[:,0];sn=s/s.max(1)[:,None]
    a=stats.setdefault(family(e['name']),{'tensors':0,'groups':0,'degenerate':0,'s1_s0_hist':[0]*(len(bins_ratio)-1),'s0_max_hist':[0]*(len(bins_norm)-1),'s1_max_hist':[0]*(len(bins_norm)-1),'min_ratio':float('inf'),'max_ratio':0.})
    a['tensors']+=1;a['groups']+=len(hdr);a['degenerate']+=int((~good).sum())
    for key,v,bins in [('s1_s0_hist',r,bins_ratio),('s0_max_hist',sn[:,0],bins_norm),('s1_max_hist',sn[:,1],bins_norm)]:a[key]=[int(x+y) for x,y in zip(a[key],np.histogram(v,bins)[0])]
    a['min_ratio']=min(a['min_ratio'],float(r.min()));a['max_ratio']=max(a['max_ratio'],float(r.max()))
out={'fixture':str(FIX),'fixture_sha256':'80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3','bins_s1_s0':[str(x) for x in bins_ratio],'bins_half_max':[str(x) for x in bins_norm],'seconds':time.time()-t,'families':stats,'definition':'h0=K0..127,h1=K128..255; s_hi/s_lo means s1/s0, not sorted max/min; left-closed histogram intervals'}
(OUT/'header_ratios.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
