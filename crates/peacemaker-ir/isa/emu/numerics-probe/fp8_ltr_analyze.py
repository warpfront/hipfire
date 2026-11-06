#!/usr/bin/env python3
"""Decodes an fp8_probe `loadtr` receipt directory (run inside it): checks every full-EXEC wave of the named sets against the
transpose mapping documented in emu/memory.rs (`global_load_tr_b64`), i.e. result byte t of lane l = byte (l & 7) of the
chunk loaded by lane (l & 16) + 4 * ((l >> 3) & 1) + (t & 3) + 8 * (t >> 2).

  cd <receipt dir> && fp8_ltr_analyze.py seq rand_a8 rand_a4 rand_a2 rand_a1 qres_s516 qres_s1032 saddr_a8 exec8
"""
import numpy as np, sys, json
def check(name, saddr=False):
    ins=np.fromfile(f'ltr.{name}.in.bin',dtype=np.uint32).reshape(-1,33)
    out=np.fromfile(f'ltr.{name}.out.bin',dtype=np.uint32).reshape(-1,3,32,2)
    n=ins.shape[0]
    mask=ins[:,0]; off=ins[:,1:].astype(np.int64)
    # reconstruct origin per (case,lane,byte)
    b=np.zeros((n,32,8),dtype=np.int64)
    org=np.zeros((n,32,8),dtype=np.int64)
    for p in range(3):
        w=out[:,p,:,:]  # n,32,2
        bytes_=np.stack([(w[:,:,j//4]>>(8*(j%4)))&255 for j in range(8)],axis=2).astype(np.int64)
        org|=bytes_<<(8*p)
    lane=np.arange(32)
    exp=np.zeros((n,32,8),dtype=np.int64)
    for l in range(32):
        for t in range(8):
            s=(l&16)|(((l>>3)&1)*4)+(t&3)+(t>>2)*8
            exp[:,l,t]=off[:,s]+(l&7)
    full=(mask==0xffffffff)
    ok=(org==exp).all(axis=2)  # n,32
    print(name,'cases',n,'full-exec cases',full.sum(),'lane-results matching',ok[full].mean() if full.any() else None,'bad cases',(~ok[full].all(axis=1)).sum())
    return org,exp,ok,off,mask
if __name__=='__main__':
    for nm in sys.argv[1:]:
        check(nm)
