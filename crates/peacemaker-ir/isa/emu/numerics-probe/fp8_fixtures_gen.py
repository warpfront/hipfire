#!/usr/bin/env python3
"""Regenerates crates/peacemaker-ir/src/emu/fp8_fixtures.rs from fp8_probe receipts (r1 = card B, r2 = card E).

  fp8_fixtures_gen.py --rcpt <dir holding r1-cvt r1-wmma r1-loadtr r2-cvt r2-wmma r2-loadtr> --out <fp8_fixtures.rs>

Deterministic: fixed random seeds pick the wmma samples/waves and load_tr waves out of the raw receipts; the run table of
v_cvt_pk_fp8_f32 is copied verbatim. The tests in emu/fp8_receipts.rs, emu/wmma.rs and emu/memory.rs consume the output.
"""
import numpy as np, sys, random, struct, argparse
KN=['v_wmma_f32_16x16x16_fp8_fp8','v_wmma_f32_16x16x16_fp8_bf8','v_wmma_f32_16x16x16_bf8_fp8','v_wmma_f32_16x16x16_bf8_bf8']
FMT=[('e4m3','e4m3'),('e4m3','e5m2'),('e5m2','e4m3'),('e5m2','e5m2')]
def dec(fmt,c):
    s=c>>7
    if fmt=='e4m3':
        e=(c>>3)&15;m=c&7
        if e==15 and m==7: return ('nan',)
        return ('f',s,(m|8) if e else m, (max(e,1)-7-3))
    e=(c>>2)&31;m=c&3
    if e==31: return ('inf',s) if m==0 else ('nan',)
    return ('f',s,(m|4) if e else m,(max(e,1)-15-2))
def tags(kind,a,b,c,d):
    fa,fb=FMT[kind]
    t=set()
    prods=[]
    special=False
    for k in range(16):
        x=dec(fa,a[k]);y=dec(fb,b[k])
        if x[0]!='f' or y[0]!='f': special=True; continue
        if x[2] and y[2]: prods.append((k,x[2]*y[2],x[3]+y[3],x[1]^y[1]))
        if (x[2] and x[3]==max(1,0)-7-3 and False): pass
    if (d&0x7fffffff)>0x7f800000: t.add('nan')
    elif (d&0x7fffffff)==0x7f800000: t.add('inf')
    if special: t.add('special')
    if not special and prods and d&0x7fffffff==0: t.add('cancel0')
    if not special and len(prods)==2:
        p,q=prods[0][0],prods[1][0]
        if p>>2==q>>2 and p>>1!=q>>1: t.add('quad2')
        elif p>>3==q>>3: t.add('step2')
        else: t.add('xstep2')
    if not special and len(prods)>=10:
        ex=[p[2] for p in prods]
        if max(ex)-min(ex)>=10: t.add('wide')
        else: t.add('dense')
    if not special and any(((a[k]&0x78)==0 and a[k]&7) if fa=='e4m3' else ((a[k]&0x7c)==0 and a[k]&3) for k in range(16)): t.add('sub')
    if not t: t.add('rand')
    return t
def main(dirs, per_tag=4, seed=5):
    rnd=random.Random(seed)
    out=[]
    for kind in range(4):
        n=KN[kind]
        picks={}
        for d in dirs:
            try:
                inb=np.memmap(f'{d}/fwmma.{n}.in.bin',dtype=np.uint8,mode='r').reshape(-1,1536)
                outb=np.memmap(f'{d}/fwmma.{n}.v0.m0.out.bin',dtype=np.uint32,mode='r').reshape(-1,8,32)
            except Exception as e:
                continue
            nc=min(len(inb),len(outb))
            for _ in range(6000):
                ci=rnd.randrange(65539,nc)
                a_all=inb[ci,:256].reshape(16,16);b_all=inb[ci,256:512].reshape(16,16)
                c_all=inb[ci,512:].view(np.uint32).reshape(16,16)
                r=rnd.randrange(16);col=rnd.randrange(16)
                a=[int(x) for x in a_all[r]];b=[int(x) for x in b_all[col]];c=int(c_all[r,col])
                dv=int(outb[ci,r&7,(r>>3)*16+col])
                for t in tags(kind,a,b,c,dv):
                    lst=picks.setdefault(t,[])
                    if len(lst)<per_tag and not any(x[1:]==(a,b,c,dv) for x in lst): lst.append((kind,a,b,c,dv))
        # single-product Cartesian rows (case 3 + a*256 + b; the product sits on the diagonal element r == c == 5)
        carts={0:[(0x00,0x7f),(0x7f,0x38),(0x00,0x00),(0x80,0x01),(0x7e,0x7e),(0x01,0x01)],
               1:[(0x00,0x7c),(0x00,0xfc),(0x38,0x7c),(0xb8,0x7c),(0x00,0x7d),(0x7f,0x7b)],
               2:[(0x7c,0x00),(0xfc,0x00),(0x7c,0x38),(0x7c,0xb8),(0x7d,0x00),(0x7b,0x7e)],
               3:[(0x00,0x7c),(0x7c,0x00),(0x3c,0x7c),(0x3c,0xfc),(0xfc,0xfc),(0x7d,0x3c),(0x7b,0x7b),(0x01,0x01)]}[kind]
        for d in dirs:
            try:
                inb=np.memmap(f'{d}/fwmma.{n}.in.bin',dtype=np.uint8,mode='r').reshape(-1,1536)
                outb=np.memmap(f'{d}/fwmma.{n}.v0.m0.out.bin',dtype=np.uint32,mode='r').reshape(-1,8,32)
            except Exception: continue
            for (av,bv) in carts:
                ci=3+av*256+bv
                a_all=inb[ci,:256].reshape(16,16);b_all=inb[ci,256:512].reshape(16,16)
                c_all=inb[ci,512:].view(np.uint32).reshape(16,16)
                r=5
                a=[int(x) for x in a_all[r]];b=[int(x) for x in b_all[r]];c=int(c_all[r,r]);dv=int(outb[ci,r&7,(r>>3)*16+r])
                lst=picks.setdefault('cart',[])
                if not any(x[1:]==(a,b,c,dv) for x in lst): lst.append((kind,a,b,c,dv))
            break
        for t,lst in sorted(picks.items()):
            for (kk,a,b,c,dv) in lst: out.append((t,kk,a,b,c,dv))
    return out

# ---------------------------------------------------------------------------------- fixture emission
def rs_arr(vals, per=8, fmt='0x{:08x}'):
    rows=[]
    for i in range(0,len(vals),per):
        rows.append('    '+', '.join(fmt.format(v) for v in vals[i:i+per])+',')
    return '\n'.join(rows)
def cvt_runs(dir_):
    r=np.fromfile(f'{dir_}/cvt.s0_lo.bin',dtype=np.uint32).reshape(-1,2)
    return [(int(s),int(w)&0xff) for s,w in r]
def waves(dirs, rnd):
    out=[]
    for kind in range(4):
        n=KN[kind]
        d=dirs[0]
        inb=np.memmap(f'{d}/fwmma.{n}.in.bin',dtype=np.uint8,mode='r').reshape(-1,1536)
        outb=np.memmap(f'{d}/fwmma.{n}.v0.m0.out.bin',dtype=np.uint32,mode='r').reshape(-1,8,32)
        nc=min(len(inb),len(outb))
        picks=[('control',0),('control_b_onehot',1)]
        want={'dense':1,'special':1,'sub':1}
        tries=0
        while want and tries<20000:
            tries+=1
            ci=rnd.randrange(65539,nc)
            a_all=inb[ci,:256].reshape(16,16);b_all=inb[ci,256:512].reshape(16,16)
            fa,fb=FMT[kind]
            # classify the whole wave by its rows/cols
            specials=any(dec(fa,int(x))[0]!='f' for x in a_all.flat) or any(dec(fb,int(x))[0]!='f' for x in b_all.flat)
            subs=any(((int(x)&0x78)==0 and int(x)&7) if fa=='e4m3' else ((int(x)&0x7c)==0 and int(x)&3) for x in a_all.flat)
            tag='special' if specials else ('sub' if subs else 'dense')
            if tag in want:
                picks.append((tag,ci)); del want[tag]
        for tag,ci in picks:
            A=inb[ci,:256].reshape(16,16);B=inb[ci,256:512].reshape(16,16);C=inb[ci,512:].view(np.uint32).reshape(16,16)
            def frag(M):
                regs=[0]*64
                for l in range(32):
                    r=l&15;h=l>>4
                    for v in range(2):
                        w=0
                        for bb in range(4): w|=int(M[r,8*h+4*v+bb])<<(8*bb)
                        regs[v*32+l]=w
                return regs
            cimg=[0]*256
            for j in range(8):
                for l in range(32):
                    cimg[j*32+l]=int(C[8*(l>>4)+j,l&15])
            dimg=[int(outb[ci,j,l]) for j in range(8) for l in range(32)]
            out.append((kind,tag,ci,frag(A),frag(B),cimg,dimg))
    return out
def ltr_waves(dirs, rnd, sets):
    out=[]
    for d,name,k in sets:
        try:
            ins=np.fromfile(f'{d}/ltr.{name}.in.bin',dtype=np.uint32).reshape(-1,33)
            res=np.fromfile(f'{d}/ltr.{name}.out.bin',dtype=np.uint32).reshape(-1,3,32,2)
        except Exception: continue
        n=len(ins)
        idxs=list(range(min(n,k))) if name=='seq' else rnd.sample(range(n),k)
        for ci in idxs:
            mask=int(ins[ci,0]); offs=[int(x) for x in ins[ci,1:]]
            org=[[0]*8 for _ in range(32)]
            sent=[[None]*8 for _ in range(32)]
            for l in range(32):
                for t in range(8):
                    o=0
                    for p in range(3):
                        w=int(res[ci,p,l,t//4]); b=(w>>(8*(t%4)))&255
                        o|=b<<(8*p)
                    org[l][t]=o
            out.append((name,mask,offs,org))
    return out

ap = argparse.ArgumentParser()
ap.add_argument('--rcpt', required=True)
ap.add_argument('--out', required=True)
_a = ap.parse_args()
R = _a.rcpt.rstrip('/') + '/'
OUT = _a.out
rnd = random.Random(20261105)

cases=main([R+'r1-wmma',R+'r2-wmma'],per_tag=4,seed=11)
wave_list=waves([R+'r2-wmma'],rnd)
runs=cvt_runs(R+'r1-cvt')
assert runs==cvt_runs(R+'r2-cvt')
ltr=[]
for d,name,k in [(R+'r1-loadtr','seq',4),(R+'r1-loadtr','rand_a8',2),(R+'r1-loadtr','rand_a4',2),(R+'r1-loadtr','rand_a2',2),(R+'r1-loadtr','rand_a1',2),(R+'r1-loadtr','qres_s516',3),(R+'r1-loadtr','qres_s1032',3),(R+'r2-loadtr','saddr_a8',2),(R+'r2-loadtr','saddr_a4',2),(R+'r2-loadtr','saddr_a1',2),(R+'r2-loadtr','exec8',6)]:
    for w in ltr_waves(None,rnd,[(d,name,k)]):
        if w[1]!=0: ltr.append(w)
def hx(vals,fmt='0x{:08x}'): return ', '.join(fmt.format(v) for v in vals)
o=[]
o.append('''// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raw gfx1201 hardware samples of the B2 attention capabilities (test data only, generated from the receipts of
//! `isa/emu/numerics-probe/fp8_probe.hip`; see `isa/emu/float_edges.md` "FP8 attention capabilities (B2)"):
//! the run table of `v_cvt_pk_fp8_f32` over all 2^32 f32 inputs, dense FP8/BF8 WMMA operand/result samples,
//! whole WMMA register waves and `global_load_tr_b64` waves. r1 = card B (UUID GPU-e475645fe0200397), r2 = card E
//! (BDF 0000:c3:00.0); the two cards agree word for word wherever both ran the same set.
#![allow(dead_code)]

/// `(first input, E4M3 byte)` of every run of the hardware table; a run lasts until the next start (the last until
/// 2^32). Sweep `cvt.s0_lo` of receipt r1 (identical across `s1_lo`/`s0_hi`/`s1_hi`, all 8 MODE values and card E).
pub const CVT_RUNS: [(u32, u8); %d] = [''' % len(runs))
for i in range(0,len(runs),4):
    o.append('    '+' '.join('(0x%08x, 0x%02x),'%(s,b) for s,b in runs[i:i+4]))
o.append('];\n')
o.append('''/// One dense WMMA output element: `kind` 0..4 = fp8_fp8, fp8_bf8, bf8_fp8, bf8_bf8; `a`/`b` the 16 raw codes of the A row /
/// B column in physical K order, `c` the accumulator input, `d` the hardware result word. `tag` names why it was kept.
pub struct WmmaCase {
    pub tag: &'static str,
    pub kind: u8,
    pub a: [u8; 16],
    pub b: [u8; 16],
    pub c: u32,
    pub d: u32,
}
''')
o.append('pub const WMMA_CASES: [WmmaCase; %d] = [' % len(cases))
for (t,kk,a,b,c,d) in cases:
    o.append('    WmmaCase { tag: "%s", kind: %d, a: [%s], b: [%s], c: 0x%08x, d: 0x%08x },'%(t,kk,hx(a,'0x{:02x}'),hx(b,'0x{:02x}'),c,d))
o.append('];\n')
o.append('''/// A whole wave: A and B are 2 VGPRs x 32 lanes (`[vgpr * 32 + lane]`), C and D 8 x 32, exactly the register images
/// the probe loaded and read back.
pub struct Wave {
    pub tag: &'static str,
    pub kind: u8,
    pub a: [u32; 64],
    pub b: [u32; 64],
    pub c: [u32; 256],
    pub d: [u32; 256],
}
''')
o.append('pub const WAVES: [Wave; %d] = ['%len(wave_list))
for (kind,tag,ci,A,B,C,D) in wave_list:
    o.append('    Wave {\n        tag: "%s",\n        kind: %d,\n        a: [%s],\n        b: [%s],\n        c: [%s],\n        d: [%s],\n    },'%(tag,kind,hx(A),hx(B),hx(C),hx(D)))
o.append('];\n')
o.append('''/// A `global_load_tr_b64` wave: EXEC `mask`, the per-lane byte offset into the plane region and `org[lane][t]`, the
/// source offset of result byte `t` of every lane (three byte planes reveal it, see the probe).
pub struct LtrWave {
    pub set: &'static str,
    pub mask: u32,
    pub offs: [u32; 32],
    pub org: [[u32; 8]; 32],
}
''')
o.append('pub const LTR_WAVES: [LtrWave; %d] = ['%len(ltr))
for (name,mask,offs,org) in ltr:
    o.append('    LtrWave {\n        set: "%s",\n        mask: 0x%08x,\n        offs: [%s],\n        org: ['%(name,mask,hx(offs,'0x{:x}')))
    for l in range(32): o.append('            [%s],'%hx(org[l],'0x{:x}'))
    o.append('        ],\n    },')
o.append('];')
open(OUT,'w').write('\n'.join(o)+'\n')
print(len(cases),len(wave_list),len(runs),len(ltr))
