#!/usr/bin/env python3
"""CPU-only algebra, packing, ISA census and cost model; no kernel implementation."""
import hashlib
import json
import random
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

E = Path('/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/IntFoldPlan')
MAX_I32 = 2**31 - 1
bounds = {}
for k in (5120, 17408, 17664, 32512, 32768):
    bounds[str(k)] = dict(c2_unscaled_box=k*120*16,
                         m1_xmaster_box=k*120*16*4,
                         m2z=k*120*1024, m2s=k*64*1024)
assert bounds['17408']['m2z'] == 2139095040 < MAX_I32
assert bounds['17664']['m2z'] > MAX_I32
assert bounds['32512']['m2s'] <= MAX_I32 < bounds['32768']['m2s']
assert 2*128*120*16*4 == 1966080 < 2**24

# Every possible first-half integer against all sign/carry boundaries of A1.
pack_count = 0
pack_min = 2**31
pack_max = -2**31
for a0 in range(-15360,13441):
    for a1 in (-15360,-8192,-1,0,1,8192,13440):
        packed = a0*65536+a1
        assert -2**31 <= packed <= MAX_I32
        lo = packed & 65535
        lo = lo - 65536 if lo & 32768 else lo
        hi = (packed+32768)//65536
        assert (hi,lo) == (a0,a1)
        pack_count += 1
        pack_min,pack_max = min(pack_min,packed), max(pack_max,packed)

rng = random.Random(20260918)
checked = []
for k in (256,512,5120,17408):
    for case in range(4):
        d = Fraction(rng.randrange(1,31),2**12)
        b_row = Fraction(rng.randrange(1,31),2**15)
        direct1 = Fraction(0); fold1 = Fraction(0)
        directz = Fraction(0); tz = 0
        directs = Fraction(0); ts = 0
        for g in range(k//256):
            b = Fraction(rng.randrange(1,31),2**14)
            tg = 0; zg = Fraction(0)
            for h in range(2):
                ex = rng.randrange(3); ew = rng.randrange(9)
                m = rng.randrange(1,17); z = Fraction(rng.randrange(-127,1),2**15)
                kh = rng.randrange(-15,1)
                qw = [rng.randrange(16) for _ in range(128)]
                qx = [rng.randrange(-8,8) for _ in range(128)]
                a = sum(w*x for w,x in zip(qw,qx)); s = sum(qx)
                assert -15360 <= a <= 13440 and -1024 <= s <= 896
                assert abs(a+kh*s) <= 15360
                tg += (m << ex)*a
                zg += z*(s*(1 << ex))
                direct1 += sum((b*m*w+z)*(d*(1 << ex)*x) for w,x in zip(qw,qx))
                tz += (1 << (ew+ex))*(a+kh*s)
                ts += (1 << (ew+ex))*sum((w-8)*x for w,x in zip(qw,qx))
                directz += sum(b_row*(1 << ew)*(w+kh)*d*(1 << ex)*x for w,x in zip(qw,qx))
                directs += sum(b_row*(1 << ew)*(w-8)*d*(1 << ex)*x for w,x in zip(qw,qx))
            assert abs(tg) <= 1966080
            fold1 += d*(b*tg+zg)
        assert direct1 == fold1
        assert directz == b_row*d*tz
        assert directs == b_row*d*ts
        checked.append(dict(k=k,case=case,exact_rational_equal=True))

text = (E/'tu-cand.dis.txt').read_text()
sym = re.search(r'^([0-9a-f]+) <gemm_mq4g256v2_residual_mmq_iu4_full_set>:\n',text,re.M)
body = text[sym.end():].split('\nDisassembly of section',1)[0]
inst = []
for line in body.splitlines():
    m = re.match(r'\s*([a-z][^/]*?)\s*// ([0-9A-Fa-f]+):',line)
    if m: inst.append((int(m.group(2),16),m.group(1).strip()))
fold = re.compile(r'v_(?:dual_)?(?:cvt_f32_i32|cvt_f32_u32|mul_f32|fmac_f32|fma_f32|add_f32|fmamk_f32|fmaak_f32)(?:_e32|_e64)?\b')
def category(op):
    if 'v_wmma_' in op: return 'wmma'
    if fold.search(op): return 'fold_valu'
    if op.startswith('v_'): return 'other_valu'
    if op.startswith('ds_load'): return 'fragment_load' if 'b64' in op else 'metadata_load'
    if op.startswith('ds_store'): return 'publish'
    if op.startswith('global_load'): return 'global_load'
    if op.startswith('global_inv'): return 'global_inv'
    if op.startswith('s_'): return 'scalar_control_wait_delay'
    return 'other'
trip = [(a,o) for a,o in inst if 0xe394 <= a <= 0xf010]
roll = [(a,o) for a,o in inst if 0xe354 <= a <= 0xe390]
counts = Counter(category(o) for a,o in trip)
assert counts['wmma'] == 32 and counts['fold_valu'] == 286 and len(trip) == 557
# Full-tile path only. All store M/N predicates true. No arbitrary branch guessing.
index = {a:i for i,(a,o) in enumerate(inst)}
pc = index[0xf014]; epi = []; seen = set()
while pc < len(inst):
    a,o = inst[pc]
    assert a not in seen
    seen.add(a); epi.append((a,o))
    op = o.split()[0]
    if op == 's_endpgm': break
    if 'branch' in op:
        assert op in ('s_branch','s_cbranch_execz','s_cbranch_execnz')
        if op in ('s_branch','s_cbranch_execnz'):
            offset = int(o.split()[1]); offset = offset-65536 if offset>32767 else offset
            pc = index[a+4+offset*4]; continue
    pc += 1
assert len(epi) == 1438
packets = len(trip)+len(roll)/2+len(epi)/40
service = packets + 32*7
assert packets == 600.45 and service == 824.45
alpha = 0.8
rates = {}
for label,p_new,overheads,ep in [('m1_simple',360,(16,48),32/40),
                                ('m1_ideal_pairing_lower_bound',296,(16,48),32/40),
                                ('m2z',192,(8,24),128/40),
                                ('m2s',136,(8,24),128/40),
                                ('impossible_zero_cost_exponent_m2s',64,(0,0),0)]:
    rows=[]
    for overhead in overheads:
        saved=286-p_new-overhead-ep
        rows.append(dict(new_fold=p_new,extra_packets=overhead,extra_epilogue=ep,
                         saved=saved,raw_packet_fraction=saved/packets,
                         service_fraction=saved/service,
                         rate_tops=190/(1-alpha*saved/service),
                         gate_us=482.7*(1-alpha*saved/service)))
    rates[label]=rows

# Supplied shares sum to 98%; leave the missing 2% as unassigned, not zero.
t0=527
shares=dict(gemm=.64,gdn=.14,fa=.13,glue=.07,unassigned=.02)
# Isolated census says int4 preparation is 105.09*4096 of the GEMM bucket.
quant_fraction=105.09*4096/(4624.3*1000*.643)
gemm_compute_fraction=1-quant_fraction
preamble_save=t0*.14*(223.5/(544.16+223.5+61.15))*.24
fa_save=t0-t0/1.0258
scan_save=t0*.14*(544.16/(544.16+223.5+61.15))*(1-190/544.16)
verdict=[]
for rate in (203,207,216,220,242.16802538188858):
    ideal=t0*(1-.64+.64*190/rate)-fa_save-preamble_save
    corrected=t0*(1-.64*gemm_compute_fraction+.64*gemm_compute_fraction*190/rate)-fa_save-preamble_save
    verdict.append(dict(rate=rate,optimistic_all_gemm_scales_us=ideal,
                        with_unchanged_quantizer_us=corrected,
                        plus_unmeasured_190us_scan_us=corrected-scan_save))
required_all=190*t0*.64/(333.3333333333333-(t0*.36-fa_save-preamble_save))
comp=t0*.64*gemm_compute_fraction
required_compute=190*comp/(333.3333333333333-(t0-comp-fa_save-preamble_save))

out=dict(scope='CPU-only; no GPU kernel/performance/correctness result',
         bounds=bounds,margin_m2z_k17408=MAX_I32-bounds['17408']['m2z'],
         packed_half_checks=pack_count,packed_min=pack_min,packed_max=pack_max,
         exact_algebra_cases=checked,
         census=dict(symbol='full_set',trip_addresses=['0xe394','0xf010'],
                     counts=dict(counts),trip_packets=len(trip),rollover_packets=len(roll),
                     fulltile_epilogue_packets=len(epi),amortized_packets=packets,
                     amortized_service_units=service),
         rate_model=dict(alpha=alpha,baseline_tops=190,baseline_us=482.7,variants=rates),
         model_verdict=dict(t0_us=t0,shares=shares,quant_fraction_of_gemm=quant_fraction,
                            preamble_saving_us=preamble_save,fa_saving_us=fa_save,
                            hypothetical_scan_saving_us=scan_save,rows=verdict,
                            required_tops_all_gemm=required_all,
                            required_tops_compute_only=required_compute),
         evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [E/'tu-cand.dis.txt',E/'gemm_iu4_shipped_775ac0612.hip',
                           E/'RESULTS-vopd.md',E/'S0-gfx1201-profile.md']})
(E/'plan_checks.json').write_text(json.dumps(out,indent=2)+'\n')
(E/'steady_set_path.txt').write_text('\n'.join(f'{a:08x} {o}' for a,o in trip)+'\n')
(E/'epilogue_set_path.txt').write_text('\n'.join(f'{a:08x} {o}' for a,o in epi)+'\n')
print(json.dumps(out,indent=2))
