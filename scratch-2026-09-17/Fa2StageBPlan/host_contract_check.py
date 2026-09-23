import bisect
import hashlib
import json
import math
import random
import struct
from pathlib import Path

ROOT = Path('/home/kaden/ClaudeCode/warpfront/wt-fa2a')
OUT = ROOT / 'scratch-2026-09-17/Fa2StageBPlan'

def half(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]

def scale16(values):
    a = max(map(abs, values))
    if a == 0:
        return 1.0
    s = half(a / 448)
    if s == 0:
        s = 2.0 ** -24
    if 448*s < a:
        bits = struct.unpack('<H', struct.pack('<e', s))[0]
        s = struct.unpack('<e', struct.pack('<H', bits+1))[0]
    assert 448*s >= a
    return s

def decode(c):
    sign = -1 if c & 128 else 1
    e, m = (c >> 3) & 15, c & 7
    if e == 15 and m == 7:
        return math.nan
    return sign * (m*2.0**-9 if e == 0 else (1+m/8)*2.0**(e-7))

POS = [decode(c) for c in range(127)]
def encode(x):
    neg = math.copysign(1, x) < 0
    x = abs(x)
    hi = bisect.bisect_left(POS, x)
    if hi >= len(POS):
        code = 126
    elif hi == 0:
        code = 0
    else:
        lo = hi-1
        code = min((lo, hi), key=lambda c: (abs(POS[c]-x), c & 1))
    return code | (128 if neg else 0)

for c in range(256):
    if math.isfinite(decode(c)):
        assert encode(decode(c)) == c

layout = []
for width in (1, 2):
    for side in ('K', 'V'):
        offsets = []
        for key in range(64):
            for dim in range(256):
                sub, dc = key >> 4, dim >> 4
                if side == 'K':
                    lane, j = 16*((dim >> 3)&1)+(key&15), dim&7
                    frag = (dc*4+sub)*32+lane
                else:
                    lane, j = 16*((key >> 3)&1)+(dim&15), key&7
                    frag = (sub*16+dc)*32+lane
                offsets.append((frag*8+j)*width)
        assert sorted(offsets) == list(range(0, 16384*width, width))
        assert all((frag*8*width) % (8*width) == 0 for frag in range(2048))
        layout.append({'side':side, 'element_bytes':width, 'elements':len(offsets), 'unique':len(set(offsets))})

scratch = []
for batch in (1,7,8,9,512):
    nrows = batch*24
    qbytes = nrows*256
    base = qbytes+nrows*4
    assert base % 16 == 0
    for splits in (1,8):
        slots = ((batch+7)//8)*4*splits
        addresses = []
        for slot in range(slots):
            for side in range(2):
                for key in range(64):
                    addresses.append(base+slot*256+side*128+2*key)
        assert len(set(addresses)) == len(addresses)
        assert min(addresses) >= base
        assert max(addresses)+2 == base+slots*256
        scratch.append({'batch':batch,'splits':splits,'q_and_sq_bytes':base,'total_bytes':base+slots*256})

rng = random.Random(20260918)
maxrel = 0.0
for trial in range(64):
    op = [0.0]*8
    ou = [0.0]*8
    bprev = 1.0
    m = -math.inf
    l = 0.0
    for sub in range(17):
        valid = (sub+trial)%5 != 0
        scores = [rng.uniform(-30,30) for _ in range(16)] if valid else [-math.inf]*16
        mn = max(m,max(scores))
        alpha = 0.0 if m == -math.inf else math.exp(m-mn)
        e = [0.0 if mn == -math.inf else math.exp(x-mn) for x in scores]
        sv = [half(2.0**rng.randint(-24,14)) for _ in range(16)]
        w = [a*b for a,b in zip(e,sv)]
        wm = max(w)
        bn = max(wm/448,2.0**-64) if wm else bprev
        pp = [decode(encode(x/bn)) for x in w]
        vv = [[decode(rng.randrange(127))*(-1 if rng.randrange(2) else 1) for _ in range(8)] for _ in range(16)]
        delta = [sum(pp[k]*vv[k][d] for k in range(16)) for d in range(8)]
        rho = alpha*bprev/bn
        op = [alpha*x+bn*y for x,y in zip(op,delta)]
        ou = [rho*x+y for x,y in zip(ou,delta)]
        maxrel = max(maxrel, max(abs(x-bn*y)/max(1,abs(x)) for x,y in zip(op,ou)))
        m,l,bprev = mn,alpha*l+sum(e),bn
    assert maxrel < 1e-10

q8_scales = [half(2.0**rng.randint(-20,8)) for _ in range(8)]
q8_values = [q8_scales[d//32]*rng.randint(-128,127) for d in range(256)]
s = scale16(q8_values)
assert all(math.isfinite(decode(encode(v/s))) for v in q8_values)
assert scale16([0.0]*256) == 1.0

qk_old = {'wmma':64,'lds':160,'address':72,'move':69,'wait':91,'delay':39,'control':113,'q_global':16}
qk_cap = dict(qk_old, lds=64, address=16)
new_per_sub = {'scale_loads':16,'scale_widen':16,'scale_address':8,'scale_wait':8,'score_extra_mul':8,'weighted_p_mul':8,'weighted_p_max_shuffle_wait':10,'b_floor_inverse':8,'p_normalize_pack_net':4,'rho':3,'misc':11}
assert sum(qk_old.values()) == 624
assert sum(qk_cap.values()) == 472
assert sum(new_per_sub.values()) == 100
assert 472+4+4*(383-48+100) == 2216

receipt = {
    'scope':'Host-only layout/algebra/budget proof; no GPU, compiler, candidate implementation or performance result.',
    'kernel_sha256':hashlib.sha256((ROOT/'kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip').read_bytes()).hexdigest(),
    'finite_codec_roundtrips':254,
    'layouts':layout,
    'scratch_cases':scratch,
    'changing_unit_trials':64,
    'changing_unit_subtiles_per_trial':17,
    'max_f64_relative_identity_error':maxrel,
    'qk_reference':qk_old,
    'qk_unmeasured_cap':qk_cap,
    'new_support_unmeasured_cap_per_subtile':new_per_sub,
    'compute_unmeasured_cap_per_wave_kt64':2216,
    'decode_required_speedup':36.4/32.7-1,
    'decode_required_time_reduction':1-32.7/36.4,
    'decode_required_ms_saved':1000/32.7-1000/36.4,
    'ag_reported_delta':0.148392-0.141396,
    'max_fp32_ou_bound':32768*65504*448*2.0**64,
    'p_floor_rounding_bound':32768*448*16*2.0**-64,
}
(OUT/'host_contract_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
