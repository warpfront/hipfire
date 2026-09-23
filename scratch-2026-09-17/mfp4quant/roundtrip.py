#!/usr/bin/env python3
"""CPU roundtrip: parent BF16 -> FWHT -> MFP4 dequant, relative error."""
import json, struct, mmap, sys
import numpy as np

TNAME = "model.language_model.layers.0.mlp.gate_proj.weight"

# --- parent BF16 ---
P = "/home/kaden/qcal/parents/qwen3.8-27b/"
idx = json.load(open(P + "model.safetensors.index.json"))
entry = idx["weight_map"][TNAME]
# entry may be str (file) ; offsets from file header
fpath = P + entry if isinstance(entry, str) else P + entry["file"]
tf = open(fpath, "rb")
n = struct.unpack("<Q", tf.read(8))[0]
header = json.loads(tf.read(n))
info = header[TNAME]
shape = info["shape"]
dtype = info["dtype"]
assert dtype == "BF16", dtype
off0, off1 = info["data_offsets"]
data_start = 8 + n
tf.seek(data_start + off0)
raw = tf.read(off1 - off0)
w = np.frombuffer(raw, dtype=">u2").copy()  # safetensors BF16 is LE; use <u2
w = np.frombuffer(raw, dtype="<u2").astype("<u4")
# bf16 -> f32: shift
w32 = (w << 16).astype("<u4").view("<f4").astype(np.float64)
m, k = shape
w32 = w32.reshape(m, k).astype(np.float32)
print("parent", TNAME, shape, "absmax", float(np.abs(w32).max()))

# --- FWHT rotation (cpu_fwht_256, seeds 42/1042) ---
def gen_signs(seed, n):
    s = seed
    out = np.empty(n, np.float32)
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0x7fffffff
        out[i] = 1.0 if ((s >> 16) & 1) else -1.0
    return out
s1 = gen_signs(42, 256); s2 = gen_signs(1042, 256)
rot = w32.reshape(-1, 256) * s1
# butterfly
st = 1
while st < 256:
    r = rot.reshape(-1, 256).copy()
    for i in range(0, 256, 2*st):
        a = r[:, i:i+st]; b = r[:, i+st:i+2*st]
        rot.reshape(-1,256)[:, i:i+st] = a + b
        rot.reshape(-1,256)[:, i+st:i+2*st] = a - b
    st <<= 1
rot = (rot * (0.0625 * s2)).astype(np.float32).reshape(m, k)
print("rotated absmax", float(np.abs(rot).max()))

# --- hfq MFP4 dequant ---
LUT = np.array([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], np.float32)
def f16bits(h):
    return struct.unpack("<e", struct.pack("<H", int(h)))[0]
A = sys.argv[1] if len(sys.argv) > 1 else "/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mfp4g32.hfq"
f = open(A, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
nt = struct.unpack_from("<I", mm, 12)[0]
data_off = struct.unpack_from("<Q", mm, 24)[0]
rawh = bytes(mm[32:data_off])
depth=0;ins=False;esc=False;end=None
for i,ch in enumerate(rawh):
    c=chr(ch)
    if ins:
        if esc: esc=False
        elif c=="\\": esc=True
        elif c=='"': ins=False
    else:
        if c=='"': ins=True
        elif c in "{[": depth+=1
        elif c in "}]":
            depth-=1
            if depth == 0:
                end = i + 1
                break
p=32+end; assert struct.unpack_from("<I",mm,p)[0]==nt; p+=4
off=None; dl=0
for _ in range(nt):
    nl=struct.unpack_from("<H",mm,p)[0]; p+=2
    name=bytes(mm[p:p+nl]).decode(); p+=nl
    qt=mm[p]; p+=1; nd=mm[p]; p+=1
    shp=struct.unpack_from("<"+"I"*nd,mm,p); p+=4*nd
    gs=struct.unpack_from("<I",mm,p)[0]; p+=4
    ln=struct.unpack_from("<Q",mm,p)[0]; p+=8
    if name==TNAME:
        assert qt==24, qt
        off=data_off + 0  # placeholder
        # data blobs are concatenated in tensor order; track cumulative
        dl=ln; mshape=shp
    # cumulative offset tracking
    # (recompute below)
    pass
# second pass for offset
p2=p  # p now at end of index; recompute: iterate again
# simpler: re-walk
p=32+end+4; cu=0; target=None
for _ in range(nt):
    nl=struct.unpack_from("<H",mm,p)[0]; p+=2
    name=bytes(mm[p:p+nl]).decode(); p+=nl
    qt=mm[p]; p+=1; nd=mm[p]; p+=1
    shp=struct.unpack_from("<"+"I"*nd,mm,p); p+=4*nd
    gs=struct.unpack_from("<I",mm,p)[0]; p+=4
    ln=struct.unpack_from("<Q",mm,p)[0]; p+=8
    if name==TNAME:
        target=(data_off+cu, shp, ln); break
    cu+=ln
assert target, "tensor not found"
toff, tshp, tln = target
tm, tk = tshp
nb = tk//32; rb = 16+17*nb
assert tln == tm*rb
dq = np.empty((tm, tk), np.float32)
for r in range(tm):
    base = toff + r*rb
    rsa = f16bits(struct.unpack_from("<H", mm, base)[0])
    for b in range(nb):
        bb = base+16+b*17
        e = mm[bb]
        sf = rsa * (2.0 ** (e-127))
        for i in range(16):
            by = mm[bb+1+i]
            dq[r, b*32+2*i] = sf * LUT[by & 0xF]
            dq[r, b*32+2*i+1] = sf * LUT[(by>>4) & 0xF]
    if r % 4096 == 0:
        print(f"row {r}/{tm}", flush=True)
num = float(np.abs(dq.astype(np.float64)-rot.astype(np.float64)).max())
den = float(np.abs(rot).max())
rel = np.linalg.norm((dq-rot).ravel())/np.linalg.norm(rot.ravel())
print(f"maxabs_err={num:.6f} refmax={den:.6f} relL2={rel:.6f}")
