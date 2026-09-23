import re, hashlib, sys

def kernel_text(path, name):
    txt = open(path).read()
    txt = re.sub(r'^.*__hip_cuid_[0-9a-f]+.*\n?', '', txt, flags=re.M)
    m = re.search(r'^' + re.escape(name) + r':.*?(?=^[A-Za-z_][\w]*:|\Z)',
                  txt, re.M | re.S)
    return m.group(0) if m else None

pairs = [
    ("base/kv.s", "kv.s", "attention_q8_0_kv"),
    ("base/kvDHIPFIRE_KV_BF16.s", "kvDHIPFIRE_KV_BF16.s", "attention_bf16_kv"),
    ("base/tile.s", "tile.s", "attention_flash_q8_0_tile"),
]
ok = True
for b, c, k in pairs:
    tb, tc = kernel_text(b, k), kernel_text(c, k)
    hb = hashlib.md5(tb.encode()).hexdigest() if tb else "MISSING"
    hc = hashlib.md5(tc.encode()).hexdigest() if tc else "MISSING"
    same = hb == hc
    ok = ok and same
    print(k + ": base=" + hb + " cur=" + hc +
          " " + ("IDENTICAL" if same else "DIFFER"))
print("ALL_IDENTICAL" if ok else "MISMATCH_FOUND")
