import collections, sys
tag = sys.argv[1]
d = collections.defaultdict(list)
for f in [f"{tag}-F.log", f"{tag}-R.log"]:
    for l in open(f):
        if l.startswith("RESULT"):
            kv = dict(x.split("=", 1) for x in l.split()[1:] if "=" in x)
            d[(kv["shape"], kv["arm"])].append((float(kv["ms"]), float(kv["tops"])))
shapes = ["set_m17408", "set_m10240", "set_m6144", "add_k17408", "add_k6144"]
print("| N8192 shape | X5 ms / TOPS / % ceil | V2C-ms native ms / TOPS / % ceil | V2C prod-layout ms / TOPS / % ceil | prod speedup vs X5 | native speedup vs prod |")
print("|---|---:|---:|---:|---:|---:|")
for s in shapes:
    row = [s]; v = {}
    for a in ["x5", "v2c_ms", "v2c_prod"]:
        ms = sum(x[0] for x in d[(s, a)]) / 2; t = sum(x[1] for x in d[(s, a)]) / 2; v[a] = ms
        row.append(f"{ms:.3f} / {t:.1f} / {100*t/253.296:.1f}%")
    row.append(f"{v['x5']/v['v2c_prod']:.3f}x"); row.append(f"{v['v2c_prod']/v['v2c_ms']:.3f}x")
    print("| " + " | ".join(row) + " |")
