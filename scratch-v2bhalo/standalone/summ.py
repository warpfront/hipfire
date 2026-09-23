"""Halo standalone table: mean of the F and R process medians per (shape, arm).
usage: summ.py <logdir> <tag> [arm ...]"""
import collections
import sys

logdir, tag = sys.argv[1], sys.argv[2]
arms = sys.argv[3:] or ["x5", "v2b_ms", "v2b_prod"]
ms = collections.defaultdict(list)
dims = {}
for f in [f"{logdir}/{tag}-F.log", f"{logdir}/{tag}-R.log"]:
    for line in open(f):
        if line.startswith("RESULT"):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            ms[(kv["shape"], kv["arm"])].append(float(kv["ms"]))
            dims[kv["shape"]] = (int(kv["M"]), int(kv["K"]), int(kv["N"]))
print("| shape | M | K | N | v2b CTAs | " + " | ".join(f"{a} ms (% of 105.543)" for a in arms)
      + " | X5 / v2b_prod | v2b_prod / v2b_ms |")
print("|---|---:|---:|---:|---:|" + "---:|" * len(arms) + "---:|---:|")
for s in sorted(dims, key=lambda s: (-dims[s][2], dims[s][1] != 5120, -dims[s][0], dims[s][1])):
    M, K, N = dims[s]
    mean = {a: sum(ms[(s, a)]) / len(ms[(s, a)]) for a in arms}
    cells = [f"{mean[a]:.3f} ({100 * 2 * M * K * N / (mean[a] * 1e9) / 105.543:.1f}%)"
             for a in arms]
    print(f"| {s} | {M} | {K} | {N} | {(M // 256) * (N // 256)} | " + " | ".join(cells)
          + f" | {mean['x5'] / mean['v2b_prod']:.3f}x | {mean['v2b_prod'] / mean['v2b_ms']:.3f}x |")
