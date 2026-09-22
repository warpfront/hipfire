#!/usr/bin/env python3
"""Uniform gate-3 trace digest: symbol/call counts, grids (work-items + workgroups),
block dims, LDS column, VGPR/SGPR, and attention-family time totals."""
import csv, collections, sys

d = sys.argv[1]
rows = list(csv.DictReader(open(f'{d}/daemon_kernel_trace.csv')))
print(f'dispatches: {len(rows)}')
buckets = collections.Counter()
fam_ns = collections.Counter()
for r in rows:
    k = r.get('Kernel_Name', '')
    if 'ttention' in k or 'fa2' in k.lower() or 'flash' in k.lower():
        wi = (r['Grid_Size_X'], r['Grid_Size_Y'], r['Grid_Size_Z'])
        b = r['Workgroup_Size_X']
        try:
            wg = (int(wi[0]) // int(b), wi[1], wi[2])
        except Exception:
            wg = ('?', wi[1], wi[2])
        key = (k, f"WI[{wi[0]},{wi[1]},{wi[2]}]", f"WG[{wg[0]},{wg[1]},{wg[2]}]",
               f"block={b}", f"lds={r['LDS_Block_Size']}",
               f"vgpr={r['VGPR_Count']}", f"sgpr={r['SGPR_Count']}")
        buckets[key] += 1
        fam_ns[k] += int(r['End_Timestamp']) - int(r['Start_Timestamp'])
for key in sorted(buckets, key=lambda k: (-buckets[k], k[0])):
    print(buckets[key], ' '.join(key))
print('--- family totals ms ---')
for k, ns in sorted(fam_ns.items(), key=lambda kv: -kv[1]):
    print(f'{k}: {ns/1e6:.2f} ms')
