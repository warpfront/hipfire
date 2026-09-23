import csv, glob, sys

def load(pat):
    rows = {}
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            if r['Kind'] != 'KERNEL_DISPATCH':
                continue
            n = r['Kernel_Name']
            d = (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1000.0  # us
            e = rows.setdefault(n, {'n': 0, 'tot': 0.0, 'vgpr': r['VGPR_Count'],
                                    'lds': r['LDS_Block_Size']})
            e['n'] += 1
            e['tot'] += d
    return rows

old = load('prof_old2/hiptrx/*_kernel_trace.csv')
new = load('prof_new2/hiptrx/*_kernel_trace.csv')
names = sorted(set(old) | set(new))
print('%-42s %6s %12s %12s %10s %5s %7s' %
      ('kernel', 'n', 'old_tot_us', 'new_tot_us', 'delta_us', 'vgpr', 'lds'))
for n in names:
    if 'copy' in n or 'memset' in n.lower():
        continue
    o = old.get(n, {'n': 0, 'tot': 0.0})
    w = new.get(n, {'n': 0, 'tot': 0.0})
    v = w.get('vgpr', o.get('vgpr', '?'))
    l = w.get('lds', o.get('lds', '?'))
    print('%-42s %6d %12.1f %12.1f %+10.1f %5s %7s' %
          (n[:42], w['n'] or o['n'], o['tot'], w['tot'], w['tot'] - o['tot'], v, l))
