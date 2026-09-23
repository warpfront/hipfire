import csv, glob, statistics

def load(pat):
    per = {}
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            if r['Kind'] != 'KERNEL_DISPATCH':
                continue
            n = r['Kernel_Name']
            if 'copy' in n or 'memset' in n.lower() or 'fill' in n.lower():
                continue
            d = (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1000.0
            per.setdefault(n, []).append(d)
    return per

new = load('prof_n_*/hiptrx/*_kernel_trace.csv')
old = load('prof_o_*/hiptrx/*_kernel_trace.csv')
print('%-34s %28s %28s %12s' % ('kernel', 'old_us med(min-max),n', 'new_us med(min-max),n', 'delta_med'))
for n in sorted(set(old) | set(new)):
    o = sorted(old.get(n, []))
    w = sorted(new.get(n, []))
    mo = statistics.median(o) if o else 0.0
    mw = statistics.median(w) if w else 0.0
    of = '%.1f(%.1f-%.1f)x%d' % (mo, o[0], o[-1], len(o)) if o else '-'
    wf = '%.1f(%.1f-%.1f)x%d' % (mw, w[0], w[-1], len(w)) if w else '-'
    print('%-34s %28s %28s %+11.1f' % (n[:34], of, wf, mw - mo))
