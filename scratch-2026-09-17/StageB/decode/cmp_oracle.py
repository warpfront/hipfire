import struct, sys

def read_f32(path):
    b = open(path, 'rb').read()
    n = len(b) // 4
    return struct.unpack('<%df' % n, b[:n * 4])

def stats(a, b):
    assert len(a) == len(b), (len(a), len(b))
    n = len(a)
    nan_a = sum(1 for x in a if x != x)
    nan_b = sum(1 for x in b if x != x)
    maxabs = 0.0
    mean = 0.0
    cnt = 0
    diffs = []
    for x, y in zip(a, b):
        if x != x or y != y:
            continue
        d = abs(x - y)
        diffs.append(d)
        if d > maxabs:
            maxabs = d
        mean += d
        cnt += 1
    mean = mean / cnt if cnt else 0.0
    diffs.sort()
    tail = 0.0
    if diffs:
        k = max(1, len(diffs) // 100)
        tail = sum(diffs[-k:]) / k
    exact = all((x != x and y != y) or struct.pack('<f', x) == struct.pack('<f', y)
                for x, y in zip(a, b))
    return dict(n=n, nan_a=nan_a, nan_b=nan_b, maxabs=maxabs, mean=mean,
                tail1pct=tail, exact=exact)

mode = sys.argv[1] if len(sys.argv) > 1 else 'ks1'
seq = sys.argv[2] if len(sys.argv) > 2 else ''
tag = mode + ('_' + seq if seq else '')
op = 'old_' + tag
np = 'new_' + tag
for kern, suf in (('scalar', 'scalar_out.bin'), ('tile', 'tile_partials.bin')):
    a = read_f32('%s.%s' % (op, suf))
    b = read_f32('%s.%s' % (np, suf))
    s = stats(a, b)
    print('%s.%s n=%d nan_old=%d nan_new=%d maxabs=%.3e mean=%.3e tail1pct=%.3e exact=%s'
          % (tag, kern, s['n'], s['nan_a'], s['nan_b'], s['maxabs'], s['mean'],
             s['tail1pct'], s['exact']))
