import struct, sys

def rd(path):
    return open(path, 'rb').read()

# row_stride = 2*(256+2) = 516; codes_len = 512
rs, cl = 516, 512
for tag in ['old_ksrand_300', 'new_ksrand_300']:
    kc = rd(tag + '.inputs_kc.bin')
    print(tag, 'len', len(kc))
    for t in [0, 1, 2]:
        for h in [0, 1]:
            s = struct.unpack('<e', kc[t * rs + cl + h * 2:t * rs + cl + h * 2 + 2])[0]
            print('  t=%d h=%d scale=%r' % (t, h, s))
    break
a = rd('old_ksrand_300.inputs_kc.bin')
b = rd('new_ksrand_300.inputs_kc.bin')
print('inputs identical:', a == b)
qa = rd('old_ksrand_300.inputs_q.bin')
qb = rd('new_ksrand_300.inputs_q.bin')
print('q identical:', qa == qb)
