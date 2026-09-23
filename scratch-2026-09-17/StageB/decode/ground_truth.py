import struct

def rd(path):
    b = open(path, 'rb').read()
    return struct.unpack('<%df' % (len(b) // 4), b)

def sw(c):
    s = (c & 0x80) << 24
    e = (c >> 3) & 0x0f
    m = c & 0x07
    if e == 0:
        f = m * 1.953125e-3
        return -f if s else f
    if e == 15:
        if m == 7:
            return float('nan')
        f = 256.0 * (1.0 + m * 0.125)
        return -f if s else f
    import struct as S
    return S.unpack('<f', S.pack('<I', s | ((e - 7 + 127) << 23) | (m << 20)))[0]

rs, cl = 516, 512
kc = open('new_ksrand_300.inputs_kc.bin', 'rb').read()
q = rd('new_ksrand_300.inputs_q.bin')
# head 0 -> kv_h 0; token 0
ks = struct.unpack('<e', kc[0 * rs + cl:0 * rs + cl + 2])[0]
codes = kc[0 * rs:0 * rs + 256]
old_dot = 0.0
new_dot = 0.0
for j in range(256):
    d = sw(codes[j])
    # emulate f32 arithmetic with numpy if available, else note double
    old_dot += q[j] * (ks * d)
    new_dot += q[j] * d
new_dot *= ks
print('ks =', repr(float(ks)))
print('f64-model old dot =', repr(old_dot))
print('f64-model new dot =', repr(new_dot))
o = rd('old_ksrand_300.scalar_out.bin')
n = rd('new_ksrand_300.scalar_out.bin')
print('old out[0:4] =', [repr(x) for x in o[0:4]])
print('new out[0:4] =', [repr(x) for x in n[0:4]])
