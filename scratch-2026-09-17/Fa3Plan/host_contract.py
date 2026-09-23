import json
from pathlib import Path
root = Path(__file__).resolve().parent
for plane in ['K', 'V']:
    offsets = set()
    for sub in range(2):
        for dc in range(16):
            for lane in range(32):
                for j in range(8):
                    off = (((dc*2+sub) if plane == 'K' else (sub*16+dc))*32+lane)*8+j
                    assert off not in offsets
                    offsets.add(off)
    assert offsets == set(range(8192))
assert {((k*64+w)^((k&3)*8)) for k in range(16) for w in range(64)} == set(range(1024))
for sub in range(2):
    for dc in range(16):
        for kg in range(2):
            for dq in range(4):
                output = []
                for half in range(2):
                    rows = [[(sub*16+kg*8+half*4+r,dc*16+4*dq+j) for j in range(4)] for r in range(4)]
                    first = []
                    for r in range(4):
                        pair = rows[r]+rows[r^1]
                        first.append([pair[i] for i in ([0,4,2,6] if r%2 == 0 else [5,1,7,3])])
                    second = []
                    for r in range(4):
                        pair = first[r]+first[r^2]
                        second.append([pair[i] for i in ([0,1,4,5] if r&2 == 0 else [6,7,2,3])])
                    output.append(second)
                for r in range(4):
                    assert output[0][r]+output[1][r] == [(sub*16+kg*8+j,dc*16+4*dq+r) for j in range(8)]
old = 3700+840/3
consumer = 3700-840-96+96
producer = 220+800+240
new = consumer+producer/3
forecast = []
for c in [1,4,8]:
    ratio = .25+.75*(new+128*(c-1))/(old+128*(c-1))
    forecast.append(dict(c=c,ratio=ratio,pp8192_us=76*ratio,pp32768_us=160*ratio))
r = dict(plane_bijection=True,scratch_bijection=True,transpose_labels_exact=True,lds_bytes=41216,old_amortized=old,consumer_packets=consumer,producer_packets=producer,new_amortized=new,forecasts=forecast)
(root/'host_contract.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
