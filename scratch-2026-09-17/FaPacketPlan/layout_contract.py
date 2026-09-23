import json
from pathlib import Path
R = Path(__file__).resolve().parent

def perm(a, b, selectors):
    both = a + b
    return [both[i] for i in selectors]

def transpose4(words):
    x = [perm(words[r], words[r ^ 1], [0,4,2,6] if r % 2 == 0 else [5,1,7,3]) for r in range(4)]
    return [perm(x[r], x[r ^ 2], [0,1,4,5] if r < 2 else [6,7,2,3]) for r in range(4)]

kseen, vseen = set(), set()
for sub in range(4):
    for p in range(8):
        for lane in range(32):
            ml, kg = lane % 16, lane // 16
            ko = ((p*4+sub)*32+lane)*16
            vo = ((sub*8+p)*32+lane)*16
            for half in range(2):
                dc = 2*p+half
                for j in range(8):
                    kseen.add((ko+half*8+j, sub*16+ml, dc*16+kg*8+j))
                    vseen.add((vo+half*8+j, sub*16+kg*8+j, dc*16+ml))
assert len(kseen) == len(vseen) == 16384
assert len({x[0] for x in kseen}) == len({x[0] for x in vseen}) == 16384
assert len({x[1:] for x in kseen}) == len({x[1:] for x in vseen}) == 16384
assert {x[0] for x in kseen} == set(range(16384))
assert {x[0] for x in vseen} == set(range(16384))
# Eight independent 2KiB scratch swizzles. Each wave owns eight complete keys.
for w in range(8):
    addresses = {((k*64+word)^((k&3)*8)) for k in range(8) for word in range(64)}
    assert addresses == set(range(512))
    for dc in range(16):
        for dq in range(4):
            for khalf in range(2):
                words = [[(w*8+khalf*4+r, dc*16+dq*4+j) for j in range(4)] for r in range(4)]
                result = transpose4(words)
                for rr in range(4):
                    assert result[rr] == [(w*8+khalf*4+j, dc*16+dq*4+rr) for j in range(4)]
# Dense all-consumer ownership, including heads straddling WG boundaries.
for batch in range(1,513):
    seen = set()
    positions = [(q*71)%113-1 for q in range(batch)]
    for bx in range((batch*6+127)//128):
        qlo, qhi = bx*128//6, min((bx*128+127)//6,batch-1)
        assert 0 <= qhi-qlo < 22
        owned_queries = set()
        for wave in range(8):
            for ml in range(16):
                row = bx*128+wave*16+ml
                if row >= 6*batch:
                    continue
                query, head = divmod(row,6)
                assert (query,head) not in seen
                seen.add((query,head))
                owned_queries.add(query)
        assert owned_queries == set(range(qlo,qhi+1))
        assert min(positions[q] for q in owned_queries) == min(positions[qlo:qhi+1])
        assert max(positions[q] for q in owned_queries) == max(positions[qlo:qhi+1])
    assert seen == {(q,h) for q in range(batch) for h in range(6)}
# Explicit boundaries, half-open and disjoint; scales are native f16 bytes.
regions = {'K':[0,16384], 'V':[16384,32768], 'scratch':[32768,49152], 'sk':[49152,49280], 'sv':[49280,49408]}
assert sum(b-a for a,b in regions.values()) == 49408
assert 2*128*256 == 65536
result = {'PASS':True,'regions':regions,'KT64_dynamic_LDS':49408,'KT128_payload_only':65536,'KT128_with_same_scratch_and_scales':82432,'paired_fragment_bytes':16,'byte_bijections':16384,'scratch_words_per_wave':512,'claim':'host label/address algebra only; no device bank/race/tail correctness or performance claim'}
result['dense_row_ownership_batches_checked'] = [1,512]
(R/'layout_contract.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
