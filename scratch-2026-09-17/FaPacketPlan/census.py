import json, re, hashlib
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parent
text=(R/'baseline.s').read_text().split('.Lfunc_end0:')[0]
blocks={}; block=0
for line in text.splitlines():
    m=re.match(r'(?:\.LBB0_|; %bb\.)(\d+)',line)
    if m: block=int(m[1])
    if re.match(r'^\s+(?:v_|s_|ds_|global_)',line):blocks.setdefault(block,[]).append(line.strip())
# Full KT64, all query lanes valid, all keys <= gmin, positive weighted P.
path=[13,14]+[16,17,15]*8+[18]+[22,23,24,21]*16+[38,39,40,41]
for sub in range(4):
    path += [44]+([45] if sub>=1 else [])+([46] if sub>=2 else [])+[47,48]+[49]*4+[50]+list(range(52,79))+[85,86,81,82,83,42,43]
path += [10,11,12]
# In block86 the execnz branch to81 is taken: its final unconditional jump
# is not executed. Other included blocks end at their selected branch.
assert blocks[86][-1]=='s_branch .LBB0_82'
blocks[86]=blocks[86][:-1]
ops=Counter(x.split()[0] for b in path for x in blocks[b])
helper=[13,14]+[16,17,15]*8+[18]+[22,23,24,21]*16+[38,11,12]
helper_packets=sum(len(blocks[b]) for b in helper)
assert sum(ops.values())==4138
assert ops['v_wmma_f32_16x16x16_fp8_fp8']==128
assert ops['v_cvt_pk_fp8_f32']==16
floor=(R/'floor-hip-amdgcn-amd-amdhsa-gfx1201.s').read_text()
floors={}
parts=re.split(r'^(_Z12floor_kernelILi\dEEvPKjPfi):.*$',floor,flags=re.M)
for idx in range(1,len(parts),2):
    mode=int(re.search(r'ILi(\d)',parts[idx])[1])
    if mode>2:continue
    bs={};b=0
    for line in parts[idx+1].split('.Lfunc_end')[0].splitlines():
        m=re.match(r'(?:\.LBB\d+_|; %bb\.)(\d+)',line)
        if m:b=int(m[1])
        if re.match(r'^\s+(?:v_|s_|ds_|global_)',line):bs.setdefault(b,[]).append(line.strip())
    op=Counter(x.split()[0] for b in [3,4,2] for x in bs[b])
    assert op['v_wmma_f32_16x16x16_fp8_fp8']==32
    assert op['v_cvt_pk_fp8_f32']==4
    assert not any('global_' in x or 'ds_load' in x or 'ds_store' in x for x in op)
    floors[mode]={'KT16_packets':sum(op.values()),'KT64_packets':4*sum(op.values()),'ops':dict(op)}
assert [floors[m]['KT16_packets'] for m in range(3)]==[399,396,255]
result={'compute_wave_KT64':sum(ops.values()),'helper_wave_KT64':helper_packets,'WG_KT64':3*sum(ops.values())+helper_packets,'amortized_per_consumer':sum(ops.values())+helper_packets/3,'path':path,'ops':dict(ops),'floors':floors}
(R/'census_verified.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['ops','path','floors']},indent=2))
print('math floor packets/KT16:', {k:v['KT16_packets'] for k,v in floors.items()})
