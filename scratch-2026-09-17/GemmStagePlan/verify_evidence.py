from pathlib import Path
import hashlib
import json
import re

ROOT = Path('/home/kaden/ClaudeCode/warpfront/wt-gemmstage/scratch-2026-09-17/GemmStagePlan')
SOURCE = Path('/home/kaden/ClaudeCode/warpfront/wt-gemmstage/kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip')
identity = json.loads((ROOT / 'shipped_identity.json').read_text())
assert identity['source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert all(x['compile_exit'] == 0 and x['device_body_identical'] for x in identity['families'])
census = json.loads((ROOT / 'isa_census.json').read_text())
for family in ('gu', 're', 'qkv', 'qkvza'):
    assert census[family + '_full']['wmma'] == 64
    assert census[family + '_full']['scratch_ops'] == 0
for family in ('gu', 're'):
    assert census[family + '_nofold']['wmma'] == 64
for stem, expect_wait in (('gu_full', True), ('gu_rawpref', False)):
    asm = (ROOT / (stem + '.s')).read_text()
    first = re.search(r'global_load_b128[^\n]*offset:64', asm).start()
    wmma = asm.index('v_wmma_', first)
    assert ('s_wait_loadcnt' in asm[first:wmma]) == expect_wait
rr = (ROOT / 'register_probe.s').read_text()
for name, n_fma in (('pure', 0), ('fold', 128)):
    body = rr.split(name + ':', 1)[1].split('.Lfunc_end', 1)[0]
    assert body.count('v_wmma_') == 64
    assert body.count('sched_barrier') == 8
    assert len(re.findall(r'v_(?:dual_)?fma[c]?_f32', body)) == n_fma
for n in (512, 8192):
    data = (ROOT / f'rawpref{n}.txt').read_text()
    for family in ('gu', 're'):
        hashes = re.findall(rf'{family}_(?:full|rawpref) .*hash=(\w+)', data)
        assert len(hashes) >= 6 and len(set(hashes)) == 1
print('PASS: shipped-source identity; WMMA retained in ablations; raw-prefetch wait removed; pinned register loop packet counts; uniform-input exact hashes for gate_up/residual at N512/N8192.')
print('NOT CLAIMED: four canonical oracle pins, WT2 KLD, cross-half depth2/depth3 speed, full-model candidate speed.')
