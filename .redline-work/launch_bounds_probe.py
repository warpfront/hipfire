from pathlib import Path
import json
import subprocess

repo = Path('/home/kaden/hipfire-rw-nofm')
source = Path('/home/kaden/hipfire-chatcache-76f2a5748/.hipfire_kernels/gfx1151/gemv_hfq4g256_moe_gate_up_indexed.hip').read_text()
out_dir = repo / '.redline-work/gate-up-launch-bounds'
out_dir.mkdir(parents=True, exist_ok=True)
for blocks in (16, 15, 14, 13, 12, 10, 8):
    hip = out_dir / f'gate_up_lb{blocks}.hip'
    hsaco = out_dir / f'gate_up_lb{blocks}.hsaco'
    hip.write_text(f'#define HIPFIRE_MOE_GATE_UP_MIN_BLOCKS {blocks}\n' + source)
    subprocess.run([
        'hipcc', '--genco', '--offload-arch=gfx1151', '-O3',
        '-fno-exceptions', '-fno-rtti',
        '-include', str(repo / 'crates/radiowave/include/radiowave/hip.h'),
        '-DRADIOWAVE_ACTIVE=1', '-o', str(hsaco), str(hip),
    ], check=True)
    inspection = json.loads(subprocess.check_output([
        repo / 'target/release/radiowave', 'inspect',
        '--input', hsaco, '--arch', 'gfx1151',
    ], text=True))
    kernel = inspection['kernels'][0]
    instructions = kernel['instructions']
    print({
        'min_blocks': blocks,
        'private': kernel['private_segment_fixed_size'],
        'vgpr': kernel['vgpr_count'],
        'sgpr': kernel['sgpr_count'],
        'vgpr_spill': kernel['vgpr_spill_count'],
        'scratch_ops': instructions.get('scratch_loads', 0) + instructions.get('scratch_stores', 0),
        'flat_loads': instructions['flat_loads'],
        'buffer_loads': instructions['buffer_loads'],
        'instructions': instructions['static_instructions'],
        'waits': instructions['wait_instructions'],
    })
