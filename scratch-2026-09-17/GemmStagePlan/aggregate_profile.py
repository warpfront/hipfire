from pathlib import Path
import csv
import json

root = Path('/home/kaden/ClaudeCode/warpfront/wt-gemmstage/scratch-2026-09-17/GemmStagePlan/profile-default')
rows = sorted(csv.DictReader((root / 'default_kernel_trace.csv').open()), key=lambda row: int(row['Dispatch_Id']))
embeddings = [i for i, row in enumerate(rows) if row['Kernel_Name'] == 'embedding_q8_batched']
assert len(embeddings) == 4, 'Expected two chunks each in one warmup and one final pass'
final = rows[embeddings[2]:]
aggregates = {}
for row in final:
    item = aggregates.setdefault(row['Kernel_Name'], {'calls': 0, 'total_us': 0.0, 'resources': set()})
    item['calls'] += 1
    item['total_us'] += (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1000
    item['resources'].add(tuple(row[key] for key in ('LDS_Block_Size', 'Scratch_Size', 'VGPR_Count', 'Workgroup_Size_X')))
for item in aggregates.values():
    item['us_per_token'] = item['total_us'] / 8192
    item['resources'] = sorted(item['resources'])
expected = {
    'gemm_gate_up_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201': 128,
    'gemm_mq4g256v2_residual_wmma_fp8_v2_b128x128_gfx1201': 256,
    'gemm_qkvza_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201': 96,
    'gemm_qkv_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201': 32,
}
for name, calls in expected.items():
    assert aggregates[name]['calls'] == calls
assert not any('s2bt8' in name for name in aggregates)
result = {
    'method': 'Numeric Dispatch_Id order; third of four embeddings starts final PP8192 pass; chunk4096. Warmup excluded.',
    'first_dispatch_id': final[0]['Dispatch_Id'],
    'rows': len(final),
    'gemm_us_per_token': sum(aggregates[name]['us_per_token'] for name in expected),
    'pack_us_per_token': aggregates['pack_f32_to_fp8_mq4v2_gfx12']['us_per_token'],
    'kernels': aggregates,
}
(root / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('first_dispatch_id', 'rows', 'gemm_us_per_token', 'pack_us_per_token')}, indent=2))
