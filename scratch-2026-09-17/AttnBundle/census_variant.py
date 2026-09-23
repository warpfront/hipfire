import json
import re
import sys
from collections import Counter
from pathlib import Path

asm_path = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
text = asm_path.read_text()
name = "attention_fp8_e4m3_fa2_gqa_packet_gfx1201"
start = text.index(name + ":")
end = text.index(".Lfunc_end5:", start)
function = text[start:end]

# Same packet convention and interior-path block census as the accepted fapkt
# census.py. Candidate packet totals are intentionally not asserted.
blocks = {}
block = 0
for line in function.splitlines():
    match = re.match(r"(?:\.LBB5_|; %bb\.)(\d+)", line)
    if match:
        block = int(match.group(1))
    if re.match(r"^\s+(?:v_|s_|ds_|global_)", line):
        blocks.setdefault(block, []).append(line.strip())

fill = [10, 11, 12, 13, 15, 16, 17] + list(range(19, 38))
header_common = [37]
header_copy = [38, 39, 40]
precompute = [41, 42, 43, 45]
subtiles = []
for sub in range(4):
    subtiles += (
        [47]
        + ([48] if sub >= 1 else [])
        + ([49] if sub >= 2 else [])
        + [50, 51]
        + [52] * 4
        + [53, 55, 56, 46]
    )
tile_latch = [8, 9]
common_path = fill + header_common + precompute + subtiles + tile_latch
header_path = common_path + header_copy

common_ops = Counter(
    instruction.split()[0]
    for block_id in common_path
    for instruction in blocks[block_id]
)
header_ops = Counter(
    instruction.split()[0]
    for block_id in header_path
    for instruction in blocks[block_id]
)
common_packets = sum(common_ops.values())
header_packets = sum(header_ops.values())
wg_packets = 6 * common_packets + 2 * header_packets
fair_packets = wg_packets / 8

assert common_ops["v_wmma_f32_16x16x16_fp8_fp8"] == 128
assert common_ops["v_cvt_pk_fp8_f32"] == 16

result = {
    "assembly": str(asm_path),
    "kernel": name,
    "source_census": "/home/kaden/ClaudeCode/warpfront/wt-fapkt/scratch-2026-09-17/fapkt2/census.py",
    "assumptions": [
        "interior full KT64",
        "all query lanes valid",
        "every key <= gmin",
        "positive weighted P",
        "one emitted instruction including waits and branches is one packet",
        "one-time function prologue and completion epilogue excluded",
        "candidate totals are measured rather than asserted",
    ],
    "waves_without_header_copy": 6,
    "waves_with_header_copy": 2,
    "packets_without_header_copy": common_packets,
    "packets_with_header_copy": header_packets,
    "workgroup_packets_KT64": wg_packets,
    "fair_packets_per_wave_KT64": fair_packets,
    "common_path": common_path,
    "header_copy_path": header_copy,
    "common_ops": dict(sorted(common_ops.items())),
    "header_ops": dict(sorted(header_ops.items())),
}
out_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "packets_without_header_copy": common_packets,
    "packets_with_header_copy": header_packets,
    "workgroup_packets_KT64": wg_packets,
    "fair_packets_per_wave_KT64": fair_packets,
    "wmma_KT64": common_ops["v_wmma_f32_16x16x16_fp8_fp8"],
    "cvt_pk_fp8_KT64": common_ops["v_cvt_pk_fp8_f32"],
}, indent=2))
