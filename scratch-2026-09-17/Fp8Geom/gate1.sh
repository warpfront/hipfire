#!/bin/sh
# Fp8Geom gate 1: full oracle (no --quick) on 128x64w8, V2 route, ordinal 2.
# Expect: 36/36 pass AND canonical N=512 hashes
#   gate_up 2adf7620c1d722e1 / qkv 09726b10ba1eb1b2 /
#   qkvza 082a7b24def26f75 / residual 373cb087fb519b9e
# Announce on hub before running; never overlap timing on ordinal 2.
set -eu
WT=/home/kaden/ClaudeCode/warpfront/wt-fp8geom
R=$WT/scratch-2026-09-17/Fp8Geom/gate1-128x64w8
export HOME=/home/kaden/.hipfire-homes/ab2
export ROCR_VISIBLE_DEVICES=2
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab2/.hipfire_kernels
export HIPFIRE_GFX12_MQ4V2_FP8_V2=1
export HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM=128x64w8
export HIPFIRE_GFX12_MQ4V2_FP8_SLABS=2
BIN=$WT/target/release/examples/tmp_gemm_v2_oracle
mkdir -p "$R"
"$BIN" --device 0 --n 512 --out "$R" > "$R.stdout" 2> "$R.stderr"; echo "RC=$?"
echo "--- canonical N=512 hashes (results.json gpu_cases) ---"
python3 - "$R" <<'EOF'
import json,sys,glob
d=json.load(open(glob.glob(sys.argv[1]+'/results.json')[0]))
print('v2_geometry:', d.get('v2_geometry'))
for c in d.get('gpu_cases',[]):
    if c.get('n')==512 and c.get('k',0)>512:
        print(c.get('family'), 'K=', c.get('k'), 'hash=', c.get('hash'),
              'repeat=', c.get('hash_repeat'), c.get('verdict'))
EOF
