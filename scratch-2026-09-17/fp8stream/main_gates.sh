#!/bin/bash
export HOME=/home/kaden/.hipfire-homes/ab2 ROCR_VISIBLE_DEVICES=2 HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab2/.hipfire_kernels HIPFIRE_GRAPH=1 HIPFIRE_LLOYD_GFX12=1
E=/home/kaden/ClaudeCode/warpfront/wt-fp8stream/scratch-2026-09-17/fp8stream/main
mkdir -p $E; cd /home/kaden/ClaudeCode/warpfront/wt-fp8stream
md5sum target/release/hipfire target/release/daemon > $E/identity.txt
summ() { python3 - "$1" <<'PY'
import json,sys
try: j=json.load(open(sys.argv[1]))
except Exception: print(sys.argv[1].split('/')[-1],"FAILED"); sys.exit()
a=[]
def walk(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k=='median': a.append(v)
            walk(v)
    elif isinstance(o,list):
        for x in o: walk(x)
walk(j); print(sys.argv[1].split('/')[-1], [round(x,1) for x in a[:5]])
PY
}
for arm in on off on; do
  if [ $arm = on ]; then export HIPFIRE_GFX12_FP8_STREAM=1; else unset HIPFIRE_GFX12_FP8_STREAM; fi
  f=$E/bench-$arm-$RANDOM.json
  timeout 1500 ./target/release/hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json > $f 2> $f.err
  summ $f
done
export HIPFIRE_GFX12_FP8_STREAM=1
timeout 900 ./target/release/hipfire bench qwen3.8:27b-mq4-xt --spec off --backend noslots --workload stateless --runs 5 --warmups 3 --max-tokens 128 --json > $E/decode-on.json 2> $E/decode-on.err
python3 -c "import json;j=json.load(open('$E/decode-on.json'));print('decode-on', j.get('tok_s_median') or [ (k,v) for k,v in j.items() if 'median' in str(k)][:3])" 2>/dev/null || grep -h "tok/s" $E/decode-on.err | tail -1
timeout 1200 python3 scripts/serve_harness.py --mode battery --model qwen3.8:27b-mq4-xt --thinking off --out $E/battery.json > $E/battery.log 2>&1; tail -3 $E/battery.log
