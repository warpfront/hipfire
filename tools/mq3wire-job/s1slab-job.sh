#!/bin/bash
# S1 over the mixed MQ4+W3 slab layers of mq3-xts: card gate. Run on hipx inside fabric-run.sh (one job per MODE, <=25 min each).
#   MODE=state : qwen4_parity --mode slowlink-state, arm S (S1 forced on) vs arm I (S1 off, 768-row incumbent chunk), rows 3400 and 16384.
#                Same-route determinism (3 repeats inside one report) is the S1 acceptance; S==I equality of every state/logit digest is the
#                (stronger, not required) byte-identity bonus: both routes run the same W3/MQ4 K2 on 768-row tiles.
#   MODE=pp    : hipfire bench --matrix --pp ${PP:-1105,8192,16384}, arms S and I.
# Optional I_VL=N pins arm I to N resident layers (compare against the S1 profile's own resident count in arm S).
# Needs BIN=<dir with hipfire daemon qwen4_parity>. mq3 model + sidecars come from the lossyeval timing home (hardlinks, read-only).
H=${HOMEDIR:-/home/kaden/qcal/release-0.4.5/mgmtp/singlecard/lossyeval/home-timing-mq3}; XN=/home/kaden/qcal/release-0.4.5/mgmtp/xtxnow
B=${BIN:?BIN=dir}; O=${OUT:-/home/kaden/qcal/release-0.4.5/mgmtp/singlecard/mq3wire/s1slab/out-$(date -u +%H%M%S)}; D=/sys/bus/pci/devices/0000:67:00.0
MODE=${MODE:?MODE=state|pp}; MODEL_NAME=qwen3.8-flash-next-base.mq3-xts; MODEL=$H/.hipfire/models/$MODEL_NAME
[ -e $MODEL ] || { echo "missing $MODEL" >&2; exit 2; }
mkdir -p $O
export HSA_USERPTR_FOR_PAGED_MEM=0 GPU_PINNED_MIN_XFER_SIZE=100000
export HIPFIRE_HOME=$H/.hipfire HOME=$H HIPFIRE_DAEMON_BIN=$B/daemon HIPFIRE_REGISTRY_URL=http://127.0.0.1:1/v1.json HIPFIRE_NO_REGISTRY_FETCH=1
for v in $(compgen -e); do case $v in HIPFIRE_DEVICES|HIPFIRE_LOCK_DIR|HIPFIRE_FABRIC_ADMISSION|HIPFIRE_HOME|HIPFIRE_DAEMON_BIN|HIPFIRE_REGISTRY_URL|HIPFIRE_NO_REGISTRY_FETCH|FABRIC_JOB) ;; HIPFIRE_*|GPU_MAX_HW_QUEUES) unset $v;; esac; done
export HIPFIRE_JINJA_CHAT=0 HIPFIRE_KV_MODE=q8 HIPFIRE_STATE_QUANT=q8
T0=$(date +%s)
{ echo "date=$(date -u +%FT%TZ) FABRIC_JOB=${FABRIC_JOB:-} MODE=$MODE"; uptime; cat /proc/loadavg; echo "GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-unset (HIP default 4)}"; env | grep -E '^(HIPFIRE|GPU_MAX|ROCR|HSA)' | sort
  md5sum $B/* 2>/dev/null; echo "link: $(cat $D/current_link_width) x $(cat $D/current_link_speed)"; free -g | head -2; cat $B/SOURCE 2>/dev/null; } > $O/pre.txt 2>&1
arm_env() { case $1 in
  S) echo "HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 HIPFIRE_QWEN4_TIER_COUNTERS=1" ;;
  A) echo "HIPFIRE_QWEN4_TIER_COUNTERS=1" ;;   # product default (selector unset = Auto: slab floor 128 rows)
  I) echo "HIPFIRE_QWEN4_SLOWLINK_PREFILL=0 HIPFIRE_PREFILL_CHUNK_ROWS=768 HIPFIRE_QWEN4_TIER_COUNTERS=1 ${I_VL:+HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=$I_VL}" ;;
esac; }
case $MODE in
state)
  python3 - <<PY
import json
def ids(seed,n):
    s=seed; out=[]
    for _ in range(n):
        s=(s*6364136223846793005+1442695040888963407)&((1<<64)-1); out.append(1000+(s>>33)%99000)
    return out
for rows in (3400,16384):
    json.dump({"prompt_ids":ids(7,rows),"max_seq":rows+256,"repeats":3},open("$O/fixture-%d.json"%rows,"w"))
PY
  for rows in 3400 16384; do for arm in S I; do
    env $(arm_env $arm) timeout 700 $B/qwen4_parity --mode slowlink-state --model $MODEL --tokens $O/fixture-$rows.json --out $O/state-$arm-$rows.json > $O/state-$arm-$rows.log 2>&1; echo "parity $arm $rows rc=$?" | tee -a $O/summary.txt
  done
  python3 - <<PY | tee -a $O/summary.txt
import json,os
try:
    a=json.load(open("$O/state-S-$rows.json")); b=json.load(open("$O/state-I-$rows.json"))
except Exception as e:
    print("rows $rows: report missing:",e); raise SystemExit
def strip(d): return {k:v for k,v in d.items() if k not in ("receipts","timing","wall_s","route","placement")}
print("rows $rows: S status",a.get("status"),"I status",b.get("status"),"| digests S==I:",strip(a)==strip(b))
PY
  grep -h "slow-link prefill (S1)" $O/state-S-$rows.log | head -2 | cut -c1-400 | tee -a $O/summary.txt
  done ;;
pp)
  for arm in ${ARMS:-S I}; do
    mkdir -p $O/pp-$arm
    ( while sleep 0.5; do cat $D/mem_info_vram_used; done > $O/pp-$arm/vram.txt 2>/dev/null ) & V=$!
    setsid env $(arm_env $arm) timeout 900 $B/hipfire bench $MODEL_NAME --matrix --pp ${PP:-1105,8192,16384} --ctx 1 --tg 1 --warmups 1 --runs 2 --spec off --max-seq 32768 --json > $O/pp-$arm/out.json 2> $O/pp-$arm/err.log
    echo "rc=$?" > $O/pp-$arm/meta.txt; kill $V; sort -n $O/pp-$arm/vram.txt | tail -1 | awk '{printf "vram_peak_MiB=%.0f\n",$1/1048576}' >> $O/pp-$arm/meta.txt; $B/hipfire stop >/dev/null 2>&1; sleep 3
    echo "pp $arm $(tr '\n' ' ' < $O/pp-$arm/meta.txt)" | tee -a $O/summary.txt
    grep -h "slow-link prefill (S1)" $O/pp-$arm/err.log | head -1 | cut -c1-500 | tee -a $O/summary.txt
  done ;;
esac
echo "ALLDONE $(date -u) elapsed $(( $(date +%s) - T0 ))s" | tee -a $O/summary.txt
