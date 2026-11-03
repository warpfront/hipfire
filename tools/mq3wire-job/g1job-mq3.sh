#!/bin/bash
# mq3wire G1: gfx1100 W3 K2 + W3 K1 byte-exact vs the CPU law on real mq3-xts layers. ONE fabric job (--class correct), <=25 min.
# Run `g1prep-mq3.sh` first (CPU only). Steps (each writes a receipt; the job stops at the first nonzero rc, resumable per step):
#   1 k2-w3-real   : W3 K2 (per-codec, real samples, Down chain, MQ4 anchor, 4 negatives) on XTX 0000:67:00.0
#   2 k2-mixed     : mixed run mq4=... w3=... on the same samples (G1c W3 reference mode)
#   3 k2-synth     : synthetic scenario set (no --sample-dir)
#   4 k1-w3-real   : W3 K1 card byte oracle (mq_i4_moe_gemv_gfx1100_w3_g1) on layers 0,23,47, VRAM + mapped host + split
# Qualification inputs: b4 base probe + divide table + the closed generic intrinsic table intrinsic-merged.c1.r3 (the fixture's NaN exp2/divide keys; CPU census on the staged samples+chain: hooks_closed, 0 missing of 133,025).
# usage (on hipx, inside fabric-run.sh --class correct --card xtx):  [CAP=1380] bash g1job-mq3.sh
X=/home/kaden/qcal/release-0.4.5/mgmtp/singlecard/mq3wire; G=$X/g1; B=$X/bin2; O=${OUT:-$G/out-$(date -u +%m%d-%H%M%S)}
K=$G/kernels; NEG=$K/negative; PROBE=$G/probe/observations.json; DIV=$G/probe/divide-observations.json; FIX=$G/probe/qwen4_moe_epilogue_fixture_pm_gfx1100.hxaco
UUID=GPU-43390a851e296ee5; BDF=0000:67:00.0; MODEL=/home/kaden/.hipfire/models/qwen3.8-flash-next-base.mq3-xts
CAP=${CAP:-1380}; T0=$(date +%s); mkdir -p $O
{ echo "date=$(date -u +%FT%TZ) FABRIC_JOB=${FABRIC_JOB:-}"; uptime; echo "GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-unset}"; env | grep -E '^(HIPFIRE|GPU_MAX|ROCR|HSA)' | sort
  md5sum $B/qwen4_moe_sym_gfx1100_g1_oracle $B/mq_i4_moe_gemv_gfx1100_w3_g1; sha256sum $K/*.hxaco; cat $X/SOURCE2; } > $O/pre.txt 2>&1
NEGS=(); for n in wrong-scale-row reversed-epoch wrong-pack-selector wrong-expert-pointer; do NEGS+=(--negative-codeobject $n=$NEG/$n/qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco); done
COMMON=(--mq4-anchor $K/qwen4_moe_iu4_sym_pm_gfx1100.hxaco --probe-table $PROBE --divide-table $DIV --intrinsic-table $G/probe/intrinsic-merged.c1.r3.json --epilogue-codeobject $FIX --card xtx --expect-uuid $UUID --expect-bdf $BDF)
run() { local name=$1; shift; [ -e $O/$name.DONE ] && { echo "skip $name"; return 0; }
  local el=$(( $(date +%s) - T0 )); [ $el -gt $((CAP - 180)) ] && { echo "SKIP $name (elapsed $el)"; return 3; }
  echo "start $name $(date -u +%T)"; "$@" > $O/$name.stdout 2> $O/$name.stderr; local rc=$?; echo "rc=$rc $name end $(date -u +%T)" | tee -a $O/steps.txt
  [ $rc = 0 ] && touch $O/$name.DONE; return $rc; }
run k2-w3-real $B/qwen4_moe_sym_gfx1100_g1_oracle --codeobject $K/qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco --codec w3 "${COMMON[@]}" --sample-dir $G/samples-mq3 --chain-dir $G/chain/w3 "${NEGS[@]}" --out $O/k2-w3-real.json || exit $?
run k2-mixed $B/qwen4_moe_sym_gfx1100_g1_oracle --mixed-codeobject mq4=$K/qwen4_moe_iu4_sym_pm_gfx1100.hxaco --mixed-codeobject w3=$K/qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco "${COMMON[@]}" --sample-dir $G/samples-mq3 --chain-dir $G/chain/w3 --out $O/k2-mixed.json || exit $?
run k2-synth $B/qwen4_moe_sym_gfx1100_g1_oracle --codeobject $K/qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco --codec w3 "${COMMON[@]}" "${NEGS[@]}" --out $O/k2-synth.json || exit $?
run k1-w3-real $B/mq_i4_moe_gemv_gfx1100_w3_g1 --codeobject $K/mq_i4_moe_gemv_mq3only_pm_gfx1100.hxaco --artifact $MODEL --layers 0,23,47 --real-experts 24 --expect-uuid $UUID --expect-bdf $BDF --expect-arch gfx1100 || exit $?
echo "ALLDONE $(date -u)"; python3 - <<E
import json,glob
for f in sorted(glob.glob("$O/*.json")):
    try: d=json.load(open(f)); print(f.split('/')[-1], "pass=",d.get("pass"), json.dumps(d.get("summary",{}))[:200])
    except Exception as e: print(f, e)
E
