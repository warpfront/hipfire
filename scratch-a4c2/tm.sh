#!/bin/bash
# tm.sh <rounds> <which: rms|hin|gated|sig|all> <dir...>
# Standalone ms per producer at the pp8192 shapes. Each (dir, producer)
# runs in its own fresh process (REPS x BATCH launches, median), rounds
# interleave the dirs. Prints "<producer> <dir> <median ms>".
cd "$(dirname "$0")"
rounds=$1; which=$2; shift 2
: "${REPS:=30}" "${BATCH:=20}"
export REPS BATCH
spec() {  # producer dir
  case $1 in
    rms) echo "rms 5120 A:$2/FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SRC.hsaco:fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2:8192:1:256:1024" ;;
    hin) echo "hin 17408 A:$2/FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_GFX12_SRC.hsaco:fused_silu_mul_mq_rotate_awq_i4_hin_gfx12:68:8192:32:0" ;;
    gated) echo "gated 6144 A:$2/GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SRC.hsaco:gated_norm_mq_rotate_awq_i4_gfx12_v2:12:8192:64:0" ;;
    sig) echo "sig 6144 A:$2/SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GFX12_SRC.hsaco:sigmoid_mul_rotate_x_mq_awq_i4_gfx12:196608:1:32:0" ;;
  esac
}
prods=$which; [ "$which" = all ] && prods="rms hin gated sig"
for r in $(seq "$rounds"); do
  for p in $prods; do
    for d in "$@"; do
      set -- "$@"
      read -r fam K s <<<"$(spec "$p" "$d")"
      ms=$(./harness "$fam" time "$K" 8192 5 0 "$s" | grep -E '^#0' | awk '{print $4}')
      echo "$p $d $ms"
    done
  done
done
