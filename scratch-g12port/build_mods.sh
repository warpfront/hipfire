#!/usr/bin/env bash
# Build the gfx1201 JIT-equivalent modules (same concatenation as kernels.rs,
# same hipcc argv as compiler.rs direct_hipcc_args) for base (2afc4a294) and
# this tree, then diff the unchanged entries' disassembly.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=scratch-g12port/mods
mkdir -p "$OUT"
BASE=2afc4a294
HIPCC=(/opt/rocm/bin/hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress -I/opt/rocm/include)

sym_hdr='#define IU4_SYMMETRIC_FOLD 1
#define gemm_mq4g256v2_residual_mmq_iu4 gemm_mq4g256v2_residual_mmq_iu4_symfold
#define gemm_mq4g256v2_residual_mmq_iu4_full_add gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold
#define gemm_mq4g256v2_residual_mmq_iu4_full_set gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold
#define gemm_mq4g256v2_gate_up_silu_mmq_iu4 gemm_mq4g256v2_gate_up_silu_mmq_iu4_symfold
#define gemm_mq4g256v2_zba_set_mmq_iu4 gemm_mq4g256v2_zba_set_mmq_iu4_symfold'

gemm_src() {  # $1 = rev or "" for worktree
  printf '%s\n' "$sym_hdr"
  if [ -n "$1" ]; then
    git show "$1:kernels/src/block_i4_128_quant.hip"
    git show "$1:kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip"
  else
    cat kernels/src/block_i4_128_quant.hip kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip
  fi
}
silu_src() {  # $1 = rev or "", $2 = extra defines
  printf '#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1\n'
  if [ -n "$1" ]; then git show "$1:kernels/src/block_i4_128_quant.hip"; else cat kernels/src/block_i4_128_quant.hip; fi
  printf '#define HIPFIRE_IU4_SIDECAR 1\n%s\n' "$2"
  if [ -n "$1" ]; then git show "$1:kernels/src/fused_silu_mul_mq_rotate_awq.hip"; else cat kernels/src/fused_silu_mul_mq_rotate_awq.hip; fi
}
rms_src() {  # $1 = rev or "", $2 = extra defines
  printf '#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1\n'
  if [ -n "$1" ]; then git show "$1:kernels/src/block_i4_128_quant.hip"; else cat kernels/src/block_i4_128_quant.hip; fi
  printf '#define HIPFIRE_IU4_SIDECAR 1\n#define HIPFIRE_RMSNORM_AWQ 1\n%s\n' "$2"
  if [ -n "$1" ]; then git show "$1:kernels/src/fused_rmsnorm_mq_rotate.hip"; else cat kernels/src/fused_rmsnorm_mq_rotate.hip; fi
}

gemm_src "$BASE" > "$OUT/gemm_base.hip"
gemm_src "" > "$OUT/gemm_new.hip"
silu_src "$BASE" '#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate_awq_i4_gfx12' > "$OUT/silu_base.hip"
silu_src "" '#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate_awq_i4_gfx12' > "$OUT/silu_new.hip"
silu_src "" '#define HIPFIRE_SILU_HIN 1
#define HIPFIRE_SILU_MQ_ROTATE_KERNEL fused_silu_mul_mq_rotate_awq_i4_hin_gfx12' > "$OUT/silu_hin.hip"
rms_src "$BASE" '#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4_gfx12' > "$OUT/rms_base.hip"
rms_src "" '#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4_gfx12' > "$OUT/rms_new.hip"
rms_src b3b6742a3 '#define HIPFIRE_RMSNORM_KEEP_X 1
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4_keep_x_gfx12' > "$OUT/rms_keep.hip"

{ printf '#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n'
  cat kernels/src/gfx12_weight_cache_policy.inc kernels/src/gemv_mq4g256v2_residual.hip; } > "$OUT/gemv_res.hip"
for m in gemm_base gemm_new silu_base silu_new silu_hin rms_base rms_new rms_keep gemv_res; do
  "${HIPCC[@]}" -o "$OUT/$m.hsaco" "$OUT/$m.hip" &
done
wait
for m in gemm_base gemm_new silu_base silu_new silu_hin rms_base rms_new rms_keep gemv_res; do
  /opt/rocm/llvm/bin/clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 --input="$OUT/$m.hsaco" --output="$OUT/$m.co"
  /opt/rocm/llvm/bin/llvm-objdump -d --no-show-raw-insn --no-leading-addr "$OUT/$m.co" \
    | sed -E 's@//.*$@@' > "$OUT/$m.dis"
  /opt/rocm/llvm/bin/llvm-readelf --notes "$OUT/$m.co" > "$OUT/$m.notes"
done

# Instruction identity of unchanged entries (function bodies, addresses stripped).
# Trailing s_nop/s_code_end lines are inter-function alignment padding.
extract() { awk -v f="<$2>:" '$0 ~ "^"f {p=1; next} /^<.*>:$/ {p=0} p' "$1" | sed -E 's/<[^>]*>//g; /^\s*$/d; /s_code_end/d' \
  | tac | awk 'body || !/^\s*s_nop 0\s*$/ {body=1; print}' | tac; }
for f in gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold gemm_mq4g256v2_residual_mmq_iu4_symfold; do
  if cmp -s <(extract "$OUT/gemm_base.dis" $f) <(extract "$OUT/gemm_new.dis" $f); then
    echo "DISASM identical: $f ($(extract "$OUT/gemm_new.dis" $f | wc -l) lines)"
  else echo "DISASM DIFFERS: $f"; fi
done
for p in silu rms; do
  f=$([ $p = silu ] && echo fused_silu_mul_mq_rotate_awq_i4_gfx12 || echo fused_rmsnorm_mq_rotate_awq_i4_gfx12)
  if cmp -s <(extract "$OUT/${p}_base.dis" $f) <(extract "$OUT/${p}_new.dis" $f); then
    echo "DISASM identical: $f ($(extract "$OUT/${p}_new.dis" $f | wc -l) lines)"
  else echo "DISASM DIFFERS: $f"; fi
done
# Resources for every entry of interest.
for m in gemm_new silu_hin rms_keep rms_new silu_new; do
  grep -E '\.name:|\.vgpr_count:|\.sgpr_count:|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size' "$OUT/$m.notes" \
    | sed "s/^/$m: /" | grep -vE 'name: +(\.|_)' || true
done
