#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
# Regenerate the coop27 PM-native code objects (no vendor assembler/compiler). Build peacemaker from the approved PM tree:
#   CARGO_TARGET_DIR=/path/out cargo build --locked --manifest-path /path/wt-land-041m/Cargo.toml -p peacemaker-isa --features toolchain --bin peacemaker
#   PEACEMAKER=/path/out/debug/peacemaker bash crates/npu-tools/native/coop27/build.sh
# ief_silu_gfx1151.s is generated: head + the incumbent's verbatim SiLU group DAG + tail (see ief_silu.head.s).
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pm=${PEACEMAKER:-peacemaker}
"$pm" native --arch=gfx1151 "$here/ief_coop_gfx1151.s" --co="$here/ief_coop_gfx1151.co"
"$pm" native --arch=gfx1151 "$here/ief_glue2_gfx1151.s" --co="$here/ief_glue2_gfx1151.co"
cat "$here/ief_silu.head.s" "$here/silu_dag_gfx1151.inc" "$here/ief_silu.tail.s" > "$here/ief_silu_gfx1151.s"
"$pm" native --arch=gfx1151 "$here/ief_silu_gfx1151.s" --co="$here/ief_silu_gfx1151.co"
# ief_silu2: the same DAG with VGPRs renamed v32..39 -> v8..15 (gate), v64..71 -> v16..23 (up), v160..207 -> v24..71.
awk '{ out = ""; s = $0
  while (match(s, /v[0-9]+/)) {
    n = substr(s, RSTART + 1, RLENGTH - 1) + 0; m = n
    if (n >= 32 && n <= 39) m = n - 24; else if (n >= 64 && n <= 71) m = n - 48; else if (n >= 160 && n <= 207) m = n - 136
    else { print "unexpected register v" n > "/dev/stderr"; exit 1 }
    out = out substr(s, 1, RSTART - 1) "v" m; s = substr(s, RSTART + RLENGTH)
  }
  print out s }' "$here/silu_dag_gfx1151.inc" > "$here/silu_dag_lowreg.inc"
cat "$here/ief_silu2.head.s" "$here/silu_dag_lowreg.inc" "$here/ief_silu2.tail.s" > "$here/ief_silu2_gfx1151.s"
"$pm" native --arch=gfx1151 "$here/ief_silu2_gfx1151.s" --co="$here/ief_silu2_gfx1151.co"

# ---- ief_fold_gfx1151: epilogue fold twins + merged wide tails (coop-epi-fuse) ----
# v2b_pm_gfx1151.s is the PM emitter output `peacemaker-isa emit --kernel iu4_v2b --epi all --arch gfx1151` of hipfire
# wt-land-041m 9aef0a35f; assembled alone it is byte-identical to the ELF inside the vendored .hxaco (the incumbent).
# ief_gu_fold / ief_add_fold: the incumbent gate_up SiLU / down ADD entry instruction for instruction (label to its end
# label), then fold_sched.inc (NPU-column SiLU / addc, flag-gated, work-claimed) before the VGPR dealloc + s_endpgm.
# ief_silu_wide / ief_addc_wide: the same scheduler in static mode as a single tail launch over all J commands.
v2b="$here/v2b_pm_gfx1151.s"
fold_s="$here/ief_fold_gfx1151.s"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
# One pass = 8 groups k (8 wave-units): input registers gb[k] (8 g | C, then 8 u | old x), output offset vh[k], C offset
# k*1024 from v11 (k < 4) or (k-4)*1024 from v16 = v11 + 4096 (the 13-bit signed offset field).
gb=(80 96 112 128 144 208 224 240)
vh=(v6 v8 v9 v10 v12 v13 v14 v15)
off() { [ "$1" -eq 0 ] && echo "" || echo " offset:$1"; }
loads() { # kind: silu (g, u from C) | addc (C, old x)
  for k in 0 1 2 3 4 5 6 7; do b=${gb[$k]}
    if [ $k -lt 4 ]; then va=v11; o=$((1024 * k)); else va=v16; o=$((1024 * (k - 4))); fi
    echo "    global_load_b128 v[$b:$((b + 3))], $va, s[38:39]$(off $o) glc dlc"
    echo "    global_load_b128 v[$((b + 4)):$((b + 7))], $va, s[38:39]$(off $((o + 16))) glc dlc"
    if [ "$1" = silu ]; then
      echo "    global_load_b128 v[$((b + 8)):$((b + 11))], $va, s[40:41]$(off $o) glc dlc"
      echo "    global_load_b128 v[$((b + 12)):$((b + 15))], $va, s[40:41]$(off $((o + 16))) glc dlc"
    else
      echo "    global_load_b128 v[$((b + 8)):$((b + 11))], ${vh[$k]}, s[16:17]"
      echo "    global_load_b128 v[$((b + 12)):$((b + 15))], ${vh[$k]}, s[16:17] offset:16"
    fi
  done
}
bodies() {
  for k in 0 1 2 3 4 5 6 7; do b=${gb[$k]}
    echo "    s_waitcnt vmcnt($((28 - 4 * k)))"
    if [ "$1" = silu ]; then
      for i in 0 1 2 3 4 5 6 7; do echo "    v_mov_b32_e32 v$((32 + i)), v$((b + i))"; done
      for i in 0 1 2 3 4 5 6 7; do echo "    v_mov_b32_e32 v$((64 + i)), v$((b + 8 + i))"; done
      cat "$here/silu_dag_gfx1151.inc"
      echo "    global_store_b128 ${vh[$k]}, v[32:35], s[16:17]"
      echo "    global_store_b128 ${vh[$k]}, v[36:39], s[16:17] offset:16"
    else
      for i in 0 1 2 3 4 5 6 7; do echo "    v_add_f32_e32 v$((b + 8 + i)), v$((b + 8 + i)), v$((b + i))"; done
      echo "    global_store_b128 ${vh[$k]}, v[$((b + 8)):$((b + 11))], s[16:17]"
      echo "    global_store_b128 ${vh[$k]}, v[$((b + 12)):$((b + 15))], s[16:17] offset:16"
    fi
  done
}
sched() { # prefix kind
  loads "$2" > "$tmp/l"; bodies "$2" > "$tmp/b"
  sed "s/@P@/$1/g" "$here/fold_sched.inc" | awk -v L="$tmp/l" -v B="$tmp/b" '
    /^;@LOADS@$/ { while ((getline x < L) > 0) print x; next }
    /^;@BODY@$/ { while ((getline x < B) > 0) print x; next }
    { print }'
}
fn_head() { printf '.text\n.protected %s\n.globl %s\n.p2align 8\n.type %s,@function\n%s:\n' "$1" "$1" "$1" "$1"; }
fn_tail() { # name vgpr sgpr wg_id_y
  cat <<D
    s_mov_b32 m0, 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.L$1_end:
.size $1, .L$1_end-$1
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel $1
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 160
	.amdhsa_user_sgpr_count 2
	.amdhsa_user_sgpr_dispatch_ptr 0
	.amdhsa_user_sgpr_queue_ptr 0
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 0
	.amdhsa_user_sgpr_private_segment_size 0
	.amdhsa_wavefront_size32 1
	.amdhsa_uses_dynamic_stack 0
	.amdhsa_enable_private_segment 0
	.amdhsa_system_sgpr_workgroup_id_x 1
	.amdhsa_system_sgpr_workgroup_id_y $4
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr $2
	.amdhsa_next_free_sgpr $3
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.L$1_end-$1)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
D
}
twin() { # name incumbent-label incumbent-end-label prefix kind
  fn_head "$1"
  awk -v s="$2:" -v e="	$3:" '$0 == s { p = 1; next } $0 == e { p = 0 } p' "$v2b"
  sched "$4" "$5"
  fn_tail "$1" 256 96 1
}
wide() { # name prefix kind
  fn_head "$1"
  printf '    v_lshrrev_b32_e32 v1, 5, v0\n    v_readfirstlane_b32 s21, v1\n'
  sched "$2" "$3"
  fn_tail "$1" 256 96 0
}
meta() { # name block
  printf '  - .args:\n'
  for a in a0 a1 a2 a3 a4 a5 c0 c1 c2 c3 flags ctr hbase; do
    printf '      - .name: %s\n        .offset: %d\n        .size: 8\n        .value_kind: by_value\n' "$a" "$o"; o=$((o + 8))
  done
  for a in round J ldy waves NW half lgx lgy upc drain_wg nwg mode chw claim; do
    printf '      - .name: %s\n        .offset: %d\n        .size: 4\n        .value_kind: by_value\n' "$a" "$o"; o=$((o + 4))
  done
  cat <<M
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 160
    .max_flat_workgroup_size: $2
    .name: $1
    .private_segment_fixed_size: 0
    .sgpr_count: 98
    .sgpr_spill_count: 0
    .symbol: $1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: $3
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
M
}
{
  printf '; GENERATED by native/build.sh from v2b_pm_gfx1151.s + fold_sched.inc + silu_dag_gfx1151.inc. Do not edit.\n'
  printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"\n.amdhsa_code_object_version 6\n'
  twin ief_gu_fold gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_gfx1151 .Lv2b_silu_end gu silu
  twin ief_add_fold gemm_mq4g256v2_residual_iu4_pm_v2b_add_gfx1151 .Lv2b_add_end ad addc
  wide ief_silu_wide sw silu
  wide ief_addc_wide aw addc
  printf '.text\n.amdgpu_metadata\n---\namdhsa.kernels:\n'
  o=0; meta ief_gu_fold 512 256
  o=0; meta ief_add_fold 512 256
  o=0; meta ief_silu_wide 256 256
  o=0; meta ief_addc_wide 256 256
  printf 'amdhsa.target: amdgcn-amd-amdhsa--gfx1151\namdhsa.version: [1, 2]\n...\n.end_amdgpu_metadata\n'
} > "$fold_s"
"$pm" native --arch=gfx1151 "$fold_s" --co="$here/ief_fold_gfx1151.co"
