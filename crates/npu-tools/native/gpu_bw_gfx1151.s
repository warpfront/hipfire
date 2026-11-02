; PM-native gfx1151 uint4 grid-stride streaming kernels. No vendor assembler.
; ABI: buffer:u64, count:u64, checksum_sink:u64, stride:u64 (32 bytes).
; read_bw retains every load in four per-thread XOR lanes; not byte-exact proof.
; Fixed block size 256; wave32. Index/count/addresses are 64-bit.
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected read_bw
.globl read_bw
.p2align 8
.type read_bw,@function
read_bw:
    s_load_b256 s[4:11], s[0:1], null
    s_lshl_b32 s3, s2, 8
    v_add_nc_u32_e32 v0, s3, v0
    v_mov_b32_e32 v1, v0
    v_mov_b32_e32 v2, 0
    s_mov_b32 s14, exec_lo
    s_waitcnt lgkmcnt(0)
    v_mov_b32_e32 v9, 0
    v_mov_b32_e32 v10, 0
    v_mov_b32_e32 v11, 0
    v_mov_b32_e32 v12, 0
.Lread_bw_loop:
    v_cmp_gt_u32_e64 s15, s7, v2
    v_cmp_eq_u32_e64 s16, s7, v2
    v_cmp_gt_u32_e64 s17, s6, v1
    s_and_b32 s16, s16, s17
    s_or_b32 s15, s15, s16
    s_and_b32 exec_lo, exec_lo, s15
    s_cbranch_execz .Lread_bw_done
    v_lshlrev_b64_e64 v[3:4], 4, v[1:2]
    v_add_co_u32 v3, vcc_lo, s4, v3
    v_add_co_ci_u32_e64 v4, vcc_lo, s5, v4, vcc_lo
    global_load_b128 v[5:8], v[3:4], off
    s_waitcnt vmcnt(0)
    v_xor_b32_e32 v9, v5, v9
    v_xor_b32_e32 v10, v6, v10
    v_xor_b32_e32 v11, v7, v11
    v_xor_b32_e32 v12, v8, v12
    v_add_co_u32 v1, vcc_lo, s10, v1
    v_add_co_ci_u32_e64 v2, vcc_lo, s11, v2, vcc_lo
    s_branch .Lread_bw_loop
.Lread_bw_done:
    s_mov_b32 exec_lo, s14
    v_lshlrev_b32_e32 v3, 4, v0
    v_mov_b32_e32 v4, 0
    v_add_co_u32 v3, vcc_lo, s8, v3
    v_add_co_ci_u32_e64 v4, vcc_lo, s9, v4, vcc_lo
    global_store_b128 v[3:4], v[9:12], off
    s_waitcnt_vscnt null, 0
    s_endpgm
.Lread_bw_end:
.size read_bw, .Lread_bw_end-read_bw
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel read_bw
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
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
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 13
    .amdhsa_next_free_sgpr 18
    .amdhsa_reserve_vcc 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_fp16_overflow 0
    .amdhsa_workgroup_processor_mode 1
    .amdhsa_memory_ordered 1
    .amdhsa_forward_progress 1
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
    .amdhsa_inst_pref_size ((instprefsize(.Lread_bw_end-read_bw)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.protected write_bw
.globl write_bw
.p2align 8
.type write_bw,@function
write_bw:
    s_load_b256 s[4:11], s[0:1], null
    s_lshl_b32 s3, s2, 8
    v_add_nc_u32_e32 v0, s3, v0
    v_mov_b32_e32 v1, v0
    v_mov_b32_e32 v2, 0
    s_mov_b32 s14, exec_lo
    s_waitcnt lgkmcnt(0)
.Lwrite_bw_loop:
    v_cmp_gt_u32_e64 s15, s7, v2
    v_cmp_eq_u32_e64 s16, s7, v2
    v_cmp_gt_u32_e64 s17, s6, v1
    s_and_b32 s16, s16, s17
    s_or_b32 s15, s15, s16
    s_and_b32 exec_lo, exec_lo, s15
    s_cbranch_execz .Lwrite_bw_done
    v_lshlrev_b64_e64 v[3:4], 4, v[1:2]
    v_add_co_u32 v3, vcc_lo, s4, v3
    v_add_co_ci_u32_e64 v4, vcc_lo, s5, v4, vcc_lo
    v_mov_b32_e32 v5, v1
    v_xor_b32_e32 v6, 0x12345678, v1
    v_mul_lo_u32 v7, 0x0019660d, v1
    v_add_nc_u32_e32 v8, 0x3c6ef35f, v1
    global_store_b128 v[3:4], v[5:8], off
    s_waitcnt_vscnt null, 0
    v_add_co_u32 v1, vcc_lo, s10, v1
    v_add_co_ci_u32_e64 v2, vcc_lo, s11, v2, vcc_lo
    s_branch .Lwrite_bw_loop
.Lwrite_bw_done:
    s_mov_b32 exec_lo, s14
    s_waitcnt_vscnt null, 0
    s_endpgm
.Lwrite_bw_end:
.size write_bw, .Lwrite_bw_end-write_bw
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel write_bw
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
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
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 13
    .amdhsa_next_free_sgpr 18
    .amdhsa_reserve_vcc 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_fp16_overflow 0
    .amdhsa_workgroup_processor_mode 1
    .amdhsa_memory_ordered 1
    .amdhsa_forward_progress 1
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
    .amdhsa_inst_pref_size ((instprefsize(.Lwrite_bw_end-write_bw)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: buffer
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .name: count
        .offset: 8
        .size: 8
        .value_kind: by_value
      - .address_space: global
        .name: checksum_sink
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .name: stride
        .offset: 24
        .size: 8
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name: read_bw
    .private_segment_fixed_size: 0
    .sgpr_count: 20
    .sgpr_spill_count: 0
    .symbol: read_bw.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 13
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: buffer
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .name: count
        .offset: 8
        .size: 8
        .value_kind: by_value
      - .address_space: global
        .name: checksum_sink
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .name: stride
        .offset: 24
        .size: 8
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name: write_bw
    .private_segment_fixed_size: 0
    .sgpr_count: 20
    .sgpr_spill_count: 0
    .symbol: write_bw.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 13
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
