.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected fused_gate_up_mq4g256v2_g0
.globl fused_gate_up_mq4g256v2_g0
.p2align 8
.type fused_gate_up_mq4g256v2_g0,@function
fused_gate_up_mq4g256v2_g0:
	s_load_b64 s[4:5], s[0:1], 0x0
	s_load_b128 s[8:11], s[0:1], 0x10
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v2, 5, v0
	v_mov_b32_e32 v3, 0
	s_wait_kmcnt 0x0
	global_load_b64 v[6:7], v3, s[4:5]
	v_cmp_lt_u32_e64 s24, v0, 16
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v5, v7, v6, s24
	global_load_b32 v4, v1, s[4:5] offset:8
	global_load_b128 v[8:11], v2, s[8:9]
	global_load_b128 v[12:15], v2, s[8:9] offset:16
	v_mov_b32_e32 v24, 0
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v4, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v4, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v4, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v4, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v4, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v4, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v4, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v4, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v6, v9, v17
	v_fmac_f32_e32 v6, v8, v16
	v_fmac_f32_e32 v6, v10, v18
	v_fmac_f32_e32 v6, v11, v19
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v6, v12, v20
	v_fmac_f32_e32 v6, v13, v21
	v_fmac_f32_e32 v6, v14, v22
	v_fmac_f32_e32 v6, v15, v23
	v_add_f32_e32 v24, v24, v6
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_add_f32_e32 v24, v24, v25
	v_add_f32_e32 v26, v26, v27
	v_add_f32_e32 v24, v24, v26
	ds_swizzle_b32 v6, v24 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmp_lt_u32_e64 s24, v0, 24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, 0, 8, s24
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v6, v3, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmp_lt_u32_e64 s24, v0, 28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, 0, 4, s24
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v6, v3, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmp_lt_u32_e64 s24, v0, 30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, 0, 2, s24
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v6, v3, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmp_lt_u32_e64 s24, v0, 31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, 0, 1, s24
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v6, v3, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	global_store_b32 v1, v24, s[10:11]
	s_wait_storecnt 0x0
	s_endpgm
.Lfused_gate_up_mq4g256v2_g0_end:
.size fused_gate_up_mq4g256v2_g0, .Lfused_gate_up_mq4g256v2_g0_end-fused_gate_up_mq4g256v2_g0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel fused_gate_up_mq4g256v2_g0
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 52
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
	.amdhsa_next_free_vgpr 32
	.amdhsa_next_free_sgpr 32
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lfused_gate_up_mq4g256v2_g0_end-fused_gate_up_mq4g256v2_g0)<<4)&4080)>>4
	.amdhsa_round_robin_scheduling 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: A_gate
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: A_up
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: x
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y_gate
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y_up
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - 
        .name: gate_m
        .offset: 40
        .size: 4
        .value_kind: by_value
      - 
        .name: up_m
        .offset: 44
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 48
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .max_flat_workgroup_size: 32
    .name: fused_gate_up_mq4g256v2_g0
    .private_segment_fixed_size: 0
    .sgpr_count: 32
    .sgpr_spill_count: 0
    .symbol: fused_gate_up_mq4g256v2_g0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 32
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
