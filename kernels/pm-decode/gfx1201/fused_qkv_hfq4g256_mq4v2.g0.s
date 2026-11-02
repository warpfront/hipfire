.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected fused_qkv_mq4g256v2_g0_group
.globl fused_qkv_mq4g256v2_g0_group
.p2align 8
.type fused_qkv_mq4g256v2_g0_group,@function
fused_qkv_mq4g256v2_g0_group:
	s_load_b64 s[4:5], s[0:1], 0x0
	s_load_b64 s[6:7], s[0:1], 0x18
	s_load_b64 s[8:9], s[0:1], 0x20
	v_lshlrev_b32_e32 v1, 2, v0
	v_add_nc_u32_e32 v2, 8, v1
	v_lshlrev_b32_e32 v3, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e64 v4, 4, 0, vcc_lo
	s_wait_kmcnt 0x0
	global_load_b32 v5, v4, s[4:5]
	global_load_b32 v6, v2, s[4:5]
	global_load_b128 v[8:11], v3, s[6:7]
	global_load_b128 v[12:15], v3, s[6:7] offset:16
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v6, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v6, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v6, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v6, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v6, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v6, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v6, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v6, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v24, v17, v9
	v_fmac_f32_e32 v24, v16, v8
	v_fmac_f32_e32 v24, v18, v10
	v_fmac_f32_e32 v24, v19, v11
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v24, v20, v12
	v_fmac_f32_e32 v24, v21, v13
	v_fmac_f32_e32 v24, v22, v14
	v_fmac_f32_e32 v24, v23, v15
	v_mov_b32_e32 v25, 0
	v_add_f32_e32 v24, v25, v24
	global_store_b32 v1, v24, s[8:9]
	s_wait_storecnt 0x0
	s_endpgm
.Lfused_qkv_mq4g256v2_g0_group_end:
.size fused_qkv_mq4g256v2_g0_group, .Lfused_qkv_mq4g256v2_g0_group_end-fused_qkv_mq4g256v2_g0_group
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel fused_qkv_mq4g256v2_g0_group
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 72
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
	.amdhsa_next_free_vgpr 26
	.amdhsa_next_free_sgpr 10
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lfused_qkv_mq4g256v2_g0_group_end-fused_qkv_mq4g256v2_g0_group)<<4)&4080)>>4
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
        .name: A_q
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: A_k
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: A_v
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: x
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y_q
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y_k
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y_v
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: q_m
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: k_m
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: v_m
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 68
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .max_flat_workgroup_size: 32
    .name: fused_qkv_mq4g256v2_g0_group
    .private_segment_fixed_size: 0
    .sgpr_count: 12
    .sgpr_spill_count: 0
    .symbol: fused_qkv_mq4g256v2_g0_group.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 26
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
