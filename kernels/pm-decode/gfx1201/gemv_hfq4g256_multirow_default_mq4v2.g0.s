.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected gemv_mq4g256v2_multirow_r2_dog_g0
.globl gemv_mq4g256v2_multirow_r2_dog_g0
.p2align 8
.type gemv_mq4g256v2_multirow_r2_dog_g0,@function
gemv_mq4g256v2_multirow_r2_dog_g0:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[8:9], s[0:1], 0x10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_kmcnt 0x0
	global_load_b32 v4, v1, s[4:5]
	global_load_b32 v5, v1, s[4:5] offset:136
	v_lshlrev_b32_e32 v1, 2, v0
	global_load_b32 v2, v1, s[4:5] offset:8
	global_load_b32 v3, v1, s[4:5] offset:144
	v_lshlrev_b32_e32 v1, 5, v0
	global_load_b128 v[12:15], v1, s[6:7]
	global_load_b128 v[16:19], v1, s[6:7] offset:16
	s_wait_loadcnt 0x3
	v_bfe_u32 v6, v2, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v12, v6
	v_bfe_u32 v6, v2, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v13, v6
	v_bfe_u32 v6, v2, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v14, v6
	v_bfe_u32 v6, v2, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v15, v6
	v_bfe_u32 v6, v2, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v16, v6
	v_bfe_u32 v6, v2, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v17, v6
	v_bfe_u32 v6, v2, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v18, v6
	v_bfe_u32 v6, v2, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v19, v6
	v_bfe_u32 v6, v3, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v13, v6
	v_bfe_u32 v6, v3, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v12, v6
	v_bfe_u32 v6, v3, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v14, v6
	v_bfe_u32 v6, v3, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v15, v6
	v_bfe_u32 v6, v3, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v16, v6
	v_bfe_u32 v6, v3, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v17, v6
	v_bfe_u32 v6, v3, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v18, v6
	v_bfe_u32 v6, v3, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v19, v6
	v_lshlrev_b32_e32 v1, 2, v0
	global_store_b32 v1, v7, s[8:9]
	global_store_b32 v1, v8, s[8:9] offset:128
	s_wait_storecnt 0x0
	s_endpgm
.Lgemv_mq4g256v2_multirow_r2_dog_g0_end:
.size gemv_mq4g256v2_multirow_r2_dog_g0, .Lgemv_mq4g256v2_multirow_r2_dog_g0_end-gemv_mq4g256v2_multirow_r2_dog_g0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq4g256v2_multirow_r2_dog_g0
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
	.amdhsa_next_free_vgpr 20
	.amdhsa_next_free_sgpr 10
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq4g256v2_multirow_r2_dog_g0_end-gemv_mq4g256v2_multirow_r2_dog_g0)<<4)&4080)>>4
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
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 32
    .name: gemv_mq4g256v2_multirow_r2_dog_g0
    .private_segment_fixed_size: 0
    .sgpr_count: 10
    .sgpr_spill_count: 0
    .symbol: gemv_mq4g256v2_multirow_r2_dog_g0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 20
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
