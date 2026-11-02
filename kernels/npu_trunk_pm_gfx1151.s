.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected npu_trunk_consume_bf16_pm
.globl npu_trunk_consume_bf16_pm
.p2align 8
.type npu_trunk_consume_bf16_pm,@function
npu_trunk_consume_bf16_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b32 s12, s[0:1], 0x10
	s_load_b32 s13, s[0:1], 0x14
	s_load_b32 s14, s[0:1], 0x18
	s_lshl_b32 s15, s2, 5
	v_add_nc_u32_e32 v1, s15, v0
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s34, s14, v1
	s_mov_b32 exec_lo, s34
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v2, s13, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v3, 0x3a4ccccd, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v4, 0x500, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v4, v2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s32, 0, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, -1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v3, v3, v30, s32
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, 0x500, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v4, v4, v30, s32
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s32, 0x500, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v3, v3, v30, s32
	s_delay_alu instid0(VALU_DEP_4)
	v_subrev_nc_u32_e32 v30, 0x500, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v4, v4, v30, s32
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v5, 0x3c4ccccd, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v6, 0x50, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s32, 0, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, -1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v5, v5, v30, s32
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, 0x50, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v30, s32
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s32, 0x50, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v5, v5, v30, s32
	s_delay_alu instid0(VALU_DEP_4)
	v_subrev_nc_u32_e32 v30, 0x50, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v30, s32
	v_mul_lo_u32 v7, s12, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v7, 0x500, v7
	v_lshrrev_b32_e32 v8, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v8, 0xa0, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v7, v8
	s_bfe_u32 s15, s3, 0x10007
	s_mul_i32 s15, s15, s12
	s_mul_i32 s15, s15, 0x50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s15, v7
	s_lshr_b32 s15, s3, 8
	s_mul_i32 s15, s15, 0x5000
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s15, v7
	v_and_b32_e32 v8, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v8, 0x2800, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v7, v8
	s_bfe_u32 s15, s3, 0x40003
	s_mul_i32 s15, s15, 0x280
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s15, v7
	v_lshrrev_b32_e32 v8, 3, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v7, v8
	s_and_b32 s15, s3, 7
	s_lshl_b32 s15, s15, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s15, v7
	v_and_b32_e32 v8, 7, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 1, v7
	buffer_load_u16 v9, v7, s[4:7], 0 offen
	s_mul_i32 s15, s3, s14
	v_add_nc_u32_e32 v10, s15, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 1, v10
	s_waitcnt vmcnt(0)
	buffer_store_b16 v9, v10, s[8:11], 0 offen
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lnpu_trunk_consume_bf16_pm_end:
.size npu_trunk_consume_bf16_pm, .Lnpu_trunk_consume_bf16_pm_end-npu_trunk_consume_bf16_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel npu_trunk_consume_bf16_pm
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 28
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
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 40
	.amdhsa_next_free_sgpr 40
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lnpu_trunk_consume_bf16_pm_end-npu_trunk_consume_bf16_pm)<<4)&1008)>>4
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
.text
.text
.protected npu_trunk_consume_f32_pm
.globl npu_trunk_consume_f32_pm
.p2align 8
.type npu_trunk_consume_f32_pm,@function
npu_trunk_consume_f32_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b32 s12, s[0:1], 0x10
	s_load_b32 s13, s[0:1], 0x14
	s_load_b32 s14, s[0:1], 0x18
	s_lshl_b32 s15, s2, 5
	v_add_nc_u32_e32 v1, s15, v0
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s34, s14, v1
	s_mov_b32 exec_lo, s34
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v2, s13, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v3, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v3, 0x3a4ccccd, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v3, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v4, 0x500, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v4, v2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s32, 0, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, -1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v3, v3, v30, s32
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, 0x500, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v4, v4, v30, s32
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s32, 0x500, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v3, v3, v30, s32
	s_delay_alu instid0(VALU_DEP_4)
	v_subrev_nc_u32_e32 v30, 0x500, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v4, v4, v30, s32
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v5, 0x3c4ccccd, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v6, 0x50, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v6, v4, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s32, 0, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, -1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v5, v5, v30, s32
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, 0x50, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v30, s32
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s32, 0x50, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v30, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v5, v5, v30, s32
	s_delay_alu instid0(VALU_DEP_4)
	v_subrev_nc_u32_e32 v30, 0x50, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v30, s32
	v_mul_lo_u32 v7, s12, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v7, 0x500, v7
	v_lshrrev_b32_e32 v8, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v8, 0xa0, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v7, v8
	s_bfe_u32 s15, s3, 0x10007
	s_mul_i32 s15, s15, s12
	s_mul_i32 s15, s15, 0x50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s15, v7
	s_lshr_b32 s15, s3, 8
	s_mul_i32 s15, s15, 0x5000
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s15, v7
	v_and_b32_e32 v8, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v8, 0x2800, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v7, v8
	s_bfe_u32 s15, s3, 0x40003
	s_mul_i32 s15, s15, 0x280
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s15, v7
	v_lshrrev_b32_e32 v8, 3, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v7, v8
	s_and_b32 s15, s3, 7
	s_lshl_b32 s15, s15, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s15, v7
	v_and_b32_e32 v8, 7, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 1, v7
	buffer_load_u16 v9, v7, s[4:7], 0 offen
	s_mul_i32 s15, s3, s14
	v_add_nc_u32_e32 v10, s15, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 2, v10
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v9, 16, v9
	buffer_store_b32 v9, v10, s[8:11], 0 offen
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lnpu_trunk_consume_f32_pm_end:
.size npu_trunk_consume_f32_pm, .Lnpu_trunk_consume_f32_pm_end-npu_trunk_consume_f32_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel npu_trunk_consume_f32_pm
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 28
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
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 40
	.amdhsa_next_free_sgpr 40
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lnpu_trunk_consume_f32_pm_end-npu_trunk_consume_f32_pm)<<4)&1008)>>4
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
.text
.text
.protected npu_trunk_i8_gemm_pm
.globl npu_trunk_i8_gemm_pm
.p2align 8
.type npu_trunk_i8_gemm_pm,@function
npu_trunk_i8_gemm_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b64 s[12:13], s[0:1], 0x10
	s_mov_b32 s14, -1
	s_mov_b32 s15, 0x31004000
	s_load_b32 s16, s[0:1], 0x18
	s_load_b32 s17, s[0:1], 0x1c
	s_load_b32 s18, s[0:1], 0x20
	s_load_b32 s19, s[0:1], 0x24
	v_and_b32_e32 v1, 15, v0
	s_lshl_b32 s20, s3, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s20, v1
	v_and_b32_e32 v2, 15, v0
	s_lshl_b32 s20, s2, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s20, v2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s20, s18, -1
	s_delay_alu instid0(VALU_DEP_1)
	v_min_u32_e32 v2, s20, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v3, s16, v2
	s_lshr_b32 s20, s19, 8
	s_lshr_b32 s21, s16, 6
	s_mul_i32 s20, s20, s21
	s_mul_i32 s20, s20, 0x1200
	v_bfe_u32 v4, v1, 7, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v4, 1, v4
	v_bfe_u32 v5, v1, 3, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v4, s20, v4
	v_lshrrev_b32_e32 v5, 8, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v5, s21, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v5, 0x1200, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4, v4, v5
	v_bfe_u32 v5, v1, 4, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 9, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4, v4, v5
	v_and_b32_e32 v5, 7, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 3, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v4, v4, v5
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	s_mov_b32 s22, 0
	.Lnpu_k:
	s_lshr_b32 s23, s22, 6
	s_mul_i32 s23, s23, 0x1200
	s_and_b32 s24, s22, 63
	s_lshr_b32 s24, s24, 3
	s_lshl_b32 s24, s24, 6
	s_add_u32 s23, s23, s24
	v_add_nc_u32_e32 v6, s23, v4
	v_add_nc_u32_e32 v7, s22, v3
	buffer_load_b32 v16, v6, s[4:7], 0 offen
	buffer_load_b32 v17, v6, s[4:7], 0 offen offset:4
	buffer_load_b32 v18, v6, s[4:7], 0 offen offset:64
	buffer_load_b32 v19, v6, s[4:7], 0 offen offset:68
	buffer_load_b128 v[20:23], v7, s[8:11], 0 offen
	s_waitcnt vmcnt(0)
	v_wmma_i32_16x16x16_iu8 v[24:31], v[20:23], v[16:19], v[24:31] neg_lo:[1,1,0]
	s_add_u32 s22, s22, 16
	s_cmp_lt_u32 s22, s16
	s_cbranch_scc1 .Lnpu_k
	v_cmp_gt_u32_e64 s34, s17, v1
	v_mul_lo_u32 v8, s18, v1
	v_lshrrev_b32_e32 v9, 4, v0
	s_lshl_b32 s20, s2, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v9, s20, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v10, 0, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s36, s18, v10
	s_and_b32 exec_lo, s34, s36
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v11, v8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b32 v24, v11, s[12:15], 0 offen
	s_mov_b32 exec_lo, -1
	v_add_nc_u32_e32 v10, 2, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s36, s18, v10
	s_and_b32 exec_lo, s34, s36
	s_waitcnt_vscnt null, 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v11, v8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b32 v25, v11, s[12:15], 0 offen
	s_mov_b32 exec_lo, -1
	v_add_nc_u32_e32 v10, 4, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s36, s18, v10
	s_and_b32 exec_lo, s34, s36
	s_waitcnt_vscnt null, 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v11, v8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b32 v26, v11, s[12:15], 0 offen
	s_mov_b32 exec_lo, -1
	v_add_nc_u32_e32 v10, 6, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s36, s18, v10
	s_and_b32 exec_lo, s34, s36
	s_waitcnt_vscnt null, 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v11, v8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b32 v27, v11, s[12:15], 0 offen
	s_mov_b32 exec_lo, -1
	v_add_nc_u32_e32 v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s36, s18, v10
	s_and_b32 exec_lo, s34, s36
	s_waitcnt_vscnt null, 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v11, v8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b32 v28, v11, s[12:15], 0 offen
	s_mov_b32 exec_lo, -1
	v_add_nc_u32_e32 v10, 10, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s36, s18, v10
	s_and_b32 exec_lo, s34, s36
	s_waitcnt_vscnt null, 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v11, v8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b32 v29, v11, s[12:15], 0 offen
	s_mov_b32 exec_lo, -1
	v_add_nc_u32_e32 v10, 12, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s36, s18, v10
	s_and_b32 exec_lo, s34, s36
	s_waitcnt_vscnt null, 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v11, v8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b32 v30, v11, s[12:15], 0 offen
	s_mov_b32 exec_lo, -1
	v_add_nc_u32_e32 v10, 14, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s36, s18, v10
	s_and_b32 exec_lo, s34, s36
	s_waitcnt_vscnt null, 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v11, v8, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b32 v31, v11, s[12:15], 0 offen
	s_mov_b32 exec_lo, -1
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lnpu_trunk_i8_gemm_pm_end:
.size npu_trunk_i8_gemm_pm, .Lnpu_trunk_i8_gemm_pm_end-npu_trunk_i8_gemm_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel npu_trunk_i8_gemm_pm
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 40
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
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 40
	.amdhsa_next_free_sgpr 40
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lnpu_trunk_i8_gemm_pm_end-npu_trunk_i8_gemm_pm)<<4)&1008)>>4
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
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: C
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: Y
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - 
        .name: padded_rows
        .offset: 16
        .size: 4
        .value_kind: by_value
      - 
        .name: col0
        .offset: 20
        .size: 4
        .value_kind: by_value
      - 
        .name: cols
        .offset: 24
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .max_flat_workgroup_size: 32
    .name: npu_trunk_consume_bf16_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 40
    .sgpr_spill_count: 0
    .symbol: npu_trunk_consume_bf16_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 40
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: C
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: Y
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - 
        .name: padded_rows
        .offset: 16
        .size: 4
        .value_kind: by_value
      - 
        .name: col0
        .offset: 20
        .size: 4
        .value_kind: by_value
      - 
        .name: cols
        .offset: 24
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .max_flat_workgroup_size: 32
    .name: npu_trunk_consume_f32_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 40
    .sgpr_spill_count: 0
    .symbol: npu_trunk_consume_f32_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 40
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: W
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: I
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: k
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: rows
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: n
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: padded_rows
        .offset: 36
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 32
    .name: npu_trunk_i8_gemm_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 40
    .sgpr_spill_count: 0
    .symbol: npu_trunk_i8_gemm_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 40
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
