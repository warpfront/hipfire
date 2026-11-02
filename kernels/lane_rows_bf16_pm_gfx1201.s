.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected gemv_bf16_xf32_x4_c1_pm_gfx1201
.globl gemv_bf16_xf32_x4_c1_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_x4_c1_pm_gfx1201,@function
gemv_bf16_xf32_x4_c1_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b32 s32, s30
	s_mov_b32 s33, s31
	s_mov_b64 s[4:5], s[16:17]
	s_mov_b32 s36, s26
	s_add_co_i32 s37, s36, s27
	s_add_co_i32 s38, s37, s28
	s_add_co_i32 s39, s38, s29
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lb16x4c1_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16x4c1_end
	s_cmp_gt_u32 s33, 1
	s_cbranch_scc1 .Lb16x4c1_end
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[2:3], s[12:13], s[14:15]
	s_cselect_b64 s[6:7], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[2:3], s[10:11], s[2:3]
	s_cselect_b64 s[6:7], s[20:21], s[6:7]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[2:3], s[8:9], s[2:3]
	s_cselect_b64 s[6:7], s[18:19], s[6:7]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_co_i32 s42, s40, s35
	s_and_b32 s56, s32, 7
	s_mov_b32 s58, s32
	s_add_co_i32 s44, s32, 0xff
	s_lshr_b32 s44, s44, 8
	s_add_co_i32 s59, s32, 31
	s_lshr_b32 s59, s59, 5
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_lshl_b32 s54, s42, 2
	s_mul_i32 s95, s42, s57
	s_mul_hi_u32 s96, s42, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mov_b64 s[64:65], s[4:5]
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	s_mov_b32 s51, 0
	s_cmp_lg_u32 s56, 0
	s_cbranch_scc1 .Lb16x4c1_scalar
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16x4c1_fold
	.Lb16x4c1_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	global_load_b128 v[24:27], v2, s[64:65]
	global_load_b128 v[28:31], v2, s[64:65] offset:16
	.Lb16x4c1_gx_done:
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v32, v24, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v33, v25, v17
	v_fmac_f32_e32 v34, v26, v18
	v_fmac_f32_e32 v35, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v32, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v33, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v34, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v35, v31, v23
	.Lb16x4c1_gc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16x4c1_group
	s_branch .Lb16x4c1_fold
	.Lb16x4c1_scalar:
	.Lb16x4c1_tile32:
	s_lshl_b32 s49, s51, 6
	s_lshl_b32 s50, s51, 7
	s_lshl_b32 s48, s51, 5
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s47, s48
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 32
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_u16 v16, v1, s[2:3]
	global_load_b32 v24, v2, s[64:65]
	.Lb16x4c1_sx_done:
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v32, v24, v16
	.Lb16x4c1_sc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s59
	s_cbranch_scc1 .Lb16x4c1_tile32
	.Lb16x4c1_fold:
	v_add_f32_e32 v33, v33, v32
	v_add_f32_e32 v35, v35, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v33
	ds_swizzle_b32 v24, v35 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v35, v35, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v35, v35, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v35, v35, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v35, v35, v24
	v_cmpx_eq_u32_e32 0, v4
	s_mul_i32 s55, s53, 0
	s_add_co_i32 s55, s55, s54
	v_mov_b32_e32 v24, s55
	global_store_b32 v24, v35, s[6:7]
	.Lb16x4c1_stored:
	s_wait_storecnt 0x0
	.Lb16x4c1_end:
	s_endpgm
.Lgemv_bf16_xf32_x4_c1_pm_gfx1201_end:
.size gemv_bf16_xf32_x4_c1_pm_gfx1201, .Lgemv_bf16_xf32_x4_c1_pm_gfx1201_end-gemv_bf16_xf32_x4_c1_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_x4_c1_pm_gfx1201
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
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
	.amdhsa_next_free_vgpr 36
	.amdhsa_next_free_sgpr 97
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_x4_c1_pm_gfx1201_end-gemv_bf16_xf32_x4_c1_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201
.globl gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201,@function
gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b32 s32, s30
	s_mov_b32 s33, s31
	s_mov_b64 s[4:5], s[16:17]
	s_mov_b32 s36, s26
	s_add_co_i32 s37, s36, s27
	s_add_co_i32 s38, s37, s28
	s_add_co_i32 s39, s38, s29
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lb16x4r4_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16x4r4_end
	s_cmp_gt_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r4_end
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[2:3], s[12:13], s[14:15]
	s_cselect_b64 s[6:7], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[2:3], s[10:11], s[2:3]
	s_cselect_b64 s[6:7], s[20:21], s[6:7]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[2:3], s[8:9], s[2:3]
	s_cselect_b64 s[6:7], s[18:19], s[6:7]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_co_i32 s42, s40, s35
	s_and_b32 s56, s32, 7
	s_mov_b32 s58, s32
	s_add_co_i32 s44, s32, 0xff
	s_lshr_b32 s44, s44, 8
	s_add_co_i32 s59, s32, 31
	s_lshr_b32 s59, s59, 5
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_lshl_b32 s54, s42, 2
	s_mul_i32 s95, s42, s57
	s_mul_hi_u32 s96, s42, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mov_b64 s[64:65], s[4:5]
	s_mul_i32 s45, s52, 1
	s_add_co_u32 s66, s4, s45
	s_add_co_ci_u32 s67, s5, 0
	s_mul_i32 s45, s52, 2
	s_add_co_u32 s68, s4, s45
	s_add_co_ci_u32 s69, s5, 0
	s_mul_i32 s45, s52, 3
	s_add_co_u32 s70, s4, s45
	s_add_co_ci_u32 s71, s5, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	s_mov_b32 s51, 0
	s_cmp_lg_u32 s56, 0
	s_cbranch_scc1 .Lb16x4r4_scalar
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16x4r4_fold
	.Lb16x4r4_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	global_load_b128 v[24:27], v2, s[64:65]
	global_load_b128 v[28:31], v2, s[64:65] offset:16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r4_gx_done
	global_load_b128 v[32:35], v2, s[66:67]
	global_load_b128 v[36:39], v2, s[66:67] offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r4_gx_done
	global_load_b128 v[40:43], v2, s[68:69]
	global_load_b128 v[44:47], v2, s[68:69] offset:16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r4_gx_done
	global_load_b128 v[48:51], v2, s[70:71]
	global_load_b128 v[52:55], v2, s[70:71] offset:16
	.Lb16x4r4_gx_done:
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v56, v24, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v57, v25, v17
	v_fmac_f32_e32 v58, v26, v18
	v_fmac_f32_e32 v59, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r4_gc_done
	v_fmac_f32_e32 v60, v32, v16
	v_fmac_f32_e32 v61, v33, v17
	v_fmac_f32_e32 v62, v34, v18
	v_fmac_f32_e32 v63, v35, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v36, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v37, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v38, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v39, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r4_gc_done
	v_fmac_f32_e32 v64, v40, v16
	v_fmac_f32_e32 v65, v41, v17
	v_fmac_f32_e32 v66, v42, v18
	v_fmac_f32_e32 v67, v43, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v44, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v45, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v46, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v47, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r4_gc_done
	v_fmac_f32_e32 v68, v48, v16
	v_fmac_f32_e32 v69, v49, v17
	v_fmac_f32_e32 v70, v50, v18
	v_fmac_f32_e32 v71, v51, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v52, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v53, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v54, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v55, v23
	.Lb16x4r4_gc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16x4r4_group
	s_branch .Lb16x4r4_fold
	.Lb16x4r4_scalar:
	.Lb16x4r4_tile32:
	s_lshl_b32 s49, s51, 6
	s_lshl_b32 s50, s51, 7
	s_lshl_b32 s48, s51, 5
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s47, s48
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 32
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_u16 v16, v1, s[2:3]
	global_load_b32 v24, v2, s[64:65]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r4_sx_done
	global_load_b32 v32, v2, s[66:67]
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r4_sx_done
	global_load_b32 v40, v2, s[68:69]
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r4_sx_done
	global_load_b32 v48, v2, s[70:71]
	.Lb16x4r4_sx_done:
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v56, v24, v16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r4_sc_done
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v60, v32, v16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r4_sc_done
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v64, v40, v16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r4_sc_done
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v48, v16
	.Lb16x4r4_sc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s59
	s_cbranch_scc1 .Lb16x4r4_tile32
	.Lb16x4r4_fold:
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v65
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v69
	ds_swizzle_b32 v24, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v71 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v27
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v59
	ds_bpermute_b32 v25, v3, v63
	ds_bpermute_b32 v26, v3, v67
	ds_bpermute_b32 v27, v3, v71
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v71, v71, v27
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v59
	ds_bpermute_b32 v25, v3, v63
	ds_bpermute_b32 v26, v3, v67
	ds_bpermute_b32 v27, v3, v71
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v71, v71, v27
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v59
	ds_bpermute_b32 v25, v3, v63
	ds_bpermute_b32 v26, v3, v67
	ds_bpermute_b32 v27, v3, v71
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v71, v71, v27
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v59
	ds_bpermute_b32 v25, v3, v63
	ds_bpermute_b32 v26, v3, v67
	ds_bpermute_b32 v27, v3, v71
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v71, v71, v27
	v_cmpx_eq_u32_e32 0, v4
	s_mul_i32 s55, s53, 0
	s_add_co_i32 s55, s55, s54
	v_mov_b32_e32 v24, s55
	global_store_b32 v24, v59, s[6:7]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r4_stored
	s_mul_i32 s55, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v25, s55
	global_store_b32 v25, v63, s[6:7]
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r4_stored
	s_mul_i32 s55, s53, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v26, s55
	global_store_b32 v26, v67, s[6:7]
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r4_stored
	s_mul_i32 s55, s53, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v27, s55
	global_store_b32 v27, v71, s[6:7]
	.Lb16x4r4_stored:
	s_wait_storecnt 0x0
	.Lb16x4r4_end:
	s_endpgm
.Lgemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201_end:
.size gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201, .Lgemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201_end-gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
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
	.amdhsa_next_free_vgpr 72
	.amdhsa_next_free_sgpr 97
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201_end-gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201
.globl gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201,@function
gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b32 s32, s30
	s_mov_b32 s33, s31
	s_mov_b64 s[4:5], s[16:17]
	s_mov_b32 s36, s26
	s_add_co_i32 s37, s36, s27
	s_add_co_i32 s38, s37, s28
	s_add_co_i32 s39, s38, s29
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lb16x4r8_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16x4r8_end
	s_cmp_gt_u32 s33, 8
	s_cbranch_scc1 .Lb16x4r8_end
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[2:3], s[12:13], s[14:15]
	s_cselect_b64 s[6:7], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[2:3], s[10:11], s[2:3]
	s_cselect_b64 s[6:7], s[20:21], s[6:7]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[2:3], s[8:9], s[2:3]
	s_cselect_b64 s[6:7], s[18:19], s[6:7]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_co_i32 s42, s40, s35
	s_and_b32 s56, s32, 7
	s_mov_b32 s58, s32
	s_add_co_i32 s44, s32, 0xff
	s_lshr_b32 s44, s44, 8
	s_add_co_i32 s59, s32, 31
	s_lshr_b32 s59, s59, 5
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_lshl_b32 s54, s42, 2
	s_mul_i32 s95, s42, s57
	s_mul_hi_u32 s96, s42, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mov_b64 s[64:65], s[4:5]
	s_mul_i32 s45, s52, 1
	s_add_co_u32 s66, s4, s45
	s_add_co_ci_u32 s67, s5, 0
	s_mul_i32 s45, s52, 2
	s_add_co_u32 s68, s4, s45
	s_add_co_ci_u32 s69, s5, 0
	s_mul_i32 s45, s52, 3
	s_add_co_u32 s70, s4, s45
	s_add_co_ci_u32 s71, s5, 0
	s_mul_i32 s45, s52, 4
	s_add_co_u32 s72, s4, s45
	s_add_co_ci_u32 s73, s5, 0
	s_mul_i32 s45, s52, 5
	s_add_co_u32 s74, s4, s45
	s_add_co_ci_u32 s75, s5, 0
	s_mul_i32 s45, s52, 6
	s_add_co_u32 s76, s4, s45
	s_add_co_ci_u32 s77, s5, 0
	s_mul_i32 s45, s52, 7
	s_add_co_u32 s78, s4, s45
	s_add_co_ci_u32 s79, s5, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	s_mov_b32 s51, 0
	s_cmp_lg_u32 s56, 0
	s_cbranch_scc1 .Lb16x4r8_scalar
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16x4r8_fold
	.Lb16x4r8_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	global_load_b128 v[24:27], v2, s[64:65]
	global_load_b128 v[28:31], v2, s[64:65] offset:16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r8_gx_done
	global_load_b128 v[32:35], v2, s[66:67]
	global_load_b128 v[36:39], v2, s[66:67] offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r8_gx_done
	global_load_b128 v[40:43], v2, s[68:69]
	global_load_b128 v[44:47], v2, s[68:69] offset:16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r8_gx_done
	global_load_b128 v[48:51], v2, s[70:71]
	global_load_b128 v[52:55], v2, s[70:71] offset:16
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r8_gx_done
	global_load_b128 v[56:59], v2, s[72:73]
	global_load_b128 v[60:63], v2, s[72:73] offset:16
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4r8_gx_done
	global_load_b128 v[64:67], v2, s[74:75]
	global_load_b128 v[68:71], v2, s[74:75] offset:16
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4r8_gx_done
	global_load_b128 v[72:75], v2, s[76:77]
	global_load_b128 v[76:79], v2, s[76:77] offset:16
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4r8_gx_done
	global_load_b128 v[80:83], v2, s[78:79]
	global_load_b128 v[84:87], v2, s[78:79] offset:16
	.Lb16x4r8_gx_done:
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v24, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v89, v25, v17
	v_fmac_f32_e32 v90, v26, v18
	v_fmac_f32_e32 v91, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r8_gc_done
	v_fmac_f32_e32 v92, v32, v16
	v_fmac_f32_e32 v93, v33, v17
	v_fmac_f32_e32 v94, v34, v18
	v_fmac_f32_e32 v95, v35, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v36, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v37, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v38, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v39, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r8_gc_done
	v_fmac_f32_e32 v96, v40, v16
	v_fmac_f32_e32 v97, v41, v17
	v_fmac_f32_e32 v98, v42, v18
	v_fmac_f32_e32 v99, v43, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v44, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v45, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v46, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v47, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r8_gc_done
	v_fmac_f32_e32 v100, v48, v16
	v_fmac_f32_e32 v101, v49, v17
	v_fmac_f32_e32 v102, v50, v18
	v_fmac_f32_e32 v103, v51, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v52, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v53, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v54, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v55, v23
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r8_gc_done
	v_fmac_f32_e32 v104, v56, v16
	v_fmac_f32_e32 v105, v57, v17
	v_fmac_f32_e32 v106, v58, v18
	v_fmac_f32_e32 v107, v59, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v60, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v61, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v62, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v63, v23
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4r8_gc_done
	v_fmac_f32_e32 v108, v64, v16
	v_fmac_f32_e32 v109, v65, v17
	v_fmac_f32_e32 v110, v66, v18
	v_fmac_f32_e32 v111, v67, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v68, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v69, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v70, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v71, v23
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4r8_gc_done
	v_fmac_f32_e32 v112, v72, v16
	v_fmac_f32_e32 v113, v73, v17
	v_fmac_f32_e32 v114, v74, v18
	v_fmac_f32_e32 v115, v75, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v76, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v77, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v78, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v79, v23
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4r8_gc_done
	v_fmac_f32_e32 v116, v80, v16
	v_fmac_f32_e32 v117, v81, v17
	v_fmac_f32_e32 v118, v82, v18
	v_fmac_f32_e32 v119, v83, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v84, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v85, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v86, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v87, v23
	.Lb16x4r8_gc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16x4r8_group
	s_branch .Lb16x4r8_fold
	.Lb16x4r8_scalar:
	.Lb16x4r8_tile32:
	s_lshl_b32 s49, s51, 6
	s_lshl_b32 s50, s51, 7
	s_lshl_b32 s48, s51, 5
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s47, s48
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 32
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_u16 v16, v1, s[2:3]
	global_load_b32 v24, v2, s[64:65]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r8_sx_done
	global_load_b32 v32, v2, s[66:67]
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r8_sx_done
	global_load_b32 v40, v2, s[68:69]
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r8_sx_done
	global_load_b32 v48, v2, s[70:71]
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r8_sx_done
	global_load_b32 v56, v2, s[72:73]
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4r8_sx_done
	global_load_b32 v64, v2, s[74:75]
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4r8_sx_done
	global_load_b32 v72, v2, s[76:77]
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4r8_sx_done
	global_load_b32 v80, v2, s[78:79]
	.Lb16x4r8_sx_done:
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v24, v16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r8_sc_done
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v92, v32, v16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r8_sc_done
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v96, v40, v16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r8_sc_done
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v48, v16
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r8_sc_done
	v_fmac_f32_e32 v104, v56, v16
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4r8_sc_done
	v_fmac_f32_e32 v108, v64, v16
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4r8_sc_done
	v_fmac_f32_e32 v112, v72, v16
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4r8_sc_done
	v_fmac_f32_e32 v116, v80, v16
	.Lb16x4r8_sc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s59
	s_cbranch_scc1 .Lb16x4r8_tile32
	.Lb16x4r8_fold:
	v_add_f32_e32 v89, v89, v88
	v_add_f32_e32 v91, v91, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v89
	v_add_f32_e32 v93, v93, v92
	v_add_f32_e32 v95, v95, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v93
	v_add_f32_e32 v97, v97, v96
	v_add_f32_e32 v99, v99, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v97
	v_add_f32_e32 v101, v101, v100
	v_add_f32_e32 v103, v103, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v101
	v_add_f32_e32 v105, v105, v104
	v_add_f32_e32 v107, v107, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v107, v107, v105
	v_add_f32_e32 v109, v109, v108
	v_add_f32_e32 v111, v111, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v111, v111, v109
	v_add_f32_e32 v113, v113, v112
	v_add_f32_e32 v115, v115, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v115, v115, v113
	v_add_f32_e32 v117, v117, v116
	v_add_f32_e32 v119, v119, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v119, v119, v117
	ds_swizzle_b32 v24, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v95 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v107 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v111 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v119 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	ds_bpermute_b32 v25, v3, v95
	ds_bpermute_b32 v26, v3, v99
	ds_bpermute_b32 v27, v3, v103
	ds_bpermute_b32 v28, v3, v107
	ds_bpermute_b32 v29, v3, v111
	ds_bpermute_b32 v30, v3, v115
	ds_bpermute_b32 v31, v3, v119
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	ds_bpermute_b32 v25, v3, v95
	ds_bpermute_b32 v26, v3, v99
	ds_bpermute_b32 v27, v3, v103
	ds_bpermute_b32 v28, v3, v107
	ds_bpermute_b32 v29, v3, v111
	ds_bpermute_b32 v30, v3, v115
	ds_bpermute_b32 v31, v3, v119
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	ds_bpermute_b32 v25, v3, v95
	ds_bpermute_b32 v26, v3, v99
	ds_bpermute_b32 v27, v3, v103
	ds_bpermute_b32 v28, v3, v107
	ds_bpermute_b32 v29, v3, v111
	ds_bpermute_b32 v30, v3, v115
	ds_bpermute_b32 v31, v3, v119
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	ds_bpermute_b32 v25, v3, v95
	ds_bpermute_b32 v26, v3, v99
	ds_bpermute_b32 v27, v3, v103
	ds_bpermute_b32 v28, v3, v107
	ds_bpermute_b32 v29, v3, v111
	ds_bpermute_b32 v30, v3, v115
	ds_bpermute_b32 v31, v3, v119
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmpx_eq_u32_e32 0, v4
	s_mul_i32 s55, s53, 0
	s_add_co_i32 s55, s55, s54
	v_mov_b32_e32 v24, s55
	global_store_b32 v24, v91, s[6:7]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r8_stored
	s_mul_i32 s55, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v25, s55
	global_store_b32 v25, v95, s[6:7]
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r8_stored
	s_mul_i32 s55, s53, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v26, s55
	global_store_b32 v26, v99, s[6:7]
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r8_stored
	s_mul_i32 s55, s53, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v27, s55
	global_store_b32 v27, v103, s[6:7]
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r8_stored
	s_mul_i32 s55, s53, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v28, s55
	global_store_b32 v28, v107, s[6:7]
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4r8_stored
	s_mul_i32 s55, s53, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v29, s55
	global_store_b32 v29, v111, s[6:7]
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4r8_stored
	s_mul_i32 s55, s53, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v30, s55
	global_store_b32 v30, v115, s[6:7]
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4r8_stored
	s_mul_i32 s55, s53, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v31, s55
	global_store_b32 v31, v119, s[6:7]
	.Lb16x4r8_stored:
	s_wait_storecnt 0x0
	.Lb16x4r8_end:
	s_endpgm
.Lgemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201_end:
.size gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201, .Lgemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201_end-gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
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
	.amdhsa_next_free_vgpr 120
	.amdhsa_next_free_sgpr 97
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201_end-gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_k4_c1_pm_gfx1201
.globl gemv_bf16_xf32_k4_c1_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_k4_c1_pm_gfx1201,@function
gemv_bf16_xf32_k4_c1_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b32 s18, s[0:1], 0x28
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b64 s[2:3], s[8:9]
	s_mov_b64 s[4:5], s[10:11]
	s_mov_b64 s[6:7], s[12:13]
	s_mov_b32 s34, s14
	s_mov_b32 s32, s15
	s_mov_b32 s33, s18
	s_cmp_ge_u32 s40, s34
	s_cbranch_scc1 .Lb16k4c1_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16k4c1_end
	s_cmp_gt_u32 s33, 1
	s_cbranch_scc1 .Lb16k4c1_end
	s_and_b32 s45, s32, 31
	s_cmp_lg_u32 s45, 0
	s_cbranch_scc1 .Lb16k4c1_end
	s_mov_b32 s60, s41
	s_mov_b32 s98, s40
	s_lshr_b32 s88, s32, 2
	s_mov_b32 s58, s88
	s_add_co_i32 s44, s88, 0xff
	s_lshr_b32 s44, s44, 8
	s_lshl_b32 s89, s88, 2
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_mul_i32 s95, s98, s57
	s_mul_hi_u32 s96, s98, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s97, s60, s88
	s_lshl_b32 s97, s97, 1
	s_add_co_u32 s2, s2, s97
	s_add_co_ci_u32 s3, s3, 0
	s_mov_b32 s45, s89
	s_mul_i32 s45, s60, s45
	s_add_co_u32 s4, s4, s45
	s_add_co_ci_u32 s5, s5, 0
	s_mov_b64 s[64:65], s[4:5]
	s_lshl_b32 s45, s60, 2
	v_mov_b32_e32 v9, s45
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	s_mov_b32 s51, 0
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16k4c1_fold
	.Lb16k4c1_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	global_load_b128 v[24:27], v2, s[64:65]
	global_load_b128 v[28:31], v2, s[64:65] offset:16
	.Lb16k4c1_gx_done:
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v32, v24, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v33, v25, v17
	v_fmac_f32_e32 v34, v26, v18
	v_fmac_f32_e32 v35, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v32, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v33, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v34, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v35, v31, v23
	.Lb16k4c1_gc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16k4c1_group
	.Lb16k4c1_fold:
	v_add_f32_e32 v33, v33, v32
	v_add_f32_e32 v35, v35, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v33
	ds_swizzle_b32 v24, v35 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v35, v35, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v35, v35, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v35, v35, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v35, v35, v24
	v_cmpx_eq_u32_e32 0, v4
	ds_store_b32 v9, v35
	.Lb16k4c1_pstored:
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4c1_pdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s46, s41, 0
	s_mov_b32 s93, s46
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4c1_iskip0
	s_lshl_b32 s45, s93, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4c1_item0_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4c1_item0_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4c1_iskip0:
	.Lb16k4c1_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_c1_pm_gfx1201_end:
.size gemv_bf16_xf32_k4_c1_pm_gfx1201, .Lgemv_bf16_xf32_k4_c1_pm_gfx1201_end-gemv_bf16_xf32_k4_c1_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_c1_pm_gfx1201
	.amdhsa_group_segment_fixed_size 16
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
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
	.amdhsa_next_free_vgpr 36
	.amdhsa_next_free_sgpr 99
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_c1_pm_gfx1201_end-gemv_bf16_xf32_k4_c1_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201
.globl gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201,@function
gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b32 s18, s[0:1], 0x28
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b64 s[2:3], s[8:9]
	s_mov_b64 s[4:5], s[10:11]
	s_mov_b64 s[6:7], s[12:13]
	s_mov_b32 s34, s14
	s_mov_b32 s32, s15
	s_mov_b32 s33, s18
	s_cmp_ge_u32 s40, s34
	s_cbranch_scc1 .Lb16k4r4_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16k4r4_end
	s_cmp_gt_u32 s33, 4
	s_cbranch_scc1 .Lb16k4r4_end
	s_and_b32 s45, s32, 31
	s_cmp_lg_u32 s45, 0
	s_cbranch_scc1 .Lb16k4r4_end
	s_mov_b32 s60, s41
	s_mov_b32 s98, s40
	s_lshr_b32 s88, s32, 2
	s_mov_b32 s58, s88
	s_add_co_i32 s44, s88, 0xff
	s_lshr_b32 s44, s44, 8
	s_lshl_b32 s89, s88, 2
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_mul_i32 s95, s98, s57
	s_mul_hi_u32 s96, s98, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s97, s60, s88
	s_lshl_b32 s97, s97, 1
	s_add_co_u32 s2, s2, s97
	s_add_co_ci_u32 s3, s3, 0
	s_mov_b32 s45, s89
	s_mul_i32 s45, s60, s45
	s_add_co_u32 s4, s4, s45
	s_add_co_ci_u32 s5, s5, 0
	s_mov_b64 s[64:65], s[4:5]
	s_mul_i32 s45, s52, 1
	s_add_co_u32 s66, s4, s45
	s_add_co_ci_u32 s67, s5, 0
	s_mul_i32 s45, s52, 2
	s_add_co_u32 s68, s4, s45
	s_add_co_ci_u32 s69, s5, 0
	s_mul_i32 s45, s52, 3
	s_add_co_u32 s70, s4, s45
	s_add_co_ci_u32 s71, s5, 0
	s_lshl_b32 s45, s60, 2
	v_mov_b32_e32 v9, s45
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	s_mov_b32 s51, 0
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16k4r4_fold
	.Lb16k4r4_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	global_load_b128 v[24:27], v2, s[64:65]
	global_load_b128 v[28:31], v2, s[64:65] offset:16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r4_gx_done
	global_load_b128 v[32:35], v2, s[66:67]
	global_load_b128 v[36:39], v2, s[66:67] offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r4_gx_done
	global_load_b128 v[40:43], v2, s[68:69]
	global_load_b128 v[44:47], v2, s[68:69] offset:16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r4_gx_done
	global_load_b128 v[48:51], v2, s[70:71]
	global_load_b128 v[52:55], v2, s[70:71] offset:16
	.Lb16k4r4_gx_done:
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v56, v24, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v57, v25, v17
	v_fmac_f32_e32 v58, v26, v18
	v_fmac_f32_e32 v59, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r4_gc_done
	v_fmac_f32_e32 v60, v32, v16
	v_fmac_f32_e32 v61, v33, v17
	v_fmac_f32_e32 v62, v34, v18
	v_fmac_f32_e32 v63, v35, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v36, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v37, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v38, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v39, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r4_gc_done
	v_fmac_f32_e32 v64, v40, v16
	v_fmac_f32_e32 v65, v41, v17
	v_fmac_f32_e32 v66, v42, v18
	v_fmac_f32_e32 v67, v43, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v44, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v45, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v46, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v47, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r4_gc_done
	v_fmac_f32_e32 v68, v48, v16
	v_fmac_f32_e32 v69, v49, v17
	v_fmac_f32_e32 v70, v50, v18
	v_fmac_f32_e32 v71, v51, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v52, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v53, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v54, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v55, v23
	.Lb16k4r4_gc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16k4r4_group
	.Lb16k4r4_fold:
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v65
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v69
	ds_swizzle_b32 v24, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v71 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v27
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v59
	ds_bpermute_b32 v25, v3, v63
	ds_bpermute_b32 v26, v3, v67
	ds_bpermute_b32 v27, v3, v71
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v71, v71, v27
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v59
	ds_bpermute_b32 v25, v3, v63
	ds_bpermute_b32 v26, v3, v67
	ds_bpermute_b32 v27, v3, v71
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v71, v71, v27
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v59
	ds_bpermute_b32 v25, v3, v63
	ds_bpermute_b32 v26, v3, v67
	ds_bpermute_b32 v27, v3, v71
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v71, v71, v27
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v59
	ds_bpermute_b32 v25, v3, v63
	ds_bpermute_b32 v26, v3, v67
	ds_bpermute_b32 v27, v3, v71
	s_wait_dscnt 0x3
	v_add_f32_e32 v59, v59, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v63, v63, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v67, v67, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v71, v71, v27
	v_cmpx_eq_u32_e32 0, v4
	ds_store_b32 v9, v59
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r4_pstored
	ds_store_b32 v9, v63 offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r4_pstored
	ds_store_b32 v9, v67 offset:32
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r4_pstored
	ds_store_b32 v9, v71 offset:48
	.Lb16k4r4_pstored:
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r4_pdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s46, s41, 0
	s_mov_b32 s93, s46
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4r4_iskip0
	s_lshl_b32 s45, s93, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4r4_item0_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4r4_item0_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r4_iskip0:
	.Lb16k4r4_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201_end:
.size gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201, .Lgemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201_end-gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201
	.amdhsa_group_segment_fixed_size 64
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
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
	.amdhsa_next_free_vgpr 72
	.amdhsa_next_free_sgpr 99
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201_end-gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201
.globl gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201,@function
gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b32 s18, s[0:1], 0x28
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b64 s[2:3], s[8:9]
	s_mov_b64 s[4:5], s[10:11]
	s_mov_b64 s[6:7], s[12:13]
	s_mov_b32 s34, s14
	s_mov_b32 s32, s15
	s_mov_b32 s33, s18
	s_cmp_ge_u32 s40, s34
	s_cbranch_scc1 .Lb16k4r8_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16k4r8_end
	s_cmp_gt_u32 s33, 8
	s_cbranch_scc1 .Lb16k4r8_end
	s_and_b32 s45, s32, 31
	s_cmp_lg_u32 s45, 0
	s_cbranch_scc1 .Lb16k4r8_end
	s_mov_b32 s60, s41
	s_mov_b32 s98, s40
	s_lshr_b32 s88, s32, 2
	s_mov_b32 s58, s88
	s_add_co_i32 s44, s88, 0xff
	s_lshr_b32 s44, s44, 8
	s_lshl_b32 s89, s88, 2
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_mul_i32 s95, s98, s57
	s_mul_hi_u32 s96, s98, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s97, s60, s88
	s_lshl_b32 s97, s97, 1
	s_add_co_u32 s2, s2, s97
	s_add_co_ci_u32 s3, s3, 0
	s_mov_b32 s45, s89
	s_mul_i32 s45, s60, s45
	s_add_co_u32 s4, s4, s45
	s_add_co_ci_u32 s5, s5, 0
	s_mov_b64 s[64:65], s[4:5]
	s_mul_i32 s45, s52, 1
	s_add_co_u32 s66, s4, s45
	s_add_co_ci_u32 s67, s5, 0
	s_mul_i32 s45, s52, 2
	s_add_co_u32 s68, s4, s45
	s_add_co_ci_u32 s69, s5, 0
	s_mul_i32 s45, s52, 3
	s_add_co_u32 s70, s4, s45
	s_add_co_ci_u32 s71, s5, 0
	s_mul_i32 s45, s52, 4
	s_add_co_u32 s72, s4, s45
	s_add_co_ci_u32 s73, s5, 0
	s_mul_i32 s45, s52, 5
	s_add_co_u32 s74, s4, s45
	s_add_co_ci_u32 s75, s5, 0
	s_mul_i32 s45, s52, 6
	s_add_co_u32 s76, s4, s45
	s_add_co_ci_u32 s77, s5, 0
	s_mul_i32 s45, s52, 7
	s_add_co_u32 s78, s4, s45
	s_add_co_ci_u32 s79, s5, 0
	s_lshl_b32 s45, s60, 2
	v_mov_b32_e32 v9, s45
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	s_mov_b32 s51, 0
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16k4r8_fold
	.Lb16k4r8_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	global_load_b128 v[24:27], v2, s[64:65]
	global_load_b128 v[28:31], v2, s[64:65] offset:16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r8_gx_done
	global_load_b128 v[32:35], v2, s[66:67]
	global_load_b128 v[36:39], v2, s[66:67] offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r8_gx_done
	global_load_b128 v[40:43], v2, s[68:69]
	global_load_b128 v[44:47], v2, s[68:69] offset:16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r8_gx_done
	global_load_b128 v[48:51], v2, s[70:71]
	global_load_b128 v[52:55], v2, s[70:71] offset:16
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4r8_gx_done
	global_load_b128 v[56:59], v2, s[72:73]
	global_load_b128 v[60:63], v2, s[72:73] offset:16
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4r8_gx_done
	global_load_b128 v[64:67], v2, s[74:75]
	global_load_b128 v[68:71], v2, s[74:75] offset:16
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4r8_gx_done
	global_load_b128 v[72:75], v2, s[76:77]
	global_load_b128 v[76:79], v2, s[76:77] offset:16
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4r8_gx_done
	global_load_b128 v[80:83], v2, s[78:79]
	global_load_b128 v[84:87], v2, s[78:79] offset:16
	.Lb16k4r8_gx_done:
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v24, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v89, v25, v17
	v_fmac_f32_e32 v90, v26, v18
	v_fmac_f32_e32 v91, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r8_gc_done
	v_fmac_f32_e32 v92, v32, v16
	v_fmac_f32_e32 v93, v33, v17
	v_fmac_f32_e32 v94, v34, v18
	v_fmac_f32_e32 v95, v35, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v36, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v37, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v38, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v39, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r8_gc_done
	v_fmac_f32_e32 v96, v40, v16
	v_fmac_f32_e32 v97, v41, v17
	v_fmac_f32_e32 v98, v42, v18
	v_fmac_f32_e32 v99, v43, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v44, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v45, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v46, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v47, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r8_gc_done
	v_fmac_f32_e32 v100, v48, v16
	v_fmac_f32_e32 v101, v49, v17
	v_fmac_f32_e32 v102, v50, v18
	v_fmac_f32_e32 v103, v51, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v52, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v53, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v54, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v55, v23
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4r8_gc_done
	v_fmac_f32_e32 v104, v56, v16
	v_fmac_f32_e32 v105, v57, v17
	v_fmac_f32_e32 v106, v58, v18
	v_fmac_f32_e32 v107, v59, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v60, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v61, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v62, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v63, v23
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4r8_gc_done
	v_fmac_f32_e32 v108, v64, v16
	v_fmac_f32_e32 v109, v65, v17
	v_fmac_f32_e32 v110, v66, v18
	v_fmac_f32_e32 v111, v67, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v68, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v69, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v70, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v71, v23
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4r8_gc_done
	v_fmac_f32_e32 v112, v72, v16
	v_fmac_f32_e32 v113, v73, v17
	v_fmac_f32_e32 v114, v74, v18
	v_fmac_f32_e32 v115, v75, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v76, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v77, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v78, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v79, v23
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4r8_gc_done
	v_fmac_f32_e32 v116, v80, v16
	v_fmac_f32_e32 v117, v81, v17
	v_fmac_f32_e32 v118, v82, v18
	v_fmac_f32_e32 v119, v83, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v84, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v85, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v86, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v87, v23
	.Lb16k4r8_gc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16k4r8_group
	.Lb16k4r8_fold:
	v_add_f32_e32 v89, v89, v88
	v_add_f32_e32 v91, v91, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v89
	v_add_f32_e32 v93, v93, v92
	v_add_f32_e32 v95, v95, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v93
	v_add_f32_e32 v97, v97, v96
	v_add_f32_e32 v99, v99, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v97
	v_add_f32_e32 v101, v101, v100
	v_add_f32_e32 v103, v103, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v101
	v_add_f32_e32 v105, v105, v104
	v_add_f32_e32 v107, v107, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v107, v107, v105
	v_add_f32_e32 v109, v109, v108
	v_add_f32_e32 v111, v111, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v111, v111, v109
	v_add_f32_e32 v113, v113, v112
	v_add_f32_e32 v115, v115, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v115, v115, v113
	v_add_f32_e32 v117, v117, v116
	v_add_f32_e32 v119, v119, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v119, v119, v117
	ds_swizzle_b32 v24, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v95 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v107 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v111 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v119 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	ds_bpermute_b32 v25, v3, v95
	ds_bpermute_b32 v26, v3, v99
	ds_bpermute_b32 v27, v3, v103
	ds_bpermute_b32 v28, v3, v107
	ds_bpermute_b32 v29, v3, v111
	ds_bpermute_b32 v30, v3, v115
	ds_bpermute_b32 v31, v3, v119
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	ds_bpermute_b32 v25, v3, v95
	ds_bpermute_b32 v26, v3, v99
	ds_bpermute_b32 v27, v3, v103
	ds_bpermute_b32 v28, v3, v107
	ds_bpermute_b32 v29, v3, v111
	ds_bpermute_b32 v30, v3, v115
	ds_bpermute_b32 v31, v3, v119
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	ds_bpermute_b32 v25, v3, v95
	ds_bpermute_b32 v26, v3, v99
	ds_bpermute_b32 v27, v3, v103
	ds_bpermute_b32 v28, v3, v107
	ds_bpermute_b32 v29, v3, v111
	ds_bpermute_b32 v30, v3, v115
	ds_bpermute_b32 v31, v3, v119
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	ds_bpermute_b32 v25, v3, v95
	ds_bpermute_b32 v26, v3, v99
	ds_bpermute_b32 v27, v3, v103
	ds_bpermute_b32 v28, v3, v107
	ds_bpermute_b32 v29, v3, v111
	ds_bpermute_b32 v30, v3, v115
	ds_bpermute_b32 v31, v3, v119
	s_wait_dscnt 0x7
	v_add_f32_e32 v91, v91, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v95, v95, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v99, v99, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v107, v107, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v111, v111, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v115, v115, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v119, v119, v31
	v_cmpx_eq_u32_e32 0, v4
	ds_store_b32 v9, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r8_pstored
	ds_store_b32 v9, v95 offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r8_pstored
	ds_store_b32 v9, v99 offset:32
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r8_pstored
	ds_store_b32 v9, v103 offset:48
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4r8_pstored
	ds_store_b32 v9, v107 offset:64
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4r8_pstored
	ds_store_b32 v9, v111 offset:80
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4r8_pstored
	ds_store_b32 v9, v115 offset:96
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4r8_pstored
	ds_store_b32 v9, v119 offset:112
	.Lb16k4r8_pstored:
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8_pdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s46, s41, 0
	s_mov_b32 s93, s46
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4r8_iskip0
	s_lshl_b32 s45, s93, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4r8_item0_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4r8_item0_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8_iskip0:
	s_add_co_i32 s46, s41, 4
	s_mov_b32 s93, s46
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4r8_iskip1
	s_lshl_b32 s45, s93, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4r8_item1_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4r8_item1_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8_iskip1:
	.Lb16k4r8_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201_end:
.size gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201, .Lgemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201_end-gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201
	.amdhsa_group_segment_fixed_size 128
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
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
	.amdhsa_next_free_vgpr 120
	.amdhsa_next_free_sgpr 99
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201_end-gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201
.globl gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201,@function
gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b32 s32, s30
	s_mov_b32 s33, s31
	s_mov_b64 s[4:5], s[16:17]
	s_add_co_i32 s36, s26, 7
	s_lshr_b32 s36, s36, 3
	s_add_co_i32 s45, s27, 7
	s_lshr_b32 s45, s45, 3
	s_add_co_i32 s37, s36, s45
	s_add_co_i32 s45, s28, 7
	s_lshr_b32 s45, s45, 3
	s_add_co_i32 s38, s37, s45
	s_add_co_i32 s45, s29, 7
	s_lshr_b32 s45, s45, 3
	s_add_co_i32 s39, s38, s45
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lb16x4r4w_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16x4r4w_end
	s_cmp_gt_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r4w_end
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[2:3], s[12:13], s[14:15]
	s_cselect_b64 s[6:7], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[2:3], s[10:11], s[2:3]
	s_cselect_b64 s[6:7], s[20:21], s[6:7]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[2:3], s[8:9], s[2:3]
	s_cselect_b64 s[6:7], s[18:19], s[6:7]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_co_i32 s42, s40, s35
	s_lshl_b32 s42, s42, 3
	s_add_co_i32 s42, s42, s41
	s_cmp_lt_i32 s42, s34
	s_cselect_b32 s43, 1, 0
	s_and_b32 s56, s32, 7
	s_mov_b32 s58, s32
	s_add_co_i32 s44, s32, 0xff
	s_lshr_b32 s44, s44, 8
	s_add_co_i32 s59, s32, 31
	s_lshr_b32 s59, s59, 5
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_lshl_b32 s54, s42, 2
	s_mul_i32 s95, s42, s57
	s_mul_hi_u32 s96, s42, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s62, s41, s52
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 2, v4
	s_lshl_b32 s45, s41, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s45, v7
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s51, 0
	s_cmp_lg_u32 s56, 0
	s_cbranch_scc1 .Lb16x4r4w_scalar
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16x4r4w_fold
	.Lb16x4r4w_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r4w_vw_done
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lb16x4r4w_vw_done:
	s_cmp_ge_u32 s41, s33
	s_cbranch_scc1 .Lb16x4r4w_vnofill
	v_add_nc_u32_e32 v5, s62, v2
	s_mov_b32 exec_lo, s47
	global_load_b128 v[24:27], v5, s[4:5]
	global_load_b128 v[28:31], v5, s[4:5] offset:16
	s_wait_loadcnt 0x1
	ds_store_b32 v6, v24
	ds_store_b32 v6, v25 offset:128
	ds_store_b32 v6, v26 offset:256
	ds_store_b32 v6, v27 offset:384
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v28 offset:512
	ds_store_b32 v6, v29 offset:640
	ds_store_b32 v6, v30 offset:768
	ds_store_b32 v6, v31 offset:896
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16x4r4w_vnofill:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r4w_vcdone
	s_mov_b32 exec_lo, s47
	ds_load_b32 v24, v7
	ds_load_b32 v25, v7 offset:128
	ds_load_b32 v26, v7 offset:256
	ds_load_b32 v27, v7 offset:384
	ds_load_b32 v28, v7 offset:512
	ds_load_b32 v29, v7 offset:640
	ds_load_b32 v30, v7 offset:768
	ds_load_b32 v31, v7 offset:896
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v32, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v33, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v34, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v35, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v32, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v33, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v34, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v35, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r4w_vcdone
	ds_load_b32 v24, v7 offset:1024
	ds_load_b32 v25, v7 offset:1152
	ds_load_b32 v26, v7 offset:1280
	ds_load_b32 v27, v7 offset:1408
	ds_load_b32 v28, v7 offset:1536
	ds_load_b32 v29, v7 offset:1664
	ds_load_b32 v30, v7 offset:1792
	ds_load_b32 v31, v7 offset:1920
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v36, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v37, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v38, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v39, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v36, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v37, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v38, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v39, v31, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r4w_vcdone
	ds_load_b32 v24, v7 offset:2048
	ds_load_b32 v25, v7 offset:2176
	ds_load_b32 v26, v7 offset:2304
	ds_load_b32 v27, v7 offset:2432
	ds_load_b32 v28, v7 offset:2560
	ds_load_b32 v29, v7 offset:2688
	ds_load_b32 v30, v7 offset:2816
	ds_load_b32 v31, v7 offset:2944
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v40, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v41, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v42, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v43, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r4w_vcdone
	ds_load_b32 v24, v7 offset:3072
	ds_load_b32 v25, v7 offset:3200
	ds_load_b32 v26, v7 offset:3328
	ds_load_b32 v27, v7 offset:3456
	ds_load_b32 v28, v7 offset:3584
	ds_load_b32 v29, v7 offset:3712
	ds_load_b32 v30, v7 offset:3840
	ds_load_b32 v31, v7 offset:3968
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v44, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v45, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v46, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v47, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	.Lb16x4r4w_vcdone:
	s_mov_b32 exec_lo, -1
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16x4r4w_group
	s_branch .Lb16x4r4w_fold
	.Lb16x4r4w_scalar:
	.Lb16x4r4w_tile32:
	s_lshl_b32 s49, s51, 6
	s_lshl_b32 s50, s51, 7
	s_lshl_b32 s48, s51, 5
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s47, s48
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 32
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r4w_sw_done
	s_mov_b32 exec_lo, s47
	global_load_u16 v16, v1, s[2:3]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lb16x4r4w_sw_done:
	s_cmp_ge_u32 s41, s33
	s_cbranch_scc1 .Lb16x4r4w_snofill
	v_add_nc_u32_e32 v5, s62, v2
	s_mov_b32 exec_lo, s47
	global_load_b32 v24, v5, s[4:5]
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v24
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16x4r4w_snofill:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r4w_scdone
	s_mov_b32 exec_lo, s47
	ds_load_b32 v24, v7
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v32, v24, v16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r4w_scdone
	ds_load_b32 v24, v7 offset:1024
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v36, v24, v16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r4w_scdone
	ds_load_b32 v24, v7 offset:2048
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v40, v24, v16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r4w_scdone
	ds_load_b32 v24, v7 offset:3072
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v44, v24, v16
	.Lb16x4r4w_scdone:
	s_mov_b32 exec_lo, -1
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s59
	s_cbranch_scc1 .Lb16x4r4w_tile32
	.Lb16x4r4w_fold:
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r4w_end
	v_add_f32_e32 v33, v33, v32
	v_add_f32_e32 v35, v35, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v33
	v_add_f32_e32 v37, v37, v36
	v_add_f32_e32 v39, v39, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v37
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	ds_swizzle_b32 v24, v35 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v39 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v27
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v27
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v27
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v27
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v27
	v_cmpx_eq_u32_e32 0, v4
	s_mul_i32 s55, s53, 0
	s_add_co_i32 s55, s55, s54
	v_mov_b32_e32 v24, s55
	global_store_b32 v24, v35, s[6:7]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r4w_stored
	s_mul_i32 s55, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v25, s55
	global_store_b32 v25, v39, s[6:7]
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r4w_stored
	s_mul_i32 s55, s53, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v26, s55
	global_store_b32 v26, v43, s[6:7]
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r4w_stored
	s_mul_i32 s55, s53, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v27, s55
	global_store_b32 v27, v47, s[6:7]
	.Lb16x4r4w_stored:
	s_wait_storecnt 0x0
	.Lb16x4r4w_end:
	s_endpgm
.Lgemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201_end:
.size gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201, .Lgemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201_end-gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201
	.amdhsa_group_segment_fixed_size 4096
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
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
	.amdhsa_next_free_vgpr 48
	.amdhsa_next_free_sgpr 97
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201_end-gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201
.globl gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201,@function
gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b32 s32, s30
	s_mov_b32 s33, s31
	s_mov_b64 s[4:5], s[16:17]
	s_add_co_i32 s36, s26, 7
	s_lshr_b32 s36, s36, 3
	s_add_co_i32 s45, s27, 7
	s_lshr_b32 s45, s45, 3
	s_add_co_i32 s37, s36, s45
	s_add_co_i32 s45, s28, 7
	s_lshr_b32 s45, s45, 3
	s_add_co_i32 s38, s37, s45
	s_add_co_i32 s45, s29, 7
	s_lshr_b32 s45, s45, 3
	s_add_co_i32 s39, s38, s45
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lb16x4r8w_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16x4r8w_end
	s_cmp_gt_u32 s33, 8
	s_cbranch_scc1 .Lb16x4r8w_end
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[2:3], s[12:13], s[14:15]
	s_cselect_b64 s[6:7], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[2:3], s[10:11], s[2:3]
	s_cselect_b64 s[6:7], s[20:21], s[6:7]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[2:3], s[8:9], s[2:3]
	s_cselect_b64 s[6:7], s[18:19], s[6:7]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_co_i32 s42, s40, s35
	s_lshl_b32 s42, s42, 3
	s_add_co_i32 s42, s42, s41
	s_cmp_lt_i32 s42, s34
	s_cselect_b32 s43, 1, 0
	s_and_b32 s56, s32, 7
	s_mov_b32 s58, s32
	s_add_co_i32 s44, s32, 0xff
	s_lshr_b32 s44, s44, 8
	s_add_co_i32 s59, s32, 31
	s_lshr_b32 s59, s59, 5
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_lshl_b32 s54, s42, 2
	s_mul_i32 s95, s42, s57
	s_mul_hi_u32 s96, s42, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s62, s41, s52
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 2, v4
	s_lshl_b32 s45, s41, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s45, v7
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	s_mov_b32 s51, 0
	s_cmp_lg_u32 s56, 0
	s_cbranch_scc1 .Lb16x4r8w_scalar
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16x4r8w_fold
	.Lb16x4r8w_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r8w_vw_done
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lb16x4r8w_vw_done:
	s_cmp_ge_u32 s41, s33
	s_cbranch_scc1 .Lb16x4r8w_vnofill
	v_add_nc_u32_e32 v5, s62, v2
	s_mov_b32 exec_lo, s47
	global_load_b128 v[24:27], v5, s[4:5]
	global_load_b128 v[28:31], v5, s[4:5] offset:16
	s_wait_loadcnt 0x1
	ds_store_b32 v6, v24
	ds_store_b32 v6, v25 offset:128
	ds_store_b32 v6, v26 offset:256
	ds_store_b32 v6, v27 offset:384
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v28 offset:512
	ds_store_b32 v6, v29 offset:640
	ds_store_b32 v6, v30 offset:768
	ds_store_b32 v6, v31 offset:896
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16x4r8w_vnofill:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r8w_vcdone
	s_mov_b32 exec_lo, s47
	ds_load_b32 v24, v7
	ds_load_b32 v25, v7 offset:128
	ds_load_b32 v26, v7 offset:256
	ds_load_b32 v27, v7 offset:384
	ds_load_b32 v28, v7 offset:512
	ds_load_b32 v29, v7 offset:640
	ds_load_b32 v30, v7 offset:768
	ds_load_b32 v31, v7 offset:896
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v32, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v33, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v34, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v35, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v32, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v33, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v34, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v35, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r8w_vcdone
	ds_load_b32 v24, v7 offset:1024
	ds_load_b32 v25, v7 offset:1152
	ds_load_b32 v26, v7 offset:1280
	ds_load_b32 v27, v7 offset:1408
	ds_load_b32 v28, v7 offset:1536
	ds_load_b32 v29, v7 offset:1664
	ds_load_b32 v30, v7 offset:1792
	ds_load_b32 v31, v7 offset:1920
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v36, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v37, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v38, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v39, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v36, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v37, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v38, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v39, v31, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r8w_vcdone
	ds_load_b32 v24, v7 offset:2048
	ds_load_b32 v25, v7 offset:2176
	ds_load_b32 v26, v7 offset:2304
	ds_load_b32 v27, v7 offset:2432
	ds_load_b32 v28, v7 offset:2560
	ds_load_b32 v29, v7 offset:2688
	ds_load_b32 v30, v7 offset:2816
	ds_load_b32 v31, v7 offset:2944
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v40, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v41, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v42, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v43, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r8w_vcdone
	ds_load_b32 v24, v7 offset:3072
	ds_load_b32 v25, v7 offset:3200
	ds_load_b32 v26, v7 offset:3328
	ds_load_b32 v27, v7 offset:3456
	ds_load_b32 v28, v7 offset:3584
	ds_load_b32 v29, v7 offset:3712
	ds_load_b32 v30, v7 offset:3840
	ds_load_b32 v31, v7 offset:3968
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v44, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v45, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v46, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v47, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r8w_vcdone
	ds_load_b32 v24, v7 offset:4096
	ds_load_b32 v25, v7 offset:4224
	ds_load_b32 v26, v7 offset:4352
	ds_load_b32 v27, v7 offset:4480
	ds_load_b32 v28, v7 offset:4608
	ds_load_b32 v29, v7 offset:4736
	ds_load_b32 v30, v7 offset:4864
	ds_load_b32 v31, v7 offset:4992
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v48, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v49, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v50, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v51, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4r8w_vcdone
	ds_load_b32 v24, v7 offset:5120
	ds_load_b32 v25, v7 offset:5248
	ds_load_b32 v26, v7 offset:5376
	ds_load_b32 v27, v7 offset:5504
	ds_load_b32 v28, v7 offset:5632
	ds_load_b32 v29, v7 offset:5760
	ds_load_b32 v30, v7 offset:5888
	ds_load_b32 v31, v7 offset:6016
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v52, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v53, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v54, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v55, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4r8w_vcdone
	ds_load_b32 v24, v7 offset:6144
	ds_load_b32 v25, v7 offset:6272
	ds_load_b32 v26, v7 offset:6400
	ds_load_b32 v27, v7 offset:6528
	ds_load_b32 v28, v7 offset:6656
	ds_load_b32 v29, v7 offset:6784
	ds_load_b32 v30, v7 offset:6912
	ds_load_b32 v31, v7 offset:7040
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v56, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v57, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v58, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v59, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4r8w_vcdone
	ds_load_b32 v24, v7 offset:7168
	ds_load_b32 v25, v7 offset:7296
	ds_load_b32 v26, v7 offset:7424
	ds_load_b32 v27, v7 offset:7552
	ds_load_b32 v28, v7 offset:7680
	ds_load_b32 v29, v7 offset:7808
	ds_load_b32 v30, v7 offset:7936
	ds_load_b32 v31, v7 offset:8064
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v60, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v61, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v62, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v63, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	.Lb16x4r8w_vcdone:
	s_mov_b32 exec_lo, -1
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16x4r8w_group
	s_branch .Lb16x4r8w_fold
	.Lb16x4r8w_scalar:
	.Lb16x4r8w_tile32:
	s_lshl_b32 s49, s51, 6
	s_lshl_b32 s50, s51, 7
	s_lshl_b32 s48, s51, 5
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s47, s48
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 32
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r8w_sw_done
	s_mov_b32 exec_lo, s47
	global_load_u16 v16, v1, s[2:3]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lb16x4r8w_sw_done:
	s_cmp_ge_u32 s41, s33
	s_cbranch_scc1 .Lb16x4r8w_snofill
	v_add_nc_u32_e32 v5, s62, v2
	s_mov_b32 exec_lo, s47
	global_load_b32 v24, v5, s[4:5]
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v24
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16x4r8w_snofill:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r8w_scdone
	s_mov_b32 exec_lo, s47
	ds_load_b32 v24, v7
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v32, v24, v16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r8w_scdone
	ds_load_b32 v24, v7 offset:1024
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v36, v24, v16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r8w_scdone
	ds_load_b32 v24, v7 offset:2048
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v40, v24, v16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r8w_scdone
	ds_load_b32 v24, v7 offset:3072
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v44, v24, v16
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r8w_scdone
	ds_load_b32 v24, v7 offset:4096
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v48, v24, v16
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4r8w_scdone
	ds_load_b32 v24, v7 offset:5120
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v52, v24, v16
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4r8w_scdone
	ds_load_b32 v24, v7 offset:6144
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v56, v24, v16
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4r8w_scdone
	ds_load_b32 v24, v7 offset:7168
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v60, v24, v16
	.Lb16x4r8w_scdone:
	s_mov_b32 exec_lo, -1
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s59
	s_cbranch_scc1 .Lb16x4r8w_tile32
	.Lb16x4r8w_fold:
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16x4r8w_end
	v_add_f32_e32 v33, v33, v32
	v_add_f32_e32 v35, v35, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v33
	v_add_f32_e32 v37, v37, v36
	v_add_f32_e32 v39, v39, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v37
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v49
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v53
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	ds_swizzle_b32 v24, v35 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v39 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v51 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v55 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	ds_bpermute_b32 v28, v3, v51
	ds_bpermute_b32 v29, v3, v55
	ds_bpermute_b32 v30, v3, v59
	ds_bpermute_b32 v31, v3, v63
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	ds_bpermute_b32 v28, v3, v51
	ds_bpermute_b32 v29, v3, v55
	ds_bpermute_b32 v30, v3, v59
	ds_bpermute_b32 v31, v3, v63
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	ds_bpermute_b32 v28, v3, v51
	ds_bpermute_b32 v29, v3, v55
	ds_bpermute_b32 v30, v3, v59
	ds_bpermute_b32 v31, v3, v63
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	ds_bpermute_b32 v28, v3, v51
	ds_bpermute_b32 v29, v3, v55
	ds_bpermute_b32 v30, v3, v59
	ds_bpermute_b32 v31, v3, v63
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmpx_eq_u32_e32 0, v4
	s_mul_i32 s55, s53, 0
	s_add_co_i32 s55, s55, s54
	v_mov_b32_e32 v24, s55
	global_store_b32 v24, v35, s[6:7]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4r8w_stored
	s_mul_i32 s55, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v25, s55
	global_store_b32 v25, v39, s[6:7]
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4r8w_stored
	s_mul_i32 s55, s53, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v26, s55
	global_store_b32 v26, v43, s[6:7]
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4r8w_stored
	s_mul_i32 s55, s53, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v27, s55
	global_store_b32 v27, v47, s[6:7]
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4r8w_stored
	s_mul_i32 s55, s53, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v28, s55
	global_store_b32 v28, v51, s[6:7]
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4r8w_stored
	s_mul_i32 s55, s53, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v29, s55
	global_store_b32 v29, v55, s[6:7]
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4r8w_stored
	s_mul_i32 s55, s53, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v30, s55
	global_store_b32 v30, v59, s[6:7]
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4r8w_stored
	s_mul_i32 s55, s53, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v31, s55
	global_store_b32 v31, v63, s[6:7]
	.Lb16x4r8w_stored:
	s_wait_storecnt 0x0
	.Lb16x4r8w_end:
	s_endpgm
.Lgemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201_end:
.size gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201, .Lgemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201_end-gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201
	.amdhsa_group_segment_fixed_size 8192
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
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
	.amdhsa_next_free_vgpr 64
	.amdhsa_next_free_sgpr 97
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201_end-gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201
.globl gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201,@function
gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b32 s18, s[0:1], 0x28
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b64 s[2:3], s[8:9]
	s_mov_b64 s[4:5], s[10:11]
	s_mov_b64 s[6:7], s[12:13]
	s_mov_b32 s34, s14
	s_mov_b32 s32, s15
	s_mov_b32 s33, s18
	s_lshl_b32 s92, s40, 1
	s_cmp_ge_u32 s92, s34
	s_cbranch_scc1 .Lb16k4r4w_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16k4r4w_end
	s_cmp_gt_u32 s33, 4
	s_cbranch_scc1 .Lb16k4r4w_end
	s_and_b32 s45, s32, 31
	s_cmp_lg_u32 s45, 0
	s_cbranch_scc1 .Lb16k4r4w_end
	s_lshr_b32 s61, s41, 2
	s_and_b32 s60, s41, 3
	s_add_co_i32 s98, s92, s61
	s_cmp_lt_u32 s98, s34
	s_cselect_b32 s43, 1, 0
	s_lshr_b32 s88, s32, 2
	s_mov_b32 s58, s88
	s_add_co_i32 s44, s88, 0xff
	s_lshr_b32 s44, s44, 8
	s_lshl_b32 s89, s88, 2
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_mul_i32 s95, s98, s57
	s_mul_hi_u32 s96, s98, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s97, s60, s88
	s_lshl_b32 s97, s97, 1
	s_add_co_u32 s2, s2, s97
	s_add_co_ci_u32 s3, s3, 0
	s_lshl_b32 s90, s52, 1
	s_mul_i32 s45, s61, s52
	s_mul_i32 s46, s60, s89
	s_add_co_i32 s91, s45, s46
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 2, v4
	s_lshl_b32 s45, s41, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s45, v7
	s_lshl_b32 s45, s60, 10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, s45, v7
	s_mul_i32 s45, s61, 64
	s_lshl_b32 s46, s60, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s45, s45, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s45
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s51, 0
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16k4r4w_fold
	.Lb16k4r4w_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s47, -1, s47
	s_wait_alu depctr_sa_sdst(0)
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s47, -1, s47
	s_add_co_i32 s45, s50, s91
	v_lshlrev_b32_e32 v2, 5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s45, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16k4r4w_hw_done
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lb16k4r4w_hw_done:
	s_lshr_b32 s45, s41, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s45, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s33
	s_cbranch_scc1 .Lb16k4r4w_st0_h0_nofill
	s_mul_i32 s46, s52, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s46, v2
	s_mov_b32 exec_lo, s47
	global_load_b128 v[24:27], v5, s[4:5]
	global_load_b128 v[28:31], v5, s[4:5] offset:16
	s_wait_loadcnt 0x1
	ds_store_b32 v6, v24
	ds_store_b32 v6, v25 offset:128
	ds_store_b32 v6, v26 offset:256
	ds_store_b32 v6, v27 offset:384
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v28 offset:512
	ds_store_b32 v6, v29 offset:640
	ds_store_b32 v6, v30 offset:768
	ds_store_b32 v6, v31 offset:896
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r4w_st0_h0_nofill:
	s_lshr_b32 s45, s41, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s45, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s33
	s_cbranch_scc1 .Lb16k4r4w_st0_h1_nofill
	s_mul_i32 s46, s52, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s46, s46, s90
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s46, v2
	s_mov_b32 exec_lo, s47
	global_load_b128 v[24:27], v5, s[4:5]
	global_load_b128 v[28:31], v5, s[4:5] offset:16
	s_wait_loadcnt 0x1
	ds_store_b32 v6, v24 offset:8192
	ds_store_b32 v6, v25 offset:8320
	ds_store_b32 v6, v26 offset:8448
	ds_store_b32 v6, v27 offset:8576
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v28 offset:8704
	ds_store_b32 v6, v29 offset:8832
	ds_store_b32 v6, v30 offset:8960
	ds_store_b32 v6, v31 offset:9088
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r4w_st0_h1_nofill:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16k4r4w_st0_cdone
	s_mov_b32 exec_lo, s47
	ds_load_b32 v24, v8
	ds_load_b32 v25, v8 offset:128
	ds_load_b32 v26, v8 offset:256
	ds_load_b32 v27, v8 offset:384
	ds_load_b32 v28, v8 offset:512
	ds_load_b32 v29, v8 offset:640
	ds_load_b32 v30, v8 offset:768
	ds_load_b32 v31, v8 offset:896
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v32, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v33, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v34, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v35, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v32, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v33, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v34, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v35, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r4w_st0_cdone
	ds_load_b32 v24, v8 offset:4096
	ds_load_b32 v25, v8 offset:4224
	ds_load_b32 v26, v8 offset:4352
	ds_load_b32 v27, v8 offset:4480
	ds_load_b32 v28, v8 offset:4608
	ds_load_b32 v29, v8 offset:4736
	ds_load_b32 v30, v8 offset:4864
	ds_load_b32 v31, v8 offset:4992
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v36, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v37, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v38, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v39, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v36, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v37, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v38, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v39, v31, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r4w_st0_cdone
	ds_load_b32 v24, v8 offset:8192
	ds_load_b32 v25, v8 offset:8320
	ds_load_b32 v26, v8 offset:8448
	ds_load_b32 v27, v8 offset:8576
	ds_load_b32 v28, v8 offset:8704
	ds_load_b32 v29, v8 offset:8832
	ds_load_b32 v30, v8 offset:8960
	ds_load_b32 v31, v8 offset:9088
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v40, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v41, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v42, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v43, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r4w_st0_cdone
	ds_load_b32 v24, v8 offset:12288
	ds_load_b32 v25, v8 offset:12416
	ds_load_b32 v26, v8 offset:12544
	ds_load_b32 v27, v8 offset:12672
	ds_load_b32 v28, v8 offset:12800
	ds_load_b32 v29, v8 offset:12928
	ds_load_b32 v30, v8 offset:13056
	ds_load_b32 v31, v8 offset:13184
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v44, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v45, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v46, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v47, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	.Lb16k4r4w_st0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lb16k4r4w_st0_done:
	s_add_co_i32 s51, s51, 1
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16k4r4w_group
	.Lb16k4r4w_fold:
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16k4r4w_pdone
	v_add_f32_e32 v33, v33, v32
	v_add_f32_e32 v35, v35, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v33
	v_add_f32_e32 v37, v37, v36
	v_add_f32_e32 v39, v39, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v37
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	ds_swizzle_b32 v24, v35 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v39 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v27
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v27
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v27
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v27
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v47, v47, v27
	v_cmpx_eq_u32_e32 0, v4
	ds_store_b32 v9, v35
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r4w_pstored
	ds_store_b32 v9, v39 offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r4w_pstored
	ds_store_b32 v9, v43 offset:32
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r4w_pstored
	ds_store_b32 v9, v47 offset:48
	.Lb16k4r4w_pstored:
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r4w_pdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s46, s41, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s94, s46, 2
	s_and_b32 s93, s46, 3
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4r4w_iskip0
	s_add_co_i32 s45, s92, s94
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s34
	s_cbranch_scc1 .Lb16k4r4w_iskip0
	s_mov_b32 s98, s45
	s_lshl_b32 s45, s46, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4r4w_item0_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4r4w_item0_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r4w_iskip0:
	.Lb16k4r4w_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201_end:
.size gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201, .Lgemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201_end-gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201
	.amdhsa_group_segment_fixed_size 16384
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
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
	.amdhsa_next_free_vgpr 48
	.amdhsa_next_free_sgpr 99
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201_end-gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201
.globl gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201,@function
gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b32 s18, s[0:1], 0x28
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b64 s[2:3], s[8:9]
	s_mov_b64 s[4:5], s[10:11]
	s_mov_b64 s[6:7], s[12:13]
	s_mov_b32 s34, s14
	s_mov_b32 s32, s15
	s_mov_b32 s33, s18
	s_lshl_b32 s92, s40, 1
	s_cmp_ge_u32 s92, s34
	s_cbranch_scc1 .Lb16k4r8w_end
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .Lb16k4r8w_end
	s_cmp_gt_u32 s33, 8
	s_cbranch_scc1 .Lb16k4r8w_end
	s_and_b32 s45, s32, 31
	s_cmp_lg_u32 s45, 0
	s_cbranch_scc1 .Lb16k4r8w_end
	s_lshr_b32 s61, s41, 2
	s_and_b32 s60, s41, 3
	s_add_co_i32 s98, s92, s61
	s_cmp_lt_u32 s98, s34
	s_cselect_b32 s43, 1, 0
	s_lshr_b32 s88, s32, 2
	s_mov_b32 s58, s88
	s_add_co_i32 s44, s88, 0xff
	s_lshr_b32 s44, s44, 8
	s_lshl_b32 s89, s88, 2
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_mul_i32 s95, s98, s57
	s_mul_hi_u32 s96, s98, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s97, s60, s88
	s_lshl_b32 s97, s97, 1
	s_add_co_u32 s2, s2, s97
	s_add_co_ci_u32 s3, s3, 0
	s_lshl_b32 s90, s52, 1
	s_mul_i32 s45, s61, s52
	s_mul_i32 s46, s60, s89
	s_add_co_i32 s91, s45, s46
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 2, v4
	s_lshl_b32 s45, s41, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v6, s45, v7
	s_lshl_b32 s45, s60, 10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, s45, v7
	s_mul_i32 s45, s61, 0x80
	s_lshl_b32 s46, s60, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s45, s45, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s45
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	s_mov_b32 s51, 0
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16k4r8w_fold
	.Lb16k4r8w_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s47, -1, s47
	s_wait_alu depctr_sa_sdst(0)
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_cselect_b32 s47, -1, s47
	s_add_co_i32 s45, s50, s91
	v_lshlrev_b32_e32 v2, 5, v4
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s45, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16k4r8w_hw_done
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	.Lb16k4r8w_hw_done:
	s_lshr_b32 s45, s41, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s45, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s33
	s_cbranch_scc1 .Lb16k4r8w_st0_h0_nofill
	s_mul_i32 s46, s52, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s46, v2
	s_mov_b32 exec_lo, s47
	global_load_b128 v[24:27], v5, s[4:5]
	global_load_b128 v[28:31], v5, s[4:5] offset:16
	s_wait_loadcnt 0x1
	ds_store_b32 v6, v24
	ds_store_b32 v6, v25 offset:128
	ds_store_b32 v6, v26 offset:256
	ds_store_b32 v6, v27 offset:384
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v28 offset:512
	ds_store_b32 v6, v29 offset:640
	ds_store_b32 v6, v30 offset:768
	ds_store_b32 v6, v31 offset:896
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8w_st0_h0_nofill:
	s_lshr_b32 s45, s41, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s45, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s33
	s_cbranch_scc1 .Lb16k4r8w_st0_h1_nofill
	s_mul_i32 s46, s52, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s46, s46, s90
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s46, v2
	s_mov_b32 exec_lo, s47
	global_load_b128 v[24:27], v5, s[4:5]
	global_load_b128 v[28:31], v5, s[4:5] offset:16
	s_wait_loadcnt 0x1
	ds_store_b32 v6, v24 offset:8192
	ds_store_b32 v6, v25 offset:8320
	ds_store_b32 v6, v26 offset:8448
	ds_store_b32 v6, v27 offset:8576
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v28 offset:8704
	ds_store_b32 v6, v29 offset:8832
	ds_store_b32 v6, v30 offset:8960
	ds_store_b32 v6, v31 offset:9088
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8w_st0_h1_nofill:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16k4r8w_st0_cdone
	s_mov_b32 exec_lo, s47
	ds_load_b32 v24, v8
	ds_load_b32 v25, v8 offset:128
	ds_load_b32 v26, v8 offset:256
	ds_load_b32 v27, v8 offset:384
	ds_load_b32 v28, v8 offset:512
	ds_load_b32 v29, v8 offset:640
	ds_load_b32 v30, v8 offset:768
	ds_load_b32 v31, v8 offset:896
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v32, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v33, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v34, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v35, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v32, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v33, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v34, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v35, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r8w_st0_cdone
	ds_load_b32 v24, v8 offset:4096
	ds_load_b32 v25, v8 offset:4224
	ds_load_b32 v26, v8 offset:4352
	ds_load_b32 v27, v8 offset:4480
	ds_load_b32 v28, v8 offset:4608
	ds_load_b32 v29, v8 offset:4736
	ds_load_b32 v30, v8 offset:4864
	ds_load_b32 v31, v8 offset:4992
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v36, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v37, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v38, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v39, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v36, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v37, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v38, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v39, v31, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r8w_st0_cdone
	ds_load_b32 v24, v8 offset:8192
	ds_load_b32 v25, v8 offset:8320
	ds_load_b32 v26, v8 offset:8448
	ds_load_b32 v27, v8 offset:8576
	ds_load_b32 v28, v8 offset:8704
	ds_load_b32 v29, v8 offset:8832
	ds_load_b32 v30, v8 offset:8960
	ds_load_b32 v31, v8 offset:9088
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v40, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v41, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v42, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v43, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r8w_st0_cdone
	ds_load_b32 v24, v8 offset:12288
	ds_load_b32 v25, v8 offset:12416
	ds_load_b32 v26, v8 offset:12544
	ds_load_b32 v27, v8 offset:12672
	ds_load_b32 v28, v8 offset:12800
	ds_load_b32 v29, v8 offset:12928
	ds_load_b32 v30, v8 offset:13056
	ds_load_b32 v31, v8 offset:13184
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v44, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v45, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v46, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v47, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	.Lb16k4r8w_st0_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lb16k4r8w_st0_done:
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4r8w_st1_done
	s_lshr_b32 s45, s41, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s45, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s33
	s_cbranch_scc1 .Lb16k4r8w_st1_h0_nofill
	s_mul_i32 s46, s52, 4
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s46, v2
	s_mov_b32 exec_lo, s47
	global_load_b128 v[24:27], v5, s[4:5]
	global_load_b128 v[28:31], v5, s[4:5] offset:16
	s_wait_loadcnt 0x1
	ds_store_b32 v6, v24
	ds_store_b32 v6, v25 offset:128
	ds_store_b32 v6, v26 offset:256
	ds_store_b32 v6, v27 offset:384
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v28 offset:512
	ds_store_b32 v6, v29 offset:640
	ds_store_b32 v6, v30 offset:768
	ds_store_b32 v6, v31 offset:896
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8w_st1_h0_nofill:
	s_lshr_b32 s45, s41, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s45, s45, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s33
	s_cbranch_scc1 .Lb16k4r8w_st1_h1_nofill
	s_mul_i32 s46, s52, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s46, s46, s90
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s46, v2
	s_mov_b32 exec_lo, s47
	global_load_b128 v[24:27], v5, s[4:5]
	global_load_b128 v[28:31], v5, s[4:5] offset:16
	s_wait_loadcnt 0x1
	ds_store_b32 v6, v24 offset:8192
	ds_store_b32 v6, v25 offset:8320
	ds_store_b32 v6, v26 offset:8448
	ds_store_b32 v6, v27 offset:8576
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v28 offset:8704
	ds_store_b32 v6, v29 offset:8832
	ds_store_b32 v6, v30 offset:8960
	ds_store_b32 v6, v31 offset:9088
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8w_st1_h1_nofill:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16k4r8w_st1_cdone
	s_mov_b32 exec_lo, s47
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4r8w_st1_cdone
	ds_load_b32 v24, v8
	ds_load_b32 v25, v8 offset:128
	ds_load_b32 v26, v8 offset:256
	ds_load_b32 v27, v8 offset:384
	ds_load_b32 v28, v8 offset:512
	ds_load_b32 v29, v8 offset:640
	ds_load_b32 v30, v8 offset:768
	ds_load_b32 v31, v8 offset:896
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v48, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v49, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v50, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v51, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4r8w_st1_cdone
	ds_load_b32 v24, v8 offset:4096
	ds_load_b32 v25, v8 offset:4224
	ds_load_b32 v26, v8 offset:4352
	ds_load_b32 v27, v8 offset:4480
	ds_load_b32 v28, v8 offset:4608
	ds_load_b32 v29, v8 offset:4736
	ds_load_b32 v30, v8 offset:4864
	ds_load_b32 v31, v8 offset:4992
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v52, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v53, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v54, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v55, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4r8w_st1_cdone
	ds_load_b32 v24, v8 offset:8192
	ds_load_b32 v25, v8 offset:8320
	ds_load_b32 v26, v8 offset:8448
	ds_load_b32 v27, v8 offset:8576
	ds_load_b32 v28, v8 offset:8704
	ds_load_b32 v29, v8 offset:8832
	ds_load_b32 v30, v8 offset:8960
	ds_load_b32 v31, v8 offset:9088
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v56, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v57, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v58, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v59, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4r8w_st1_cdone
	ds_load_b32 v24, v8 offset:12288
	ds_load_b32 v25, v8 offset:12416
	ds_load_b32 v26, v8 offset:12544
	ds_load_b32 v27, v8 offset:12672
	ds_load_b32 v28, v8 offset:12800
	ds_load_b32 v29, v8 offset:12928
	ds_load_b32 v30, v8 offset:13056
	ds_load_b32 v31, v8 offset:13184
	s_wait_dscnt 0x7
	v_fmac_f32_e32 v60, v24, v16
	s_wait_dscnt 0x6
	v_fmac_f32_e32 v61, v25, v17
	s_wait_dscnt 0x5
	v_fmac_f32_e32 v62, v26, v18
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v63, v27, v19
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	.Lb16k4r8w_st1_cdone:
	s_mov_b32 exec_lo, -1
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lb16k4r8w_st1_done:
	s_add_co_i32 s51, s51, 1
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16k4r8w_group
	.Lb16k4r8w_fold:
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lb16k4r8w_pdone
	v_add_f32_e32 v33, v33, v32
	v_add_f32_e32 v35, v35, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v33
	v_add_f32_e32 v37, v37, v36
	v_add_f32_e32 v39, v39, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v37
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v49
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v53
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	ds_swizzle_b32 v24, v35 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v39 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v51 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v55 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	ds_bpermute_b32 v28, v3, v51
	ds_bpermute_b32 v29, v3, v55
	ds_bpermute_b32 v30, v3, v59
	ds_bpermute_b32 v31, v3, v63
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	ds_bpermute_b32 v28, v3, v51
	ds_bpermute_b32 v29, v3, v55
	ds_bpermute_b32 v30, v3, v59
	ds_bpermute_b32 v31, v3, v63
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	ds_bpermute_b32 v28, v3, v51
	ds_bpermute_b32 v29, v3, v55
	ds_bpermute_b32 v30, v3, v59
	ds_bpermute_b32 v31, v3, v63
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v35
	ds_bpermute_b32 v25, v3, v39
	ds_bpermute_b32 v26, v3, v43
	ds_bpermute_b32 v27, v3, v47
	ds_bpermute_b32 v28, v3, v51
	ds_bpermute_b32 v29, v3, v55
	ds_bpermute_b32 v30, v3, v59
	ds_bpermute_b32 v31, v3, v63
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v35, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v39, v39, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v43, v43, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v47, v47, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v51, v51, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v55, v55, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v59, v59, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v63, v63, v31
	v_cmpx_eq_u32_e32 0, v4
	ds_store_b32 v9, v35
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4r8w_pstored
	ds_store_b32 v9, v39 offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4r8w_pstored
	ds_store_b32 v9, v43 offset:32
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4r8w_pstored
	ds_store_b32 v9, v47 offset:48
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4r8w_pstored
	ds_store_b32 v9, v51 offset:64
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4r8w_pstored
	ds_store_b32 v9, v55 offset:80
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4r8w_pstored
	ds_store_b32 v9, v59 offset:96
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4r8w_pstored
	ds_store_b32 v9, v63 offset:112
	.Lb16k4r8w_pstored:
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8w_pdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s46, s41, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s94, s46, 3
	s_and_b32 s93, s46, 7
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4r8w_iskip0
	s_add_co_i32 s45, s92, s94
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s34
	s_cbranch_scc1 .Lb16k4r8w_iskip0
	s_mov_b32 s98, s45
	s_lshl_b32 s45, s46, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4r8w_item0_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4r8w_item0_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8w_iskip0:
	s_add_co_i32 s46, s41, 8
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s94, s46, 3
	s_and_b32 s93, s46, 7
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4r8w_iskip1
	s_add_co_i32 s45, s92, s94
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s45, s34
	s_cbranch_scc1 .Lb16k4r8w_iskip1
	s_mov_b32 s98, s45
	s_lshl_b32 s45, s46, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4r8w_item1_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4r8w_item1_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4r8w_iskip1:
	.Lb16k4r8w_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201_end:
.size gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201, .Lgemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201_end-gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201
	.amdhsa_group_segment_fixed_size 16384
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
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
	.amdhsa_next_free_vgpr 64
	.amdhsa_next_free_sgpr 99
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201_end-gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_rows_x4_t8_pm_gfx1201
.globl gemv_bf16_xf32_rows_x4_t8_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_rows_x4_t8_pm_gfx1201,@function
gemv_bf16_xf32_rows_x4_t8_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b32 s32, s30
	s_mov_b64 s[4:5], s[16:17]
	s_mov_b32 s36, s26
	s_add_co_i32 s37, s36, s27
	s_add_co_i32 s38, s37, s28
	s_add_co_i32 s39, s38, s29
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lb16x4t8_end
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .Lb16x4t8_end
	s_cmp_gt_u32 s31, 64
	s_cbranch_scc1 .Lb16x4t8_end
	s_and_b32 s99, ttmp7, 0xffff
	s_lshl_b32 s99, s99, 3
	s_cmp_ge_u32 s99, s31
	s_cbranch_scc1 .Lb16x4t8_end
	s_sub_co_i32 s55, s31, s99
	s_min_u32 s33, s55, 8
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[2:3], s[12:13], s[14:15]
	s_cselect_b64 s[6:7], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[2:3], s[10:11], s[2:3]
	s_cselect_b64 s[6:7], s[20:21], s[6:7]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[2:3], s[8:9], s[2:3]
	s_cselect_b64 s[6:7], s[18:19], s[6:7]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_co_i32 s42, s40, s35
	s_and_b32 s56, s32, 7
	s_mov_b32 s58, s32
	s_add_co_i32 s44, s32, 0xff
	s_lshr_b32 s44, s44, 8
	s_add_co_i32 s59, s32, 31
	s_lshr_b32 s59, s59, 5
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_lshl_b32 s54, s42, 2
	s_mul_i32 s95, s42, s57
	s_mul_hi_u32 s96, s42, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s45, s99, s52
	s_add_co_u32 s4, s4, s45
	s_add_co_ci_u32 s5, s5, 0
	s_mul_i32 s45, s99, s53
	s_add_co_u32 s6, s6, s45
	s_add_co_ci_u32 s7, s7, 0
	s_mov_b64 s[64:65], s[4:5]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_row_pointers_done
	s_mul_i32 s45, s52, 1
	s_add_co_u32 s66, s4, s45
	s_add_co_ci_u32 s67, s5, 0
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_row_pointers_done
	s_mul_i32 s45, s52, 2
	s_add_co_u32 s68, s4, s45
	s_add_co_ci_u32 s69, s5, 0
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_row_pointers_done
	s_mul_i32 s45, s52, 3
	s_add_co_u32 s70, s4, s45
	s_add_co_ci_u32 s71, s5, 0
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_row_pointers_done
	s_mul_i32 s45, s52, 4
	s_add_co_u32 s72, s4, s45
	s_add_co_ci_u32 s73, s5, 0
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_row_pointers_done
	s_mul_i32 s45, s52, 5
	s_add_co_u32 s74, s4, s45
	s_add_co_ci_u32 s75, s5, 0
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_row_pointers_done
	s_mul_i32 s45, s52, 6
	s_add_co_u32 s76, s4, s45
	s_add_co_ci_u32 s77, s5, 0
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_row_pointers_done
	s_mul_i32 s45, s52, 7
	s_add_co_u32 s78, s4, s45
	s_add_co_ci_u32 s79, s5, 0
	.Lb16x4t8_row_pointers_done:
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_zero_rows_done
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_zero_rows_done
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_zero_rows_done
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_zero_rows_done
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_zero_rows_done
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_zero_rows_done
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_zero_rows_done
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	.Lb16x4t8_zero_rows_done:
	s_mov_b32 s51, 0
	s_cmp_lg_u32 s56, 0
	s_cbranch_scc1 .Lb16x4t8_scalar
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16x4t8_fold
	.Lb16x4t8_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	global_load_b128 v[24:27], v2, s[64:65]
	global_load_b128 v[28:31], v2, s[64:65] offset:16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_gx_done
	global_load_b128 v[32:35], v2, s[66:67]
	global_load_b128 v[36:39], v2, s[66:67] offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_gx_done
	global_load_b128 v[40:43], v2, s[68:69]
	global_load_b128 v[44:47], v2, s[68:69] offset:16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_gx_done
	global_load_b128 v[48:51], v2, s[70:71]
	global_load_b128 v[52:55], v2, s[70:71] offset:16
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_gx_done
	global_load_b128 v[56:59], v2, s[72:73]
	global_load_b128 v[60:63], v2, s[72:73] offset:16
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_gx_done
	global_load_b128 v[64:67], v2, s[74:75]
	global_load_b128 v[68:71], v2, s[74:75] offset:16
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_gx_done
	global_load_b128 v[72:75], v2, s[76:77]
	global_load_b128 v[76:79], v2, s[76:77] offset:16
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_gx_done
	global_load_b128 v[80:83], v2, s[78:79]
	global_load_b128 v[84:87], v2, s[78:79] offset:16
	.Lb16x4t8_gx_done:
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v24, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v89, v25, v17
	v_fmac_f32_e32 v90, v26, v18
	v_fmac_f32_e32 v91, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_gc_done
	v_fmac_f32_e32 v92, v32, v16
	v_fmac_f32_e32 v93, v33, v17
	v_fmac_f32_e32 v94, v34, v18
	v_fmac_f32_e32 v95, v35, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v36, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v37, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v38, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v39, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_gc_done
	v_fmac_f32_e32 v96, v40, v16
	v_fmac_f32_e32 v97, v41, v17
	v_fmac_f32_e32 v98, v42, v18
	v_fmac_f32_e32 v99, v43, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v44, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v45, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v46, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v47, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_gc_done
	v_fmac_f32_e32 v100, v48, v16
	v_fmac_f32_e32 v101, v49, v17
	v_fmac_f32_e32 v102, v50, v18
	v_fmac_f32_e32 v103, v51, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v52, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v53, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v54, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v55, v23
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_gc_done
	v_fmac_f32_e32 v104, v56, v16
	v_fmac_f32_e32 v105, v57, v17
	v_fmac_f32_e32 v106, v58, v18
	v_fmac_f32_e32 v107, v59, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v60, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v61, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v62, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v63, v23
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_gc_done
	v_fmac_f32_e32 v108, v64, v16
	v_fmac_f32_e32 v109, v65, v17
	v_fmac_f32_e32 v110, v66, v18
	v_fmac_f32_e32 v111, v67, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v68, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v69, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v70, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v71, v23
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_gc_done
	v_fmac_f32_e32 v112, v72, v16
	v_fmac_f32_e32 v113, v73, v17
	v_fmac_f32_e32 v114, v74, v18
	v_fmac_f32_e32 v115, v75, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v76, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v77, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v78, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v79, v23
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_gc_done
	v_fmac_f32_e32 v116, v80, v16
	v_fmac_f32_e32 v117, v81, v17
	v_fmac_f32_e32 v118, v82, v18
	v_fmac_f32_e32 v119, v83, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v84, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v85, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v86, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v87, v23
	.Lb16x4t8_gc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16x4t8_group
	s_branch .Lb16x4t8_fold
	.Lb16x4t8_scalar:
	.Lb16x4t8_tile32:
	s_lshl_b32 s49, s51, 6
	s_lshl_b32 s50, s51, 7
	s_lshl_b32 s48, s51, 5
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s47, s48
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 32
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_u16 v16, v1, s[2:3]
	global_load_b32 v24, v2, s[64:65]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_sx_done
	global_load_b32 v32, v2, s[66:67]
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_sx_done
	global_load_b32 v40, v2, s[68:69]
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_sx_done
	global_load_b32 v48, v2, s[70:71]
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_sx_done
	global_load_b32 v56, v2, s[72:73]
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_sx_done
	global_load_b32 v64, v2, s[74:75]
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_sx_done
	global_load_b32 v72, v2, s[76:77]
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_sx_done
	global_load_b32 v80, v2, s[78:79]
	.Lb16x4t8_sx_done:
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v24, v16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_sc_done
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v92, v32, v16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_sc_done
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v96, v40, v16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_sc_done
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v48, v16
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_sc_done
	v_fmac_f32_e32 v104, v56, v16
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_sc_done
	v_fmac_f32_e32 v108, v64, v16
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_sc_done
	v_fmac_f32_e32 v112, v72, v16
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_sc_done
	v_fmac_f32_e32 v116, v80, v16
	.Lb16x4t8_sc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s59
	s_cbranch_scc1 .Lb16x4t8_tile32
	.Lb16x4t8_fold:
	v_add_f32_e32 v89, v89, v88
	v_add_f32_e32 v91, v91, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v89
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_fold_rows_done
	v_add_f32_e32 v93, v93, v92
	v_add_f32_e32 v95, v95, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v93
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_fold_rows_done
	v_add_f32_e32 v97, v97, v96
	v_add_f32_e32 v99, v99, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v97
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_fold_rows_done
	v_add_f32_e32 v101, v101, v100
	v_add_f32_e32 v103, v103, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v101
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_fold_rows_done
	v_add_f32_e32 v105, v105, v104
	v_add_f32_e32 v107, v107, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v107, v107, v105
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_fold_rows_done
	v_add_f32_e32 v109, v109, v108
	v_add_f32_e32 v111, v111, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v111, v111, v109
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_fold_rows_done
	v_add_f32_e32 v113, v113, v112
	v_add_f32_e32 v115, v115, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v115, v115, v113
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_fold_rows_done
	v_add_f32_e32 v117, v117, v116
	v_add_f32_e32 v119, v119, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v119, v119, v117
	.Lb16x4t8_fold_rows_done:
	ds_swizzle_b32 v24, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_swizzle_rows_done
	ds_swizzle_b32 v25, v95 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_swizzle_rows_done
	ds_swizzle_b32 v26, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_swizzle_rows_done
	ds_swizzle_b32 v27, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_swizzle_rows_done
	ds_swizzle_b32 v28, v107 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_swizzle_rows_done
	ds_swizzle_b32 v29, v111 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_swizzle_rows_done
	ds_swizzle_b32 v30, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_swizzle_rows_done
	ds_swizzle_b32 v31, v119 offset:swizzle(BITMASK_PERM,"1pppp")
	.Lb16x4t8_swizzle_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_swizzle_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_swizzle_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_swizzle_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_swizzle_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_swizzle_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_swizzle_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_swizzle_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16x4t8_swizzle_add_rows_done:
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_permute8_rows_done
	ds_bpermute_b32 v25, v3, v95
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_permute8_rows_done
	ds_bpermute_b32 v26, v3, v99
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_permute8_rows_done
	ds_bpermute_b32 v27, v3, v103
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_permute8_rows_done
	ds_bpermute_b32 v28, v3, v107
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_permute8_rows_done
	ds_bpermute_b32 v29, v3, v111
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_permute8_rows_done
	ds_bpermute_b32 v30, v3, v115
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_permute8_rows_done
	ds_bpermute_b32 v31, v3, v119
	.Lb16x4t8_permute8_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_permute8_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_permute8_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_permute8_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_permute8_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_permute8_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_permute8_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_permute8_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16x4t8_permute8_add_rows_done:
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_permute4_rows_done
	ds_bpermute_b32 v25, v3, v95
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_permute4_rows_done
	ds_bpermute_b32 v26, v3, v99
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_permute4_rows_done
	ds_bpermute_b32 v27, v3, v103
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_permute4_rows_done
	ds_bpermute_b32 v28, v3, v107
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_permute4_rows_done
	ds_bpermute_b32 v29, v3, v111
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_permute4_rows_done
	ds_bpermute_b32 v30, v3, v115
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_permute4_rows_done
	ds_bpermute_b32 v31, v3, v119
	.Lb16x4t8_permute4_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_permute4_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_permute4_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_permute4_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_permute4_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_permute4_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_permute4_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_permute4_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16x4t8_permute4_add_rows_done:
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_permute2_rows_done
	ds_bpermute_b32 v25, v3, v95
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_permute2_rows_done
	ds_bpermute_b32 v26, v3, v99
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_permute2_rows_done
	ds_bpermute_b32 v27, v3, v103
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_permute2_rows_done
	ds_bpermute_b32 v28, v3, v107
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_permute2_rows_done
	ds_bpermute_b32 v29, v3, v111
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_permute2_rows_done
	ds_bpermute_b32 v30, v3, v115
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_permute2_rows_done
	ds_bpermute_b32 v31, v3, v119
	.Lb16x4t8_permute2_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_permute2_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_permute2_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_permute2_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_permute2_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_permute2_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_permute2_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_permute2_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16x4t8_permute2_add_rows_done:
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_permute1_rows_done
	ds_bpermute_b32 v25, v3, v95
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_permute1_rows_done
	ds_bpermute_b32 v26, v3, v99
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_permute1_rows_done
	ds_bpermute_b32 v27, v3, v103
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_permute1_rows_done
	ds_bpermute_b32 v28, v3, v107
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_permute1_rows_done
	ds_bpermute_b32 v29, v3, v111
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_permute1_rows_done
	ds_bpermute_b32 v30, v3, v115
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_permute1_rows_done
	ds_bpermute_b32 v31, v3, v119
	.Lb16x4t8_permute1_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_permute1_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_permute1_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_permute1_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_permute1_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_permute1_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_permute1_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_permute1_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16x4t8_permute1_add_rows_done:
	v_cmpx_eq_u32_e32 0, v4
	s_mul_i32 s55, s53, 0
	s_add_co_i32 s55, s55, s54
	v_mov_b32_e32 v24, s55
	global_store_b32 v24, v91, s[6:7]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16x4t8_stored
	s_mul_i32 s55, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v25, s55
	global_store_b32 v25, v95, s[6:7]
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16x4t8_stored
	s_mul_i32 s55, s53, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v26, s55
	global_store_b32 v26, v99, s[6:7]
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16x4t8_stored
	s_mul_i32 s55, s53, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v27, s55
	global_store_b32 v27, v103, s[6:7]
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16x4t8_stored
	s_mul_i32 s55, s53, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v28, s55
	global_store_b32 v28, v107, s[6:7]
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16x4t8_stored
	s_mul_i32 s55, s53, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v29, s55
	global_store_b32 v29, v111, s[6:7]
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16x4t8_stored
	s_mul_i32 s55, s53, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v30, s55
	global_store_b32 v30, v115, s[6:7]
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16x4t8_stored
	s_mul_i32 s55, s53, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s55, s54
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v31, s55
	global_store_b32 v31, v119, s[6:7]
	.Lb16x4t8_stored:
	s_wait_storecnt 0x0
	.Lb16x4t8_end:
	s_endpgm
.Lgemv_bf16_xf32_rows_x4_t8_pm_gfx1201_end:
.size gemv_bf16_xf32_rows_x4_t8_pm_gfx1201, .Lgemv_bf16_xf32_rows_x4_t8_pm_gfx1201_end-gemv_bf16_xf32_rows_x4_t8_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_rows_x4_t8_pm_gfx1201
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
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
	.amdhsa_next_free_vgpr 120
	.amdhsa_next_free_sgpr 100
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_rows_x4_t8_pm_gfx1201_end-gemv_bf16_xf32_rows_x4_t8_pm_gfx1201)<<4)&4080)>>4
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
.text
.protected gemv_bf16_xf32_k4_rows_t8_pm_gfx1201
.globl gemv_bf16_xf32_k4_rows_t8_pm_gfx1201
.p2align 8
.type gemv_bf16_xf32_k4_rows_t8_pm_gfx1201,@function
gemv_bf16_xf32_k4_rows_t8_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b32 s18, s[0:1], 0x28
	v_lshrrev_b32_e32 v5, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s41, v5
	v_and_b32_e32 v4, 31, v0
	s_mov_b32 s40, ttmp9
	s_wait_kmcnt 0x0
	s_mov_b64 s[2:3], s[8:9]
	s_mov_b64 s[4:5], s[10:11]
	s_mov_b64 s[6:7], s[12:13]
	s_mov_b32 s34, s14
	s_mov_b32 s32, s15
	s_cmp_ge_u32 s40, s34
	s_cbranch_scc1 .Lb16k4t8_end
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lb16k4t8_end
	s_cmp_gt_u32 s18, 64
	s_cbranch_scc1 .Lb16k4t8_end
	s_and_b32 s99, ttmp7, 0xffff
	s_lshl_b32 s99, s99, 3
	s_cmp_ge_u32 s99, s18
	s_cbranch_scc1 .Lb16k4t8_end
	s_sub_co_i32 s55, s18, s99
	s_min_u32 s33, s55, 8
	s_and_b32 s45, s32, 31
	s_cmp_lg_u32 s45, 0
	s_cbranch_scc1 .Lb16k4t8_end
	s_mov_b32 s60, s41
	s_mov_b32 s98, s40
	s_lshr_b32 s88, s32, 2
	s_mov_b32 s58, s88
	s_add_co_i32 s44, s88, 0xff
	s_lshr_b32 s44, s44, 8
	s_lshl_b32 s89, s88, 2
	s_lshl_b32 s52, s32, 2
	s_lshl_b32 s57, s32, 1
	s_lshl_b32 s53, s34, 2
	s_mul_i32 s95, s98, s57
	s_mul_hi_u32 s96, s98, s57
	s_add_co_u32 s2, s2, s95
	s_add_co_ci_u32 s3, s3, s96
	s_mul_i32 s97, s60, s88
	s_lshl_b32 s97, s97, 1
	s_add_co_u32 s2, s2, s97
	s_add_co_ci_u32 s3, s3, 0
	s_mov_b32 s45, s89
	s_mul_i32 s45, s60, s45
	s_add_co_u32 s4, s4, s45
	s_add_co_ci_u32 s5, s5, 0
	s_mul_i32 s45, s99, s52
	s_add_co_u32 s4, s4, s45
	s_add_co_ci_u32 s5, s5, 0
	s_mul_i32 s45, s99, s53
	s_add_co_u32 s6, s6, s45
	s_add_co_ci_u32 s7, s7, 0
	s_mov_b64 s[64:65], s[4:5]
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_row_pointers_done
	s_mul_i32 s45, s52, 1
	s_add_co_u32 s66, s4, s45
	s_add_co_ci_u32 s67, s5, 0
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_row_pointers_done
	s_mul_i32 s45, s52, 2
	s_add_co_u32 s68, s4, s45
	s_add_co_ci_u32 s69, s5, 0
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_row_pointers_done
	s_mul_i32 s45, s52, 3
	s_add_co_u32 s70, s4, s45
	s_add_co_ci_u32 s71, s5, 0
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_row_pointers_done
	s_mul_i32 s45, s52, 4
	s_add_co_u32 s72, s4, s45
	s_add_co_ci_u32 s73, s5, 0
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_row_pointers_done
	s_mul_i32 s45, s52, 5
	s_add_co_u32 s74, s4, s45
	s_add_co_ci_u32 s75, s5, 0
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_row_pointers_done
	s_mul_i32 s45, s52, 6
	s_add_co_u32 s76, s4, s45
	s_add_co_ci_u32 s77, s5, 0
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_row_pointers_done
	s_mul_i32 s45, s52, 7
	s_add_co_u32 s78, s4, s45
	s_add_co_ci_u32 s79, s5, 0
	.Lb16k4t8_row_pointers_done:
	s_lshl_b32 s45, s60, 2
	v_mov_b32_e32 v9, s45
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_zero_rows_done
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_zero_rows_done
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_zero_rows_done
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_zero_rows_done
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_zero_rows_done
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_zero_rows_done
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_zero_rows_done
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	.Lb16k4t8_zero_rows_done:
	s_mov_b32 s51, 0
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lb16k4t8_fold
	.Lb16k4t8_group:
	s_lshl_b32 s49, s51, 9
	s_lshl_b32 s50, s51, 10
	s_lshl_b32 s48, s51, 8
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s48, s58, s48
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s47, s48, 3
	s_lshl_b32 s47, -1, s47
	s_not_b32 s47, s47
	s_cmp_ge_u32 s48, 0x100
	s_cselect_b32 s47, -1, s47
	v_lshlrev_b32_e32 v2, 5, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s50, v2
	v_lshlrev_b32_e32 v1, 4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s49, v1
	s_mov_b32 exec_lo, s47
	global_load_b128 v[16:19], v1, s[2:3]
	global_load_b128 v[24:27], v2, s[64:65]
	global_load_b128 v[28:31], v2, s[64:65] offset:16
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_gx_done
	global_load_b128 v[32:35], v2, s[66:67]
	global_load_b128 v[36:39], v2, s[66:67] offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_gx_done
	global_load_b128 v[40:43], v2, s[68:69]
	global_load_b128 v[44:47], v2, s[68:69] offset:16
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_gx_done
	global_load_b128 v[48:51], v2, s[70:71]
	global_load_b128 v[52:55], v2, s[70:71] offset:16
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_gx_done
	global_load_b128 v[56:59], v2, s[72:73]
	global_load_b128 v[60:63], v2, s[72:73] offset:16
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_gx_done
	global_load_b128 v[64:67], v2, s[74:75]
	global_load_b128 v[68:71], v2, s[74:75] offset:16
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_gx_done
	global_load_b128 v[72:75], v2, s[76:77]
	global_load_b128 v[76:79], v2, s[76:77] offset:16
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_gx_done
	global_load_b128 v[80:83], v2, s[78:79]
	global_load_b128 v[84:87], v2, s[78:79] offset:16
	.Lb16k4t8_gx_done:
	s_wait_loadcnt 0x0
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v24, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v89, v25, v17
	v_fmac_f32_e32 v90, v26, v18
	v_fmac_f32_e32 v91, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v88, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v89, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v90, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v91, v31, v23
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_gc_done
	v_fmac_f32_e32 v92, v32, v16
	v_fmac_f32_e32 v93, v33, v17
	v_fmac_f32_e32 v94, v34, v18
	v_fmac_f32_e32 v95, v35, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v92, v36, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v93, v37, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v38, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v95, v39, v23
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_gc_done
	v_fmac_f32_e32 v96, v40, v16
	v_fmac_f32_e32 v97, v41, v17
	v_fmac_f32_e32 v98, v42, v18
	v_fmac_f32_e32 v99, v43, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v96, v44, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v97, v45, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v98, v46, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v99, v47, v23
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_gc_done
	v_fmac_f32_e32 v100, v48, v16
	v_fmac_f32_e32 v101, v49, v17
	v_fmac_f32_e32 v102, v50, v18
	v_fmac_f32_e32 v103, v51, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v100, v52, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v101, v53, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v102, v54, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v55, v23
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_gc_done
	v_fmac_f32_e32 v104, v56, v16
	v_fmac_f32_e32 v105, v57, v17
	v_fmac_f32_e32 v106, v58, v18
	v_fmac_f32_e32 v107, v59, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v104, v60, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v105, v61, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v106, v62, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v107, v63, v23
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_gc_done
	v_fmac_f32_e32 v108, v64, v16
	v_fmac_f32_e32 v109, v65, v17
	v_fmac_f32_e32 v110, v66, v18
	v_fmac_f32_e32 v111, v67, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v108, v68, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v109, v69, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v70, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v111, v71, v23
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_gc_done
	v_fmac_f32_e32 v112, v72, v16
	v_fmac_f32_e32 v113, v73, v17
	v_fmac_f32_e32 v114, v74, v18
	v_fmac_f32_e32 v115, v75, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v76, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v113, v77, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v114, v78, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v115, v79, v23
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_gc_done
	v_fmac_f32_e32 v116, v80, v16
	v_fmac_f32_e32 v117, v81, v17
	v_fmac_f32_e32 v118, v82, v18
	v_fmac_f32_e32 v119, v83, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v116, v84, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v85, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v118, v86, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v119, v87, v23
	.Lb16k4t8_gc_done:
	s_mov_b32 exec_lo, -1
	s_add_co_i32 s51, s51, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_u32 s51, s44
	s_cbranch_scc1 .Lb16k4t8_group
	.Lb16k4t8_fold:
	v_add_f32_e32 v89, v89, v88
	v_add_f32_e32 v91, v91, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v89
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_fold_rows_done
	v_add_f32_e32 v93, v93, v92
	v_add_f32_e32 v95, v95, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v93
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_fold_rows_done
	v_add_f32_e32 v97, v97, v96
	v_add_f32_e32 v99, v99, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v97
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_fold_rows_done
	v_add_f32_e32 v101, v101, v100
	v_add_f32_e32 v103, v103, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v101
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_fold_rows_done
	v_add_f32_e32 v105, v105, v104
	v_add_f32_e32 v107, v107, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v107, v107, v105
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_fold_rows_done
	v_add_f32_e32 v109, v109, v108
	v_add_f32_e32 v111, v111, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v111, v111, v109
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_fold_rows_done
	v_add_f32_e32 v113, v113, v112
	v_add_f32_e32 v115, v115, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v115, v115, v113
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_fold_rows_done
	v_add_f32_e32 v117, v117, v116
	v_add_f32_e32 v119, v119, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v119, v119, v117
	.Lb16k4t8_fold_rows_done:
	ds_swizzle_b32 v24, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_swizzle_rows_done
	ds_swizzle_b32 v25, v95 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_swizzle_rows_done
	ds_swizzle_b32 v26, v99 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_swizzle_rows_done
	ds_swizzle_b32 v27, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_swizzle_rows_done
	ds_swizzle_b32 v28, v107 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_swizzle_rows_done
	ds_swizzle_b32 v29, v111 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_swizzle_rows_done
	ds_swizzle_b32 v30, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_swizzle_rows_done
	ds_swizzle_b32 v31, v119 offset:swizzle(BITMASK_PERM,"1pppp")
	.Lb16k4t8_swizzle_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_swizzle_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_swizzle_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_swizzle_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_swizzle_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_swizzle_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_swizzle_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_swizzle_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16k4t8_swizzle_add_rows_done:
	v_cmp_gt_u32_e32 vcc_lo, 24, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_permute8_rows_done
	ds_bpermute_b32 v25, v3, v95
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_permute8_rows_done
	ds_bpermute_b32 v26, v3, v99
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_permute8_rows_done
	ds_bpermute_b32 v27, v3, v103
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_permute8_rows_done
	ds_bpermute_b32 v28, v3, v107
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_permute8_rows_done
	ds_bpermute_b32 v29, v3, v111
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_permute8_rows_done
	ds_bpermute_b32 v30, v3, v115
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_permute8_rows_done
	ds_bpermute_b32 v31, v3, v119
	.Lb16k4t8_permute8_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_permute8_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_permute8_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_permute8_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_permute8_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_permute8_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_permute8_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_permute8_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16k4t8_permute8_add_rows_done:
	v_cmp_gt_u32_e32 vcc_lo, 28, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_permute4_rows_done
	ds_bpermute_b32 v25, v3, v95
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_permute4_rows_done
	ds_bpermute_b32 v26, v3, v99
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_permute4_rows_done
	ds_bpermute_b32 v27, v3, v103
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_permute4_rows_done
	ds_bpermute_b32 v28, v3, v107
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_permute4_rows_done
	ds_bpermute_b32 v29, v3, v111
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_permute4_rows_done
	ds_bpermute_b32 v30, v3, v115
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_permute4_rows_done
	ds_bpermute_b32 v31, v3, v119
	.Lb16k4t8_permute4_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_permute4_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_permute4_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_permute4_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_permute4_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_permute4_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_permute4_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_permute4_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16k4t8_permute4_add_rows_done:
	v_cmp_gt_u32_e32 vcc_lo, 30, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_permute2_rows_done
	ds_bpermute_b32 v25, v3, v95
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_permute2_rows_done
	ds_bpermute_b32 v26, v3, v99
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_permute2_rows_done
	ds_bpermute_b32 v27, v3, v103
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_permute2_rows_done
	ds_bpermute_b32 v28, v3, v107
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_permute2_rows_done
	ds_bpermute_b32 v29, v3, v111
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_permute2_rows_done
	ds_bpermute_b32 v30, v3, v115
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_permute2_rows_done
	ds_bpermute_b32 v31, v3, v119
	.Lb16k4t8_permute2_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_permute2_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_permute2_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_permute2_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_permute2_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_permute2_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_permute2_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_permute2_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16k4t8_permute2_add_rows_done:
	v_cmp_gt_u32_e32 vcc_lo, 31, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v24, v3, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_permute1_rows_done
	ds_bpermute_b32 v25, v3, v95
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_permute1_rows_done
	ds_bpermute_b32 v26, v3, v99
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_permute1_rows_done
	ds_bpermute_b32 v27, v3, v103
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_permute1_rows_done
	ds_bpermute_b32 v28, v3, v107
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_permute1_rows_done
	ds_bpermute_b32 v29, v3, v111
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_permute1_rows_done
	ds_bpermute_b32 v30, v3, v115
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_permute1_rows_done
	ds_bpermute_b32 v31, v3, v119
	.Lb16k4t8_permute1_rows_done:
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v24
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_permute1_add_rows_done
	v_add_f32_e32 v95, v95, v25
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_permute1_add_rows_done
	v_add_f32_e32 v99, v99, v26
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_permute1_add_rows_done
	v_add_f32_e32 v103, v103, v27
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_permute1_add_rows_done
	v_add_f32_e32 v107, v107, v28
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_permute1_add_rows_done
	v_add_f32_e32 v111, v111, v29
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_permute1_add_rows_done
	v_add_f32_e32 v115, v115, v30
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_permute1_add_rows_done
	v_add_f32_e32 v119, v119, v31
	.Lb16k4t8_permute1_add_rows_done:
	v_cmpx_eq_u32_e32 0, v4
	ds_store_b32 v9, v91
	s_cmp_le_u32 s33, 1
	s_cbranch_scc1 .Lb16k4t8_pstored
	ds_store_b32 v9, v95 offset:16
	s_cmp_le_u32 s33, 2
	s_cbranch_scc1 .Lb16k4t8_pstored
	ds_store_b32 v9, v99 offset:32
	s_cmp_le_u32 s33, 3
	s_cbranch_scc1 .Lb16k4t8_pstored
	ds_store_b32 v9, v103 offset:48
	s_cmp_le_u32 s33, 4
	s_cbranch_scc1 .Lb16k4t8_pstored
	ds_store_b32 v9, v107 offset:64
	s_cmp_le_u32 s33, 5
	s_cbranch_scc1 .Lb16k4t8_pstored
	ds_store_b32 v9, v111 offset:80
	s_cmp_le_u32 s33, 6
	s_cbranch_scc1 .Lb16k4t8_pstored
	ds_store_b32 v9, v115 offset:96
	s_cmp_le_u32 s33, 7
	s_cbranch_scc1 .Lb16k4t8_pstored
	ds_store_b32 v9, v119 offset:112
	.Lb16k4t8_pstored:
	s_wait_dscnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4t8_pdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_add_co_i32 s46, s41, 0
	s_mov_b32 s93, s46
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4t8_iskip0
	s_lshl_b32 s45, s93, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4t8_item0_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4t8_item0_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4t8_iskip0:
	s_add_co_i32 s46, s41, 4
	s_mov_b32 s93, s46
	s_cmp_ge_u32 s93, s33
	s_cbranch_scc1 .Lb16k4t8_iskip1
	s_lshl_b32 s45, s93, 4
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s45
	ds_load_b128 v[12:15], v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v12, v13
	v_add_f32_e32 v25, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v24, v25
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 .Lb16k4t8_item1_done
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, s16, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v25, 0x80000000, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, v30, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v27, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v27
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_xor_b32_e32 v25, 0x80000000, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, v30, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v25, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v27, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v30, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0x7fff, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v25, 0xffff, v30
	v_or_b32_e32 v26, 0x10000, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v25, v26, v30, vcc_lo
	v_and_b32_e32 v26, 0x7f800000, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v26
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v24, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v30, 0xffff0000, v24
	.Lb16k4t8_item1_done:
	s_mul_i32 s63, s93, s53
	s_lshl_b32 s45, s98, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s63, s63, s45
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s63
	v_mov_b32_e32 v11, v30
	s_mov_b32 exec_lo, 1
	global_store_b32 v10, v11, s[6:7]
	s_wait_storecnt 0x0
	s_mov_b32 exec_lo, -1
	.Lb16k4t8_iskip1:
	.Lb16k4t8_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_rows_t8_pm_gfx1201_end:
.size gemv_bf16_xf32_k4_rows_t8_pm_gfx1201, .Lgemv_bf16_xf32_k4_rows_t8_pm_gfx1201_end-gemv_bf16_xf32_k4_rows_t8_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_rows_t8_pm_gfx1201
	.amdhsa_group_segment_fixed_size 128
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
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
	.amdhsa_next_free_vgpr 120
	.amdhsa_next_free_sgpr 100
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_rows_t8_pm_gfx1201_end-gemv_bf16_xf32_k4_rows_t8_pm_gfx1201)<<4)&4080)>>4
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
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: gemv_bf16_xf32_x4_c1_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 99
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_x4_c1_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 36
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 99
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 99
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 120
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
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
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 16
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 128
    .name: gemv_bf16_xf32_k4_c1_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_c1_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 36
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
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
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 64
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 128
    .name: gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
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
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 128
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 128
    .name: gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 120
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 99
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 48
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 99
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 64
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
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
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name: gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 48
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
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
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name: gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 64
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: gemv_bf16_xf32_rows_x4_t8_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 102
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_rows_x4_t8_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 120
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
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
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 128
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 128
    .name: gemv_bf16_xf32_k4_rows_t8_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 102
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_rows_t8_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 120
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
