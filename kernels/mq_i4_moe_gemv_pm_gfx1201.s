.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected mq_i4_moe_gemv_mq3i4_r256_gate_up
.globl mq_i4_moe_gemv_mq3i4_r256_gate_up
.p2align 8
.type mq_i4_moe_gemv_mq3i4_r256_gate_up,@function
mq_i4_moe_gemv_mq3i4_r256_gate_up:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s80, s[0:1], 0x58
	s_mov_b32 s30, ttmp9
	s_and_b32 s31, ttmp7, 0xffff
	s_wait_kmcnt 0x0
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_exit
	s_lshl_b32 s51, s31, 2
	s_add_co_u32 s36, s12, s51
	s_add_co_ci_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], 0x0
	s_add_co_u32 s36, s10, s51
	s_add_co_ci_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], 0x0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_exit
	s_cmp_ge_u32 s34, s80
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_exit
	s_lshl_b32 s51, s34, 3
	s_add_co_u32 s36, s8, s51
	s_add_co_ci_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], 0x0
	s_lshr_b32 s48, s23, 7
	s_lshr_b32 s49, s23, 3
	s_mul_i32 s49, s49, 3
	s_mul_i32 s50, s22, s49
	s_mul_i32 s51, s30, s49
	s_wait_kmcnt 0x0
	s_add_co_u32 s44, s38, s51
	s_add_co_ci_u32 s45, s39, 0
	s_mul_i32 s52, s30, s48
	s_lshl_b32 s52, s52, 1
	s_add_co_i32 s52, s52, s50
	s_add_co_u32 s46, s38, s52
	s_add_co_ci_u32 s47, s39, 0
	s_lshr_b32 s77, s22, 1
	s_sub_co_i32 s78, s30, s77
	s_cmp_lt_u32 s30, s77
	s_cselect_b64 s[58:59], s[18:19], s[20:21]
	s_cselect_b32 s79, s30, s78
	s_lshl_b32 s60, s79, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_add_co_i32 s63, s35, -1
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v3, 4, v0
	v_mul_u32_u24_e32 v4, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v4, 1, v4
	v_and_b32_e32 v5, 1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 2, v5
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
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_min_u32_e32 v35, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v35, s32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 2, v35
	global_load_b32 v28, v35, s[14:15]
	v_cvt_f32_u32_e32 v33, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v33, v33
	s_wait_loadcnt 0x0
	v_cvt_f32_u32_e32 v30, v28
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v30, v30, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v31, v30, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v31, v28, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, -1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, v32, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, s26, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v31, v31, v32, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, 1, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, v32, s66
	v_cmp_le_i32_e64 s64, 0, v28
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v29, 0x7fffff00, v30, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v29, v29, s61
	v_mul_lo_u32 v34, v28, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, s60, v34
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v29, 0
	v_mov_b32_e32 v92, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v34, 0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v100, s73
	v_readlane_b32 s74, v29, 1
	v_mov_b32_e32 v93, s74
	v_readlane_b32 s74, v34, 1
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v101, s74
	v_readlane_b32 s75, v29, 2
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v94, s75
	v_readlane_b32 s75, v34, 2
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v102, s75
	v_readlane_b32 s76, v29, 3
	v_mov_b32_e32 v95, s76
	v_readlane_b32 s76, v34, 3
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v103, s76
	v_readlane_b32 s73, v29, 4
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v96, s73
	v_readlane_b32 s73, v34, 4
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v104, s73
	v_readlane_b32 s74, v29, 5
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v97, s74
	v_readlane_b32 s74, v34, 5
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v105, s74
	v_readlane_b32 s75, v29, 6
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v98, s75
	v_readlane_b32 s75, v34, 6
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v106, s75
	v_readlane_b32 s76, v29, 7
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v99, s76
	v_readlane_b32 s76, v34, 7
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v107, s76
	s_and_b32 s54, s48, -4
	s_cmp_eq_u32 s54, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_tail
	.Lmqi4moe_mq3i4_r256_gate_up_quad:
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u8 v9, v1, s[44:45]
	global_load_u8 v10, v1, s[44:45] offset:1
	global_load_u16 v11, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v92, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v93, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v94, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v95, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v8, v96, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v8, v97, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v8, v98, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v8, v99, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_q0_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v11.l
	v_lshl_or_b32 v9, v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v9, v5, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v9, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 1, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, -7, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v11, v24
	v_bfe_u32 v25, v9, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 1, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, -7, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v11, v25
	v_bfe_u32 v26, v9, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, -7, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v11, v26
	v_bfe_u32 v27, v9, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 1, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, -7, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v11, v27
	v_mul_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_done
	v_mul_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_done
	v_mul_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_done
	v_mul_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_done
	v_mul_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_done
	v_mul_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_done
	v_mul_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q0_done
	v_mul_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v14
	.Lmqi4moe_mq3i4_r256_gate_up_q0_done:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u8 v9, v1, s[44:45]
	global_load_u8 v10, v1, s[44:45] offset:1
	global_load_u16 v11, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v92, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v93, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v94, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v95, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v8, v96, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v8, v97, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v8, v98, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v8, v99, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_q1_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v11.l
	v_lshl_or_b32 v9, v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v9, v5, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v9, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 1, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, -7, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v11, v24
	v_bfe_u32 v25, v9, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 1, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, -7, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v11, v25
	v_bfe_u32 v26, v9, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, -7, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v11, v26
	v_bfe_u32 v27, v9, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 1, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, -7, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v11, v27
	v_mul_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_done
	v_mul_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_done
	v_mul_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_done
	v_mul_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_done
	v_mul_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_done
	v_mul_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_done
	v_mul_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v85, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q1_done
	v_mul_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v14
	.Lmqi4moe_mq3i4_r256_gate_up_q1_done:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u8 v9, v1, s[44:45]
	global_load_u8 v10, v1, s[44:45] offset:1
	global_load_u16 v11, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v92, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v93, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v94, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v95, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v8, v96, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v8, v97, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v8, v98, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v8, v99, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_q2_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v11.l
	v_lshl_or_b32 v9, v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v9, v5, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v9, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 1, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, -7, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v11, v24
	v_bfe_u32 v25, v9, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 1, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, -7, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v11, v25
	v_bfe_u32 v26, v9, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, -7, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v11, v26
	v_bfe_u32 v27, v9, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 1, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, -7, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v11, v27
	v_mul_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_done
	v_mul_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_done
	v_mul_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_done
	v_mul_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_done
	v_mul_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78, v78, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_done
	v_mul_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v82, v82, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_done
	v_mul_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v86, v86, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q2_done
	v_mul_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v90, v90, v14
	.Lmqi4moe_mq3i4_r256_gate_up_q2_done:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u8 v9, v1, s[44:45]
	global_load_u8 v10, v1, s[44:45] offset:1
	global_load_u16 v11, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v92, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v93, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v94, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v95, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v8, v96, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v8, v97, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v8, v98, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v8, v99, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_q3_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v11.l
	v_lshl_or_b32 v9, v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v9, v5, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v9, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 1, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, -7, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v11, v24
	v_bfe_u32 v25, v9, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 1, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, -7, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v11, v25
	v_bfe_u32 v26, v9, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, -7, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v11, v26
	v_bfe_u32 v27, v9, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 1, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, -7, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v11, v27
	v_mul_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_done
	v_mul_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_done
	v_mul_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_done
	v_mul_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_done
	v_mul_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v79, v79, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_done
	v_mul_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_done
	v_mul_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v87, v87, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_q3_done
	v_mul_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v14
	.Lmqi4moe_mq3i4_r256_gate_up_q3_done:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s54
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_quad
	.Lmqi4moe_mq3i4_r256_gate_up_tail:
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_reduce
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u8 v9, v1, s[44:45]
	global_load_u8 v10, v1, s[44:45] offset:1
	global_load_u16 v11, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v92, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v93, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v94, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v95, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v8, v96, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v8, v97, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v8, v98, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v8, v99, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t0_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v11.l
	v_lshl_or_b32 v9, v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v9, v5, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v9, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 1, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, -7, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v11, v24
	v_bfe_u32 v25, v9, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 1, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, -7, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v11, v25
	v_bfe_u32 v26, v9, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, -7, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v11, v26
	v_bfe_u32 v27, v9, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 1, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, -7, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v11, v27
	v_mul_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_done
	v_mul_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_done
	v_mul_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_done
	v_mul_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_done
	v_mul_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_done
	v_mul_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_done
	v_mul_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t0_done
	v_mul_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v14
	.Lmqi4moe_mq3i4_r256_gate_up_t0_done:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_reduce
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u8 v9, v1, s[44:45]
	global_load_u8 v10, v1, s[44:45] offset:1
	global_load_u16 v11, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v92, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v93, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v94, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v95, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v8, v96, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v8, v97, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v8, v98, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v8, v99, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t1_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v11.l
	v_lshl_or_b32 v9, v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v9, v5, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v9, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 1, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, -7, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v11, v24
	v_bfe_u32 v25, v9, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 1, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, -7, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v11, v25
	v_bfe_u32 v26, v9, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, -7, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v11, v26
	v_bfe_u32 v27, v9, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 1, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, -7, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v11, v27
	v_mul_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_done
	v_mul_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_done
	v_mul_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_done
	v_mul_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_done
	v_mul_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_done
	v_mul_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_done
	v_mul_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v85, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t1_done
	v_mul_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v14
	.Lmqi4moe_mq3i4_r256_gate_up_t1_done:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_reduce
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u8 v9, v1, s[44:45]
	global_load_u8 v10, v1, s[44:45] offset:1
	global_load_u16 v11, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v92, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v93, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v94, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v95, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v8, v96, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v8, v97, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v8, v98, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v8, v99, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t2_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v11.l
	v_lshl_or_b32 v9, v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v9, v5, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v9, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 1, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, -7, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v11, v24
	v_bfe_u32 v25, v9, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 1, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, -7, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v11, v25
	v_bfe_u32 v26, v9, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, -7, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v11, v26
	v_bfe_u32 v27, v9, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 1, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, -7, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v11, v27
	v_mul_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_done
	v_mul_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_done
	v_mul_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_done
	v_mul_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_done
	v_mul_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78, v78, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_done
	v_mul_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v82, v82, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_done
	v_mul_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v86, v86, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t2_done
	v_mul_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v90, v90, v14
	.Lmqi4moe_mq3i4_r256_gate_up_t2_done:
	s_add_co_i32 s53, s53, 1
	.Lmqi4moe_mq3i4_r256_gate_up_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v16, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v16, v16, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v17, v17, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v18, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v18, v18, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v19, v19, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v20, v20, v0, 2
	v_add_f32_e32 v60, v60, v61
	v_add_f32_e32 v62, v62, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v62
	ds_bpermute_b32 v15, v16, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s68
	ds_bpermute_b32 v15, v17, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s69
	ds_bpermute_b32 v15, v18, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s70
	ds_bpermute_b32 v15, v19, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s71
	ds_bpermute_b32 v15, v20, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v64, v64, v65
	v_add_f32_e32 v66, v66, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v66
	ds_bpermute_b32 v15, v16, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s68
	ds_bpermute_b32 v15, v17, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s69
	ds_bpermute_b32 v15, v18, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s70
	ds_bpermute_b32 v15, v19, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s71
	ds_bpermute_b32 v15, v20, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v68, v68, v69
	v_add_f32_e32 v70, v70, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v70
	ds_bpermute_b32 v15, v16, v68
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v68, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v15, s68
	ds_bpermute_b32 v15, v17, v68
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v68, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v15, s69
	ds_bpermute_b32 v15, v18, v68
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v68, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v15, s70
	ds_bpermute_b32 v15, v19, v68
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v68, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v15, s71
	ds_bpermute_b32 v15, v20, v68
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v68, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v15, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v72, v72, v73
	v_add_f32_e32 v74, v74, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v74
	ds_bpermute_b32 v15, v16, v72
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v72, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v15, s68
	ds_bpermute_b32 v15, v17, v72
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v72, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v15, s69
	ds_bpermute_b32 v15, v18, v72
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v72, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v15, s70
	ds_bpermute_b32 v15, v19, v72
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v72, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v15, s71
	ds_bpermute_b32 v15, v20, v72
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v72, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v15, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v76, v76, v77
	v_add_f32_e32 v78, v78, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v78
	ds_bpermute_b32 v15, v16, v76
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v76, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v15, s68
	ds_bpermute_b32 v15, v17, v76
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v76, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v15, s69
	ds_bpermute_b32 v15, v18, v76
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v76, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v15, s70
	ds_bpermute_b32 v15, v19, v76
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v76, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v15, s71
	ds_bpermute_b32 v15, v20, v76
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v76, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v15, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v80, v80, v81
	v_add_f32_e32 v82, v82, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v82
	ds_bpermute_b32 v15, v16, v80
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v80, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v15, s68
	ds_bpermute_b32 v15, v17, v80
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v80, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v15, s69
	ds_bpermute_b32 v15, v18, v80
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v80, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v15, s70
	ds_bpermute_b32 v15, v19, v80
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v80, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v15, s71
	ds_bpermute_b32 v15, v20, v80
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v80, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v15, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v84, v84, v85
	v_add_f32_e32 v86, v86, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v86
	ds_bpermute_b32 v15, v16, v84
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v84, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v15, s68
	ds_bpermute_b32 v15, v17, v84
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v84, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v15, s69
	ds_bpermute_b32 v15, v18, v84
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v84, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v15, s70
	ds_bpermute_b32 v15, v19, v84
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v84, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v15, s71
	ds_bpermute_b32 v15, v20, v84
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v84, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v15, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v88, v88, v89
	v_add_f32_e32 v90, v90, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v90
	ds_bpermute_b32 v15, v16, v88
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v88, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v15, s68
	ds_bpermute_b32 v15, v17, v88
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v88, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v15, s69
	ds_bpermute_b32 v15, v18, v88
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v88, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v15, s70
	ds_bpermute_b32 v15, v19, v88
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v88, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v15, s71
	ds_bpermute_b32 v15, v20, v88
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v88, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v15, s72
	.Lmqi4moe_mq3i4_r256_gate_up_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v100, v60, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_st_done
	global_store_b32 v101, v64, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_st_done
	global_store_b32 v102, v68, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_st_done
	global_store_b32 v103, v72, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_st_done
	global_store_b32 v104, v76, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_st_done
	global_store_b32 v105, v80, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_st_done
	global_store_b32 v106, v84, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_st_done
	global_store_b32 v107, v88, s[58:59]
	.Lmqi4moe_mq3i4_r256_gate_up_st_done:
	s_wait_storecnt 0x0
	.Lmqi4moe_mq3i4_r256_gate_up_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq3i4_r256_gate_up_end:
.size mq_i4_moe_gemv_mq3i4_r256_gate_up, .Lmq_i4_moe_gemv_mq3i4_r256_gate_up_end-mq_i4_moe_gemv_mq3i4_r256_gate_up
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq3i4_r256_gate_up
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
	.amdhsa_next_free_vgpr 112
	.amdhsa_next_free_sgpr 81
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq3i4_r256_gate_up_end-mq_i4_moe_gemv_mq3i4_r256_gate_up)<<4)&4080)>>4
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
.protected mq_i4_moe_gemv_mq3i4_r128_down
.globl mq_i4_moe_gemv_mq3i4_r128_down
.p2align 8
.type mq_i4_moe_gemv_mq3i4_r128_down,@function
mq_i4_moe_gemv_mq3i4_r128_down:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s80, s[0:1], 0x58
	s_mov_b32 s30, ttmp9
	s_and_b32 s31, ttmp7, 0xffff
	s_wait_kmcnt 0x0
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_exit
	s_lshl_b32 s51, s31, 2
	s_add_co_u32 s36, s12, s51
	s_add_co_ci_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], 0x0
	s_add_co_u32 s36, s10, s51
	s_add_co_ci_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], 0x0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_exit
	s_cmp_ge_u32 s34, s80
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_exit
	s_lshl_b32 s51, s34, 3
	s_add_co_u32 s36, s8, s51
	s_add_co_ci_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], 0x0
	s_lshr_b32 s48, s23, 7
	s_lshr_b32 s49, s23, 3
	s_mul_i32 s49, s49, 3
	s_mul_i32 s50, s22, s49
	s_mul_i32 s51, s30, s49
	s_wait_kmcnt 0x0
	s_add_co_u32 s44, s38, s51
	s_add_co_ci_u32 s45, s39, 0
	s_mul_i32 s52, s30, s48
	s_lshl_b32 s52, s52, 1
	s_add_co_i32 s52, s52, s50
	s_add_co_u32 s46, s38, s52
	s_add_co_ci_u32 s47, s39, 0
	s_mov_b32 s58, s18
	s_mov_b32 s59, s19
	s_lshl_b32 s60, s30, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_add_co_i32 s63, s35, -1
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v3, 4, v0
	v_mul_u32_u24_e32 v4, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v4, 1, v4
	v_and_b32_e32 v5, 1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 2, v5
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_min_u32_e32 v35, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v35, s32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 2, v35
	global_load_b32 v28, v35, s[14:15]
	v_cvt_f32_u32_e32 v33, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v33, v33
	s_wait_loadcnt 0x0
	v_cvt_f32_u32_e32 v30, v28
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v30, v30, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v31, v30, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v31, v28, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, -1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, v32, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, s26, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v31, v31, v32, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, 1, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, v32, s66
	v_cmp_le_i32_e64 s64, 0, v28
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v29, 0x7fffff00, v30, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v29, v29, s61
	v_mul_lo_u32 v34, v28, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, s60, v34
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v29, 0
	v_mov_b32_e32 v68, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v34, 0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v76, s73
	v_readlane_b32 s74, v29, 1
	v_mov_b32_e32 v69, s74
	v_readlane_b32 s74, v34, 1
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v77, s74
	v_readlane_b32 s75, v29, 2
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v70, s75
	v_readlane_b32 s75, v34, 2
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v78, s75
	v_readlane_b32 s76, v29, 3
	v_mov_b32_e32 v71, s76
	v_readlane_b32 s76, v34, 3
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v79, s76
	v_readlane_b32 s73, v29, 4
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v72, s73
	v_readlane_b32 s73, v34, 4
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v80, s73
	v_readlane_b32 s74, v29, 5
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v73, s74
	v_readlane_b32 s74, v34, 5
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v81, s74
	v_readlane_b32 s75, v29, 6
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v74, s75
	v_readlane_b32 s75, v34, 6
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v82, s75
	v_readlane_b32 s76, v29, 7
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v75, s76
	v_readlane_b32 s76, v34, 7
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v83, s76
	s_cmp_eq_u32 s48, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_reduce
	.Lmqi4moe_mq3i4_r128_down_group:
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u8 v9, v1, s[44:45]
	global_load_u8 v10, v1, s[44:45] offset:1
	global_load_u16 v11, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v68, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v69, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v70, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v71, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_xdone
	v_add_nc_u32_e32 v8, v72, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_xdone
	v_add_nc_u32_e32 v8, v73, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_xdone
	v_add_nc_u32_e32 v8, v74, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_xdone
	v_add_nc_u32_e32 v8, v75, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq3i4_r128_down_g_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v11.l
	v_lshl_or_b32 v9, v10, 8, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v9, v5, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v24, v9, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 1, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, -7, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v11, v24
	v_bfe_u32 v25, v9, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 1, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, -7, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v11, v25
	v_bfe_u32 v26, v9, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, -7, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v11, v26
	v_bfe_u32 v27, v9, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 1, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, -7, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v11, v27
	v_mul_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_done
	v_mul_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_done
	v_mul_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_done
	v_mul_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_done
	v_mul_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_done
	v_mul_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_done
	v_mul_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_g_done
	v_mul_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v14
	.Lmqi4moe_mq3i4_r128_down_g_done:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_group
	.Lmqi4moe_mq3i4_r128_down_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v16, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v16, v16, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v17, v17, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v18, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v18, v18, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v19, v19, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v20, v20, v0, 2
	ds_bpermute_b32 v15, v16, v60
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s68
	ds_bpermute_b32 v15, v17, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s69
	ds_bpermute_b32 v15, v18, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s70
	ds_bpermute_b32 v15, v19, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s71
	ds_bpermute_b32 v15, v20, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_red_done
	ds_bpermute_b32 v15, v16, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s68
	ds_bpermute_b32 v15, v17, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s69
	ds_bpermute_b32 v15, v18, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s70
	ds_bpermute_b32 v15, v19, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s71
	ds_bpermute_b32 v15, v20, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_red_done
	ds_bpermute_b32 v15, v16, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s68
	ds_bpermute_b32 v15, v17, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s69
	ds_bpermute_b32 v15, v18, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s70
	ds_bpermute_b32 v15, v19, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s71
	ds_bpermute_b32 v15, v20, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_red_done
	ds_bpermute_b32 v15, v16, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s68
	ds_bpermute_b32 v15, v17, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s69
	ds_bpermute_b32 v15, v18, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s70
	ds_bpermute_b32 v15, v19, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s71
	ds_bpermute_b32 v15, v20, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_red_done
	ds_bpermute_b32 v15, v16, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s68
	ds_bpermute_b32 v15, v17, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s69
	ds_bpermute_b32 v15, v18, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s70
	ds_bpermute_b32 v15, v19, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s71
	ds_bpermute_b32 v15, v20, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_red_done
	ds_bpermute_b32 v15, v16, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s68
	ds_bpermute_b32 v15, v17, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s69
	ds_bpermute_b32 v15, v18, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s70
	ds_bpermute_b32 v15, v19, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s71
	ds_bpermute_b32 v15, v20, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_red_done
	ds_bpermute_b32 v15, v16, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s68
	ds_bpermute_b32 v15, v17, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s69
	ds_bpermute_b32 v15, v18, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s70
	ds_bpermute_b32 v15, v19, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s71
	ds_bpermute_b32 v15, v20, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_red_done
	ds_bpermute_b32 v15, v16, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s68
	ds_bpermute_b32 v15, v17, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s69
	ds_bpermute_b32 v15, v18, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s70
	ds_bpermute_b32 v15, v19, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s71
	ds_bpermute_b32 v15, v20, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s72
	.Lmqi4moe_mq3i4_r128_down_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v76, v60, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_st_done
	global_store_b32 v77, v61, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_st_done
	global_store_b32 v78, v62, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_st_done
	global_store_b32 v79, v63, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_st_done
	global_store_b32 v80, v64, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_st_done
	global_store_b32 v81, v65, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_st_done
	global_store_b32 v82, v66, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_st_done
	global_store_b32 v83, v67, s[58:59]
	.Lmqi4moe_mq3i4_r128_down_st_done:
	s_wait_storecnt 0x0
	.Lmqi4moe_mq3i4_r128_down_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq3i4_r128_down_end:
.size mq_i4_moe_gemv_mq3i4_r128_down, .Lmq_i4_moe_gemv_mq3i4_r128_down_end-mq_i4_moe_gemv_mq3i4_r128_down
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq3i4_r128_down
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
	.amdhsa_next_free_vgpr 88
	.amdhsa_next_free_sgpr 81
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq3i4_r128_down_end-mq_i4_moe_gemv_mq3i4_r128_down)<<4)&4080)>>4
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
.protected mq_i4_moe_gemv_mq4sym_r256_gate_up
.globl mq_i4_moe_gemv_mq4sym_r256_gate_up
.p2align 8
.type mq_i4_moe_gemv_mq4sym_r256_gate_up,@function
mq_i4_moe_gemv_mq4sym_r256_gate_up:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s80, s[0:1], 0x58
	s_mov_b32 s30, ttmp9
	s_and_b32 s31, ttmp7, 0xffff
	s_wait_kmcnt 0x0
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_lshl_b32 s51, s31, 2
	s_add_co_u32 s36, s12, s51
	s_add_co_ci_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], 0x0
	s_add_co_u32 s36, s10, s51
	s_add_co_ci_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], 0x0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_cmp_ge_u32 s34, s80
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_lshl_b32 s51, s34, 3
	s_add_co_u32 s36, s8, s51
	s_add_co_ci_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], 0x0
	s_lshr_b32 s48, s23, 8
	s_mul_i32 s49, s48, 0x88
	s_mul_i32 s51, s30, s49
	s_wait_kmcnt 0x0
	s_add_co_u32 s44, s38, s51
	s_add_co_ci_u32 s45, s39, 0
	s_lshr_b32 s77, s22, 1
	s_sub_co_i32 s78, s30, s77
	s_cmp_lt_u32 s30, s77
	s_cselect_b64 s[58:59], s[18:19], s[20:21]
	s_cselect_b32 s79, s30, s78
	s_lshl_b32 s60, s79, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_add_co_i32 s63, s35, -1
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v3, 5, v0
	v_lshlrev_b32_e32 v4, 2, v0
	v_lshrrev_b32_e32 v6, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v6, 2, v6
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
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_min_u32_e32 v39, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, s32, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 2, v39
	global_load_b32 v32, v39, s[14:15]
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	s_wait_loadcnt 0x0
	v_cvt_f32_u32_e32 v34, v32
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v34, v34, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v35, v34, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v35, v32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, -1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, s26, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v35, v35, v36, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, 1, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s66
	v_cmp_le_i32_e64 s64, 0, v32
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v34, 1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v33, 0x7fffff00, v34, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v33, v33, s61
	v_mul_lo_u32 v38, v32, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v38, s60, v38
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v33, 0
	v_mov_b32_e32 v128, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v38, 0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v136, s73
	v_readlane_b32 s74, v33, 1
	v_mov_b32_e32 v129, s74
	v_readlane_b32 s74, v38, 1
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v137, s74
	v_readlane_b32 s75, v33, 2
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v130, s75
	v_readlane_b32 s75, v38, 2
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v138, s75
	v_readlane_b32 s76, v33, 3
	v_mov_b32_e32 v131, s76
	v_readlane_b32 s76, v38, 3
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v139, s76
	v_readlane_b32 s73, v33, 4
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v132, s73
	v_readlane_b32 s73, v38, 4
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v140, s73
	v_readlane_b32 s74, v33, 5
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v133, s74
	v_readlane_b32 s74, v38, 5
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v141, s74
	v_readlane_b32 s75, v33, 6
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v134, s75
	v_readlane_b32 s75, v38, 6
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v142, s75
	v_readlane_b32 s76, v33, 7
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v135, s76
	v_readlane_b32 s76, v38, 7
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v143, s76
	s_and_b32 s54, s48, -4
	s_cmp_eq_u32 s54, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_tail
	.Lmqi4moe_mq4sym_r256_gate_up_quad:
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v6
	v_add_nc_u32_e32 v7, s56, v3
	global_load_b32 v9, v1, s[44:45] offset:8
	global_load_b32 v12, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v128, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	buffer_load_b128 v[36:39], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v129, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	buffer_load_b128 v[44:47], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v130, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	buffer_load_b128 v[52:55], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v131, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	buffer_load_b128 v[60:63], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v8, v132, v7
	buffer_load_b128 v[64:67], v8, s[40:43], null offen
	buffer_load_b128 v[68:71], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v8, v133, v7
	buffer_load_b128 v[72:75], v8, s[40:43], null offen
	buffer_load_b128 v[76:79], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v8, v134, v7
	buffer_load_b128 v[80:83], v8, s[40:43], null offen
	buffer_load_b128 v[84:87], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v8, v135, v7
	buffer_load_b128 v[88:91], v8, s[40:43], null offen
	buffer_load_b128 v[92:95], v8, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_q0_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v12.l
	v_lshrrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v12, v12.l
	v_bfe_u32 v24, v9, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v11, v24, v12
	v_bfe_u32 v25, v9, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v11, v25, v12
	v_bfe_u32 v26, v9, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v11, v26, v12
	v_bfe_u32 v27, v9, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v11, v27, v12
	v_bfe_u32 v28, v9, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v11, v28, v12
	v_bfe_u32 v29, v9, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v11, v29, v12
	v_bfe_u32 v30, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v11, v30, v12
	v_bfe_u32 v31, v9, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v11, v31, v12
	v_mul_f32_e32 v13, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v96, v96, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v100, v100, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v13, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v104, v104, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v108, v108, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v13, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v112, v112, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v14, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v116, v116, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v13, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v120, v120, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v14, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v124, v124, v14
	.Lmqi4moe_mq4sym_r256_gate_up_q0_done:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v6
	v_add_nc_u32_e32 v7, s56, v3
	global_load_b32 v9, v1, s[44:45] offset:8
	global_load_b32 v12, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v128, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	buffer_load_b128 v[36:39], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v129, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	buffer_load_b128 v[44:47], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v130, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	buffer_load_b128 v[52:55], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v131, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	buffer_load_b128 v[60:63], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v8, v132, v7
	buffer_load_b128 v[64:67], v8, s[40:43], null offen
	buffer_load_b128 v[68:71], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v8, v133, v7
	buffer_load_b128 v[72:75], v8, s[40:43], null offen
	buffer_load_b128 v[76:79], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v8, v134, v7
	buffer_load_b128 v[80:83], v8, s[40:43], null offen
	buffer_load_b128 v[84:87], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v8, v135, v7
	buffer_load_b128 v[88:91], v8, s[40:43], null offen
	buffer_load_b128 v[92:95], v8, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_q1_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v12.l
	v_lshrrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v12, v12.l
	v_bfe_u32 v24, v9, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v11, v24, v12
	v_bfe_u32 v25, v9, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v11, v25, v12
	v_bfe_u32 v26, v9, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v11, v26, v12
	v_bfe_u32 v27, v9, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v11, v27, v12
	v_bfe_u32 v28, v9, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v11, v28, v12
	v_bfe_u32 v29, v9, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v11, v29, v12
	v_bfe_u32 v30, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v11, v30, v12
	v_bfe_u32 v31, v9, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v11, v31, v12
	v_mul_f32_e32 v13, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v101, v101, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v13, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v109, v109, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v13, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v14, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v117, v117, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v13, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v14, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v125, v125, v14
	.Lmqi4moe_mq4sym_r256_gate_up_q1_done:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v6
	v_add_nc_u32_e32 v7, s56, v3
	global_load_b32 v9, v1, s[44:45] offset:8
	global_load_b32 v12, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v128, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	buffer_load_b128 v[36:39], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v129, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	buffer_load_b128 v[44:47], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v130, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	buffer_load_b128 v[52:55], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v131, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	buffer_load_b128 v[60:63], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v8, v132, v7
	buffer_load_b128 v[64:67], v8, s[40:43], null offen
	buffer_load_b128 v[68:71], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v8, v133, v7
	buffer_load_b128 v[72:75], v8, s[40:43], null offen
	buffer_load_b128 v[76:79], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v8, v134, v7
	buffer_load_b128 v[80:83], v8, s[40:43], null offen
	buffer_load_b128 v[84:87], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v8, v135, v7
	buffer_load_b128 v[88:91], v8, s[40:43], null offen
	buffer_load_b128 v[92:95], v8, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_q2_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v12.l
	v_lshrrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v12, v12.l
	v_bfe_u32 v24, v9, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v11, v24, v12
	v_bfe_u32 v25, v9, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v11, v25, v12
	v_bfe_u32 v26, v9, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v11, v26, v12
	v_bfe_u32 v27, v9, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v11, v27, v12
	v_bfe_u32 v28, v9, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v11, v28, v12
	v_bfe_u32 v29, v9, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v11, v29, v12
	v_bfe_u32 v30, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v11, v30, v12
	v_bfe_u32 v31, v9, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v11, v31, v12
	v_mul_f32_e32 v13, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v98, v98, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v102, v102, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v13, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v106, v106, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v110, v110, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v13, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v114, v114, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v14, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v118, v118, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v13, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v122, v122, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v14, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v126, v126, v14
	.Lmqi4moe_mq4sym_r256_gate_up_q2_done:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v6
	v_add_nc_u32_e32 v7, s56, v3
	global_load_b32 v9, v1, s[44:45] offset:8
	global_load_b32 v12, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v128, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	buffer_load_b128 v[36:39], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v129, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	buffer_load_b128 v[44:47], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v130, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	buffer_load_b128 v[52:55], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v131, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	buffer_load_b128 v[60:63], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v8, v132, v7
	buffer_load_b128 v[64:67], v8, s[40:43], null offen
	buffer_load_b128 v[68:71], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v8, v133, v7
	buffer_load_b128 v[72:75], v8, s[40:43], null offen
	buffer_load_b128 v[76:79], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v8, v134, v7
	buffer_load_b128 v[80:83], v8, s[40:43], null offen
	buffer_load_b128 v[84:87], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v8, v135, v7
	buffer_load_b128 v[88:91], v8, s[40:43], null offen
	buffer_load_b128 v[92:95], v8, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_q3_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v12.l
	v_lshrrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v12, v12.l
	v_bfe_u32 v24, v9, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v11, v24, v12
	v_bfe_u32 v25, v9, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v11, v25, v12
	v_bfe_u32 v26, v9, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v11, v26, v12
	v_bfe_u32 v27, v9, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v11, v27, v12
	v_bfe_u32 v28, v9, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v11, v28, v12
	v_bfe_u32 v29, v9, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v11, v29, v12
	v_bfe_u32 v30, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v11, v30, v12
	v_bfe_u32 v31, v9, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v11, v31, v12
	v_mul_f32_e32 v13, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v13, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v107, v107, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v111, v111, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v13, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v115, v115, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v14, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v119, v119, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v13, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v123, v123, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v14, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v127, v127, v14
	.Lmqi4moe_mq4sym_r256_gate_up_q3_done:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s54
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_quad
	.Lmqi4moe_mq4sym_r256_gate_up_tail:
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v6
	v_add_nc_u32_e32 v7, s56, v3
	global_load_b32 v9, v1, s[44:45] offset:8
	global_load_b32 v12, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v128, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	buffer_load_b128 v[36:39], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v129, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	buffer_load_b128 v[44:47], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v130, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	buffer_load_b128 v[52:55], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v131, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	buffer_load_b128 v[60:63], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v8, v132, v7
	buffer_load_b128 v[64:67], v8, s[40:43], null offen
	buffer_load_b128 v[68:71], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v8, v133, v7
	buffer_load_b128 v[72:75], v8, s[40:43], null offen
	buffer_load_b128 v[76:79], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v8, v134, v7
	buffer_load_b128 v[80:83], v8, s[40:43], null offen
	buffer_load_b128 v[84:87], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v8, v135, v7
	buffer_load_b128 v[88:91], v8, s[40:43], null offen
	buffer_load_b128 v[92:95], v8, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t0_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v12.l
	v_lshrrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v12, v12.l
	v_bfe_u32 v24, v9, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v11, v24, v12
	v_bfe_u32 v25, v9, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v11, v25, v12
	v_bfe_u32 v26, v9, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v11, v26, v12
	v_bfe_u32 v27, v9, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v11, v27, v12
	v_bfe_u32 v28, v9, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v11, v28, v12
	v_bfe_u32 v29, v9, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v11, v29, v12
	v_bfe_u32 v30, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v11, v30, v12
	v_bfe_u32 v31, v9, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v11, v31, v12
	v_mul_f32_e32 v13, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v96, v96, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v100, v100, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v13, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v104, v104, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v108, v108, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v13, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v112, v112, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v14, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v116, v116, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v13, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v120, v120, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v14, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v124, v124, v14
	.Lmqi4moe_mq4sym_r256_gate_up_t0_done:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v6
	v_add_nc_u32_e32 v7, s56, v3
	global_load_b32 v9, v1, s[44:45] offset:8
	global_load_b32 v12, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v128, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	buffer_load_b128 v[36:39], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v129, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	buffer_load_b128 v[44:47], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v130, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	buffer_load_b128 v[52:55], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v131, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	buffer_load_b128 v[60:63], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v8, v132, v7
	buffer_load_b128 v[64:67], v8, s[40:43], null offen
	buffer_load_b128 v[68:71], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v8, v133, v7
	buffer_load_b128 v[72:75], v8, s[40:43], null offen
	buffer_load_b128 v[76:79], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v8, v134, v7
	buffer_load_b128 v[80:83], v8, s[40:43], null offen
	buffer_load_b128 v[84:87], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v8, v135, v7
	buffer_load_b128 v[88:91], v8, s[40:43], null offen
	buffer_load_b128 v[92:95], v8, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t1_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v12.l
	v_lshrrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v12, v12.l
	v_bfe_u32 v24, v9, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v11, v24, v12
	v_bfe_u32 v25, v9, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v11, v25, v12
	v_bfe_u32 v26, v9, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v11, v26, v12
	v_bfe_u32 v27, v9, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v11, v27, v12
	v_bfe_u32 v28, v9, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v11, v28, v12
	v_bfe_u32 v29, v9, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v11, v29, v12
	v_bfe_u32 v30, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v11, v30, v12
	v_bfe_u32 v31, v9, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v11, v31, v12
	v_mul_f32_e32 v13, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v101, v101, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v13, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v109, v109, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v13, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v14, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v117, v117, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v13, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v14, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v125, v125, v14
	.Lmqi4moe_mq4sym_r256_gate_up_t1_done:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v6
	v_add_nc_u32_e32 v7, s56, v3
	global_load_b32 v9, v1, s[44:45] offset:8
	global_load_b32 v12, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v128, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	buffer_load_b128 v[36:39], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v129, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	buffer_load_b128 v[44:47], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v130, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	buffer_load_b128 v[52:55], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v131, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	buffer_load_b128 v[60:63], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v8, v132, v7
	buffer_load_b128 v[64:67], v8, s[40:43], null offen
	buffer_load_b128 v[68:71], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v8, v133, v7
	buffer_load_b128 v[72:75], v8, s[40:43], null offen
	buffer_load_b128 v[76:79], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v8, v134, v7
	buffer_load_b128 v[80:83], v8, s[40:43], null offen
	buffer_load_b128 v[84:87], v8, s[40:43], null offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v8, v135, v7
	buffer_load_b128 v[88:91], v8, s[40:43], null offen
	buffer_load_b128 v[92:95], v8, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t2_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v12.l
	v_lshrrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v12, v12.l
	v_bfe_u32 v24, v9, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v11, v24, v12
	v_bfe_u32 v25, v9, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v11, v25, v12
	v_bfe_u32 v26, v9, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v11, v26, v12
	v_bfe_u32 v27, v9, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v11, v27, v12
	v_bfe_u32 v28, v9, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v11, v28, v12
	v_bfe_u32 v29, v9, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v11, v29, v12
	v_bfe_u32 v30, v9, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v11, v30, v12
	v_bfe_u32 v31, v9, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v11, v31, v12
	v_mul_f32_e32 v13, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v98, v98, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v102, v102, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v13, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v106, v106, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v110, v110, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v13, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v114, v114, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v14, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v118, v118, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v13, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v28, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v29, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v30, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v31, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v122, v122, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v14, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v28, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v29, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v30, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v31, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v126, v126, v14
	.Lmqi4moe_mq4sym_r256_gate_up_t2_done:
	s_add_co_i32 s53, s53, 1
	.Lmqi4moe_mq4sym_r256_gate_up_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v16, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v16, v16, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v17, v17, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v18, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v18, v18, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v19, v19, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v20, v20, v0, 2
	v_add_f32_e32 v96, v96, v97
	v_add_f32_e32 v98, v98, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v96, v96, v98
	ds_bpermute_b32 v15, v16, v96
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v96, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v15, s68
	ds_bpermute_b32 v15, v17, v96
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v96, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v15, s69
	ds_bpermute_b32 v15, v18, v96
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v96, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v15, s70
	ds_bpermute_b32 v15, v19, v96
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v96, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v15, s71
	ds_bpermute_b32 v15, v20, v96
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v96, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v15, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v100, v100, v101
	v_add_f32_e32 v102, v102, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v100, v100, v102
	ds_bpermute_b32 v15, v16, v100
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v100, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v15, s68
	ds_bpermute_b32 v15, v17, v100
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v100, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v15, s69
	ds_bpermute_b32 v15, v18, v100
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v100, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v15, s70
	ds_bpermute_b32 v15, v19, v100
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v100, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v15, s71
	ds_bpermute_b32 v15, v20, v100
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v100, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v15, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v104, v104, v105
	v_add_f32_e32 v106, v106, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v104, v104, v106
	ds_bpermute_b32 v15, v16, v104
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v104, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v15, s68
	ds_bpermute_b32 v15, v17, v104
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v104, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v15, s69
	ds_bpermute_b32 v15, v18, v104
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v104, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v15, s70
	ds_bpermute_b32 v15, v19, v104
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v104, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v15, s71
	ds_bpermute_b32 v15, v20, v104
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v104, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v15, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v108, v108, v109
	v_add_f32_e32 v110, v110, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v108, v108, v110
	ds_bpermute_b32 v15, v16, v108
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v108, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v15, s68
	ds_bpermute_b32 v15, v17, v108
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v108, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v15, s69
	ds_bpermute_b32 v15, v18, v108
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v108, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v15, s70
	ds_bpermute_b32 v15, v19, v108
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v108, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v15, s71
	ds_bpermute_b32 v15, v20, v108
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v108, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v15, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v112, v112, v113
	v_add_f32_e32 v114, v114, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v112, v112, v114
	ds_bpermute_b32 v15, v16, v112
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v112, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v15, s68
	ds_bpermute_b32 v15, v17, v112
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v112, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v15, s69
	ds_bpermute_b32 v15, v18, v112
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v112, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v15, s70
	ds_bpermute_b32 v15, v19, v112
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v112, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v15, s71
	ds_bpermute_b32 v15, v20, v112
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v112, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v15, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v116, v116, v117
	v_add_f32_e32 v118, v118, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v116, v116, v118
	ds_bpermute_b32 v15, v16, v116
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v116, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v15, s68
	ds_bpermute_b32 v15, v17, v116
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v116, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v15, s69
	ds_bpermute_b32 v15, v18, v116
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v116, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v15, s70
	ds_bpermute_b32 v15, v19, v116
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v116, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v15, s71
	ds_bpermute_b32 v15, v20, v116
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v116, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v15, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v120, v120, v121
	v_add_f32_e32 v122, v122, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v120, v120, v122
	ds_bpermute_b32 v15, v16, v120
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v120, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v15, s68
	ds_bpermute_b32 v15, v17, v120
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v120, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v15, s69
	ds_bpermute_b32 v15, v18, v120
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v120, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v15, s70
	ds_bpermute_b32 v15, v19, v120
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v120, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v15, s71
	ds_bpermute_b32 v15, v20, v120
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v120, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v15, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v124, v124, v125
	v_add_f32_e32 v126, v126, v127
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v124, v124, v126
	ds_bpermute_b32 v15, v16, v124
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v124, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v124, v124, v15, s68
	ds_bpermute_b32 v15, v17, v124
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v124, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v124, v124, v15, s69
	ds_bpermute_b32 v15, v18, v124
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v124, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v124, v124, v15, s70
	ds_bpermute_b32 v15, v19, v124
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v124, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v124, v124, v15, s71
	ds_bpermute_b32 v15, v20, v124
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v124, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v124, v124, v15, s72
	.Lmqi4moe_mq4sym_r256_gate_up_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v136, v96, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v137, v100, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v138, v104, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v139, v108, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v140, v112, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v141, v116, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v142, v120, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v143, v124, s[58:59]
	.Lmqi4moe_mq4sym_r256_gate_up_st_done:
	s_wait_storecnt 0x0
	.Lmqi4moe_mq4sym_r256_gate_up_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq4sym_r256_gate_up_end:
.size mq_i4_moe_gemv_mq4sym_r256_gate_up, .Lmq_i4_moe_gemv_mq4sym_r256_gate_up_end-mq_i4_moe_gemv_mq4sym_r256_gate_up
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq4sym_r256_gate_up
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
	.amdhsa_next_free_vgpr 144
	.amdhsa_next_free_sgpr 81
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq4sym_r256_gate_up_end-mq_i4_moe_gemv_mq4sym_r256_gate_up)<<4)&4080)>>4
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
.protected mq_i4_moe_gemv_mq4sym_r128_down
.globl mq_i4_moe_gemv_mq4sym_r128_down
.p2align 8
.type mq_i4_moe_gemv_mq4sym_r128_down,@function
mq_i4_moe_gemv_mq4sym_r128_down:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s80, s[0:1], 0x58
	s_mov_b32 s30, ttmp9
	s_and_b32 s31, ttmp7, 0xffff
	s_wait_kmcnt 0x0
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_lshl_b32 s51, s31, 2
	s_add_co_u32 s36, s12, s51
	s_add_co_ci_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], 0x0
	s_add_co_u32 s36, s10, s51
	s_add_co_ci_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], 0x0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_cmp_ge_u32 s34, s80
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_lshl_b32 s51, s34, 3
	s_add_co_u32 s36, s8, s51
	s_add_co_ci_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], 0x0
	s_lshr_b32 s48, s23, 7
	s_mul_i32 s49, s48, 0x44
	s_mul_i32 s51, s30, s49
	s_wait_kmcnt 0x0
	s_add_co_u32 s44, s38, s51
	s_add_co_ci_u32 s45, s39, 0
	s_mov_b32 s58, s18
	s_mov_b32 s59, s19
	s_lshl_b32 s60, s30, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_add_co_i32 s63, s35, -1
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v3, 4, v0
	v_lshlrev_b32_e32 v4, 1, v0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_min_u32_e32 v35, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v35, s32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 2, v35
	global_load_b32 v28, v35, s[14:15]
	v_cvt_f32_u32_e32 v33, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v33, v33
	s_wait_loadcnt 0x0
	v_cvt_f32_u32_e32 v30, v28
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v30, v30, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v31, v30, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v31, v28, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, -1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, v32, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, s26, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v31, v31, v32, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v32, 1, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, v32, s66
	v_cmp_le_i32_e64 s64, 0, v28
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v29, 0x7fffff00, v30, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v29, v29, s61
	v_mul_lo_u32 v34, v28, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, s60, v34
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v29, 0
	v_mov_b32_e32 v68, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v34, 0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v76, s73
	v_readlane_b32 s74, v29, 1
	v_mov_b32_e32 v69, s74
	v_readlane_b32 s74, v34, 1
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v77, s74
	v_readlane_b32 s75, v29, 2
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v70, s75
	v_readlane_b32 s75, v34, 2
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v78, s75
	v_readlane_b32 s76, v29, 3
	v_mov_b32_e32 v71, s76
	v_readlane_b32 s76, v34, 3
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v79, s76
	v_readlane_b32 s73, v29, 4
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v72, s73
	v_readlane_b32 s73, v34, 4
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v80, s73
	v_readlane_b32 s74, v29, 5
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v73, s74
	v_readlane_b32 s74, v34, 5
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v81, s74
	v_readlane_b32 s75, v29, 6
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v74, s75
	v_readlane_b32 s75, v34, 6
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v82, s75
	v_readlane_b32 s76, v29, 7
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v75, s76
	v_readlane_b32 s76, v34, 7
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v83, s76
	s_cmp_eq_u32 s48, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_reduce
	.Lmqi4moe_mq4sym_r128_down_group:
	s_mul_i32 s55, s53, 0x44
	s_lshl_b32 s56, s53, 9
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s55
	v_add_nc_u32_e32 v7, s56, v3
	global_load_u16 v9, v1, s[44:45] offset:4
	global_load_b32 v12, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, v68, v7
	buffer_load_b128 v[28:31], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v8, v69, v7
	buffer_load_b128 v[32:35], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v8, v70, v7
	buffer_load_b128 v[36:39], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v8, v71, v7
	buffer_load_b128 v[40:43], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	v_add_nc_u32_e32 v8, v72, v7
	buffer_load_b128 v[44:47], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	v_add_nc_u32_e32 v8, v73, v7
	buffer_load_b128 v[48:51], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	v_add_nc_u32_e32 v8, v74, v7
	buffer_load_b128 v[52:55], v8, s[40:43], null offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	v_add_nc_u32_e32 v8, v75, v7
	buffer_load_b128 v[56:59], v8, s[40:43], null offen
	.Lmqi4moe_mq4sym_r128_down_g_xdone:
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v11, v12.l
	v_lshrrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v12, v12.l
	v_bfe_u32 v24, v9, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v11, v24, v12
	v_bfe_u32 v25, v9, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v11, v25, v12
	v_bfe_u32 v26, v9, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v11, v26, v12
	v_bfe_u32 v27, v9, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v11, v27, v12
	v_mul_f32_e32 v13, v25, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v13
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v14, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v14
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v13, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v13
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v14, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v14
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v13, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v13
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v14, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v14
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v13, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v13
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v14, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v14
	.Lmqi4moe_mq4sym_r128_down_g_done:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_group
	.Lmqi4moe_mq4sym_r128_down_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v16, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v16, v16, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v17, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v17, v17, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v18, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v18, v18, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v19, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v19, v19, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v20, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v20, v20, v0, 2
	ds_bpermute_b32 v15, v16, v60
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s68
	ds_bpermute_b32 v15, v17, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s69
	ds_bpermute_b32 v15, v18, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s70
	ds_bpermute_b32 v15, v19, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s71
	ds_bpermute_b32 v15, v20, v60
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v60, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v15, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v15, v16, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s68
	ds_bpermute_b32 v15, v17, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s69
	ds_bpermute_b32 v15, v18, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s70
	ds_bpermute_b32 v15, v19, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s71
	ds_bpermute_b32 v15, v20, v61
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v61, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v15, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v15, v16, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s68
	ds_bpermute_b32 v15, v17, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s69
	ds_bpermute_b32 v15, v18, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s70
	ds_bpermute_b32 v15, v19, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s71
	ds_bpermute_b32 v15, v20, v62
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v62, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v15, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v15, v16, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s68
	ds_bpermute_b32 v15, v17, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s69
	ds_bpermute_b32 v15, v18, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s70
	ds_bpermute_b32 v15, v19, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s71
	ds_bpermute_b32 v15, v20, v63
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v63, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v15, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v15, v16, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s68
	ds_bpermute_b32 v15, v17, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s69
	ds_bpermute_b32 v15, v18, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s70
	ds_bpermute_b32 v15, v19, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s71
	ds_bpermute_b32 v15, v20, v64
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v64, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v15, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v15, v16, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s68
	ds_bpermute_b32 v15, v17, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s69
	ds_bpermute_b32 v15, v18, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s70
	ds_bpermute_b32 v15, v19, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s71
	ds_bpermute_b32 v15, v20, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v65, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v15, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v15, v16, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s68
	ds_bpermute_b32 v15, v17, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s69
	ds_bpermute_b32 v15, v18, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s70
	ds_bpermute_b32 v15, v19, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s71
	ds_bpermute_b32 v15, v20, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v66, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v15, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v15, v16, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s68
	ds_bpermute_b32 v15, v17, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s69
	ds_bpermute_b32 v15, v18, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s70
	ds_bpermute_b32 v15, v19, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s71
	ds_bpermute_b32 v15, v20, v67
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v67, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v15, s72
	.Lmqi4moe_mq4sym_r128_down_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v76, v60, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v77, v61, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v78, v62, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v79, v63, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v80, v64, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v81, v65, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v82, v66, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v83, v67, s[58:59]
	.Lmqi4moe_mq4sym_r128_down_st_done:
	s_wait_storecnt 0x0
	.Lmqi4moe_mq4sym_r128_down_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq4sym_r128_down_end:
.size mq_i4_moe_gemv_mq4sym_r128_down, .Lmq_i4_moe_gemv_mq4sym_r128_down_end-mq_i4_moe_gemv_mq4sym_r128_down
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq4sym_r128_down
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
	.amdhsa_next_free_vgpr 88
	.amdhsa_next_free_sgpr 81
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq4sym_r128_down_end-mq_i4_moe_gemv_mq4sym_r128_down)<<4)&4080)>>4
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
.protected mq_i4_moe_gemv_mq3i4_r256_gate_up_t8
.globl mq_i4_moe_gemv_mq3i4_r256_gate_up_t8
.p2align 8
.type mq_i4_moe_gemv_mq3i4_r256_gate_up_t8,@function
mq_i4_moe_gemv_mq3i4_r256_gate_up_t8:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s84, s[0:1], 0x58
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 4, v1
	v_lshlrev_b32_e32 v3, 4, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s80, v2
	s_mov_b32 s81, ttmp9
	s_and_b32 s31, ttmp7, 0xffff
	s_lshr_b32 s85, ttmp7, 16
	s_wait_kmcnt 0x0
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_cmp_eq_u32 s24, 0x50
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_lshl_b32 s51, s85, 3
	s_cmp_ge_u32 s51, s25
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_cmp_eq_u32 s26, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_and_b32 s51, s23, 0x7f
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_mul_i32 s51, s85, 0x788
	s_add_co_u32 s10, s10, s51
	s_add_co_ci_u32 s11, s11, 0
	s_add_co_u32 s12, s12, s51
	s_add_co_ci_u32 s13, s13, 0
	s_add_co_u32 s14, s14, s51
	s_add_co_ci_u32 s15, s15, 0
	s_lshl_b32 s81, s81, 3
	s_cmp_ge_u32 s81, s22
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_lshl_b32 s51, s31, 2
	s_add_co_u32 s36, s12, s51
	s_add_co_ci_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], 0x0
	s_add_co_u32 s36, s10, s51
	s_add_co_ci_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], 0x0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_mov_b32 s51, 0x50
	s_cmp_gt_u32 s32, s51
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s34, s84
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_add_co_i32 s63, s35, -1
	v_min_u32_e32 v39, s63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, s32, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 2, v39
	global_load_b32 v32, v39, s[14:15]
	s_wait_loadcnt 0x0
	v_cmp_le_u32_e32 vcc_lo, s51, v32
	s_cbranch_vccnz .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	s_lshl_b32 s51, s34, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s36, s8, s51
	s_add_co_ci_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], 0x0
	s_add_co_i32 s30, s81, s80
	s_lshr_b32 s48, s23, 7
	s_lshr_b32 s49, s23, 3
	s_mul_i32 s49, s49, 3
	s_mul_i32 s50, s22, s49
	s_mul_i32 s51, s30, s49
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s44, s38, s51
	s_add_co_ci_u32 s45, s39, 0
	s_mul_i32 s52, s30, s48
	s_lshl_b32 s52, s52, 1
	s_add_co_i32 s52, s52, s50
	s_add_co_u32 s46, s38, s52
	s_add_co_ci_u32 s47, s39, 0
	s_lshr_b32 s77, s22, 1
	s_sub_co_i32 s78, s30, s77
	s_cmp_lt_u32 s30, s77
	s_cselect_b64 s[58:59], s[18:19], s[20:21]
	s_cselect_b32 s79, s30, s78
	s_lshl_b32 s60, s79, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v7, 4, v1
	v_mul_u32_u24_e32 v8, 3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v8, 1, v8
	v_and_b32_e32 v9, 1, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 2, v9
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
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	v_cvt_f32_u32_e32 v34, v32
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v34, v34, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v35, v34, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v35, v32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, -1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, s26, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v35, v35, v36, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, 1, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s66
	v_cmp_le_i32_e64 s64, 0, v32
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v34, 1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v40, 0x7fffff00, v34, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v40, v40, s61
	v_mul_lo_u32 v41, v32, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v41, s60, v41
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	s_and_b32 s54, s48, -4
	s_cmp_eq_u32 s54, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_tail
	.Lmqi4moe_mq3i4_r256_gate_up_t8_quad:
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s57
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_wdone
	global_load_u8 v13, v5, s[44:45]
	global_load_u8 v14, v5, s[44:45] offset:1
	global_load_u16 v15, v6, s[46:47]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q0_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_udone
	v_cvt_f32_f16_e64 v15, v15.l
	v_lshl_or_b32 v13, v14, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v13, v9, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v28, v13, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, -7, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v15, v28
	v_bfe_u32 v29, v13, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, -7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v15, v29
	v_bfe_u32 v30, v13, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, -7, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v15, v30
	v_bfe_u32 v31, v13, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, -7, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v15, v31
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v18
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q0_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q0_gend:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s57
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_wdone
	global_load_u8 v13, v5, s[44:45]
	global_load_u8 v14, v5, s[44:45] offset:1
	global_load_u16 v15, v6, s[46:47]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q1_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_udone
	v_cvt_f32_f16_e64 v15, v15.l
	v_lshl_or_b32 v13, v14, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v13, v9, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v28, v13, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, -7, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v15, v28
	v_bfe_u32 v29, v13, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, -7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v15, v29
	v_bfe_u32 v30, v13, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, -7, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v15, v30
	v_bfe_u32 v31, v13, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, -7, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v15, v31
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v18
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q1_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q1_gend:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s57
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_wdone
	global_load_u8 v13, v5, s[44:45]
	global_load_u8 v14, v5, s[44:45] offset:1
	global_load_u16 v15, v6, s[46:47]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q2_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_udone
	v_cvt_f32_f16_e64 v15, v15.l
	v_lshl_or_b32 v13, v14, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v13, v9, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v28, v13, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, -7, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v15, v28
	v_bfe_u32 v29, v13, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, -7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v15, v29
	v_bfe_u32 v30, v13, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, -7, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v15, v30
	v_bfe_u32 v31, v13, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, -7, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v15, v31
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v18
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q2_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q2_gend:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s57
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_wdone
	global_load_u8 v13, v5, s[44:45]
	global_load_u8 v14, v5, s[44:45] offset:1
	global_load_u16 v15, v6, s[46:47]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q3_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_udone
	v_cvt_f32_f16_e64 v15, v15.l
	v_lshl_or_b32 v13, v14, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v13, v9, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v28, v13, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, -7, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v15, v28
	v_bfe_u32 v29, v13, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, -7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v15, v29
	v_bfe_u32 v30, v13, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, -7, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v15, v30
	v_bfe_u32 v31, v13, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, -7, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v15, v31
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v18
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q3_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq3i4_r256_gate_up_t8_q3_gend:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s54
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r256_gate_up_t8_quad
	.Lmqi4moe_mq3i4_r256_gate_up_t8_tail:
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_reduce
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s57
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_wdone
	global_load_u8 v13, v5, s[44:45]
	global_load_u8 v14, v5, s[44:45] offset:1
	global_load_u16 v15, v6, s[46:47]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t0_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_udone
	v_cvt_f32_f16_e64 v15, v15.l
	v_lshl_or_b32 v13, v14, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v13, v9, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v28, v13, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, -7, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v15, v28
	v_bfe_u32 v29, v13, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, -7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v15, v29
	v_bfe_u32 v30, v13, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, -7, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v15, v30
	v_bfe_u32 v31, v13, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, -7, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v15, v31
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v18
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t0_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t0_gend:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_reduce
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s57
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_wdone
	global_load_u8 v13, v5, s[44:45]
	global_load_u8 v14, v5, s[44:45] offset:1
	global_load_u16 v15, v6, s[46:47]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t1_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_udone
	v_cvt_f32_f16_e64 v15, v15.l
	v_lshl_or_b32 v13, v14, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v13, v9, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v28, v13, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, -7, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v15, v28
	v_bfe_u32 v29, v13, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, -7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v15, v29
	v_bfe_u32 v30, v13, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, -7, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v15, v30
	v_bfe_u32 v31, v13, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, -7, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v15, v31
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v18
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t1_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t1_gend:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_reduce
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s57
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_wdone
	global_load_u8 v13, v5, s[44:45]
	global_load_u8 v14, v5, s[44:45] offset:1
	global_load_u16 v15, v6, s[46:47]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t2_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_udone
	v_cvt_f32_f16_e64 v15, v15.l
	v_lshl_or_b32 v13, v14, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v13, v9, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v28, v13, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, -7, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v15, v28
	v_bfe_u32 v29, v13, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, -7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v15, v29
	v_bfe_u32 v30, v13, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, -7, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v15, v30
	v_bfe_u32 v31, v13, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, -7, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v15, v31
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v18
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t2_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq3i4_r256_gate_up_t8_t2_gend:
	s_add_co_i32 s53, s53, 1
	.Lmqi4moe_mq3i4_r256_gate_up_t8_reduce:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_exit
	v_cmp_gt_u32_e64 s68, 16, v1
	v_cndmask_b32_e64 v23, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v23, v23, v1, 2
	v_cmp_gt_u32_e64 s69, 24, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v24, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v24, v24, v1, 2
	v_cmp_gt_u32_e64 s70, 28, v1
	v_cndmask_b32_e64 v25, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v25, v25, v1, 2
	v_cmp_gt_u32_e64 s71, 30, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v26, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v26, v26, v1, 2
	v_cmp_gt_u32_e64 s72, 31, v1
	v_cndmask_b32_e64 v27, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v27, v27, v1, 2
	v_add_f32_e32 v42, v42, v43
	v_add_f32_e32 v44, v44, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v44
	v_add_f32_e32 v46, v46, v47
	v_add_f32_e32 v48, v48, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v48
	v_add_f32_e32 v50, v50, v51
	v_add_f32_e32 v52, v52, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v52
	v_add_f32_e32 v54, v54, v55
	v_add_f32_e32 v56, v56, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v56
	ds_bpermute_b32 v19, v23, v42
	ds_bpermute_b32 v20, v23, v46
	ds_bpermute_b32 v21, v23, v50
	ds_bpermute_b32 v22, v23, v54
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s68
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v46, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v20, s68
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v50, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v21, s68
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v54, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v22, s68
	ds_bpermute_b32 v19, v24, v42
	ds_bpermute_b32 v20, v24, v46
	ds_bpermute_b32 v21, v24, v50
	ds_bpermute_b32 v22, v24, v54
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s69
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v46, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v20, s69
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v50, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v21, s69
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v54, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v22, s69
	ds_bpermute_b32 v19, v25, v42
	ds_bpermute_b32 v20, v25, v46
	ds_bpermute_b32 v21, v25, v50
	ds_bpermute_b32 v22, v25, v54
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s70
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v46, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v20, s70
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v50, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v21, s70
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v54, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v22, s70
	ds_bpermute_b32 v19, v26, v42
	ds_bpermute_b32 v20, v26, v46
	ds_bpermute_b32 v21, v26, v50
	ds_bpermute_b32 v22, v26, v54
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s71
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v46, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v20, s71
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v50, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v21, s71
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v54, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v22, s71
	ds_bpermute_b32 v19, v27, v42
	ds_bpermute_b32 v20, v27, v46
	ds_bpermute_b32 v21, v27, v50
	ds_bpermute_b32 v22, v27, v54
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s72
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v46, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v20, s72
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v50, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v21, s72
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v54, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v22, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_red_done
	v_add_f32_e32 v58, v58, v59
	v_add_f32_e32 v60, v60, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v60
	v_add_f32_e32 v62, v62, v63
	v_add_f32_e32 v64, v64, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v64
	v_add_f32_e32 v66, v66, v67
	v_add_f32_e32 v68, v68, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v68
	v_add_f32_e32 v70, v70, v71
	v_add_f32_e32 v72, v72, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v72
	ds_bpermute_b32 v19, v23, v58
	ds_bpermute_b32 v20, v23, v62
	ds_bpermute_b32 v21, v23, v66
	ds_bpermute_b32 v22, v23, v70
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v58, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v19, s68
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v62, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v20, s68
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v66, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v21, s68
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v70, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v22, s68
	ds_bpermute_b32 v19, v24, v58
	ds_bpermute_b32 v20, v24, v62
	ds_bpermute_b32 v21, v24, v66
	ds_bpermute_b32 v22, v24, v70
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v58, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v19, s69
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v62, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v20, s69
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v66, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v21, s69
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v70, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v22, s69
	ds_bpermute_b32 v19, v25, v58
	ds_bpermute_b32 v20, v25, v62
	ds_bpermute_b32 v21, v25, v66
	ds_bpermute_b32 v22, v25, v70
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v58, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v19, s70
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v62, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v20, s70
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v66, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v21, s70
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v70, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v22, s70
	ds_bpermute_b32 v19, v26, v58
	ds_bpermute_b32 v20, v26, v62
	ds_bpermute_b32 v21, v26, v66
	ds_bpermute_b32 v22, v26, v70
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v58, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v19, s71
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v62, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v20, s71
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v66, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v21, s71
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v70, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v22, s71
	ds_bpermute_b32 v19, v27, v58
	ds_bpermute_b32 v20, v27, v62
	ds_bpermute_b32 v21, v27, v66
	ds_bpermute_b32 v22, v27, v70
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v58, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v19, s72
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v62, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v20, s72
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v66, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v21, s72
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v70, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v22, s72
	.Lmqi4moe_mq3i4_r256_gate_up_t8_red_done:
	v_cmpx_eq_u32_e32 0, v1
	v_readlane_b32 s73, v41, 0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s73
	global_store_b32 v12, v42, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_st_done
	v_readlane_b32 s74, v41, 1
	s_wait_storecnt 0x0
	v_mov_b32_e32 v12, s74
	global_store_b32 v12, v46, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_st_done
	v_readlane_b32 s75, v41, 2
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s75
	global_store_b32 v12, v50, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_st_done
	v_readlane_b32 s76, v41, 3
	s_wait_storecnt 0x0
	v_mov_b32_e32 v12, s76
	global_store_b32 v12, v54, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_st_done
	v_readlane_b32 s73, v41, 4
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s73
	global_store_b32 v12, v58, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_st_done
	v_readlane_b32 s74, v41, 5
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s74
	global_store_b32 v12, v62, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_st_done
	v_readlane_b32 s75, v41, 6
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s75
	global_store_b32 v12, v66, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r256_gate_up_t8_st_done
	v_readlane_b32 s76, v41, 7
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s76
	global_store_b32 v12, v70, s[58:59]
	.Lmqi4moe_mq3i4_r256_gate_up_t8_st_done:
	s_wait_storecnt 0x0
	.Lmqi4moe_mq3i4_r256_gate_up_t8_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq3i4_r256_gate_up_t8_end:
.size mq_i4_moe_gemv_mq3i4_r256_gate_up_t8, .Lmq_i4_moe_gemv_mq3i4_r256_gate_up_t8_end-mq_i4_moe_gemv_mq3i4_r256_gate_up_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq3i4_r256_gate_up_t8
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
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 1
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 80
	.amdhsa_next_free_sgpr 86
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq3i4_r256_gate_up_t8_end-mq_i4_moe_gemv_mq3i4_r256_gate_up_t8)<<4)&4080)>>4
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
.protected mq_i4_moe_gemv_mq3i4_r128_down_t8
.globl mq_i4_moe_gemv_mq3i4_r128_down_t8
.p2align 8
.type mq_i4_moe_gemv_mq3i4_r128_down_t8,@function
mq_i4_moe_gemv_mq3i4_r128_down_t8:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s84, s[0:1], 0x58
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 4, v1
	v_lshlrev_b32_e32 v3, 4, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s80, v2
	s_mov_b32 s81, ttmp9
	s_and_b32 s31, ttmp7, 0xffff
	s_lshr_b32 s85, ttmp7, 16
	s_wait_kmcnt 0x0
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_cmp_eq_u32 s24, 0x50
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_lshl_b32 s51, s85, 3
	s_cmp_ge_u32 s51, s25
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_cmp_eq_u32 s26, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_and_b32 s51, s23, 0x7f
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_mul_i32 s51, s85, 0x788
	s_add_co_u32 s10, s10, s51
	s_add_co_ci_u32 s11, s11, 0
	s_add_co_u32 s12, s12, s51
	s_add_co_ci_u32 s13, s13, 0
	s_add_co_u32 s14, s14, s51
	s_add_co_ci_u32 s15, s15, 0
	s_lshl_b32 s81, s81, 3
	s_cmp_ge_u32 s81, s22
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_lshl_b32 s51, s31, 2
	s_add_co_u32 s36, s12, s51
	s_add_co_ci_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], 0x0
	s_add_co_u32 s36, s10, s51
	s_add_co_ci_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], 0x0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_mov_b32 s51, 0x50
	s_cmp_gt_u32 s32, s51
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s34, s84
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_add_co_i32 s63, s35, -1
	v_min_u32_e32 v39, s63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, s32, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 2, v39
	global_load_b32 v32, v39, s[14:15]
	s_wait_loadcnt 0x0
	v_cmp_le_u32_e32 vcc_lo, s51, v32
	s_cbranch_vccnz .Lmqi4moe_mq3i4_r128_down_t8_exit
	s_lshl_b32 s51, s34, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s36, s8, s51
	s_add_co_ci_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], 0x0
	s_add_co_i32 s30, s81, s80
	s_lshr_b32 s48, s23, 7
	s_lshr_b32 s49, s23, 3
	s_mul_i32 s49, s49, 3
	s_mul_i32 s50, s22, s49
	s_mul_i32 s51, s30, s49
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s44, s38, s51
	s_add_co_ci_u32 s45, s39, 0
	s_mul_i32 s52, s30, s48
	s_lshl_b32 s52, s52, 1
	s_add_co_i32 s52, s52, s50
	s_add_co_u32 s46, s38, s52
	s_add_co_ci_u32 s47, s39, 0
	s_mov_b32 s58, s18
	s_mov_b32 s59, s19
	s_lshl_b32 s60, s30, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v7, 4, v1
	v_mul_u32_u24_e32 v8, 3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v8, 1, v8
	v_and_b32_e32 v9, 1, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 2, v9
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	v_cvt_f32_u32_e32 v34, v32
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v34, v34, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v35, v34, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v35, v32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, -1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, s26, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v35, v35, v36, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, 1, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s66
	v_cmp_le_i32_e64 s64, 0, v32
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v34, 1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v40, 0x7fffff00, v34, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v40, v40, s61
	v_mul_lo_u32 v41, v32, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v41, s60, v41
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	s_cmp_eq_u32 s48, 0
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_reduce
	.Lmqi4moe_mq3i4_r128_down_t8_group:
	s_mul_i32 s55, s53, 48
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s57
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_wdone
	global_load_u8 v13, v5, s[44:45]
	global_load_u8 v14, v5, s[44:45] offset:1
	global_load_u16 v15, v6, s[46:47]
	.Lmqi4moe_mq3i4_r128_down_t8_g_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq3i4_r128_down_t8_g_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq3i4_r128_down_t8_g_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_udone
	v_cvt_f32_f16_e64 v15, v15.l
	v_lshl_or_b32 v13, v14, 8, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v13, v9, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v28, v13, 0, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, -7, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v15, v28
	v_bfe_u32 v29, v13, 3, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, -7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v15, v29
	v_bfe_u32 v30, v13, 6, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, -7, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v15, v30
	v_bfe_u32 v31, v13, 9, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, -7, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v15, v31
	.Lmqi4moe_mq3i4_r128_down_t8_g_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v18
	.Lmqi4moe_mq3i4_r128_down_t8_g_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq3i4_r128_down_t8_g_gend:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc1 .Lmqi4moe_mq3i4_r128_down_t8_group
	.Lmqi4moe_mq3i4_r128_down_t8_reduce:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_exit
	v_cmp_gt_u32_e64 s68, 16, v1
	v_cndmask_b32_e64 v23, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v23, v23, v1, 2
	v_cmp_gt_u32_e64 s69, 24, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v24, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v24, v24, v1, 2
	v_cmp_gt_u32_e64 s70, 28, v1
	v_cndmask_b32_e64 v25, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v25, v25, v1, 2
	v_cmp_gt_u32_e64 s71, 30, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v26, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v26, v26, v1, 2
	v_cmp_gt_u32_e64 s72, 31, v1
	v_cndmask_b32_e64 v27, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v27, v27, v1, 2
	ds_bpermute_b32 v19, v23, v42
	ds_bpermute_b32 v20, v23, v43
	ds_bpermute_b32 v21, v23, v44
	ds_bpermute_b32 v22, v23, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s68
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s68
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s68
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s68
	ds_bpermute_b32 v19, v24, v42
	ds_bpermute_b32 v20, v24, v43
	ds_bpermute_b32 v21, v24, v44
	ds_bpermute_b32 v22, v24, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s69
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s69
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s69
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s69
	ds_bpermute_b32 v19, v25, v42
	ds_bpermute_b32 v20, v25, v43
	ds_bpermute_b32 v21, v25, v44
	ds_bpermute_b32 v22, v25, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s70
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s70
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s70
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s70
	ds_bpermute_b32 v19, v26, v42
	ds_bpermute_b32 v20, v26, v43
	ds_bpermute_b32 v21, v26, v44
	ds_bpermute_b32 v22, v26, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s71
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s71
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s71
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s71
	ds_bpermute_b32 v19, v27, v42
	ds_bpermute_b32 v20, v27, v43
	ds_bpermute_b32 v21, v27, v44
	ds_bpermute_b32 v22, v27, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s72
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s72
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s72
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_red_done
	ds_bpermute_b32 v19, v23, v46
	ds_bpermute_b32 v20, v23, v47
	ds_bpermute_b32 v21, v23, v48
	ds_bpermute_b32 v22, v23, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s68
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s68
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s68
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s68
	ds_bpermute_b32 v19, v24, v46
	ds_bpermute_b32 v20, v24, v47
	ds_bpermute_b32 v21, v24, v48
	ds_bpermute_b32 v22, v24, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s69
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s69
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s69
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s69
	ds_bpermute_b32 v19, v25, v46
	ds_bpermute_b32 v20, v25, v47
	ds_bpermute_b32 v21, v25, v48
	ds_bpermute_b32 v22, v25, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s70
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s70
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s70
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s70
	ds_bpermute_b32 v19, v26, v46
	ds_bpermute_b32 v20, v26, v47
	ds_bpermute_b32 v21, v26, v48
	ds_bpermute_b32 v22, v26, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s71
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s71
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s71
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s71
	ds_bpermute_b32 v19, v27, v46
	ds_bpermute_b32 v20, v27, v47
	ds_bpermute_b32 v21, v27, v48
	ds_bpermute_b32 v22, v27, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s72
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s72
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s72
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s72
	.Lmqi4moe_mq3i4_r128_down_t8_red_done:
	v_cmpx_eq_u32_e32 0, v1
	v_readlane_b32 s73, v41, 0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s73
	global_store_b32 v12, v42, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_st_done
	v_readlane_b32 s74, v41, 1
	s_wait_storecnt 0x0
	v_mov_b32_e32 v12, s74
	global_store_b32 v12, v43, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_st_done
	v_readlane_b32 s75, v41, 2
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s75
	global_store_b32 v12, v44, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_st_done
	v_readlane_b32 s76, v41, 3
	s_wait_storecnt 0x0
	v_mov_b32_e32 v12, s76
	global_store_b32 v12, v45, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_st_done
	v_readlane_b32 s73, v41, 4
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s73
	global_store_b32 v12, v46, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_st_done
	v_readlane_b32 s74, v41, 5
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s74
	global_store_b32 v12, v47, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_st_done
	v_readlane_b32 s75, v41, 6
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s75
	global_store_b32 v12, v48, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq3i4_r128_down_t8_st_done
	v_readlane_b32 s76, v41, 7
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s76
	global_store_b32 v12, v49, s[58:59]
	.Lmqi4moe_mq3i4_r128_down_t8_st_done:
	s_wait_storecnt 0x0
	.Lmqi4moe_mq3i4_r128_down_t8_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq3i4_r128_down_t8_end:
.size mq_i4_moe_gemv_mq3i4_r128_down_t8, .Lmq_i4_moe_gemv_mq3i4_r128_down_t8_end-mq_i4_moe_gemv_mq3i4_r128_down_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq3i4_r128_down_t8
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
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 1
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 56
	.amdhsa_next_free_sgpr 86
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq3i4_r128_down_t8_end-mq_i4_moe_gemv_mq3i4_r128_down_t8)<<4)&4080)>>4
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
.protected mq_i4_moe_gemv_mq4sym_r256_gate_up_t8
.globl mq_i4_moe_gemv_mq4sym_r256_gate_up_t8
.p2align 8
.type mq_i4_moe_gemv_mq4sym_r256_gate_up_t8,@function
mq_i4_moe_gemv_mq4sym_r256_gate_up_t8:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s84, s[0:1], 0x58
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 4, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v3, 10, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v3, v1, 4, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_readfirstlane_b32 s80, v2
	s_mov_b32 s81, ttmp9
	s_and_b32 s31, ttmp7, 0xffff
	s_lshr_b32 s85, ttmp7, 16
	s_wait_kmcnt 0x0
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_cmp_eq_u32 s24, 0x50
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_lshl_b32 s51, s85, 3
	s_cmp_ge_u32 s51, s25
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_cmp_eq_u32 s26, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_and_b32 s51, s23, 0xff
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_mul_i32 s51, s85, 0x788
	s_add_co_u32 s10, s10, s51
	s_add_co_ci_u32 s11, s11, 0
	s_add_co_u32 s12, s12, s51
	s_add_co_ci_u32 s13, s13, 0
	s_add_co_u32 s14, s14, s51
	s_add_co_ci_u32 s15, s15, 0
	s_lshl_b32 s81, s81, 3
	s_cmp_ge_u32 s81, s22
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_lshl_b32 s51, s31, 2
	s_add_co_u32 s36, s12, s51
	s_add_co_ci_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], 0x0
	s_add_co_u32 s36, s10, s51
	s_add_co_ci_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], 0x0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_mov_b32 s51, 0x50
	s_cmp_gt_u32 s32, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s34, s84
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_add_co_i32 s63, s35, -1
	v_min_u32_e32 v43, s63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v43, s32, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v43, 2, v43
	global_load_b32 v36, v43, s[14:15]
	s_wait_loadcnt 0x0
	v_cmp_le_u32_e32 vcc_lo, s51, v36
	s_cbranch_vccnz .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	s_lshl_b32 s51, s34, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s36, s8, s51
	s_add_co_ci_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], 0x0
	s_add_co_i32 s30, s81, s80
	s_lshr_b32 s48, s23, 8
	s_mul_i32 s49, s48, 0x88
	s_mul_i32 s51, s30, s49
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s44, s38, s51
	s_add_co_ci_u32 s45, s39, 0
	s_lshr_b32 s77, s22, 1
	s_sub_co_i32 s78, s30, s77
	s_cmp_lt_u32 s30, s77
	s_cselect_b64 s[58:59], s[18:19], s[20:21]
	s_cselect_b32 s79, s30, s78
	s_lshl_b32 s60, s79, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v7, 5, v1
	v_lshlrev_b32_e32 v8, 2, v1
	v_lshrrev_b32_e32 v10, 4, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v10, 2, v10
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
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_cvt_f32_u32_e32 v41, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v41, v41
	v_cvt_f32_u32_e32 v38, v36
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v38, v38, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v38, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v39, v38, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v39, v36, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v39
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v40, -1, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v38, v38, v40, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v40, s26, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v39, v39, v40, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v39
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v40, 1, v38
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v38, v38, v40, s66
	v_cmp_le_i32_e64 s64, 0, v36
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v38, 1, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, 0x7fffff00, v38, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v44, v44, s61
	v_mul_lo_u32 v45, v36, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v45, s60, v45
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	s_and_b32 s54, s48, -4
	s_cmp_eq_u32 s54, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_tail
	.Lmqi4moe_mq4sym_r256_gate_up_t8_quad:
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_add_nc_u32_e32 v6, s55, v10
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_wdone
	global_load_b32 v13, v5, s[44:45] offset:8
	global_load_b32 v16, v6, s[44:45]
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q0_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_pdone
	v_readlane_b32 s83, v44, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[36:39], v12, s[40:43], null offen
	buffer_load_b128 v[40:43], v12, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_sdone
	ds_store_b128 v3, v[36:39]
	ds_store_b128 v3, v[40:43] offset:512
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_udone
	v_cvt_f32_f16_e64 v15, v16.l
	v_lshrrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v16, v16.l
	v_bfe_u32 v28, v13, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v15, v28, v16
	v_bfe_u32 v29, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v15, v29, v16
	v_bfe_u32 v30, v13, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v15, v30, v16
	v_bfe_u32 v31, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v15, v31, v16
	v_bfe_u32 v32, v13, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v32, v15, v32, v16
	v_bfe_u32 v33, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v15, v33, v16
	v_bfe_u32 v34, v13, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v15, v34, v16
	v_bfe_u32 v35, v13, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v35, v15, v35, v16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[36:39], v4
	ds_load_b128 v[40:43], v4 offset:512
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:1024
	ds_load_b128 v[40:43], v4 offset:1536
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:2048
	ds_load_b128 v[40:43], v4 offset:2560
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:3072
	ds_load_b128 v[40:43], v4 offset:3584
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:4096
	ds_load_b128 v[40:43], v4 offset:4608
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:5120
	ds_load_b128 v[40:43], v4 offset:5632
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:6144
	ds_load_b128 v[40:43], v4 offset:6656
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:7168
	ds_load_b128 v[40:43], v4 offset:7680
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v18
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q0_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q0_gend:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_add_nc_u32_e32 v6, s55, v10
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_wdone
	global_load_b32 v13, v5, s[44:45] offset:8
	global_load_b32 v16, v6, s[44:45]
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q1_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_pdone
	v_readlane_b32 s83, v44, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[36:39], v12, s[40:43], null offen
	buffer_load_b128 v[40:43], v12, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_sdone
	ds_store_b128 v3, v[36:39]
	ds_store_b128 v3, v[40:43] offset:512
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_udone
	v_cvt_f32_f16_e64 v15, v16.l
	v_lshrrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v16, v16.l
	v_bfe_u32 v28, v13, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v15, v28, v16
	v_bfe_u32 v29, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v15, v29, v16
	v_bfe_u32 v30, v13, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v15, v30, v16
	v_bfe_u32 v31, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v15, v31, v16
	v_bfe_u32 v32, v13, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v32, v15, v32, v16
	v_bfe_u32 v33, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v15, v33, v16
	v_bfe_u32 v34, v13, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v15, v34, v16
	v_bfe_u32 v35, v13, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v35, v15, v35, v16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[36:39], v4
	ds_load_b128 v[40:43], v4 offset:512
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:1024
	ds_load_b128 v[40:43], v4 offset:1536
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:2048
	ds_load_b128 v[40:43], v4 offset:2560
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:3072
	ds_load_b128 v[40:43], v4 offset:3584
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:4096
	ds_load_b128 v[40:43], v4 offset:4608
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:5120
	ds_load_b128 v[40:43], v4 offset:5632
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:6144
	ds_load_b128 v[40:43], v4 offset:6656
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:7168
	ds_load_b128 v[40:43], v4 offset:7680
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v18
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q1_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q1_gend:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_add_nc_u32_e32 v6, s55, v10
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_wdone
	global_load_b32 v13, v5, s[44:45] offset:8
	global_load_b32 v16, v6, s[44:45]
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q2_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_pdone
	v_readlane_b32 s83, v44, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[36:39], v12, s[40:43], null offen
	buffer_load_b128 v[40:43], v12, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_sdone
	ds_store_b128 v3, v[36:39]
	ds_store_b128 v3, v[40:43] offset:512
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_udone
	v_cvt_f32_f16_e64 v15, v16.l
	v_lshrrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v16, v16.l
	v_bfe_u32 v28, v13, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v15, v28, v16
	v_bfe_u32 v29, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v15, v29, v16
	v_bfe_u32 v30, v13, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v15, v30, v16
	v_bfe_u32 v31, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v15, v31, v16
	v_bfe_u32 v32, v13, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v32, v15, v32, v16
	v_bfe_u32 v33, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v15, v33, v16
	v_bfe_u32 v34, v13, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v15, v34, v16
	v_bfe_u32 v35, v13, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v35, v15, v35, v16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[36:39], v4
	ds_load_b128 v[40:43], v4 offset:512
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:1024
	ds_load_b128 v[40:43], v4 offset:1536
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:2048
	ds_load_b128 v[40:43], v4 offset:2560
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:3072
	ds_load_b128 v[40:43], v4 offset:3584
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:4096
	ds_load_b128 v[40:43], v4 offset:4608
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:5120
	ds_load_b128 v[40:43], v4 offset:5632
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:6144
	ds_load_b128 v[40:43], v4 offset:6656
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:7168
	ds_load_b128 v[40:43], v4 offset:7680
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v18
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q2_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q2_gend:
	s_add_co_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_add_nc_u32_e32 v6, s55, v10
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_wdone
	global_load_b32 v13, v5, s[44:45] offset:8
	global_load_b32 v16, v6, s[44:45]
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q3_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_pdone
	v_readlane_b32 s83, v44, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[36:39], v12, s[40:43], null offen
	buffer_load_b128 v[40:43], v12, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_sdone
	ds_store_b128 v3, v[36:39]
	ds_store_b128 v3, v[40:43] offset:512
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_udone
	v_cvt_f32_f16_e64 v15, v16.l
	v_lshrrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v16, v16.l
	v_bfe_u32 v28, v13, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v15, v28, v16
	v_bfe_u32 v29, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v15, v29, v16
	v_bfe_u32 v30, v13, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v15, v30, v16
	v_bfe_u32 v31, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v15, v31, v16
	v_bfe_u32 v32, v13, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v32, v15, v32, v16
	v_bfe_u32 v33, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v15, v33, v16
	v_bfe_u32 v34, v13, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v15, v34, v16
	v_bfe_u32 v35, v13, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v35, v15, v35, v16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[36:39], v4
	ds_load_b128 v[40:43], v4 offset:512
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[36:39], v4 offset:1024
	ds_load_b128 v[40:43], v4 offset:1536
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v53, v53, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[36:39], v4 offset:2048
	ds_load_b128 v[40:43], v4 offset:2560
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[36:39], v4 offset:3072
	ds_load_b128 v[40:43], v4 offset:3584
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[36:39], v4 offset:4096
	ds_load_b128 v[40:43], v4 offset:4608
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[36:39], v4 offset:5120
	ds_load_b128 v[40:43], v4 offset:5632
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[36:39], v4 offset:6144
	ds_load_b128 v[40:43], v4 offset:6656
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone
	ds_load_b128 v[36:39], v4 offset:7168
	ds_load_b128 v[40:43], v4 offset:7680
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v18
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q3_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq4sym_r256_gate_up_t8_q3_gend:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s54
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_t8_quad
	.Lmqi4moe_mq4sym_r256_gate_up_t8_tail:
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_add_nc_u32_e32 v6, s55, v10
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_wdone
	global_load_b32 v13, v5, s[44:45] offset:8
	global_load_b32 v16, v6, s[44:45]
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t0_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_pdone
	v_readlane_b32 s83, v44, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[36:39], v12, s[40:43], null offen
	buffer_load_b128 v[40:43], v12, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_sdone
	ds_store_b128 v3, v[36:39]
	ds_store_b128 v3, v[40:43] offset:512
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_udone
	v_cvt_f32_f16_e64 v15, v16.l
	v_lshrrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v16, v16.l
	v_bfe_u32 v28, v13, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v15, v28, v16
	v_bfe_u32 v29, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v15, v29, v16
	v_bfe_u32 v30, v13, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v15, v30, v16
	v_bfe_u32 v31, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v15, v31, v16
	v_bfe_u32 v32, v13, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v32, v15, v32, v16
	v_bfe_u32 v33, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v15, v33, v16
	v_bfe_u32 v34, v13, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v15, v34, v16
	v_bfe_u32 v35, v13, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v35, v15, v35, v16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[36:39], v4
	ds_load_b128 v[40:43], v4 offset:512
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:1024
	ds_load_b128 v[40:43], v4 offset:1536
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:2048
	ds_load_b128 v[40:43], v4 offset:2560
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:3072
	ds_load_b128 v[40:43], v4 offset:3584
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:4096
	ds_load_b128 v[40:43], v4 offset:4608
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:5120
	ds_load_b128 v[40:43], v4 offset:5632
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:6144
	ds_load_b128 v[40:43], v4 offset:6656
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone
	ds_load_b128 v[36:39], v4 offset:7168
	ds_load_b128 v[40:43], v4 offset:7680
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v18
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t0_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t0_gend:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_add_nc_u32_e32 v6, s55, v10
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_wdone
	global_load_b32 v13, v5, s[44:45] offset:8
	global_load_b32 v16, v6, s[44:45]
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t1_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_pdone
	v_readlane_b32 s83, v44, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[36:39], v12, s[40:43], null offen
	buffer_load_b128 v[40:43], v12, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_sdone
	ds_store_b128 v3, v[36:39]
	ds_store_b128 v3, v[40:43] offset:512
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_udone
	v_cvt_f32_f16_e64 v15, v16.l
	v_lshrrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v16, v16.l
	v_bfe_u32 v28, v13, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v15, v28, v16
	v_bfe_u32 v29, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v15, v29, v16
	v_bfe_u32 v30, v13, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v15, v30, v16
	v_bfe_u32 v31, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v15, v31, v16
	v_bfe_u32 v32, v13, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v32, v15, v32, v16
	v_bfe_u32 v33, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v15, v33, v16
	v_bfe_u32 v34, v13, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v15, v34, v16
	v_bfe_u32 v35, v13, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v35, v15, v35, v16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[36:39], v4
	ds_load_b128 v[40:43], v4 offset:512
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:1024
	ds_load_b128 v[40:43], v4 offset:1536
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:2048
	ds_load_b128 v[40:43], v4 offset:2560
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:3072
	ds_load_b128 v[40:43], v4 offset:3584
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:4096
	ds_load_b128 v[40:43], v4 offset:4608
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:5120
	ds_load_b128 v[40:43], v4 offset:5632
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:6144
	ds_load_b128 v[40:43], v4 offset:6656
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone
	ds_load_b128 v[36:39], v4 offset:7168
	ds_load_b128 v[40:43], v4 offset:7680
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v18
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t1_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t1_gend:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_add_nc_u32_e32 v6, s55, v10
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_wdone
	global_load_b32 v13, v5, s[44:45] offset:8
	global_load_b32 v16, v6, s[44:45]
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t2_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_pdone
	v_readlane_b32 s83, v44, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[36:39], v12, s[40:43], null offen
	buffer_load_b128 v[40:43], v12, s[40:43], null offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_sdone
	ds_store_b128 v3, v[36:39]
	ds_store_b128 v3, v[40:43] offset:512
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_udone
	v_cvt_f32_f16_e64 v15, v16.l
	v_lshrrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v16, v16.l
	v_bfe_u32 v28, v13, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v15, v28, v16
	v_bfe_u32 v29, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v15, v29, v16
	v_bfe_u32 v30, v13, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v15, v30, v16
	v_bfe_u32 v31, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v15, v31, v16
	v_bfe_u32 v32, v13, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v32, v15, v32, v16
	v_bfe_u32 v33, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v15, v33, v16
	v_bfe_u32 v34, v13, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v15, v34, v16
	v_bfe_u32 v35, v13, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v35, v15, v35, v16
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[36:39], v4
	ds_load_b128 v[40:43], v4 offset:512
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:1024
	ds_load_b128 v[40:43], v4 offset:1536
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v52, v52, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:2048
	ds_load_b128 v[40:43], v4 offset:2560
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:3072
	ds_load_b128 v[40:43], v4 offset:3584
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:4096
	ds_load_b128 v[40:43], v4 offset:4608
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:5120
	ds_load_b128 v[40:43], v4 offset:5632
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:6144
	ds_load_b128 v[40:43], v4 offset:6656
	s_wait_dscnt 0x1
	v_mul_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone
	ds_load_b128 v[36:39], v4 offset:7168
	ds_load_b128 v[40:43], v4 offset:7680
	s_wait_dscnt 0x1
	v_mul_f32_e32 v18, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v39
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v32, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v33, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v35, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v18
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t2_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq4sym_r256_gate_up_t8_t2_gend:
	s_add_co_i32 s53, s53, 1
	.Lmqi4moe_mq4sym_r256_gate_up_t8_reduce:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_exit
	v_cmp_gt_u32_e64 s68, 16, v1
	v_cndmask_b32_e64 v23, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v23, v23, v1, 2
	v_cmp_gt_u32_e64 s69, 24, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v24, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v24, v24, v1, 2
	v_cmp_gt_u32_e64 s70, 28, v1
	v_cndmask_b32_e64 v25, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v25, v25, v1, 2
	v_cmp_gt_u32_e64 s71, 30, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v26, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v26, v26, v1, 2
	v_cmp_gt_u32_e64 s72, 31, v1
	v_cndmask_b32_e64 v27, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v27, v27, v1, 2
	v_add_f32_e32 v46, v46, v47
	v_add_f32_e32 v48, v48, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v48
	v_add_f32_e32 v50, v50, v51
	v_add_f32_e32 v52, v52, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v50, v50, v52
	v_add_f32_e32 v54, v54, v55
	v_add_f32_e32 v56, v56, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v54, v54, v56
	v_add_f32_e32 v58, v58, v59
	v_add_f32_e32 v60, v60, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v60
	ds_bpermute_b32 v19, v23, v46
	ds_bpermute_b32 v20, v23, v50
	ds_bpermute_b32 v21, v23, v54
	ds_bpermute_b32 v22, v23, v58
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s68
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v50, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v20, s68
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v54, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v21, s68
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v58, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v22, s68
	ds_bpermute_b32 v19, v24, v46
	ds_bpermute_b32 v20, v24, v50
	ds_bpermute_b32 v21, v24, v54
	ds_bpermute_b32 v22, v24, v58
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s69
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v50, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v20, s69
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v54, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v21, s69
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v58, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v22, s69
	ds_bpermute_b32 v19, v25, v46
	ds_bpermute_b32 v20, v25, v50
	ds_bpermute_b32 v21, v25, v54
	ds_bpermute_b32 v22, v25, v58
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s70
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v50, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v20, s70
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v54, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v21, s70
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v58, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v22, s70
	ds_bpermute_b32 v19, v26, v46
	ds_bpermute_b32 v20, v26, v50
	ds_bpermute_b32 v21, v26, v54
	ds_bpermute_b32 v22, v26, v58
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s71
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v50, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v20, s71
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v54, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v21, s71
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v58, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v22, s71
	ds_bpermute_b32 v19, v27, v46
	ds_bpermute_b32 v20, v27, v50
	ds_bpermute_b32 v21, v27, v54
	ds_bpermute_b32 v22, v27, v58
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s72
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v50, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v50, v50, v20, s72
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v54, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, v21, s72
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v58, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v22, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_red_done
	v_add_f32_e32 v62, v62, v63
	v_add_f32_e32 v64, v64, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v64
	v_add_f32_e32 v66, v66, v67
	v_add_f32_e32 v68, v68, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v68
	v_add_f32_e32 v70, v70, v71
	v_add_f32_e32 v72, v72, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v72
	v_add_f32_e32 v74, v74, v75
	v_add_f32_e32 v76, v76, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v76
	ds_bpermute_b32 v19, v23, v62
	ds_bpermute_b32 v20, v23, v66
	ds_bpermute_b32 v21, v23, v70
	ds_bpermute_b32 v22, v23, v74
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v62, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v19, s68
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v66, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v20, s68
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v70, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v21, s68
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v74, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v74, v74, v22, s68
	ds_bpermute_b32 v19, v24, v62
	ds_bpermute_b32 v20, v24, v66
	ds_bpermute_b32 v21, v24, v70
	ds_bpermute_b32 v22, v24, v74
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v62, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v19, s69
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v66, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v20, s69
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v70, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v21, s69
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v74, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v74, v74, v22, s69
	ds_bpermute_b32 v19, v25, v62
	ds_bpermute_b32 v20, v25, v66
	ds_bpermute_b32 v21, v25, v70
	ds_bpermute_b32 v22, v25, v74
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v62, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v19, s70
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v66, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v20, s70
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v70, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v21, s70
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v74, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v74, v74, v22, s70
	ds_bpermute_b32 v19, v26, v62
	ds_bpermute_b32 v20, v26, v66
	ds_bpermute_b32 v21, v26, v70
	ds_bpermute_b32 v22, v26, v74
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v62, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v19, s71
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v66, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v20, s71
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v70, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v21, s71
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v74, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v74, v74, v22, s71
	ds_bpermute_b32 v19, v27, v62
	ds_bpermute_b32 v20, v27, v66
	ds_bpermute_b32 v21, v27, v70
	ds_bpermute_b32 v22, v27, v74
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v62, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v19, s72
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v66, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v20, s72
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v70, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v21, s72
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v74, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v74, v74, v22, s72
	.Lmqi4moe_mq4sym_r256_gate_up_t8_red_done:
	v_cmpx_eq_u32_e32 0, v1
	v_readlane_b32 s73, v45, 0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s73
	global_store_b32 v12, v46, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_st_done
	v_readlane_b32 s74, v45, 1
	s_wait_storecnt 0x0
	v_mov_b32_e32 v12, s74
	global_store_b32 v12, v50, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_st_done
	v_readlane_b32 s75, v45, 2
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s75
	global_store_b32 v12, v54, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_st_done
	v_readlane_b32 s76, v45, 3
	s_wait_storecnt 0x0
	v_mov_b32_e32 v12, s76
	global_store_b32 v12, v58, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_st_done
	v_readlane_b32 s73, v45, 4
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s73
	global_store_b32 v12, v62, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_st_done
	v_readlane_b32 s74, v45, 5
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s74
	global_store_b32 v12, v66, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_st_done
	v_readlane_b32 s75, v45, 6
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s75
	global_store_b32 v12, v70, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t8_st_done
	v_readlane_b32 s76, v45, 7
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s76
	global_store_b32 v12, v74, s[58:59]
	.Lmqi4moe_mq4sym_r256_gate_up_t8_st_done:
	s_wait_storecnt 0x0
	.Lmqi4moe_mq4sym_r256_gate_up_t8_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq4sym_r256_gate_up_t8_end:
.size mq_i4_moe_gemv_mq4sym_r256_gate_up_t8, .Lmq_i4_moe_gemv_mq4sym_r256_gate_up_t8_end-mq_i4_moe_gemv_mq4sym_r256_gate_up_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq4sym_r256_gate_up_t8
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
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 1
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 80
	.amdhsa_next_free_sgpr 86
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq4sym_r256_gate_up_t8_end-mq_i4_moe_gemv_mq4sym_r256_gate_up_t8)<<4)&4080)>>4
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
.protected mq_i4_moe_gemv_mq4sym_r128_down_t8
.globl mq_i4_moe_gemv_mq4sym_r128_down_t8
.p2align 8
.type mq_i4_moe_gemv_mq4sym_r128_down_t8,@function
mq_i4_moe_gemv_mq4sym_r128_down_t8:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s84, s[0:1], 0x58
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 4, v1
	v_lshlrev_b32_e32 v3, 4, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s80, v2
	s_mov_b32 s81, ttmp9
	s_and_b32 s31, ttmp7, 0xffff
	s_lshr_b32 s85, ttmp7, 16
	s_wait_kmcnt 0x0
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_cmp_eq_u32 s24, 0x50
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_lshl_b32 s51, s85, 3
	s_cmp_ge_u32 s51, s25
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_cmp_eq_u32 s26, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_and_b32 s51, s23, 0x7f
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_mul_i32 s51, s85, 0x788
	s_add_co_u32 s10, s10, s51
	s_add_co_ci_u32 s11, s11, 0
	s_add_co_u32 s12, s12, s51
	s_add_co_ci_u32 s13, s13, 0
	s_add_co_u32 s14, s14, s51
	s_add_co_ci_u32 s15, s15, 0
	s_lshl_b32 s81, s81, 3
	s_cmp_ge_u32 s81, s22
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_lshl_b32 s51, s31, 2
	s_add_co_u32 s36, s12, s51
	s_add_co_ci_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], 0x0
	s_add_co_u32 s36, s10, s51
	s_add_co_ci_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], 0x0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_mov_b32 s51, 0x50
	s_cmp_gt_u32 s32, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s34, s84
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_add_co_i32 s63, s35, -1
	v_min_u32_e32 v39, s63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, s32, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 2, v39
	global_load_b32 v32, v39, s[14:15]
	s_wait_loadcnt 0x0
	v_cmp_le_u32_e32 vcc_lo, s51, v32
	s_cbranch_vccnz .Lmqi4moe_mq4sym_r128_down_t8_exit
	s_lshl_b32 s51, s34, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s36, s8, s51
	s_add_co_ci_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], 0x0
	s_add_co_i32 s30, s81, s80
	s_lshr_b32 s48, s23, 7
	s_mul_i32 s49, s48, 0x44
	s_mul_i32 s51, s30, s49
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_u32 s44, s38, s51
	s_add_co_ci_u32 s45, s39, 0
	s_mov_b32 s58, s18
	s_mov_b32 s59, s19
	s_lshl_b32 s60, s30, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v7, 4, v1
	v_lshlrev_b32_e32 v8, 1, v1
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	v_cvt_f32_u32_e32 v34, v32
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v34, v34, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v35, v34, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v35, v32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, -1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, s26, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v35, v35, v36, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, 1, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s66
	v_cmp_le_i32_e64 s64, 0, v32
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v34, 1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v40, 0x7fffff00, v34, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v40, v40, s61
	v_mul_lo_u32 v41, v32, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v41, s60, v41
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	s_cmp_eq_u32 s48, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_reduce
	.Lmqi4moe_mq4sym_r128_down_t8_group:
	s_mul_i32 s55, s53, 0x44
	s_lshl_b32 s56, s53, 9
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v5, s55, v8
	v_mov_b32_e32 v6, s55
	v_add_nc_u32_e32 v11, s56, v7
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_wdone
	global_load_u16 v13, v5, s[44:45] offset:4
	global_load_b32 v16, v6, s[44:45]
	.Lmqi4moe_mq4sym_r128_down_t8_g_wdone:
	s_add_co_i32 s82, s80, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_pdone
	v_readlane_b32 s83, v40, s82
	s_wait_alu depctr_va_sdst(0)
	v_add_nc_u32_e32 v12, s83, v11
	buffer_load_b128 v[32:35], v12, s[40:43], null offen
	.Lmqi4moe_mq4sym_r128_down_t8_g_t0_pdone:
	s_wait_loadcnt 0x0
	s_cmp_gt_u32 s35, s82
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_sdone
	ds_store_b128 v3, v[32:35]
	.Lmqi4moe_mq4sym_r128_down_t8_g_t0_sdone:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_udone
	v_cvt_f32_f16_e64 v15, v16.l
	v_lshrrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v16, v16.l
	v_bfe_u32 v28, v13, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v28, v15, v28, v16
	v_bfe_u32 v29, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, v15, v29, v16
	v_bfe_u32 v30, v13, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v30, v15, v30, v16
	v_bfe_u32 v31, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v31, v15, v31, v16
	.Lmqi4moe_mq4sym_r128_down_t8_g_t0_udone:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v42, v42, v17
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:512
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v18
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1024
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v44, v44, v17
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:1536
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v45, v45, v18
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2048
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v46, v46, v17
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:2560
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v18
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3072
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v48, v48, v17
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone
	ds_load_b128 v[32:35], v4 offset:3584
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v18
	.Lmqi4moe_mq4sym_r128_down_t8_g_t0_cdone:
	s_barrier_signal -1
	s_barrier_wait 0xffff
	.Lmqi4moe_mq4sym_r128_down_t8_g_gend:
	s_add_co_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_t8_group
	.Lmqi4moe_mq4sym_r128_down_t8_reduce:
	s_cmp_lt_u32 s30, s22
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_exit
	v_cmp_gt_u32_e64 s68, 16, v1
	v_cndmask_b32_e64 v23, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v23, v23, v1, 2
	v_cmp_gt_u32_e64 s69, 24, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v24, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v24, v24, v1, 2
	v_cmp_gt_u32_e64 s70, 28, v1
	v_cndmask_b32_e64 v25, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v25, v25, v1, 2
	v_cmp_gt_u32_e64 s71, 30, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v26, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v26, v26, v1, 2
	v_cmp_gt_u32_e64 s72, 31, v1
	v_cndmask_b32_e64 v27, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v27, v27, v1, 2
	ds_bpermute_b32 v19, v23, v42
	ds_bpermute_b32 v20, v23, v43
	ds_bpermute_b32 v21, v23, v44
	ds_bpermute_b32 v22, v23, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s68
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s68
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s68
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s68
	ds_bpermute_b32 v19, v24, v42
	ds_bpermute_b32 v20, v24, v43
	ds_bpermute_b32 v21, v24, v44
	ds_bpermute_b32 v22, v24, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s69
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s69
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s69
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s69
	ds_bpermute_b32 v19, v25, v42
	ds_bpermute_b32 v20, v25, v43
	ds_bpermute_b32 v21, v25, v44
	ds_bpermute_b32 v22, v25, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s70
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s70
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s70
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s70
	ds_bpermute_b32 v19, v26, v42
	ds_bpermute_b32 v20, v26, v43
	ds_bpermute_b32 v21, v26, v44
	ds_bpermute_b32 v22, v26, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s71
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s71
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s71
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s71
	ds_bpermute_b32 v19, v27, v42
	ds_bpermute_b32 v20, v27, v43
	ds_bpermute_b32 v21, v27, v44
	ds_bpermute_b32 v22, v27, v45
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v42, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v42, v42, v19, s72
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v43, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v20, s72
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v44, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v44, v44, v21, s72
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v45, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v45, v45, v22, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_red_done
	ds_bpermute_b32 v19, v23, v46
	ds_bpermute_b32 v20, v23, v47
	ds_bpermute_b32 v21, v23, v48
	ds_bpermute_b32 v22, v23, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s68
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s68
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s68
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s68
	ds_bpermute_b32 v19, v24, v46
	ds_bpermute_b32 v20, v24, v47
	ds_bpermute_b32 v21, v24, v48
	ds_bpermute_b32 v22, v24, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s69
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s69
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s69
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s69
	ds_bpermute_b32 v19, v25, v46
	ds_bpermute_b32 v20, v25, v47
	ds_bpermute_b32 v21, v25, v48
	ds_bpermute_b32 v22, v25, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s70
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s70
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s70
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s70
	ds_bpermute_b32 v19, v26, v46
	ds_bpermute_b32 v20, v26, v47
	ds_bpermute_b32 v21, v26, v48
	ds_bpermute_b32 v22, v26, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s71
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s71
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s71
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s71
	ds_bpermute_b32 v19, v27, v46
	ds_bpermute_b32 v20, v27, v47
	ds_bpermute_b32 v21, v27, v48
	ds_bpermute_b32 v22, v27, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v19, v46, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, v19, s72
	s_wait_dscnt 0x2
	v_add_f32_e32 v20, v47, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v47, v47, v20, s72
	s_wait_dscnt 0x1
	v_add_f32_e32 v21, v48, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v48, v48, v21, s72
	s_wait_dscnt 0x0
	v_add_f32_e32 v22, v49, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v49, v49, v22, s72
	.Lmqi4moe_mq4sym_r128_down_t8_red_done:
	v_cmpx_eq_u32_e32 0, v1
	v_readlane_b32 s73, v41, 0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s73
	global_store_b32 v12, v42, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_st_done
	v_readlane_b32 s74, v41, 1
	s_wait_storecnt 0x0
	v_mov_b32_e32 v12, s74
	global_store_b32 v12, v43, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_st_done
	v_readlane_b32 s75, v41, 2
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s75
	global_store_b32 v12, v44, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_st_done
	v_readlane_b32 s76, v41, 3
	s_wait_storecnt 0x0
	v_mov_b32_e32 v12, s76
	global_store_b32 v12, v45, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_st_done
	v_readlane_b32 s73, v41, 4
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s73
	global_store_b32 v12, v46, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_st_done
	v_readlane_b32 s74, v41, 5
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s74
	global_store_b32 v12, v47, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_st_done
	v_readlane_b32 s75, v41, 6
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s75
	global_store_b32 v12, v48, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_t8_st_done
	v_readlane_b32 s76, v41, 7
	s_wait_storecnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_mov_b32_e32 v12, s76
	global_store_b32 v12, v49, s[58:59]
	.Lmqi4moe_mq4sym_r128_down_t8_st_done:
	s_wait_storecnt 0x0
	.Lmqi4moe_mq4sym_r128_down_t8_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq4sym_r128_down_t8_end:
.size mq_i4_moe_gemv_mq4sym_r128_down_t8, .Lmq_i4_moe_gemv_mq4sym_r128_down_t8_end-mq_i4_moe_gemv_mq4sym_r128_down_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq4sym_r128_down_t8
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
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 1
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 56
	.amdhsa_next_free_sgpr 86
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq4sym_r128_down_t8_end-mq_i4_moe_gemv_mq4sym_r128_down_t8)<<4)&4080)>>4
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
        .name: expert_weight_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_offsets
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: slot_ids
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: Xrot
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: U
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 68
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_stride
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: y_slot_stride
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: reserved
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: mq_i4_moe_gemv_mq3i4_r256_gate_up
    .private_segment_fixed_size: 0
    .sgpr_count: 81
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq3i4_r256_gate_up.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 112
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: expert_weight_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_offsets
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: slot_ids
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: Xrot
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: U
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 68
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_stride
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: y_slot_stride
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: reserved
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: mq_i4_moe_gemv_mq3i4_r128_down
    .private_segment_fixed_size: 0
    .sgpr_count: 81
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq3i4_r128_down.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 88
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: expert_weight_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_offsets
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: slot_ids
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: Xrot
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: U
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 68
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_stride
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: y_slot_stride
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: reserved
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: mq_i4_moe_gemv_mq4sym_r256_gate_up
    .private_segment_fixed_size: 0
    .sgpr_count: 81
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq4sym_r256_gate_up.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 144
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: expert_weight_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_offsets
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: slot_ids
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: Xrot
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: U
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 68
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_stride
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: y_slot_stride
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: reserved
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: mq_i4_moe_gemv_mq4sym_r128_down
    .private_segment_fixed_size: 0
    .sgpr_count: 81
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq4sym_r128_down.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 88
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: expert_weight_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_offsets
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: slot_ids
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: Xrot
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: U
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 68
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_stride
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: y_slot_stride
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: reserved
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_mq3i4_r256_gate_up_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 88
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq3i4_r256_gate_up_t8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 80
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: expert_weight_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_offsets
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: slot_ids
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: Xrot
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: U
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 68
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_stride
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: y_slot_stride
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: reserved
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_mq3i4_r128_down_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 88
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq3i4_r128_down_t8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 56
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: expert_weight_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_offsets
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: slot_ids
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: Xrot
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: U
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 68
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_stride
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: y_slot_stride
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: reserved
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_mq4sym_r256_gate_up_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 88
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq4sym_r256_gate_up_t8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 80
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: expert_weight_ptrs
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_ids
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: expert_offsets
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: slot_ids
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: Xrot
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: U
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: C
        .offset: 68
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_div
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: x_src_rows
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: x_row_stride
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: y_slot_stride
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: reserved
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_mq4sym_r128_down_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 88
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq4sym_r128_down_t8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 56
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
