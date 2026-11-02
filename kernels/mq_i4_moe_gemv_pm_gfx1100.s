.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.amdhsa_code_object_version 6
.text
.protected mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100
.globl mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100
.p2align 8
.type mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100,@function
mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r256_gate_up_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r256_gate_up_exit
	s_lshl_b32 s51, s31, 2
	s_add_u32 s36, s12, s51
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s51
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r256_gate_up_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r256_gate_up_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r256_gate_up_exit
	s_lshl_b32 s51, s34, 3
	s_add_u32 s36, s8, s51
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	s_lshr_b32 s48, s23, 7
	s_lshr_b32 s49, s23, 2
	s_mul_i32 s50, s22, s49
	s_mul_i32 s51, s30, s49
	s_waitcnt lgkmcnt(0)
	s_add_u32 s44, s38, s51
	s_addc_u32 s45, s39, 0
	s_mul_i32 s52, s30, s48
	s_lshl_b32 s52, s52, 1
	s_add_i32 s52, s52, s50
	s_add_u32 s46, s38, s52
	s_addc_u32 s47, s39, 0
	s_lshr_b32 s77, s22, 1
	s_sub_i32 s78, s30, s77
	s_cmp_lt_u32 s30, s77
	s_cselect_b64 s[58:59], s[18:19], s[20:21]
	s_cselect_b32 s79, s30, s78
	s_lshl_b32 s60, s79, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_add_i32 s63, s35, -1
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v3, 4, v0
	v_mov_b32_e32 v4, v0
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
	v_min_u32_e32 v31, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, s32, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 2, v31
	global_load_b32 v24, v31, s[14:15]
	v_cvt_f32_u32_e32 v29, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v29, v29
	s_waitcnt vmcnt(0)
	v_cvt_f32_u32_e32 v26, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v26, v26, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v27, v26, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v27, v24, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, -1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, s26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v27, v27, v28, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s66
	v_cmp_le_i32_e64 s64, 0, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v25, 0x7fffff00, v26, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v25, v25, s61
	v_mul_lo_u32 v30, v24, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, s60, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v25, 0
	v_mov_b32_e32 v88, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v30, 0
	v_mov_b32_e32 v96, s73
	v_readlane_b32 s74, v25, 1
	v_mov_b32_e32 v89, s74
	v_readlane_b32 s74, v30, 1
	v_mov_b32_e32 v97, s74
	v_readlane_b32 s75, v25, 2
	v_mov_b32_e32 v90, s75
	v_readlane_b32 s75, v30, 2
	v_mov_b32_e32 v98, s75
	v_readlane_b32 s76, v25, 3
	v_mov_b32_e32 v91, s76
	v_readlane_b32 s76, v30, 3
	v_mov_b32_e32 v99, s76
	v_readlane_b32 s73, v25, 4
	v_mov_b32_e32 v92, s73
	v_readlane_b32 s73, v30, 4
	v_mov_b32_e32 v100, s73
	v_readlane_b32 s74, v25, 5
	v_mov_b32_e32 v93, s74
	v_readlane_b32 s74, v30, 5
	v_mov_b32_e32 v101, s74
	v_readlane_b32 s75, v25, 6
	v_mov_b32_e32 v94, s75
	v_readlane_b32 s75, v30, 6
	v_mov_b32_e32 v102, s75
	v_readlane_b32 s76, v25, 7
	v_mov_b32_e32 v95, s76
	v_readlane_b32 s76, v30, 7
	v_mov_b32_e32 v103, s76
	s_and_b32 s54, s48, -4
	s_cmp_eq_u32 s54, 0
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r256_gate_up_tail
	.Lmqi4moe_mq2i4_r256_gate_up_quad:
	s_mul_i32 s55, s53, 32
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u8 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v88, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v89, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v90, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v91, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v92, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v93, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v94, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v95, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq2i4_r256_gate_up_q0_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v9.l
	v_bfe_u32 v20, v8, 0, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 1, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, -3, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v9, v20
	v_bfe_u32 v21, v8, 2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 1, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, -3, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v9, v21
	v_bfe_u32 v22, v8, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 1, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, -3, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v9, v22
	v_bfe_u32 v23, v8, 6, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 1, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, -3, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v9, v23
	v_mul_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_done
	v_mul_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_done
	v_mul_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_done
	v_mul_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_done
	v_mul_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_done
	v_mul_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_done
	v_mul_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q0_done
	v_mul_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v12
	.Lmqi4moe_mq2i4_r256_gate_up_q0_done:
	s_add_i32 s53, s53, 1
	s_mul_i32 s55, s53, 32
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u8 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v88, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v89, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v90, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v91, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v92, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v93, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v94, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v95, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq2i4_r256_gate_up_q1_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v9.l
	v_bfe_u32 v20, v8, 0, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 1, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, -3, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v9, v20
	v_bfe_u32 v21, v8, 2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 1, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, -3, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v9, v21
	v_bfe_u32 v22, v8, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 1, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, -3, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v9, v22
	v_bfe_u32 v23, v8, 6, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 1, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, -3, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v9, v23
	v_mul_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_done
	v_mul_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_done
	v_mul_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_done
	v_mul_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_done
	v_mul_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_done
	v_mul_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_done
	v_mul_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q1_done
	v_mul_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v85, v12
	.Lmqi4moe_mq2i4_r256_gate_up_q1_done:
	s_add_i32 s53, s53, 1
	s_mul_i32 s55, s53, 32
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u8 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v88, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v89, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v90, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v91, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v92, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v93, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v94, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v95, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq2i4_r256_gate_up_q2_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v9.l
	v_bfe_u32 v20, v8, 0, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 1, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, -3, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v9, v20
	v_bfe_u32 v21, v8, 2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 1, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, -3, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v9, v21
	v_bfe_u32 v22, v8, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 1, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, -3, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v9, v22
	v_bfe_u32 v23, v8, 6, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 1, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, -3, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v9, v23
	v_mul_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_done
	v_mul_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_done
	v_mul_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_done
	v_mul_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_done
	v_mul_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_done
	v_mul_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78, v78, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_done
	v_mul_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v82, v82, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q2_done
	v_mul_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v86, v86, v12
	.Lmqi4moe_mq2i4_r256_gate_up_q2_done:
	s_add_i32 s53, s53, 1
	s_mul_i32 s55, s53, 32
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u8 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v88, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v89, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v90, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v91, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v92, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v93, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v94, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v95, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq2i4_r256_gate_up_q3_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v9.l
	v_bfe_u32 v20, v8, 0, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 1, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, -3, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v9, v20
	v_bfe_u32 v21, v8, 2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 1, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, -3, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v9, v21
	v_bfe_u32 v22, v8, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 1, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, -3, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v9, v22
	v_bfe_u32 v23, v8, 6, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 1, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, -3, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v9, v23
	v_mul_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_done
	v_mul_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_done
	v_mul_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_done
	v_mul_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_done
	v_mul_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_done
	v_mul_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v79, v79, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_done
	v_mul_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_q3_done
	v_mul_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v87, v87, v12
	.Lmqi4moe_mq2i4_r256_gate_up_q3_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s54
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r256_gate_up_quad
	.Lmqi4moe_mq2i4_r256_gate_up_tail:
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_reduce
	s_mul_i32 s55, s53, 32
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u8 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v88, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v89, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v90, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v91, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v92, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v93, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v94, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v95, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq2i4_r256_gate_up_t0_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v9.l
	v_bfe_u32 v20, v8, 0, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 1, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, -3, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v9, v20
	v_bfe_u32 v21, v8, 2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 1, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, -3, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v9, v21
	v_bfe_u32 v22, v8, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 1, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, -3, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v9, v22
	v_bfe_u32 v23, v8, 6, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 1, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, -3, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v9, v23
	v_mul_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_done
	v_mul_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_done
	v_mul_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_done
	v_mul_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_done
	v_mul_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_done
	v_mul_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_done
	v_mul_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t0_done
	v_mul_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v12
	.Lmqi4moe_mq2i4_r256_gate_up_t0_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_reduce
	s_mul_i32 s55, s53, 32
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u8 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v88, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v89, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v90, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v91, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v92, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v93, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v94, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v95, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq2i4_r256_gate_up_t1_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v9.l
	v_bfe_u32 v20, v8, 0, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 1, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, -3, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v9, v20
	v_bfe_u32 v21, v8, 2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 1, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, -3, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v9, v21
	v_bfe_u32 v22, v8, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 1, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, -3, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v9, v22
	v_bfe_u32 v23, v8, 6, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 1, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, -3, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v9, v23
	v_mul_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_done
	v_mul_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_done
	v_mul_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_done
	v_mul_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_done
	v_mul_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_done
	v_mul_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_done
	v_mul_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t1_done
	v_mul_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v85, v12
	.Lmqi4moe_mq2i4_r256_gate_up_t1_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_reduce
	s_mul_i32 s55, s53, 32
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u8 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v88, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v89, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v90, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v91, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v92, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v93, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v94, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v95, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq2i4_r256_gate_up_t2_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v9.l
	v_bfe_u32 v20, v8, 0, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 1, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, -3, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v9, v20
	v_bfe_u32 v21, v8, 2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 1, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, -3, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v9, v21
	v_bfe_u32 v22, v8, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 1, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, -3, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v9, v22
	v_bfe_u32 v23, v8, 6, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 1, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, -3, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v9, v23
	v_mul_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_done
	v_mul_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_done
	v_mul_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_done
	v_mul_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_done
	v_mul_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_done
	v_mul_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78, v78, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_done
	v_mul_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v82, v82, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_t2_done
	v_mul_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v86, v86, v12
	.Lmqi4moe_mq2i4_r256_gate_up_t2_done:
	s_add_i32 s53, s53, 1
	.Lmqi4moe_mq2i4_r256_gate_up_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v14, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v14, v14, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	v_cndmask_b32_e64 v15, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v15, v15, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v16, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v16, v16, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	v_cndmask_b32_e64 v17, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v17, v17, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	v_cndmask_b32_e64 v18, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v18, v18, v0, 2
	v_add_f32_e32 v56, v56, v57
	v_add_f32_e32 v58, v58, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v58
	ds_bpermute_b32 v13, v14, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s68
	ds_bpermute_b32 v13, v15, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s69
	ds_bpermute_b32 v13, v16, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s70
	ds_bpermute_b32 v13, v17, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s71
	ds_bpermute_b32 v13, v18, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_red_done
	v_add_f32_e32 v60, v60, v61
	v_add_f32_e32 v62, v62, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v62
	ds_bpermute_b32 v13, v14, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s68
	ds_bpermute_b32 v13, v15, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s69
	ds_bpermute_b32 v13, v16, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s70
	ds_bpermute_b32 v13, v17, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s71
	ds_bpermute_b32 v13, v18, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_red_done
	v_add_f32_e32 v64, v64, v65
	v_add_f32_e32 v66, v66, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v66
	ds_bpermute_b32 v13, v14, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v64, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v13, s68
	ds_bpermute_b32 v13, v15, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v64, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v13, s69
	ds_bpermute_b32 v13, v16, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v64, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v13, s70
	ds_bpermute_b32 v13, v17, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v64, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v13, s71
	ds_bpermute_b32 v13, v18, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v64, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v13, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_red_done
	v_add_f32_e32 v68, v68, v69
	v_add_f32_e32 v70, v70, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v70
	ds_bpermute_b32 v13, v14, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v68, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v13, s68
	ds_bpermute_b32 v13, v15, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v68, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v13, s69
	ds_bpermute_b32 v13, v16, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v68, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v13, s70
	ds_bpermute_b32 v13, v17, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v68, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v13, s71
	ds_bpermute_b32 v13, v18, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v68, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v13, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_red_done
	v_add_f32_e32 v72, v72, v73
	v_add_f32_e32 v74, v74, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v74
	ds_bpermute_b32 v13, v14, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v72, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v13, s68
	ds_bpermute_b32 v13, v15, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v72, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v13, s69
	ds_bpermute_b32 v13, v16, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v72, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v13, s70
	ds_bpermute_b32 v13, v17, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v72, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v13, s71
	ds_bpermute_b32 v13, v18, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v72, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v13, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_red_done
	v_add_f32_e32 v76, v76, v77
	v_add_f32_e32 v78, v78, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v78
	ds_bpermute_b32 v13, v14, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v76, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v13, s68
	ds_bpermute_b32 v13, v15, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v76, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v13, s69
	ds_bpermute_b32 v13, v16, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v76, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v13, s70
	ds_bpermute_b32 v13, v17, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v76, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v13, s71
	ds_bpermute_b32 v13, v18, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v76, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v13, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_red_done
	v_add_f32_e32 v80, v80, v81
	v_add_f32_e32 v82, v82, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v82
	ds_bpermute_b32 v13, v14, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v80, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v13, s68
	ds_bpermute_b32 v13, v15, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v80, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v13, s69
	ds_bpermute_b32 v13, v16, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v80, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v13, s70
	ds_bpermute_b32 v13, v17, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v80, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v13, s71
	ds_bpermute_b32 v13, v18, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v80, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v13, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_red_done
	v_add_f32_e32 v84, v84, v85
	v_add_f32_e32 v86, v86, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v86
	ds_bpermute_b32 v13, v14, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v84, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v13, s68
	ds_bpermute_b32 v13, v15, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v84, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v13, s69
	ds_bpermute_b32 v13, v16, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v84, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v13, s70
	ds_bpermute_b32 v13, v17, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v84, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v13, s71
	ds_bpermute_b32 v13, v18, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v84, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v13, s72
	.Lmqi4moe_mq2i4_r256_gate_up_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v96, v56, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_st_done
	global_store_b32 v97, v60, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_st_done
	global_store_b32 v98, v64, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_st_done
	global_store_b32 v99, v68, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_st_done
	global_store_b32 v100, v72, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_st_done
	global_store_b32 v101, v76, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_st_done
	global_store_b32 v102, v80, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r256_gate_up_st_done
	global_store_b32 v103, v84, s[58:59]
	.Lmqi4moe_mq2i4_r256_gate_up_st_done:
	s_waitcnt_vscnt null, 0x0
	.Lmqi4moe_mq2i4_r256_gate_up_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100_end:
.size mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100, .Lmq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100_end-mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 88
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
	.amdhsa_next_free_vgpr 104
	.amdhsa_next_free_sgpr 80
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100_end-mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100
.globl mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100
.p2align 8
.type mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100,@function
mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r128_down_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r128_down_exit
	s_lshl_b32 s51, s31, 2
	s_add_u32 s36, s12, s51
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s51
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r128_down_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r128_down_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r128_down_exit
	s_lshl_b32 s51, s34, 3
	s_add_u32 s36, s8, s51
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	s_lshr_b32 s48, s23, 7
	s_lshr_b32 s49, s23, 2
	s_mul_i32 s50, s22, s49
	s_mul_i32 s51, s30, s49
	s_waitcnt lgkmcnt(0)
	s_add_u32 s44, s38, s51
	s_addc_u32 s45, s39, 0
	s_mul_i32 s52, s30, s48
	s_lshl_b32 s52, s52, 1
	s_add_i32 s52, s52, s50
	s_add_u32 s46, s38, s52
	s_addc_u32 s47, s39, 0
	s_mov_b32 s58, s18
	s_mov_b32 s59, s19
	s_lshl_b32 s60, s30, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_add_i32 s63, s35, -1
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v3, 4, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_min_u32_e32 v31, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, s32, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 2, v31
	global_load_b32 v24, v31, s[14:15]
	v_cvt_f32_u32_e32 v29, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v29, v29
	s_waitcnt vmcnt(0)
	v_cvt_f32_u32_e32 v26, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v26, v26, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v27, v26, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v27, v24, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, -1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, s26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v27, v27, v28, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s66
	v_cmp_le_i32_e64 s64, 0, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v25, 0x7fffff00, v26, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v25, v25, s61
	v_mul_lo_u32 v30, v24, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, s60, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v25, 0
	v_mov_b32_e32 v64, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v30, 0
	v_mov_b32_e32 v72, s73
	v_readlane_b32 s74, v25, 1
	v_mov_b32_e32 v65, s74
	v_readlane_b32 s74, v30, 1
	v_mov_b32_e32 v73, s74
	v_readlane_b32 s75, v25, 2
	v_mov_b32_e32 v66, s75
	v_readlane_b32 s75, v30, 2
	v_mov_b32_e32 v74, s75
	v_readlane_b32 s76, v25, 3
	v_mov_b32_e32 v67, s76
	v_readlane_b32 s76, v30, 3
	v_mov_b32_e32 v75, s76
	v_readlane_b32 s73, v25, 4
	v_mov_b32_e32 v68, s73
	v_readlane_b32 s73, v30, 4
	v_mov_b32_e32 v76, s73
	v_readlane_b32 s74, v25, 5
	v_mov_b32_e32 v69, s74
	v_readlane_b32 s74, v30, 5
	v_mov_b32_e32 v77, s74
	v_readlane_b32 s75, v25, 6
	v_mov_b32_e32 v70, s75
	v_readlane_b32 s75, v30, 6
	v_mov_b32_e32 v78, s75
	v_readlane_b32 s76, v25, 7
	v_mov_b32_e32 v71, s76
	v_readlane_b32 s76, v30, 7
	v_mov_b32_e32 v79, s76
	s_cmp_eq_u32 s48, 0
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r128_down_reduce
	.Lmqi4moe_mq2i4_r128_down_group:
	s_mul_i32 s55, s53, 32
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u8 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v64, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v65, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v66, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v67, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v68, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v69, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v70, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v71, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq2i4_r128_down_g_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v9.l
	v_bfe_u32 v20, v8, 0, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 1, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, -3, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v9, v20
	v_bfe_u32 v21, v8, 2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 1, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, -3, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v9, v21
	v_bfe_u32 v22, v8, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 1, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, -3, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v9, v22
	v_bfe_u32 v23, v8, 6, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 1, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, -3, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v9, v23
	v_mul_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_done
	v_mul_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_done
	v_mul_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_done
	v_mul_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_done
	v_mul_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_done
	v_mul_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_done
	v_mul_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_g_done
	v_mul_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v12
	.Lmqi4moe_mq2i4_r128_down_g_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc1 .Lmqi4moe_mq2i4_r128_down_group
	.Lmqi4moe_mq2i4_r128_down_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v14, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v14, v14, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	v_cndmask_b32_e64 v15, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v15, v15, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v16, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v16, v16, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	v_cndmask_b32_e64 v17, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v17, v17, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	v_cndmask_b32_e64 v18, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v18, v18, v0, 2
	ds_bpermute_b32 v13, v14, v56
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s68
	ds_bpermute_b32 v13, v15, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s69
	ds_bpermute_b32 v13, v16, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s70
	ds_bpermute_b32 v13, v17, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s71
	ds_bpermute_b32 v13, v18, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_red_done
	ds_bpermute_b32 v13, v14, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s68
	ds_bpermute_b32 v13, v15, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s69
	ds_bpermute_b32 v13, v16, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s70
	ds_bpermute_b32 v13, v17, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s71
	ds_bpermute_b32 v13, v18, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_red_done
	ds_bpermute_b32 v13, v14, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s68
	ds_bpermute_b32 v13, v15, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s69
	ds_bpermute_b32 v13, v16, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s70
	ds_bpermute_b32 v13, v17, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s71
	ds_bpermute_b32 v13, v18, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_red_done
	ds_bpermute_b32 v13, v14, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s68
	ds_bpermute_b32 v13, v15, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s69
	ds_bpermute_b32 v13, v16, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s70
	ds_bpermute_b32 v13, v17, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s71
	ds_bpermute_b32 v13, v18, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_red_done
	ds_bpermute_b32 v13, v14, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s68
	ds_bpermute_b32 v13, v15, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s69
	ds_bpermute_b32 v13, v16, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s70
	ds_bpermute_b32 v13, v17, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s71
	ds_bpermute_b32 v13, v18, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_red_done
	ds_bpermute_b32 v13, v14, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s68
	ds_bpermute_b32 v13, v15, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s69
	ds_bpermute_b32 v13, v16, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s70
	ds_bpermute_b32 v13, v17, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s71
	ds_bpermute_b32 v13, v18, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_red_done
	ds_bpermute_b32 v13, v14, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s68
	ds_bpermute_b32 v13, v15, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s69
	ds_bpermute_b32 v13, v16, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s70
	ds_bpermute_b32 v13, v17, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s71
	ds_bpermute_b32 v13, v18, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_red_done
	ds_bpermute_b32 v13, v14, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s68
	ds_bpermute_b32 v13, v15, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s69
	ds_bpermute_b32 v13, v16, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s70
	ds_bpermute_b32 v13, v17, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s71
	ds_bpermute_b32 v13, v18, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s72
	.Lmqi4moe_mq2i4_r128_down_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v72, v56, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_st_done
	global_store_b32 v73, v57, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_st_done
	global_store_b32 v74, v58, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_st_done
	global_store_b32 v75, v59, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_st_done
	global_store_b32 v76, v60, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_st_done
	global_store_b32 v77, v61, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_st_done
	global_store_b32 v78, v62, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq2i4_r128_down_st_done
	global_store_b32 v79, v63, s[58:59]
	.Lmqi4moe_mq2i4_r128_down_st_done:
	s_waitcnt_vscnt null, 0x0
	.Lmqi4moe_mq2i4_r128_down_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100_end:
.size mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100, .Lmq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100_end-mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 88
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
	.amdhsa_next_free_vgpr 80
	.amdhsa_next_free_sgpr 80
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100_end-mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100
.globl mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100
.p2align 8
.type mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100,@function
mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_lshl_b32 s51, s31, 2
	s_add_u32 s36, s12, s51
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s51
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_exit
	s_lshl_b32 s51, s34, 3
	s_add_u32 s36, s8, s51
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	s_lshr_b32 s48, s23, 8
	s_mul_i32 s49, s48, 0x88
	s_mul_i32 s51, s30, s49
	s_waitcnt lgkmcnt(0)
	s_add_u32 s44, s38, s51
	s_addc_u32 s45, s39, 0
	s_lshr_b32 s77, s22, 1
	s_sub_i32 s78, s30, s77
	s_cmp_lt_u32 s30, s77
	s_cselect_b64 s[58:59], s[18:19], s[20:21]
	s_cselect_b32 s79, s30, s78
	s_lshl_b32 s60, s79, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_add_i32 s63, s35, -1
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v3, 5, v0
	v_lshlrev_b32_e32 v4, 2, v0
	v_lshrrev_b32_e32 v5, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 2, v5
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
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_min_u32_e32 v35, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v35, s32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 2, v35
	global_load_b32 v28, v35, s[14:15]
	v_cvt_f32_u32_e32 v33, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v33, v33
	s_waitcnt vmcnt(0)
	v_cvt_f32_u32_e32 v30, v28
	s_waitcnt_depctr depctr_va_vdst(0)
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
	v_mov_b32_e32 v124, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v34, 0
	v_mov_b32_e32 v132, s73
	v_readlane_b32 s74, v29, 1
	v_mov_b32_e32 v125, s74
	v_readlane_b32 s74, v34, 1
	v_mov_b32_e32 v133, s74
	v_readlane_b32 s75, v29, 2
	v_mov_b32_e32 v126, s75
	v_readlane_b32 s75, v34, 2
	v_mov_b32_e32 v134, s75
	v_readlane_b32 s76, v29, 3
	v_mov_b32_e32 v127, s76
	v_readlane_b32 s76, v34, 3
	v_mov_b32_e32 v135, s76
	v_readlane_b32 s73, v29, 4
	v_mov_b32_e32 v128, s73
	v_readlane_b32 s73, v34, 4
	v_mov_b32_e32 v136, s73
	v_readlane_b32 s74, v29, 5
	v_mov_b32_e32 v129, s74
	v_readlane_b32 s74, v34, 5
	v_mov_b32_e32 v137, s74
	v_readlane_b32 s75, v29, 6
	v_mov_b32_e32 v130, s75
	v_readlane_b32 s75, v34, 6
	v_mov_b32_e32 v138, s75
	v_readlane_b32 s76, v29, 7
	v_mov_b32_e32 v131, s76
	v_readlane_b32 s76, v34, 7
	v_mov_b32_e32 v139, s76
	s_and_b32 s54, s48, -4
	s_cmp_eq_u32 s54, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_tail
	.Lmqi4moe_mq4sym_r256_gate_up_quad:
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v5
	v_add_nc_u32_e32 v6, s56, v3
	global_load_b32 v8, v1, s[44:45] offset:8
	global_load_b32 v10, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v124, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v125, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v126, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v127, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v128, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	buffer_load_b128 v[64:67], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v129, v6
	buffer_load_b128 v[68:71], v7, s[40:43], 0 offen
	buffer_load_b128 v[72:75], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v130, v6
	buffer_load_b128 v[76:79], v7, s[40:43], 0 offen
	buffer_load_b128 v[80:83], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v131, v6
	buffer_load_b128 v[84:87], v7, s[40:43], 0 offen
	buffer_load_b128 v[88:91], v7, s[40:43], 0 offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_q0_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v10.l
	v_lshrrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v10, v10.l
	v_bfe_u32 v20, v8, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v9, v20, v10
	v_bfe_u32 v21, v8, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v9, v21, v10
	v_bfe_u32 v22, v8, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v22, v9, v22, v10
	v_bfe_u32 v23, v8, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v23, v9, v23, v10
	v_bfe_u32 v24, v8, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v9, v24, v10
	v_bfe_u32 v25, v8, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v9, v25, v10
	v_bfe_u32 v26, v8, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v9, v26, v10
	v_bfe_u32 v27, v8, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v9, v27, v10
	v_mul_f32_e32 v11, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v96, v96, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v11, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v100, v100, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v104, v104, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v11, v21, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v108, v108, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v12, v21, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v112, v112, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v11, v21, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v116, v116, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q0_done
	v_mul_f32_e32 v12, v21, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v120, v120, v12
	.Lmqi4moe_mq4sym_r256_gate_up_q0_done:
	s_add_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v5
	v_add_nc_u32_e32 v6, s56, v3
	global_load_b32 v8, v1, s[44:45] offset:8
	global_load_b32 v10, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v124, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v125, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v126, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v127, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v128, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	buffer_load_b128 v[64:67], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v129, v6
	buffer_load_b128 v[68:71], v7, s[40:43], 0 offen
	buffer_load_b128 v[72:75], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v130, v6
	buffer_load_b128 v[76:79], v7, s[40:43], 0 offen
	buffer_load_b128 v[80:83], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v131, v6
	buffer_load_b128 v[84:87], v7, s[40:43], 0 offen
	buffer_load_b128 v[88:91], v7, s[40:43], 0 offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_q1_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v10.l
	v_lshrrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v10, v10.l
	v_bfe_u32 v20, v8, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v9, v20, v10
	v_bfe_u32 v21, v8, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v9, v21, v10
	v_bfe_u32 v22, v8, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v22, v9, v22, v10
	v_bfe_u32 v23, v8, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v23, v9, v23, v10
	v_bfe_u32 v24, v8, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v9, v24, v10
	v_bfe_u32 v25, v8, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v9, v25, v10
	v_bfe_u32 v26, v8, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v9, v26, v10
	v_bfe_u32 v27, v8, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v9, v27, v10
	v_mul_f32_e32 v11, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v93, v93, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v11, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v101, v101, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v11, v21, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v109, v109, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v12, v21, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v11, v21, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v117, v117, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q1_done
	v_mul_f32_e32 v12, v21, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v12
	.Lmqi4moe_mq4sym_r256_gate_up_q1_done:
	s_add_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v5
	v_add_nc_u32_e32 v6, s56, v3
	global_load_b32 v8, v1, s[44:45] offset:8
	global_load_b32 v10, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v124, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v125, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v126, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v127, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v128, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	buffer_load_b128 v[64:67], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v129, v6
	buffer_load_b128 v[68:71], v7, s[40:43], 0 offen
	buffer_load_b128 v[72:75], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v130, v6
	buffer_load_b128 v[76:79], v7, s[40:43], 0 offen
	buffer_load_b128 v[80:83], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v131, v6
	buffer_load_b128 v[84:87], v7, s[40:43], 0 offen
	buffer_load_b128 v[88:91], v7, s[40:43], 0 offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_q2_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v10.l
	v_lshrrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v10, v10.l
	v_bfe_u32 v20, v8, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v9, v20, v10
	v_bfe_u32 v21, v8, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v9, v21, v10
	v_bfe_u32 v22, v8, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v22, v9, v22, v10
	v_bfe_u32 v23, v8, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v23, v9, v23, v10
	v_bfe_u32 v24, v8, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v9, v24, v10
	v_bfe_u32 v25, v8, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v9, v25, v10
	v_bfe_u32 v26, v8, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v9, v26, v10
	v_bfe_u32 v27, v8, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v9, v27, v10
	v_mul_f32_e32 v11, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v94, v94, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v98, v98, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v11, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v102, v102, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v106, v106, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v11, v21, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v110, v110, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v12, v21, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v114, v114, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v11, v21, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v118, v118, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q2_done
	v_mul_f32_e32 v12, v21, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v122, v122, v12
	.Lmqi4moe_mq4sym_r256_gate_up_q2_done:
	s_add_i32 s53, s53, 1
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v5
	v_add_nc_u32_e32 v6, s56, v3
	global_load_b32 v8, v1, s[44:45] offset:8
	global_load_b32 v10, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v124, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v125, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v126, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v127, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v128, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	buffer_load_b128 v[64:67], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v129, v6
	buffer_load_b128 v[68:71], v7, s[40:43], 0 offen
	buffer_load_b128 v[72:75], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v130, v6
	buffer_load_b128 v[76:79], v7, s[40:43], 0 offen
	buffer_load_b128 v[80:83], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v131, v6
	buffer_load_b128 v[84:87], v7, s[40:43], 0 offen
	buffer_load_b128 v[88:91], v7, s[40:43], 0 offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_q3_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v10.l
	v_lshrrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v10, v10.l
	v_bfe_u32 v20, v8, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v9, v20, v10
	v_bfe_u32 v21, v8, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v9, v21, v10
	v_bfe_u32 v22, v8, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v22, v9, v22, v10
	v_bfe_u32 v23, v8, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v23, v9, v23, v10
	v_bfe_u32 v24, v8, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v9, v24, v10
	v_bfe_u32 v25, v8, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v9, v25, v10
	v_bfe_u32 v26, v8, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v9, v26, v10
	v_bfe_u32 v27, v8, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v9, v27, v10
	v_mul_f32_e32 v11, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v99, v99, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v11, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v103, v103, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v107, v107, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v11, v21, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v111, v111, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v12, v21, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v115, v115, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v11, v21, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v119, v119, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_q3_done
	v_mul_f32_e32 v12, v21, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v123, v123, v12
	.Lmqi4moe_mq4sym_r256_gate_up_q3_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s54
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r256_gate_up_quad
	.Lmqi4moe_mq4sym_r256_gate_up_tail:
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v5
	v_add_nc_u32_e32 v6, s56, v3
	global_load_b32 v8, v1, s[44:45] offset:8
	global_load_b32 v10, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v124, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v125, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v126, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v127, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v128, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	buffer_load_b128 v[64:67], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v129, v6
	buffer_load_b128 v[68:71], v7, s[40:43], 0 offen
	buffer_load_b128 v[72:75], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v130, v6
	buffer_load_b128 v[76:79], v7, s[40:43], 0 offen
	buffer_load_b128 v[80:83], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v131, v6
	buffer_load_b128 v[84:87], v7, s[40:43], 0 offen
	buffer_load_b128 v[88:91], v7, s[40:43], 0 offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t0_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v10.l
	v_lshrrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v10, v10.l
	v_bfe_u32 v20, v8, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v9, v20, v10
	v_bfe_u32 v21, v8, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v9, v21, v10
	v_bfe_u32 v22, v8, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v22, v9, v22, v10
	v_bfe_u32 v23, v8, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v23, v9, v23, v10
	v_bfe_u32 v24, v8, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v9, v24, v10
	v_bfe_u32 v25, v8, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v9, v25, v10
	v_bfe_u32 v26, v8, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v9, v26, v10
	v_bfe_u32 v27, v8, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v9, v27, v10
	v_mul_f32_e32 v11, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v96, v96, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v11, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v100, v100, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v104, v104, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v11, v21, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v108, v108, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v12, v21, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v112, v112, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v11, v21, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v116, v116, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t0_done
	v_mul_f32_e32 v12, v21, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v120, v120, v12
	.Lmqi4moe_mq4sym_r256_gate_up_t0_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v5
	v_add_nc_u32_e32 v6, s56, v3
	global_load_b32 v8, v1, s[44:45] offset:8
	global_load_b32 v10, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v124, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v125, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v126, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v127, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v128, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	buffer_load_b128 v[64:67], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v129, v6
	buffer_load_b128 v[68:71], v7, s[40:43], 0 offen
	buffer_load_b128 v[72:75], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v130, v6
	buffer_load_b128 v[76:79], v7, s[40:43], 0 offen
	buffer_load_b128 v[80:83], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v131, v6
	buffer_load_b128 v[84:87], v7, s[40:43], 0 offen
	buffer_load_b128 v[88:91], v7, s[40:43], 0 offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t1_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v10.l
	v_lshrrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v10, v10.l
	v_bfe_u32 v20, v8, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v9, v20, v10
	v_bfe_u32 v21, v8, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v9, v21, v10
	v_bfe_u32 v22, v8, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v22, v9, v22, v10
	v_bfe_u32 v23, v8, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v23, v9, v23, v10
	v_bfe_u32 v24, v8, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v9, v24, v10
	v_bfe_u32 v25, v8, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v9, v25, v10
	v_bfe_u32 v26, v8, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v9, v26, v10
	v_bfe_u32 v27, v8, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v9, v27, v10
	v_mul_f32_e32 v11, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v93, v93, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v97, v97, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v11, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v101, v101, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v105, v105, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v11, v21, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v109, v109, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v12, v21, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v113, v113, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v11, v21, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v117, v117, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t1_done
	v_mul_f32_e32 v12, v21, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v121, v121, v12
	.Lmqi4moe_mq4sym_r256_gate_up_t1_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_reduce
	s_mul_i32 s55, s53, 0x88
	s_lshl_b32 s56, s53, 10
	v_add_nc_u32_e32 v1, s55, v4
	v_add_nc_u32_e32 v2, s55, v5
	v_add_nc_u32_e32 v6, s56, v3
	global_load_b32 v8, v1, s[44:45] offset:8
	global_load_b32 v10, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v124, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v125, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v126, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v127, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v128, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	buffer_load_b128 v[64:67], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v129, v6
	buffer_load_b128 v[68:71], v7, s[40:43], 0 offen
	buffer_load_b128 v[72:75], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v130, v6
	buffer_load_b128 v[76:79], v7, s[40:43], 0 offen
	buffer_load_b128 v[80:83], v7, s[40:43], 0 offen offset:16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v131, v6
	buffer_load_b128 v[84:87], v7, s[40:43], 0 offen
	buffer_load_b128 v[88:91], v7, s[40:43], 0 offen offset:16
	.Lmqi4moe_mq4sym_r256_gate_up_t2_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v10.l
	v_lshrrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v10, v10.l
	v_bfe_u32 v20, v8, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v9, v20, v10
	v_bfe_u32 v21, v8, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v9, v21, v10
	v_bfe_u32 v22, v8, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v22, v9, v22, v10
	v_bfe_u32 v23, v8, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v23, v9, v23, v10
	v_bfe_u32 v24, v8, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v24, v9, v24, v10
	v_bfe_u32 v25, v8, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v9, v25, v10
	v_bfe_u32 v26, v8, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v26, v9, v26, v10
	v_bfe_u32 v27, v8, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v27, v9, v27, v10
	v_mul_f32_e32 v11, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v94, v94, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v98, v98, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v11, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v102, v102, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v106, v106, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v11, v21, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v110, v110, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v12, v21, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v114, v114, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v11, v21, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v24, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v25, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v26, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v27, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v118, v118, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_t2_done
	v_mul_f32_e32 v12, v21, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v24, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v25, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v26, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v27, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v122, v122, v12
	.Lmqi4moe_mq4sym_r256_gate_up_t2_done:
	s_add_i32 s53, s53, 1
	.Lmqi4moe_mq4sym_r256_gate_up_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v14, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v14, v14, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	v_cndmask_b32_e64 v15, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v15, v15, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v16, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v16, v16, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	v_cndmask_b32_e64 v17, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v17, v17, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	v_cndmask_b32_e64 v18, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v18, v18, v0, 2
	v_add_f32_e32 v92, v92, v93
	v_add_f32_e32 v94, v94, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v94
	ds_bpermute_b32 v13, v14, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v92, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v13, s68
	ds_bpermute_b32 v13, v15, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v92, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v13, s69
	ds_bpermute_b32 v13, v16, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v92, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v13, s70
	ds_bpermute_b32 v13, v17, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v92, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v13, s71
	ds_bpermute_b32 v13, v18, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v92, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v13, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v96, v96, v97
	v_add_f32_e32 v98, v98, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v96, v96, v98
	ds_bpermute_b32 v13, v14, v96
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v96, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v13, s68
	ds_bpermute_b32 v13, v15, v96
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v96, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v13, s69
	ds_bpermute_b32 v13, v16, v96
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v96, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v13, s70
	ds_bpermute_b32 v13, v17, v96
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v96, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v13, s71
	ds_bpermute_b32 v13, v18, v96
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v96, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v13, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v100, v100, v101
	v_add_f32_e32 v102, v102, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v100, v100, v102
	ds_bpermute_b32 v13, v14, v100
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v100, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v13, s68
	ds_bpermute_b32 v13, v15, v100
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v100, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v13, s69
	ds_bpermute_b32 v13, v16, v100
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v100, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v13, s70
	ds_bpermute_b32 v13, v17, v100
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v100, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v13, s71
	ds_bpermute_b32 v13, v18, v100
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v100, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v100, v100, v13, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v104, v104, v105
	v_add_f32_e32 v106, v106, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v104, v104, v106
	ds_bpermute_b32 v13, v14, v104
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v104, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v13, s68
	ds_bpermute_b32 v13, v15, v104
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v104, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v13, s69
	ds_bpermute_b32 v13, v16, v104
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v104, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v13, s70
	ds_bpermute_b32 v13, v17, v104
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v104, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v13, s71
	ds_bpermute_b32 v13, v18, v104
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v104, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v13, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v108, v108, v109
	v_add_f32_e32 v110, v110, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v108, v108, v110
	ds_bpermute_b32 v13, v14, v108
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v108, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v13, s68
	ds_bpermute_b32 v13, v15, v108
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v108, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v13, s69
	ds_bpermute_b32 v13, v16, v108
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v108, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v13, s70
	ds_bpermute_b32 v13, v17, v108
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v108, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v13, s71
	ds_bpermute_b32 v13, v18, v108
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v108, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v108, v108, v13, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v112, v112, v113
	v_add_f32_e32 v114, v114, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v112, v112, v114
	ds_bpermute_b32 v13, v14, v112
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v112, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v13, s68
	ds_bpermute_b32 v13, v15, v112
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v112, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v13, s69
	ds_bpermute_b32 v13, v16, v112
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v112, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v13, s70
	ds_bpermute_b32 v13, v17, v112
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v112, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v13, s71
	ds_bpermute_b32 v13, v18, v112
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v112, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v112, v112, v13, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v116, v116, v117
	v_add_f32_e32 v118, v118, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v116, v116, v118
	ds_bpermute_b32 v13, v14, v116
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v116, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v13, s68
	ds_bpermute_b32 v13, v15, v116
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v116, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v13, s69
	ds_bpermute_b32 v13, v16, v116
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v116, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v13, s70
	ds_bpermute_b32 v13, v17, v116
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v116, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v13, s71
	ds_bpermute_b32 v13, v18, v116
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v116, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v116, v116, v13, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v120, v120, v121
	v_add_f32_e32 v122, v122, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v120, v120, v122
	ds_bpermute_b32 v13, v14, v120
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v120, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v13, s68
	ds_bpermute_b32 v13, v15, v120
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v120, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v13, s69
	ds_bpermute_b32 v13, v16, v120
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v120, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v13, s70
	ds_bpermute_b32 v13, v17, v120
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v120, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v13, s71
	ds_bpermute_b32 v13, v18, v120
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v120, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v13, s72
	.Lmqi4moe_mq4sym_r256_gate_up_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v132, v92, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v133, v96, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v134, v100, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v135, v104, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v136, v108, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v137, v112, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v138, v116, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r256_gate_up_st_done
	global_store_b32 v139, v120, s[58:59]
	.Lmqi4moe_mq4sym_r256_gate_up_st_done:
	s_waitcnt_vscnt null, 0x0
	.Lmqi4moe_mq4sym_r256_gate_up_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100_end:
.size mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100, .Lmq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100_end-mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 88
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
	.amdhsa_next_free_sgpr 80
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100_end-mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100
.globl mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100
.p2align 8
.type mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100,@function
mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_lshl_b32 s51, s31, 2
	s_add_u32 s36, s12, s51
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s51
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_exit
	s_lshl_b32 s51, s34, 3
	s_add_u32 s36, s8, s51
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	s_lshr_b32 s48, s23, 7
	s_mul_i32 s49, s48, 0x44
	s_mul_i32 s51, s30, s49
	s_waitcnt lgkmcnt(0)
	s_add_u32 s44, s38, s51
	s_addc_u32 s45, s39, 0
	s_mov_b32 s58, s18
	s_mov_b32 s59, s19
	s_lshl_b32 s60, s30, 2
	s_lshl_b32 s61, s28, 2
	s_lshl_b32 s62, s29, 2
	s_add_i32 s63, s35, -1
	s_mov_b32 s40, s16
	s_mov_b32 s41, s17
	s_mul_i32 s51, s27, s61
	s_mov_b32 s42, s51
	s_mov_b32 s43, 0x31004000
	s_mov_b32 s53, 0
	v_lshlrev_b32_e32 v3, 4, v0
	v_lshlrev_b32_e32 v4, 1, v0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_min_u32_e32 v31, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, s32, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 2, v31
	global_load_b32 v24, v31, s[14:15]
	v_cvt_f32_u32_e32 v29, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v29, v29
	s_waitcnt vmcnt(0)
	v_cvt_f32_u32_e32 v26, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v26, v26, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v27, v26, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v27, v24, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s66, 0, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, -1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s66
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, s26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v27, v27, v28, s66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s66, s26, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s66
	v_cmp_le_i32_e64 s64, 0, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v25, 0x7fffff00, v26, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v25, v25, s61
	v_mul_lo_u32 v30, v24, s62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, s60, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v25, 0
	v_mov_b32_e32 v64, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v30, 0
	v_mov_b32_e32 v72, s73
	v_readlane_b32 s74, v25, 1
	v_mov_b32_e32 v65, s74
	v_readlane_b32 s74, v30, 1
	v_mov_b32_e32 v73, s74
	v_readlane_b32 s75, v25, 2
	v_mov_b32_e32 v66, s75
	v_readlane_b32 s75, v30, 2
	v_mov_b32_e32 v74, s75
	v_readlane_b32 s76, v25, 3
	v_mov_b32_e32 v67, s76
	v_readlane_b32 s76, v30, 3
	v_mov_b32_e32 v75, s76
	v_readlane_b32 s73, v25, 4
	v_mov_b32_e32 v68, s73
	v_readlane_b32 s73, v30, 4
	v_mov_b32_e32 v76, s73
	v_readlane_b32 s74, v25, 5
	v_mov_b32_e32 v69, s74
	v_readlane_b32 s74, v30, 5
	v_mov_b32_e32 v77, s74
	v_readlane_b32 s75, v25, 6
	v_mov_b32_e32 v70, s75
	v_readlane_b32 s75, v30, 6
	v_mov_b32_e32 v78, s75
	v_readlane_b32 s76, v25, 7
	v_mov_b32_e32 v71, s76
	v_readlane_b32 s76, v30, 7
	v_mov_b32_e32 v79, s76
	s_cmp_eq_u32 s48, 0
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_reduce
	.Lmqi4moe_mq4sym_r128_down_group:
	s_mul_i32 s55, s53, 0x44
	s_lshl_b32 s56, s53, 9
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s55
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45] offset:4
	global_load_b32 v10, v2, s[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v64, v6
	buffer_load_b128 v[24:27], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v65, v6
	buffer_load_b128 v[28:31], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v66, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v67, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v68, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v69, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v70, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v71, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	.Lmqi4moe_mq4sym_r128_down_g_xdone:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v9, v10.l
	v_lshrrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v10, v10.l
	v_bfe_u32 v20, v8, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v9, v20, v10
	v_bfe_u32 v21, v8, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v9, v21, v10
	v_bfe_u32 v22, v8, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v22, v9, v22, v10
	v_bfe_u32 v23, v8, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v23, v9, v23, v10
	v_mul_f32_e32 v11, v21, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v56, v56, v11
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v12, v21, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v57, v57, v12
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v11, v21, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v58, v58, v11
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v12, v21, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v12
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v11, v21, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v11
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v12, v21, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v61, v61, v12
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v11, v21, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v20, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v22, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v62, v62, v11
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_g_done
	v_mul_f32_e32 v12, v21, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v20, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v22, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v12
	.Lmqi4moe_mq4sym_r128_down_g_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc1 .Lmqi4moe_mq4sym_r128_down_group
	.Lmqi4moe_mq4sym_r128_down_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v14, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v14, v14, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	v_cndmask_b32_e64 v15, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v15, v15, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v16, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v16, v16, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	v_cndmask_b32_e64 v17, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v17, v17, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	v_cndmask_b32_e64 v18, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v18, v18, v0, 2
	ds_bpermute_b32 v13, v14, v56
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s68
	ds_bpermute_b32 v13, v15, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s69
	ds_bpermute_b32 v13, v16, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s70
	ds_bpermute_b32 v13, v17, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s71
	ds_bpermute_b32 v13, v18, v56
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v56, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v56, v56, v13, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v13, v14, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s68
	ds_bpermute_b32 v13, v15, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s69
	ds_bpermute_b32 v13, v16, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s70
	ds_bpermute_b32 v13, v17, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s71
	ds_bpermute_b32 v13, v18, v57
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v57, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v57, v57, v13, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v13, v14, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s68
	ds_bpermute_b32 v13, v15, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s69
	ds_bpermute_b32 v13, v16, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s70
	ds_bpermute_b32 v13, v17, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s71
	ds_bpermute_b32 v13, v18, v58
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v58, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v58, v58, v13, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v13, v14, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s68
	ds_bpermute_b32 v13, v15, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s69
	ds_bpermute_b32 v13, v16, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s70
	ds_bpermute_b32 v13, v17, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s71
	ds_bpermute_b32 v13, v18, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v59, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v59, v59, v13, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v13, v14, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s68
	ds_bpermute_b32 v13, v15, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s69
	ds_bpermute_b32 v13, v16, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s70
	ds_bpermute_b32 v13, v17, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s71
	ds_bpermute_b32 v13, v18, v60
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v60, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v60, v60, v13, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v13, v14, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s68
	ds_bpermute_b32 v13, v15, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s69
	ds_bpermute_b32 v13, v16, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s70
	ds_bpermute_b32 v13, v17, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s71
	ds_bpermute_b32 v13, v18, v61
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v61, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, v61, v13, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v13, v14, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s68
	ds_bpermute_b32 v13, v15, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s69
	ds_bpermute_b32 v13, v16, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s70
	ds_bpermute_b32 v13, v17, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s71
	ds_bpermute_b32 v13, v18, v62
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v62, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v13, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_red_done
	ds_bpermute_b32 v13, v14, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s68
	ds_bpermute_b32 v13, v15, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s69
	ds_bpermute_b32 v13, v16, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s70
	ds_bpermute_b32 v13, v17, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s71
	ds_bpermute_b32 v13, v18, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v63, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v13, s72
	.Lmqi4moe_mq4sym_r128_down_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v72, v56, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v73, v57, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v74, v58, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v75, v59, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v76, v60, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v77, v61, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v78, v62, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lmqi4moe_mq4sym_r128_down_st_done
	global_store_b32 v79, v63, s[58:59]
	.Lmqi4moe_mq4sym_r128_down_st_done:
	s_waitcnt_vscnt null, 0x0
	.Lmqi4moe_mq4sym_r128_down_exit:
	s_endpgm
.Lmq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100_end:
.size mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100, .Lmq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100_end-mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 88
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
	.amdhsa_next_free_vgpr 80
	.amdhsa_next_free_sgpr 80
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100_end-mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100)<<4)&1008)>>4
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .max_flat_workgroup_size: 32
    .name: mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 80
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 104
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .max_flat_workgroup_size: 32
    .name: mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 80
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100.kd
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .max_flat_workgroup_size: 32
    .name: mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 80
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100.kd
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .max_flat_workgroup_size: 32
    .name: mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 80
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 80
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
