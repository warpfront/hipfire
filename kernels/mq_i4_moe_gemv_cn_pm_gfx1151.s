.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected mq_i4_moe_gemv_cn_mq3i4_r256_gate_up
.globl mq_i4_moe_gemv_cn_mq3i4_r256_gate_up
.p2align 8
.type mq_i4_moe_gemv_cn_mq3i4_r256_gate_up,@function
mq_i4_moe_gemv_cn_mq3i4_r256_gate_up:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s100, s[0:1], 0x58
	v_lshrrev_b32_e32 v80, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v80
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v5, 4, v0
	v_mov_b32_e32 v4, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_u32_u24_e32 v80, 12, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v6, 5, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v6, 2, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v7, 31, v80
	s_mul_i32 s41, s40, 0x1400
	v_add_nc_u32_e32 v1, s41, v2
	s_lshl_b32 s30, s2, 3
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s3, s24
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_exit
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_exit
	s_cmp_lg_u32 s23, 0xa00
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_exit
	s_lshl_b32 s41, s3, 2
	s_add_u32 s36, s12, s41
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s41
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_add_i32 s30, s30, s40
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_exit
	s_mul_i32 s41, s25, 10
	s_cmp_gt_u32 s33, s41
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_exit
	s_cmp_ge_u32 s34, s100
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_exit
	s_lshl_b32 s41, s34, 3
	s_add_u32 s36, s8, s41
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	s_add_i32 s43, s35, -1
	s_min_u32 s41, s43, 0
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s58, s[36:37], null
	s_min_u32 s41, s43, 1
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s59, s[36:37], null
	s_min_u32 s41, s43, 2
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s60, s[36:37], null
	s_min_u32 s41, s43, 3
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s61, s[36:37], null
	s_min_u32 s41, s43, 4
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s62, s[36:37], null
	s_min_u32 s41, s43, 5
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s63, s[36:37], null
	s_min_u32 s41, s43, 6
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s64, s[36:37], null
	s_min_u32 s41, s43, 7
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s65, s[36:37], null
	s_min_u32 s41, s43, s40
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s57, s[36:37], null
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v80, s57
	v_cvt_f32_u32_e32 v85, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v85, v85
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_u32_e32 v82, v80
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v82, v82, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v83, v82, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v83, v80, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s80, 0, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v84, -1, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v82, v82, v84, s80
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v84, s26, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v83, v83, v84, s80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s80, s26, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v84, 1, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v82, v82, v84, s80
	v_cmp_le_i32_e64 s78, 0, v80
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v82, 1, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v81, 0x7fffff00, v82, s78
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s66, v81
	s_lshl_b32 s55, s28, 2
	s_mul_i32 s67, s66, s55
	v_add_nc_u32_e32 v2, s67, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, 0x1000, v2
	s_lshr_b32 s69, s22, 1
	s_sub_i32 s70, s30, s69
	s_cmp_lt_u32 s30, s69
	s_cselect_b64 s[52:53], s[18:19], s[20:21]
	s_cselect_b32 s71, s30, s70
	s_lshl_b32 s54, s71, 2
	s_lshl_b32 s56, s29, 2
	s_cmp_lt_u32 s40, s35
	s_cselect_b32 s68, -1, 0
	s_cmp_lt_u32 s66, s27
	s_cselect_b32 s77, -1, 0
	s_and_b32 s77, s77, s68
	s_xor_b32 s82, s68, s77
	.Lcn_mq3i4_r256_gate_up_body:
	s_mul_i32 s41, s30, 0x3c0
	s_add_u32 s48, s38, s41
	s_addc_u32 s49, s39, 0
	s_mul_i32 s41, s22, 0x3c0
	s_mul_i32 s42, s30, 40
	s_add_i32 s41, s41, s42
	s_add_u32 s50, s38, s41
	s_addc_u32 s51, s39, 0
	s_load_b256 s[88:95], s[50:51], null
	s_load_b64 s[96:97], s[50:51], 0x20
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_z0
	s_mov_b32 exec_lo, s82
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
	s_mov_b32 exec_lo, -1
	.Lcn_mq3i4_r256_gate_up_z0:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[80:83], v2, s[16:17]
	global_load_b128 v[84:87], v2, s[16:17] offset:512
	global_load_b128 v[88:91], v2, s[16:17] offset:1024
	global_load_b128 v[92:95], v2, s[16:17] offset:1536
	global_load_b128 v[96:99], v2, s[16:17] offset:2048
	global_load_b128 v[100:103], v2, s[16:17] offset:2560
	global_load_b128 v[104:107], v2, s[16:17] offset:3072
	global_load_b128 v[108:111], v2, s[16:17] offset:3584
	global_load_b128 v[112:115], v3, s[16:17]
	global_load_b128 v[116:119], v3, s[16:17] offset:512
	s_mov_b32 exec_lo, -1
	global_load_b64 v[40:41], v6, s[48:49]
	global_load_b64 v[42:43], v6, s[48:49] offset:48
	global_load_b64 v[44:45], v6, s[48:49] offset:96
	global_load_b64 v[46:47], v6, s[48:49] offset:144
	global_load_b64 v[48:49], v6, s[48:49] offset:192
	global_load_b64 v[50:51], v6, s[48:49] offset:240
	global_load_b64 v[52:53], v6, s[48:49] offset:288
	global_load_b64 v[54:55], v6, s[48:49] offset:336
	global_load_b64 v[56:57], v6, s[48:49] offset:384
	global_load_b64 v[58:59], v6, s[48:49] offset:432
	global_load_b64 v[60:61], v6, s[48:49] offset:480
	global_load_b64 v[62:63], v6, s[48:49] offset:528
	global_load_b64 v[64:65], v6, s[48:49] offset:576
	global_load_b64 v[66:67], v6, s[48:49] offset:624
	global_load_b64 v[68:69], v6, s[48:49] offset:672
	global_load_b64 v[70:71], v6, s[48:49] offset:720
	global_load_b64 v[72:73], v6, s[48:49] offset:768
	global_load_b64 v[74:75], v6, s[48:49] offset:816
	global_load_b64 v[76:77], v6, s[48:49] offset:864
	global_load_b64 v[78:79], v6, s[48:49] offset:912
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	s_waitcnt vmcnt(20)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[80:83]
	ds_store_b128 v1, v[84:87] offset:512
	ds_store_b128 v1, v[88:91] offset:1024
	ds_store_b128 v1, v[92:95] offset:1536
	ds_store_b128 v1, v[96:99] offset:2048
	ds_store_b128 v1, v[100:103] offset:2560
	ds_store_b128 v1, v[104:107] offset:3072
	ds_store_b128 v1, v[108:111] offset:3584
	ds_store_b128 v1, v[112:115] offset:4096
	ds_store_b128 v1, v[116:119] offset:4608
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq3i4_r256_gate_up_c0:
	ds_load_b128 v[92:95], v5
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_lw
	ds_load_b128 v[96:99], v5 offset:5120
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_lw
	ds_load_b128 v[100:103], v5 offset:10240
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_lw
	ds_load_b128 v[104:107], v5 offset:15360
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_lw
	ds_load_b128 v[108:111], v5 offset:20480
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_lw
	ds_load_b128 v[112:115], v5 offset:25600
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_lw
	ds_load_b128 v[116:119], v5 offset:30720
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_lw
	ds_load_b128 v[120:123], v5 offset:35840
	.Lcn_mq3i4_r256_gate_up_g0_b0_lw:
	v_mov_b32_e32 v84, s88
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(19)
	v_lshrrev_b64 v[86:87], v7, v[40:41]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_g0_b0_ld:
	ds_load_b128 v[92:95], v5 offset:512
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_lw
	ds_load_b128 v[96:99], v5 offset:5632
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_lw
	ds_load_b128 v[100:103], v5 offset:10752
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_lw
	ds_load_b128 v[104:107], v5 offset:15872
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_lw
	ds_load_b128 v[108:111], v5 offset:20992
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_lw
	ds_load_b128 v[112:115], v5 offset:26112
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_lw
	ds_load_b128 v[116:119], v5 offset:31232
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_lw
	ds_load_b128 v[120:123], v5 offset:36352
	.Lcn_mq3i4_r256_gate_up_g1_b0_lw:
	s_lshr_b32 s98, s88, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(18)
	v_lshrrev_b64 v[86:87], v7, v[42:43]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_g1_b0_ld:
	ds_load_b128 v[92:95], v5 offset:1024
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_lw
	ds_load_b128 v[96:99], v5 offset:6144
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_lw
	ds_load_b128 v[100:103], v5 offset:11264
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_lw
	ds_load_b128 v[104:107], v5 offset:16384
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_lw
	ds_load_b128 v[108:111], v5 offset:21504
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_lw
	ds_load_b128 v[112:115], v5 offset:26624
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_lw
	ds_load_b128 v[116:119], v5 offset:31744
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_lw
	ds_load_b128 v[120:123], v5 offset:36864
	.Lcn_mq3i4_r256_gate_up_g2_b0_lw:
	v_mov_b32_e32 v84, s89
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(17)
	v_lshrrev_b64 v[86:87], v7, v[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_g2_b0_ld:
	ds_load_b128 v[92:95], v5 offset:1536
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_lw
	ds_load_b128 v[96:99], v5 offset:6656
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_lw
	ds_load_b128 v[100:103], v5 offset:11776
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_lw
	ds_load_b128 v[104:107], v5 offset:16896
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_lw
	ds_load_b128 v[108:111], v5 offset:22016
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_lw
	ds_load_b128 v[112:115], v5 offset:27136
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_lw
	ds_load_b128 v[116:119], v5 offset:32256
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_lw
	ds_load_b128 v[120:123], v5 offset:37376
	.Lcn_mq3i4_r256_gate_up_g3_b0_lw:
	s_lshr_b32 s99, s89, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(16)
	v_lshrrev_b64 v[86:87], v7, v[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_g3_b0_ld:
	ds_load_b128 v[92:95], v5 offset:2048
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_lw
	ds_load_b128 v[96:99], v5 offset:7168
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_lw
	ds_load_b128 v[100:103], v5 offset:12288
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_lw
	ds_load_b128 v[104:107], v5 offset:17408
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_lw
	ds_load_b128 v[108:111], v5 offset:22528
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_lw
	ds_load_b128 v[112:115], v5 offset:27648
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_lw
	ds_load_b128 v[116:119], v5 offset:32768
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_lw
	ds_load_b128 v[120:123], v5 offset:37888
	.Lcn_mq3i4_r256_gate_up_g4_b0_lw:
	v_mov_b32_e32 v84, s90
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(15)
	v_lshrrev_b64 v[86:87], v7, v[48:49]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_g4_b0_ld:
	ds_load_b128 v[92:95], v5 offset:2560
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_lw
	ds_load_b128 v[96:99], v5 offset:7680
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_lw
	ds_load_b128 v[100:103], v5 offset:12800
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_lw
	ds_load_b128 v[104:107], v5 offset:17920
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_lw
	ds_load_b128 v[108:111], v5 offset:23040
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_lw
	ds_load_b128 v[112:115], v5 offset:28160
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_lw
	ds_load_b128 v[116:119], v5 offset:33280
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_lw
	ds_load_b128 v[120:123], v5 offset:38400
	.Lcn_mq3i4_r256_gate_up_g5_b0_lw:
	s_lshr_b32 s98, s90, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(14)
	v_lshrrev_b64 v[86:87], v7, v[50:51]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_g5_b0_ld:
	ds_load_b128 v[92:95], v5 offset:3072
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_lw
	ds_load_b128 v[96:99], v5 offset:8192
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_lw
	ds_load_b128 v[100:103], v5 offset:13312
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_lw
	ds_load_b128 v[104:107], v5 offset:18432
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_lw
	ds_load_b128 v[108:111], v5 offset:23552
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_lw
	ds_load_b128 v[112:115], v5 offset:28672
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_lw
	ds_load_b128 v[116:119], v5 offset:33792
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_lw
	ds_load_b128 v[120:123], v5 offset:38912
	.Lcn_mq3i4_r256_gate_up_g6_b0_lw:
	v_mov_b32_e32 v84, s91
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(13)
	v_lshrrev_b64 v[86:87], v7, v[52:53]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_g6_b0_ld:
	ds_load_b128 v[92:95], v5 offset:3584
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_lw
	ds_load_b128 v[96:99], v5 offset:8704
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_lw
	ds_load_b128 v[100:103], v5 offset:13824
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_lw
	ds_load_b128 v[104:107], v5 offset:18944
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_lw
	ds_load_b128 v[108:111], v5 offset:24064
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_lw
	ds_load_b128 v[112:115], v5 offset:29184
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_lw
	ds_load_b128 v[116:119], v5 offset:34304
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_lw
	ds_load_b128 v[120:123], v5 offset:39424
	.Lcn_mq3i4_r256_gate_up_g7_b0_lw:
	s_lshr_b32 s99, s91, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(12)
	v_lshrrev_b64 v[86:87], v7, v[54:55]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_g7_b0_ld:
	ds_load_b128 v[92:95], v5 offset:4096
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_lw
	ds_load_b128 v[96:99], v5 offset:9216
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_lw
	ds_load_b128 v[100:103], v5 offset:14336
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_lw
	ds_load_b128 v[104:107], v5 offset:19456
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_lw
	ds_load_b128 v[108:111], v5 offset:24576
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_lw
	ds_load_b128 v[112:115], v5 offset:29696
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_lw
	ds_load_b128 v[116:119], v5 offset:34816
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_lw
	ds_load_b128 v[120:123], v5 offset:39936
	.Lcn_mq3i4_r256_gate_up_g8_b0_lw:
	v_mov_b32_e32 v84, s92
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(11)
	v_lshrrev_b64 v[86:87], v7, v[56:57]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_g8_b0_ld:
	ds_load_b128 v[92:95], v5 offset:4608
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_lw
	ds_load_b128 v[96:99], v5 offset:9728
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_lw
	ds_load_b128 v[100:103], v5 offset:14848
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_lw
	ds_load_b128 v[104:107], v5 offset:19968
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_lw
	ds_load_b128 v[108:111], v5 offset:25088
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_lw
	ds_load_b128 v[112:115], v5 offset:30208
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_lw
	ds_load_b128 v[116:119], v5 offset:35328
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_lw
	ds_load_b128 v[120:123], v5 offset:40448
	.Lcn_mq3i4_r256_gate_up_g9_b0_lw:
	s_lshr_b32 s98, s92, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	s_waitcnt vmcnt(10)
	v_lshrrev_b64 v[86:87], v7, v[58:59]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_g9_b0_ld:
	.Lcn_mq3i4_r256_gate_up_s1:
	v_add_nc_u32_e32 v2, 0x1400, v2
	v_add_nc_u32_e32 v3, 0x1400, v3
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_z1
	s_mov_b32 exec_lo, s82
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
	s_mov_b32 exec_lo, -1
	.Lcn_mq3i4_r256_gate_up_z1:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[80:83], v2, s[16:17]
	global_load_b128 v[84:87], v2, s[16:17] offset:512
	global_load_b128 v[88:91], v2, s[16:17] offset:1024
	global_load_b128 v[92:95], v2, s[16:17] offset:1536
	global_load_b128 v[96:99], v2, s[16:17] offset:2048
	global_load_b128 v[100:103], v2, s[16:17] offset:2560
	global_load_b128 v[104:107], v2, s[16:17] offset:3072
	global_load_b128 v[108:111], v2, s[16:17] offset:3584
	global_load_b128 v[112:115], v3, s[16:17]
	global_load_b128 v[116:119], v3, s[16:17] offset:512
	s_mov_b32 exec_lo, -1
	s_barrier
	s_waitcnt vmcnt(0)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[80:83]
	ds_store_b128 v1, v[84:87] offset:512
	ds_store_b128 v1, v[88:91] offset:1024
	ds_store_b128 v1, v[92:95] offset:1536
	ds_store_b128 v1, v[96:99] offset:2048
	ds_store_b128 v1, v[100:103] offset:2560
	ds_store_b128 v1, v[104:107] offset:3072
	ds_store_b128 v1, v[108:111] offset:3584
	ds_store_b128 v1, v[112:115] offset:4096
	ds_store_b128 v1, v[116:119] offset:4608
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq3i4_r256_gate_up_c1:
	ds_load_b128 v[92:95], v5
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_lw
	ds_load_b128 v[96:99], v5 offset:5120
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_lw
	ds_load_b128 v[100:103], v5 offset:10240
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_lw
	ds_load_b128 v[104:107], v5 offset:15360
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_lw
	ds_load_b128 v[108:111], v5 offset:20480
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_lw
	ds_load_b128 v[112:115], v5 offset:25600
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_lw
	ds_load_b128 v[116:119], v5 offset:30720
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_lw
	ds_load_b128 v[120:123], v5 offset:35840
	.Lcn_mq3i4_r256_gate_up_g10_b0_lw:
	v_mov_b32_e32 v84, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[60:61]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g10_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_g10_b0_ld:
	ds_load_b128 v[92:95], v5 offset:512
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_lw
	ds_load_b128 v[96:99], v5 offset:5632
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_lw
	ds_load_b128 v[100:103], v5 offset:10752
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_lw
	ds_load_b128 v[104:107], v5 offset:15872
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_lw
	ds_load_b128 v[108:111], v5 offset:20992
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_lw
	ds_load_b128 v[112:115], v5 offset:26112
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_lw
	ds_load_b128 v[116:119], v5 offset:31232
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_lw
	ds_load_b128 v[120:123], v5 offset:36352
	.Lcn_mq3i4_r256_gate_up_g11_b0_lw:
	s_lshr_b32 s99, s93, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[62:63]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g11_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_g11_b0_ld:
	ds_load_b128 v[92:95], v5 offset:1024
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_lw
	ds_load_b128 v[96:99], v5 offset:6144
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_lw
	ds_load_b128 v[100:103], v5 offset:11264
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_lw
	ds_load_b128 v[104:107], v5 offset:16384
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_lw
	ds_load_b128 v[108:111], v5 offset:21504
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_lw
	ds_load_b128 v[112:115], v5 offset:26624
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_lw
	ds_load_b128 v[116:119], v5 offset:31744
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_lw
	ds_load_b128 v[120:123], v5 offset:36864
	.Lcn_mq3i4_r256_gate_up_g12_b0_lw:
	v_mov_b32_e32 v84, s94
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[64:65]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g12_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_g12_b0_ld:
	ds_load_b128 v[92:95], v5 offset:1536
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_lw
	ds_load_b128 v[96:99], v5 offset:6656
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_lw
	ds_load_b128 v[100:103], v5 offset:11776
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_lw
	ds_load_b128 v[104:107], v5 offset:16896
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_lw
	ds_load_b128 v[108:111], v5 offset:22016
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_lw
	ds_load_b128 v[112:115], v5 offset:27136
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_lw
	ds_load_b128 v[116:119], v5 offset:32256
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_lw
	ds_load_b128 v[120:123], v5 offset:37376
	.Lcn_mq3i4_r256_gate_up_g13_b0_lw:
	s_lshr_b32 s98, s94, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[66:67]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g13_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_g13_b0_ld:
	ds_load_b128 v[92:95], v5 offset:2048
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_lw
	ds_load_b128 v[96:99], v5 offset:7168
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_lw
	ds_load_b128 v[100:103], v5 offset:12288
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_lw
	ds_load_b128 v[104:107], v5 offset:17408
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_lw
	ds_load_b128 v[108:111], v5 offset:22528
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_lw
	ds_load_b128 v[112:115], v5 offset:27648
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_lw
	ds_load_b128 v[116:119], v5 offset:32768
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_lw
	ds_load_b128 v[120:123], v5 offset:37888
	.Lcn_mq3i4_r256_gate_up_g14_b0_lw:
	v_mov_b32_e32 v84, s95
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[68:69]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g14_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_g14_b0_ld:
	ds_load_b128 v[92:95], v5 offset:2560
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_lw
	ds_load_b128 v[96:99], v5 offset:7680
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_lw
	ds_load_b128 v[100:103], v5 offset:12800
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_lw
	ds_load_b128 v[104:107], v5 offset:17920
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_lw
	ds_load_b128 v[108:111], v5 offset:23040
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_lw
	ds_load_b128 v[112:115], v5 offset:28160
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_lw
	ds_load_b128 v[116:119], v5 offset:33280
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_lw
	ds_load_b128 v[120:123], v5 offset:38400
	.Lcn_mq3i4_r256_gate_up_g15_b0_lw:
	s_lshr_b32 s99, s95, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[70:71]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g15_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_g15_b0_ld:
	ds_load_b128 v[92:95], v5 offset:3072
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_lw
	ds_load_b128 v[96:99], v5 offset:8192
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_lw
	ds_load_b128 v[100:103], v5 offset:13312
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_lw
	ds_load_b128 v[104:107], v5 offset:18432
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_lw
	ds_load_b128 v[108:111], v5 offset:23552
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_lw
	ds_load_b128 v[112:115], v5 offset:28672
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_lw
	ds_load_b128 v[116:119], v5 offset:33792
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_lw
	ds_load_b128 v[120:123], v5 offset:38912
	.Lcn_mq3i4_r256_gate_up_g16_b0_lw:
	v_mov_b32_e32 v84, s96
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[72:73]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g16_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_g16_b0_ld:
	ds_load_b128 v[92:95], v5 offset:3584
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_lw
	ds_load_b128 v[96:99], v5 offset:8704
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_lw
	ds_load_b128 v[100:103], v5 offset:13824
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_lw
	ds_load_b128 v[104:107], v5 offset:18944
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_lw
	ds_load_b128 v[108:111], v5 offset:24064
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_lw
	ds_load_b128 v[112:115], v5 offset:29184
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_lw
	ds_load_b128 v[116:119], v5 offset:34304
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_lw
	ds_load_b128 v[120:123], v5 offset:39424
	.Lcn_mq3i4_r256_gate_up_g17_b0_lw:
	s_lshr_b32 s98, s96, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[74:75]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g17_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_g17_b0_ld:
	ds_load_b128 v[92:95], v5 offset:4096
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_lw
	ds_load_b128 v[96:99], v5 offset:9216
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_lw
	ds_load_b128 v[100:103], v5 offset:14336
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_lw
	ds_load_b128 v[104:107], v5 offset:19456
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_lw
	ds_load_b128 v[108:111], v5 offset:24576
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_lw
	ds_load_b128 v[112:115], v5 offset:29696
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_lw
	ds_load_b128 v[116:119], v5 offset:34816
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_lw
	ds_load_b128 v[120:123], v5 offset:39936
	.Lcn_mq3i4_r256_gate_up_g18_b0_lw:
	v_mov_b32_e32 v84, s97
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[76:77]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g18_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_g18_b0_ld:
	ds_load_b128 v[92:95], v5 offset:4608
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_lw
	ds_load_b128 v[96:99], v5 offset:9728
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_lw
	ds_load_b128 v[100:103], v5 offset:14848
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_lw
	ds_load_b128 v[104:107], v5 offset:19968
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_lw
	ds_load_b128 v[108:111], v5 offset:25088
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_lw
	ds_load_b128 v[112:115], v5 offset:30208
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_lw
	ds_load_b128 v[116:119], v5 offset:35328
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_lw
	ds_load_b128 v[120:123], v5 offset:40448
	.Lcn_mq3i4_r256_gate_up_g19_b0_lw:
	s_lshr_b32 s99, s97, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[78:79]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_g19_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_g19_b0_ld:
	.Lcn_mq3i4_r256_gate_up_epi:
	v_cmp_gt_u32_e64 s72, 16, v0
	v_cndmask_b32_e64 v80, 0, 16, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v80, v80, v0, 2
	v_cmp_gt_u32_e64 s73, 24, v0
	v_cndmask_b32_e64 v81, 0, 8, s73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v81, v81, v0, 2
	v_cmp_gt_u32_e64 s74, 28, v0
	v_cndmask_b32_e64 v82, 0, 4, s74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v82, v82, v0, 2
	v_cmp_gt_u32_e64 s75, 30, v0
	v_cndmask_b32_e64 v83, 0, 2, s75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v83, v83, v0, 2
	v_cmp_gt_u32_e64 s76, 31, v0
	v_cndmask_b32_e64 v84, 0, 1, s76
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v84, v84, v0, 2
	v_add_f32_e32 v8, v8, v9
	v_add_f32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v10
	ds_bpermute_b32 v85, v80, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s72
	ds_bpermute_b32 v85, v81, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s73
	ds_bpermute_b32 v85, v82, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s74
	ds_bpermute_b32 v85, v83, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s75
	ds_bpermute_b32 v85, v84, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s76
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v12, v12, v13
	v_add_f32_e32 v14, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v14
	ds_bpermute_b32 v85, v80, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s72
	ds_bpermute_b32 v85, v81, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s73
	ds_bpermute_b32 v85, v82, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s74
	ds_bpermute_b32 v85, v83, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s75
	ds_bpermute_b32 v85, v84, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s76
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	ds_bpermute_b32 v85, v80, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s72
	ds_bpermute_b32 v85, v81, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s73
	ds_bpermute_b32 v85, v82, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s74
	ds_bpermute_b32 v85, v83, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s75
	ds_bpermute_b32 v85, v84, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s76
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v20, v20, v21
	v_add_f32_e32 v22, v22, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v22
	ds_bpermute_b32 v85, v80, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s72
	ds_bpermute_b32 v85, v81, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s73
	ds_bpermute_b32 v85, v82, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s74
	ds_bpermute_b32 v85, v83, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s75
	ds_bpermute_b32 v85, v84, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s76
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v24, v24, v25
	v_add_f32_e32 v26, v26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v26
	ds_bpermute_b32 v85, v80, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s72
	ds_bpermute_b32 v85, v81, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s73
	ds_bpermute_b32 v85, v82, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s74
	ds_bpermute_b32 v85, v83, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s75
	ds_bpermute_b32 v85, v84, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s76
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v28, v28, v29
	v_add_f32_e32 v30, v30, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v30
	ds_bpermute_b32 v85, v80, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s72
	ds_bpermute_b32 v85, v81, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s73
	ds_bpermute_b32 v85, v82, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s74
	ds_bpermute_b32 v85, v83, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s75
	ds_bpermute_b32 v85, v84, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s76
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v32, v32, v33
	v_add_f32_e32 v34, v34, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v34
	ds_bpermute_b32 v85, v80, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s72
	ds_bpermute_b32 v85, v81, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s73
	ds_bpermute_b32 v85, v82, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s74
	ds_bpermute_b32 v85, v83, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s75
	ds_bpermute_b32 v85, v84, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s76
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_red_done
	v_add_f32_e32 v36, v36, v37
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v38
	ds_bpermute_b32 v85, v80, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s72
	ds_bpermute_b32 v85, v81, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s73
	ds_bpermute_b32 v85, v82, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s74
	ds_bpermute_b32 v85, v83, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s75
	ds_bpermute_b32 v85, v84, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s76
	.Lcn_mq3i4_r256_gate_up_red_done:
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s41, s58, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v86, s41
	global_store_b32 v86, v8, s[52:53]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_st_done
	s_mul_i32 s41, s59, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v87, s41
	global_store_b32 v87, v12, s[52:53]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_st_done
	s_mul_i32 s41, s60, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v88, s41
	global_store_b32 v88, v16, s[52:53]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_st_done
	s_mul_i32 s41, s61, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v89, s41
	global_store_b32 v89, v20, s[52:53]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_st_done
	s_mul_i32 s41, s62, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v90, s41
	global_store_b32 v90, v24, s[52:53]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_st_done
	s_mul_i32 s41, s63, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v91, s41
	global_store_b32 v91, v28, s[52:53]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_st_done
	s_mul_i32 s41, s64, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v92, s41
	global_store_b32 v92, v32, s[52:53]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_st_done
	s_mul_i32 s41, s65, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v93, s41
	global_store_b32 v93, v36, s[52:53]
	.Lcn_mq3i4_r256_gate_up_st_done:
	.Lcn_mq3i4_r256_gate_up_exit:
	s_endpgm
.Lmq_i4_moe_gemv_cn_mq3i4_r256_gate_up_end:
.size mq_i4_moe_gemv_cn_mq3i4_r256_gate_up, .Lmq_i4_moe_gemv_cn_mq3i4_r256_gate_up_end-mq_i4_moe_gemv_cn_mq3i4_r256_gate_up
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_cn_mq3i4_r256_gate_up
	.amdhsa_group_segment_fixed_size 40960
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
	.amdhsa_next_free_vgpr 124
	.amdhsa_next_free_sgpr 101
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_cn_mq3i4_r256_gate_up_end-mq_i4_moe_gemv_cn_mq3i4_r256_gate_up)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_cn_mq3i4_r128_down
.globl mq_i4_moe_gemv_cn_mq3i4_r128_down
.p2align 8
.type mq_i4_moe_gemv_cn_mq3i4_r128_down,@function
mq_i4_moe_gemv_cn_mq3i4_r128_down:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s100, s[0:1], 0x58
	v_lshrrev_b32_e32 v32, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v32
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 4, v0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_u32_u24_e32 v32, 12, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v5, 5, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 2, v5
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v6, 31, v32
	s_mul_i32 s41, s40, 0xa00
	v_add_nc_u32_e32 v1, s41, v2
	s_lshl_b32 s30, s2, 3
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s3, s24
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_exit
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_exit
	s_cmp_lg_u32 s23, 0x280
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_exit
	s_lshl_b32 s41, s3, 2
	s_add_u32 s36, s12, s41
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s41
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_add_i32 s30, s30, s40
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_exit
	s_mul_i32 s41, s25, 10
	s_cmp_gt_u32 s33, s41
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_exit
	s_cmp_ge_u32 s34, s100
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_exit
	s_lshl_b32 s41, s34, 3
	s_add_u32 s36, s8, s41
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	s_add_i32 s43, s35, -1
	s_min_u32 s41, s43, 0
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s58, s[36:37], null
	s_min_u32 s41, s43, 1
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s59, s[36:37], null
	s_min_u32 s41, s43, 2
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s60, s[36:37], null
	s_min_u32 s41, s43, 3
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s61, s[36:37], null
	s_min_u32 s41, s43, 4
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s62, s[36:37], null
	s_min_u32 s41, s43, 5
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s63, s[36:37], null
	s_min_u32 s41, s43, 6
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s64, s[36:37], null
	s_min_u32 s41, s43, 7
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s65, s[36:37], null
	s_min_u32 s41, s43, s40
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s57, s[36:37], null
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v32, s57
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_3)
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
	v_cmp_gt_i32_e64 s80, 0, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, -1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s80
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, s26, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v35, v35, v36, s80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s80, s26, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, 1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s80
	v_cmp_le_i32_e64 s78, 0, v32
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v34, 1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v33, 0x7fffff00, v34, s78
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s66, v33
	s_lshl_b32 s55, s28, 2
	s_mul_i32 s67, s66, s55
	v_add_nc_u32_e32 v2, s67, v2
	s_mov_b32 s52, s18
	s_mov_b32 s53, s19
	s_lshl_b32 s54, s30, 2
	s_lshl_b32 s56, s29, 2
	s_cmp_lt_u32 s40, s35
	s_cselect_b32 s68, -1, 0
	s_cmp_lt_u32 s66, s27
	s_cselect_b32 s77, -1, 0
	s_and_b32 s77, s77, s68
	s_xor_b32 s82, s68, s77
	.Lcn_mq3i4_r128_down_body:
	s_mul_i32 s41, s30, 0xf0
	s_add_u32 s48, s38, s41
	s_addc_u32 s49, s39, 0
	s_mul_i32 s41, s22, 0xf0
	s_mul_i32 s42, s30, 10
	s_add_i32 s41, s41, s42
	s_add_u32 s50, s38, s41
	s_addc_u32 s51, s39, 0
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_z0
	s_mov_b32 exec_lo, s82
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
	s_mov_b32 exec_lo, -1
	.Lcn_mq3i4_r128_down_z0:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[32:35], v2, s[16:17]
	global_load_b128 v[36:39], v2, s[16:17] offset:512
	global_load_b128 v[40:43], v2, s[16:17] offset:1024
	global_load_b128 v[44:47], v2, s[16:17] offset:1536
	global_load_b128 v[48:51], v2, s[16:17] offset:2048
	s_mov_b32 exec_lo, -1
	global_load_b64 v[16:17], v5, s[48:49]
	global_load_u16 v26, v3, s[50:51]
	global_load_b64 v[18:19], v5, s[48:49] offset:48
	global_load_u16 v27, v3, s[50:51] offset:2
	global_load_b64 v[20:21], v5, s[48:49] offset:96
	global_load_u16 v28, v3, s[50:51] offset:4
	global_load_b64 v[22:23], v5, s[48:49] offset:144
	global_load_u16 v29, v3, s[50:51] offset:6
	global_load_b64 v[24:25], v5, s[48:49] offset:192
	global_load_u16 v30, v3, s[50:51] offset:8
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	s_waitcnt vmcnt(10)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[32:35]
	ds_store_b128 v1, v[36:39] offset:512
	ds_store_b128 v1, v[40:43] offset:1024
	ds_store_b128 v1, v[44:47] offset:1536
	ds_store_b128 v1, v[48:51] offset:2048
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq3i4_r128_down_c0:
	ds_load_b128 v[44:47], v4
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_lw
	ds_load_b128 v[48:51], v4 offset:2560
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_lw
	ds_load_b128 v[52:55], v4 offset:5120
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_lw
	ds_load_b128 v[56:59], v4 offset:7680
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_lw
	ds_load_b128 v[60:63], v4 offset:10240
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_lw
	ds_load_b128 v[64:67], v4 offset:12800
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_lw
	ds_load_b128 v[68:71], v4 offset:15360
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_lw
	ds_load_b128 v[72:75], v4 offset:17920
	.Lcn_mq3i4_r128_down_g0_b0_lw:
	s_waitcnt vmcnt(8)
	v_cvt_f32_f16_e64 v36, v26.l
	v_lshrrev_b64 v[38:39], v6, v[16:17]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g0_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_g0_b0_ld:
	ds_load_b128 v[44:47], v4 offset:512
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_lw
	ds_load_b128 v[48:51], v4 offset:3072
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_lw
	ds_load_b128 v[52:55], v4 offset:5632
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_lw
	ds_load_b128 v[56:59], v4 offset:8192
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_lw
	ds_load_b128 v[60:63], v4 offset:10752
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_lw
	ds_load_b128 v[64:67], v4 offset:13312
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_lw
	ds_load_b128 v[68:71], v4 offset:15872
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_lw
	ds_load_b128 v[72:75], v4 offset:18432
	.Lcn_mq3i4_r128_down_g1_b0_lw:
	s_waitcnt vmcnt(6)
	v_cvt_f32_f16_e64 v36, v27.l
	v_lshrrev_b64 v[38:39], v6, v[18:19]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g1_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_g1_b0_ld:
	ds_load_b128 v[44:47], v4 offset:1024
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_lw
	ds_load_b128 v[48:51], v4 offset:3584
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_lw
	ds_load_b128 v[52:55], v4 offset:6144
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_lw
	ds_load_b128 v[56:59], v4 offset:8704
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_lw
	ds_load_b128 v[60:63], v4 offset:11264
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_lw
	ds_load_b128 v[64:67], v4 offset:13824
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_lw
	ds_load_b128 v[68:71], v4 offset:16384
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_lw
	ds_load_b128 v[72:75], v4 offset:18944
	.Lcn_mq3i4_r128_down_g2_b0_lw:
	s_waitcnt vmcnt(4)
	v_cvt_f32_f16_e64 v36, v28.l
	v_lshrrev_b64 v[38:39], v6, v[20:21]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g2_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_g2_b0_ld:
	ds_load_b128 v[44:47], v4 offset:1536
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_lw
	ds_load_b128 v[48:51], v4 offset:4096
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_lw
	ds_load_b128 v[52:55], v4 offset:6656
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_lw
	ds_load_b128 v[56:59], v4 offset:9216
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_lw
	ds_load_b128 v[60:63], v4 offset:11776
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_lw
	ds_load_b128 v[64:67], v4 offset:14336
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_lw
	ds_load_b128 v[68:71], v4 offset:16896
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_lw
	ds_load_b128 v[72:75], v4 offset:19456
	.Lcn_mq3i4_r128_down_g3_b0_lw:
	s_waitcnt vmcnt(2)
	v_cvt_f32_f16_e64 v36, v29.l
	v_lshrrev_b64 v[38:39], v6, v[22:23]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g3_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_g3_b0_ld:
	ds_load_b128 v[44:47], v4 offset:2048
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_lw
	ds_load_b128 v[48:51], v4 offset:4608
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_lw
	ds_load_b128 v[52:55], v4 offset:7168
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_lw
	ds_load_b128 v[56:59], v4 offset:9728
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_lw
	ds_load_b128 v[60:63], v4 offset:12288
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_lw
	ds_load_b128 v[64:67], v4 offset:14848
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_lw
	ds_load_b128 v[68:71], v4 offset:17408
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_lw
	ds_load_b128 v[72:75], v4 offset:19968
	.Lcn_mq3i4_r128_down_g4_b0_lw:
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e64 v36, v30.l
	v_lshrrev_b64 v[38:39], v6, v[24:25]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_g4_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_g4_b0_ld:
	.Lcn_mq3i4_r128_down_epi:
	v_cmp_gt_u32_e64 s72, 16, v0
	v_cndmask_b32_e64 v32, 0, 16, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v32, v32, v0, 2
	v_cmp_gt_u32_e64 s73, 24, v0
	v_cndmask_b32_e64 v33, 0, 8, s73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v33, v33, v0, 2
	v_cmp_gt_u32_e64 s74, 28, v0
	v_cndmask_b32_e64 v34, 0, 4, s74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v34, v34, v0, 2
	v_cmp_gt_u32_e64 s75, 30, v0
	v_cndmask_b32_e64 v35, 0, 2, s75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v35, v35, v0, 2
	v_cmp_gt_u32_e64 s76, 31, v0
	v_cndmask_b32_e64 v36, 0, 1, s76
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v36, v36, v0, 2
	ds_bpermute_b32 v37, v32, v7
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s72
	ds_bpermute_b32 v37, v33, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s73
	ds_bpermute_b32 v37, v34, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s74
	ds_bpermute_b32 v37, v35, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s75
	ds_bpermute_b32 v37, v36, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s76
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_red_done
	ds_bpermute_b32 v37, v32, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s72
	ds_bpermute_b32 v37, v33, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s73
	ds_bpermute_b32 v37, v34, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s74
	ds_bpermute_b32 v37, v35, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s75
	ds_bpermute_b32 v37, v36, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s76
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_red_done
	ds_bpermute_b32 v37, v32, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s72
	ds_bpermute_b32 v37, v33, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s73
	ds_bpermute_b32 v37, v34, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s74
	ds_bpermute_b32 v37, v35, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s75
	ds_bpermute_b32 v37, v36, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s76
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_red_done
	ds_bpermute_b32 v37, v32, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s72
	ds_bpermute_b32 v37, v33, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s73
	ds_bpermute_b32 v37, v34, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s74
	ds_bpermute_b32 v37, v35, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s75
	ds_bpermute_b32 v37, v36, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s76
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_red_done
	ds_bpermute_b32 v37, v32, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s72
	ds_bpermute_b32 v37, v33, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s73
	ds_bpermute_b32 v37, v34, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s74
	ds_bpermute_b32 v37, v35, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s75
	ds_bpermute_b32 v37, v36, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s76
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_red_done
	ds_bpermute_b32 v37, v32, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s72
	ds_bpermute_b32 v37, v33, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s73
	ds_bpermute_b32 v37, v34, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s74
	ds_bpermute_b32 v37, v35, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s75
	ds_bpermute_b32 v37, v36, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s76
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_red_done
	ds_bpermute_b32 v37, v32, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s72
	ds_bpermute_b32 v37, v33, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s73
	ds_bpermute_b32 v37, v34, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s74
	ds_bpermute_b32 v37, v35, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s75
	ds_bpermute_b32 v37, v36, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s76
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_red_done
	ds_bpermute_b32 v37, v32, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s72
	ds_bpermute_b32 v37, v33, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s73
	ds_bpermute_b32 v37, v34, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s74
	ds_bpermute_b32 v37, v35, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s75
	ds_bpermute_b32 v37, v36, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s76
	.Lcn_mq3i4_r128_down_red_done:
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s41, s58, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v38, s41
	global_store_b32 v38, v7, s[52:53]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_st_done
	s_mul_i32 s41, s59, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v39, s41
	global_store_b32 v39, v8, s[52:53]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_st_done
	s_mul_i32 s41, s60, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v40, s41
	global_store_b32 v40, v9, s[52:53]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_st_done
	s_mul_i32 s41, s61, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v41, s41
	global_store_b32 v41, v10, s[52:53]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_st_done
	s_mul_i32 s41, s62, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v42, s41
	global_store_b32 v42, v11, s[52:53]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_st_done
	s_mul_i32 s41, s63, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v43, s41
	global_store_b32 v43, v12, s[52:53]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_st_done
	s_mul_i32 s41, s64, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v44, s41
	global_store_b32 v44, v13, s[52:53]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_st_done
	s_mul_i32 s41, s65, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v45, s41
	global_store_b32 v45, v14, s[52:53]
	.Lcn_mq3i4_r128_down_st_done:
	.Lcn_mq3i4_r128_down_exit:
	s_endpgm
.Lmq_i4_moe_gemv_cn_mq3i4_r128_down_end:
.size mq_i4_moe_gemv_cn_mq3i4_r128_down, .Lmq_i4_moe_gemv_cn_mq3i4_r128_down_end-mq_i4_moe_gemv_cn_mq3i4_r128_down
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_cn_mq3i4_r128_down
	.amdhsa_group_segment_fixed_size 20480
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
	.amdhsa_next_free_vgpr 76
	.amdhsa_next_free_sgpr 101
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_cn_mq3i4_r128_down_end-mq_i4_moe_gemv_cn_mq3i4_r128_down)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_cn_mq4sym_r256_gate_up
.globl mq_i4_moe_gemv_cn_mq4sym_r256_gate_up
.p2align 8
.type mq_i4_moe_gemv_cn_mq4sym_r256_gate_up,@function
mq_i4_moe_gemv_cn_mq4sym_r256_gate_up:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s100, s[0:1], 0x58
	v_lshrrev_b32_e32 v60, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v60
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v5, 5, v0
	v_mov_b32_e32 v4, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v6, 2, v0
	v_lshrrev_b32_e32 v7, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 2, v7
	s_mul_i32 s41, s40, 0x1400
	v_add_nc_u32_e32 v1, s41, v2
	s_lshl_b32 s30, s2, 3
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s3, s24
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_exit
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_exit
	s_cmp_lg_u32 s23, 0xa00
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_exit
	s_lshl_b32 s41, s3, 2
	s_add_u32 s36, s12, s41
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s41
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_add_i32 s30, s30, s40
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_exit
	s_mul_i32 s41, s25, 10
	s_cmp_gt_u32 s33, s41
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_exit
	s_cmp_ge_u32 s34, s100
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_exit
	s_lshl_b32 s41, s34, 3
	s_add_u32 s36, s8, s41
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	s_add_i32 s43, s35, -1
	s_min_u32 s41, s43, 0
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s58, s[36:37], null
	s_min_u32 s41, s43, 1
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s59, s[36:37], null
	s_min_u32 s41, s43, 2
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s60, s[36:37], null
	s_min_u32 s41, s43, 3
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s61, s[36:37], null
	s_min_u32 s41, s43, 4
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s62, s[36:37], null
	s_min_u32 s41, s43, 5
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s63, s[36:37], null
	s_min_u32 s41, s43, 6
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s64, s[36:37], null
	s_min_u32 s41, s43, 7
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s65, s[36:37], null
	s_min_u32 s41, s43, s40
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s57, s[36:37], null
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v60, s57
	v_cvt_f32_u32_e32 v65, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v65, v65
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_u32_e32 v62, v60
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v62, v62, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v62, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v63, v62, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v63, v60, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s80, 0, v63
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v64, -1, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v64, s80
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v64, s26, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v64, s80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s80, s26, v63
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v64, 1, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v64, s80
	v_cmp_le_i32_e64 s78, 0, v60
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v62, 1, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, 0x7fffff00, v62, s78
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s66, v61
	s_lshl_b32 s55, s28, 2
	s_mul_i32 s67, s66, s55
	v_add_nc_u32_e32 v2, s67, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, 0x1000, v2
	s_lshr_b32 s69, s22, 1
	s_sub_i32 s70, s30, s69
	s_cmp_lt_u32 s30, s69
	s_cselect_b64 s[52:53], s[18:19], s[20:21]
	s_cselect_b32 s71, s30, s70
	s_lshl_b32 s54, s71, 2
	s_lshl_b32 s56, s29, 2
	s_cmp_lt_u32 s40, s35
	s_cselect_b32 s68, -1, 0
	s_cmp_lt_u32 s66, s27
	s_cselect_b32 s77, -1, 0
	s_and_b32 s77, s77, s68
	s_xor_b32 s82, s68, s77
	.Lcn_mq4sym_r256_gate_up_body:
	s_mul_i32 s41, s30, 0x550
	s_add_u32 s48, s38, s41
	s_addc_u32 s49, s39, 0
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_z0
	s_mov_b32 exec_lo, s82
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
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	s_mov_b32 exec_lo, -1
	.Lcn_mq4sym_r256_gate_up_z0:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[60:63], v2, s[16:17]
	global_load_b128 v[64:67], v2, s[16:17] offset:512
	global_load_b128 v[68:71], v2, s[16:17] offset:1024
	global_load_b128 v[72:75], v2, s[16:17] offset:1536
	global_load_b128 v[76:79], v2, s[16:17] offset:2048
	global_load_b128 v[80:83], v2, s[16:17] offset:2560
	global_load_b128 v[84:87], v2, s[16:17] offset:3072
	global_load_b128 v[88:91], v2, s[16:17] offset:3584
	global_load_b128 v[92:95], v3, s[16:17]
	global_load_b128 v[96:99], v3, s[16:17] offset:512
	s_mov_b32 exec_lo, -1
	global_load_b32 v40, v7, s[48:49]
	global_load_b32 v50, v6, s[48:49] offset:8
	global_load_b32 v41, v7, s[48:49] offset:136
	global_load_b32 v51, v6, s[48:49] offset:144
	global_load_b32 v42, v7, s[48:49] offset:272
	global_load_b32 v52, v6, s[48:49] offset:280
	global_load_b32 v43, v7, s[48:49] offset:408
	global_load_b32 v53, v6, s[48:49] offset:416
	global_load_b32 v44, v7, s[48:49] offset:544
	global_load_b32 v54, v6, s[48:49] offset:552
	global_load_b32 v45, v7, s[48:49] offset:680
	global_load_b32 v55, v6, s[48:49] offset:688
	global_load_b32 v46, v7, s[48:49] offset:816
	global_load_b32 v56, v6, s[48:49] offset:824
	global_load_b32 v47, v7, s[48:49] offset:952
	global_load_b32 v57, v6, s[48:49] offset:960
	global_load_b32 v48, v7, s[48:49] offset:1088
	global_load_b32 v58, v6, s[48:49] offset:1096
	global_load_b32 v49, v7, s[48:49] offset:1224
	global_load_b32 v59, v6, s[48:49] offset:1232
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	s_waitcnt vmcnt(20)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[60:63]
	ds_store_b128 v1, v[64:67] offset:512
	ds_store_b128 v1, v[68:71] offset:1024
	ds_store_b128 v1, v[72:75] offset:1536
	ds_store_b128 v1, v[76:79] offset:2048
	ds_store_b128 v1, v[80:83] offset:2560
	ds_store_b128 v1, v[84:87] offset:3072
	ds_store_b128 v1, v[88:91] offset:3584
	ds_store_b128 v1, v[92:95] offset:4096
	ds_store_b128 v1, v[96:99] offset:4608
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq4sym_r256_gate_up_c0:
	ds_load_b128 v[72:75], v5
	ds_load_b128 v[76:79], v5 offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b0_lw
	ds_load_b128 v[80:83], v5 offset:5120
	ds_load_b128 v[84:87], v5 offset:5136
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b0_lw
	ds_load_b128 v[88:91], v5 offset:10240
	ds_load_b128 v[92:95], v5 offset:10256
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b0_lw
	ds_load_b128 v[96:99], v5 offset:15360
	ds_load_b128 v[100:103], v5 offset:15376
	.Lcn_mq4sym_r256_gate_up_g0_b0_lw:
	s_waitcnt vmcnt(19)
	v_cvt_f32_f16_e64 v68, v40.l
	v_lshrrev_b32_e32 v69, 16, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	s_waitcnt vmcnt(18)
	v_bfe_u32 v60, v50, 0, 4
	v_bfe_u32 v61, v50, 4, 4
	v_bfe_u32 v62, v50, 8, 4
	v_bfe_u32 v63, v50, 12, 4
	v_bfe_u32 v64, v50, 16, 4
	v_bfe_u32 v65, v50, 20, 4
	v_bfe_u32 v66, v50, 24, 4
	v_bfe_u32 v67, v50, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v71
	.Lcn_mq4sym_r256_gate_up_g0_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_done
	ds_load_b128 v[72:75], v5 offset:20480
	ds_load_b128 v[76:79], v5 offset:20496
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b1_lw
	ds_load_b128 v[80:83], v5 offset:25600
	ds_load_b128 v[84:87], v5 offset:25616
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b1_lw
	ds_load_b128 v[88:91], v5 offset:30720
	ds_load_b128 v[92:95], v5 offset:30736
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b1_lw
	ds_load_b128 v[96:99], v5 offset:35840
	ds_load_b128 v[100:103], v5 offset:35856
	.Lcn_mq4sym_r256_gate_up_g0_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g0_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v71
	.Lcn_mq4sym_r256_gate_up_g0_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g0_done:
	ds_load_b128 v[72:75], v5 offset:1024
	ds_load_b128 v[76:79], v5 offset:1040
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b0_lw
	ds_load_b128 v[80:83], v5 offset:6144
	ds_load_b128 v[84:87], v5 offset:6160
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b0_lw
	ds_load_b128 v[88:91], v5 offset:11264
	ds_load_b128 v[92:95], v5 offset:11280
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b0_lw
	ds_load_b128 v[96:99], v5 offset:16384
	ds_load_b128 v[100:103], v5 offset:16400
	.Lcn_mq4sym_r256_gate_up_g1_b0_lw:
	s_waitcnt vmcnt(17)
	v_cvt_f32_f16_e64 v68, v41.l
	v_lshrrev_b32_e32 v69, 16, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	s_waitcnt vmcnt(16)
	v_bfe_u32 v60, v51, 0, 4
	v_bfe_u32 v61, v51, 4, 4
	v_bfe_u32 v62, v51, 8, 4
	v_bfe_u32 v63, v51, 12, 4
	v_bfe_u32 v64, v51, 16, 4
	v_bfe_u32 v65, v51, 20, 4
	v_bfe_u32 v66, v51, 24, 4
	v_bfe_u32 v67, v51, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v71
	.Lcn_mq4sym_r256_gate_up_g1_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_done
	ds_load_b128 v[72:75], v5 offset:21504
	ds_load_b128 v[76:79], v5 offset:21520
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b1_lw
	ds_load_b128 v[80:83], v5 offset:26624
	ds_load_b128 v[84:87], v5 offset:26640
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b1_lw
	ds_load_b128 v[88:91], v5 offset:31744
	ds_load_b128 v[92:95], v5 offset:31760
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b1_lw
	ds_load_b128 v[96:99], v5 offset:36864
	ds_load_b128 v[100:103], v5 offset:36880
	.Lcn_mq4sym_r256_gate_up_g1_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g1_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v71
	.Lcn_mq4sym_r256_gate_up_g1_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g1_done:
	ds_load_b128 v[72:75], v5 offset:2048
	ds_load_b128 v[76:79], v5 offset:2064
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b0_lw
	ds_load_b128 v[80:83], v5 offset:7168
	ds_load_b128 v[84:87], v5 offset:7184
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b0_lw
	ds_load_b128 v[88:91], v5 offset:12288
	ds_load_b128 v[92:95], v5 offset:12304
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b0_lw
	ds_load_b128 v[96:99], v5 offset:17408
	ds_load_b128 v[100:103], v5 offset:17424
	.Lcn_mq4sym_r256_gate_up_g2_b0_lw:
	s_waitcnt vmcnt(15)
	v_cvt_f32_f16_e64 v68, v42.l
	v_lshrrev_b32_e32 v69, 16, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	s_waitcnt vmcnt(14)
	v_bfe_u32 v60, v52, 0, 4
	v_bfe_u32 v61, v52, 4, 4
	v_bfe_u32 v62, v52, 8, 4
	v_bfe_u32 v63, v52, 12, 4
	v_bfe_u32 v64, v52, 16, 4
	v_bfe_u32 v65, v52, 20, 4
	v_bfe_u32 v66, v52, 24, 4
	v_bfe_u32 v67, v52, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v71
	.Lcn_mq4sym_r256_gate_up_g2_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_done
	ds_load_b128 v[72:75], v5 offset:22528
	ds_load_b128 v[76:79], v5 offset:22544
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b1_lw
	ds_load_b128 v[80:83], v5 offset:27648
	ds_load_b128 v[84:87], v5 offset:27664
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b1_lw
	ds_load_b128 v[88:91], v5 offset:32768
	ds_load_b128 v[92:95], v5 offset:32784
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b1_lw
	ds_load_b128 v[96:99], v5 offset:37888
	ds_load_b128 v[100:103], v5 offset:37904
	.Lcn_mq4sym_r256_gate_up_g2_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g2_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v71
	.Lcn_mq4sym_r256_gate_up_g2_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g2_done:
	ds_load_b128 v[72:75], v5 offset:3072
	ds_load_b128 v[76:79], v5 offset:3088
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b0_lw
	ds_load_b128 v[80:83], v5 offset:8192
	ds_load_b128 v[84:87], v5 offset:8208
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b0_lw
	ds_load_b128 v[88:91], v5 offset:13312
	ds_load_b128 v[92:95], v5 offset:13328
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b0_lw
	ds_load_b128 v[96:99], v5 offset:18432
	ds_load_b128 v[100:103], v5 offset:18448
	.Lcn_mq4sym_r256_gate_up_g3_b0_lw:
	s_waitcnt vmcnt(13)
	v_cvt_f32_f16_e64 v68, v43.l
	v_lshrrev_b32_e32 v69, 16, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	s_waitcnt vmcnt(12)
	v_bfe_u32 v60, v53, 0, 4
	v_bfe_u32 v61, v53, 4, 4
	v_bfe_u32 v62, v53, 8, 4
	v_bfe_u32 v63, v53, 12, 4
	v_bfe_u32 v64, v53, 16, 4
	v_bfe_u32 v65, v53, 20, 4
	v_bfe_u32 v66, v53, 24, 4
	v_bfe_u32 v67, v53, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v71
	.Lcn_mq4sym_r256_gate_up_g3_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_done
	ds_load_b128 v[72:75], v5 offset:23552
	ds_load_b128 v[76:79], v5 offset:23568
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b1_lw
	ds_load_b128 v[80:83], v5 offset:28672
	ds_load_b128 v[84:87], v5 offset:28688
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b1_lw
	ds_load_b128 v[88:91], v5 offset:33792
	ds_load_b128 v[92:95], v5 offset:33808
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b1_lw
	ds_load_b128 v[96:99], v5 offset:38912
	ds_load_b128 v[100:103], v5 offset:38928
	.Lcn_mq4sym_r256_gate_up_g3_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g3_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v71
	.Lcn_mq4sym_r256_gate_up_g3_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g3_done:
	ds_load_b128 v[72:75], v5 offset:4096
	ds_load_b128 v[76:79], v5 offset:4112
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b0_lw
	ds_load_b128 v[80:83], v5 offset:9216
	ds_load_b128 v[84:87], v5 offset:9232
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b0_lw
	ds_load_b128 v[88:91], v5 offset:14336
	ds_load_b128 v[92:95], v5 offset:14352
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b0_lw
	ds_load_b128 v[96:99], v5 offset:19456
	ds_load_b128 v[100:103], v5 offset:19472
	.Lcn_mq4sym_r256_gate_up_g4_b0_lw:
	s_waitcnt vmcnt(11)
	v_cvt_f32_f16_e64 v68, v44.l
	v_lshrrev_b32_e32 v69, 16, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	s_waitcnt vmcnt(10)
	v_bfe_u32 v60, v54, 0, 4
	v_bfe_u32 v61, v54, 4, 4
	v_bfe_u32 v62, v54, 8, 4
	v_bfe_u32 v63, v54, 12, 4
	v_bfe_u32 v64, v54, 16, 4
	v_bfe_u32 v65, v54, 20, 4
	v_bfe_u32 v66, v54, 24, 4
	v_bfe_u32 v67, v54, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v71
	.Lcn_mq4sym_r256_gate_up_g4_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_done
	ds_load_b128 v[72:75], v5 offset:24576
	ds_load_b128 v[76:79], v5 offset:24592
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b1_lw
	ds_load_b128 v[80:83], v5 offset:29696
	ds_load_b128 v[84:87], v5 offset:29712
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b1_lw
	ds_load_b128 v[88:91], v5 offset:34816
	ds_load_b128 v[92:95], v5 offset:34832
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b1_lw
	ds_load_b128 v[96:99], v5 offset:39936
	ds_load_b128 v[100:103], v5 offset:39952
	.Lcn_mq4sym_r256_gate_up_g4_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g4_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v71
	.Lcn_mq4sym_r256_gate_up_g4_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g4_done:
	.Lcn_mq4sym_r256_gate_up_s1:
	v_add_nc_u32_e32 v2, 0x1400, v2
	v_add_nc_u32_e32 v3, 0x1400, v3
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_z1
	s_mov_b32 exec_lo, s82
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
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	s_mov_b32 exec_lo, -1
	.Lcn_mq4sym_r256_gate_up_z1:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[60:63], v2, s[16:17]
	global_load_b128 v[64:67], v2, s[16:17] offset:512
	global_load_b128 v[68:71], v2, s[16:17] offset:1024
	global_load_b128 v[72:75], v2, s[16:17] offset:1536
	global_load_b128 v[76:79], v2, s[16:17] offset:2048
	global_load_b128 v[80:83], v2, s[16:17] offset:2560
	global_load_b128 v[84:87], v2, s[16:17] offset:3072
	global_load_b128 v[88:91], v2, s[16:17] offset:3584
	global_load_b128 v[92:95], v3, s[16:17]
	global_load_b128 v[96:99], v3, s[16:17] offset:512
	s_mov_b32 exec_lo, -1
	s_barrier
	s_waitcnt vmcnt(0)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[60:63]
	ds_store_b128 v1, v[64:67] offset:512
	ds_store_b128 v1, v[68:71] offset:1024
	ds_store_b128 v1, v[72:75] offset:1536
	ds_store_b128 v1, v[76:79] offset:2048
	ds_store_b128 v1, v[80:83] offset:2560
	ds_store_b128 v1, v[84:87] offset:3072
	ds_store_b128 v1, v[88:91] offset:3584
	ds_store_b128 v1, v[92:95] offset:4096
	ds_store_b128 v1, v[96:99] offset:4608
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq4sym_r256_gate_up_c1:
	ds_load_b128 v[72:75], v5
	ds_load_b128 v[76:79], v5 offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b0_lw
	ds_load_b128 v[80:83], v5 offset:5120
	ds_load_b128 v[84:87], v5 offset:5136
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b0_lw
	ds_load_b128 v[88:91], v5 offset:10240
	ds_load_b128 v[92:95], v5 offset:10256
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b0_lw
	ds_load_b128 v[96:99], v5 offset:15360
	ds_load_b128 v[100:103], v5 offset:15376
	.Lcn_mq4sym_r256_gate_up_g5_b0_lw:
	v_cvt_f32_f16_e64 v68, v45.l
	v_lshrrev_b32_e32 v69, 16, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v55, 0, 4
	v_bfe_u32 v61, v55, 4, 4
	v_bfe_u32 v62, v55, 8, 4
	v_bfe_u32 v63, v55, 12, 4
	v_bfe_u32 v64, v55, 16, 4
	v_bfe_u32 v65, v55, 20, 4
	v_bfe_u32 v66, v55, 24, 4
	v_bfe_u32 v67, v55, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v71
	.Lcn_mq4sym_r256_gate_up_g5_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_done
	ds_load_b128 v[72:75], v5 offset:20480
	ds_load_b128 v[76:79], v5 offset:20496
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b1_lw
	ds_load_b128 v[80:83], v5 offset:25600
	ds_load_b128 v[84:87], v5 offset:25616
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b1_lw
	ds_load_b128 v[88:91], v5 offset:30720
	ds_load_b128 v[92:95], v5 offset:30736
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b1_lw
	ds_load_b128 v[96:99], v5 offset:35840
	ds_load_b128 v[100:103], v5 offset:35856
	.Lcn_mq4sym_r256_gate_up_g5_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g5_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v71
	.Lcn_mq4sym_r256_gate_up_g5_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g5_done:
	ds_load_b128 v[72:75], v5 offset:1024
	ds_load_b128 v[76:79], v5 offset:1040
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b0_lw
	ds_load_b128 v[80:83], v5 offset:6144
	ds_load_b128 v[84:87], v5 offset:6160
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b0_lw
	ds_load_b128 v[88:91], v5 offset:11264
	ds_load_b128 v[92:95], v5 offset:11280
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b0_lw
	ds_load_b128 v[96:99], v5 offset:16384
	ds_load_b128 v[100:103], v5 offset:16400
	.Lcn_mq4sym_r256_gate_up_g6_b0_lw:
	v_cvt_f32_f16_e64 v68, v46.l
	v_lshrrev_b32_e32 v69, 16, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v56, 0, 4
	v_bfe_u32 v61, v56, 4, 4
	v_bfe_u32 v62, v56, 8, 4
	v_bfe_u32 v63, v56, 12, 4
	v_bfe_u32 v64, v56, 16, 4
	v_bfe_u32 v65, v56, 20, 4
	v_bfe_u32 v66, v56, 24, 4
	v_bfe_u32 v67, v56, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v71
	.Lcn_mq4sym_r256_gate_up_g6_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_done
	ds_load_b128 v[72:75], v5 offset:21504
	ds_load_b128 v[76:79], v5 offset:21520
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b1_lw
	ds_load_b128 v[80:83], v5 offset:26624
	ds_load_b128 v[84:87], v5 offset:26640
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b1_lw
	ds_load_b128 v[88:91], v5 offset:31744
	ds_load_b128 v[92:95], v5 offset:31760
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b1_lw
	ds_load_b128 v[96:99], v5 offset:36864
	ds_load_b128 v[100:103], v5 offset:36880
	.Lcn_mq4sym_r256_gate_up_g6_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g6_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v71
	.Lcn_mq4sym_r256_gate_up_g6_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g6_done:
	ds_load_b128 v[72:75], v5 offset:2048
	ds_load_b128 v[76:79], v5 offset:2064
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b0_lw
	ds_load_b128 v[80:83], v5 offset:7168
	ds_load_b128 v[84:87], v5 offset:7184
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b0_lw
	ds_load_b128 v[88:91], v5 offset:12288
	ds_load_b128 v[92:95], v5 offset:12304
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b0_lw
	ds_load_b128 v[96:99], v5 offset:17408
	ds_load_b128 v[100:103], v5 offset:17424
	.Lcn_mq4sym_r256_gate_up_g7_b0_lw:
	v_cvt_f32_f16_e64 v68, v47.l
	v_lshrrev_b32_e32 v69, 16, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v57, 0, 4
	v_bfe_u32 v61, v57, 4, 4
	v_bfe_u32 v62, v57, 8, 4
	v_bfe_u32 v63, v57, 12, 4
	v_bfe_u32 v64, v57, 16, 4
	v_bfe_u32 v65, v57, 20, 4
	v_bfe_u32 v66, v57, 24, 4
	v_bfe_u32 v67, v57, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v71
	.Lcn_mq4sym_r256_gate_up_g7_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_done
	ds_load_b128 v[72:75], v5 offset:22528
	ds_load_b128 v[76:79], v5 offset:22544
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b1_lw
	ds_load_b128 v[80:83], v5 offset:27648
	ds_load_b128 v[84:87], v5 offset:27664
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b1_lw
	ds_load_b128 v[88:91], v5 offset:32768
	ds_load_b128 v[92:95], v5 offset:32784
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b1_lw
	ds_load_b128 v[96:99], v5 offset:37888
	ds_load_b128 v[100:103], v5 offset:37904
	.Lcn_mq4sym_r256_gate_up_g7_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g7_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v71
	.Lcn_mq4sym_r256_gate_up_g7_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g7_done:
	ds_load_b128 v[72:75], v5 offset:3072
	ds_load_b128 v[76:79], v5 offset:3088
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b0_lw
	ds_load_b128 v[80:83], v5 offset:8192
	ds_load_b128 v[84:87], v5 offset:8208
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b0_lw
	ds_load_b128 v[88:91], v5 offset:13312
	ds_load_b128 v[92:95], v5 offset:13328
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b0_lw
	ds_load_b128 v[96:99], v5 offset:18432
	ds_load_b128 v[100:103], v5 offset:18448
	.Lcn_mq4sym_r256_gate_up_g8_b0_lw:
	v_cvt_f32_f16_e64 v68, v48.l
	v_lshrrev_b32_e32 v69, 16, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v58, 0, 4
	v_bfe_u32 v61, v58, 4, 4
	v_bfe_u32 v62, v58, 8, 4
	v_bfe_u32 v63, v58, 12, 4
	v_bfe_u32 v64, v58, 16, 4
	v_bfe_u32 v65, v58, 20, 4
	v_bfe_u32 v66, v58, 24, 4
	v_bfe_u32 v67, v58, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v71
	.Lcn_mq4sym_r256_gate_up_g8_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_done
	ds_load_b128 v[72:75], v5 offset:23552
	ds_load_b128 v[76:79], v5 offset:23568
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b1_lw
	ds_load_b128 v[80:83], v5 offset:28672
	ds_load_b128 v[84:87], v5 offset:28688
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b1_lw
	ds_load_b128 v[88:91], v5 offset:33792
	ds_load_b128 v[92:95], v5 offset:33808
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b1_lw
	ds_load_b128 v[96:99], v5 offset:38912
	ds_load_b128 v[100:103], v5 offset:38928
	.Lcn_mq4sym_r256_gate_up_g8_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g8_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v71
	.Lcn_mq4sym_r256_gate_up_g8_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g8_done:
	ds_load_b128 v[72:75], v5 offset:4096
	ds_load_b128 v[76:79], v5 offset:4112
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b0_lw
	ds_load_b128 v[80:83], v5 offset:9216
	ds_load_b128 v[84:87], v5 offset:9232
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b0_lw
	ds_load_b128 v[88:91], v5 offset:14336
	ds_load_b128 v[92:95], v5 offset:14352
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b0_lw
	ds_load_b128 v[96:99], v5 offset:19456
	ds_load_b128 v[100:103], v5 offset:19472
	.Lcn_mq4sym_r256_gate_up_g9_b0_lw:
	v_cvt_f32_f16_e64 v68, v49.l
	v_lshrrev_b32_e32 v69, 16, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v59, 0, 4
	v_bfe_u32 v61, v59, 4, 4
	v_bfe_u32 v62, v59, 8, 4
	v_bfe_u32 v63, v59, 12, 4
	v_bfe_u32 v64, v59, 16, 4
	v_bfe_u32 v65, v59, 20, 4
	v_bfe_u32 v66, v59, 24, 4
	v_bfe_u32 v67, v59, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v71
	.Lcn_mq4sym_r256_gate_up_g9_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_done
	ds_load_b128 v[72:75], v5 offset:24576
	ds_load_b128 v[76:79], v5 offset:24592
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b1_lw
	ds_load_b128 v[80:83], v5 offset:29696
	ds_load_b128 v[84:87], v5 offset:29712
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b1_lw
	ds_load_b128 v[88:91], v5 offset:34816
	ds_load_b128 v[92:95], v5 offset:34832
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b1_lw
	ds_load_b128 v[96:99], v5 offset:39936
	ds_load_b128 v[100:103], v5 offset:39952
	.Lcn_mq4sym_r256_gate_up_g9_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_g9_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v71
	.Lcn_mq4sym_r256_gate_up_g9_b1_ld:
	.Lcn_mq4sym_r256_gate_up_g9_done:
	.Lcn_mq4sym_r256_gate_up_epi:
	v_cmp_gt_u32_e64 s72, 16, v0
	v_cndmask_b32_e64 v60, 0, 16, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v60, v60, v0, 2
	v_cmp_gt_u32_e64 s73, 24, v0
	v_cndmask_b32_e64 v61, 0, 8, s73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v61, v61, v0, 2
	v_cmp_gt_u32_e64 s74, 28, v0
	v_cndmask_b32_e64 v62, 0, 4, s74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v62, v62, v0, 2
	v_cmp_gt_u32_e64 s75, 30, v0
	v_cndmask_b32_e64 v63, 0, 2, s75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v63, v63, v0, 2
	v_cmp_gt_u32_e64 s76, 31, v0
	v_cndmask_b32_e64 v64, 0, 1, s76
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v64, v64, v0, 2
	v_add_f32_e32 v8, v8, v9
	v_add_f32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v10
	ds_bpermute_b32 v65, v60, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s72
	ds_bpermute_b32 v65, v61, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s73
	ds_bpermute_b32 v65, v62, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s74
	ds_bpermute_b32 v65, v63, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s75
	ds_bpermute_b32 v65, v64, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s76
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v12, v12, v13
	v_add_f32_e32 v14, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v14
	ds_bpermute_b32 v65, v60, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s72
	ds_bpermute_b32 v65, v61, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s73
	ds_bpermute_b32 v65, v62, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s74
	ds_bpermute_b32 v65, v63, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s75
	ds_bpermute_b32 v65, v64, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s76
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	ds_bpermute_b32 v65, v60, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s72
	ds_bpermute_b32 v65, v61, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s73
	ds_bpermute_b32 v65, v62, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s74
	ds_bpermute_b32 v65, v63, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s75
	ds_bpermute_b32 v65, v64, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s76
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v20, v20, v21
	v_add_f32_e32 v22, v22, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v22
	ds_bpermute_b32 v65, v60, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s72
	ds_bpermute_b32 v65, v61, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s73
	ds_bpermute_b32 v65, v62, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s74
	ds_bpermute_b32 v65, v63, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s75
	ds_bpermute_b32 v65, v64, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s76
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v24, v24, v25
	v_add_f32_e32 v26, v26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v26
	ds_bpermute_b32 v65, v60, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s72
	ds_bpermute_b32 v65, v61, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s73
	ds_bpermute_b32 v65, v62, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s74
	ds_bpermute_b32 v65, v63, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s75
	ds_bpermute_b32 v65, v64, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s76
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v28, v28, v29
	v_add_f32_e32 v30, v30, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v30
	ds_bpermute_b32 v65, v60, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s72
	ds_bpermute_b32 v65, v61, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s73
	ds_bpermute_b32 v65, v62, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s74
	ds_bpermute_b32 v65, v63, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s75
	ds_bpermute_b32 v65, v64, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s76
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v32, v32, v33
	v_add_f32_e32 v34, v34, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v34
	ds_bpermute_b32 v65, v60, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s72
	ds_bpermute_b32 v65, v61, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s73
	ds_bpermute_b32 v65, v62, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s74
	ds_bpermute_b32 v65, v63, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s75
	ds_bpermute_b32 v65, v64, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s76
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_red_done
	v_add_f32_e32 v36, v36, v37
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v38
	ds_bpermute_b32 v65, v60, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s72
	ds_bpermute_b32 v65, v61, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s73
	ds_bpermute_b32 v65, v62, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s74
	ds_bpermute_b32 v65, v63, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s75
	ds_bpermute_b32 v65, v64, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s76
	.Lcn_mq4sym_r256_gate_up_red_done:
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s41, s58, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v66, s41
	global_store_b32 v66, v8, s[52:53]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_st_done
	s_mul_i32 s41, s59, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v67, s41
	global_store_b32 v67, v12, s[52:53]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_st_done
	s_mul_i32 s41, s60, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v68, s41
	global_store_b32 v68, v16, s[52:53]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_st_done
	s_mul_i32 s41, s61, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v69, s41
	global_store_b32 v69, v20, s[52:53]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_st_done
	s_mul_i32 s41, s62, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v70, s41
	global_store_b32 v70, v24, s[52:53]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_st_done
	s_mul_i32 s41, s63, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v71, s41
	global_store_b32 v71, v28, s[52:53]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_st_done
	s_mul_i32 s41, s64, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v72, s41
	global_store_b32 v72, v32, s[52:53]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_st_done
	s_mul_i32 s41, s65, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v73, s41
	global_store_b32 v73, v36, s[52:53]
	.Lcn_mq4sym_r256_gate_up_st_done:
	.Lcn_mq4sym_r256_gate_up_exit:
	s_endpgm
.Lmq_i4_moe_gemv_cn_mq4sym_r256_gate_up_end:
.size mq_i4_moe_gemv_cn_mq4sym_r256_gate_up, .Lmq_i4_moe_gemv_cn_mq4sym_r256_gate_up_end-mq_i4_moe_gemv_cn_mq4sym_r256_gate_up
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_cn_mq4sym_r256_gate_up
	.amdhsa_group_segment_fixed_size 40960
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
	.amdhsa_next_free_vgpr 104
	.amdhsa_next_free_sgpr 101
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_cn_mq4sym_r256_gate_up_end-mq_i4_moe_gemv_cn_mq4sym_r256_gate_up)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_cn_mq4sym_r128_down
.globl mq_i4_moe_gemv_cn_mq4sym_r128_down
.p2align 8
.type mq_i4_moe_gemv_cn_mq4sym_r128_down,@function
mq_i4_moe_gemv_cn_mq4sym_r128_down:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s100, s[0:1], 0x58
	v_lshrrev_b32_e32 v24, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v24
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 4, v0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v5, 1, v0
	s_mul_i32 s41, s40, 0xa00
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v1, s41, v2
	s_lshl_b32 s30, s2, 3
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s3, s24
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_exit
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_exit
	s_cmp_lg_u32 s23, 0x280
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_exit
	s_lshl_b32 s41, s3, 2
	s_add_u32 s36, s12, s41
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s41
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_add_i32 s30, s30, s40
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_exit
	s_mul_i32 s41, s25, 10
	s_cmp_gt_u32 s33, s41
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_exit
	s_cmp_ge_u32 s34, s100
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_exit
	s_lshl_b32 s41, s34, 3
	s_add_u32 s36, s8, s41
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	s_add_i32 s43, s35, -1
	s_min_u32 s41, s43, 0
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s58, s[36:37], null
	s_min_u32 s41, s43, 1
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s59, s[36:37], null
	s_min_u32 s41, s43, 2
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s60, s[36:37], null
	s_min_u32 s41, s43, 3
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s61, s[36:37], null
	s_min_u32 s41, s43, 4
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s62, s[36:37], null
	s_min_u32 s41, s43, 5
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s63, s[36:37], null
	s_min_u32 s41, s43, 6
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s64, s[36:37], null
	s_min_u32 s41, s43, 7
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s65, s[36:37], null
	s_min_u32 s41, s43, s40
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s57, s[36:37], null
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v24, s57
	v_cvt_f32_u32_e32 v29, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_u32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v26, v26, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v27, v26, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v27, v24, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s80, 0, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, -1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s80
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, s26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v27, v27, v28, s80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s80, s26, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s80
	v_cmp_le_i32_e64 s78, 0, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v25, 0x7fffff00, v26, s78
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s66, v25
	s_lshl_b32 s55, s28, 2
	s_mul_i32 s67, s66, s55
	v_add_nc_u32_e32 v2, s67, v2
	s_mov_b32 s52, s18
	s_mov_b32 s53, s19
	s_lshl_b32 s54, s30, 2
	s_lshl_b32 s56, s29, 2
	s_cmp_lt_u32 s40, s35
	s_cselect_b32 s68, -1, 0
	s_cmp_lt_u32 s66, s27
	s_cselect_b32 s77, -1, 0
	s_and_b32 s77, s77, s68
	s_xor_b32 s82, s68, s77
	.Lcn_mq4sym_r128_down_body:
	s_mul_i32 s41, s30, 0x154
	s_add_u32 s48, s38, s41
	s_addc_u32 s49, s39, 0
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_z0
	s_mov_b32 exec_lo, s82
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
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
	s_mov_b32 exec_lo, -1
	.Lcn_mq4sym_r128_down_z0:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[24:27], v2, s[16:17]
	global_load_b128 v[28:31], v2, s[16:17] offset:512
	global_load_b128 v[32:35], v2, s[16:17] offset:1024
	global_load_b128 v[36:39], v2, s[16:17] offset:1536
	global_load_b128 v[40:43], v2, s[16:17] offset:2048
	s_mov_b32 exec_lo, -1
	global_load_b32 v14, v3, s[48:49]
	global_load_u16 v19, v5, s[48:49] offset:4
	global_load_b32 v15, v3, s[48:49] offset:68
	global_load_u16 v20, v5, s[48:49] offset:72
	global_load_b32 v16, v3, s[48:49] offset:136
	global_load_u16 v21, v5, s[48:49] offset:140
	global_load_b32 v17, v3, s[48:49] offset:204
	global_load_u16 v22, v5, s[48:49] offset:208
	global_load_b32 v18, v3, s[48:49] offset:272
	global_load_u16 v23, v5, s[48:49] offset:276
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	s_waitcnt vmcnt(10)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[24:27]
	ds_store_b128 v1, v[28:31] offset:512
	ds_store_b128 v1, v[32:35] offset:1024
	ds_store_b128 v1, v[36:39] offset:1536
	ds_store_b128 v1, v[40:43] offset:2048
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq4sym_r128_down_c0:
	ds_load_b128 v[32:35], v4
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_lw
	ds_load_b128 v[36:39], v4 offset:2560
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_lw
	ds_load_b128 v[40:43], v4 offset:5120
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_lw
	ds_load_b128 v[44:47], v4 offset:7680
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_lw
	ds_load_b128 v[48:51], v4 offset:10240
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_lw
	ds_load_b128 v[52:55], v4 offset:12800
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_lw
	ds_load_b128 v[56:59], v4 offset:15360
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_lw
	ds_load_b128 v[60:63], v4 offset:17920
	.Lcn_mq4sym_r128_down_g0_b0_lw:
	s_waitcnt vmcnt(9)
	v_cvt_f32_f16_e64 v28, v14.l
	v_lshrrev_b32_e32 v29, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	s_waitcnt vmcnt(8)
	v_bfe_u32 v24, v19, 0, 4
	v_bfe_u32 v25, v19, 4, 4
	v_bfe_u32 v26, v19, 8, 4
	v_bfe_u32 v27, v19, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g0_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_g0_b0_ld:
	ds_load_b128 v[32:35], v4 offset:512
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_lw
	ds_load_b128 v[36:39], v4 offset:3072
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_lw
	ds_load_b128 v[40:43], v4 offset:5632
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_lw
	ds_load_b128 v[44:47], v4 offset:8192
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_lw
	ds_load_b128 v[48:51], v4 offset:10752
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_lw
	ds_load_b128 v[52:55], v4 offset:13312
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_lw
	ds_load_b128 v[56:59], v4 offset:15872
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_lw
	ds_load_b128 v[60:63], v4 offset:18432
	.Lcn_mq4sym_r128_down_g1_b0_lw:
	s_waitcnt vmcnt(7)
	v_cvt_f32_f16_e64 v28, v15.l
	v_lshrrev_b32_e32 v29, 16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	s_waitcnt vmcnt(6)
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g1_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_g1_b0_ld:
	ds_load_b128 v[32:35], v4 offset:1024
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_lw
	ds_load_b128 v[36:39], v4 offset:3584
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_lw
	ds_load_b128 v[40:43], v4 offset:6144
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_lw
	ds_load_b128 v[44:47], v4 offset:8704
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_lw
	ds_load_b128 v[48:51], v4 offset:11264
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_lw
	ds_load_b128 v[52:55], v4 offset:13824
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_lw
	ds_load_b128 v[56:59], v4 offset:16384
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_lw
	ds_load_b128 v[60:63], v4 offset:18944
	.Lcn_mq4sym_r128_down_g2_b0_lw:
	s_waitcnt vmcnt(5)
	v_cvt_f32_f16_e64 v28, v16.l
	v_lshrrev_b32_e32 v29, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	s_waitcnt vmcnt(4)
	v_bfe_u32 v24, v21, 0, 4
	v_bfe_u32 v25, v21, 4, 4
	v_bfe_u32 v26, v21, 8, 4
	v_bfe_u32 v27, v21, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g2_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_g2_b0_ld:
	ds_load_b128 v[32:35], v4 offset:1536
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_lw
	ds_load_b128 v[36:39], v4 offset:4096
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_lw
	ds_load_b128 v[40:43], v4 offset:6656
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_lw
	ds_load_b128 v[44:47], v4 offset:9216
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_lw
	ds_load_b128 v[48:51], v4 offset:11776
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_lw
	ds_load_b128 v[52:55], v4 offset:14336
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_lw
	ds_load_b128 v[56:59], v4 offset:16896
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_lw
	ds_load_b128 v[60:63], v4 offset:19456
	.Lcn_mq4sym_r128_down_g3_b0_lw:
	s_waitcnt vmcnt(3)
	v_cvt_f32_f16_e64 v28, v17.l
	v_lshrrev_b32_e32 v29, 16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	s_waitcnt vmcnt(2)
	v_bfe_u32 v24, v22, 0, 4
	v_bfe_u32 v25, v22, 4, 4
	v_bfe_u32 v26, v22, 8, 4
	v_bfe_u32 v27, v22, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g3_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_g3_b0_ld:
	ds_load_b128 v[32:35], v4 offset:2048
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_lw
	ds_load_b128 v[36:39], v4 offset:4608
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_lw
	ds_load_b128 v[40:43], v4 offset:7168
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_lw
	ds_load_b128 v[44:47], v4 offset:9728
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_lw
	ds_load_b128 v[48:51], v4 offset:12288
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_lw
	ds_load_b128 v[52:55], v4 offset:14848
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_lw
	ds_load_b128 v[56:59], v4 offset:17408
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_lw
	ds_load_b128 v[60:63], v4 offset:19968
	.Lcn_mq4sym_r128_down_g4_b0_lw:
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e64 v28, v18.l
	v_lshrrev_b32_e32 v29, 16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	s_waitcnt vmcnt(0)
	v_bfe_u32 v24, v23, 0, 4
	v_bfe_u32 v25, v23, 4, 4
	v_bfe_u32 v26, v23, 8, 4
	v_bfe_u32 v27, v23, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_g4_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_g4_b0_ld:
	.Lcn_mq4sym_r128_down_epi:
	v_cmp_gt_u32_e64 s72, 16, v0
	v_cndmask_b32_e64 v24, 0, 16, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v24, v24, v0, 2
	v_cmp_gt_u32_e64 s73, 24, v0
	v_cndmask_b32_e64 v25, 0, 8, s73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v25, v25, v0, 2
	v_cmp_gt_u32_e64 s74, 28, v0
	v_cndmask_b32_e64 v26, 0, 4, s74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v26, v26, v0, 2
	v_cmp_gt_u32_e64 s75, 30, v0
	v_cndmask_b32_e64 v27, 0, 2, s75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v27, v27, v0, 2
	v_cmp_gt_u32_e64 s76, 31, v0
	v_cndmask_b32_e64 v28, 0, 1, s76
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v28, v28, v0, 2
	ds_bpermute_b32 v29, v24, v6
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s72
	ds_bpermute_b32 v29, v25, v6
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s73
	ds_bpermute_b32 v29, v26, v6
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s74
	ds_bpermute_b32 v29, v27, v6
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s75
	ds_bpermute_b32 v29, v28, v6
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s76
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_red_done
	ds_bpermute_b32 v29, v24, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s72
	ds_bpermute_b32 v29, v25, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s73
	ds_bpermute_b32 v29, v26, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s74
	ds_bpermute_b32 v29, v27, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s75
	ds_bpermute_b32 v29, v28, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s76
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_red_done
	ds_bpermute_b32 v29, v24, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s72
	ds_bpermute_b32 v29, v25, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s73
	ds_bpermute_b32 v29, v26, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s74
	ds_bpermute_b32 v29, v27, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s75
	ds_bpermute_b32 v29, v28, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s76
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_red_done
	ds_bpermute_b32 v29, v24, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s72
	ds_bpermute_b32 v29, v25, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s73
	ds_bpermute_b32 v29, v26, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s74
	ds_bpermute_b32 v29, v27, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s75
	ds_bpermute_b32 v29, v28, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s76
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_red_done
	ds_bpermute_b32 v29, v24, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s72
	ds_bpermute_b32 v29, v25, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s73
	ds_bpermute_b32 v29, v26, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s74
	ds_bpermute_b32 v29, v27, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s75
	ds_bpermute_b32 v29, v28, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s76
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_red_done
	ds_bpermute_b32 v29, v24, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s72
	ds_bpermute_b32 v29, v25, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s73
	ds_bpermute_b32 v29, v26, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s74
	ds_bpermute_b32 v29, v27, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s75
	ds_bpermute_b32 v29, v28, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s76
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_red_done
	ds_bpermute_b32 v29, v24, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s72
	ds_bpermute_b32 v29, v25, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s73
	ds_bpermute_b32 v29, v26, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s74
	ds_bpermute_b32 v29, v27, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s75
	ds_bpermute_b32 v29, v28, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s76
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_red_done
	ds_bpermute_b32 v29, v24, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s72
	ds_bpermute_b32 v29, v25, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s73
	ds_bpermute_b32 v29, v26, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s74
	ds_bpermute_b32 v29, v27, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s75
	ds_bpermute_b32 v29, v28, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s76
	.Lcn_mq4sym_r128_down_red_done:
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s41, s58, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v30, s41
	global_store_b32 v30, v6, s[52:53]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_st_done
	s_mul_i32 s41, s59, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v31, s41
	global_store_b32 v31, v7, s[52:53]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_st_done
	s_mul_i32 s41, s60, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v32, s41
	global_store_b32 v32, v8, s[52:53]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_st_done
	s_mul_i32 s41, s61, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v33, s41
	global_store_b32 v33, v9, s[52:53]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_st_done
	s_mul_i32 s41, s62, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v34, s41
	global_store_b32 v34, v10, s[52:53]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_st_done
	s_mul_i32 s41, s63, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v35, s41
	global_store_b32 v35, v11, s[52:53]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_st_done
	s_mul_i32 s41, s64, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v36, s41
	global_store_b32 v36, v12, s[52:53]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_st_done
	s_mul_i32 s41, s65, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v37, s41
	global_store_b32 v37, v13, s[52:53]
	.Lcn_mq4sym_r128_down_st_done:
	.Lcn_mq4sym_r128_down_exit:
	s_endpgm
.Lmq_i4_moe_gemv_cn_mq4sym_r128_down_end:
.size mq_i4_moe_gemv_cn_mq4sym_r128_down, .Lmq_i4_moe_gemv_cn_mq4sym_r128_down_end-mq_i4_moe_gemv_cn_mq4sym_r128_down
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_cn_mq4sym_r128_down
	.amdhsa_group_segment_fixed_size 20480
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
	.amdhsa_next_free_vgpr 64
	.amdhsa_next_free_sgpr 101
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_cn_mq4sym_r128_down_end-mq_i4_moe_gemv_cn_mq4sym_r128_down)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8
.globl mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8
.p2align 8
.type mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8,@function
mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s100, s[0:1], 0x58
	v_lshrrev_b32_e32 v80, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v80
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v5, 4, v0
	v_mov_b32_e32 v4, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_u32_u24_e32 v80, 12, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v6, 5, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v6, 2, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v7, 31, v80
	s_mul_i32 s41, s40, 0x1400
	v_add_nc_u32_e32 v1, s41, v2
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_lg_u32 s24, 0x50
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_sub_i32 s41, s25, 1
	s_lshr_b32 s41, s41, 3
	s_cmp_gt_u32 s4, s41
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_mul_i32 s41, s4, 0x788
	s_add_u32 s10, s10, s41
	s_addc_u32 s11, s11, 0
	s_add_u32 s12, s12, s41
	s_addc_u32 s13, s13, 0
	s_add_u32 s14, s14, s41
	s_addc_u32 s15, s15, 0
	s_lshl_b32 s30, s2, 3
	s_cmp_ge_u32 s3, s24
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_lg_u32 s23, 0xa00
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_lshl_b32 s41, s3, 2
	s_add_u32 s36, s12, s41
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s41
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_add_i32 s30, s30, s40
	s_cmp_lt_u32 s30, s22
	s_cselect_b32 s44, 1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_gt_u32 s32, 0x50
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_gt_u32 s33, 0x50
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_mul_i32 s41, s25, 10
	s_cmp_gt_u32 s33, s41
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s34, s100
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_add_i32 s43, s35, -1
	s_min_u32 s41, s43, 0
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s58, s[36:37], null
	s_min_u32 s41, s43, 1
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s59, s[36:37], null
	s_min_u32 s41, s43, 2
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s60, s[36:37], null
	s_min_u32 s41, s43, 3
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s61, s[36:37], null
	s_min_u32 s41, s43, 4
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s62, s[36:37], null
	s_min_u32 s41, s43, 5
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s63, s[36:37], null
	s_min_u32 s41, s43, 6
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s64, s[36:37], null
	s_min_u32 s41, s43, 7
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s65, s[36:37], null
	s_min_u32 s41, s43, s40
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s57, s[36:37], null
	s_mul_i32 s42, s25, 10
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s58, s42
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s59, s42
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s60, s42
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s61, s42
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s62, s42
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s63, s42
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s64, s42
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_cmp_ge_u32 s65, s42
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	s_lshl_b32 s41, s34, 3
	s_add_u32 s36, s8, s41
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	v_mov_b32_e32 v80, s57
	v_cvt_f32_u32_e32 v85, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v85, v85
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_u32_e32 v82, v80
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v82, v82, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v83, v82, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v83, v80, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s80, 0, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v84, -1, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v82, v82, v84, s80
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v84, s26, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v83, v83, v84, s80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s80, s26, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v84, 1, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v82, v82, v84, s80
	v_cmp_le_i32_e64 s78, 0, v80
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v82, 1, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v81, 0x7fffff00, v82, s78
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s66, v81
	s_lshl_b32 s55, s28, 2
	s_mul_i32 s67, s66, s55
	v_add_nc_u32_e32 v2, s67, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, 0x1000, v2
	s_lshr_b32 s69, s22, 1
	s_sub_i32 s70, s30, s69
	s_cmp_lt_u32 s30, s69
	s_cselect_b64 s[52:53], s[18:19], s[20:21]
	s_cselect_b32 s71, s30, s70
	s_lshl_b32 s54, s71, 2
	s_lshl_b32 s56, s29, 2
	s_cmp_lt_u32 s40, s35
	s_cselect_b32 s68, -1, 0
	s_cmp_lt_u32 s66, s27
	s_cselect_b32 s77, -1, 0
	s_and_b32 s77, s77, s68
	s_xor_b32 s82, s68, s77
	.Lcn_mq3i4_r256_gate_up_t8_body:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_rowptr_done
	s_mul_i32 s41, s30, 0x3c0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s48, s38, s41
	s_addc_u32 s49, s39, 0
	s_mul_i32 s41, s22, 0x3c0
	s_mul_i32 s42, s30, 40
	s_add_i32 s41, s41, s42
	s_add_u32 s50, s38, s41
	s_addc_u32 s51, s39, 0
	s_load_b256 s[88:95], s[50:51], null
	s_load_b64 s[96:97], s[50:51], 0x20
	.Lcn_mq3i4_r256_gate_up_t8_rowptr_done:
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_z0
	s_mov_b32 exec_lo, s82
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
	s_mov_b32 exec_lo, -1
	.Lcn_mq3i4_r256_gate_up_t8_z0:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[80:83], v2, s[16:17]
	global_load_b128 v[84:87], v2, s[16:17] offset:512
	global_load_b128 v[88:91], v2, s[16:17] offset:1024
	global_load_b128 v[92:95], v2, s[16:17] offset:1536
	global_load_b128 v[96:99], v2, s[16:17] offset:2048
	global_load_b128 v[100:103], v2, s[16:17] offset:2560
	global_load_b128 v[104:107], v2, s[16:17] offset:3072
	global_load_b128 v[108:111], v2, s[16:17] offset:3584
	global_load_b128 v[112:115], v3, s[16:17]
	global_load_b128 v[116:119], v3, s[16:17] offset:512
	s_mov_b32 exec_lo, -1
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_weights_done
	global_load_b64 v[40:41], v6, s[48:49]
	global_load_b64 v[42:43], v6, s[48:49] offset:48
	global_load_b64 v[44:45], v6, s[48:49] offset:96
	global_load_b64 v[46:47], v6, s[48:49] offset:144
	global_load_b64 v[48:49], v6, s[48:49] offset:192
	global_load_b64 v[50:51], v6, s[48:49] offset:240
	global_load_b64 v[52:53], v6, s[48:49] offset:288
	global_load_b64 v[54:55], v6, s[48:49] offset:336
	global_load_b64 v[56:57], v6, s[48:49] offset:384
	global_load_b64 v[58:59], v6, s[48:49] offset:432
	global_load_b64 v[60:61], v6, s[48:49] offset:480
	global_load_b64 v[62:63], v6, s[48:49] offset:528
	global_load_b64 v[64:65], v6, s[48:49] offset:576
	global_load_b64 v[66:67], v6, s[48:49] offset:624
	global_load_b64 v[68:69], v6, s[48:49] offset:672
	global_load_b64 v[70:71], v6, s[48:49] offset:720
	global_load_b64 v[72:73], v6, s[48:49] offset:768
	global_load_b64 v[74:75], v6, s[48:49] offset:816
	global_load_b64 v[76:77], v6, s[48:49] offset:864
	global_load_b64 v[78:79], v6, s[48:49] offset:912
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	.Lcn_mq3i4_r256_gate_up_t8_weights_done:
	s_waitcnt vmcnt(0)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[80:83]
	ds_store_b128 v1, v[84:87] offset:512
	ds_store_b128 v1, v[88:91] offset:1024
	ds_store_b128 v1, v[92:95] offset:1536
	ds_store_b128 v1, v[96:99] offset:2048
	ds_store_b128 v1, v[100:103] offset:2560
	ds_store_b128 v1, v[104:107] offset:3072
	ds_store_b128 v1, v[108:111] offset:3584
	ds_store_b128 v1, v[112:115] offset:4096
	ds_store_b128 v1, v[116:119] offset:4608
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq3i4_r256_gate_up_t8_c0:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_s1
	ds_load_b128 v[92:95], v5
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[96:99], v5 offset:5120
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[100:103], v5 offset:10240
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[104:107], v5 offset:15360
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[108:111], v5 offset:20480
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[112:115], v5 offset:25600
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[116:119], v5 offset:30720
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[120:123], v5 offset:35840
	.Lcn_mq3i4_r256_gate_up_t8_g0_b0_lw:
	v_mov_b32_e32 v84, s88
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[40:41]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_t8_g0_b0_ld:
	ds_load_b128 v[92:95], v5 offset:512
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[96:99], v5 offset:5632
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[100:103], v5 offset:10752
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[104:107], v5 offset:15872
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[108:111], v5 offset:20992
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[112:115], v5 offset:26112
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[116:119], v5 offset:31232
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[120:123], v5 offset:36352
	.Lcn_mq3i4_r256_gate_up_t8_g1_b0_lw:
	s_lshr_b32 s98, s88, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[42:43]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_t8_g1_b0_ld:
	ds_load_b128 v[92:95], v5 offset:1024
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[96:99], v5 offset:6144
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[100:103], v5 offset:11264
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[104:107], v5 offset:16384
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[108:111], v5 offset:21504
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[112:115], v5 offset:26624
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[116:119], v5 offset:31744
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[120:123], v5 offset:36864
	.Lcn_mq3i4_r256_gate_up_t8_g2_b0_lw:
	v_mov_b32_e32 v84, s89
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[44:45]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_t8_g2_b0_ld:
	ds_load_b128 v[92:95], v5 offset:1536
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[96:99], v5 offset:6656
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[100:103], v5 offset:11776
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[104:107], v5 offset:16896
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[108:111], v5 offset:22016
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[112:115], v5 offset:27136
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[116:119], v5 offset:32256
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[120:123], v5 offset:37376
	.Lcn_mq3i4_r256_gate_up_t8_g3_b0_lw:
	s_lshr_b32 s99, s89, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_t8_g3_b0_ld:
	ds_load_b128 v[92:95], v5 offset:2048
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[96:99], v5 offset:7168
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[100:103], v5 offset:12288
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[104:107], v5 offset:17408
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[108:111], v5 offset:22528
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[112:115], v5 offset:27648
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[116:119], v5 offset:32768
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[120:123], v5 offset:37888
	.Lcn_mq3i4_r256_gate_up_t8_g4_b0_lw:
	v_mov_b32_e32 v84, s90
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[48:49]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_t8_g4_b0_ld:
	ds_load_b128 v[92:95], v5 offset:2560
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[96:99], v5 offset:7680
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[100:103], v5 offset:12800
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[104:107], v5 offset:17920
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[108:111], v5 offset:23040
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[112:115], v5 offset:28160
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[116:119], v5 offset:33280
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[120:123], v5 offset:38400
	.Lcn_mq3i4_r256_gate_up_t8_g5_b0_lw:
	s_lshr_b32 s98, s90, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[50:51]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_t8_g5_b0_ld:
	ds_load_b128 v[92:95], v5 offset:3072
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[96:99], v5 offset:8192
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[100:103], v5 offset:13312
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[104:107], v5 offset:18432
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[108:111], v5 offset:23552
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[112:115], v5 offset:28672
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[116:119], v5 offset:33792
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[120:123], v5 offset:38912
	.Lcn_mq3i4_r256_gate_up_t8_g6_b0_lw:
	v_mov_b32_e32 v84, s91
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[52:53]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_t8_g6_b0_ld:
	ds_load_b128 v[92:95], v5 offset:3584
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[96:99], v5 offset:8704
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[100:103], v5 offset:13824
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[104:107], v5 offset:18944
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[108:111], v5 offset:24064
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[112:115], v5 offset:29184
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[116:119], v5 offset:34304
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[120:123], v5 offset:39424
	.Lcn_mq3i4_r256_gate_up_t8_g7_b0_lw:
	s_lshr_b32 s99, s91, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[54:55]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_t8_g7_b0_ld:
	ds_load_b128 v[92:95], v5 offset:4096
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[96:99], v5 offset:9216
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[100:103], v5 offset:14336
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[104:107], v5 offset:19456
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[108:111], v5 offset:24576
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[112:115], v5 offset:29696
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[116:119], v5 offset:34816
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[120:123], v5 offset:39936
	.Lcn_mq3i4_r256_gate_up_t8_g8_b0_lw:
	v_mov_b32_e32 v84, s92
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[56:57]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_t8_g8_b0_ld:
	ds_load_b128 v[92:95], v5 offset:4608
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[96:99], v5 offset:9728
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[100:103], v5 offset:14848
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[104:107], v5 offset:19968
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[108:111], v5 offset:25088
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[112:115], v5 offset:30208
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[116:119], v5 offset:35328
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[120:123], v5 offset:40448
	.Lcn_mq3i4_r256_gate_up_t8_g9_b0_lw:
	s_lshr_b32 s98, s92, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[58:59]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_t8_g9_b0_ld:
	.Lcn_mq3i4_r256_gate_up_t8_s1:
	v_add_nc_u32_e32 v2, 0x1400, v2
	v_add_nc_u32_e32 v3, 0x1400, v3
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_z1
	s_mov_b32 exec_lo, s82
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
	s_mov_b32 exec_lo, -1
	.Lcn_mq3i4_r256_gate_up_t8_z1:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[80:83], v2, s[16:17]
	global_load_b128 v[84:87], v2, s[16:17] offset:512
	global_load_b128 v[88:91], v2, s[16:17] offset:1024
	global_load_b128 v[92:95], v2, s[16:17] offset:1536
	global_load_b128 v[96:99], v2, s[16:17] offset:2048
	global_load_b128 v[100:103], v2, s[16:17] offset:2560
	global_load_b128 v[104:107], v2, s[16:17] offset:3072
	global_load_b128 v[108:111], v2, s[16:17] offset:3584
	global_load_b128 v[112:115], v3, s[16:17]
	global_load_b128 v[116:119], v3, s[16:17] offset:512
	s_mov_b32 exec_lo, -1
	s_barrier
	s_waitcnt vmcnt(0)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[80:83]
	ds_store_b128 v1, v[84:87] offset:512
	ds_store_b128 v1, v[88:91] offset:1024
	ds_store_b128 v1, v[92:95] offset:1536
	ds_store_b128 v1, v[96:99] offset:2048
	ds_store_b128 v1, v[100:103] offset:2560
	ds_store_b128 v1, v[104:107] offset:3072
	ds_store_b128 v1, v[108:111] offset:3584
	ds_store_b128 v1, v[112:115] offset:4096
	ds_store_b128 v1, v[116:119] offset:4608
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq3i4_r256_gate_up_t8_c1:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_epi
	ds_load_b128 v[92:95], v5
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_lw
	ds_load_b128 v[96:99], v5 offset:5120
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_lw
	ds_load_b128 v[100:103], v5 offset:10240
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_lw
	ds_load_b128 v[104:107], v5 offset:15360
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_lw
	ds_load_b128 v[108:111], v5 offset:20480
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_lw
	ds_load_b128 v[112:115], v5 offset:25600
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_lw
	ds_load_b128 v[116:119], v5 offset:30720
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_lw
	ds_load_b128 v[120:123], v5 offset:35840
	.Lcn_mq3i4_r256_gate_up_t8_g10_b0_lw:
	v_mov_b32_e32 v84, s93
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[60:61]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g10_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_t8_g10_b0_ld:
	ds_load_b128 v[92:95], v5 offset:512
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_lw
	ds_load_b128 v[96:99], v5 offset:5632
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_lw
	ds_load_b128 v[100:103], v5 offset:10752
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_lw
	ds_load_b128 v[104:107], v5 offset:15872
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_lw
	ds_load_b128 v[108:111], v5 offset:20992
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_lw
	ds_load_b128 v[112:115], v5 offset:26112
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_lw
	ds_load_b128 v[116:119], v5 offset:31232
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_lw
	ds_load_b128 v[120:123], v5 offset:36352
	.Lcn_mq3i4_r256_gate_up_t8_g11_b0_lw:
	s_lshr_b32 s99, s93, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[62:63]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g11_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_t8_g11_b0_ld:
	ds_load_b128 v[92:95], v5 offset:1024
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_lw
	ds_load_b128 v[96:99], v5 offset:6144
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_lw
	ds_load_b128 v[100:103], v5 offset:11264
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_lw
	ds_load_b128 v[104:107], v5 offset:16384
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_lw
	ds_load_b128 v[108:111], v5 offset:21504
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_lw
	ds_load_b128 v[112:115], v5 offset:26624
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_lw
	ds_load_b128 v[116:119], v5 offset:31744
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_lw
	ds_load_b128 v[120:123], v5 offset:36864
	.Lcn_mq3i4_r256_gate_up_t8_g12_b0_lw:
	v_mov_b32_e32 v84, s94
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[64:65]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g12_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_t8_g12_b0_ld:
	ds_load_b128 v[92:95], v5 offset:1536
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_lw
	ds_load_b128 v[96:99], v5 offset:6656
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_lw
	ds_load_b128 v[100:103], v5 offset:11776
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_lw
	ds_load_b128 v[104:107], v5 offset:16896
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_lw
	ds_load_b128 v[108:111], v5 offset:22016
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_lw
	ds_load_b128 v[112:115], v5 offset:27136
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_lw
	ds_load_b128 v[116:119], v5 offset:32256
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_lw
	ds_load_b128 v[120:123], v5 offset:37376
	.Lcn_mq3i4_r256_gate_up_t8_g13_b0_lw:
	s_lshr_b32 s98, s94, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[66:67]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g13_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_t8_g13_b0_ld:
	ds_load_b128 v[92:95], v5 offset:2048
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_lw
	ds_load_b128 v[96:99], v5 offset:7168
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_lw
	ds_load_b128 v[100:103], v5 offset:12288
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_lw
	ds_load_b128 v[104:107], v5 offset:17408
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_lw
	ds_load_b128 v[108:111], v5 offset:22528
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_lw
	ds_load_b128 v[112:115], v5 offset:27648
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_lw
	ds_load_b128 v[116:119], v5 offset:32768
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_lw
	ds_load_b128 v[120:123], v5 offset:37888
	.Lcn_mq3i4_r256_gate_up_t8_g14_b0_lw:
	v_mov_b32_e32 v84, s95
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[68:69]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g14_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_t8_g14_b0_ld:
	ds_load_b128 v[92:95], v5 offset:2560
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_lw
	ds_load_b128 v[96:99], v5 offset:7680
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_lw
	ds_load_b128 v[100:103], v5 offset:12800
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_lw
	ds_load_b128 v[104:107], v5 offset:17920
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_lw
	ds_load_b128 v[108:111], v5 offset:23040
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_lw
	ds_load_b128 v[112:115], v5 offset:28160
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_lw
	ds_load_b128 v[116:119], v5 offset:33280
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_lw
	ds_load_b128 v[120:123], v5 offset:38400
	.Lcn_mq3i4_r256_gate_up_t8_g15_b0_lw:
	s_lshr_b32 s99, s95, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[70:71]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g15_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_t8_g15_b0_ld:
	ds_load_b128 v[92:95], v5 offset:3072
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_lw
	ds_load_b128 v[96:99], v5 offset:8192
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_lw
	ds_load_b128 v[100:103], v5 offset:13312
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_lw
	ds_load_b128 v[104:107], v5 offset:18432
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_lw
	ds_load_b128 v[108:111], v5 offset:23552
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_lw
	ds_load_b128 v[112:115], v5 offset:28672
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_lw
	ds_load_b128 v[116:119], v5 offset:33792
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_lw
	ds_load_b128 v[120:123], v5 offset:38912
	.Lcn_mq3i4_r256_gate_up_t8_g16_b0_lw:
	v_mov_b32_e32 v84, s96
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[72:73]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g16_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v89
	.Lcn_mq3i4_r256_gate_up_t8_g16_b0_ld:
	ds_load_b128 v[92:95], v5 offset:3584
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_lw
	ds_load_b128 v[96:99], v5 offset:8704
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_lw
	ds_load_b128 v[100:103], v5 offset:13824
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_lw
	ds_load_b128 v[104:107], v5 offset:18944
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_lw
	ds_load_b128 v[108:111], v5 offset:24064
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_lw
	ds_load_b128 v[112:115], v5 offset:29184
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_lw
	ds_load_b128 v[116:119], v5 offset:34304
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_lw
	ds_load_b128 v[120:123], v5 offset:39424
	.Lcn_mq3i4_r256_gate_up_t8_g17_b0_lw:
	s_lshr_b32 s98, s96, 16
	v_mov_b32_e32 v84, s98
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[74:75]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g17_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v89
	.Lcn_mq3i4_r256_gate_up_t8_g17_b0_ld:
	ds_load_b128 v[92:95], v5 offset:4096
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_lw
	ds_load_b128 v[96:99], v5 offset:9216
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_lw
	ds_load_b128 v[100:103], v5 offset:14336
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_lw
	ds_load_b128 v[104:107], v5 offset:19456
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_lw
	ds_load_b128 v[108:111], v5 offset:24576
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_lw
	ds_load_b128 v[112:115], v5 offset:29696
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_lw
	ds_load_b128 v[116:119], v5 offset:34816
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_lw
	ds_load_b128 v[120:123], v5 offset:39936
	.Lcn_mq3i4_r256_gate_up_t8_g18_b0_lw:
	v_mov_b32_e32 v84, s97
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[76:77]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g18_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v89
	.Lcn_mq3i4_r256_gate_up_t8_g18_b0_ld:
	ds_load_b128 v[92:95], v5 offset:4608
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_lw
	ds_load_b128 v[96:99], v5 offset:9728
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_lw
	ds_load_b128 v[100:103], v5 offset:14848
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_lw
	ds_load_b128 v[104:107], v5 offset:19968
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_lw
	ds_load_b128 v[108:111], v5 offset:25088
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_lw
	ds_load_b128 v[112:115], v5 offset:30208
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_lw
	ds_load_b128 v[116:119], v5 offset:35328
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_lw
	ds_load_b128 v[120:123], v5 offset:40448
	.Lcn_mq3i4_r256_gate_up_t8_g19_b0_lw:
	s_lshr_b32 s99, s97, 16
	v_mov_b32_e32 v84, s99
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v84, v84.l
	v_lshrrev_b64 v[86:87], v7, v[78:79]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v80, v86, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v81, v86, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v82, v86, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v83, v86, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v80, v80, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v81, v81, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v82, v82, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v83, v83, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v80, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v81, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v83, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v80, v84, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v81, v84, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v82, v84, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v83, v84, v83
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v88, v80, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v88
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_ld
	v_mul_f32_e32 v89, v80, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v89
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_ld
	v_mul_f32_e32 v88, v80, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v88
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_ld
	v_mul_f32_e32 v89, v80, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v89
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_ld
	v_mul_f32_e32 v88, v80, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v88
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_ld
	v_mul_f32_e32 v89, v80, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v89
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_ld
	v_mul_f32_e32 v88, v80, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v81, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v82, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v88, v83, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v88
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_g19_b0_ld
	v_mul_f32_e32 v89, v80, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v81, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v82, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v89, v83, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v89
	.Lcn_mq3i4_r256_gate_up_t8_g19_b0_ld:
	.Lcn_mq3i4_r256_gate_up_t8_epi:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r256_gate_up_t8_exit
	v_cmp_gt_u32_e64 s72, 16, v0
	v_cndmask_b32_e64 v80, 0, 16, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v80, v80, v0, 2
	v_cmp_gt_u32_e64 s73, 24, v0
	v_cndmask_b32_e64 v81, 0, 8, s73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v81, v81, v0, 2
	v_cmp_gt_u32_e64 s74, 28, v0
	v_cndmask_b32_e64 v82, 0, 4, s74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v82, v82, v0, 2
	v_cmp_gt_u32_e64 s75, 30, v0
	v_cndmask_b32_e64 v83, 0, 2, s75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v83, v83, v0, 2
	v_cmp_gt_u32_e64 s76, 31, v0
	v_cndmask_b32_e64 v84, 0, 1, s76
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v84, v84, v0, 2
	v_add_f32_e32 v8, v8, v9
	v_add_f32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v10
	ds_bpermute_b32 v85, v80, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s72
	ds_bpermute_b32 v85, v81, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s73
	ds_bpermute_b32 v85, v82, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s74
	ds_bpermute_b32 v85, v83, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s75
	ds_bpermute_b32 v85, v84, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v8, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v85, s76
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_red_done
	v_add_f32_e32 v12, v12, v13
	v_add_f32_e32 v14, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v14
	ds_bpermute_b32 v85, v80, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s72
	ds_bpermute_b32 v85, v81, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s73
	ds_bpermute_b32 v85, v82, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s74
	ds_bpermute_b32 v85, v83, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s75
	ds_bpermute_b32 v85, v84, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v12, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v85, s76
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_red_done
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	ds_bpermute_b32 v85, v80, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s72
	ds_bpermute_b32 v85, v81, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s73
	ds_bpermute_b32 v85, v82, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s74
	ds_bpermute_b32 v85, v83, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s75
	ds_bpermute_b32 v85, v84, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v16, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v85, s76
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_red_done
	v_add_f32_e32 v20, v20, v21
	v_add_f32_e32 v22, v22, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v22
	ds_bpermute_b32 v85, v80, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s72
	ds_bpermute_b32 v85, v81, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s73
	ds_bpermute_b32 v85, v82, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s74
	ds_bpermute_b32 v85, v83, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s75
	ds_bpermute_b32 v85, v84, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v20, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v85, s76
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_red_done
	v_add_f32_e32 v24, v24, v25
	v_add_f32_e32 v26, v26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v26
	ds_bpermute_b32 v85, v80, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s72
	ds_bpermute_b32 v85, v81, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s73
	ds_bpermute_b32 v85, v82, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s74
	ds_bpermute_b32 v85, v83, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s75
	ds_bpermute_b32 v85, v84, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v24, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v85, s76
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_red_done
	v_add_f32_e32 v28, v28, v29
	v_add_f32_e32 v30, v30, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v30
	ds_bpermute_b32 v85, v80, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s72
	ds_bpermute_b32 v85, v81, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s73
	ds_bpermute_b32 v85, v82, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s74
	ds_bpermute_b32 v85, v83, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s75
	ds_bpermute_b32 v85, v84, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v28, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v85, s76
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_red_done
	v_add_f32_e32 v32, v32, v33
	v_add_f32_e32 v34, v34, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v34
	ds_bpermute_b32 v85, v80, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s72
	ds_bpermute_b32 v85, v81, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s73
	ds_bpermute_b32 v85, v82, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s74
	ds_bpermute_b32 v85, v83, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s75
	ds_bpermute_b32 v85, v84, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v32, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v85, s76
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_red_done
	v_add_f32_e32 v36, v36, v37
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v38
	ds_bpermute_b32 v85, v80, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s72
	ds_bpermute_b32 v85, v81, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s73
	ds_bpermute_b32 v85, v82, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s74
	ds_bpermute_b32 v85, v83, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s75
	ds_bpermute_b32 v85, v84, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v36, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v85, s76
	.Lcn_mq3i4_r256_gate_up_t8_red_done:
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s41, s58, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v86, s41
	global_store_b32 v86, v8, s[52:53]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_st_done
	s_mul_i32 s41, s59, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v87, s41
	global_store_b32 v87, v12, s[52:53]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_st_done
	s_mul_i32 s41, s60, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v88, s41
	global_store_b32 v88, v16, s[52:53]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_st_done
	s_mul_i32 s41, s61, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v89, s41
	global_store_b32 v89, v20, s[52:53]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_st_done
	s_mul_i32 s41, s62, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v90, s41
	global_store_b32 v90, v24, s[52:53]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_st_done
	s_mul_i32 s41, s63, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v91, s41
	global_store_b32 v91, v28, s[52:53]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_st_done
	s_mul_i32 s41, s64, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v92, s41
	global_store_b32 v92, v32, s[52:53]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r256_gate_up_t8_st_done
	s_mul_i32 s41, s65, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v93, s41
	global_store_b32 v93, v36, s[52:53]
	.Lcn_mq3i4_r256_gate_up_t8_st_done:
	.Lcn_mq3i4_r256_gate_up_t8_exit:
	s_endpgm
.Lmq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8_end:
.size mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8, .Lmq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8_end-mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8
	.amdhsa_group_segment_fixed_size 40960
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
	.amdhsa_next_free_vgpr 124
	.amdhsa_next_free_sgpr 101
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8_end-mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_cn_mq3i4_r128_down_t8
.globl mq_i4_moe_gemv_cn_mq3i4_r128_down_t8
.p2align 8
.type mq_i4_moe_gemv_cn_mq3i4_r128_down_t8,@function
mq_i4_moe_gemv_cn_mq3i4_r128_down_t8:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s100, s[0:1], 0x58
	v_lshrrev_b32_e32 v32, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v32
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 4, v0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_u32_u24_e32 v32, 12, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v5, 5, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 2, v5
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v6, 31, v32
	s_mul_i32 s41, s40, 0xa00
	v_add_nc_u32_e32 v1, s41, v2
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_lg_u32 s24, 0x50
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_sub_i32 s41, s25, 1
	s_lshr_b32 s41, s41, 3
	s_cmp_gt_u32 s4, s41
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_mul_i32 s41, s4, 0x788
	s_add_u32 s10, s10, s41
	s_addc_u32 s11, s11, 0
	s_add_u32 s12, s12, s41
	s_addc_u32 s13, s13, 0
	s_add_u32 s14, s14, s41
	s_addc_u32 s15, s15, 0
	s_lshl_b32 s30, s2, 3
	s_cmp_ge_u32 s3, s24
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_lg_u32 s23, 0x280
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_lshl_b32 s41, s3, 2
	s_add_u32 s36, s12, s41
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s41
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_add_i32 s30, s30, s40
	s_cmp_lt_u32 s30, s22
	s_cselect_b32 s44, 1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_gt_u32 s32, 0x50
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_gt_u32 s33, 0x50
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_mul_i32 s41, s25, 10
	s_cmp_gt_u32 s33, s41
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s34, s100
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_add_i32 s43, s35, -1
	s_min_u32 s41, s43, 0
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s58, s[36:37], null
	s_min_u32 s41, s43, 1
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s59, s[36:37], null
	s_min_u32 s41, s43, 2
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s60, s[36:37], null
	s_min_u32 s41, s43, 3
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s61, s[36:37], null
	s_min_u32 s41, s43, 4
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s62, s[36:37], null
	s_min_u32 s41, s43, 5
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s63, s[36:37], null
	s_min_u32 s41, s43, 6
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s64, s[36:37], null
	s_min_u32 s41, s43, 7
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s65, s[36:37], null
	s_min_u32 s41, s43, s40
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s57, s[36:37], null
	s_mul_i32 s42, s25, 10
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s58, s42
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s59, s42
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s60, s42
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s61, s42
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s62, s42
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s63, s42
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s64, s42
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_cmp_ge_u32 s65, s42
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	s_lshl_b32 s41, s34, 3
	s_add_u32 s36, s8, s41
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	v_mov_b32_e32 v32, s57
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_3)
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
	v_cmp_gt_i32_e64 s80, 0, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, -1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s80
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, s26, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v35, v35, v36, s80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s80, s26, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v36, 1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v36, s80
	v_cmp_le_i32_e64 s78, 0, v32
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v34, 1, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v33, 0x7fffff00, v34, s78
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s66, v33
	s_lshl_b32 s55, s28, 2
	s_mul_i32 s67, s66, s55
	v_add_nc_u32_e32 v2, s67, v2
	s_mov_b32 s52, s18
	s_mov_b32 s53, s19
	s_lshl_b32 s54, s30, 2
	s_lshl_b32 s56, s29, 2
	s_cmp_lt_u32 s40, s35
	s_cselect_b32 s68, -1, 0
	s_cmp_lt_u32 s66, s27
	s_cselect_b32 s77, -1, 0
	s_and_b32 s77, s77, s68
	s_xor_b32 s82, s68, s77
	.Lcn_mq3i4_r128_down_t8_body:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_rowptr_done
	s_mul_i32 s41, s30, 0xf0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s48, s38, s41
	s_addc_u32 s49, s39, 0
	s_mul_i32 s41, s22, 0xf0
	s_mul_i32 s42, s30, 10
	s_add_i32 s41, s41, s42
	s_add_u32 s50, s38, s41
	s_addc_u32 s51, s39, 0
	.Lcn_mq3i4_r128_down_t8_rowptr_done:
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_z0
	s_mov_b32 exec_lo, s82
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
	s_mov_b32 exec_lo, -1
	.Lcn_mq3i4_r128_down_t8_z0:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[32:35], v2, s[16:17]
	global_load_b128 v[36:39], v2, s[16:17] offset:512
	global_load_b128 v[40:43], v2, s[16:17] offset:1024
	global_load_b128 v[44:47], v2, s[16:17] offset:1536
	global_load_b128 v[48:51], v2, s[16:17] offset:2048
	s_mov_b32 exec_lo, -1
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_weights_done
	global_load_b64 v[16:17], v5, s[48:49]
	global_load_u16 v26, v3, s[50:51]
	global_load_b64 v[18:19], v5, s[48:49] offset:48
	global_load_u16 v27, v3, s[50:51] offset:2
	global_load_b64 v[20:21], v5, s[48:49] offset:96
	global_load_u16 v28, v3, s[50:51] offset:4
	global_load_b64 v[22:23], v5, s[48:49] offset:144
	global_load_u16 v29, v3, s[50:51] offset:6
	global_load_b64 v[24:25], v5, s[48:49] offset:192
	global_load_u16 v30, v3, s[50:51] offset:8
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	.Lcn_mq3i4_r128_down_t8_weights_done:
	s_waitcnt vmcnt(0)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[32:35]
	ds_store_b128 v1, v[36:39] offset:512
	ds_store_b128 v1, v[40:43] offset:1024
	ds_store_b128 v1, v[44:47] offset:1536
	ds_store_b128 v1, v[48:51] offset:2048
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq3i4_r128_down_t8_c0:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_epi
	ds_load_b128 v[44:47], v4
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_lw
	ds_load_b128 v[48:51], v4 offset:2560
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_lw
	ds_load_b128 v[52:55], v4 offset:5120
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_lw
	ds_load_b128 v[56:59], v4 offset:7680
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_lw
	ds_load_b128 v[60:63], v4 offset:10240
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_lw
	ds_load_b128 v[64:67], v4 offset:12800
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_lw
	ds_load_b128 v[68:71], v4 offset:15360
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_lw
	ds_load_b128 v[72:75], v4 offset:17920
	.Lcn_mq3i4_r128_down_t8_g0_b0_lw:
	v_cvt_f32_f16_e64 v36, v26.l
	v_lshrrev_b64 v[38:39], v6, v[16:17]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_t8_g0_b0_ld:
	ds_load_b128 v[44:47], v4 offset:512
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_lw
	ds_load_b128 v[48:51], v4 offset:3072
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_lw
	ds_load_b128 v[52:55], v4 offset:5632
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_lw
	ds_load_b128 v[56:59], v4 offset:8192
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_lw
	ds_load_b128 v[60:63], v4 offset:10752
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_lw
	ds_load_b128 v[64:67], v4 offset:13312
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_lw
	ds_load_b128 v[68:71], v4 offset:15872
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_lw
	ds_load_b128 v[72:75], v4 offset:18432
	.Lcn_mq3i4_r128_down_t8_g1_b0_lw:
	v_cvt_f32_f16_e64 v36, v27.l
	v_lshrrev_b64 v[38:39], v6, v[18:19]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_t8_g1_b0_ld:
	ds_load_b128 v[44:47], v4 offset:1024
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_lw
	ds_load_b128 v[48:51], v4 offset:3584
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_lw
	ds_load_b128 v[52:55], v4 offset:6144
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_lw
	ds_load_b128 v[56:59], v4 offset:8704
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_lw
	ds_load_b128 v[60:63], v4 offset:11264
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_lw
	ds_load_b128 v[64:67], v4 offset:13824
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_lw
	ds_load_b128 v[68:71], v4 offset:16384
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_lw
	ds_load_b128 v[72:75], v4 offset:18944
	.Lcn_mq3i4_r128_down_t8_g2_b0_lw:
	v_cvt_f32_f16_e64 v36, v28.l
	v_lshrrev_b64 v[38:39], v6, v[20:21]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_t8_g2_b0_ld:
	ds_load_b128 v[44:47], v4 offset:1536
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_lw
	ds_load_b128 v[48:51], v4 offset:4096
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_lw
	ds_load_b128 v[52:55], v4 offset:6656
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_lw
	ds_load_b128 v[56:59], v4 offset:9216
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_lw
	ds_load_b128 v[60:63], v4 offset:11776
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_lw
	ds_load_b128 v[64:67], v4 offset:14336
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_lw
	ds_load_b128 v[68:71], v4 offset:16896
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_lw
	ds_load_b128 v[72:75], v4 offset:19456
	.Lcn_mq3i4_r128_down_t8_g3_b0_lw:
	v_cvt_f32_f16_e64 v36, v29.l
	v_lshrrev_b64 v[38:39], v6, v[22:23]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_t8_g3_b0_ld:
	ds_load_b128 v[44:47], v4 offset:2048
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_lw
	ds_load_b128 v[48:51], v4 offset:4608
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_lw
	ds_load_b128 v[52:55], v4 offset:7168
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_lw
	ds_load_b128 v[56:59], v4 offset:9728
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_lw
	ds_load_b128 v[60:63], v4 offset:12288
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_lw
	ds_load_b128 v[64:67], v4 offset:14848
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_lw
	ds_load_b128 v[68:71], v4 offset:17408
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_lw
	ds_load_b128 v[72:75], v4 offset:19968
	.Lcn_mq3i4_r128_down_t8_g4_b0_lw:
	v_cvt_f32_f16_e64 v36, v30.l
	v_lshrrev_b64 v[38:39], v6, v[24:25]
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_u32 v32, v38, 0, 3
	s_delay_alu instid0(VALU_DEP_2)
	v_bfe_u32 v33, v38, 3, 3
	s_delay_alu instid0(VALU_DEP_3)
	v_bfe_u32 v34, v38, 6, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v35, v38, 9, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v32, v32, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v34, v34, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 1, -7
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v34, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_i32_e32 v35, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v32, v36, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v33, v36, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v34, v36, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v35, v36, v35
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v40, v32, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v40
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v41, v32, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v41
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v40, v32, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v40
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v41, v32, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v41
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v40, v32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v41, v32, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v40, v32, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v34, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v35, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v40
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v41, v32, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v33, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v34, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v41, v35, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v41
	.Lcn_mq3i4_r128_down_t8_g4_b0_ld:
	.Lcn_mq3i4_r128_down_t8_epi:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq3i4_r128_down_t8_exit
	v_cmp_gt_u32_e64 s72, 16, v0
	v_cndmask_b32_e64 v32, 0, 16, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v32, v32, v0, 2
	v_cmp_gt_u32_e64 s73, 24, v0
	v_cndmask_b32_e64 v33, 0, 8, s73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v33, v33, v0, 2
	v_cmp_gt_u32_e64 s74, 28, v0
	v_cndmask_b32_e64 v34, 0, 4, s74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v34, v34, v0, 2
	v_cmp_gt_u32_e64 s75, 30, v0
	v_cndmask_b32_e64 v35, 0, 2, s75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v35, v35, v0, 2
	v_cmp_gt_u32_e64 s76, 31, v0
	v_cndmask_b32_e64 v36, 0, 1, s76
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v36, v36, v0, 2
	ds_bpermute_b32 v37, v32, v7
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s72
	ds_bpermute_b32 v37, v33, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s73
	ds_bpermute_b32 v37, v34, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s74
	ds_bpermute_b32 v37, v35, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s75
	ds_bpermute_b32 v37, v36, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v7, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v37, s76
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_red_done
	ds_bpermute_b32 v37, v32, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s72
	ds_bpermute_b32 v37, v33, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s73
	ds_bpermute_b32 v37, v34, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s74
	ds_bpermute_b32 v37, v35, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s75
	ds_bpermute_b32 v37, v36, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v8, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v37, s76
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_red_done
	ds_bpermute_b32 v37, v32, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s72
	ds_bpermute_b32 v37, v33, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s73
	ds_bpermute_b32 v37, v34, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s74
	ds_bpermute_b32 v37, v35, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s75
	ds_bpermute_b32 v37, v36, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v9, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v37, s76
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_red_done
	ds_bpermute_b32 v37, v32, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s72
	ds_bpermute_b32 v37, v33, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s73
	ds_bpermute_b32 v37, v34, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s74
	ds_bpermute_b32 v37, v35, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s75
	ds_bpermute_b32 v37, v36, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v10, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v37, s76
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_red_done
	ds_bpermute_b32 v37, v32, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s72
	ds_bpermute_b32 v37, v33, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s73
	ds_bpermute_b32 v37, v34, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s74
	ds_bpermute_b32 v37, v35, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s75
	ds_bpermute_b32 v37, v36, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v11, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v37, s76
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_red_done
	ds_bpermute_b32 v37, v32, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s72
	ds_bpermute_b32 v37, v33, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s73
	ds_bpermute_b32 v37, v34, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s74
	ds_bpermute_b32 v37, v35, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s75
	ds_bpermute_b32 v37, v36, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v12, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v37, s76
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_red_done
	ds_bpermute_b32 v37, v32, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s72
	ds_bpermute_b32 v37, v33, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s73
	ds_bpermute_b32 v37, v34, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s74
	ds_bpermute_b32 v37, v35, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s75
	ds_bpermute_b32 v37, v36, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v13, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v37, s76
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_red_done
	ds_bpermute_b32 v37, v32, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s72
	ds_bpermute_b32 v37, v33, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s73
	ds_bpermute_b32 v37, v34, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s74
	ds_bpermute_b32 v37, v35, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s75
	ds_bpermute_b32 v37, v36, v14
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v14, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v14, v14, v37, s76
	.Lcn_mq3i4_r128_down_t8_red_done:
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s41, s58, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v38, s41
	global_store_b32 v38, v7, s[52:53]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_st_done
	s_mul_i32 s41, s59, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v39, s41
	global_store_b32 v39, v8, s[52:53]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_st_done
	s_mul_i32 s41, s60, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v40, s41
	global_store_b32 v40, v9, s[52:53]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_st_done
	s_mul_i32 s41, s61, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v41, s41
	global_store_b32 v41, v10, s[52:53]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_st_done
	s_mul_i32 s41, s62, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v42, s41
	global_store_b32 v42, v11, s[52:53]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_st_done
	s_mul_i32 s41, s63, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v43, s41
	global_store_b32 v43, v12, s[52:53]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_st_done
	s_mul_i32 s41, s64, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v44, s41
	global_store_b32 v44, v13, s[52:53]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq3i4_r128_down_t8_st_done
	s_mul_i32 s41, s65, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v45, s41
	global_store_b32 v45, v14, s[52:53]
	.Lcn_mq3i4_r128_down_t8_st_done:
	.Lcn_mq3i4_r128_down_t8_exit:
	s_endpgm
.Lmq_i4_moe_gemv_cn_mq3i4_r128_down_t8_end:
.size mq_i4_moe_gemv_cn_mq3i4_r128_down_t8, .Lmq_i4_moe_gemv_cn_mq3i4_r128_down_t8_end-mq_i4_moe_gemv_cn_mq3i4_r128_down_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_cn_mq3i4_r128_down_t8
	.amdhsa_group_segment_fixed_size 20480
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
	.amdhsa_next_free_vgpr 76
	.amdhsa_next_free_sgpr 101
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_cn_mq3i4_r128_down_t8_end-mq_i4_moe_gemv_cn_mq3i4_r128_down_t8)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8
.globl mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8
.p2align 8
.type mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8,@function
mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s100, s[0:1], 0x58
	v_lshrrev_b32_e32 v60, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v60
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v5, 5, v0
	v_mov_b32_e32 v4, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v6, 2, v0
	v_lshrrev_b32_e32 v7, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 2, v7
	s_mul_i32 s41, s40, 0x1400
	v_add_nc_u32_e32 v1, s41, v2
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_lg_u32 s24, 0x50
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_sub_i32 s41, s25, 1
	s_lshr_b32 s41, s41, 3
	s_cmp_gt_u32 s4, s41
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_mul_i32 s41, s4, 0x788
	s_add_u32 s10, s10, s41
	s_addc_u32 s11, s11, 0
	s_add_u32 s12, s12, s41
	s_addc_u32 s13, s13, 0
	s_add_u32 s14, s14, s41
	s_addc_u32 s15, s15, 0
	s_lshl_b32 s30, s2, 3
	s_cmp_ge_u32 s3, s24
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_lg_u32 s23, 0xa00
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_lshl_b32 s41, s3, 2
	s_add_u32 s36, s12, s41
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s41
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_add_i32 s30, s30, s40
	s_cmp_lt_u32 s30, s22
	s_cselect_b32 s44, 1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_gt_u32 s32, 0x50
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_gt_u32 s33, 0x50
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_mul_i32 s41, s25, 10
	s_cmp_gt_u32 s33, s41
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s34, s100
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_add_i32 s43, s35, -1
	s_min_u32 s41, s43, 0
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s58, s[36:37], null
	s_min_u32 s41, s43, 1
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s59, s[36:37], null
	s_min_u32 s41, s43, 2
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s60, s[36:37], null
	s_min_u32 s41, s43, 3
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s61, s[36:37], null
	s_min_u32 s41, s43, 4
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s62, s[36:37], null
	s_min_u32 s41, s43, 5
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s63, s[36:37], null
	s_min_u32 s41, s43, 6
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s64, s[36:37], null
	s_min_u32 s41, s43, 7
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s65, s[36:37], null
	s_min_u32 s41, s43, s40
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s57, s[36:37], null
	s_mul_i32 s42, s25, 10
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s58, s42
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s59, s42
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s60, s42
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s61, s42
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s62, s42
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s63, s42
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s64, s42
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_cmp_ge_u32 s65, s42
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	s_lshl_b32 s41, s34, 3
	s_add_u32 s36, s8, s41
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	v_mov_b32_e32 v60, s57
	v_cvt_f32_u32_e32 v65, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v65, v65
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_u32_e32 v62, v60
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v62, v62, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v62, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v63, v62, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v63, v60, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s80, 0, v63
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v64, -1, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v64, s80
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v64, s26, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v63, v63, v64, s80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s80, s26, v63
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v64, 1, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, v64, s80
	v_cmp_le_i32_e64 s78, 0, v60
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v62, 1, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v61, 0x7fffff00, v62, s78
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s66, v61
	s_lshl_b32 s55, s28, 2
	s_mul_i32 s67, s66, s55
	v_add_nc_u32_e32 v2, s67, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, 0x1000, v2
	s_lshr_b32 s69, s22, 1
	s_sub_i32 s70, s30, s69
	s_cmp_lt_u32 s30, s69
	s_cselect_b64 s[52:53], s[18:19], s[20:21]
	s_cselect_b32 s71, s30, s70
	s_lshl_b32 s54, s71, 2
	s_lshl_b32 s56, s29, 2
	s_cmp_lt_u32 s40, s35
	s_cselect_b32 s68, -1, 0
	s_cmp_lt_u32 s66, s27
	s_cselect_b32 s77, -1, 0
	s_and_b32 s77, s77, s68
	s_xor_b32 s82, s68, s77
	.Lcn_mq4sym_r256_gate_up_t8_body:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_rowptr_done
	s_mul_i32 s41, s30, 0x550
	s_waitcnt lgkmcnt(0)
	s_add_u32 s48, s38, s41
	s_addc_u32 s49, s39, 0
	.Lcn_mq4sym_r256_gate_up_t8_rowptr_done:
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_z0
	s_mov_b32 exec_lo, s82
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
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	s_mov_b32 exec_lo, -1
	.Lcn_mq4sym_r256_gate_up_t8_z0:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[60:63], v2, s[16:17]
	global_load_b128 v[64:67], v2, s[16:17] offset:512
	global_load_b128 v[68:71], v2, s[16:17] offset:1024
	global_load_b128 v[72:75], v2, s[16:17] offset:1536
	global_load_b128 v[76:79], v2, s[16:17] offset:2048
	global_load_b128 v[80:83], v2, s[16:17] offset:2560
	global_load_b128 v[84:87], v2, s[16:17] offset:3072
	global_load_b128 v[88:91], v2, s[16:17] offset:3584
	global_load_b128 v[92:95], v3, s[16:17]
	global_load_b128 v[96:99], v3, s[16:17] offset:512
	s_mov_b32 exec_lo, -1
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_weights_done
	global_load_b32 v40, v7, s[48:49]
	global_load_b32 v50, v6, s[48:49] offset:8
	global_load_b32 v41, v7, s[48:49] offset:136
	global_load_b32 v51, v6, s[48:49] offset:144
	global_load_b32 v42, v7, s[48:49] offset:272
	global_load_b32 v52, v6, s[48:49] offset:280
	global_load_b32 v43, v7, s[48:49] offset:408
	global_load_b32 v53, v6, s[48:49] offset:416
	global_load_b32 v44, v7, s[48:49] offset:544
	global_load_b32 v54, v6, s[48:49] offset:552
	global_load_b32 v45, v7, s[48:49] offset:680
	global_load_b32 v55, v6, s[48:49] offset:688
	global_load_b32 v46, v7, s[48:49] offset:816
	global_load_b32 v56, v6, s[48:49] offset:824
	global_load_b32 v47, v7, s[48:49] offset:952
	global_load_b32 v57, v6, s[48:49] offset:960
	global_load_b32 v48, v7, s[48:49] offset:1088
	global_load_b32 v58, v6, s[48:49] offset:1096
	global_load_b32 v49, v7, s[48:49] offset:1224
	global_load_b32 v59, v6, s[48:49] offset:1232
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	.Lcn_mq4sym_r256_gate_up_t8_weights_done:
	s_waitcnt vmcnt(0)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[60:63]
	ds_store_b128 v1, v[64:67] offset:512
	ds_store_b128 v1, v[68:71] offset:1024
	ds_store_b128 v1, v[72:75] offset:1536
	ds_store_b128 v1, v[76:79] offset:2048
	ds_store_b128 v1, v[80:83] offset:2560
	ds_store_b128 v1, v[84:87] offset:3072
	ds_store_b128 v1, v[88:91] offset:3584
	ds_store_b128 v1, v[92:95] offset:4096
	ds_store_b128 v1, v[96:99] offset:4608
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq4sym_r256_gate_up_t8_c0:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_s1
	ds_load_b128 v[72:75], v5
	ds_load_b128 v[76:79], v5 offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[80:83], v5 offset:5120
	ds_load_b128 v[84:87], v5 offset:5136
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[88:91], v5 offset:10240
	ds_load_b128 v[92:95], v5 offset:10256
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b0_lw
	ds_load_b128 v[96:99], v5 offset:15360
	ds_load_b128 v[100:103], v5 offset:15376
	.Lcn_mq4sym_r256_gate_up_t8_g0_b0_lw:
	v_cvt_f32_f16_e64 v68, v40.l
	v_lshrrev_b32_e32 v69, 16, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v50, 0, 4
	v_bfe_u32 v61, v50, 4, 4
	v_bfe_u32 v62, v50, 8, 4
	v_bfe_u32 v63, v50, 12, 4
	v_bfe_u32 v64, v50, 16, 4
	v_bfe_u32 v65, v50, 20, 4
	v_bfe_u32 v66, v50, 24, 4
	v_bfe_u32 v67, v50, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v71
	.Lcn_mq4sym_r256_gate_up_t8_g0_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_done
	ds_load_b128 v[72:75], v5 offset:20480
	ds_load_b128 v[76:79], v5 offset:20496
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b1_lw
	ds_load_b128 v[80:83], v5 offset:25600
	ds_load_b128 v[84:87], v5 offset:25616
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b1_lw
	ds_load_b128 v[88:91], v5 offset:30720
	ds_load_b128 v[92:95], v5 offset:30736
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b1_lw
	ds_load_b128 v[96:99], v5 offset:35840
	ds_load_b128 v[100:103], v5 offset:35856
	.Lcn_mq4sym_r256_gate_up_t8_g0_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g0_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v71
	.Lcn_mq4sym_r256_gate_up_t8_g0_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g0_done:
	ds_load_b128 v[72:75], v5 offset:1024
	ds_load_b128 v[76:79], v5 offset:1040
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[80:83], v5 offset:6144
	ds_load_b128 v[84:87], v5 offset:6160
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[88:91], v5 offset:11264
	ds_load_b128 v[92:95], v5 offset:11280
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b0_lw
	ds_load_b128 v[96:99], v5 offset:16384
	ds_load_b128 v[100:103], v5 offset:16400
	.Lcn_mq4sym_r256_gate_up_t8_g1_b0_lw:
	v_cvt_f32_f16_e64 v68, v41.l
	v_lshrrev_b32_e32 v69, 16, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v51, 0, 4
	v_bfe_u32 v61, v51, 4, 4
	v_bfe_u32 v62, v51, 8, 4
	v_bfe_u32 v63, v51, 12, 4
	v_bfe_u32 v64, v51, 16, 4
	v_bfe_u32 v65, v51, 20, 4
	v_bfe_u32 v66, v51, 24, 4
	v_bfe_u32 v67, v51, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v71
	.Lcn_mq4sym_r256_gate_up_t8_g1_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_done
	ds_load_b128 v[72:75], v5 offset:21504
	ds_load_b128 v[76:79], v5 offset:21520
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b1_lw
	ds_load_b128 v[80:83], v5 offset:26624
	ds_load_b128 v[84:87], v5 offset:26640
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b1_lw
	ds_load_b128 v[88:91], v5 offset:31744
	ds_load_b128 v[92:95], v5 offset:31760
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b1_lw
	ds_load_b128 v[96:99], v5 offset:36864
	ds_load_b128 v[100:103], v5 offset:36880
	.Lcn_mq4sym_r256_gate_up_t8_g1_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g1_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v71
	.Lcn_mq4sym_r256_gate_up_t8_g1_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g1_done:
	ds_load_b128 v[72:75], v5 offset:2048
	ds_load_b128 v[76:79], v5 offset:2064
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[80:83], v5 offset:7168
	ds_load_b128 v[84:87], v5 offset:7184
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[88:91], v5 offset:12288
	ds_load_b128 v[92:95], v5 offset:12304
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b0_lw
	ds_load_b128 v[96:99], v5 offset:17408
	ds_load_b128 v[100:103], v5 offset:17424
	.Lcn_mq4sym_r256_gate_up_t8_g2_b0_lw:
	v_cvt_f32_f16_e64 v68, v42.l
	v_lshrrev_b32_e32 v69, 16, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v52, 0, 4
	v_bfe_u32 v61, v52, 4, 4
	v_bfe_u32 v62, v52, 8, 4
	v_bfe_u32 v63, v52, 12, 4
	v_bfe_u32 v64, v52, 16, 4
	v_bfe_u32 v65, v52, 20, 4
	v_bfe_u32 v66, v52, 24, 4
	v_bfe_u32 v67, v52, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v71
	.Lcn_mq4sym_r256_gate_up_t8_g2_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_done
	ds_load_b128 v[72:75], v5 offset:22528
	ds_load_b128 v[76:79], v5 offset:22544
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b1_lw
	ds_load_b128 v[80:83], v5 offset:27648
	ds_load_b128 v[84:87], v5 offset:27664
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b1_lw
	ds_load_b128 v[88:91], v5 offset:32768
	ds_load_b128 v[92:95], v5 offset:32784
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b1_lw
	ds_load_b128 v[96:99], v5 offset:37888
	ds_load_b128 v[100:103], v5 offset:37904
	.Lcn_mq4sym_r256_gate_up_t8_g2_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g2_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v71
	.Lcn_mq4sym_r256_gate_up_t8_g2_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g2_done:
	ds_load_b128 v[72:75], v5 offset:3072
	ds_load_b128 v[76:79], v5 offset:3088
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[80:83], v5 offset:8192
	ds_load_b128 v[84:87], v5 offset:8208
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[88:91], v5 offset:13312
	ds_load_b128 v[92:95], v5 offset:13328
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b0_lw
	ds_load_b128 v[96:99], v5 offset:18432
	ds_load_b128 v[100:103], v5 offset:18448
	.Lcn_mq4sym_r256_gate_up_t8_g3_b0_lw:
	v_cvt_f32_f16_e64 v68, v43.l
	v_lshrrev_b32_e32 v69, 16, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v53, 0, 4
	v_bfe_u32 v61, v53, 4, 4
	v_bfe_u32 v62, v53, 8, 4
	v_bfe_u32 v63, v53, 12, 4
	v_bfe_u32 v64, v53, 16, 4
	v_bfe_u32 v65, v53, 20, 4
	v_bfe_u32 v66, v53, 24, 4
	v_bfe_u32 v67, v53, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v71
	.Lcn_mq4sym_r256_gate_up_t8_g3_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_done
	ds_load_b128 v[72:75], v5 offset:23552
	ds_load_b128 v[76:79], v5 offset:23568
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b1_lw
	ds_load_b128 v[80:83], v5 offset:28672
	ds_load_b128 v[84:87], v5 offset:28688
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b1_lw
	ds_load_b128 v[88:91], v5 offset:33792
	ds_load_b128 v[92:95], v5 offset:33808
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b1_lw
	ds_load_b128 v[96:99], v5 offset:38912
	ds_load_b128 v[100:103], v5 offset:38928
	.Lcn_mq4sym_r256_gate_up_t8_g3_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g3_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v71
	.Lcn_mq4sym_r256_gate_up_t8_g3_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g3_done:
	ds_load_b128 v[72:75], v5 offset:4096
	ds_load_b128 v[76:79], v5 offset:4112
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[80:83], v5 offset:9216
	ds_load_b128 v[84:87], v5 offset:9232
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[88:91], v5 offset:14336
	ds_load_b128 v[92:95], v5 offset:14352
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b0_lw
	ds_load_b128 v[96:99], v5 offset:19456
	ds_load_b128 v[100:103], v5 offset:19472
	.Lcn_mq4sym_r256_gate_up_t8_g4_b0_lw:
	v_cvt_f32_f16_e64 v68, v44.l
	v_lshrrev_b32_e32 v69, 16, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v54, 0, 4
	v_bfe_u32 v61, v54, 4, 4
	v_bfe_u32 v62, v54, 8, 4
	v_bfe_u32 v63, v54, 12, 4
	v_bfe_u32 v64, v54, 16, 4
	v_bfe_u32 v65, v54, 20, 4
	v_bfe_u32 v66, v54, 24, 4
	v_bfe_u32 v67, v54, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v71
	.Lcn_mq4sym_r256_gate_up_t8_g4_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_done
	ds_load_b128 v[72:75], v5 offset:24576
	ds_load_b128 v[76:79], v5 offset:24592
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b1_lw
	ds_load_b128 v[80:83], v5 offset:29696
	ds_load_b128 v[84:87], v5 offset:29712
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b1_lw
	ds_load_b128 v[88:91], v5 offset:34816
	ds_load_b128 v[92:95], v5 offset:34832
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b1_lw
	ds_load_b128 v[96:99], v5 offset:39936
	ds_load_b128 v[100:103], v5 offset:39952
	.Lcn_mq4sym_r256_gate_up_t8_g4_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g4_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v71
	.Lcn_mq4sym_r256_gate_up_t8_g4_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g4_done:
	.Lcn_mq4sym_r256_gate_up_t8_s1:
	v_add_nc_u32_e32 v2, 0x1400, v2
	v_add_nc_u32_e32 v3, 0x1400, v3
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_z1
	s_mov_b32 exec_lo, s82
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
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	s_mov_b32 exec_lo, -1
	.Lcn_mq4sym_r256_gate_up_t8_z1:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[60:63], v2, s[16:17]
	global_load_b128 v[64:67], v2, s[16:17] offset:512
	global_load_b128 v[68:71], v2, s[16:17] offset:1024
	global_load_b128 v[72:75], v2, s[16:17] offset:1536
	global_load_b128 v[76:79], v2, s[16:17] offset:2048
	global_load_b128 v[80:83], v2, s[16:17] offset:2560
	global_load_b128 v[84:87], v2, s[16:17] offset:3072
	global_load_b128 v[88:91], v2, s[16:17] offset:3584
	global_load_b128 v[92:95], v3, s[16:17]
	global_load_b128 v[96:99], v3, s[16:17] offset:512
	s_mov_b32 exec_lo, -1
	s_barrier
	s_waitcnt vmcnt(0)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[60:63]
	ds_store_b128 v1, v[64:67] offset:512
	ds_store_b128 v1, v[68:71] offset:1024
	ds_store_b128 v1, v[72:75] offset:1536
	ds_store_b128 v1, v[76:79] offset:2048
	ds_store_b128 v1, v[80:83] offset:2560
	ds_store_b128 v1, v[84:87] offset:3072
	ds_store_b128 v1, v[88:91] offset:3584
	ds_store_b128 v1, v[92:95] offset:4096
	ds_store_b128 v1, v[96:99] offset:4608
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq4sym_r256_gate_up_t8_c1:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_epi
	ds_load_b128 v[72:75], v5
	ds_load_b128 v[76:79], v5 offset:16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[80:83], v5 offset:5120
	ds_load_b128 v[84:87], v5 offset:5136
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[88:91], v5 offset:10240
	ds_load_b128 v[92:95], v5 offset:10256
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b0_lw
	ds_load_b128 v[96:99], v5 offset:15360
	ds_load_b128 v[100:103], v5 offset:15376
	.Lcn_mq4sym_r256_gate_up_t8_g5_b0_lw:
	v_cvt_f32_f16_e64 v68, v45.l
	v_lshrrev_b32_e32 v69, 16, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v55, 0, 4
	v_bfe_u32 v61, v55, 4, 4
	v_bfe_u32 v62, v55, 8, 4
	v_bfe_u32 v63, v55, 12, 4
	v_bfe_u32 v64, v55, 16, 4
	v_bfe_u32 v65, v55, 20, 4
	v_bfe_u32 v66, v55, 24, 4
	v_bfe_u32 v67, v55, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v71
	.Lcn_mq4sym_r256_gate_up_t8_g5_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_done
	ds_load_b128 v[72:75], v5 offset:20480
	ds_load_b128 v[76:79], v5 offset:20496
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b1_lw
	ds_load_b128 v[80:83], v5 offset:25600
	ds_load_b128 v[84:87], v5 offset:25616
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b1_lw
	ds_load_b128 v[88:91], v5 offset:30720
	ds_load_b128 v[92:95], v5 offset:30736
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b1_lw
	ds_load_b128 v[96:99], v5 offset:35840
	ds_load_b128 v[100:103], v5 offset:35856
	.Lcn_mq4sym_r256_gate_up_t8_g5_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g5_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v71
	.Lcn_mq4sym_r256_gate_up_t8_g5_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g5_done:
	ds_load_b128 v[72:75], v5 offset:1024
	ds_load_b128 v[76:79], v5 offset:1040
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[80:83], v5 offset:6144
	ds_load_b128 v[84:87], v5 offset:6160
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[88:91], v5 offset:11264
	ds_load_b128 v[92:95], v5 offset:11280
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b0_lw
	ds_load_b128 v[96:99], v5 offset:16384
	ds_load_b128 v[100:103], v5 offset:16400
	.Lcn_mq4sym_r256_gate_up_t8_g6_b0_lw:
	v_cvt_f32_f16_e64 v68, v46.l
	v_lshrrev_b32_e32 v69, 16, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v56, 0, 4
	v_bfe_u32 v61, v56, 4, 4
	v_bfe_u32 v62, v56, 8, 4
	v_bfe_u32 v63, v56, 12, 4
	v_bfe_u32 v64, v56, 16, 4
	v_bfe_u32 v65, v56, 20, 4
	v_bfe_u32 v66, v56, 24, 4
	v_bfe_u32 v67, v56, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v22, v22, v71
	.Lcn_mq4sym_r256_gate_up_t8_g6_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_done
	ds_load_b128 v[72:75], v5 offset:21504
	ds_load_b128 v[76:79], v5 offset:21520
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b1_lw
	ds_load_b128 v[80:83], v5 offset:26624
	ds_load_b128 v[84:87], v5 offset:26640
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b1_lw
	ds_load_b128 v[88:91], v5 offset:31744
	ds_load_b128 v[92:95], v5 offset:31760
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b1_lw
	ds_load_b128 v[96:99], v5 offset:36864
	ds_load_b128 v[100:103], v5 offset:36880
	.Lcn_mq4sym_r256_gate_up_t8_g6_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v26, v26, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v30, v30, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g6_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v38, v38, v71
	.Lcn_mq4sym_r256_gate_up_t8_g6_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g6_done:
	ds_load_b128 v[72:75], v5 offset:2048
	ds_load_b128 v[76:79], v5 offset:2064
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[80:83], v5 offset:7168
	ds_load_b128 v[84:87], v5 offset:7184
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[88:91], v5 offset:12288
	ds_load_b128 v[92:95], v5 offset:12304
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b0_lw
	ds_load_b128 v[96:99], v5 offset:17408
	ds_load_b128 v[100:103], v5 offset:17424
	.Lcn_mq4sym_r256_gate_up_t8_g7_b0_lw:
	v_cvt_f32_f16_e64 v68, v47.l
	v_lshrrev_b32_e32 v69, 16, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v57, 0, 4
	v_bfe_u32 v61, v57, 4, 4
	v_bfe_u32 v62, v57, 8, 4
	v_bfe_u32 v63, v57, 12, 4
	v_bfe_u32 v64, v57, 16, 4
	v_bfe_u32 v65, v57, 20, 4
	v_bfe_u32 v66, v57, 24, 4
	v_bfe_u32 v67, v57, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v15, v15, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v19, v19, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v23, v23, v71
	.Lcn_mq4sym_r256_gate_up_t8_g7_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_done
	ds_load_b128 v[72:75], v5 offset:22528
	ds_load_b128 v[76:79], v5 offset:22544
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b1_lw
	ds_load_b128 v[80:83], v5 offset:27648
	ds_load_b128 v[84:87], v5 offset:27664
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b1_lw
	ds_load_b128 v[88:91], v5 offset:32768
	ds_load_b128 v[92:95], v5 offset:32784
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b1_lw
	ds_load_b128 v[96:99], v5 offset:37888
	ds_load_b128 v[100:103], v5 offset:37904
	.Lcn_mq4sym_r256_gate_up_t8_g7_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v27, v27, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v31, v31, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v35, v35, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g7_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v39, v39, v71
	.Lcn_mq4sym_r256_gate_up_t8_g7_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g7_done:
	ds_load_b128 v[72:75], v5 offset:3072
	ds_load_b128 v[76:79], v5 offset:3088
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[80:83], v5 offset:8192
	ds_load_b128 v[84:87], v5 offset:8208
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[88:91], v5 offset:13312
	ds_load_b128 v[92:95], v5 offset:13328
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b0_lw
	ds_load_b128 v[96:99], v5 offset:18432
	ds_load_b128 v[100:103], v5 offset:18448
	.Lcn_mq4sym_r256_gate_up_t8_g8_b0_lw:
	v_cvt_f32_f16_e64 v68, v48.l
	v_lshrrev_b32_e32 v69, 16, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v58, 0, 4
	v_bfe_u32 v61, v58, 4, 4
	v_bfe_u32 v62, v58, 8, 4
	v_bfe_u32 v63, v58, 12, 4
	v_bfe_u32 v64, v58, 16, 4
	v_bfe_u32 v65, v58, 20, 4
	v_bfe_u32 v66, v58, 24, 4
	v_bfe_u32 v67, v58, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v71
	.Lcn_mq4sym_r256_gate_up_t8_g8_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_done
	ds_load_b128 v[72:75], v5 offset:23552
	ds_load_b128 v[76:79], v5 offset:23568
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b1_lw
	ds_load_b128 v[80:83], v5 offset:28672
	ds_load_b128 v[84:87], v5 offset:28688
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b1_lw
	ds_load_b128 v[88:91], v5 offset:33792
	ds_load_b128 v[92:95], v5 offset:33808
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b1_lw
	ds_load_b128 v[96:99], v5 offset:38912
	ds_load_b128 v[100:103], v5 offset:38928
	.Lcn_mq4sym_r256_gate_up_t8_g8_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g8_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v71
	.Lcn_mq4sym_r256_gate_up_t8_g8_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g8_done:
	ds_load_b128 v[72:75], v5 offset:4096
	ds_load_b128 v[76:79], v5 offset:4112
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[80:83], v5 offset:9216
	ds_load_b128 v[84:87], v5 offset:9232
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[88:91], v5 offset:14336
	ds_load_b128 v[92:95], v5 offset:14352
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b0_lw
	ds_load_b128 v[96:99], v5 offset:19456
	ds_load_b128 v[100:103], v5 offset:19472
	.Lcn_mq4sym_r256_gate_up_t8_g9_b0_lw:
	v_cvt_f32_f16_e64 v68, v49.l
	v_lshrrev_b32_e32 v69, 16, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v69, v69.l
	v_bfe_u32 v60, v59, 0, 4
	v_bfe_u32 v61, v59, 4, 4
	v_bfe_u32 v62, v59, 8, 4
	v_bfe_u32 v63, v59, 12, 4
	v_bfe_u32 v64, v59, 16, 4
	v_bfe_u32 v65, v59, 20, 4
	v_bfe_u32 v66, v59, 24, 4
	v_bfe_u32 v67, v59, 28, 4
	v_cvt_f32_u32_e32 v60, v60
	v_cvt_f32_u32_e32 v61, v61
	v_cvt_f32_u32_e32 v62, v62
	v_cvt_f32_u32_e32 v63, v63
	v_cvt_f32_u32_e32 v64, v64
	v_cvt_f32_u32_e32 v65, v65
	v_cvt_f32_u32_e32 v66, v66
	v_cvt_f32_u32_e32 v67, v67
	v_fma_f32 v60, v68, v60, v69
	v_fma_f32 v61, v68, v61, v69
	v_fma_f32 v62, v68, v62, v69
	v_fma_f32 v63, v68, v63, v69
	v_fma_f32 v64, v68, v64, v69
	v_fma_f32 v65, v68, v65, v69
	v_fma_f32 v66, v68, v66, v69
	v_fma_f32 v67, v68, v67, v69
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v70
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v71
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v70
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b0_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v21, v21, v71
	.Lcn_mq4sym_r256_gate_up_t8_g9_b0_ld:
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_done
	ds_load_b128 v[72:75], v5 offset:24576
	ds_load_b128 v[76:79], v5 offset:24592
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b1_lw
	ds_load_b128 v[80:83], v5 offset:29696
	ds_load_b128 v[84:87], v5 offset:29712
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b1_lw
	ds_load_b128 v[88:91], v5 offset:34816
	ds_load_b128 v[92:95], v5 offset:34832
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b1_lw
	ds_load_b128 v[96:99], v5 offset:39936
	ds_load_b128 v[100:103], v5 offset:39952
	.Lcn_mq4sym_r256_gate_up_t8_g9_b1_lw:
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v70, v61, v73
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v74
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v76
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v70
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b1_ld
	v_mul_f32_e32 v71, v61, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v80
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v84
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v71
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b1_ld
	v_mul_f32_e32 v70, v61, v89
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v60, v88
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v62, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v63, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v64, v92
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v65, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v66, v94
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v70, v67, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v70
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_g9_b1_ld
	v_mul_f32_e32 v71, v61, v97
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v60, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v62, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v63, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v64, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v65, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v66, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v67, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v37, v37, v71
	.Lcn_mq4sym_r256_gate_up_t8_g9_b1_ld:
	.Lcn_mq4sym_r256_gate_up_t8_g9_done:
	.Lcn_mq4sym_r256_gate_up_t8_epi:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r256_gate_up_t8_exit
	v_cmp_gt_u32_e64 s72, 16, v0
	v_cndmask_b32_e64 v60, 0, 16, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v60, v60, v0, 2
	v_cmp_gt_u32_e64 s73, 24, v0
	v_cndmask_b32_e64 v61, 0, 8, s73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v61, v61, v0, 2
	v_cmp_gt_u32_e64 s74, 28, v0
	v_cndmask_b32_e64 v62, 0, 4, s74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v62, v62, v0, 2
	v_cmp_gt_u32_e64 s75, 30, v0
	v_cndmask_b32_e64 v63, 0, 2, s75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v63, v63, v0, 2
	v_cmp_gt_u32_e64 s76, 31, v0
	v_cndmask_b32_e64 v64, 0, 1, s76
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v64, v64, v0, 2
	v_add_f32_e32 v8, v8, v9
	v_add_f32_e32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v10
	ds_bpermute_b32 v65, v60, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s72
	ds_bpermute_b32 v65, v61, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s73
	ds_bpermute_b32 v65, v62, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s74
	ds_bpermute_b32 v65, v63, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s75
	ds_bpermute_b32 v65, v64, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v8, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v65, s76
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_red_done
	v_add_f32_e32 v12, v12, v13
	v_add_f32_e32 v14, v14, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v14
	ds_bpermute_b32 v65, v60, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s72
	ds_bpermute_b32 v65, v61, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s73
	ds_bpermute_b32 v65, v62, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s74
	ds_bpermute_b32 v65, v63, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s75
	ds_bpermute_b32 v65, v64, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v12, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v65, s76
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_red_done
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	ds_bpermute_b32 v65, v60, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s72
	ds_bpermute_b32 v65, v61, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s73
	ds_bpermute_b32 v65, v62, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s74
	ds_bpermute_b32 v65, v63, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s75
	ds_bpermute_b32 v65, v64, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v16, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, v16, v65, s76
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_red_done
	v_add_f32_e32 v20, v20, v21
	v_add_f32_e32 v22, v22, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v20, v20, v22
	ds_bpermute_b32 v65, v60, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s72
	ds_bpermute_b32 v65, v61, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s73
	ds_bpermute_b32 v65, v62, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s74
	ds_bpermute_b32 v65, v63, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s75
	ds_bpermute_b32 v65, v64, v20
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v20, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, v65, s76
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_red_done
	v_add_f32_e32 v24, v24, v25
	v_add_f32_e32 v26, v26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v26
	ds_bpermute_b32 v65, v60, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s72
	ds_bpermute_b32 v65, v61, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s73
	ds_bpermute_b32 v65, v62, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s74
	ds_bpermute_b32 v65, v63, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s75
	ds_bpermute_b32 v65, v64, v24
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v24, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, v65, s76
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_red_done
	v_add_f32_e32 v28, v28, v29
	v_add_f32_e32 v30, v30, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v30
	ds_bpermute_b32 v65, v60, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s72
	ds_bpermute_b32 v65, v61, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s73
	ds_bpermute_b32 v65, v62, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s74
	ds_bpermute_b32 v65, v63, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s75
	ds_bpermute_b32 v65, v64, v28
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v28, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, v65, s76
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_red_done
	v_add_f32_e32 v32, v32, v33
	v_add_f32_e32 v34, v34, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v34
	ds_bpermute_b32 v65, v60, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s72
	ds_bpermute_b32 v65, v61, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s73
	ds_bpermute_b32 v65, v62, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s74
	ds_bpermute_b32 v65, v63, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s75
	ds_bpermute_b32 v65, v64, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v32, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, v65, s76
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_red_done
	v_add_f32_e32 v36, v36, v37
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v36, v36, v38
	ds_bpermute_b32 v65, v60, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s72
	ds_bpermute_b32 v65, v61, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s73
	ds_bpermute_b32 v65, v62, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s74
	ds_bpermute_b32 v65, v63, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s75
	ds_bpermute_b32 v65, v64, v36
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v36, v36, v65, s76
	.Lcn_mq4sym_r256_gate_up_t8_red_done:
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s41, s58, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v66, s41
	global_store_b32 v66, v8, s[52:53]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_st_done
	s_mul_i32 s41, s59, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v67, s41
	global_store_b32 v67, v12, s[52:53]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_st_done
	s_mul_i32 s41, s60, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v68, s41
	global_store_b32 v68, v16, s[52:53]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_st_done
	s_mul_i32 s41, s61, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v69, s41
	global_store_b32 v69, v20, s[52:53]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_st_done
	s_mul_i32 s41, s62, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v70, s41
	global_store_b32 v70, v24, s[52:53]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_st_done
	s_mul_i32 s41, s63, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v71, s41
	global_store_b32 v71, v28, s[52:53]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_st_done
	s_mul_i32 s41, s64, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v72, s41
	global_store_b32 v72, v32, s[52:53]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r256_gate_up_t8_st_done
	s_mul_i32 s41, s65, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v73, s41
	global_store_b32 v73, v36, s[52:53]
	.Lcn_mq4sym_r256_gate_up_t8_st_done:
	.Lcn_mq4sym_r256_gate_up_t8_exit:
	s_endpgm
.Lmq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8_end:
.size mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8, .Lmq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8_end-mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8
	.amdhsa_group_segment_fixed_size 40960
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
	.amdhsa_next_free_vgpr 104
	.amdhsa_next_free_sgpr 101
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8_end-mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8)<<4)&1008)>>4
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
.protected mq_i4_moe_gemv_cn_mq4sym_r128_down_t8
.globl mq_i4_moe_gemv_cn_mq4sym_r128_down_t8
.p2align 8
.type mq_i4_moe_gemv_cn_mq4sym_r128_down_t8,@function
mq_i4_moe_gemv_cn_mq4sym_r128_down_t8:
	s_load_b256 s[8:15], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s100, s[0:1], 0x58
	v_lshrrev_b32_e32 v24, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s40, v24
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v4, 4, v0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v5, 1, v0
	s_mul_i32 s41, s40, 0xa00
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v1, s41, v2
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_lg_u32 s24, 0x50
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_sub_i32 s41, s25, 1
	s_lshr_b32 s41, s41, 3
	s_cmp_gt_u32 s4, s41
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_mul_i32 s41, s4, 0x788
	s_add_u32 s10, s10, s41
	s_addc_u32 s11, s11, 0
	s_add_u32 s12, s12, s41
	s_addc_u32 s13, s13, 0
	s_add_u32 s14, s14, s41
	s_addc_u32 s15, s15, 0
	s_lshl_b32 s30, s2, 3
	s_cmp_ge_u32 s3, s24
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_lg_u32 s23, 0x280
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_lshl_b32 s41, s3, 2
	s_add_u32 s36, s12, s41
	s_addc_u32 s37, s13, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s10, s41
	s_addc_u32 s37, s11, 0
	s_load_b32 s34, s[36:37], null
	s_add_i32 s30, s30, s40
	s_cmp_lt_u32 s30, s22
	s_cselect_b32 s44, 1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_gt_u32 s32, 0x50
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_gt_u32 s33, 0x50
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_mul_i32 s41, s25, 10
	s_cmp_gt_u32 s33, s41
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s34, s100
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_add_i32 s43, s35, -1
	s_min_u32 s41, s43, 0
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s58, s[36:37], null
	s_min_u32 s41, s43, 1
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s59, s[36:37], null
	s_min_u32 s41, s43, 2
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s60, s[36:37], null
	s_min_u32 s41, s43, 3
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s61, s[36:37], null
	s_min_u32 s41, s43, 4
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s62, s[36:37], null
	s_min_u32 s41, s43, 5
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s63, s[36:37], null
	s_min_u32 s41, s43, 6
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s64, s[36:37], null
	s_min_u32 s41, s43, 7
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s65, s[36:37], null
	s_min_u32 s41, s43, s40
	s_add_i32 s41, s41, s32
	s_lshl_b32 s41, s41, 2
	s_add_u32 s36, s14, s41
	s_addc_u32 s37, s15, 0
	s_load_b32 s57, s[36:37], null
	s_mul_i32 s42, s25, 10
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s58, s42
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s59, s42
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s60, s42
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s61, s42
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s62, s42
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s63, s42
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s64, s42
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_cmp_ge_u32 s65, s42
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	s_lshl_b32 s41, s34, 3
	s_add_u32 s36, s8, s41
	s_addc_u32 s37, s9, 0
	s_load_b64 s[38:39], s[36:37], null
	v_mov_b32_e32 v24, s57
	v_cvt_f32_u32_e32 v29, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_u32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v26, v26, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v27, v26, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v27, v24, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s80, 0, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, -1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s80
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, s26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v27, v27, v28, s80
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s80, s26, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v28, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, v28, s80
	v_cmp_le_i32_e64 s78, 0, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_u32_u24_e32 v26, 1, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v25, 0x7fffff00, v26, s78
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s66, v25
	s_lshl_b32 s55, s28, 2
	s_mul_i32 s67, s66, s55
	v_add_nc_u32_e32 v2, s67, v2
	s_mov_b32 s52, s18
	s_mov_b32 s53, s19
	s_lshl_b32 s54, s30, 2
	s_lshl_b32 s56, s29, 2
	s_cmp_lt_u32 s40, s35
	s_cselect_b32 s68, -1, 0
	s_cmp_lt_u32 s66, s27
	s_cselect_b32 s77, -1, 0
	s_and_b32 s77, s77, s68
	s_xor_b32 s82, s68, s77
	.Lcn_mq4sym_r128_down_t8_body:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_rowptr_done
	s_mul_i32 s41, s30, 0x154
	s_waitcnt lgkmcnt(0)
	s_add_u32 s48, s38, s41
	s_addc_u32 s49, s39, 0
	.Lcn_mq4sym_r128_down_t8_rowptr_done:
	s_cmp_eq_u32 s82, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_z0
	s_mov_b32 exec_lo, s82
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
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
	s_mov_b32 exec_lo, -1
	.Lcn_mq4sym_r128_down_t8_z0:
	s_mov_b32 exec_lo, s77
	global_load_b128 v[24:27], v2, s[16:17]
	global_load_b128 v[28:31], v2, s[16:17] offset:512
	global_load_b128 v[32:35], v2, s[16:17] offset:1024
	global_load_b128 v[36:39], v2, s[16:17] offset:1536
	global_load_b128 v[40:43], v2, s[16:17] offset:2048
	s_mov_b32 exec_lo, -1
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_weights_done
	global_load_b32 v14, v3, s[48:49]
	global_load_u16 v19, v5, s[48:49] offset:4
	global_load_b32 v15, v3, s[48:49] offset:68
	global_load_u16 v20, v5, s[48:49] offset:72
	global_load_b32 v16, v3, s[48:49] offset:136
	global_load_u16 v21, v5, s[48:49] offset:140
	global_load_b32 v17, v3, s[48:49] offset:204
	global_load_u16 v22, v5, s[48:49] offset:208
	global_load_b32 v18, v3, s[48:49] offset:272
	global_load_u16 v23, v5, s[48:49] offset:276
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	.Lcn_mq4sym_r128_down_t8_weights_done:
	s_waitcnt vmcnt(0)
	s_mov_b32 exec_lo, s68
	ds_store_b128 v1, v[24:27]
	ds_store_b128 v1, v[28:31] offset:512
	ds_store_b128 v1, v[32:35] offset:1024
	ds_store_b128 v1, v[36:39] offset:1536
	ds_store_b128 v1, v[40:43] offset:2048
	s_mov_b32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	s_barrier
	.Lcn_mq4sym_r128_down_t8_c0:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_epi
	ds_load_b128 v[32:35], v4
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_lw
	ds_load_b128 v[36:39], v4 offset:2560
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_lw
	ds_load_b128 v[40:43], v4 offset:5120
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_lw
	ds_load_b128 v[44:47], v4 offset:7680
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_lw
	ds_load_b128 v[48:51], v4 offset:10240
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_lw
	ds_load_b128 v[52:55], v4 offset:12800
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_lw
	ds_load_b128 v[56:59], v4 offset:15360
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_lw
	ds_load_b128 v[60:63], v4 offset:17920
	.Lcn_mq4sym_r128_down_t8_g0_b0_lw:
	v_cvt_f32_f16_e64 v28, v14.l
	v_lshrrev_b32_e32 v29, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	v_bfe_u32 v24, v19, 0, 4
	v_bfe_u32 v25, v19, 4, 4
	v_bfe_u32 v26, v19, 8, 4
	v_bfe_u32 v27, v19, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g0_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_t8_g0_b0_ld:
	ds_load_b128 v[32:35], v4 offset:512
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_lw
	ds_load_b128 v[36:39], v4 offset:3072
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_lw
	ds_load_b128 v[40:43], v4 offset:5632
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_lw
	ds_load_b128 v[44:47], v4 offset:8192
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_lw
	ds_load_b128 v[48:51], v4 offset:10752
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_lw
	ds_load_b128 v[52:55], v4 offset:13312
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_lw
	ds_load_b128 v[56:59], v4 offset:15872
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_lw
	ds_load_b128 v[60:63], v4 offset:18432
	.Lcn_mq4sym_r128_down_t8_g1_b0_lw:
	v_cvt_f32_f16_e64 v28, v15.l
	v_lshrrev_b32_e32 v29, 16, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	v_bfe_u32 v24, v20, 0, 4
	v_bfe_u32 v25, v20, 4, 4
	v_bfe_u32 v26, v20, 8, 4
	v_bfe_u32 v27, v20, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g1_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_t8_g1_b0_ld:
	ds_load_b128 v[32:35], v4 offset:1024
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_lw
	ds_load_b128 v[36:39], v4 offset:3584
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_lw
	ds_load_b128 v[40:43], v4 offset:6144
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_lw
	ds_load_b128 v[44:47], v4 offset:8704
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_lw
	ds_load_b128 v[48:51], v4 offset:11264
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_lw
	ds_load_b128 v[52:55], v4 offset:13824
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_lw
	ds_load_b128 v[56:59], v4 offset:16384
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_lw
	ds_load_b128 v[60:63], v4 offset:18944
	.Lcn_mq4sym_r128_down_t8_g2_b0_lw:
	v_cvt_f32_f16_e64 v28, v16.l
	v_lshrrev_b32_e32 v29, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	v_bfe_u32 v24, v21, 0, 4
	v_bfe_u32 v25, v21, 4, 4
	v_bfe_u32 v26, v21, 8, 4
	v_bfe_u32 v27, v21, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g2_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_t8_g2_b0_ld:
	ds_load_b128 v[32:35], v4 offset:1536
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_lw
	ds_load_b128 v[36:39], v4 offset:4096
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_lw
	ds_load_b128 v[40:43], v4 offset:6656
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_lw
	ds_load_b128 v[44:47], v4 offset:9216
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_lw
	ds_load_b128 v[48:51], v4 offset:11776
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_lw
	ds_load_b128 v[52:55], v4 offset:14336
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_lw
	ds_load_b128 v[56:59], v4 offset:16896
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_lw
	ds_load_b128 v[60:63], v4 offset:19456
	.Lcn_mq4sym_r128_down_t8_g3_b0_lw:
	v_cvt_f32_f16_e64 v28, v17.l
	v_lshrrev_b32_e32 v29, 16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	v_bfe_u32 v24, v22, 0, 4
	v_bfe_u32 v25, v22, 4, 4
	v_bfe_u32 v26, v22, 8, 4
	v_bfe_u32 v27, v22, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g3_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_t8_g3_b0_ld:
	ds_load_b128 v[32:35], v4 offset:2048
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_lw
	ds_load_b128 v[36:39], v4 offset:4608
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_lw
	ds_load_b128 v[40:43], v4 offset:7168
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_lw
	ds_load_b128 v[44:47], v4 offset:9728
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_lw
	ds_load_b128 v[48:51], v4 offset:12288
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_lw
	ds_load_b128 v[52:55], v4 offset:14848
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_lw
	ds_load_b128 v[56:59], v4 offset:17408
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_lw
	ds_load_b128 v[60:63], v4 offset:19968
	.Lcn_mq4sym_r128_down_t8_g4_b0_lw:
	v_cvt_f32_f16_e64 v28, v18.l
	v_lshrrev_b32_e32 v29, 16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_f16_e64 v29, v29.l
	v_bfe_u32 v24, v23, 0, 4
	v_bfe_u32 v25, v23, 4, 4
	v_bfe_u32 v26, v23, 8, 4
	v_bfe_u32 v27, v23, 12, 4
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_u32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v24, v28, v24, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v25, v28, v25, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v26, v28, v26, v29
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v27, v28, v27, v29
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v30, v25, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v30
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v31, v25, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v31
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v30, v25, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v8, v30
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v31, v25, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v31
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v30, v25, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v30
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v31, v25, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v31
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v30, v25, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v24, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v26, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v30, v27, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v30
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_g4_b0_ld
	v_mul_f32_e32 v31, v25, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v24, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v26, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v31, v27, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v31
	.Lcn_mq4sym_r128_down_t8_g4_b0_ld:
	.Lcn_mq4sym_r128_down_t8_epi:
	s_cmp_eq_u32 s44, 0
	s_cbranch_scc1 .Lcn_mq4sym_r128_down_t8_exit
	v_cmp_gt_u32_e64 s72, 16, v0
	v_cndmask_b32_e64 v24, 0, 16, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v24, v24, v0, 2
	v_cmp_gt_u32_e64 s73, 24, v0
	v_cndmask_b32_e64 v25, 0, 8, s73
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v25, v25, v0, 2
	v_cmp_gt_u32_e64 s74, 28, v0
	v_cndmask_b32_e64 v26, 0, 4, s74
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v26, v26, v0, 2
	v_cmp_gt_u32_e64 s75, 30, v0
	v_cndmask_b32_e64 v27, 0, 2, s75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v27, v27, v0, 2
	v_cmp_gt_u32_e64 s76, 31, v0
	v_cndmask_b32_e64 v28, 0, 1, s76
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v28, v28, v0, 2
	ds_bpermute_b32 v29, v24, v6
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s72
	ds_bpermute_b32 v29, v25, v6
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s73
	ds_bpermute_b32 v29, v26, v6
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s74
	ds_bpermute_b32 v29, v27, v6
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s75
	ds_bpermute_b32 v29, v28, v6
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v6, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v6, v6, v29, s76
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_red_done
	ds_bpermute_b32 v29, v24, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s72
	ds_bpermute_b32 v29, v25, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s73
	ds_bpermute_b32 v29, v26, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s74
	ds_bpermute_b32 v29, v27, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s75
	ds_bpermute_b32 v29, v28, v7
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v7, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v7, v7, v29, s76
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_red_done
	ds_bpermute_b32 v29, v24, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s72
	ds_bpermute_b32 v29, v25, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s73
	ds_bpermute_b32 v29, v26, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s74
	ds_bpermute_b32 v29, v27, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s75
	ds_bpermute_b32 v29, v28, v8
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v8, v8, v29, s76
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_red_done
	ds_bpermute_b32 v29, v24, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s72
	ds_bpermute_b32 v29, v25, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s73
	ds_bpermute_b32 v29, v26, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s74
	ds_bpermute_b32 v29, v27, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s75
	ds_bpermute_b32 v29, v28, v9
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v9, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v29, s76
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_red_done
	ds_bpermute_b32 v29, v24, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s72
	ds_bpermute_b32 v29, v25, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s73
	ds_bpermute_b32 v29, v26, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s74
	ds_bpermute_b32 v29, v27, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s75
	ds_bpermute_b32 v29, v28, v10
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v10, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v29, s76
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_red_done
	ds_bpermute_b32 v29, v24, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s72
	ds_bpermute_b32 v29, v25, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s73
	ds_bpermute_b32 v29, v26, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s74
	ds_bpermute_b32 v29, v27, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s75
	ds_bpermute_b32 v29, v28, v11
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v11, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v29, s76
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_red_done
	ds_bpermute_b32 v29, v24, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s72
	ds_bpermute_b32 v29, v25, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s73
	ds_bpermute_b32 v29, v26, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s74
	ds_bpermute_b32 v29, v27, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s75
	ds_bpermute_b32 v29, v28, v12
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v12, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v29, s76
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_red_done
	ds_bpermute_b32 v29, v24, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s72
	ds_bpermute_b32 v29, v25, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s73
	ds_bpermute_b32 v29, v26, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s74
	ds_bpermute_b32 v29, v27, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s75
	ds_bpermute_b32 v29, v28, v13
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v13, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v13, v13, v29, s76
	.Lcn_mq4sym_r128_down_t8_red_done:
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s41, s58, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v30, s41
	global_store_b32 v30, v6, s[52:53]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_st_done
	s_mul_i32 s41, s59, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v31, s41
	global_store_b32 v31, v7, s[52:53]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_st_done
	s_mul_i32 s41, s60, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v32, s41
	global_store_b32 v32, v8, s[52:53]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_st_done
	s_mul_i32 s41, s61, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v33, s41
	global_store_b32 v33, v9, s[52:53]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_st_done
	s_mul_i32 s41, s62, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v34, s41
	global_store_b32 v34, v10, s[52:53]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_st_done
	s_mul_i32 s41, s63, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v35, s41
	global_store_b32 v35, v11, s[52:53]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_st_done
	s_mul_i32 s41, s64, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v36, s41
	global_store_b32 v36, v12, s[52:53]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Lcn_mq4sym_r128_down_t8_st_done
	s_mul_i32 s41, s65, s56
	s_add_i32 s41, s41, s54
	v_mov_b32_e32 v37, s41
	global_store_b32 v37, v13, s[52:53]
	.Lcn_mq4sym_r128_down_t8_st_done:
	.Lcn_mq4sym_r128_down_t8_exit:
	s_endpgm
.Lmq_i4_moe_gemv_cn_mq4sym_r128_down_t8_end:
.size mq_i4_moe_gemv_cn_mq4sym_r128_down_t8, .Lmq_i4_moe_gemv_cn_mq4sym_r128_down_t8_end-mq_i4_moe_gemv_cn_mq4sym_r128_down_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mq_i4_moe_gemv_cn_mq4sym_r128_down_t8
	.amdhsa_group_segment_fixed_size 20480
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
	.amdhsa_next_free_vgpr 64
	.amdhsa_next_free_sgpr 101
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lmq_i4_moe_gemv_cn_mq4sym_r128_down_t8_end-mq_i4_moe_gemv_cn_mq4sym_r128_down_t8)<<4)&1008)>>4
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
    .group_segment_fixed_size: 40960
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_cn_mq3i4_r256_gate_up
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_cn_mq3i4_r256_gate_up.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 124
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
    .group_segment_fixed_size: 20480
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_cn_mq3i4_r128_down
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_cn_mq3i4_r128_down.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 76
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
    .group_segment_fixed_size: 40960
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_cn_mq4sym_r256_gate_up
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_cn_mq4sym_r256_gate_up.kd
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
    .group_segment_fixed_size: 20480
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_cn_mq4sym_r128_down
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_cn_mq4sym_r128_down.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 64
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
    .group_segment_fixed_size: 40960
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 124
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
    .group_segment_fixed_size: 20480
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_cn_mq3i4_r128_down_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_cn_mq3i4_r128_down_t8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 76
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
    .group_segment_fixed_size: 40960
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8.kd
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
    .group_segment_fixed_size: 20480
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: mq_i4_moe_gemv_cn_mq4sym_r128_down_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .symbol: mq_i4_moe_gemv_cn_mq4sym_r128_down_t8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 64
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
