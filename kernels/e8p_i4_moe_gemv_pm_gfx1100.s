.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.amdhsa_code_object_version 6
.text
.protected e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100
.globl e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100
.p2align 8
.type e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100,@function
e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100:
	v_lshlrev_b32_e32 v24, 2, v0
	s_load_b256 s[88:95], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s4, s[0:1], 0x58
	s_mov_b32 s30, s14
	s_mov_b32 s31, s15
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_exit
	s_lshl_b32 s51, s31, 2
	s_add_u32 s36, s92, s51
	s_addc_u32 s37, s93, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s90, s51
	s_addc_u32 s37, s91, 0
	s_load_b32 s34, s[36:37], null
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_exit
	s_cmp_ge_u32 s34, s4
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_exit
	s_lshl_b32 s51, s34, 3
	s_add_u32 s36, s88, s51
	s_addc_u32 s37, s89, 0
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
	v_and_b32_e32 v4, 30, v0
	v_and_b32_e32 v5, 1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 4, v5
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
	v_min_u32_e32 v39, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, s32, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 2, v39
	global_load_b32 v32, v39, s[94:95]
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	s_waitcnt vmcnt(0)
	v_cvt_f32_u32_e32 v34, v32
	s_waitcnt_depctr depctr_va_vdst(0)
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
	v_mov_b32_e32 v96, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v38, 0
	v_mov_b32_e32 v104, s73
	v_readlane_b32 s74, v33, 1
	v_mov_b32_e32 v97, s74
	v_readlane_b32 s74, v38, 1
	v_mov_b32_e32 v105, s74
	v_readlane_b32 s75, v33, 2
	v_mov_b32_e32 v98, s75
	v_readlane_b32 s75, v38, 2
	v_mov_b32_e32 v106, s75
	v_readlane_b32 s76, v33, 3
	v_mov_b32_e32 v99, s76
	v_readlane_b32 s76, v38, 3
	v_mov_b32_e32 v107, s76
	v_readlane_b32 s73, v33, 4
	v_mov_b32_e32 v100, s73
	v_readlane_b32 s73, v38, 4
	v_mov_b32_e32 v108, s73
	v_readlane_b32 s74, v33, 5
	v_mov_b32_e32 v101, s74
	v_readlane_b32 s74, v38, 5
	v_mov_b32_e32 v109, s74
	v_readlane_b32 s75, v33, 6
	v_mov_b32_e32 v102, s75
	v_readlane_b32 s75, v38, 6
	v_mov_b32_e32 v110, s75
	v_readlane_b32 s76, v33, 7
	v_mov_b32_e32 v103, s76
	v_readlane_b32 s76, v38, 7
	v_mov_b32_e32 v111, s76
	v_lshrrev_b32_e32 v34, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v0, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v35, v35, v34
	ds_store_b32 v24, v35 offset:1024
	v_add_nc_u32_e32 v32, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 4, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v32, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v36, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v36, v36, v34
	ds_store_b32 v24, v36 offset:1152
	v_add_nc_u32_e32 v32, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 4, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v32, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v37, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v37, v37, v34
	ds_store_b32 v24, v37 offset:1280
	v_add_nc_u32_e32 v32, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 4, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v32, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v38, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v38, v38, v34
	ds_store_b32 v24, v38 offset:1408
	s_mov_b32 s80, 0x11111111
	s_mov_b32 s81, 0xd1111111
	s_mov_b32 s82, 0x51111111
	s_mov_b32 s83, 0xf3111111
	s_mov_b32 s84, 0x33111111
	s_mov_b32 s85, 0xb3111111
	s_mov_b32 s86, 0x15111111
	s_mov_b32 s87, 0xd5111111
	v_writelane_b32 v39, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s87, 7
	s_mov_b32 s80, 0xf1311111
	s_mov_b32 s81, 0x31311111
	s_mov_b32 s82, 0xb1311111
	s_mov_b32 s83, 0x13311111
	s_mov_b32 s84, 0xd3311111
	s_mov_b32 s85, 0xf5311111
	s_mov_b32 s86, 0x11511111
	s_mov_b32 s87, 0xd1511111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s87, 15
	s_mov_b32 s80, 0xf3511111
	s_mov_b32 s81, 0xf1131111
	s_mov_b32 s82, 0x31131111
	s_mov_b32 s83, 0xb1131111
	s_mov_b32 s84, 0x13131111
	s_mov_b32 s85, 0xd3131111
	s_mov_b32 s86, 0xf5131111
	s_mov_b32 s87, 0x11331111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s87, 23
	s_mov_b32 s80, 0xd1331111
	s_mov_b32 s81, 0xf3331111
	s_mov_b32 s82, 0x33331111
	s_mov_b32 s83, 0xf1531111
	s_mov_b32 s84, 0x11151111
	s_mov_b32 s85, 0xd1151111
	s_mov_b32 s86, 0xf3151111
	s_mov_b32 s87, 0xf1351111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s87, 31
	ds_store_b32 v24, v39
	s_mov_b32 s80, 0xf1113111
	s_mov_b32 s81, 0x31113111
	s_mov_b32 s82, 0xb1113111
	s_mov_b32 s83, 0x13113111
	s_mov_b32 s84, 0xd3113111
	s_mov_b32 s85, 0xf5113111
	s_mov_b32 s86, 0x11313111
	s_mov_b32 s87, 0xd1313111
	v_writelane_b32 v40, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s87, 7
	s_mov_b32 s80, 0xf3313111
	s_mov_b32 s81, 0x33313111
	s_mov_b32 s82, 0xf1513111
	s_mov_b32 s83, 0x11133111
	s_mov_b32 s84, 0xd1133111
	s_mov_b32 s85, 0xf3133111
	s_mov_b32 s86, 0x33133111
	s_mov_b32 s87, 0xf1333111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s87, 15
	s_mov_b32 s80, 0x31333111
	s_mov_b32 s81, 0x13333111
	s_mov_b32 s82, 0xf1153111
	s_mov_b32 s83, 0x11115111
	s_mov_b32 s84, 0xd1115111
	s_mov_b32 s85, 0xf3115111
	s_mov_b32 s86, 0xf1315111
	s_mov_b32 s87, 0xf1135111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s87, 23
	s_mov_b32 s80, 0xf1111311
	s_mov_b32 s81, 0x31111311
	s_mov_b32 s82, 0xb1111311
	s_mov_b32 s83, 0x13111311
	s_mov_b32 s84, 0xd3111311
	s_mov_b32 s85, 0xf5111311
	s_mov_b32 s86, 0x11311311
	s_mov_b32 s87, 0xd1311311
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s87, 31
	ds_store_b32 v24, v40 offset:128
	s_mov_b32 s80, 0xf3311311
	s_mov_b32 s81, 0x33311311
	s_mov_b32 s82, 0xf1511311
	s_mov_b32 s83, 0x11131311
	s_mov_b32 s84, 0xd1131311
	s_mov_b32 s85, 0xf3131311
	s_mov_b32 s86, 0x33131311
	s_mov_b32 s87, 0xf1331311
	v_writelane_b32 v41, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s87, 7
	s_mov_b32 s80, 0x31331311
	s_mov_b32 s81, 0x13331311
	s_mov_b32 s82, 0xf1151311
	s_mov_b32 s83, 0x11113311
	s_mov_b32 s84, 0xd1113311
	s_mov_b32 s85, 0xf3113311
	s_mov_b32 s86, 0x33113311
	s_mov_b32 s87, 0xf1313311
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s87, 15
	s_mov_b32 s80, 0x31313311
	s_mov_b32 s81, 0x13313311
	s_mov_b32 s82, 0xf1133311
	s_mov_b32 s83, 0x31133311
	s_mov_b32 s84, 0x13133311
	s_mov_b32 s85, 0x11333311
	s_mov_b32 s86, 0xf1115311
	s_mov_b32 s87, 0x11111511
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s87, 23
	s_mov_b32 s80, 0xd1111511
	s_mov_b32 s81, 0xf3111511
	s_mov_b32 s82, 0xf1311511
	s_mov_b32 s83, 0xf1131511
	s_mov_b32 s84, 0xf1113511
	s_mov_b32 s85, 0xf1111131
	s_mov_b32 s86, 0x31111131
	s_mov_b32 s87, 0xb1111131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s87, 31
	ds_store_b32 v24, v41 offset:256
	s_mov_b32 s80, 0x13111131
	s_mov_b32 s81, 0xd3111131
	s_mov_b32 s82, 0xf5111131
	s_mov_b32 s83, 0x11311131
	s_mov_b32 s84, 0xd1311131
	s_mov_b32 s85, 0xf3311131
	s_mov_b32 s86, 0x33311131
	s_mov_b32 s87, 0xf1511131
	v_writelane_b32 v42, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s87, 7
	s_mov_b32 s80, 0x11131131
	s_mov_b32 s81, 0xd1131131
	s_mov_b32 s82, 0xf3131131
	s_mov_b32 s83, 0x33131131
	s_mov_b32 s84, 0xf1331131
	s_mov_b32 s85, 0x31331131
	s_mov_b32 s86, 0x13331131
	s_mov_b32 s87, 0xf1151131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s87, 15
	s_mov_b32 s80, 0x11113131
	s_mov_b32 s81, 0xd1113131
	s_mov_b32 s82, 0xf3113131
	s_mov_b32 s83, 0x33113131
	s_mov_b32 s84, 0xf1313131
	s_mov_b32 s85, 0x31313131
	s_mov_b32 s86, 0x13313131
	s_mov_b32 s87, 0xf1133131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s87, 23
	s_mov_b32 s80, 0x31133131
	s_mov_b32 s81, 0x13133131
	s_mov_b32 s82, 0x11333131
	s_mov_b32 s83, 0xf1115131
	s_mov_b32 s84, 0x11111331
	s_mov_b32 s85, 0xd1111331
	s_mov_b32 s86, 0xf3111331
	s_mov_b32 s87, 0x33111331
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s87, 31
	ds_store_b32 v24, v42 offset:384
	s_mov_b32 s80, 0xf1311331
	s_mov_b32 s81, 0x31311331
	s_mov_b32 s82, 0x13311331
	s_mov_b32 s83, 0xf1131331
	s_mov_b32 s84, 0x31131331
	s_mov_b32 s85, 0x13131331
	s_mov_b32 s86, 0x11331331
	s_mov_b32 s87, 0xf1113331
	v_writelane_b32 v43, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s87, 7
	s_mov_b32 s80, 0x31113331
	s_mov_b32 s81, 0x13113331
	s_mov_b32 s82, 0x11313331
	s_mov_b32 s83, 0x11133331
	s_mov_b32 s84, 0xf1111531
	s_mov_b32 s85, 0x11111151
	s_mov_b32 s86, 0xd1111151
	s_mov_b32 s87, 0xf3111151
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s87, 15
	s_mov_b32 s80, 0xf1311151
	s_mov_b32 s81, 0xf1131151
	s_mov_b32 s82, 0xf1113151
	s_mov_b32 s83, 0xf1111351
	s_mov_b32 s84, 0xf1111113
	s_mov_b32 s85, 0x31111113
	s_mov_b32 s86, 0xb1111113
	s_mov_b32 s87, 0x13111113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s87, 23
	s_mov_b32 s80, 0xd3111113
	s_mov_b32 s81, 0xf5111113
	s_mov_b32 s82, 0x11311113
	s_mov_b32 s83, 0xd1311113
	s_mov_b32 s84, 0xf3311113
	s_mov_b32 s85, 0x33311113
	s_mov_b32 s86, 0xf1511113
	s_mov_b32 s87, 0x11131113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s87, 31
	ds_store_b32 v24, v43 offset:512
	s_mov_b32 s80, 0xd1131113
	s_mov_b32 s81, 0xf3131113
	s_mov_b32 s82, 0x33131113
	s_mov_b32 s83, 0xf1331113
	s_mov_b32 s84, 0x31331113
	s_mov_b32 s85, 0x13331113
	s_mov_b32 s86, 0xf1151113
	s_mov_b32 s87, 0x11113113
	v_writelane_b32 v44, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s87, 7
	s_mov_b32 s80, 0xd1113113
	s_mov_b32 s81, 0xf3113113
	s_mov_b32 s82, 0x33113113
	s_mov_b32 s83, 0xf1313113
	s_mov_b32 s84, 0x31313113
	s_mov_b32 s85, 0x13313113
	s_mov_b32 s86, 0xf1133113
	s_mov_b32 s87, 0x31133113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s87, 15
	s_mov_b32 s80, 0x13133113
	s_mov_b32 s81, 0x11333113
	s_mov_b32 s82, 0xf1115113
	s_mov_b32 s83, 0x11111313
	s_mov_b32 s84, 0xd1111313
	s_mov_b32 s85, 0xf3111313
	s_mov_b32 s86, 0x33111313
	s_mov_b32 s87, 0xf1311313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s87, 23
	s_mov_b32 s80, 0x31311313
	s_mov_b32 s81, 0x13311313
	s_mov_b32 s82, 0xf1131313
	s_mov_b32 s83, 0x31131313
	s_mov_b32 s84, 0x13131313
	s_mov_b32 s85, 0x11331313
	s_mov_b32 s86, 0xf1113313
	s_mov_b32 s87, 0x31113313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s87, 31
	ds_store_b32 v24, v44 offset:640
	s_mov_b32 s80, 0x13113313
	s_mov_b32 s81, 0x11313313
	s_mov_b32 s82, 0x11133313
	s_mov_b32 s83, 0xf1111513
	s_mov_b32 s84, 0x11111133
	s_mov_b32 s85, 0xd1111133
	s_mov_b32 s86, 0xf3111133
	s_mov_b32 s87, 0x33111133
	v_writelane_b32 v45, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s87, 7
	s_mov_b32 s80, 0xf1311133
	s_mov_b32 s81, 0x31311133
	s_mov_b32 s82, 0x13311133
	s_mov_b32 s83, 0xf1131133
	s_mov_b32 s84, 0x31131133
	s_mov_b32 s85, 0x13131133
	s_mov_b32 s86, 0x11331133
	s_mov_b32 s87, 0xf1113133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s87, 15
	s_mov_b32 s80, 0x31113133
	s_mov_b32 s81, 0x13113133
	s_mov_b32 s82, 0x11313133
	s_mov_b32 s83, 0x11133133
	s_mov_b32 s84, 0xf1111333
	s_mov_b32 s85, 0x31111333
	s_mov_b32 s86, 0x13111333
	s_mov_b32 s87, 0x11311333
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s87, 23
	s_mov_b32 s80, 0x11131333
	s_mov_b32 s81, 0x11113333
	s_mov_b32 s82, 0xf1111153
	s_mov_b32 s83, 0x11111115
	s_mov_b32 s84, 0xd1111115
	s_mov_b32 s85, 0xf3111115
	s_mov_b32 s86, 0xf1311115
	s_mov_b32 s87, 0xf1131115
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s87, 31
	ds_store_b32 v24, v45 offset:768
	s_mov_b32 s80, 0xf1113115
	s_mov_b32 s81, 0xf1111315
	s_mov_b32 s82, 0xf1111135
	s_mov_b32 s83, 0xd3331113
	s_mov_b32 s84, 0xd3331131
	s_mov_b32 s85, 0xd3331311
	s_mov_b32 s86, 0xd3333111
	s_mov_b32 s87, 0xf1331333
	v_writelane_b32 v46, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s87, 7
	s_mov_b32 s80, 0xf3131333
	s_mov_b32 s81, 0xf3311333
	s_mov_b32 s82, 0xd1131333
	s_mov_b32 s83, 0xd1311333
	s_mov_b32 s84, 0xd3111333
	s_mov_b32 s85, 0xf1333133
	s_mov_b32 s86, 0xf3133133
	s_mov_b32 s87, 0xf3313133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s87, 15
	s_mov_b32 s80, 0xd1133133
	s_mov_b32 s81, 0xd1313133
	s_mov_b32 s82, 0xd3113133
	s_mov_b32 s83, 0xf1333313
	s_mov_b32 s84, 0xf3133313
	s_mov_b32 s85, 0xf3313313
	s_mov_b32 s86, 0xd1133313
	s_mov_b32 s87, 0xd1313313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s87, 23
	s_mov_b32 s80, 0xd3113331
	s_mov_b32 s81, 0xf1333331
	s_mov_b32 s82, 0xf3133331
	s_mov_b32 s83, 0xf3313331
	s_mov_b32 s84, 0xd1133331
	s_mov_b32 s85, 0xd1313331
	s_mov_b32 s86, 0xd3313311
	s_mov_b32 s87, 0xf3331133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s87, 31
	ds_store_b32 v24, v46 offset:896
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b32 s54, s48, -4
	s_cmp_eq_u32 s54, 0
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_tail
	.Le8pi4moe_r256_gate_up_quad:
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_q0_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q0_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v17
	.Le8pi4moe_r256_gate_up_q0_done:
	s_add_i32 s53, s53, 1
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_q1_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v85, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q1_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v93, v93, v17
	.Le8pi4moe_r256_gate_up_q1_done:
	s_add_i32 s53, s53, 1
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_q2_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78, v78, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v82, v82, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v86, v86, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v90, v90, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q2_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v94, v94, v17
	.Le8pi4moe_r256_gate_up_q2_done:
	s_add_i32 s53, s53, 1
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_q3_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v79, v79, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v87, v87, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_q3_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v17
	.Le8pi4moe_r256_gate_up_q3_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s54
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_quad
	.Le8pi4moe_r256_gate_up_tail:
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_reduce
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t0_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t0_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v17
	.Le8pi4moe_r256_gate_up_t0_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_reduce
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t1_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v85, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t1_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v93, v93, v17
	.Le8pi4moe_r256_gate_up_t1_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_reduce
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t2_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78, v78, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v82, v82, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v86, v86, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v90, v90, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t2_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v94, v94, v17
	.Le8pi4moe_r256_gate_up_t2_done:
	s_add_i32 s53, s53, 1
	.Le8pi4moe_r256_gate_up_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v19, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v19, v19, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	v_cndmask_b32_e64 v20, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v20, v20, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v21, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v21, v21, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	v_cndmask_b32_e64 v22, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v22, v22, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	v_cndmask_b32_e64 v23, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v23, v23, v0, 2
	v_add_f32_e32 v64, v64, v65
	v_add_f32_e32 v66, v66, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v66
	ds_bpermute_b32 v18, v19, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s68
	ds_bpermute_b32 v18, v20, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s69
	ds_bpermute_b32 v18, v21, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s70
	ds_bpermute_b32 v18, v22, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s71
	ds_bpermute_b32 v18, v23, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_red_done
	v_add_f32_e32 v68, v68, v69
	v_add_f32_e32 v70, v70, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v70
	ds_bpermute_b32 v18, v19, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s68
	ds_bpermute_b32 v18, v20, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s69
	ds_bpermute_b32 v18, v21, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s70
	ds_bpermute_b32 v18, v22, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s71
	ds_bpermute_b32 v18, v23, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_red_done
	v_add_f32_e32 v72, v72, v73
	v_add_f32_e32 v74, v74, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v74
	ds_bpermute_b32 v18, v19, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s68
	ds_bpermute_b32 v18, v20, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s69
	ds_bpermute_b32 v18, v21, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s70
	ds_bpermute_b32 v18, v22, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s71
	ds_bpermute_b32 v18, v23, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_red_done
	v_add_f32_e32 v76, v76, v77
	v_add_f32_e32 v78, v78, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v78
	ds_bpermute_b32 v18, v19, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s68
	ds_bpermute_b32 v18, v20, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s69
	ds_bpermute_b32 v18, v21, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s70
	ds_bpermute_b32 v18, v22, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s71
	ds_bpermute_b32 v18, v23, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_red_done
	v_add_f32_e32 v80, v80, v81
	v_add_f32_e32 v82, v82, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v82
	ds_bpermute_b32 v18, v19, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s68
	ds_bpermute_b32 v18, v20, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s69
	ds_bpermute_b32 v18, v21, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s70
	ds_bpermute_b32 v18, v22, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s71
	ds_bpermute_b32 v18, v23, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_red_done
	v_add_f32_e32 v84, v84, v85
	v_add_f32_e32 v86, v86, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v86
	ds_bpermute_b32 v18, v19, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s68
	ds_bpermute_b32 v18, v20, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s69
	ds_bpermute_b32 v18, v21, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s70
	ds_bpermute_b32 v18, v22, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s71
	ds_bpermute_b32 v18, v23, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_red_done
	v_add_f32_e32 v88, v88, v89
	v_add_f32_e32 v90, v90, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v90
	ds_bpermute_b32 v18, v19, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s68
	ds_bpermute_b32 v18, v20, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s69
	ds_bpermute_b32 v18, v21, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s70
	ds_bpermute_b32 v18, v22, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s71
	ds_bpermute_b32 v18, v23, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_red_done
	v_add_f32_e32 v92, v92, v93
	v_add_f32_e32 v94, v94, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v94
	ds_bpermute_b32 v18, v19, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s68
	ds_bpermute_b32 v18, v20, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s69
	ds_bpermute_b32 v18, v21, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s70
	ds_bpermute_b32 v18, v22, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s71
	ds_bpermute_b32 v18, v23, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s72
	.Le8pi4moe_r256_gate_up_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v104, v64, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_st_done
	global_store_b32 v105, v68, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_st_done
	global_store_b32 v106, v72, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_st_done
	global_store_b32 v107, v76, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_st_done
	global_store_b32 v108, v80, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_st_done
	global_store_b32 v109, v84, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_st_done
	global_store_b32 v110, v88, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_st_done
	global_store_b32 v111, v92, s[58:59]
	.Le8pi4moe_r256_gate_up_st_done:
	s_waitcnt_vscnt null, 0x0
	.Le8pi4moe_r256_gate_up_exit:
	s_endpgm
.Le8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_end:
.size e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100, .Le8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_end-e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100
	.amdhsa_group_segment_fixed_size 1536
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_sgpr 96
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Le8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_end-e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100)<<4)&1008)>>4
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
.protected e8p_i4_moe_gemv_r128_down_pm_gfx1100
.globl e8p_i4_moe_gemv_r128_down_pm_gfx1100
.p2align 8
.type e8p_i4_moe_gemv_r128_down_pm_gfx1100,@function
e8p_i4_moe_gemv_r128_down_pm_gfx1100:
	v_lshlrev_b32_e32 v24, 2, v0
	s_load_b256 s[88:95], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s4, s[0:1], 0x58
	s_mov_b32 s30, s14
	s_mov_b32 s31, s15
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Le8pi4moe_r128_down_exit
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Le8pi4moe_r128_down_exit
	s_lshl_b32 s51, s31, 2
	s_add_u32 s36, s92, s51
	s_addc_u32 s37, s93, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s90, s51
	s_addc_u32 s37, s91, 0
	s_load_b32 s34, s[36:37], null
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Le8pi4moe_r128_down_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Le8pi4moe_r128_down_exit
	s_mul_i32 s51, s25, 10
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Le8pi4moe_r128_down_exit
	s_cmp_ge_u32 s34, s4
	s_cbranch_scc1 .Le8pi4moe_r128_down_exit
	s_lshl_b32 s51, s34, 3
	s_add_u32 s36, s88, s51
	s_addc_u32 s37, s89, 0
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
	v_and_b32_e32 v4, 30, v0
	v_and_b32_e32 v5, 1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 4, v5
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_min_u32_e32 v39, s63, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, s32, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 2, v39
	global_load_b32 v32, v39, s[94:95]
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	s_waitcnt vmcnt(0)
	v_cvt_f32_u32_e32 v34, v32
	s_waitcnt_depctr depctr_va_vdst(0)
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
	v_mov_b32_e32 v72, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v38, 0
	v_mov_b32_e32 v80, s73
	v_readlane_b32 s74, v33, 1
	v_mov_b32_e32 v73, s74
	v_readlane_b32 s74, v38, 1
	v_mov_b32_e32 v81, s74
	v_readlane_b32 s75, v33, 2
	v_mov_b32_e32 v74, s75
	v_readlane_b32 s75, v38, 2
	v_mov_b32_e32 v82, s75
	v_readlane_b32 s76, v33, 3
	v_mov_b32_e32 v75, s76
	v_readlane_b32 s76, v38, 3
	v_mov_b32_e32 v83, s76
	v_readlane_b32 s73, v33, 4
	v_mov_b32_e32 v76, s73
	v_readlane_b32 s73, v38, 4
	v_mov_b32_e32 v84, s73
	v_readlane_b32 s74, v33, 5
	v_mov_b32_e32 v77, s74
	v_readlane_b32 s74, v38, 5
	v_mov_b32_e32 v85, s74
	v_readlane_b32 s75, v33, 6
	v_mov_b32_e32 v78, s75
	v_readlane_b32 s75, v38, 6
	v_mov_b32_e32 v86, s75
	v_readlane_b32 s76, v33, 7
	v_mov_b32_e32 v79, s76
	v_readlane_b32 s76, v38, 7
	v_mov_b32_e32 v87, s76
	v_lshrrev_b32_e32 v34, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v0, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v35, v35, v34
	ds_store_b32 v24, v35 offset:1024
	v_add_nc_u32_e32 v32, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 4, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v32, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v36, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v36, v36, v34
	ds_store_b32 v24, v36 offset:1152
	v_add_nc_u32_e32 v32, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 4, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v32, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v37, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v37, v37, v34
	ds_store_b32 v24, v37 offset:1280
	v_add_nc_u32_e32 v32, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 4, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v32, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v38, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v38, v38, v34
	ds_store_b32 v24, v38 offset:1408
	s_mov_b32 s80, 0x11111111
	s_mov_b32 s81, 0xd1111111
	s_mov_b32 s82, 0x51111111
	s_mov_b32 s83, 0xf3111111
	s_mov_b32 s84, 0x33111111
	s_mov_b32 s85, 0xb3111111
	s_mov_b32 s86, 0x15111111
	s_mov_b32 s87, 0xd5111111
	v_writelane_b32 v39, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s87, 7
	s_mov_b32 s80, 0xf1311111
	s_mov_b32 s81, 0x31311111
	s_mov_b32 s82, 0xb1311111
	s_mov_b32 s83, 0x13311111
	s_mov_b32 s84, 0xd3311111
	s_mov_b32 s85, 0xf5311111
	s_mov_b32 s86, 0x11511111
	s_mov_b32 s87, 0xd1511111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s87, 15
	s_mov_b32 s80, 0xf3511111
	s_mov_b32 s81, 0xf1131111
	s_mov_b32 s82, 0x31131111
	s_mov_b32 s83, 0xb1131111
	s_mov_b32 s84, 0x13131111
	s_mov_b32 s85, 0xd3131111
	s_mov_b32 s86, 0xf5131111
	s_mov_b32 s87, 0x11331111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s87, 23
	s_mov_b32 s80, 0xd1331111
	s_mov_b32 s81, 0xf3331111
	s_mov_b32 s82, 0x33331111
	s_mov_b32 s83, 0xf1531111
	s_mov_b32 s84, 0x11151111
	s_mov_b32 s85, 0xd1151111
	s_mov_b32 s86, 0xf3151111
	s_mov_b32 s87, 0xf1351111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v39, s87, 31
	ds_store_b32 v24, v39
	s_mov_b32 s80, 0xf1113111
	s_mov_b32 s81, 0x31113111
	s_mov_b32 s82, 0xb1113111
	s_mov_b32 s83, 0x13113111
	s_mov_b32 s84, 0xd3113111
	s_mov_b32 s85, 0xf5113111
	s_mov_b32 s86, 0x11313111
	s_mov_b32 s87, 0xd1313111
	v_writelane_b32 v40, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s87, 7
	s_mov_b32 s80, 0xf3313111
	s_mov_b32 s81, 0x33313111
	s_mov_b32 s82, 0xf1513111
	s_mov_b32 s83, 0x11133111
	s_mov_b32 s84, 0xd1133111
	s_mov_b32 s85, 0xf3133111
	s_mov_b32 s86, 0x33133111
	s_mov_b32 s87, 0xf1333111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s87, 15
	s_mov_b32 s80, 0x31333111
	s_mov_b32 s81, 0x13333111
	s_mov_b32 s82, 0xf1153111
	s_mov_b32 s83, 0x11115111
	s_mov_b32 s84, 0xd1115111
	s_mov_b32 s85, 0xf3115111
	s_mov_b32 s86, 0xf1315111
	s_mov_b32 s87, 0xf1135111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s87, 23
	s_mov_b32 s80, 0xf1111311
	s_mov_b32 s81, 0x31111311
	s_mov_b32 s82, 0xb1111311
	s_mov_b32 s83, 0x13111311
	s_mov_b32 s84, 0xd3111311
	s_mov_b32 s85, 0xf5111311
	s_mov_b32 s86, 0x11311311
	s_mov_b32 s87, 0xd1311311
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v40, s87, 31
	ds_store_b32 v24, v40 offset:128
	s_mov_b32 s80, 0xf3311311
	s_mov_b32 s81, 0x33311311
	s_mov_b32 s82, 0xf1511311
	s_mov_b32 s83, 0x11131311
	s_mov_b32 s84, 0xd1131311
	s_mov_b32 s85, 0xf3131311
	s_mov_b32 s86, 0x33131311
	s_mov_b32 s87, 0xf1331311
	v_writelane_b32 v41, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s87, 7
	s_mov_b32 s80, 0x31331311
	s_mov_b32 s81, 0x13331311
	s_mov_b32 s82, 0xf1151311
	s_mov_b32 s83, 0x11113311
	s_mov_b32 s84, 0xd1113311
	s_mov_b32 s85, 0xf3113311
	s_mov_b32 s86, 0x33113311
	s_mov_b32 s87, 0xf1313311
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s87, 15
	s_mov_b32 s80, 0x31313311
	s_mov_b32 s81, 0x13313311
	s_mov_b32 s82, 0xf1133311
	s_mov_b32 s83, 0x31133311
	s_mov_b32 s84, 0x13133311
	s_mov_b32 s85, 0x11333311
	s_mov_b32 s86, 0xf1115311
	s_mov_b32 s87, 0x11111511
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s87, 23
	s_mov_b32 s80, 0xd1111511
	s_mov_b32 s81, 0xf3111511
	s_mov_b32 s82, 0xf1311511
	s_mov_b32 s83, 0xf1131511
	s_mov_b32 s84, 0xf1113511
	s_mov_b32 s85, 0xf1111131
	s_mov_b32 s86, 0x31111131
	s_mov_b32 s87, 0xb1111131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v41, s87, 31
	ds_store_b32 v24, v41 offset:256
	s_mov_b32 s80, 0x13111131
	s_mov_b32 s81, 0xd3111131
	s_mov_b32 s82, 0xf5111131
	s_mov_b32 s83, 0x11311131
	s_mov_b32 s84, 0xd1311131
	s_mov_b32 s85, 0xf3311131
	s_mov_b32 s86, 0x33311131
	s_mov_b32 s87, 0xf1511131
	v_writelane_b32 v42, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s87, 7
	s_mov_b32 s80, 0x11131131
	s_mov_b32 s81, 0xd1131131
	s_mov_b32 s82, 0xf3131131
	s_mov_b32 s83, 0x33131131
	s_mov_b32 s84, 0xf1331131
	s_mov_b32 s85, 0x31331131
	s_mov_b32 s86, 0x13331131
	s_mov_b32 s87, 0xf1151131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s87, 15
	s_mov_b32 s80, 0x11113131
	s_mov_b32 s81, 0xd1113131
	s_mov_b32 s82, 0xf3113131
	s_mov_b32 s83, 0x33113131
	s_mov_b32 s84, 0xf1313131
	s_mov_b32 s85, 0x31313131
	s_mov_b32 s86, 0x13313131
	s_mov_b32 s87, 0xf1133131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s87, 23
	s_mov_b32 s80, 0x31133131
	s_mov_b32 s81, 0x13133131
	s_mov_b32 s82, 0x11333131
	s_mov_b32 s83, 0xf1115131
	s_mov_b32 s84, 0x11111331
	s_mov_b32 s85, 0xd1111331
	s_mov_b32 s86, 0xf3111331
	s_mov_b32 s87, 0x33111331
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v42, s87, 31
	ds_store_b32 v24, v42 offset:384
	s_mov_b32 s80, 0xf1311331
	s_mov_b32 s81, 0x31311331
	s_mov_b32 s82, 0x13311331
	s_mov_b32 s83, 0xf1131331
	s_mov_b32 s84, 0x31131331
	s_mov_b32 s85, 0x13131331
	s_mov_b32 s86, 0x11331331
	s_mov_b32 s87, 0xf1113331
	v_writelane_b32 v43, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s87, 7
	s_mov_b32 s80, 0x31113331
	s_mov_b32 s81, 0x13113331
	s_mov_b32 s82, 0x11313331
	s_mov_b32 s83, 0x11133331
	s_mov_b32 s84, 0xf1111531
	s_mov_b32 s85, 0x11111151
	s_mov_b32 s86, 0xd1111151
	s_mov_b32 s87, 0xf3111151
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s87, 15
	s_mov_b32 s80, 0xf1311151
	s_mov_b32 s81, 0xf1131151
	s_mov_b32 s82, 0xf1113151
	s_mov_b32 s83, 0xf1111351
	s_mov_b32 s84, 0xf1111113
	s_mov_b32 s85, 0x31111113
	s_mov_b32 s86, 0xb1111113
	s_mov_b32 s87, 0x13111113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s87, 23
	s_mov_b32 s80, 0xd3111113
	s_mov_b32 s81, 0xf5111113
	s_mov_b32 s82, 0x11311113
	s_mov_b32 s83, 0xd1311113
	s_mov_b32 s84, 0xf3311113
	s_mov_b32 s85, 0x33311113
	s_mov_b32 s86, 0xf1511113
	s_mov_b32 s87, 0x11131113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v43, s87, 31
	ds_store_b32 v24, v43 offset:512
	s_mov_b32 s80, 0xd1131113
	s_mov_b32 s81, 0xf3131113
	s_mov_b32 s82, 0x33131113
	s_mov_b32 s83, 0xf1331113
	s_mov_b32 s84, 0x31331113
	s_mov_b32 s85, 0x13331113
	s_mov_b32 s86, 0xf1151113
	s_mov_b32 s87, 0x11113113
	v_writelane_b32 v44, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s87, 7
	s_mov_b32 s80, 0xd1113113
	s_mov_b32 s81, 0xf3113113
	s_mov_b32 s82, 0x33113113
	s_mov_b32 s83, 0xf1313113
	s_mov_b32 s84, 0x31313113
	s_mov_b32 s85, 0x13313113
	s_mov_b32 s86, 0xf1133113
	s_mov_b32 s87, 0x31133113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s87, 15
	s_mov_b32 s80, 0x13133113
	s_mov_b32 s81, 0x11333113
	s_mov_b32 s82, 0xf1115113
	s_mov_b32 s83, 0x11111313
	s_mov_b32 s84, 0xd1111313
	s_mov_b32 s85, 0xf3111313
	s_mov_b32 s86, 0x33111313
	s_mov_b32 s87, 0xf1311313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s87, 23
	s_mov_b32 s80, 0x31311313
	s_mov_b32 s81, 0x13311313
	s_mov_b32 s82, 0xf1131313
	s_mov_b32 s83, 0x31131313
	s_mov_b32 s84, 0x13131313
	s_mov_b32 s85, 0x11331313
	s_mov_b32 s86, 0xf1113313
	s_mov_b32 s87, 0x31113313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v44, s87, 31
	ds_store_b32 v24, v44 offset:640
	s_mov_b32 s80, 0x13113313
	s_mov_b32 s81, 0x11313313
	s_mov_b32 s82, 0x11133313
	s_mov_b32 s83, 0xf1111513
	s_mov_b32 s84, 0x11111133
	s_mov_b32 s85, 0xd1111133
	s_mov_b32 s86, 0xf3111133
	s_mov_b32 s87, 0x33111133
	v_writelane_b32 v45, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s87, 7
	s_mov_b32 s80, 0xf1311133
	s_mov_b32 s81, 0x31311133
	s_mov_b32 s82, 0x13311133
	s_mov_b32 s83, 0xf1131133
	s_mov_b32 s84, 0x31131133
	s_mov_b32 s85, 0x13131133
	s_mov_b32 s86, 0x11331133
	s_mov_b32 s87, 0xf1113133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s87, 15
	s_mov_b32 s80, 0x31113133
	s_mov_b32 s81, 0x13113133
	s_mov_b32 s82, 0x11313133
	s_mov_b32 s83, 0x11133133
	s_mov_b32 s84, 0xf1111333
	s_mov_b32 s85, 0x31111333
	s_mov_b32 s86, 0x13111333
	s_mov_b32 s87, 0x11311333
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s87, 23
	s_mov_b32 s80, 0x11131333
	s_mov_b32 s81, 0x11113333
	s_mov_b32 s82, 0xf1111153
	s_mov_b32 s83, 0x11111115
	s_mov_b32 s84, 0xd1111115
	s_mov_b32 s85, 0xf3111115
	s_mov_b32 s86, 0xf1311115
	s_mov_b32 s87, 0xf1131115
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v45, s87, 31
	ds_store_b32 v24, v45 offset:768
	s_mov_b32 s80, 0xf1113115
	s_mov_b32 s81, 0xf1111315
	s_mov_b32 s82, 0xf1111135
	s_mov_b32 s83, 0xd3331113
	s_mov_b32 s84, 0xd3331131
	s_mov_b32 s85, 0xd3331311
	s_mov_b32 s86, 0xd3333111
	s_mov_b32 s87, 0xf1331333
	v_writelane_b32 v46, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s87, 7
	s_mov_b32 s80, 0xf3131333
	s_mov_b32 s81, 0xf3311333
	s_mov_b32 s82, 0xd1131333
	s_mov_b32 s83, 0xd1311333
	s_mov_b32 s84, 0xd3111333
	s_mov_b32 s85, 0xf1333133
	s_mov_b32 s86, 0xf3133133
	s_mov_b32 s87, 0xf3313133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s87, 15
	s_mov_b32 s80, 0xd1133133
	s_mov_b32 s81, 0xd1313133
	s_mov_b32 s82, 0xd3113133
	s_mov_b32 s83, 0xf1333313
	s_mov_b32 s84, 0xf3133313
	s_mov_b32 s85, 0xf3313313
	s_mov_b32 s86, 0xd1133313
	s_mov_b32 s87, 0xd1313313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s87, 23
	s_mov_b32 s80, 0xd3113331
	s_mov_b32 s81, 0xf1333331
	s_mov_b32 s82, 0xf3133331
	s_mov_b32 s83, 0xf3313331
	s_mov_b32 s84, 0xd1133331
	s_mov_b32 s85, 0xd1313331
	s_mov_b32 s86, 0xd3313311
	s_mov_b32 s87, 0xf3331133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v46, s87, 31
	ds_store_b32 v24, v46 offset:896
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s48, 0
	s_cbranch_scc1 .Le8pi4moe_r128_down_reduce
	.Le8pi4moe_r128_down_group:
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v72, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v73, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v74, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v75, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v76, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v77, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v78, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_xdone
	v_add_nc_u32_e32 v7, v79, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r128_down_g_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_g_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v17
	.Le8pi4moe_r128_down_g_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc1 .Le8pi4moe_r128_down_group
	.Le8pi4moe_r128_down_reduce:
	v_cmp_gt_u32_e64 s68, 16, v0
	v_cndmask_b32_e64 v19, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v19, v19, v0, 2
	v_cmp_gt_u32_e64 s69, 24, v0
	v_cndmask_b32_e64 v20, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v20, v20, v0, 2
	v_cmp_gt_u32_e64 s70, 28, v0
	v_cndmask_b32_e64 v21, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v21, v21, v0, 2
	v_cmp_gt_u32_e64 s71, 30, v0
	v_cndmask_b32_e64 v22, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v22, v22, v0, 2
	v_cmp_gt_u32_e64 s72, 31, v0
	v_cndmask_b32_e64 v23, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v23, v23, v0, 2
	ds_bpermute_b32 v18, v19, v64
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s68
	ds_bpermute_b32 v18, v20, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s69
	ds_bpermute_b32 v18, v21, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s70
	ds_bpermute_b32 v18, v22, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s71
	ds_bpermute_b32 v18, v23, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_red_done
	ds_bpermute_b32 v18, v19, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s68
	ds_bpermute_b32 v18, v20, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s69
	ds_bpermute_b32 v18, v21, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s70
	ds_bpermute_b32 v18, v22, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s71
	ds_bpermute_b32 v18, v23, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_red_done
	ds_bpermute_b32 v18, v19, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s68
	ds_bpermute_b32 v18, v20, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s69
	ds_bpermute_b32 v18, v21, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s70
	ds_bpermute_b32 v18, v22, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s71
	ds_bpermute_b32 v18, v23, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_red_done
	ds_bpermute_b32 v18, v19, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s68
	ds_bpermute_b32 v18, v20, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s69
	ds_bpermute_b32 v18, v21, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s70
	ds_bpermute_b32 v18, v22, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s71
	ds_bpermute_b32 v18, v23, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_red_done
	ds_bpermute_b32 v18, v19, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s68
	ds_bpermute_b32 v18, v20, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s69
	ds_bpermute_b32 v18, v21, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s70
	ds_bpermute_b32 v18, v22, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s71
	ds_bpermute_b32 v18, v23, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_red_done
	ds_bpermute_b32 v18, v19, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s68
	ds_bpermute_b32 v18, v20, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s69
	ds_bpermute_b32 v18, v21, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s70
	ds_bpermute_b32 v18, v22, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s71
	ds_bpermute_b32 v18, v23, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_red_done
	ds_bpermute_b32 v18, v19, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s68
	ds_bpermute_b32 v18, v20, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s69
	ds_bpermute_b32 v18, v21, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s70
	ds_bpermute_b32 v18, v22, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s71
	ds_bpermute_b32 v18, v23, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_red_done
	ds_bpermute_b32 v18, v19, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s68
	ds_bpermute_b32 v18, v20, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s69
	ds_bpermute_b32 v18, v21, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s70
	ds_bpermute_b32 v18, v22, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s71
	ds_bpermute_b32 v18, v23, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s72
	.Le8pi4moe_r128_down_red_done:
	v_cmpx_eq_u32_e32 0, v0
	global_store_b32 v80, v64, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_st_done
	global_store_b32 v81, v65, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_st_done
	global_store_b32 v82, v66, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_st_done
	global_store_b32 v83, v67, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_st_done
	global_store_b32 v84, v68, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_st_done
	global_store_b32 v85, v69, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_st_done
	global_store_b32 v86, v70, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_st_done
	global_store_b32 v87, v71, s[58:59]
	.Le8pi4moe_r128_down_st_done:
	s_waitcnt_vscnt null, 0x0
	.Le8pi4moe_r128_down_exit:
	s_endpgm
.Le8p_i4_moe_gemv_r128_down_pm_gfx1100_end:
.size e8p_i4_moe_gemv_r128_down_pm_gfx1100, .Le8p_i4_moe_gemv_r128_down_pm_gfx1100_end-e8p_i4_moe_gemv_r128_down_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel e8p_i4_moe_gemv_r128_down_pm_gfx1100
	.amdhsa_group_segment_fixed_size 1536
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
	.amdhsa_user_sgpr_count 14
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
	.amdhsa_next_free_sgpr 96
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Le8p_i4_moe_gemv_r128_down_pm_gfx1100_end-e8p_i4_moe_gemv_r128_down_pm_gfx1100)<<4)&1008)>>4
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
.protected e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8
.globl e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8
.p2align 8
.type e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8,@function
e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8:
	v_and_b32_e32 v24, 0x7f, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 2, v24
	v_and_b32_e32 v7, 31, v0
	v_lshrrev_b32_e32 v6, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s5, v6
	s_load_b256 s[88:95], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s6, s[0:1], 0x58
	s_lshl_b32 s30, s13, 3
	s_add_i32 s30, s30, s5
	s_mov_b32 s31, s14
	s_mov_b32 s7, s15
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_cmp_eq_u32 s24, 0x50
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_exit
	s_lshl_b32 s51, s7, 3
	s_cmp_ge_u32 s51, s25
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_and_b32 s51, s30, -8
	s_cmp_ge_u32 s51, s22
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_mul_i32 s51, s7, 0x788
	s_add_u32 s90, s90, s51
	s_addc_u32 s91, s91, 0
	s_add_u32 s92, s92, s51
	s_addc_u32 s93, s93, 0
	s_add_u32 s94, s94, s51
	s_addc_u32 s95, s95, 0
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_lshl_b32 s51, s31, 2
	s_add_u32 s36, s92, s51
	s_addc_u32 s37, s93, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s90, s51
	s_addc_u32 s37, s91, 0
	s_load_b32 s34, s[36:37], null
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_mov_b32 s51, 0x50
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_cmp_ge_u32 s34, s6
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_lshl_b32 s51, s34, 3
	s_add_u32 s36, s88, s51
	s_addc_u32 s37, s89, 0
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
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v3, 4, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v4, 30, v7
	v_and_b32_e32 v5, 1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 4, v5
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
	v_min_u32_e32 v39, s63, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, s32, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 2, v39
	global_load_b32 v32, v39, s[94:95]
	s_mul_i32 s52, s25, 10
	s_waitcnt vmcnt(0)
	v_readlane_b32 s51, v32, 0
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	v_readlane_b32 s51, v32, 1
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	v_readlane_b32 s51, v32, 2
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	v_readlane_b32 s51, v32, 3
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	v_readlane_b32 s51, v32, 4
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	v_readlane_b32 s51, v32, 5
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	v_readlane_b32 s51, v32, 6
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	v_readlane_b32 s51, v32, 7
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	v_cvt_f32_u32_e32 v34, v32
	s_waitcnt_depctr depctr_va_vdst(0)
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
	v_mov_b32_e32 v96, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v38, 0
	v_mov_b32_e32 v104, s73
	v_readlane_b32 s74, v33, 1
	v_mov_b32_e32 v97, s74
	v_readlane_b32 s74, v38, 1
	v_mov_b32_e32 v105, s74
	v_readlane_b32 s75, v33, 2
	v_mov_b32_e32 v98, s75
	v_readlane_b32 s75, v38, 2
	v_mov_b32_e32 v106, s75
	v_readlane_b32 s76, v33, 3
	v_mov_b32_e32 v99, s76
	v_readlane_b32 s76, v38, 3
	v_mov_b32_e32 v107, s76
	v_readlane_b32 s73, v33, 4
	v_mov_b32_e32 v100, s73
	v_readlane_b32 s73, v38, 4
	v_mov_b32_e32 v108, s73
	v_readlane_b32 s74, v33, 5
	v_mov_b32_e32 v101, s74
	v_readlane_b32 s74, v38, 5
	v_mov_b32_e32 v109, s74
	v_readlane_b32 s75, v33, 6
	v_mov_b32_e32 v102, s75
	v_readlane_b32 s75, v38, 6
	v_mov_b32_e32 v110, s75
	v_readlane_b32 s76, v33, 7
	v_mov_b32_e32 v103, s76
	v_readlane_b32 s76, v38, 7
	v_mov_b32_e32 v111, s76
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_lut0_skip
	s_mov_b32 s80, 0x11111111
	s_mov_b32 s81, 0xd1111111
	s_mov_b32 s82, 0x51111111
	s_mov_b32 s83, 0xf3111111
	s_mov_b32 s84, 0x33111111
	s_mov_b32 s85, 0xb3111111
	s_mov_b32 s86, 0x15111111
	s_mov_b32 s87, 0xd5111111
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xf1311111
	s_mov_b32 s81, 0x31311111
	s_mov_b32 s82, 0xb1311111
	s_mov_b32 s83, 0x13311111
	s_mov_b32 s84, 0xd3311111
	s_mov_b32 s85, 0xf5311111
	s_mov_b32 s86, 0x11511111
	s_mov_b32 s87, 0xd1511111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0xf3511111
	s_mov_b32 s81, 0xf1131111
	s_mov_b32 s82, 0x31131111
	s_mov_b32 s83, 0xb1131111
	s_mov_b32 s84, 0x13131111
	s_mov_b32 s85, 0xd3131111
	s_mov_b32 s86, 0xf5131111
	s_mov_b32 s87, 0x11331111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xd1331111
	s_mov_b32 s81, 0xf3331111
	s_mov_b32 s82, 0x33331111
	s_mov_b32 s83, 0xf1531111
	s_mov_b32 s84, 0x11151111
	s_mov_b32 s85, 0xd1151111
	s_mov_b32 s86, 0xf3151111
	s_mov_b32 s87, 0xf1351111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r256_gate_up_t8_lut0_skip:
	s_cmp_eq_u32 s5, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_lut1_skip
	s_mov_b32 s80, 0xf1113111
	s_mov_b32 s81, 0x31113111
	s_mov_b32 s82, 0xb1113111
	s_mov_b32 s83, 0x13113111
	s_mov_b32 s84, 0xd3113111
	s_mov_b32 s85, 0xf5113111
	s_mov_b32 s86, 0x11313111
	s_mov_b32 s87, 0xd1313111
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xf3313111
	s_mov_b32 s81, 0x33313111
	s_mov_b32 s82, 0xf1513111
	s_mov_b32 s83, 0x11133111
	s_mov_b32 s84, 0xd1133111
	s_mov_b32 s85, 0xf3133111
	s_mov_b32 s86, 0x33133111
	s_mov_b32 s87, 0xf1333111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x31333111
	s_mov_b32 s81, 0x13333111
	s_mov_b32 s82, 0xf1153111
	s_mov_b32 s83, 0x11115111
	s_mov_b32 s84, 0xd1115111
	s_mov_b32 s85, 0xf3115111
	s_mov_b32 s86, 0xf1315111
	s_mov_b32 s87, 0xf1135111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xf1111311
	s_mov_b32 s81, 0x31111311
	s_mov_b32 s82, 0xb1111311
	s_mov_b32 s83, 0x13111311
	s_mov_b32 s84, 0xd3111311
	s_mov_b32 s85, 0xf5111311
	s_mov_b32 s86, 0x11311311
	s_mov_b32 s87, 0xd1311311
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r256_gate_up_t8_lut1_skip:
	s_cmp_eq_u32 s5, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_lut2_skip
	s_mov_b32 s80, 0xf3311311
	s_mov_b32 s81, 0x33311311
	s_mov_b32 s82, 0xf1511311
	s_mov_b32 s83, 0x11131311
	s_mov_b32 s84, 0xd1131311
	s_mov_b32 s85, 0xf3131311
	s_mov_b32 s86, 0x33131311
	s_mov_b32 s87, 0xf1331311
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0x31331311
	s_mov_b32 s81, 0x13331311
	s_mov_b32 s82, 0xf1151311
	s_mov_b32 s83, 0x11113311
	s_mov_b32 s84, 0xd1113311
	s_mov_b32 s85, 0xf3113311
	s_mov_b32 s86, 0x33113311
	s_mov_b32 s87, 0xf1313311
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x31313311
	s_mov_b32 s81, 0x13313311
	s_mov_b32 s82, 0xf1133311
	s_mov_b32 s83, 0x31133311
	s_mov_b32 s84, 0x13133311
	s_mov_b32 s85, 0x11333311
	s_mov_b32 s86, 0xf1115311
	s_mov_b32 s87, 0x11111511
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xd1111511
	s_mov_b32 s81, 0xf3111511
	s_mov_b32 s82, 0xf1311511
	s_mov_b32 s83, 0xf1131511
	s_mov_b32 s84, 0xf1113511
	s_mov_b32 s85, 0xf1111131
	s_mov_b32 s86, 0x31111131
	s_mov_b32 s87, 0xb1111131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r256_gate_up_t8_lut2_skip:
	s_cmp_eq_u32 s5, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_lut3_skip
	s_mov_b32 s80, 0x13111131
	s_mov_b32 s81, 0xd3111131
	s_mov_b32 s82, 0xf5111131
	s_mov_b32 s83, 0x11311131
	s_mov_b32 s84, 0xd1311131
	s_mov_b32 s85, 0xf3311131
	s_mov_b32 s86, 0x33311131
	s_mov_b32 s87, 0xf1511131
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0x11131131
	s_mov_b32 s81, 0xd1131131
	s_mov_b32 s82, 0xf3131131
	s_mov_b32 s83, 0x33131131
	s_mov_b32 s84, 0xf1331131
	s_mov_b32 s85, 0x31331131
	s_mov_b32 s86, 0x13331131
	s_mov_b32 s87, 0xf1151131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x11113131
	s_mov_b32 s81, 0xd1113131
	s_mov_b32 s82, 0xf3113131
	s_mov_b32 s83, 0x33113131
	s_mov_b32 s84, 0xf1313131
	s_mov_b32 s85, 0x31313131
	s_mov_b32 s86, 0x13313131
	s_mov_b32 s87, 0xf1133131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0x31133131
	s_mov_b32 s81, 0x13133131
	s_mov_b32 s82, 0x11333131
	s_mov_b32 s83, 0xf1115131
	s_mov_b32 s84, 0x11111331
	s_mov_b32 s85, 0xd1111331
	s_mov_b32 s86, 0xf3111331
	s_mov_b32 s87, 0x33111331
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r256_gate_up_t8_lut3_skip:
	s_cmp_eq_u32 s5, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_lut4_skip
	s_mov_b32 s80, 0xf1311331
	s_mov_b32 s81, 0x31311331
	s_mov_b32 s82, 0x13311331
	s_mov_b32 s83, 0xf1131331
	s_mov_b32 s84, 0x31131331
	s_mov_b32 s85, 0x13131331
	s_mov_b32 s86, 0x11331331
	s_mov_b32 s87, 0xf1113331
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0x31113331
	s_mov_b32 s81, 0x13113331
	s_mov_b32 s82, 0x11313331
	s_mov_b32 s83, 0x11133331
	s_mov_b32 s84, 0xf1111531
	s_mov_b32 s85, 0x11111151
	s_mov_b32 s86, 0xd1111151
	s_mov_b32 s87, 0xf3111151
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0xf1311151
	s_mov_b32 s81, 0xf1131151
	s_mov_b32 s82, 0xf1113151
	s_mov_b32 s83, 0xf1111351
	s_mov_b32 s84, 0xf1111113
	s_mov_b32 s85, 0x31111113
	s_mov_b32 s86, 0xb1111113
	s_mov_b32 s87, 0x13111113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xd3111113
	s_mov_b32 s81, 0xf5111113
	s_mov_b32 s82, 0x11311113
	s_mov_b32 s83, 0xd1311113
	s_mov_b32 s84, 0xf3311113
	s_mov_b32 s85, 0x33311113
	s_mov_b32 s86, 0xf1511113
	s_mov_b32 s87, 0x11131113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r256_gate_up_t8_lut4_skip:
	s_cmp_eq_u32 s5, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_lut5_skip
	s_mov_b32 s80, 0xd1131113
	s_mov_b32 s81, 0xf3131113
	s_mov_b32 s82, 0x33131113
	s_mov_b32 s83, 0xf1331113
	s_mov_b32 s84, 0x31331113
	s_mov_b32 s85, 0x13331113
	s_mov_b32 s86, 0xf1151113
	s_mov_b32 s87, 0x11113113
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xd1113113
	s_mov_b32 s81, 0xf3113113
	s_mov_b32 s82, 0x33113113
	s_mov_b32 s83, 0xf1313113
	s_mov_b32 s84, 0x31313113
	s_mov_b32 s85, 0x13313113
	s_mov_b32 s86, 0xf1133113
	s_mov_b32 s87, 0x31133113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x13133113
	s_mov_b32 s81, 0x11333113
	s_mov_b32 s82, 0xf1115113
	s_mov_b32 s83, 0x11111313
	s_mov_b32 s84, 0xd1111313
	s_mov_b32 s85, 0xf3111313
	s_mov_b32 s86, 0x33111313
	s_mov_b32 s87, 0xf1311313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0x31311313
	s_mov_b32 s81, 0x13311313
	s_mov_b32 s82, 0xf1131313
	s_mov_b32 s83, 0x31131313
	s_mov_b32 s84, 0x13131313
	s_mov_b32 s85, 0x11331313
	s_mov_b32 s86, 0xf1113313
	s_mov_b32 s87, 0x31113313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r256_gate_up_t8_lut5_skip:
	s_cmp_eq_u32 s5, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_lut6_skip
	s_mov_b32 s80, 0x13113313
	s_mov_b32 s81, 0x11313313
	s_mov_b32 s82, 0x11133313
	s_mov_b32 s83, 0xf1111513
	s_mov_b32 s84, 0x11111133
	s_mov_b32 s85, 0xd1111133
	s_mov_b32 s86, 0xf3111133
	s_mov_b32 s87, 0x33111133
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xf1311133
	s_mov_b32 s81, 0x31311133
	s_mov_b32 s82, 0x13311133
	s_mov_b32 s83, 0xf1131133
	s_mov_b32 s84, 0x31131133
	s_mov_b32 s85, 0x13131133
	s_mov_b32 s86, 0x11331133
	s_mov_b32 s87, 0xf1113133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x31113133
	s_mov_b32 s81, 0x13113133
	s_mov_b32 s82, 0x11313133
	s_mov_b32 s83, 0x11133133
	s_mov_b32 s84, 0xf1111333
	s_mov_b32 s85, 0x31111333
	s_mov_b32 s86, 0x13111333
	s_mov_b32 s87, 0x11311333
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0x11131333
	s_mov_b32 s81, 0x11113333
	s_mov_b32 s82, 0xf1111153
	s_mov_b32 s83, 0x11111115
	s_mov_b32 s84, 0xd1111115
	s_mov_b32 s85, 0xf3111115
	s_mov_b32 s86, 0xf1311115
	s_mov_b32 s87, 0xf1131115
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r256_gate_up_t8_lut6_skip:
	s_cmp_eq_u32 s5, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_lut7_skip
	s_mov_b32 s80, 0xf1113115
	s_mov_b32 s81, 0xf1111315
	s_mov_b32 s82, 0xf1111135
	s_mov_b32 s83, 0xd3331113
	s_mov_b32 s84, 0xd3331131
	s_mov_b32 s85, 0xd3331311
	s_mov_b32 s86, 0xd3333111
	s_mov_b32 s87, 0xf1331333
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xf3131333
	s_mov_b32 s81, 0xf3311333
	s_mov_b32 s82, 0xd1131333
	s_mov_b32 s83, 0xd1311333
	s_mov_b32 s84, 0xd3111333
	s_mov_b32 s85, 0xf1333133
	s_mov_b32 s86, 0xf3133133
	s_mov_b32 s87, 0xf3313133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0xd1133133
	s_mov_b32 s81, 0xd1313133
	s_mov_b32 s82, 0xd3113133
	s_mov_b32 s83, 0xf1333313
	s_mov_b32 s84, 0xf3133313
	s_mov_b32 s85, 0xf3313313
	s_mov_b32 s86, 0xd1133313
	s_mov_b32 s87, 0xd1313313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xd3113331
	s_mov_b32 s81, 0xf1333331
	s_mov_b32 s82, 0xf3133331
	s_mov_b32 s83, 0xf3313331
	s_mov_b32 s84, 0xd1133331
	s_mov_b32 s85, 0xd1313331
	s_mov_b32 s86, 0xd3313311
	s_mov_b32 s87, 0xf3331133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r256_gate_up_t8_lut7_skip:
	s_cmp_gt_u32 s5, 3
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_stage_hi
	v_lshrrev_b32_e32 v34, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v0, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v35, v35, v34
	ds_store_b32 v24, v35 offset:1024
	ds_store_b32 v24, v36
	s_waitcnt lgkmcnt(0)
	s_branch .Le8pi4moe_r256_gate_up_t8_stage_join
	.Le8pi4moe_r256_gate_up_t8_stage_hi:
	ds_store_b32 v24, v36 offset:512
	s_waitcnt lgkmcnt(0)
	.Le8pi4moe_r256_gate_up_t8_stage_join:
	s_barrier
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_exit
	s_and_b32 s54, s48, -4
	s_cmp_eq_u32 s54, 0
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_tail
	.Le8pi4moe_r256_gate_up_t8_quad:
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t8_q0_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q0_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v17
	.Le8pi4moe_r256_gate_up_t8_q0_done:
	s_add_i32 s53, s53, 1
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t8_q1_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v85, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q1_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v93, v93, v17
	.Le8pi4moe_r256_gate_up_t8_q1_done:
	s_add_i32 s53, s53, 1
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t8_q2_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78, v78, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v82, v82, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v86, v86, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v90, v90, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q2_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v94, v94, v17
	.Le8pi4moe_r256_gate_up_t8_q2_done:
	s_add_i32 s53, s53, 1
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t8_q3_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v75, v75, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v79, v79, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v83, v83, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v87, v87, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v91, v91, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_q3_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v95, v17
	.Le8pi4moe_r256_gate_up_t8_q3_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s54
	s_cbranch_scc1 .Le8pi4moe_r256_gate_up_t8_quad
	.Le8pi4moe_r256_gate_up_t8_tail:
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_reduce
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t8_t0_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t0_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v17
	.Le8pi4moe_r256_gate_up_t8_t0_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_reduce
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t8_t1_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v73, v73, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v77, v77, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v81, v81, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v85, v85, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v89, v89, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t1_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v93, v93, v17
	.Le8pi4moe_r256_gate_up_t8_t1_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_reduce
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v96, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v97, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v98, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v99, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_xdone
	v_add_nc_u32_e32 v7, v100, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_xdone
	v_add_nc_u32_e32 v7, v101, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_xdone
	v_add_nc_u32_e32 v7, v102, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_xdone
	v_add_nc_u32_e32 v7, v103, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r256_gate_up_t8_t2_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v74, v74, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78, v78, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v82, v82, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v86, v86, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v90, v90, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_t2_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v94, v94, v17
	.Le8pi4moe_r256_gate_up_t8_t2_done:
	s_add_i32 s53, s53, 1
	.Le8pi4moe_r256_gate_up_t8_reduce:
	v_and_b32_e32 v7, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s68, 16, v7
	v_cndmask_b32_e64 v19, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v19, v19, v7, 2
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_gt_u32_e64 s69, 24, v7
	v_cndmask_b32_e64 v20, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v20, v20, v7, 2
	v_cmp_gt_u32_e64 s70, 28, v7
	v_cndmask_b32_e64 v21, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v21, v21, v7, 2
	v_cmp_gt_u32_e64 s71, 30, v7
	v_cndmask_b32_e64 v22, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v22, v22, v7, 2
	v_cmp_gt_u32_e64 s72, 31, v7
	v_cndmask_b32_e64 v23, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v23, v23, v7, 2
	v_add_f32_e32 v64, v64, v65
	v_add_f32_e32 v66, v66, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v66
	ds_bpermute_b32 v18, v19, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s68
	ds_bpermute_b32 v18, v20, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s69
	ds_bpermute_b32 v18, v21, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s70
	ds_bpermute_b32 v18, v22, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s71
	ds_bpermute_b32 v18, v23, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_red_done
	v_add_f32_e32 v68, v68, v69
	v_add_f32_e32 v70, v70, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v70
	ds_bpermute_b32 v18, v19, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s68
	ds_bpermute_b32 v18, v20, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s69
	ds_bpermute_b32 v18, v21, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s70
	ds_bpermute_b32 v18, v22, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s71
	ds_bpermute_b32 v18, v23, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_red_done
	v_add_f32_e32 v72, v72, v73
	v_add_f32_e32 v74, v74, v75
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v72, v72, v74
	ds_bpermute_b32 v18, v19, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s68
	ds_bpermute_b32 v18, v20, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s69
	ds_bpermute_b32 v18, v21, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s70
	ds_bpermute_b32 v18, v22, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s71
	ds_bpermute_b32 v18, v23, v72
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v72, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v72, v72, v18, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_red_done
	v_add_f32_e32 v76, v76, v77
	v_add_f32_e32 v78, v78, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v76, v76, v78
	ds_bpermute_b32 v18, v19, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s68
	ds_bpermute_b32 v18, v20, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s69
	ds_bpermute_b32 v18, v21, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s70
	ds_bpermute_b32 v18, v22, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s71
	ds_bpermute_b32 v18, v23, v76
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v76, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v76, v76, v18, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_red_done
	v_add_f32_e32 v80, v80, v81
	v_add_f32_e32 v82, v82, v83
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v80, v80, v82
	ds_bpermute_b32 v18, v19, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s68
	ds_bpermute_b32 v18, v20, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s69
	ds_bpermute_b32 v18, v21, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s70
	ds_bpermute_b32 v18, v22, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s71
	ds_bpermute_b32 v18, v23, v80
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v80, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v18, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_red_done
	v_add_f32_e32 v84, v84, v85
	v_add_f32_e32 v86, v86, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v84, v84, v86
	ds_bpermute_b32 v18, v19, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s68
	ds_bpermute_b32 v18, v20, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s69
	ds_bpermute_b32 v18, v21, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s70
	ds_bpermute_b32 v18, v22, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s71
	ds_bpermute_b32 v18, v23, v84
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v84, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v84, v84, v18, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_red_done
	v_add_f32_e32 v88, v88, v89
	v_add_f32_e32 v90, v90, v91
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v88, v88, v90
	ds_bpermute_b32 v18, v19, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s68
	ds_bpermute_b32 v18, v20, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s69
	ds_bpermute_b32 v18, v21, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s70
	ds_bpermute_b32 v18, v22, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s71
	ds_bpermute_b32 v18, v23, v88
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v88, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v88, v88, v18, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_red_done
	v_add_f32_e32 v92, v92, v93
	v_add_f32_e32 v94, v94, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v92, v92, v94
	ds_bpermute_b32 v18, v19, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s68
	ds_bpermute_b32 v18, v20, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s69
	ds_bpermute_b32 v18, v21, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s70
	ds_bpermute_b32 v18, v22, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s71
	ds_bpermute_b32 v18, v23, v92
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v92, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v92, v92, v18, s72
	.Le8pi4moe_r256_gate_up_t8_red_done:
	v_cmpx_eq_u32_e32 0, v7
	global_store_b32 v104, v64, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_st_done
	global_store_b32 v105, v68, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_st_done
	global_store_b32 v106, v72, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_st_done
	global_store_b32 v107, v76, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_st_done
	global_store_b32 v108, v80, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_st_done
	global_store_b32 v109, v84, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_st_done
	global_store_b32 v110, v88, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r256_gate_up_t8_st_done
	global_store_b32 v111, v92, s[58:59]
	.Le8pi4moe_r256_gate_up_t8_st_done:
	s_waitcnt_vscnt null, 0x0
	.Le8pi4moe_r256_gate_up_t8_exit:
	s_endpgm
.Le8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8_end:
.size e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8, .Le8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8_end-e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8
	.amdhsa_group_segment_fixed_size 1536
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
	.amdhsa_user_sgpr_count 13
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
	.amdhsa_next_free_vgpr 112
	.amdhsa_next_free_sgpr 96
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Le8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8_end-e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8)<<4)&1008)>>4
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
.protected e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8
.globl e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8
.p2align 8
.type e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8,@function
e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8:
	v_and_b32_e32 v24, 0x7f, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 2, v24
	v_and_b32_e32 v7, 31, v0
	v_lshrrev_b32_e32 v6, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s5, v6
	s_load_b256 s[88:95], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b64 s[28:29], s[0:1], 0x50
	s_load_b32 s6, s[0:1], 0x58
	s_lshl_b32 s30, s13, 3
	s_add_i32 s30, s30, s5
	s_mov_b32 s31, s14
	s_mov_b32 s7, s15
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_cmp_gt_u32 s25, 64
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_cmp_eq_u32 s24, 0x50
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_exit
	s_lshl_b32 s51, s7, 3
	s_cmp_ge_u32 s51, s25
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_and_b32 s51, s30, -8
	s_cmp_ge_u32 s51, s22
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_mul_i32 s51, s7, 0x788
	s_add_u32 s90, s90, s51
	s_addc_u32 s91, s91, 0
	s_add_u32 s92, s92, s51
	s_addc_u32 s93, s93, 0
	s_add_u32 s94, s94, s51
	s_addc_u32 s95, s95, 0
	s_cmp_ge_u32 s31, s24
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_lshl_b32 s51, s31, 2
	s_add_u32 s36, s92, s51
	s_addc_u32 s37, s93, 0
	s_load_b64 s[32:33], s[36:37], null
	s_add_u32 s36, s90, s51
	s_addc_u32 s37, s91, 0
	s_load_b32 s34, s[36:37], null
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s35, s33, s32
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_cmp_gt_u32 s35, 8
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_mov_b32 s51, 0x50
	s_cmp_gt_u32 s33, s51
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_cmp_ge_u32 s34, s6
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_lshl_b32 s51, s34, 3
	s_add_u32 s36, s88, s51
	s_addc_u32 s37, s89, 0
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
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v3, 4, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v4, 30, v7
	v_and_b32_e32 v5, 1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v5, 4, v5
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_min_u32_e32 v39, s63, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, s32, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 2, v39
	global_load_b32 v32, v39, s[94:95]
	s_mul_i32 s52, s25, 10
	s_waitcnt vmcnt(0)
	v_readlane_b32 s51, v32, 0
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	v_readlane_b32 s51, v32, 1
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	v_readlane_b32 s51, v32, 2
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	v_readlane_b32 s51, v32, 3
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	v_readlane_b32 s51, v32, 4
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	v_readlane_b32 s51, v32, 5
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	v_readlane_b32 s51, v32, 6
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	v_readlane_b32 s51, v32, 7
	s_cmp_ge_u32 s51, s52
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	v_cvt_f32_u32_e32 v37, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v37, v37
	v_cvt_f32_u32_e32 v34, v32
	s_waitcnt_depctr depctr_va_vdst(0)
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
	v_mov_b32_e32 v72, s73
	s_delay_alu instid0(VALU_DEP_3)
	v_readlane_b32 s73, v38, 0
	v_mov_b32_e32 v80, s73
	v_readlane_b32 s74, v33, 1
	v_mov_b32_e32 v73, s74
	v_readlane_b32 s74, v38, 1
	v_mov_b32_e32 v81, s74
	v_readlane_b32 s75, v33, 2
	v_mov_b32_e32 v74, s75
	v_readlane_b32 s75, v38, 2
	v_mov_b32_e32 v82, s75
	v_readlane_b32 s76, v33, 3
	v_mov_b32_e32 v75, s76
	v_readlane_b32 s76, v38, 3
	v_mov_b32_e32 v83, s76
	v_readlane_b32 s73, v33, 4
	v_mov_b32_e32 v76, s73
	v_readlane_b32 s73, v38, 4
	v_mov_b32_e32 v84, s73
	v_readlane_b32 s74, v33, 5
	v_mov_b32_e32 v77, s74
	v_readlane_b32 s74, v38, 5
	v_mov_b32_e32 v85, s74
	v_readlane_b32 s75, v33, 6
	v_mov_b32_e32 v78, s75
	v_readlane_b32 s75, v38, 6
	v_mov_b32_e32 v86, s75
	v_readlane_b32 s76, v33, 7
	v_mov_b32_e32 v79, s76
	v_readlane_b32 s76, v38, 7
	v_mov_b32_e32 v87, s76
	s_cmp_eq_u32 s5, 0
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_lut0_skip
	s_mov_b32 s80, 0x11111111
	s_mov_b32 s81, 0xd1111111
	s_mov_b32 s82, 0x51111111
	s_mov_b32 s83, 0xf3111111
	s_mov_b32 s84, 0x33111111
	s_mov_b32 s85, 0xb3111111
	s_mov_b32 s86, 0x15111111
	s_mov_b32 s87, 0xd5111111
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xf1311111
	s_mov_b32 s81, 0x31311111
	s_mov_b32 s82, 0xb1311111
	s_mov_b32 s83, 0x13311111
	s_mov_b32 s84, 0xd3311111
	s_mov_b32 s85, 0xf5311111
	s_mov_b32 s86, 0x11511111
	s_mov_b32 s87, 0xd1511111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0xf3511111
	s_mov_b32 s81, 0xf1131111
	s_mov_b32 s82, 0x31131111
	s_mov_b32 s83, 0xb1131111
	s_mov_b32 s84, 0x13131111
	s_mov_b32 s85, 0xd3131111
	s_mov_b32 s86, 0xf5131111
	s_mov_b32 s87, 0x11331111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xd1331111
	s_mov_b32 s81, 0xf3331111
	s_mov_b32 s82, 0x33331111
	s_mov_b32 s83, 0xf1531111
	s_mov_b32 s84, 0x11151111
	s_mov_b32 s85, 0xd1151111
	s_mov_b32 s86, 0xf3151111
	s_mov_b32 s87, 0xf1351111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r128_down_t8_lut0_skip:
	s_cmp_eq_u32 s5, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_lut1_skip
	s_mov_b32 s80, 0xf1113111
	s_mov_b32 s81, 0x31113111
	s_mov_b32 s82, 0xb1113111
	s_mov_b32 s83, 0x13113111
	s_mov_b32 s84, 0xd3113111
	s_mov_b32 s85, 0xf5113111
	s_mov_b32 s86, 0x11313111
	s_mov_b32 s87, 0xd1313111
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xf3313111
	s_mov_b32 s81, 0x33313111
	s_mov_b32 s82, 0xf1513111
	s_mov_b32 s83, 0x11133111
	s_mov_b32 s84, 0xd1133111
	s_mov_b32 s85, 0xf3133111
	s_mov_b32 s86, 0x33133111
	s_mov_b32 s87, 0xf1333111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x31333111
	s_mov_b32 s81, 0x13333111
	s_mov_b32 s82, 0xf1153111
	s_mov_b32 s83, 0x11115111
	s_mov_b32 s84, 0xd1115111
	s_mov_b32 s85, 0xf3115111
	s_mov_b32 s86, 0xf1315111
	s_mov_b32 s87, 0xf1135111
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xf1111311
	s_mov_b32 s81, 0x31111311
	s_mov_b32 s82, 0xb1111311
	s_mov_b32 s83, 0x13111311
	s_mov_b32 s84, 0xd3111311
	s_mov_b32 s85, 0xf5111311
	s_mov_b32 s86, 0x11311311
	s_mov_b32 s87, 0xd1311311
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r128_down_t8_lut1_skip:
	s_cmp_eq_u32 s5, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_lut2_skip
	s_mov_b32 s80, 0xf3311311
	s_mov_b32 s81, 0x33311311
	s_mov_b32 s82, 0xf1511311
	s_mov_b32 s83, 0x11131311
	s_mov_b32 s84, 0xd1131311
	s_mov_b32 s85, 0xf3131311
	s_mov_b32 s86, 0x33131311
	s_mov_b32 s87, 0xf1331311
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0x31331311
	s_mov_b32 s81, 0x13331311
	s_mov_b32 s82, 0xf1151311
	s_mov_b32 s83, 0x11113311
	s_mov_b32 s84, 0xd1113311
	s_mov_b32 s85, 0xf3113311
	s_mov_b32 s86, 0x33113311
	s_mov_b32 s87, 0xf1313311
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x31313311
	s_mov_b32 s81, 0x13313311
	s_mov_b32 s82, 0xf1133311
	s_mov_b32 s83, 0x31133311
	s_mov_b32 s84, 0x13133311
	s_mov_b32 s85, 0x11333311
	s_mov_b32 s86, 0xf1115311
	s_mov_b32 s87, 0x11111511
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xd1111511
	s_mov_b32 s81, 0xf3111511
	s_mov_b32 s82, 0xf1311511
	s_mov_b32 s83, 0xf1131511
	s_mov_b32 s84, 0xf1113511
	s_mov_b32 s85, 0xf1111131
	s_mov_b32 s86, 0x31111131
	s_mov_b32 s87, 0xb1111131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r128_down_t8_lut2_skip:
	s_cmp_eq_u32 s5, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_lut3_skip
	s_mov_b32 s80, 0x13111131
	s_mov_b32 s81, 0xd3111131
	s_mov_b32 s82, 0xf5111131
	s_mov_b32 s83, 0x11311131
	s_mov_b32 s84, 0xd1311131
	s_mov_b32 s85, 0xf3311131
	s_mov_b32 s86, 0x33311131
	s_mov_b32 s87, 0xf1511131
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0x11131131
	s_mov_b32 s81, 0xd1131131
	s_mov_b32 s82, 0xf3131131
	s_mov_b32 s83, 0x33131131
	s_mov_b32 s84, 0xf1331131
	s_mov_b32 s85, 0x31331131
	s_mov_b32 s86, 0x13331131
	s_mov_b32 s87, 0xf1151131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x11113131
	s_mov_b32 s81, 0xd1113131
	s_mov_b32 s82, 0xf3113131
	s_mov_b32 s83, 0x33113131
	s_mov_b32 s84, 0xf1313131
	s_mov_b32 s85, 0x31313131
	s_mov_b32 s86, 0x13313131
	s_mov_b32 s87, 0xf1133131
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0x31133131
	s_mov_b32 s81, 0x13133131
	s_mov_b32 s82, 0x11333131
	s_mov_b32 s83, 0xf1115131
	s_mov_b32 s84, 0x11111331
	s_mov_b32 s85, 0xd1111331
	s_mov_b32 s86, 0xf3111331
	s_mov_b32 s87, 0x33111331
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r128_down_t8_lut3_skip:
	s_cmp_eq_u32 s5, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_lut4_skip
	s_mov_b32 s80, 0xf1311331
	s_mov_b32 s81, 0x31311331
	s_mov_b32 s82, 0x13311331
	s_mov_b32 s83, 0xf1131331
	s_mov_b32 s84, 0x31131331
	s_mov_b32 s85, 0x13131331
	s_mov_b32 s86, 0x11331331
	s_mov_b32 s87, 0xf1113331
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0x31113331
	s_mov_b32 s81, 0x13113331
	s_mov_b32 s82, 0x11313331
	s_mov_b32 s83, 0x11133331
	s_mov_b32 s84, 0xf1111531
	s_mov_b32 s85, 0x11111151
	s_mov_b32 s86, 0xd1111151
	s_mov_b32 s87, 0xf3111151
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0xf1311151
	s_mov_b32 s81, 0xf1131151
	s_mov_b32 s82, 0xf1113151
	s_mov_b32 s83, 0xf1111351
	s_mov_b32 s84, 0xf1111113
	s_mov_b32 s85, 0x31111113
	s_mov_b32 s86, 0xb1111113
	s_mov_b32 s87, 0x13111113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xd3111113
	s_mov_b32 s81, 0xf5111113
	s_mov_b32 s82, 0x11311113
	s_mov_b32 s83, 0xd1311113
	s_mov_b32 s84, 0xf3311113
	s_mov_b32 s85, 0x33311113
	s_mov_b32 s86, 0xf1511113
	s_mov_b32 s87, 0x11131113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r128_down_t8_lut4_skip:
	s_cmp_eq_u32 s5, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_lut5_skip
	s_mov_b32 s80, 0xd1131113
	s_mov_b32 s81, 0xf3131113
	s_mov_b32 s82, 0x33131113
	s_mov_b32 s83, 0xf1331113
	s_mov_b32 s84, 0x31331113
	s_mov_b32 s85, 0x13331113
	s_mov_b32 s86, 0xf1151113
	s_mov_b32 s87, 0x11113113
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xd1113113
	s_mov_b32 s81, 0xf3113113
	s_mov_b32 s82, 0x33113113
	s_mov_b32 s83, 0xf1313113
	s_mov_b32 s84, 0x31313113
	s_mov_b32 s85, 0x13313113
	s_mov_b32 s86, 0xf1133113
	s_mov_b32 s87, 0x31133113
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x13133113
	s_mov_b32 s81, 0x11333113
	s_mov_b32 s82, 0xf1115113
	s_mov_b32 s83, 0x11111313
	s_mov_b32 s84, 0xd1111313
	s_mov_b32 s85, 0xf3111313
	s_mov_b32 s86, 0x33111313
	s_mov_b32 s87, 0xf1311313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0x31311313
	s_mov_b32 s81, 0x13311313
	s_mov_b32 s82, 0xf1131313
	s_mov_b32 s83, 0x31131313
	s_mov_b32 s84, 0x13131313
	s_mov_b32 s85, 0x11331313
	s_mov_b32 s86, 0xf1113313
	s_mov_b32 s87, 0x31113313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r128_down_t8_lut5_skip:
	s_cmp_eq_u32 s5, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_lut6_skip
	s_mov_b32 s80, 0x13113313
	s_mov_b32 s81, 0x11313313
	s_mov_b32 s82, 0x11133313
	s_mov_b32 s83, 0xf1111513
	s_mov_b32 s84, 0x11111133
	s_mov_b32 s85, 0xd1111133
	s_mov_b32 s86, 0xf3111133
	s_mov_b32 s87, 0x33111133
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xf1311133
	s_mov_b32 s81, 0x31311133
	s_mov_b32 s82, 0x13311133
	s_mov_b32 s83, 0xf1131133
	s_mov_b32 s84, 0x31131133
	s_mov_b32 s85, 0x13131133
	s_mov_b32 s86, 0x11331133
	s_mov_b32 s87, 0xf1113133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0x31113133
	s_mov_b32 s81, 0x13113133
	s_mov_b32 s82, 0x11313133
	s_mov_b32 s83, 0x11133133
	s_mov_b32 s84, 0xf1111333
	s_mov_b32 s85, 0x31111333
	s_mov_b32 s86, 0x13111333
	s_mov_b32 s87, 0x11311333
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0x11131333
	s_mov_b32 s81, 0x11113333
	s_mov_b32 s82, 0xf1111153
	s_mov_b32 s83, 0x11111115
	s_mov_b32 s84, 0xd1111115
	s_mov_b32 s85, 0xf3111115
	s_mov_b32 s86, 0xf1311115
	s_mov_b32 s87, 0xf1131115
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r128_down_t8_lut6_skip:
	s_cmp_eq_u32 s5, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_lut7_skip
	s_mov_b32 s80, 0xf1113115
	s_mov_b32 s81, 0xf1111315
	s_mov_b32 s82, 0xf1111135
	s_mov_b32 s83, 0xd3331113
	s_mov_b32 s84, 0xd3331131
	s_mov_b32 s85, 0xd3331311
	s_mov_b32 s86, 0xd3333111
	s_mov_b32 s87, 0xf1331333
	v_writelane_b32 v36, s80, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 3
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 5
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 6
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 7
	s_mov_b32 s80, 0xf3131333
	s_mov_b32 s81, 0xf3311333
	s_mov_b32 s82, 0xd1131333
	s_mov_b32 s83, 0xd1311333
	s_mov_b32 s84, 0xd3111333
	s_mov_b32 s85, 0xf1333133
	s_mov_b32 s86, 0xf3133133
	s_mov_b32 s87, 0xf3313133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 9
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 10
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 11
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 12
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 13
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 14
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 15
	s_mov_b32 s80, 0xd1133133
	s_mov_b32 s81, 0xd1313133
	s_mov_b32 s82, 0xd3113133
	s_mov_b32 s83, 0xf1333313
	s_mov_b32 s84, 0xf3133313
	s_mov_b32 s85, 0xf3313313
	s_mov_b32 s86, 0xd1133313
	s_mov_b32 s87, 0xd1313313
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 16
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 17
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 18
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 19
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 20
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 21
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 22
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 23
	s_mov_b32 s80, 0xd3113331
	s_mov_b32 s81, 0xf1333331
	s_mov_b32 s82, 0xf3133331
	s_mov_b32 s83, 0xf3313331
	s_mov_b32 s84, 0xd1133331
	s_mov_b32 s85, 0xd1313331
	s_mov_b32 s86, 0xd3313311
	s_mov_b32 s87, 0xf3331133
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s80, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s81, 25
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s82, 26
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s83, 27
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s84, 28
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s85, 29
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s86, 30
	s_delay_alu instid0(VALU_DEP_1)
	v_writelane_b32 v36, s87, 31
	.Le8pi4moe_r128_down_t8_lut7_skip:
	s_cmp_gt_u32 s5, 3
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_stage_hi
	v_lshrrev_b32_e32 v34, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v0, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v34, 1, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_xor_b32_e32 v33, v33, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 7, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v34, 0x80, v33, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0xf000f, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 6, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x3030303, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v34, v34, 3, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v34, 0x11111111, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v35, v35, v34
	ds_store_b32 v24, v35 offset:1024
	ds_store_b32 v24, v36
	s_waitcnt lgkmcnt(0)
	s_branch .Le8pi4moe_r128_down_t8_stage_join
	.Le8pi4moe_r128_down_t8_stage_hi:
	ds_store_b32 v24, v36 offset:512
	s_waitcnt lgkmcnt(0)
	.Le8pi4moe_r128_down_t8_stage_join:
	s_barrier
	s_cmp_ge_u32 s30, s22
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_exit
	s_cmp_eq_u32 s48, 0
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_reduce
	.Le8pi4moe_r128_down_t8_group:
	s_lshl_b32 s55, s53, 5
	s_lshl_b32 s56, s53, 9
	s_lshl_b32 s57, s53, 1
	v_add_nc_u32_e32 v1, s55, v4
	v_mov_b32_e32 v2, s57
	v_add_nc_u32_e32 v6, s56, v3
	global_load_u16 v8, v1, s[44:45]
	global_load_u16 v9, v2, s[46:47]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v7, v72, v6
	buffer_load_b128 v[32:35], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_xdone
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v7, v73, v6
	buffer_load_b128 v[36:39], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_xdone
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v7, v74, v6
	buffer_load_b128 v[40:43], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_xdone
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, v75, v6
	buffer_load_b128 v[44:47], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_xdone
	v_add_nc_u32_e32 v7, v76, v6
	buffer_load_b128 v[48:51], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_xdone
	v_add_nc_u32_e32 v7, v77, v6
	buffer_load_b128 v[52:55], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_xdone
	v_add_nc_u32_e32 v7, v78, v6
	buffer_load_b128 v[56:59], v7, s[40:43], 0 offen
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_xdone
	v_add_nc_u32_e32 v7, v79, v6
	buffer_load_b128 v[60:63], v7, s[40:43], 0 offen
	.Le8pi4moe_r128_down_t8_g_xdone:
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v10, 0x3fc, v10
	v_lshrrev_b32_e32 v11, 6, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v11, 0x1fc, v11
	ds_load_b32 v12, v10
	ds_load_b32 v13, v11 offset:1024
	v_bfe_i32 v14, v8, 15, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v14, 0xeeeeeeef, v14
	v_cvt_f32_f16_e64 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, 0x3e800000, v9
	s_waitcnt lgkmcnt(0)
	v_xor_b32_e32 v12, v13, v12
	v_and_b32_e32 v13, 0x11111111, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v12, v13, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v15, v5, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_bfe_i32 v28, v15, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 1, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v9, v28
	v_bfe_i32 v29, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 1, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v9, v29
	v_bfe_i32 v30, v15, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 1, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v9, v30
	v_bfe_i32 v31, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 1, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_i32_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v9, v31
	v_mul_f32_e32 v16, v28, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v64, v64, v16
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_done
	v_mul_f32_e32 v17, v28, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v17
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_done
	v_mul_f32_e32 v16, v28, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v16
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_done
	v_mul_f32_e32 v17, v28, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v17
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_done
	v_mul_f32_e32 v16, v28, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v68, v68, v16
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_done
	v_mul_f32_e32 v17, v28, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v69, v69, v17
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_done
	v_mul_f32_e32 v16, v28, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v29, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v30, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v31, v59
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v70, v70, v16
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_g_done
	v_mul_f32_e32 v17, v28, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v29, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v30, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v31, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v17
	.Le8pi4moe_r128_down_t8_g_done:
	s_add_i32 s53, s53, 1
	s_cmp_lt_u32 s53, s48
	s_cbranch_scc1 .Le8pi4moe_r128_down_t8_group
	.Le8pi4moe_r128_down_t8_reduce:
	v_and_b32_e32 v7, 31, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s68, 16, v7
	v_cndmask_b32_e64 v19, 0, 16, s68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v19, v19, v7, 2
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_gt_u32_e64 s69, 24, v7
	v_cndmask_b32_e64 v20, 0, 8, s69
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v20, v20, v7, 2
	v_cmp_gt_u32_e64 s70, 28, v7
	v_cndmask_b32_e64 v21, 0, 4, s70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v21, v21, v7, 2
	v_cmp_gt_u32_e64 s71, 30, v7
	v_cndmask_b32_e64 v22, 0, 2, s71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v22, v22, v7, 2
	v_cmp_gt_u32_e64 s72, 31, v7
	v_cndmask_b32_e64 v23, 0, 1, s72
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v23, v23, v7, 2
	ds_bpermute_b32 v18, v19, v64
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s68
	ds_bpermute_b32 v18, v20, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s69
	ds_bpermute_b32 v18, v21, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s70
	ds_bpermute_b32 v18, v22, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s71
	ds_bpermute_b32 v18, v23, v64
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v64, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v64, v64, v18, s72
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_red_done
	ds_bpermute_b32 v18, v19, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s68
	ds_bpermute_b32 v18, v20, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s69
	ds_bpermute_b32 v18, v21, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s70
	ds_bpermute_b32 v18, v22, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s71
	ds_bpermute_b32 v18, v23, v65
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v65, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v65, v65, v18, s72
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_red_done
	ds_bpermute_b32 v18, v19, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s68
	ds_bpermute_b32 v18, v20, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s69
	ds_bpermute_b32 v18, v21, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s70
	ds_bpermute_b32 v18, v22, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s71
	ds_bpermute_b32 v18, v23, v66
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v66, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v66, v66, v18, s72
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_red_done
	ds_bpermute_b32 v18, v19, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s68
	ds_bpermute_b32 v18, v20, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s69
	ds_bpermute_b32 v18, v21, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s70
	ds_bpermute_b32 v18, v22, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s71
	ds_bpermute_b32 v18, v23, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v67, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v67, v67, v18, s72
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_red_done
	ds_bpermute_b32 v18, v19, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s68
	ds_bpermute_b32 v18, v20, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s69
	ds_bpermute_b32 v18, v21, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s70
	ds_bpermute_b32 v18, v22, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s71
	ds_bpermute_b32 v18, v23, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v68, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v68, v68, v18, s72
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_red_done
	ds_bpermute_b32 v18, v19, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s68
	ds_bpermute_b32 v18, v20, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s69
	ds_bpermute_b32 v18, v21, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s70
	ds_bpermute_b32 v18, v22, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s71
	ds_bpermute_b32 v18, v23, v69
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v69, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v69, v69, v18, s72
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_red_done
	ds_bpermute_b32 v18, v19, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s68
	ds_bpermute_b32 v18, v20, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s69
	ds_bpermute_b32 v18, v21, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s70
	ds_bpermute_b32 v18, v22, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s71
	ds_bpermute_b32 v18, v23, v70
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v70, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, v18, s72
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_red_done
	ds_bpermute_b32 v18, v19, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s68
	ds_bpermute_b32 v18, v20, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s69
	ds_bpermute_b32 v18, v21, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s70
	ds_bpermute_b32 v18, v22, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s71
	ds_bpermute_b32 v18, v23, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v18, v71, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v71, v71, v18, s72
	.Le8pi4moe_r128_down_t8_red_done:
	v_cmpx_eq_u32_e32 0, v7
	global_store_b32 v80, v64, s[58:59]
	s_cmp_gt_u32 s35, 1
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_st_done
	global_store_b32 v81, v65, s[58:59]
	s_cmp_gt_u32 s35, 2
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_st_done
	global_store_b32 v82, v66, s[58:59]
	s_cmp_gt_u32 s35, 3
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_st_done
	global_store_b32 v83, v67, s[58:59]
	s_cmp_gt_u32 s35, 4
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_st_done
	global_store_b32 v84, v68, s[58:59]
	s_cmp_gt_u32 s35, 5
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_st_done
	global_store_b32 v85, v69, s[58:59]
	s_cmp_gt_u32 s35, 6
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_st_done
	global_store_b32 v86, v70, s[58:59]
	s_cmp_gt_u32 s35, 7
	s_cbranch_scc0 .Le8pi4moe_r128_down_t8_st_done
	global_store_b32 v87, v71, s[58:59]
	.Le8pi4moe_r128_down_t8_st_done:
	s_waitcnt_vscnt null, 0x0
	.Le8pi4moe_r128_down_t8_exit:
	s_endpgm
.Le8p_i4_moe_gemv_r128_down_pm_gfx1100_t8_end:
.size e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8, .Le8p_i4_moe_gemv_r128_down_pm_gfx1100_t8_end-e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8
	.amdhsa_group_segment_fixed_size 1536
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
	.amdhsa_user_sgpr_count 13
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
	.amdhsa_next_free_vgpr 88
	.amdhsa_next_free_sgpr 96
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Le8p_i4_moe_gemv_r128_down_pm_gfx1100_t8_end-e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8)<<4)&1008)>>4
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
    .group_segment_fixed_size: 1536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 96
    .sgpr_spill_count: 0
    .symbol: e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100.kd
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
    .group_segment_fixed_size: 1536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: e8p_i4_moe_gemv_r128_down_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 96
    .sgpr_spill_count: 0
    .symbol: e8p_i4_moe_gemv_r128_down_pm_gfx1100.kd
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
    .group_segment_fixed_size: 1536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 96
    .sgpr_spill_count: 0
    .symbol: e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8.kd
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
    .group_segment_fixed_size: 1536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name: e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8
    .private_segment_fixed_size: 0
    .sgpr_count: 96
    .sgpr_spill_count: 0
    .symbol: e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 88
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
