	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	re_nocompute_nb         ; -- Begin function re_nocompute_nb
	.globl	re_nocompute_nb
	.p2align	8
	.type	re_nocompute_nb,@function
re_nocompute_nb:                        ; @re_nocompute_nb
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b96 s[16:18], s[0:1], 0x28
	s_load_b64 s[6:7], s[0:1], 0x20
	s_load_b256 s[8:15], s[0:1], 0x0
	v_bfe_u32 v2, v0, 4, 1
	v_lshrrev_b32_e32 v1, 1, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v96, 3, v2
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s17, 0xff
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	v_lshlrev_b32_e32 v2, 3, v2
	s_mov_b32 s0, 0
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b32 s0, -1
                                        ; implicit-def: $vgpr2
.LBB0_3:
	v_and_b32_e32 v95, 15, v0
	v_and_b32_e32 v93, 0x60, v0
	v_and_b32_e32 v94, 64, v1
	s_lshl_b32 s19, ttmp7, 7
	s_lshl_b32 s22, ttmp9, 7
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s21, 0
	s_cbranch_vccnz .LBB0_16
; %bb.4:
	v_lshrrev_b32_e32 v2, 2, v0
	s_ashr_i32 s0, s17, 31
	s_add_co_i32 s4, s16, -1
	s_lshr_b32 s0, s0, 24
	s_add_co_i32 s20, s18, -1
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v3, s22, v2
	s_add_co_i32 s0, s17, s0
	v_add_nc_u32_e32 v4, s19, v2
	s_ashr_i32 s23, s0, 8
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v2, 64, v3
	v_min_i32_e32 v5, s4, v3
	v_and_b32_e32 v6, 0x7f, v0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s5, s23, 0x88
	v_cmp_gt_i32_e64 s1, s16, v3
	v_min_i32_e32 v9, s4, v2
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v5, s5, v5
	v_or_b32_e32 v10, s22, v6
	v_lshrrev_b32_e32 v14, 4, v0
	v_dual_mov_b32 v65, v8 :: v_dual_lshlrev_b32 v16, 2, v1
	v_mul_lo_u32 v3, s5, v9
	v_add_nc_u32_e32 v9, s19, v1
	v_min_i32_e32 v12, s4, v10
	v_add_co_u32 v5, s3, s8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, s9, 0, s3
	v_cmp_gt_i32_e64 s3, s16, v2
	v_min_i32_e32 v2, s20, v9
	v_mul_lo_u32 v12, s5, v12
	v_add_co_u32 v3, s4, s8, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s9, 0, s4
	v_mul_lo_u32 v97, v2, s23
	v_cmp_gt_i32_e64 s4, s18, v9
	v_and_b32_e32 v9, 3, v0
	v_or_b32_e32 v2, 16, v14
	v_add_co_u32 v98, s8, s8, v12
	v_lshrrev_b32_e32 v12, 7, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_or_b32 v14, v14, 12, v9
	v_and_or_b32 v15, v2, 28, v9
	v_lshlrev_b32_e32 v2, 4, v9
	v_lshlrev_b32_e32 v9, 3, v9
	v_or_b32_e32 v1, v94, v95
	v_cmp_gt_i32_e64 s5, s16, v10
	v_lshlrev_b32_e32 v10, 1, v0
	v_lshl_add_u32 v18, v12, 2, 0
	v_dual_mov_b32 v67, v8 :: v_dual_lshlrev_b32 v100, 4, v12
	v_dual_mov_b32 v69, v8 :: v_dual_lshlrev_b32 v12, 2, v1
	v_mov_b32_e32 v1, v8
	v_dual_mov_b32 v66, v8 :: v_dual_lshlrev_b32 v17, 3, v6
	v_mov_b32_e32 v6, v8
	v_add_co_u32 v101, vcc_lo, v5, v9
	v_dual_mov_b32 v5, v8 :: v_dual_and_b32 v10, 0x78, v10
	v_add_co_ci_u32_e64 v102, null, 0, v11, vcc_lo
	v_add_co_u32 v103, vcc_lo, v3, v9
	v_mov_b32_e32 v9, v8
	v_add_nc_u32_e32 v7, 64, v4
	v_mad_u32_u24 v15, 0x120, v15, v10
	v_mad_u32_u24 v14, 0x120, v14, v10
	v_mov_b32_e32 v10, v8
	v_cmp_gt_i32_e64 s0, s18, v4
	v_cmp_gt_i32_e64 s2, s18, v7
	v_min_i32_e32 v7, s20, v7
	v_min_i32_e32 v4, s20, v4
	v_dual_mov_b32 v72, v8 :: v_dual_add_nc_u32 v105, 0, v16
	v_mov_b32_e32 v11, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[85:86], null, v7, s17, v[2:3]
	v_mov_b32_e32 v7, v8
	v_mad_co_u64_u32 v[86:87], null, v4, s17, v[2:3]
	v_or_b32_e32 v2, v96, v93
	v_mov_b32_e32 v68, v8
	v_dual_mov_b32 v70, v8 :: v_dual_mov_b32 v71, v8
	v_dual_mov_b32 v57, v65 :: v_dual_add_nc_u32 v106, v18, v17
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v2, v8 :: v_dual_lshlrev_b32 v19, 3, v2
	v_dual_mov_b32 v3, v8 :: v_dual_mov_b32 v4, v8
	v_dual_mov_b32 v64, v72 :: v_dual_mov_b32 v49, v65
	v_dual_mov_b32 v56, v72 :: v_dual_mov_b32 v41, v65
	v_dual_mov_b32 v48, v72 :: v_dual_mov_b32 v33, v65
	v_dual_mov_b32 v84, v11 :: v_dual_mov_b32 v81, v8
	v_dual_mov_b32 v59, v67 :: v_dual_add_nc_u32 v108, 0, v19
	v_dual_mov_b32 v40, v72 :: v_dual_mov_b32 v25, v65
	v_dual_mov_b32 v82, v9 :: v_dual_mov_b32 v17, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v104, null, 0, v13, vcc_lo
	v_dual_mov_b32 v58, v66 :: v_dual_add_nc_u32 v107, 0, v12
	v_dual_mov_b32 v60, v68 :: v_dual_add_nc_u32 v109, 0, v14
	v_dual_mov_b32 v61, v69 :: v_dual_add_nc_u32 v110, 0, v15
	v_dual_mov_b32 v32, v72 :: v_dual_mov_b32 v83, v10
	v_dual_mov_b32 v24, v72 :: v_dual_mov_b32 v9, v65
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v77, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v99, null, s9, 0, s8
	v_dual_mov_b32 v62, v70 :: v_dual_mov_b32 v63, v71
	v_dual_mov_b32 v50, v66 :: v_dual_mov_b32 v51, v67
	v_dual_mov_b32 v52, v68 :: v_dual_mov_b32 v53, v69
	v_dual_mov_b32 v54, v70 :: v_dual_mov_b32 v55, v71
	v_dual_mov_b32 v42, v66 :: v_dual_mov_b32 v43, v67
	v_dual_mov_b32 v44, v68 :: v_dual_mov_b32 v45, v69
	v_dual_mov_b32 v46, v70 :: v_dual_mov_b32 v47, v71
	v_dual_mov_b32 v34, v66 :: v_dual_mov_b32 v35, v67
	v_dual_mov_b32 v36, v68 :: v_dual_mov_b32 v37, v69
	v_dual_mov_b32 v38, v70 :: v_dual_mov_b32 v39, v71
	v_dual_mov_b32 v26, v66 :: v_dual_mov_b32 v27, v67
	v_dual_mov_b32 v28, v68 :: v_dual_mov_b32 v29, v69
	v_dual_mov_b32 v30, v70 :: v_dual_mov_b32 v31, v71
	v_dual_mov_b32 v18, v66 :: v_dual_mov_b32 v19, v67
	v_dual_mov_b32 v20, v68 :: v_dual_mov_b32 v21, v69
	v_dual_mov_b32 v22, v70 :: v_dual_mov_b32 v23, v71
	v_dual_mov_b32 v10, v66 :: v_dual_mov_b32 v11, v67
	v_dual_mov_b32 v12, v68 :: v_dual_mov_b32 v13, v69
	v_dual_mov_b32 v14, v70 :: v_dual_mov_b32 v15, v71
	v_dual_mov_b32 v16, v72 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v78, v6 :: v_dual_mov_b32 v75, v3
	v_dual_mov_b32 v76, v4 :: v_dual_mov_b32 v73, v1
	v_mov_b32_e32 v74, v2
	s_mov_b32 s17, 0x4e4c4a48
	s_mov_b32 s24, 0x4040404
	s_mov_b32 s25, 0
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_add_co_i32 s25, s25, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s25, s23
	s_cbranch_scc1 .LBB0_17
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
                                        ;       Child Loop BB0_10 Depth 3
	v_add_lshl_u32 v111, s25, v97, 1
	s_lshl_b32 s20, s25, 8
	s_mul_i32 s26, s25, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[10:11], s[20:21]
	s_mov_b32 s20, -1
	s_mov_b32 s29, 0
	s_branch .LBB0_8
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=2
	s_mov_b32 s29, 1
	s_and_b32 vcc_lo, exec_lo, s27
	s_mov_b32 s20, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_5
.LBB0_8:                                ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_10 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v7, s29, v111
	s_xor_b32 s27, s20, -1
	s_lshl_b32 s20, s29, 2
	s_mov_b32 s28, -1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s20, s20, s26
	v_lshlrev_b64_e32 v[1:2], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s13, v2, vcc_lo
	global_load_b32 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v1, 0, v1, s4
	ds_store_b32 v105, v1 offset:18432
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v98, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v99, vcc_lo
	s_lshl_b32 s20, s29, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[8:9], s[20:21]
	global_load_b32 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v100, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s5
	ds_store_b32 v106, v1 offset:18944
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s20, s30, v86
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s31, 0, s20
	global_load_b128 v[3:6], v[1:2], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	v_cndmask_b32_e64 v6, 0, v6, s0
	v_cndmask_b32_e64 v5, 0, v5, s0
	ds_store_2addr_b64 v109, v[3:4], v[5:6] offset1:16
	v_add_co_u32 v3, s20, s30, v85
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s31, 0, s20
	s_lshl_b32 s20, s29, 6
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s20, s26, s20
	global_load_b128 v[87:90], v[3:4], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v6, 0, v88, s2
	v_cndmask_b32_e64 v5, 0, v87, s2
	v_cndmask_b32_e64 v88, 0, v90, s2
	v_cndmask_b32_e64 v87, 0, v89, s2
	ds_store_2addr_b64 v110, v[5:6], v[87:88] offset1:16
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, vcc_lo, v101, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v102, vcc_lo
	global_load_b64 v[87:88], v[5:6], off offset:8
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v7, 0, v87, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v87, 0xf0f0f0f, v7
	v_lshrrev_b32_e32 v7, 4, v7
	v_and_b32_e32 v7, 0xf0f0f0f, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v7, v87, 0x5010400
	v_perm_b32 v7, v7, v87, 0x7030602
	v_and_b32_e32 v87, 0x7070707, v89
	v_lshrrev_b32_e32 v89, 1, v89
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v90, s17, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v89, v89, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v87, v90, v89
	v_and_b32_e32 v87, 0x7070707, v7
	v_lshrrev_b32_e32 v7, 1, v7
	v_perm_b32 v90, s17, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v7, v7, s24, 0x3020100
	v_perm_b32 v90, v87, v90, v7
	v_cndmask_b32_e64 v7, 0, v88, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v87, 0xf0f0f0f, v7
	v_lshrrev_b32_e32 v7, 4, v7
	v_and_b32_e32 v7, 0xf0f0f0f, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v88, v7, v87, 0x5010400
	v_perm_b32 v7, v7, v87, 0x7030602
	v_and_b32_e32 v87, 0x7070707, v88
	v_lshrrev_b32_e32 v88, 1, v88
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v91, s17, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v88, v88, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v87, v87, v91, v88
	v_and_b32_e32 v88, 0x7070707, v7
	v_lshrrev_b32_e32 v7, 1, v7
	v_perm_b32 v91, s17, 0x44403800, v88
	v_or_b32_e32 v88, 0x50505050, v88
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v7, v7, s24, 0x3020100
	v_perm_b32 v88, v88, v91, v7
	v_add_nc_u32_e32 v7, 0x2000, v109
	ds_store_2addr_b64 v7, v[89:90], v[87:88] offset0:128 offset1:144
	v_add_co_u32 v87, vcc_lo, v103, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v88, null, 0, v104, vcc_lo
	global_load_b64 v[89:90], v[87:88], off offset:8
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v89, 0, v89, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v91, 0xf0f0f0f, v89
	v_lshrrev_b32_e32 v89, 4, v89
	v_and_b32_e32 v89, 0xf0f0f0f, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v92, v89, v91, 0x5010400
	v_perm_b32 v89, v89, v91, 0x7030602
	v_and_b32_e32 v91, 0x7070707, v92
	v_lshrrev_b32_e32 v92, 1, v92
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v112, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	v_and_or_b32 v92, v92, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v91, v91, v112, v92
	v_and_b32_e32 v92, 0x7070707, v89
	v_lshrrev_b32_e32 v89, 1, v89
	v_perm_b32 v112, s17, 0x44403800, v92
	v_or_b32_e32 v92, 0x50505050, v92
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v89, v89, s24, 0x3020100
	v_perm_b32 v92, v92, v112, v89
	v_cndmask_b32_e64 v89, 0, v90, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v90, 0xf0f0f0f, v89
	v_lshrrev_b32_e32 v89, 4, v89
	v_and_b32_e32 v89, 0xf0f0f0f, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v112, v89, v90, 0x5010400
	v_perm_b32 v90, v89, v90, 0x7030602
	v_and_b32_e32 v89, 0x7070707, v112
	v_lshrrev_b32_e32 v112, 1, v112
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v113, s17, 0x44403800, v89
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v112, v112, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v89, v113, v112
	v_and_b32_e32 v112, 0x7070707, v90
	v_lshrrev_b32_e32 v90, 1, v90
	v_perm_b32 v113, s17, 0x44403800, v112
	v_or_b32_e32 v112, 0x50505050, v112
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v90, v90, s24, 0x3020100
	v_perm_b32 v90, v112, v113, v90
	v_add_nc_u32_e32 v112, 0x2000, v110
	ds_store_2addr_b64 v112, v[91:92], v[89:90] offset0:128 offset1:144
	s_branch .LBB0_10
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=3
	s_mov_b32 s29, -1
	s_and_b32 vcc_lo, exec_lo, s20
	s_mov_b32 s28, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_7
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_8 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_xor_b32 s20, s28, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_13
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_14
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
	s_branch .LBB0_15
.LBB0_13:                               ;   in Loop: Header=BB0_10 Depth=3
	s_clause 0x1
	global_load_b128 v[73:76], v[1:2], off offset:64
	global_load_b128 v[77:80], v[3:4], off offset:64
	s_clause 0x1
	global_load_b64 v[81:82], v[5:6], off offset:40
	global_load_b64 v[83:84], v[87:88], off offset:40
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v74, 0, v74, s0
	v_cndmask_b32_e64 v73, 0, v73, s0
	v_cndmask_b32_e64 v76, 0, v76, s0
	v_cndmask_b32_e64 v75, 0, v75, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v78, 0, v78, s2
	v_cndmask_b32_e64 v77, 0, v77, s2
	v_cndmask_b32_e64 v80, 0, v80, s2
	v_cndmask_b32_e64 v79, 0, v79, s2
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v81, 0, v81, s1
	v_cndmask_b32_e64 v82, 0, v82, s1
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v83, 0, v83, s3
	v_cndmask_b32_e64 v84, 0, v84, s3
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
.LBB0_14:                               ;   in Loop: Header=BB0_10 Depth=3
	v_lshrrev_b32_e32 v90, 4, v81
	v_and_b32_e32 v89, 0xf0f0f0f, v81
	ds_store_2addr_b64 v109, v[73:74], v[75:76] offset1:16
	ds_store_2addr_b64 v110, v[77:78], v[79:80] offset1:16
	v_and_b32_e32 v90, 0xf0f0f0f, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v91, v90, v89, 0x5010400
	v_perm_b32 v90, v90, v89, 0x7030602
	v_and_b32_e32 v89, 0x7070707, v91
	v_lshrrev_b32_e32 v91, 1, v91
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v92, s17, 0x44403800, v89
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v91, v91, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v89, v92, v91
	v_and_b32_e32 v91, 0x7070707, v90
	v_lshrrev_b32_e32 v90, 1, v90
	v_perm_b32 v92, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v90, v90, s24, 0x3020100
	v_perm_b32 v90, v91, v92, v90
	v_lshrrev_b32_e32 v92, 4, v82
	v_and_b32_e32 v91, 0xf0f0f0f, v82
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v92, 0xf0f0f0f, v92
	v_perm_b32 v113, v92, v91, 0x5010400
	v_perm_b32 v92, v92, v91, 0x7030602
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_b32_e32 v91, 0x7070707, v113
	v_lshrrev_b32_e32 v113, 1, v113
	v_perm_b32 v114, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v113, v113, s24, 0x3020100
	v_perm_b32 v91, v91, v114, v113
	v_and_b32_e32 v113, 0x7070707, v92
	v_lshrrev_b32_e32 v92, 1, v92
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v114, s17, 0x44403800, v113
	v_or_b32_e32 v113, 0x50505050, v113
	v_and_or_b32 v92, v92, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_perm_b32 v92, v113, v114, v92
	ds_store_2addr_b64 v7, v[89:90], v[91:92] offset0:128 offset1:144
	v_lshrrev_b32_e32 v90, 4, v83
	v_and_b32_e32 v89, 0xf0f0f0f, v83
	v_and_b32_e32 v90, 0xf0f0f0f, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v91, v90, v89, 0x5010400
	v_perm_b32 v90, v90, v89, 0x7030602
	v_and_b32_e32 v89, 0x7070707, v91
	v_lshrrev_b32_e32 v91, 1, v91
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v92, s17, 0x44403800, v89
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v91, v91, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v89, v92, v91
	v_and_b32_e32 v91, 0x7070707, v90
	v_lshrrev_b32_e32 v90, 1, v90
	v_perm_b32 v92, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v90, v90, s24, 0x3020100
	v_perm_b32 v90, v91, v92, v90
	v_lshrrev_b32_e32 v92, 4, v84
	v_and_b32_e32 v91, 0xf0f0f0f, v84
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v92, 0xf0f0f0f, v92
	v_perm_b32 v113, v92, v91, 0x5010400
	v_perm_b32 v92, v92, v91, 0x7030602
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_b32_e32 v91, 0x7070707, v113
	v_lshrrev_b32_e32 v113, 1, v113
	v_perm_b32 v114, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v113, v113, s24, 0x3020100
	v_perm_b32 v91, v91, v114, v113
	v_and_b32_e32 v113, 0x7070707, v92
	v_lshrrev_b32_e32 v92, 1, v92
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v114, s17, 0x44403800, v113
	v_or_b32_e32 v113, 0x50505050, v113
	v_and_or_b32 v92, v92, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v92, v113, v114, v92
	ds_store_2addr_b64 v112, v[89:90], v[91:92] offset0:128 offset1:144
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
.LBB0_15:                               ;   in Loop: Header=BB0_10 Depth=3
	v_add_nc_u32_e32 v89, 0x4800, v107
	ds_load_2addr_b32 v[91:92], v89 offset1:16
	ds_load_2addr_b32 v[89:90], v89 offset0:32 offset1:48
	v_add_nc_u32_e32 v113, 0x4a00, v108
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v65, 0, v113, v65
	v_fma_f32 v49, 0, v113, v49
	v_fma_f32 v33, 0, v113, v33
	v_fma_f32 v17, 0, v113, v17
	v_add_nc_u32_e32 v113, 0x4a08, v108
	v_fmac_f32_e32 v65, v114, v91
	v_fmac_f32_e32 v49, v114, v92
	v_fmac_f32_e32 v33, v114, v89
	v_fmac_f32_e32 v17, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v66, 0, v113, v66
	v_fma_f32 v50, 0, v113, v50
	v_fma_f32 v34, 0, v113, v34
	v_fma_f32 v18, 0, v113, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v66, v114, v91 :: v_dual_add_nc_u32 v113, 0x4a10, v108
	v_fmac_f32_e32 v50, v114, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v34, v114, v89
	v_fmac_f32_e32 v18, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v67, 0, v113, v67
	v_fma_f32 v51, 0, v113, v51
	v_fma_f32 v35, 0, v113, v35
	v_fma_f32 v19, 0, v113, v19
	v_add_nc_u32_e32 v113, 0x4a18, v108
	v_fmac_f32_e32 v67, v114, v91
	v_fmac_f32_e32 v51, v114, v92
	v_fmac_f32_e32 v35, v114, v89
	v_fmac_f32_e32 v19, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v68, 0, v113, v68
	v_fma_f32 v52, 0, v113, v52
	v_fma_f32 v36, 0, v113, v36
	v_fma_f32 v20, 0, v113, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v68, v114, v91 :: v_dual_add_nc_u32 v113, 0x4a20, v108
	v_fmac_f32_e32 v52, v114, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v36, v114, v89
	v_fmac_f32_e32 v20, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v69, 0, v113, v69
	v_fma_f32 v53, 0, v113, v53
	v_fma_f32 v37, 0, v113, v37
	v_fma_f32 v21, 0, v113, v21
	v_add_nc_u32_e32 v113, 0x4a28, v108
	v_fmac_f32_e32 v69, v114, v91
	v_fmac_f32_e32 v53, v114, v92
	v_fmac_f32_e32 v37, v114, v89
	v_fmac_f32_e32 v21, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v70, 0, v113, v70
	v_fma_f32 v54, 0, v113, v54
	v_fma_f32 v38, 0, v113, v38
	v_fma_f32 v22, 0, v113, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v70, v114, v91 :: v_dual_add_nc_u32 v113, 0x4a30, v108
	v_fmac_f32_e32 v54, v114, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v38, v114, v89
	v_fmac_f32_e32 v22, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v71, 0, v113, v71
	v_fma_f32 v55, 0, v113, v55
	v_fma_f32 v39, 0, v113, v39
	v_fma_f32 v23, 0, v113, v23
	v_add_nc_u32_e32 v113, 0x4a38, v108
	v_fmac_f32_e32 v71, v114, v91
	v_fmac_f32_e32 v55, v114, v92
	v_fmac_f32_e32 v39, v114, v89
	v_fmac_f32_e32 v23, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v72, 0, v113
	v_fmac_f32_e32 v56, 0, v113
	v_fmac_f32_e32 v40, 0, v113
	v_dual_fmac_f32 v24, 0, v113 :: v_dual_add_nc_u32 v113, 0x4a80, v108
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v72, v114, v91
	v_fmac_f32_e32 v56, v114, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v40, v114, v89
	v_fmac_f32_e32 v24, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v57, 0, v113, v57
	v_fma_f32 v41, 0, v113, v41
	v_fma_f32 v25, 0, v113, v25
	v_fma_f32 v9, 0, v113, v9
	v_add_nc_u32_e32 v113, 0x4a88, v108
	v_fmac_f32_e32 v57, v114, v91
	v_fmac_f32_e32 v41, v114, v92
	v_fmac_f32_e32 v25, v114, v89
	v_fmac_f32_e32 v9, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v58, 0, v113, v58
	v_fma_f32 v42, 0, v113, v42
	v_fma_f32 v26, 0, v113, v26
	v_fma_f32 v10, 0, v113, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v58, v114, v91 :: v_dual_add_nc_u32 v113, 0x4a90, v108
	v_fmac_f32_e32 v42, v114, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v26, v114, v89
	v_fmac_f32_e32 v10, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v59, 0, v113, v59
	v_fma_f32 v43, 0, v113, v43
	v_fma_f32 v27, 0, v113, v27
	v_fma_f32 v11, 0, v113, v11
	v_add_nc_u32_e32 v113, 0x4a98, v108
	v_fmac_f32_e32 v59, v114, v91
	v_fmac_f32_e32 v43, v114, v92
	v_fmac_f32_e32 v27, v114, v89
	v_fmac_f32_e32 v11, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v60, 0, v113, v60
	v_fma_f32 v44, 0, v113, v44
	v_fma_f32 v28, 0, v113, v28
	v_fma_f32 v12, 0, v113, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v60, v114, v91 :: v_dual_add_nc_u32 v113, 0x4aa0, v108
	v_fmac_f32_e32 v44, v114, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v28, v114, v89
	v_fmac_f32_e32 v12, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v61, 0, v113, v61
	v_fma_f32 v45, 0, v113, v45
	v_fma_f32 v29, 0, v113, v29
	v_fma_f32 v13, 0, v113, v13
	v_add_nc_u32_e32 v113, 0x4aa8, v108
	v_fmac_f32_e32 v61, v114, v91
	v_fmac_f32_e32 v45, v114, v92
	v_fmac_f32_e32 v29, v114, v89
	v_fmac_f32_e32 v13, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v62, 0, v113, v62
	v_fma_f32 v46, 0, v113, v46
	v_fma_f32 v30, 0, v113, v30
	v_fma_f32 v14, 0, v113, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v62, v114, v91 :: v_dual_add_nc_u32 v113, 0x4ab0, v108
	v_fmac_f32_e32 v46, v114, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v30, v114, v89
	v_fmac_f32_e32 v14, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fma_f32 v63, 0, v113, v63
	v_fma_f32 v47, 0, v113, v47
	v_fma_f32 v31, 0, v113, v31
	v_fma_f32 v15, 0, v113, v15
	v_add_nc_u32_e32 v113, 0x4ab8, v108
	v_fmac_f32_e32 v63, v114, v91
	v_fmac_f32_e32 v47, v114, v92
	v_fmac_f32_e32 v31, v114, v89
	v_fmac_f32_e32 v15, v114, v90
	ds_load_2addr_b32 v[113:114], v113 offset1:1
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v64, 0, v113
	v_fmac_f32_e32 v48, 0, v113
	v_fmac_f32_e32 v32, 0, v113
	v_fmac_f32_e32 v16, 0, v113
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v64, v114, v91
	v_fmac_f32_e32 v48, v114, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v32, v114, v89
	v_fmac_f32_e32 v16, v114, v90
	s_branch .LBB0_9
.LBB0_16:
	v_mov_b32_e32 v9, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v16, v9
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v11, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v15, v9
	v_mov_b32_e32 v24, v16
	v_mov_b32_e32 v32, v16
	v_mov_b32_e32 v40, v16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v48, v16 :: v_dual_mov_b32 v47, v15
	v_dual_mov_b32 v56, v16 :: v_dual_mov_b32 v55, v15
	v_dual_mov_b32 v64, v16 :: v_dual_mov_b32 v63, v15
	v_dual_mov_b32 v72, v16 :: v_dual_mov_b32 v71, v15
	v_dual_mov_b32 v23, v15 :: v_dual_mov_b32 v22, v14
	v_dual_mov_b32 v21, v13 :: v_dual_mov_b32 v20, v12
	v_dual_mov_b32 v19, v11 :: v_dual_mov_b32 v18, v10
	v_mov_b32_e32 v17, v9
	v_dual_mov_b32 v31, v15 :: v_dual_mov_b32 v30, v14
	v_dual_mov_b32 v29, v13 :: v_dual_mov_b32 v28, v12
	v_dual_mov_b32 v27, v11 :: v_dual_mov_b32 v26, v10
	v_mov_b32_e32 v25, v9
	v_dual_mov_b32 v39, v15 :: v_dual_mov_b32 v38, v14
	v_dual_mov_b32 v37, v13 :: v_dual_mov_b32 v36, v12
	v_dual_mov_b32 v35, v11 :: v_dual_mov_b32 v34, v10
	v_dual_mov_b32 v33, v9 :: v_dual_mov_b32 v46, v14
	v_dual_mov_b32 v45, v13 :: v_dual_mov_b32 v44, v12
	v_dual_mov_b32 v43, v11 :: v_dual_mov_b32 v42, v10
	v_dual_mov_b32 v41, v9 :: v_dual_mov_b32 v54, v14
	v_dual_mov_b32 v53, v13 :: v_dual_mov_b32 v52, v12
	v_dual_mov_b32 v51, v11 :: v_dual_mov_b32 v50, v10
	v_dual_mov_b32 v49, v9 :: v_dual_mov_b32 v62, v14
	v_dual_mov_b32 v61, v13 :: v_dual_mov_b32 v60, v12
	v_dual_mov_b32 v59, v11 :: v_dual_mov_b32 v58, v10
	v_dual_mov_b32 v57, v9 :: v_dual_mov_b32 v70, v14
	v_dual_mov_b32 v69, v13 :: v_dual_mov_b32 v68, v12
	v_dual_mov_b32 v67, v11 :: v_dual_mov_b32 v66, v10
	v_mov_b32_e32 v65, v9
	s_branch .LBB0_18
.LBB0_17:
	v_mov_b32_e32 v2, v96
.LBB0_18:
	v_lshlrev_b32_e32 v1, 6, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v2, v2, v95
	v_and_b32_e32 v5, 31, v0
	v_and_b32_e32 v1, 0x3800, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v3, 31, v2
	v_add_nc_u32_e32 v6, 18, v2
	v_add_nc_u32_e32 v7, 19, v2
	v_add_nc_u32_e32 v8, 20, v2
	v_add_nc_u32_e32 v4, 0, v1
	v_add_nc_u32_e32 v1, 17, v2
	v_xor_b32_e32 v3, 16, v3
	v_add_nc_u32_e32 v77, 21, v2
	v_add_nc_u32_e32 v78, 22, v2
	v_lshl_add_u32 v76, v95, 7, v4
	v_and_b32_e32 v1, 31, v1
	v_and_b32_e32 v75, 31, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v6, v2, 2, v76
	v_lshl_add_u32 v74, v1, 2, v76
	v_add_nc_u32_e32 v1, 23, v2
	v_and_b32_e32 v2, 31, v7
	v_lshl_add_u32 v73, v3, 2, v76
	v_and_b32_e32 v3, 31, v8
	v_and_b32_e32 v7, 31, v77
	v_and_b32_e32 v8, 31, v78
	v_and_b32_e32 v1, 31, v1
	v_lshl_add_u32 v75, v75, 2, v76
	ds_store_2addr_b32 v6, v65, v66 offset1:1
	ds_store_2addr_b32 v6, v67, v68 offset0:2 offset1:3
	ds_store_2addr_b32 v6, v69, v70 offset0:4 offset1:5
	ds_store_2addr_b32 v6, v71, v72 offset0:6 offset1:7
	ds_store_b32 v73, v57
	ds_store_b32 v74, v58
	ds_store_b32 v75, v59
	v_lshl_add_u32 v65, v2, 2, v76
	v_lshl_add_u32 v66, v3, 2, v76
	v_lshl_add_u32 v67, v7, 2, v76
	v_lshl_add_u32 v68, v8, 2, v76
	v_lshl_add_u32 v69, v1, 2, v76
	ds_store_b32 v65, v60
	ds_store_b32 v66, v61
	ds_store_b32 v67, v62
	ds_store_b32 v68, v63
	ds_store_b32 v69, v64
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_or3_b32 v3, s22, v93, v5
	v_add_nc_u32_e32 v1, s19, v94
	v_lshl_add_u32 v5, v5, 2, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s0, s16, v3
	v_cmp_gt_i32_e64 s1, s18, v1
	v_ashrrev_i32_e32 v2, 31, v1
	v_cmp_le_i32_e32 vcc_lo, s16, v3
	s_and_b32 s0, s1, s0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_mad_co_u64_u32 v[7:8], null, v1, s16, v[3:4]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[57:58], 2, v[1:2]
	ds_load_b32 v59, v5
	v_add_co_u32 v57, s0, s14, v57
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v58, null, s15, v58, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, s0, s6, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s7, v8, s0
	global_load_b32 v57, v[57:58], off
	global_load_b32 v58, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v58, v57, v59
	global_store_b32 v[7:8], v58, off
.LBB0_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v8, 1, v1
	v_add_nc_u32_e32 v7, 1, v0
	s_xor_b32 s1, vcc_lo, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s0, s18, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_mad_co_u64_u32 v[57:58], null, v8, s16, v[3:4]
	v_mov_b32_e32 v58, 0
	v_lshlrev_b64_e32 v[59:60], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v59, vcc_lo, s14, v59
	v_lshlrev_b64_e32 v[57:58], 2, v[57:58]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v60, null, s15, v60, vcc_lo
	v_add_co_u32 v57, vcc_lo, s6, v57
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v58, null, s7, v58, vcc_lo
	global_load_b32 v8, v[59:60], off offset:4
	global_load_b32 v59, v[57:58], off
	v_and_b32_e32 v60, 31, v7
	v_lshl_add_u32 v60, v60, 2, v4
	ds_load_b32 v60, v60 offset:128
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v59, v8, v60
	global_store_b32 v[57:58], v59, off
.LBB0_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v57, 2, v1
	v_add_nc_u32_e32 v8, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v57
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_mad_co_u64_u32 v[57:58], null, v57, s16, v[3:4]
	v_dual_mov_b32 v58, 0 :: v_dual_and_b32 v61, 31, v8
	v_lshlrev_b64_e32 v[59:60], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v61, v61, 2, v4
	v_lshlrev_b64_e32 v[57:58], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v59, vcc_lo, s14, v59
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v60, null, s15, v60, vcc_lo
	ds_load_b32 v61, v61 offset:256
	v_add_co_u32 v57, vcc_lo, s6, v57
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, s7, v58, vcc_lo
	global_load_b32 v59, v[59:60], off offset:8
	global_load_b32 v60, v[57:58], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v60, v59, v61
	global_store_b32 v[57:58], v60, off
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v58, 3, v1
	v_add_nc_u32_e32 v57, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v58
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_mad_co_u64_u32 v[58:59], null, v58, s16, v[3:4]
	v_dual_mov_b32 v59, 0 :: v_dual_and_b32 v62, 31, v57
	v_lshlrev_b64_e32 v[60:61], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v62, v62, 2, v4
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v60, vcc_lo, s14, v60
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v61, null, s15, v61, vcc_lo
	ds_load_b32 v62, v62 offset:384
	v_add_co_u32 v58, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, s7, v59, vcc_lo
	global_load_b32 v60, v[60:61], off offset:12
	global_load_b32 v61, v[58:59], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v61, v60, v62
	global_store_b32 v[58:59], v61, off
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v59, 4, v1
	v_add_nc_u32_e32 v58, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v59
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_mad_co_u64_u32 v[59:60], null, v59, s16, v[3:4]
	v_dual_mov_b32 v60, 0 :: v_dual_and_b32 v63, 31, v58
	v_lshlrev_b64_e32 v[61:62], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v63, v63, 2, v4
	v_lshlrev_b64_e32 v[59:60], 2, v[59:60]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v61, vcc_lo, s14, v61
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v62, null, s15, v62, vcc_lo
	ds_load_b32 v63, v63 offset:512
	v_add_co_u32 v59, vcc_lo, s6, v59
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v60, null, s7, v60, vcc_lo
	global_load_b32 v61, v[61:62], off offset:16
	global_load_b32 v62, v[59:60], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v62, v61, v63
	global_store_b32 v[59:60], v62, off
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v60, 5, v1
	v_add_nc_u32_e32 v59, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v60
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_mad_co_u64_u32 v[60:61], null, v60, s16, v[3:4]
	v_dual_mov_b32 v61, 0 :: v_dual_and_b32 v64, 31, v59
	v_lshlrev_b64_e32 v[62:63], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v64, v64, 2, v4
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v62, vcc_lo, s14, v62
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, s15, v63, vcc_lo
	ds_load_b32 v64, v64 offset:640
	v_add_co_u32 v60, vcc_lo, s6, v60
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v61, null, s7, v61, vcc_lo
	global_load_b32 v62, v[62:63], off offset:20
	global_load_b32 v63, v[60:61], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v63, v62, v64
	global_store_b32 v[60:61], v63, off
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v61, 6, v1
	v_add_nc_u32_e32 v60, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v61
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_mad_co_u64_u32 v[61:62], null, v61, s16, v[3:4]
	v_mov_b32_e32 v62, 0
	v_lshlrev_b64_e32 v[63:64], 2, v[1:2]
	v_and_b32_e32 v70, 31, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v70, v70, 2, v4
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v63, vcc_lo, s14, v63
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v64, null, s15, v64, vcc_lo
	ds_load_b32 v70, v70 offset:768
	v_add_co_u32 v61, vcc_lo, s6, v61
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v62, null, s7, v62, vcc_lo
	global_load_b32 v63, v[63:64], off offset:24
	global_load_b32 v64, v[61:62], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v64, v63, v70
	global_store_b32 v[61:62], v64, off
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v62, 7, v1
	v_add_nc_u32_e32 v61, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v62
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_mad_co_u64_u32 v[62:63], null, v62, s16, v[3:4]
	v_mov_b32_e32 v63, 0
	v_lshlrev_b64_e32 v[70:71], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v70, vcc_lo, s14, v70
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v71, null, s15, v71, vcc_lo
	v_add_co_u32 v62, vcc_lo, s6, v62
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v63, null, s7, v63, vcc_lo
	global_load_b32 v64, v[70:71], off offset:28
	global_load_b32 v70, v[62:63], off
	v_and_b32_e32 v71, 31, v61
	v_lshl_add_u32 v71, v71, 2, v4
	ds_load_b32 v71, v71 offset:896
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v70, v64, v71
	global_store_b32 v[62:63], v70, off
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v63, 8, v1
	v_add_nc_u32_e32 v62, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v63
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_and_b32_e32 v72, 31, v62
	v_mad_co_u64_u32 v[63:64], null, v63, s16, v[3:4]
	v_lshlrev_b64_e32 v[70:71], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v72, v72, 2, v4
	v_add_co_u32 v70, vcc_lo, s14, v70
	ds_load_b32 v72, v72 offset:1024
	v_mov_b32_e32 v64, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v71, null, s15, v71, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[63:64], 2, v[63:64]
	v_add_co_u32 v63, vcc_lo, s6, v63
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v64, null, s7, v64, vcc_lo
	global_load_b32 v70, v[70:71], off offset:32
	global_load_b32 v71, v[63:64], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v71, v70, v72
	global_store_b32 v[63:64], v71, off
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v64, 9, v1
	v_add_nc_u32_e32 v63, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v64
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_mad_co_u64_u32 v[70:71], null, v64, s16, v[3:4]
	v_mov_b32_e32 v71, 0
	v_lshlrev_b64_e32 v[76:77], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v76, vcc_lo, s14, v76
	v_lshlrev_b64_e32 v[70:71], 2, v[70:71]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, s15, v77, vcc_lo
	v_add_co_u32 v70, vcc_lo, s6, v70
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v71, null, s7, v71, vcc_lo
	global_load_b32 v64, v[76:77], off offset:36
	global_load_b32 v72, v[70:71], off
	v_and_b32_e32 v76, 31, v63
	v_lshl_add_u32 v76, v76, 2, v4
	ds_load_b32 v76, v76 offset:1152
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v72, v64, v76
	global_store_b32 v[70:71], v72, off
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v70, 10, v1
	v_add_nc_u32_e32 v64, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v70
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_mad_co_u64_u32 v[70:71], null, v70, s16, v[3:4]
	v_mov_b32_e32 v71, 0
	v_lshlrev_b64_e32 v[76:77], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v76, vcc_lo, s14, v76
	v_lshlrev_b64_e32 v[70:71], 2, v[70:71]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, s15, v77, vcc_lo
	v_add_co_u32 v70, vcc_lo, s6, v70
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v71, null, s7, v71, vcc_lo
	global_load_b32 v72, v[76:77], off offset:40
	global_load_b32 v76, v[70:71], off
	v_and_b32_e32 v77, 31, v64
	v_lshl_add_u32 v77, v77, 2, v4
	ds_load_b32 v77, v77 offset:1280
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v76, v72, v77
	global_store_b32 v[70:71], v76, off
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v71, 11, v1
	v_add_nc_u32_e32 v70, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v71
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_and_b32_e32 v78, 31, v70
	v_mad_co_u64_u32 v[71:72], null, v71, s16, v[3:4]
	v_lshlrev_b64_e32 v[76:77], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v78, v78, 2, v4
	v_add_co_u32 v76, vcc_lo, s14, v76
	ds_load_b32 v78, v78 offset:1408
	v_mov_b32_e32 v72, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s15, v77, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[71:72], 2, v[71:72]
	v_add_co_u32 v71, vcc_lo, s6, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v72, null, s7, v72, vcc_lo
	global_load_b32 v76, v[76:77], off offset:44
	global_load_b32 v77, v[71:72], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v77, v76, v78
	global_store_b32 v[71:72], v77, off
.LBB0_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v72, 12, v1
	v_add_nc_u32_e32 v71, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v72
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_mad_co_u64_u32 v[76:77], null, v72, s16, v[3:4]
	v_mov_b32_e32 v77, 0
	v_lshlrev_b64_e32 v[78:79], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v78, vcc_lo, s14, v78
	v_lshlrev_b64_e32 v[76:77], 2, v[76:77]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v79, null, s15, v79, vcc_lo
	v_add_co_u32 v76, vcc_lo, s6, v76
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v77, null, s7, v77, vcc_lo
	global_load_b32 v72, v[78:79], off offset:48
	global_load_b32 v78, v[76:77], off
	v_and_b32_e32 v79, 31, v71
	v_lshl_add_u32 v79, v79, 2, v4
	ds_load_b32 v79, v79 offset:1536
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v78, v72, v79
	global_store_b32 v[76:77], v78, off
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v76, 13, v1
	v_add_nc_u32_e32 v72, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v76
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_mad_co_u64_u32 v[76:77], null, v76, s16, v[3:4]
	v_dual_mov_b32 v77, 0 :: v_dual_and_b32 v80, 31, v72
	v_lshlrev_b64_e32 v[78:79], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v80, v80, 2, v4
	v_lshlrev_b64_e32 v[76:77], 2, v[76:77]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v78, vcc_lo, s14, v78
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, s15, v79, vcc_lo
	ds_load_b32 v80, v80 offset:1664
	v_add_co_u32 v76, vcc_lo, s6, v76
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s7, v77, vcc_lo
	global_load_b32 v78, v[78:79], off offset:52
	global_load_b32 v79, v[76:77], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v79, v78, v80
	global_store_b32 v[76:77], v79, off
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v77, 14, v1
	v_add_nc_u32_e32 v76, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v77
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_mad_co_u64_u32 v[77:78], null, v77, s16, v[3:4]
	v_dual_mov_b32 v78, 0 :: v_dual_and_b32 v81, 31, v76
	v_lshlrev_b64_e32 v[79:80], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v81, v81, 2, v4
	v_lshlrev_b64_e32 v[77:78], 2, v[77:78]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v79, vcc_lo, s14, v79
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v80, null, s15, v80, vcc_lo
	ds_load_b32 v81, v81 offset:1792
	v_add_co_u32 v77, vcc_lo, s6, v77
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, s7, v78, vcc_lo
	global_load_b32 v79, v[79:80], off offset:56
	global_load_b32 v80, v[77:78], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v80, v79, v81
	global_store_b32 v[77:78], v80, off
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v77, 15, v1
	v_add_nc_u32_e32 v0, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v77
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_mad_co_u64_u32 v[77:78], null, v77, s16, v[3:4]
	v_dual_mov_b32 v78, 0 :: v_dual_and_b32 v81, 31, v0
	v_lshlrev_b64_e32 v[79:80], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v81, v81, 2, v4
	v_lshlrev_b64_e32 v[77:78], 2, v[77:78]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v79, vcc_lo, s14, v79
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v80, null, s15, v80, vcc_lo
	ds_load_b32 v81, v81 offset:1920
	v_add_co_u32 v77, vcc_lo, s6, v77
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, s7, v78, vcc_lo
	global_load_b32 v79, v[79:80], off offset:60
	global_load_b32 v80, v[77:78], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v80, v79, v81
	global_store_b32 v[77:78], v80, off
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v49, v50 offset1:1
	ds_store_2addr_b32 v6, v51, v52 offset0:2 offset1:3
	ds_store_2addr_b32 v6, v53, v54 offset0:4 offset1:5
	ds_store_2addr_b32 v6, v55, v56 offset0:6 offset1:7
	ds_store_b32 v73, v41
	ds_store_b32 v74, v42
	ds_store_b32 v75, v43
	ds_store_b32 v65, v44
	ds_store_b32 v66, v45
	ds_store_b32 v67, v46
	ds_store_b32 v68, v47
	ds_store_b32 v69, v48
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v41, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	ds_load_b32 v45, v5
	v_add_co_u32 v43, vcc_lo, s14, v43
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:64
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 17, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v7
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:128
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:68
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v8
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:256
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:72
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 19, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v57
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:384
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:76
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 20, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v58
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:512
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:80
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 21, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v59
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:640
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:84
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 22, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v60
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:768
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:88
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 23, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v61
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:896
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:92
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 24, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v62
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:1024
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:96
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 25, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v63
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:1152
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:100
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 26, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v64
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:1280
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:104
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 27, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v70
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:1408
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:108
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 28, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v71
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:1536
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:112
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 29, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_78
; %bb.77:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v72
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:1664
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:116
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 30, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v76
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:1792
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:120
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 31, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_mad_co_u64_u32 v[41:42], null, v41, s16, v[3:4]
	v_dual_mov_b32 v42, 0 :: v_dual_and_b32 v45, 31, v0
	v_lshlrev_b64_e32 v[43:44], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v45, v45, 2, v4
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v43, vcc_lo, s14, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, s15, v44, vcc_lo
	ds_load_b32 v45, v45 offset:1920
	v_add_co_u32 v41, vcc_lo, s6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, s7, v42, vcc_lo
	global_load_b32 v43, v[43:44], off offset:124
	global_load_b32 v44, v[41:42], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v44, v43, v45
	global_store_b32 v[41:42], v44, off
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v33, v34 offset1:1
	ds_store_2addr_b32 v6, v35, v36 offset0:2 offset1:3
	ds_store_2addr_b32 v6, v37, v38 offset0:4 offset1:5
	ds_store_2addr_b32 v6, v39, v40 offset0:6 offset1:7
	ds_store_b32 v73, v25
	ds_store_b32 v74, v26
	ds_store_b32 v75, v27
	ds_store_b32 v65, v28
	ds_store_b32 v66, v29
	ds_store_b32 v67, v30
	ds_store_b32 v68, v31
	ds_store_b32 v69, v32
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v25, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	ds_load_b32 v29, v5
	v_add_co_u32 v27, vcc_lo, s14, v27
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:128
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 33, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v7
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:128
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:132
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 34, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v8
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:256
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:136
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 35, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v57
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:384
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:140
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 36, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_92
; %bb.91:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v58
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:512
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:144
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 37, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v59
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:640
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:148
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 38, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_96
; %bb.95:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v60
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:768
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:152
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 39, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v61
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:896
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:156
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 40, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v62
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1024
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:160
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 41, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v63
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1152
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:164
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 42, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v64
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1280
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:168
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 43, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v70
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1408
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:172
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 44, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v71
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1536
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:176
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 45, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v72
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1664
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:180
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 46, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v76
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1792
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:184
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 47, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_114
; %bb.113:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v0
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1920
	v_add_co_u32 v25, vcc_lo, s6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s7, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:188
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v17, v18 offset1:1
	ds_store_2addr_b32 v6, v19, v20 offset0:2 offset1:3
	ds_store_2addr_b32 v6, v21, v22 offset0:4 offset1:5
	ds_store_2addr_b32 v6, v23, v24 offset0:6 offset1:7
	ds_store_b32 v73, v9
	ds_store_b32 v74, v10
	ds_store_b32 v75, v11
	ds_store_b32 v65, v12
	ds_store_b32 v66, v13
	ds_store_b32 v67, v14
	ds_store_b32 v68, v15
	ds_store_b32 v69, v16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v6, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v6
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_116
; %bb.115:
	v_mad_co_u64_u32 v[9:10], null, v6, s16, v[3:4]
	ds_load_b32 v5, v5
	v_mov_b32_e32 v10, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v11, vcc_lo, s14, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v12, null, s15, v12, vcc_lo
	v_add_co_u32 v9, vcc_lo, s6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s7, v10, vcc_lo
	global_load_b32 v6, v[11:12], off offset:192
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v11, v6, v5
	global_store_b32 v[9:10], v11, off
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 49, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v7, 31, v7
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v7, v7, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v9, vcc_lo, s14, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s15, v10, vcc_lo
	ds_load_b32 v7, v7 offset:128
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v9, v[9:10], off offset:196
	global_load_b32 v10, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v10, v9, v7
	global_store_b32 v[5:6], v10, off
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_and_b32_e32 v8, 31, v8
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v8, v8, 2, v4
	v_add_co_u32 v9, vcc_lo, s14, v9
	ds_load_b32 v8, v8 offset:256
	v_mov_b32_e32 v6, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s15, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[9:10], off offset:200
	global_load_b32 v9, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v9, v7, v8
	global_store_b32 v[5:6], v9, off
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 51, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_122
; %bb.121:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v57
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:384
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:204
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 52, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_124
; %bb.123:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v58
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:512
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:208
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 53, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_126
; %bb.125:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v59
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:640
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:212
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 54, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_128
; %bb.127:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v60
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:768
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:216
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 55, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_130
; %bb.129:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v61
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:896
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:220
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 56, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_132
; %bb.131:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v62
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:1024
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:224
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 57, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_134
; %bb.133:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v63
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:1152
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:228
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 58, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_136
; %bb.135:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v64
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:1280
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:232
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 59, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_138
; %bb.137:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v70
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:1408
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:236
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_140
; %bb.139:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v71
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:1536
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:240
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 61, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v72
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:1664
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:244
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 62, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v76
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v9, v9, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v7, vcc_lo, s14, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	ds_load_b32 v9, v9 offset:1792
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:248
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_146
; %bb.145:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v0, v0, 2, v4
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v1, vcc_lo, s14, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s15, v2, vcc_lo
	ds_load_b32 v0, v0 offset:1920
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v1, v[1:2], off offset:252
	global_load_b32 v2, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v2, v1, v0
	global_store_b32 v[5:6], v2, off
.LBB0_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end0:
	.size	re_nocompute_nb, .Lfunc_end0-re_nocompute_nb
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel re_nocompute_nb
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 115
		.amdhsa_next_free_sgpr 32
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-re_nocompute_nb)<<4)&4080)>>4
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
                                        ; -- End function
	.set .Lre_nocompute_nb.num_vgpr, 115
	.set .Lre_nocompute_nb.num_agpr, 0
	.set .Lre_nocompute_nb.numbered_sgpr, 32
	.set .Lre_nocompute_nb.num_named_barrier, 0
	.set .Lre_nocompute_nb.private_seg_size, 0
	.set .Lre_nocompute_nb.uses_vcc, 1
	.set .Lre_nocompute_nb.uses_flat_scratch, 0
	.set .Lre_nocompute_nb.has_dyn_sized_stack, 0
	.set .Lre_nocompute_nb.has_recursion, 0
	.set .Lre_nocompute_nb.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16148
; TotalNumSgprs: 34
; NumVgprs: 115
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 115
; Occupancy: 12
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_5ef301fea5751346,@object ; @__hip_cuid_5ef301fea5751346
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5ef301fea5751346
__hip_cuid_5ef301fea5751346:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5ef301fea5751346, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5ef301fea5751346
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           re_nocompute_nb
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         re_nocompute_nb.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     115
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
