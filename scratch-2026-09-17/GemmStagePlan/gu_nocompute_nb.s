	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gu_nocompute_nb         ; -- Begin function gu_nocompute_nb
	.globl	gu_nocompute_nb
	.p2align	8
	.type	gu_nocompute_nb,@function
gu_nocompute_nb:                        ; @gu_nocompute_nb
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[24:27], s[0:1], 0x38
	s_load_b256 s[16:23], s[0:1], 0x0
	s_load_b256 s[8:15], s[0:1], 0x20
	v_bfe_u32 v2, v0, 4, 1
	v_lshrrev_b32_e32 v1, 1, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v96, 3, v2
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s26, 0xff
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	v_lshlrev_b32_e32 v2, 3, v2
	s_mov_b32 s0, 0
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b32 s0, -1
                                        ; implicit-def: $vgpr2
.LBB0_3:
	v_and_b32_e32 v94, 15, v0
	v_and_b32_e32 v95, 0x60, v0
	v_and_b32_e32 v93, 64, v1
	s_lshl_b32 s29, ttmp7, 7
	s_add_co_i32 s28, s25, s24
	s_lshl_b32 s30, ttmp9, 7
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s7, 0
	s_cbranch_vccnz .LBB0_16
; %bb.4:
	v_lshrrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v3, s17
	s_add_co_i32 s1, s28, -1
	v_dual_mov_b32 v5, s16 :: v_dual_and_b32 v6, 0x7f, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v4, s30, v2
	s_ashr_i32 s0, s26, 31
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v7, s29, v1
	s_lshr_b32 s0, s0, 24
	v_min_i32_e32 v10, s1, v4
	v_or_b32_e32 v15, s30, v6
	s_add_co_i32 s0, s26, s0
	s_add_co_i32 s6, s27, -1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s16, s0, 8
	v_cmp_gt_i32_e32 vcc_lo, s24, v10
	v_add_nc_u32_e32 v9, 64, v4
	v_min_i32_e32 v17, s1, v15
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s5, s16, 0x88
	v_min_i32_e32 v19, s6, v7
	v_cndmask_b32_e64 v14, s24, 0, vcc_lo
	v_min_i32_e32 v12, s1, v9
	v_cmp_gt_i32_e64 s1, s28, v4
	v_add_nc_u32_e32 v11, s29, v2
	v_or_b32_e32 v13, 64, v2
	v_sub_nc_u32_e32 v10, v10, v14
	v_cmp_gt_i32_e64 s4, s24, v12
	v_mul_lo_u32 v97, v19, s16
	v_mad_u32_u24 v19, 0x48, v2, 0
	v_mul_u32_u24_e32 v13, 0x48, v13
	v_mul_lo_u32 v4, s5, v10
	v_cndmask_b32_e64 v14, s24, 0, s4
	v_dual_mov_b32 v65, v8 :: v_dual_mov_b32 v70, v8
	v_dual_mov_b32 v67, v8 :: v_dual_mov_b32 v72, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_sub_nc_u32_e32 v10, v12, v14
	v_cndmask_b32_e32 v12, s18, v5, vcc_lo
	v_cndmask_b32_e32 v14, s19, v3, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s24, v17
	s_mov_b32 s17, 0x4e4c4a48
	v_mul_lo_u32 v10, s5, v10
	v_add_co_u32 v4, s3, v12, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v18, s24, 0, vcc_lo
	v_cndmask_b32_e64 v12, s19, v3, s4
	v_cndmask_b32_e32 v3, s19, v3, vcc_lo
	v_add_co_ci_u32_e64 v14, null, 0, v14, s3
	v_cmp_gt_i32_e64 s3, s28, v9
	v_cndmask_b32_e64 v9, s18, v5, s4
	v_sub_nc_u32_e32 v17, v17, v18
	s_mov_b32 s19, 0
	v_mov_b32_e32 v68, v8
	v_mov_b32_e32 v66, v8
	v_add_co_u32 v9, s4, v9, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v12, s4
	v_mul_lo_u32 v12, s5, v17
	v_cmp_gt_i32_e64 s4, s27, v7
	v_cmp_gt_i32_e64 s5, s28, v15
	v_and_b32_e32 v7, 3, v0
	v_mul_u32_u24_e32 v15, 0x48, v2
	v_dual_mov_b32 v2, v8 :: v_dual_cndmask_b32 v5, s18, v5
	v_dual_mov_b32 v6, v8 :: v_dual_lshlrev_b32 v17, 3, v6
	s_mov_b32 s18, 0x4040404
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v98, vcc_lo, v5, v12
	v_lshlrev_b32_e32 v12, 4, v7
	v_lshlrev_b32_e32 v7, 3, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, 0, v3, vcc_lo
	v_lshrrev_b32_e32 v5, 7, v0
	v_add_nc_u32_e32 v20, 0, v12
	v_add_co_u32 v101, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v102, null, 0, v14, vcc_lo
	v_mov_b32_e32 v4, v8
	v_add_co_u32 v103, vcc_lo, v9, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v104, null, 0, v10, vcc_lo
	v_mov_b32_e32 v10, v8
	v_dual_mov_b32 v9, v8 :: v_dual_add_nc_u32 v16, 64, v11
	v_cmp_gt_i32_e64 s0, s27, v11
	v_min_i32_e32 v11, s6, v11
	v_lshl_add_u32 v18, v5, 2, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_i32_e32 v3, s6, v16
	v_cmp_gt_i32_e64 s2, s27, v16
	v_lshlrev_b32_e32 v16, 2, v1
	v_or_b32_e32 v1, v93, v94
	v_dual_mov_b32 v69, v8 :: v_dual_add_nc_u32 v106, v18, v17
	v_mad_co_u64_u32 v[85:86], null, v3, s26, v[12:13]
	v_or_b32_e32 v3, v96, v95
	v_mad_co_u64_u32 v[86:87], null, v11, s26, v[12:13]
	v_mov_b32_e32 v11, v8
	v_lshlrev_b32_e32 v21, 2, v1
	v_dual_mov_b32 v1, v8 :: v_dual_lshlrev_b32 v100, 4, v5
	v_dual_mov_b32 v5, v8 :: v_dual_lshlrev_b32 v22, 3, v3
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v7, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v71, v8 :: v_dual_add_nc_u32 v108, 0, v22
	v_dual_mov_b32 v25, v65 :: v_dual_mov_b32 v32, v72
	v_dual_mov_b32 v27, v67 :: v_dual_mov_b32 v84, v11
	v_add_nc_u32_e32 v105, 0, v16
	v_add_nc_u32_e32 v109, v19, v12
	v_add3_u32 v110, v20, v15, 0x2400
	v_add3_u32 v111, v20, v13, 0x2400
	v_dual_mov_b32 v29, v69 :: v_dual_mov_b32 v82, v9
	v_mov_b32_e32 v83, v10
	v_mov_b32_e32 v81, v8
	v_dual_mov_b32 v9, v65 :: v_dual_mov_b32 v16, v72
	v_add_nc_u32_e32 v107, 0, v21
	v_dual_mov_b32 v57, v65 :: v_dual_mov_b32 v64, v72
	v_dual_mov_b32 v49, v65 :: v_dual_mov_b32 v56, v72
	v_dual_mov_b32 v41, v65 :: v_dual_mov_b32 v48, v72
	v_dual_mov_b32 v33, v65 :: v_dual_mov_b32 v40, v72
	v_dual_mov_b32 v17, v65 :: v_dual_mov_b32 v24, v72
	v_dual_mov_b32 v11, v67 :: v_dual_mov_b32 v80, v8
	v_dual_mov_b32 v58, v66 :: v_dual_mov_b32 v59, v67
	v_mov_b32_e32 v50, v66
	v_dual_mov_b32 v60, v68 :: v_dual_mov_b32 v61, v69
	v_mov_b32_e32 v52, v68
	v_dual_mov_b32 v62, v70 :: v_dual_mov_b32 v63, v71
	v_dual_mov_b32 v54, v70 :: v_dual_mov_b32 v51, v67
	v_dual_mov_b32 v42, v66 :: v_dual_mov_b32 v53, v69
	v_dual_mov_b32 v44, v68 :: v_dual_mov_b32 v55, v71
	v_dual_mov_b32 v46, v70 :: v_dual_mov_b32 v43, v67
	v_dual_mov_b32 v34, v66 :: v_dual_mov_b32 v45, v69
	v_dual_mov_b32 v36, v68 :: v_dual_mov_b32 v47, v71
	v_dual_mov_b32 v38, v70 :: v_dual_mov_b32 v35, v67
	v_dual_mov_b32 v26, v66 :: v_dual_mov_b32 v37, v69
	v_dual_mov_b32 v28, v68 :: v_dual_mov_b32 v39, v71
	v_dual_mov_b32 v30, v70 :: v_dual_mov_b32 v31, v71
	v_dual_mov_b32 v18, v66 :: v_dual_mov_b32 v19, v67
	v_mov_b32_e32 v10, v66
	v_dual_mov_b32 v20, v68 :: v_dual_mov_b32 v21, v69
	v_mov_b32_e32 v12, v68
	v_dual_mov_b32 v22, v70 :: v_dual_mov_b32 v23, v71
	v_dual_mov_b32 v14, v70 :: v_dual_mov_b32 v13, v69
	v_dual_mov_b32 v78, v6 :: v_dual_mov_b32 v15, v71
	v_dual_mov_b32 v76, v4 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v74, v2 :: v_dual_mov_b32 v77, v5
	v_mov_b32_e32 v75, v3
	v_mov_b32_e32 v73, v1
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_add_co_i32 s19, s19, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s19, s16
	s_cbranch_scc1 .LBB0_17
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
                                        ;       Child Loop BB0_10 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	v_add_lshl_u32 v112, s19, v97, 1
	s_lshl_b32 s6, s19, 8
	s_mul_i32 s26, s19, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[14:15], s[20:21], s[6:7]
	s_mov_b32 s6, -1
	s_mov_b32 s34, 0
	s_branch .LBB0_8
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=2
	s_mov_b32 s34, 1
	s_and_b32 vcc_lo, exec_lo, s31
	s_mov_b32 s6, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_5
.LBB0_8:                                ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_10 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v7, s34, v112
	s_xor_b32 s31, s6, -1
	s_lshl_b32 s6, s34, 2
	s_mov_b32 s33, -1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s6, s26
	v_lshlrev_b64_e32 v[1:2], 2, v[7:8]
	v_add_nc_u32_e32 v7, 0x1200, v109
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s22, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s23, v2, vcc_lo
	global_load_b32 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v1, 0, v1, s4
	ds_store_b32 v105, v1 offset:18432
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v98, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v99, vcc_lo
	s_lshl_b32 s6, s34, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[14:15], s[6:7]
	global_load_b32 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v100, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s5
	ds_store_b32 v106, v1 offset:18944
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s6, s36, v86
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s6
	global_load_b128 v[3:6], v[1:2], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	v_cndmask_b32_e64 v6, 0, v6, s0
	v_cndmask_b32_e64 v5, 0, v5, s0
	ds_store_2addr_b64 v109, v[3:4], v[5:6] offset1:1
	v_add_co_u32 v3, s6, s36, v85
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s37, 0, s6
	s_lshl_b32 s6, s34, 6
	s_mov_b32 s34, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s26, s6
	global_load_b128 v[87:90], v[3:4], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v6, 0, v88, s2
	v_cndmask_b32_e64 v5, 0, v87, s2
	v_cndmask_b32_e64 v88, 0, v90, s2
	v_cndmask_b32_e64 v87, 0, v89, s2
	ds_store_2addr_b64 v7, v[5:6], v[87:88] offset1:1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, vcc_lo, v101, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v102, vcc_lo
	global_load_b64 v[87:88], v[5:6], off offset:8
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v87, 0, v87, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v89, 0xf0f0f0f, v87
	v_lshrrev_b32_e32 v87, 4, v87
	v_and_b32_e32 v87, 0xf0f0f0f, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v90, v87, v89, 0x5010400
	v_perm_b32 v87, v87, v89, 0x7030602
	v_and_b32_e32 v89, 0x7070707, v90
	v_lshrrev_b32_e32 v90, 1, v90
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v91, s17, 0x44403800, v89
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v90, v90, s18, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v89, v91, v90
	v_and_b32_e32 v90, 0x7070707, v87
	v_lshrrev_b32_e32 v87, 1, v87
	v_perm_b32 v91, s17, 0x44403800, v90
	v_or_b32_e32 v90, 0x50505050, v90
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v87, v87, s18, 0x3020100
	v_perm_b32 v90, v90, v91, v87
	v_cndmask_b32_e64 v87, 0, v88, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v88, 0xf0f0f0f, v87
	v_lshrrev_b32_e32 v87, 4, v87
	v_and_b32_e32 v87, 0xf0f0f0f, v87
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v91, v87, v88, 0x5010400
	v_perm_b32 v88, v87, v88, 0x7030602
	v_and_b32_e32 v87, 0x7070707, v91
	v_lshrrev_b32_e32 v91, 1, v91
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v92, s17, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v91, v91, s18, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v87, v87, v92, v91
	v_and_b32_e32 v91, 0x7070707, v88
	v_lshrrev_b32_e32 v88, 1, v88
	v_perm_b32 v92, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v88, v88, s18, 0x3020100
	v_perm_b32 v88, v91, v92, v88
	ds_store_2addr_b64 v110, v[89:90], v[87:88] offset1:1
	v_add_co_u32 v87, vcc_lo, v103, s6
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
	v_perm_b32 v113, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	v_and_or_b32 v92, v92, s18, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v91, v91, v113, v92
	v_and_b32_e32 v92, 0x7070707, v89
	v_lshrrev_b32_e32 v89, 1, v89
	v_perm_b32 v113, s17, 0x44403800, v92
	v_or_b32_e32 v92, 0x50505050, v92
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v89, v89, s18, 0x3020100
	v_perm_b32 v92, v92, v113, v89
	v_cndmask_b32_e64 v89, 0, v90, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v90, 0xf0f0f0f, v89
	v_lshrrev_b32_e32 v89, 4, v89
	v_and_b32_e32 v89, 0xf0f0f0f, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v113, v89, v90, 0x5010400
	v_perm_b32 v90, v89, v90, 0x7030602
	v_and_b32_e32 v89, 0x7070707, v113
	v_lshrrev_b32_e32 v113, 1, v113
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v114, s17, 0x44403800, v89
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v113, v113, s18, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v89, v114, v113
	v_and_b32_e32 v113, 0x7070707, v90
	v_lshrrev_b32_e32 v90, 1, v90
	v_perm_b32 v114, s17, 0x44403800, v113
	v_or_b32_e32 v113, 0x50505050, v113
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v90, v90, s18, 0x3020100
	v_perm_b32 v90, v113, v114, v90
	ds_store_2addr_b64 v111, v[91:92], v[89:90] offset1:1
	s_branch .LBB0_10
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=3
	s_mov_b32 s34, -1
	s_and_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s33, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_7
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_8 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_xor_b32 s6, s33, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_13
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s33
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_14
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
	s_branch .LBB0_15
.LBB0_13:                               ;   in Loop: Header=BB0_10 Depth=3
	s_clause 0x1
	global_load_b128 v[73:76], v[1:2], off offset:64
	global_load_b128 v[77:80], v[3:4], off offset:64
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
	s_and_not1_b32 vcc_lo, exec_lo, s33
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
.LBB0_14:                               ;   in Loop: Header=BB0_10 Depth=3
	v_lshrrev_b32_e32 v90, 4, v81
	v_and_b32_e32 v89, 0xf0f0f0f, v81
	ds_store_2addr_b64 v109, v[73:74], v[75:76] offset1:1
	ds_store_2addr_b64 v7, v[77:78], v[79:80] offset1:1
	v_and_b32_e32 v90, 0xf0f0f0f, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v91, v90, v89, 0x5010400
	v_perm_b32 v90, v90, v89, 0x7030602
	v_and_b32_e32 v89, 0x7070707, v91
	v_lshrrev_b32_e32 v91, 1, v91
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v92, s17, 0x44403800, v89
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v91, v91, s18, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v89, v92, v91
	v_and_b32_e32 v91, 0x7070707, v90
	v_lshrrev_b32_e32 v90, 1, v90
	v_perm_b32 v92, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v90, v90, s18, 0x3020100
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
	v_and_or_b32 v113, v113, s18, 0x3020100
	v_perm_b32 v91, v91, v114, v113
	v_and_b32_e32 v113, 0x7070707, v92
	v_lshrrev_b32_e32 v92, 1, v92
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v114, s17, 0x44403800, v113
	v_or_b32_e32 v113, 0x50505050, v113
	v_and_or_b32 v92, v92, s18, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_perm_b32 v92, v113, v114, v92
	ds_store_2addr_b64 v110, v[89:90], v[91:92] offset1:1
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
	v_and_or_b32 v91, v91, s18, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v89, v89, v92, v91
	v_and_b32_e32 v91, 0x7070707, v90
	v_lshrrev_b32_e32 v90, 1, v90
	v_perm_b32 v92, s17, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v90, v90, s18, 0x3020100
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
	v_and_or_b32 v113, v113, s18, 0x3020100
	v_perm_b32 v91, v91, v114, v113
	v_and_b32_e32 v113, 0x7070707, v92
	v_lshrrev_b32_e32 v92, 1, v92
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v114, s17, 0x44403800, v113
	v_or_b32_e32 v113, 0x50505050, v113
	v_and_or_b32 v92, v92, s18, 0x3020100
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v92, v113, v114, v92
	ds_store_2addr_b64 v111, v[89:90], v[91:92] offset1:1
	s_and_not1_b32 vcc_lo, exec_lo, s34
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
	v_add_nc_u32_e32 v2, v2, v94
	v_and_b32_e32 v76, 31, v0
	v_and_b32_e32 v1, 0x3800, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_and_b32_e32 v5, 31, v2
	v_add_nc_u32_e32 v6, 17, v2
	v_add_nc_u32_e32 v7, 18, v2
	v_add_nc_u32_e32 v4, 0, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v5, 16, v5
	v_mov_b32_e32 v1, s11
	v_or3_b32 v3, s30, v95, v76
	v_and_b32_e32 v6, 31, v6
	v_lshl_add_u32 v77, v94, 7, v4
	v_and_b32_e32 v7, 31, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cmp_gt_i32_e64 s0, s24, v3
	v_cmp_gt_i32_e64 s1, s28, v3
	v_lshl_add_u32 v73, v5, 2, v77
	v_add_nc_u32_e32 v5, 20, v2
	v_lshl_add_u32 v8, v2, 2, v77
	v_lshl_add_u32 v74, v6, 2, v77
	v_lshl_add_u32 v75, v7, 2, v77
	ds_store_2addr_b32 v8, v65, v66 offset1:1
	ds_store_2addr_b32 v8, v67, v68 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v69, v70 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v71, v72 offset0:6 offset1:7
	ds_store_b32 v73, v57
	ds_store_b32 v74, v58
	ds_store_b32 v75, v59
	v_and_b32_e32 v5, 31, v5
	v_cmp_le_i32_e32 vcc_lo, s28, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u32 v59, v5, 2, v77
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v5, s13, v1, s0
	v_dual_mov_b32 v1, s10 :: v_dual_add_nc_u32 v78, 19, v2
	v_add_nc_u32_e32 v6, 21, v2
	v_add_nc_u32_e32 v7, 22, v2
	v_add_nc_u32_e32 v2, 23, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v57, 31, v78
	v_and_b32_e32 v6, 31, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v7, 31, v7
	v_and_b32_e32 v2, 31, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v58, v57, 2, v77
	v_lshl_add_u32 v65, v6, 2, v77
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v66, v7, 2, v77
	v_lshl_add_u32 v67, v2, 2, v77
	ds_store_b32 v58, v60
	ds_store_b32 v59, v61
	ds_store_b32 v65, v62
	ds_store_b32 v66, v63
	ds_store_b32 v67, v64
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v2, s29, v93
	v_cndmask_b32_e64 v7, s24, 0, s0
	v_mov_b32_e32 v57, s24
	v_cndmask_b32_e64 v6, s12, v1, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_nc_u32_e32 v1, v3, v7
	v_cndmask_b32_e64 v7, s25, v57, s0
	v_cmp_gt_i32_e64 s0, s27, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u32 v57, v76, 2, v4
	s_and_b32 s0, s0, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_lshlrev_b64_e32 v[60:61], 2, v[2:3]
	ds_load_b32 v63, v57
	v_add_co_u32 v60, s0, s8, v60
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v61, null, s9, v61, s0
	global_load_b32 v62, v[60:61], off
	v_mad_co_u64_u32 v[60:61], null, v2, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v61, 0 :: v_dual_mul_f32 v62, v62, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	v_add_co_u32 v60, s0, v6, v60
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, v5, v61, s0
	global_store_b32 v[60:61], v62, off
.LBB0_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v61, 1, v2
	v_add_nc_u32_e32 v60, 1, v0
	s_xor_b32 s1, vcc_lo, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s0, s27, v61
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_lshlrev_b64_e32 v[62:63], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, vcc_lo, s8, v62
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, s9, v63, vcc_lo
	global_load_b32 v63, v[62:63], off offset:4
	v_and_b32_e32 v62, 31, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v62, v62, 2, v4
	ds_load_b32 v64, v62 offset:128
	v_mad_co_u64_u32 v[61:62], null, v61, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v62, 0 :: v_dual_mul_f32 v63, v63, v64
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v61, vcc_lo, v6, v61
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v62, null, v5, v62, vcc_lo
	global_store_b32 v[61:62], v63, off
.LBB0_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v62, 2, v2
	v_add_nc_u32_e32 v61, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v62
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_lshlrev_b64_e32 v[63:64], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v63, vcc_lo, s8, v63
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v64, null, s9, v64, vcc_lo
	global_load_b32 v64, v[63:64], off offset:8
	v_and_b32_e32 v63, 31, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v63, v63, 2, v4
	ds_load_b32 v68, v63 offset:256
	v_mad_co_u64_u32 v[62:63], null, v62, v7, v[1:2]
	v_mov_b32_e32 v63, 0
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, vcc_lo, v6, v62
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, v5, v63, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v64, v64, v68
	global_store_b32 v[62:63], v64, off
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v63, 3, v2
	v_add_nc_u32_e32 v62, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v63
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_lshlrev_b64_e32 v[68:69], 2, v[2:3]
	v_and_b32_e32 v64, 31, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v64, v64, 2, v4
	v_add_co_u32 v68, vcc_lo, s8, v68
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v69, null, s9, v69, vcc_lo
	global_load_b32 v68, v[68:69], off offset:12
	ds_load_b32 v69, v64 offset:384
	v_mad_co_u64_u32 v[63:64], null, v63, v7, v[1:2]
	v_mov_b32_e32 v64, 0
	v_lshlrev_b64_e32 v[63:64], 2, v[63:64]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v63, vcc_lo, v6, v63
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v64, null, v5, v64, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v68, v68, v69
	global_store_b32 v[63:64], v68, off
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v64, 4, v2
	v_add_nc_u32_e32 v63, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v64
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_lshlrev_b64_e32 v[68:69], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v68, vcc_lo, s8, v68
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v69, null, s9, v69, vcc_lo
	global_load_b32 v70, v[68:69], off offset:16
	v_and_b32_e32 v68, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v68, v68, 2, v4
	ds_load_b32 v71, v68 offset:512
	v_mad_co_u64_u32 v[68:69], null, v64, v7, v[1:2]
	v_mov_b32_e32 v69, 0
	v_lshlrev_b64_e32 v[68:69], 2, v[68:69]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v68, vcc_lo, v6, v68
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v69, null, v5, v69, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v64, v70, v71
	global_store_b32 v[68:69], v64, off
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v68, 5, v2
	v_add_nc_u32_e32 v64, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v68
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_lshlrev_b64_e32 v[69:70], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v69, vcc_lo, s8, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s9, v70, vcc_lo
	global_load_b32 v70, v[69:70], off offset:20
	v_and_b32_e32 v69, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v69, v69, 2, v4
	ds_load_b32 v71, v69 offset:640
	v_mad_co_u64_u32 v[68:69], null, v68, v7, v[1:2]
	v_mov_b32_e32 v69, 0
	v_lshlrev_b64_e32 v[68:69], 2, v[68:69]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v68, vcc_lo, v6, v68
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v69, null, v5, v69, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v70, v70, v71
	global_store_b32 v[68:69], v70, off
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v69, 6, v2
	v_add_nc_u32_e32 v68, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v69
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_lshlrev_b64_e32 v[70:71], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v70, vcc_lo, s8, v70
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v71, null, s9, v71, vcc_lo
	global_load_b32 v71, v[70:71], off offset:24
	v_and_b32_e32 v70, 31, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v70, v70, 2, v4
	ds_load_b32 v72, v70 offset:768
	v_mad_co_u64_u32 v[69:70], null, v69, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v70, 0 :: v_dual_mul_f32 v71, v71, v72
	v_lshlrev_b64_e32 v[69:70], 2, v[69:70]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v69, vcc_lo, v6, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, v5, v70, vcc_lo
	global_store_b32 v[69:70], v71, off
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v70, 7, v2
	v_add_nc_u32_e32 v69, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v70
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_lshlrev_b64_e32 v[71:72], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v71, vcc_lo, s8, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, s9, v72, vcc_lo
	global_load_b32 v72, v[71:72], off offset:28
	v_and_b32_e32 v71, 31, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v71, v71, 2, v4
	ds_load_b32 v76, v71 offset:896
	v_mad_co_u64_u32 v[70:71], null, v70, v7, v[1:2]
	v_mov_b32_e32 v71, 0
	v_lshlrev_b64_e32 v[70:71], 2, v[70:71]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v70, vcc_lo, v6, v70
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v71, null, v5, v71, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v72, v72, v76
	global_store_b32 v[70:71], v72, off
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v71, 8, v2
	v_add_nc_u32_e32 v70, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v71
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_lshlrev_b64_e32 v[76:77], 2, v[2:3]
	v_and_b32_e32 v72, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v72, v72, 2, v4
	v_add_co_u32 v76, vcc_lo, s8, v76
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v77, null, s9, v77, vcc_lo
	global_load_b32 v76, v[76:77], off offset:32
	ds_load_b32 v77, v72 offset:1024
	v_mad_co_u64_u32 v[71:72], null, v71, v7, v[1:2]
	v_mov_b32_e32 v72, 0
	v_lshlrev_b64_e32 v[71:72], 2, v[71:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v71, vcc_lo, v6, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, v5, v72, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v76, v76, v77
	global_store_b32 v[71:72], v76, off
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v72, 9, v2
	v_add_nc_u32_e32 v71, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v72
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_lshlrev_b64_e32 v[76:77], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v76, vcc_lo, s8, v76
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s9, v77, vcc_lo
	global_load_b32 v78, v[76:77], off offset:36
	v_and_b32_e32 v76, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v76, v76, 2, v4
	ds_load_b32 v79, v76 offset:1152
	v_mad_co_u64_u32 v[76:77], null, v72, v7, v[1:2]
	v_mov_b32_e32 v77, 0
	v_lshlrev_b64_e32 v[76:77], 2, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v76, vcc_lo, v6, v76
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, v5, v77, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v72, v78, v79
	global_store_b32 v[76:77], v72, off
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v76, 10, v2
	v_add_nc_u32_e32 v72, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v76
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_lshlrev_b64_e32 v[77:78], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v77, vcc_lo, s8, v77
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, s9, v78, vcc_lo
	global_load_b32 v78, v[77:78], off offset:40
	v_and_b32_e32 v77, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v77, v77, 2, v4
	ds_load_b32 v79, v77 offset:1280
	v_mad_co_u64_u32 v[76:77], null, v76, v7, v[1:2]
	v_mov_b32_e32 v77, 0
	v_lshlrev_b64_e32 v[76:77], 2, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v76, vcc_lo, v6, v76
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, v5, v77, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v78, v78, v79
	global_store_b32 v[76:77], v78, off
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v77, 11, v2
	v_add_nc_u32_e32 v76, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v77
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_lshlrev_b64_e32 v[78:79], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v78, vcc_lo, s8, v78
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, s9, v79, vcc_lo
	global_load_b32 v79, v[78:79], off offset:44
	v_and_b32_e32 v78, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v78, v78, 2, v4
	ds_load_b32 v80, v78 offset:1408
	v_mad_co_u64_u32 v[77:78], null, v77, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v78, 0 :: v_dual_mul_f32 v79, v79, v80
	v_lshlrev_b64_e32 v[77:78], 2, v[77:78]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v77, vcc_lo, v6, v77
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, v5, v78, vcc_lo
	global_store_b32 v[77:78], v79, off
.LBB0_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v78, 12, v2
	v_add_nc_u32_e32 v77, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v78
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_lshlrev_b64_e32 v[79:80], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v79, vcc_lo, s8, v79
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v80, null, s9, v80, vcc_lo
	global_load_b32 v80, v[79:80], off offset:48
	v_and_b32_e32 v79, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v79, v79, 2, v4
	ds_load_b32 v81, v79 offset:1536
	v_mad_co_u64_u32 v[78:79], null, v78, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v79, 0 :: v_dual_mul_f32 v80, v80, v81
	v_lshlrev_b64_e32 v[78:79], 2, v[78:79]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v78, vcc_lo, v6, v78
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, v5, v79, vcc_lo
	global_store_b32 v[78:79], v80, off
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v79, 13, v2
	v_add_nc_u32_e32 v78, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v79
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_lshlrev_b64_e32 v[80:81], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v80, vcc_lo, s8, v80
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, s9, v81, vcc_lo
	global_load_b32 v81, v[80:81], off offset:52
	v_and_b32_e32 v80, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v80, v80, 2, v4
	ds_load_b32 v82, v80 offset:1664
	v_mad_co_u64_u32 v[79:80], null, v79, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v80, 0 :: v_dual_mul_f32 v81, v81, v82
	v_lshlrev_b64_e32 v[79:80], 2, v[79:80]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v79, vcc_lo, v6, v79
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v80, null, v5, v80, vcc_lo
	global_store_b32 v[79:80], v81, off
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v80, 14, v2
	v_add_nc_u32_e32 v79, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v80
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_lshlrev_b64_e32 v[81:82], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v81, vcc_lo, s8, v81
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, s9, v82, vcc_lo
	global_load_b32 v82, v[81:82], off offset:56
	v_and_b32_e32 v81, 31, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v81, v81, 2, v4
	ds_load_b32 v83, v81 offset:1792
	v_mad_co_u64_u32 v[80:81], null, v80, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v81, 0 :: v_dual_mul_f32 v82, v82, v83
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v80, vcc_lo, v6, v80
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, v5, v81, vcc_lo
	global_store_b32 v[80:81], v82, off
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v80, 15, v2
	v_add_nc_u32_e32 v0, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v80
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_lshlrev_b64_e32 v[81:82], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v81, vcc_lo, s8, v81
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, s9, v82, vcc_lo
	global_load_b32 v82, v[81:82], off offset:60
	v_and_b32_e32 v81, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v81, v81, 2, v4
	ds_load_b32 v83, v81 offset:1920
	v_mad_co_u64_u32 v[80:81], null, v80, v7, v[1:2]
	v_mov_b32_e32 v81, 0
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v80, vcc_lo, v6, v80
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, v5, v81, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v82, v82, v83
	global_store_b32 v[80:81], v82, off
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v49, v50 offset1:1
	ds_store_2addr_b32 v8, v51, v52 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v53, v54 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v55, v56 offset0:6 offset1:7
	ds_store_b32 v73, v41
	ds_store_b32 v74, v42
	ds_store_b32 v75, v43
	ds_store_b32 v58, v44
	ds_store_b32 v59, v45
	ds_store_b32 v65, v46
	ds_store_b32 v66, v47
	ds_store_b32 v67, v48
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v41, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	ds_load_b32 v44, v57
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:64
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v42, 0 :: v_dual_mul_f32 v43, v43, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	global_store_b32 v[41:42], v43, off
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 17, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:68
	v_and_b32_e32 v42, 31, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:128
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v42, 0 :: v_dual_mul_f32 v43, v43, v44
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	global_store_b32 v[41:42], v43, off
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 18, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:72
	v_and_b32_e32 v42, 31, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:256
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 19, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:76
	v_and_b32_e32 v42, 31, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:384
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 20, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:80
	v_and_b32_e32 v42, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:512
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 21, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:84
	v_and_b32_e32 v42, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:640
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v42, 0 :: v_dual_mul_f32 v43, v43, v44
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	global_store_b32 v[41:42], v43, off
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 22, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:88
	v_and_b32_e32 v42, 31, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:768
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v42, 0 :: v_dual_mul_f32 v43, v43, v44
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	global_store_b32 v[41:42], v43, off
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 23, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:92
	v_and_b32_e32 v42, 31, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:896
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 24, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:96
	v_and_b32_e32 v42, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:1024
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 25, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:100
	v_and_b32_e32 v42, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:1152
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 26, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:104
	v_and_b32_e32 v42, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:1280
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v42, 0 :: v_dual_mul_f32 v43, v43, v44
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	global_store_b32 v[41:42], v43, off
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 27, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:108
	v_and_b32_e32 v42, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:1408
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v42, 0 :: v_dual_mul_f32 v43, v43, v44
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	global_store_b32 v[41:42], v43, off
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 28, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:112
	v_and_b32_e32 v42, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:1536
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 29, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_78
; %bb.77:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:116
	v_and_b32_e32 v42, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:1664
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 30, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:120
	v_and_b32_e32 v42, 31, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:1792
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	v_mov_b32_e32 v42, 0
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v43, v43, v44
	global_store_b32 v[41:42], v43, off
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v41, 31, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v41
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_lshlrev_b64_e32 v[42:43], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s8, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, s9, v43, vcc_lo
	global_load_b32 v43, v[42:43], off offset:124
	v_and_b32_e32 v42, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v42, v42, 2, v4
	ds_load_b32 v44, v42 offset:1920
	v_mad_co_u64_u32 v[41:42], null, v41, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v42, 0 :: v_dual_mul_f32 v43, v43, v44
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, vcc_lo, v6, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v5, v42, vcc_lo
	global_store_b32 v[41:42], v43, off
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v33, v34 offset1:1
	ds_store_2addr_b32 v8, v35, v36 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v37, v38 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v39, v40 offset0:6 offset1:7
	ds_store_b32 v73, v25
	ds_store_b32 v74, v26
	ds_store_b32 v75, v27
	ds_store_b32 v58, v28
	ds_store_b32 v59, v29
	ds_store_b32 v65, v30
	ds_store_b32 v66, v31
	ds_store_b32 v67, v32
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v25, 32, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	ds_load_b32 v28, v57
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:128
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v26, 0 :: v_dual_mul_f32 v27, v27, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 33, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:132
	v_and_b32_e32 v26, 31, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:128
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v26, 0 :: v_dual_mul_f32 v27, v27, v28
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 34, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:136
	v_and_b32_e32 v26, 31, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:256
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 35, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:140
	v_and_b32_e32 v26, 31, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:384
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 36, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_92
; %bb.91:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:144
	v_and_b32_e32 v26, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:512
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 37, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:148
	v_and_b32_e32 v26, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:640
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v26, 0 :: v_dual_mul_f32 v27, v27, v28
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 38, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_96
; %bb.95:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:152
	v_and_b32_e32 v26, 31, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:768
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v26, 0 :: v_dual_mul_f32 v27, v27, v28
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 39, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:156
	v_and_b32_e32 v26, 31, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:896
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 40, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:160
	v_and_b32_e32 v26, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1024
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 41, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:164
	v_and_b32_e32 v26, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1152
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 42, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:168
	v_and_b32_e32 v26, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1280
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v26, 0 :: v_dual_mul_f32 v27, v27, v28
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 43, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:172
	v_and_b32_e32 v26, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1408
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v26, 0 :: v_dual_mul_f32 v27, v27, v28
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 44, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:176
	v_and_b32_e32 v26, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1536
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 45, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:180
	v_and_b32_e32 v26, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1664
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 46, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:184
	v_and_b32_e32 v26, 31, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1792
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	global_store_b32 v[25:26], v27, off
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 47, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_114
; %bb.113:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:188
	v_and_b32_e32 v26, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1920
	v_mad_co_u64_u32 v[25:26], null, v25, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v26, 0 :: v_dual_mul_f32 v27, v27, v28
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v6, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v5, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v17, v18 offset1:1
	ds_store_2addr_b32 v8, v19, v20 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v21, v22 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v23, v24 offset0:6 offset1:7
	ds_store_b32 v73, v9
	ds_store_b32 v74, v10
	ds_store_b32 v75, v11
	ds_store_b32 v58, v12
	ds_store_b32 v59, v13
	ds_store_b32 v65, v14
	ds_store_b32 v66, v15
	ds_store_b32 v67, v16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v8, 48, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_116
; %bb.115:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	ds_load_b32 v11, v57
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:192
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mul_f32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	global_store_b32 v[8:9], v10, off
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 49, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:196
	v_and_b32_e32 v9, 31, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:128
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 50, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:200
	v_and_b32_e32 v9, 31, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:256
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 51, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_122
; %bb.121:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:204
	v_and_b32_e32 v9, 31, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:384
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 52, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_124
; %bb.123:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:208
	v_and_b32_e32 v9, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:512
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mul_f32 v10, v10, v11
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	global_store_b32 v[8:9], v10, off
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 53, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_126
; %bb.125:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:212
	v_and_b32_e32 v9, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:640
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 54, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_128
; %bb.127:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:216
	v_and_b32_e32 v9, 31, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:768
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 55, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_130
; %bb.129:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:220
	v_and_b32_e32 v9, 31, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:896
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 56, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_132
; %bb.131:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:224
	v_and_b32_e32 v9, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1024
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 57, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_134
; %bb.133:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:228
	v_and_b32_e32 v9, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1152
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mul_f32 v10, v10, v11
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	global_store_b32 v[8:9], v10, off
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 58, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_136
; %bb.135:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:232
	v_and_b32_e32 v9, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1280
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 59, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_138
; %bb.137:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:236
	v_and_b32_e32 v9, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1408
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 60, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_140
; %bb.139:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:240
	v_and_b32_e32 v9, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1536
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 61, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:244
	v_and_b32_e32 v9, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1664
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	v_mov_b32_e32 v9, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v10, v10, v11
	global_store_b32 v[8:9], v10, off
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 62, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:248
	v_and_b32_e32 v9, 31, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1792
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mul_f32 v10, v10, v11
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	global_store_b32 v[8:9], v10, off
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 63, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v8
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_146
; %bb.145:
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v0, v0, 2, v4
	v_add_co_u32 v2, vcc_lo, s8, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, s9, v3, vcc_lo
	global_load_b32 v2, v[2:3], off offset:252
	ds_load_b32 v3, v0 offset:1920
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v8, v7, v[1:2]
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, v6, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
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
	.size	gu_nocompute_nb, .Lfunc_end0-gu_nocompute_nb
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gu_nocompute_nb
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 115
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gu_nocompute_nb)<<4)&4080)>>4
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
	.set .Lgu_nocompute_nb.num_vgpr, 115
	.set .Lgu_nocompute_nb.num_agpr, 0
	.set .Lgu_nocompute_nb.numbered_sgpr, 38
	.set .Lgu_nocompute_nb.num_named_barrier, 0
	.set .Lgu_nocompute_nb.private_seg_size, 0
	.set .Lgu_nocompute_nb.uses_vcc, 1
	.set .Lgu_nocompute_nb.uses_flat_scratch, 0
	.set .Lgu_nocompute_nb.has_dyn_sized_stack, 0
	.set .Lgu_nocompute_nb.has_recursion, 0
	.set .Lgu_nocompute_nb.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15784
; TotalNumSgprs: 40
; NumVgprs: 115
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 40
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
	.type	__hip_cuid_cefdda29c1f9ed9c,@object ; @__hip_cuid_cefdda29c1f9ed9c
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_cefdda29c1f9ed9c
__hip_cuid_cefdda29c1f9ed9c:
	.byte	0                               ; 0x0
	.size	__hip_cuid_cefdda29c1f9ed9c, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_cefdda29c1f9ed9c
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
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           gu_nocompute_nb
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gu_nocompute_nb.kd
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
