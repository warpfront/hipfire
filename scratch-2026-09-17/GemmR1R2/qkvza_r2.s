	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	qkvza_r2                ; -- Begin function qkvza_r2
	.globl	qkvza_r2
	.p2align	8
	.type	qkvza_r2,@function
qkvza_r2:                               ; @qkvza_r2
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[24:27], s[0:1], 0x58
	s_load_b512 s[8:23], s[0:1], 0x0
	s_load_b64 s[28:29], s[0:1], 0x68
	v_lshrrev_b32_e32 v7, 2, v0
	s_lshl_b32 s33, ttmp9, 7
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v3, s33, v7
	s_wait_kmcnt 0x0
	s_add_co_i32 s30, s25, s24
	v_mov_b32_e32 v1, s14
	s_add_co_i32 s31, s30, s26
	v_mov_b32_e32 v2, s15
	s_add_co_i32 s34, s31, s27
	v_mov_b32_e32 v4, s31
	s_add_co_i32 s5, s34, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v12, s5, v3
	v_cmpx_gt_i32_e64 s31, v12
	s_cbranch_execz .LBB0_4
; %bb.1:
	v_dual_mov_b32 v1, s12 :: v_dual_mov_b32 v2, s13
	v_mov_b32_e32 v4, s30
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s30, v12
; %bb.2:
	v_cmp_gt_i32_e32 vcc_lo, s24, v12
	v_mov_b32_e32 v1, s9
	v_mov_b32_e32 v5, s8
	v_cndmask_b32_e64 v4, s24, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v2, s11, v1, vcc_lo
	v_cndmask_b32_e32 v1, s10, v5, vcc_lo
; %bb.3:
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_4:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_mov_b32 v9, s31 :: v_dual_add_nc_u32 v10, 64, v3
	v_dual_mov_b32 v5, s14 :: v_dual_mov_b32 v6, s15
	s_mov_b32 s2, exec_lo
	v_min_i32_e32 v8, s5, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s31, v8
	s_cbranch_execz .LBB0_8
; %bb.5:
	v_dual_mov_b32 v5, s12 :: v_dual_mov_b32 v6, s13
	v_mov_b32_e32 v9, s30
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_i32_e64 s30, v8
; %bb.6:
	v_cmp_gt_i32_e32 vcc_lo, s24, v8
	v_mov_b32_e32 v5, s9
	v_mov_b32_e32 v11, s8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, s24, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v6, s11, v5, vcc_lo
	v_cndmask_b32_e32 v5, s10, v11, vcc_lo
; %bb.7:
	s_or_b32 exec_lo, exec_lo, s3
.LBB0_8:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s2
	s_ashr_i32 s2, s28, 31
	v_sub_nc_u32_e32 v11, v12, v4
	s_lshr_b32 s2, s2, 24
	s_lshl_b32 s35, ttmp7, 7
	s_add_co_i32 s2, s28, s2
	v_mov_b32_e32 v4, 0
	s_ashr_i32 s36, s2, 8
	v_cmp_gt_i32_e64 s2, s34, v10
	s_mul_i32 s7, s36, 0x88
	v_lshlrev_b32_e32 v12, 3, v7
	v_mul_lo_u32 v10, s7, v11
	v_and_b32_e32 v11, 3, v0
	v_or_b32_e32 v15, s35, v7
	s_add_co_i32 s37, s29, -1
	v_and_b32_e32 v18, 0x7f, v0
	v_lshrrev_b32_e32 v7, 2, v7
	v_lshrrev_b32_e32 v53, 1, v0
	v_cmp_gt_i32_e64 s3, s29, v15
	v_add_co_u32 v13, vcc_lo, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, 0, v2, vcc_lo
	v_lshlrev_b32_e32 v1, 4, v11
	v_and_b32_e32 v2, 0x78, v12
	v_min_i32_e32 v10, s37, v15
	v_lshrrev_b32_e32 v12, 4, v0
	v_or_b32_e32 v15, 64, v15
	v_or_b32_e32 v7, 16, v7
	s_cmp_lt_i32 s28, 0x100
	v_mad_co_u64_u32 v[205:206], null, v10, s28, v[1:2]
	v_or_b32_e32 v10, s33, v18
	v_and_or_b32 v12, v12, 12, v11
	v_min_i32_e32 v16, s37, v15
	v_and_or_b32 v7, v7, 28, v11
	v_cmp_gt_i32_e64 s4, s29, v15
	v_min_i32_e32 v17, s5, v10
	v_mad_u32_u24 v12, 0x120, v12, v2
	v_mad_co_u64_u32 v[206:207], null, v16, s28, v[1:2]
	v_sub_nc_u32_e32 v1, v8, v9
	v_mad_u32_u24 v2, 0x120, v7, v2
	v_cmp_gt_i32_e32 vcc_lo, s24, v17
	v_dual_mov_b32 v8, s9 :: v_dual_mov_b32 v9, s8
	v_cmp_gt_i32_e64 s6, s30, v17
	v_lshlrev_b32_e32 v15, 3, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, s24, 0, vcc_lo
	v_or_b32_e32 v11, s35, v53
	v_cndmask_b32_e32 v8, s11, v8, vcc_lo
	v_cndmask_b32_e32 v9, s10, v9, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s31, v17
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v7, s30, v7, s6
	v_mul_lo_u32 v1, s7, v1
	v_cmp_gt_i32_e64 s5, s34, v3
	v_min_i32_e32 v3, s37, v11
	v_cndmask_b32_e64 v8, s13, v8, s6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, s31, v7, vcc_lo
	v_cndmask_b32_e64 v9, s12, v9, s6
	v_add_nc_u32_e32 v251, 0, v12
	v_add_nc_u32_e32 v252, 0, v2
	v_add_co_u32 v16, s6, v5, v1
	v_mul_lo_u32 v1, v3, s36
	v_dual_cndmask_b32 v3, s15, v8 :: v_dual_lshlrev_b32 v8, 2, v53
	v_sub_nc_u32_e32 v7, v17, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, 0, v6, s6
	v_cndmask_b32_e32 v5, s14, v9, vcc_lo
	v_lshlrev_b32_e32 v9, 3, v18
	v_mul_lo_u32 v6, s7, v7
	v_lshrrev_b32_e32 v7, 7, v0
	v_lshlrev_b32_e32 v104, 1, v1
	v_cmp_gt_i32_e64 s6, s29, v11
	v_cmp_gt_i32_e64 s7, s34, v10
	v_add_nc_u32_e32 v248, 0, v8
	v_lshlrev_b32_e32 v249, 4, v7
	s_cselect_b32 s9, -1, 0
	v_add_co_u32 v102, vcc_lo, v5, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v103, null, 0, v3, vcc_lo
	v_lshl_or_b32 v3, v7, 2, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s9
	s_mov_b32 s8, -1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v250, 0, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_10
; %bb.9:
	v_add_co_u32 v8, vcc_lo, v13, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, 0, v14, vcc_lo
	v_add_co_u32 v11, vcc_lo, v16, v15
	global_load_b32 v30, v[102:103], off
	s_clause 0x1
	global_load_b128 v[18:21], v205, s[16:17]
	global_load_b128 v[22:25], v206, s[16:17]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, 0, v17, vcc_lo
	v_lshlrev_b32_e32 v3, 1, v1
	global_load_b64 v[1:2], v[8:9], off offset:8
	global_load_b64 v[26:27], v[11:12], off offset:8
	global_load_b128 v[141:144], v206, s[16:17] offset:64
	s_mov_b32 s10, 0x4e4c4a48
	v_lshlrev_b64_e32 v[28:29], 2, v[3:4]
	global_load_b128 v[4:7], v205, s[16:17] offset:64
	s_mov_b32 s11, 0x4040404
	v_add_nc_u32_e32 v32, 0x2000, v252
	v_add_nc_u32_e32 v31, 0x2000, v251
	v_add_co_u32 v28, vcc_lo, s18, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s19, v29, vcc_lo
	global_load_b32 v3, v[28:29], off
	global_load_b64 v[9:10], v[8:9], off offset:40
	global_load_b64 v[11:12], v[11:12], off offset:40
	s_wait_loadcnt 0x9
	v_lshrrev_b32_e32 v8, v249, v30
	s_wait_loadcnt 0x8
	v_cndmask_b32_e64 v19, 0, v19, s3
	v_cndmask_b32_e64 v18, 0, v18, s3
	v_cndmask_b32_e64 v21, 0, v21, s3
	v_cndmask_b32_e64 v20, 0, v20, s3
	v_cvt_f32_f16_e32 v8, v8.l
	s_wait_loadcnt 0x6
	v_cndmask_b32_e64 v1, 0, v1, s5
	v_cndmask_b32_e64 v2, 0, v2, s5
	s_wait_loadcnt 0x5
	v_cndmask_b32_e64 v26, 0, v26, s2
	v_cndmask_b32_e64 v27, 0, v27, s2
	v_cndmask_b32_e64 v30, 0, v8, s7
	s_wait_loadcnt 0x3
	v_dual_mov_b32 v96, v5 :: v_dual_mov_b32 v5, v7
	v_and_b32_e32 v8, 0xf0f0f0f, v2
	v_and_b32_e32 v7, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_lshrrev_b32_e32 v2, 4, v2
	v_and_b32_e32 v28, 0xf0f0f0f, v26
	v_lshrrev_b32_e32 v26, 4, v26
	v_lshrrev_b32_e32 v29, 4, v27
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	v_and_b32_e32 v2, 0xf0f0f0f, v2
	v_and_b32_e32 v27, 0xf0f0f0f, v27
	v_and_b32_e32 v26, 0xf0f0f0f, v26
	v_and_b32_e32 v29, 0xf0f0f0f, v29
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v3, 0, v3, s6
	v_perm_b32 v33, v1, v7, 0x5010400
	v_perm_b32 v1, v1, v7, 0x7030602
	v_perm_b32 v7, v2, v8, 0x5010400
	v_perm_b32 v2, v2, v8, 0x7030602
	v_perm_b32 v8, v26, v28, 0x5010400
	v_perm_b32 v26, v26, v28, 0x7030602
	v_perm_b32 v28, v29, v27, 0x5010400
	v_perm_b32 v27, v29, v27, 0x7030602
	ds_store_b32 v248, v3 offset:18432
	v_and_b32_e32 v3, 0x7070707, v33
	v_lshrrev_b32_e32 v29, 1, v33
	v_and_b32_e32 v33, 0x7070707, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_and_b32_e32 v34, 0x7070707, v7
	v_lshrrev_b32_e32 v7, 1, v7
	v_and_b32_e32 v35, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	v_and_b32_e32 v36, 0x7070707, v8
	v_lshrrev_b32_e32 v8, 1, v8
	v_and_b32_e32 v37, 0x7070707, v26
	v_lshrrev_b32_e32 v26, 1, v26
	v_and_b32_e32 v38, 0x7070707, v28
	v_lshrrev_b32_e32 v28, 1, v28
	v_and_b32_e32 v39, 0x7070707, v27
	v_lshrrev_b32_e32 v27, 1, v27
	s_wait_alu depctr_sa_sdst(0)
	v_perm_b32 v40, s10, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v29, v29, s11, 0x3020100
	v_perm_b32 v41, s10, 0x44403800, v33
	v_or_b32_e32 v33, 0x50505050, v33
	v_and_or_b32 v42, v1, s11, 0x3020100
	v_perm_b32 v43, s10, 0x44403800, v34
	v_or_b32_e32 v34, 0x50505050, v34
	v_and_or_b32 v7, v7, s11, 0x3020100
	v_perm_b32 v44, s10, 0x44403800, v35
	v_or_b32_e32 v35, 0x50505050, v35
	v_and_or_b32 v45, v2, s11, 0x3020100
	v_perm_b32 v46, s10, 0x44403800, v36
	v_or_b32_e32 v36, 0x50505050, v36
	v_and_or_b32 v47, v8, s11, 0x3020100
	v_perm_b32 v48, s10, 0x44403800, v37
	v_or_b32_e32 v37, 0x50505050, v37
	v_and_or_b32 v49, v26, s11, 0x3020100
	v_perm_b32 v50, s10, 0x44403800, v38
	v_or_b32_e32 v38, 0x50505050, v38
	v_and_or_b32 v28, v28, s11, 0x3020100
	v_perm_b32 v51, s10, 0x44403800, v39
	v_or_b32_e32 v39, 0x50505050, v39
	v_and_or_b32 v52, v27, s11, 0x3020100
	v_mov_b32_e32 v95, v144
	v_cndmask_b32_e64 v23, 0, v23, s4
	v_cndmask_b32_e64 v22, 0, v22, s4
	v_cndmask_b32_e64 v25, 0, v25, s4
	v_cndmask_b32_e64 v24, 0, v24, s4
	v_perm_b32 v1, v3, v40, v29
	v_perm_b32 v2, v33, v41, v42
	v_perm_b32 v7, v34, v43, v7
	v_perm_b32 v8, v35, v44, v45
	v_perm_b32 v26, v36, v46, v47
	v_perm_b32 v27, v37, v48, v49
	v_perm_b32 v28, v38, v50, v28
	v_perm_b32 v29, v39, v51, v52
	ds_store_b32 v250, v30 offset:18944
	ds_store_2addr_b64 v251, v[18:19], v[20:21] offset1:16
	ds_store_2addr_b64 v252, v[22:23], v[24:25] offset1:16
	ds_store_2addr_b64 v31, v[1:2], v[7:8] offset0:128 offset1:144
	ds_store_2addr_b64 v32, v[26:27], v[28:29] offset0:128 offset1:144
	s_branch .LBB0_11
.LBB0_10:
	v_dual_mov_b32 v7, v4 :: v_dual_mov_b32 v96, 0
	v_dual_mov_b32 v5, v4 :: v_dual_mov_b32 v6, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v12, v7
	v_dual_mov_b32 v11, v6 :: v_dual_mov_b32 v10, v5
	v_dual_mov_b32 v9, v4 :: v_dual_mov_b32 v8, v3
	v_mov_b32_e32 v7, v2
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_mov_b32_e32 v95, 0
.LBB0_11:
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_bfe_u32 v1, v0, 4, 1
	s_and_not1_b32 vcc_lo, exec_lo, s9
                                        ; implicit-def: $vgpr7
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v18, 3, v1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_13
; %bb.12:
	v_lshlrev_b32_e32 v7, 3, v1
	s_mov_b32 s8, 0
.LBB0_13:
	v_dual_mov_b32 v38, 0 :: v_dual_and_b32 v23, 31, v0
	v_dual_mov_b32 v39, 0 :: v_dual_and_b32 v8, 15, v0
	v_dual_mov_b32 v41, 0 :: v_dual_and_b32 v24, 0x1c0, v53
	v_dual_mov_b32 v40, 0 :: v_dual_and_b32 v25, 0x60, v0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v232, 0
	v_dual_mov_b32 v255, 0 :: v_dual_mov_b32 v234, 0
	v_dual_mov_b32 v231, 0 :: v_dual_mov_b32 v236, 0
	v_dual_mov_b32 v233, 0 :: v_dual_mov_b32 v238, 0
	v_dual_mov_b32 v235, 0 :: v_dual_mov_b32 v240, 0
	v_dual_mov_b32 v237, 0 :: v_dual_mov_b32 v242, 0
	v_dual_mov_b32 v239, 0 :: v_dual_mov_b32 v244, 0
	v_dual_mov_b32 v241, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v243, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v245, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v174, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v175, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_25
; %bb.14:
	v_or_b32_e32 v2, v24, v8
	v_or_b32_e32 v3, v18, v25
	scratch_store_b32 off, v0, off offset:360 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v0, 2, v24
	v_lshrrev_b32_e32 v1, 2, v25
	v_dual_mov_b32 v105, 0 :: v_dual_lshlrev_b32 v2, 2, v2
	v_lshlrev_b32_e32 v3, 3, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_u32_u24 v0, 0x120, v0, 0
	v_mad_u32_u24 v1, 0x120, v1, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v174, v105 :: v_dual_lshlrev_b32 v7, 3, v23
	v_dual_mov_b32 v172, v105 :: v_dual_add_nc_u32 v3, 0, v3
	v_mov_b32_e32 v170, v105
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v1, v1, v7
	v_dual_mov_b32 v175, v105 :: v_dual_add_nc_u32 v176, v0, v7
	v_dual_mov_b32 v173, v105 :: v_dual_add_nc_u32 v0, 0, v2
	v_dual_mov_b32 v169, v105 :: v_dual_add_nc_u32 v2, 0x4a10, v3
	v_add_nc_u32_e32 v246, 0x2400, v1
	v_dual_mov_b32 v164, v105 :: v_dual_add_nc_u32 v1, 0x2000, v251
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v67, v105 :: v_dual_add_nc_u32 v0, 0x4800, v0
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v2, off offset:8
	scratch_store_b32 off, v1, off
	v_dual_mov_b32 v165, v105 :: v_dual_add_nc_u32 v2, 0x4a18, v3
	v_dual_mov_b32 v162, v105 :: v_dual_add_nc_u32 v1, 0x2000, v252
	scratch_store_b32 off, v0, off offset:64 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, v205
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v2, off offset:12
	scratch_store_b32 off, v1, off offset:4
	v_dual_mov_b32 v163, v105 :: v_dual_add_nc_u32 v2, 0x4a20, v3
	scratch_store_b32 off, v23, off offset:364 ; 4-byte Folded Spill
	v_mov_b32_e32 v161, v105
	v_mov_b32_e32 v159, v105
	scratch_store_b32 off, v2, off offset:16 ; 4-byte Folded Spill
	v_dual_mov_b32 v245, v105 :: v_dual_add_nc_u32 v2, 0x4a28, v3
	v_mov_b32_e32 v243, v105
	v_add_co_u32 v207, vcc_lo, v13, v15
	scratch_store_b32 off, v2, off offset:20 ; 4-byte Folded Spill
	v_dual_mov_b32 v241, v105 :: v_dual_add_nc_u32 v2, 0x4a30, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v208, null, 0, v14, vcc_lo
	v_add_co_u32 v209, vcc_lo, v16, v15
	scratch_store_b32 off, v2, off offset:24 ; 4-byte Folded Spill
	v_dual_mov_b32 v239, v105 :: v_dual_add_nc_u32 v2, 0x4a38, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v210, null, 0, v17, vcc_lo
	v_mov_b32_e32 v237, v105
	scratch_store_b32 off, v2, off offset:28 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v2, 0x4a80, v3
	v_dual_mov_b32 v166, v105 :: v_dual_add_nc_u32 v229, 0x4a00, v3
	v_dual_mov_b32 v235, v105 :: v_dual_add_nc_u32 v254, 0x4a08, v3
	v_mov_b32_e32 v171, v105
	scratch_store_b32 off, v2, off offset:32 ; 4-byte Folded Spill
	v_dual_mov_b32 v233, v105 :: v_dual_add_nc_u32 v2, 0x4a88, v3
	v_dual_mov_b32 v69, v105 :: v_dual_add_nc_u32 v230, 0x800, v176
	v_dual_mov_b32 v160, v105 :: v_dual_mov_b32 v65, v105
	scratch_store_b32 off, v2, off offset:36 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v2, 0x4a90, v3
	v_dual_mov_b32 v244, v105 :: v_dual_mov_b32 v63, v105
	v_dual_mov_b32 v242, v105 :: v_dual_mov_b32 v61, v105
	scratch_store_b32 off, v2, off offset:40 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v2, 0x4a98, v3
	v_dual_mov_b32 v240, v105 :: v_dual_mov_b32 v59, v105
	v_dual_mov_b32 v238, v105 :: v_dual_mov_b32 v57, v105
	scratch_store_b32 off, v2, off offset:44 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v2, 0x4aa0, v3
	v_dual_mov_b32 v236, v105 :: v_dual_mov_b32 v51, v105
	v_dual_mov_b32 v234, v105 :: v_dual_mov_b32 v55, v105
	scratch_store_b32 off, v2, off offset:48 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v2, 0x4aa8, v3
	v_dual_mov_b32 v232, v105 :: v_dual_mov_b32 v53, v105
	v_mov_b32_e32 v231, v105
	v_mov_b32_e32 v255, v105
	scratch_store_b32 off, v2, off offset:52 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v2, 0x4ab0, v3
	v_dual_mov_b32 v68, v105 :: v_dual_mov_b32 v49, v105
	v_dual_mov_b32 v66, v105 :: v_dual_mov_b32 v47, v105
	scratch_store_b32 off, v2, off offset:56 ; 4-byte Folded Spill
	v_dual_mov_b32 v3, v105 :: v_dual_add_nc_u32 v2, 0x4ab8, v3
	v_dual_mov_b32 v64, v105 :: v_dual_mov_b32 v45, v105
	v_dual_mov_b32 v62, v105 :: v_dual_mov_b32 v43, v105
	v_dual_mov_b32 v60, v105 :: v_dual_mov_b32 v41, v105
	v_dual_mov_b32 v58, v105 :: v_dual_mov_b32 v39, v105
	v_mov_b32_e32 v52, v105
	v_mov_b32_e32 v50, v105
	v_mov_b32_e32 v56, v105
	v_mov_b32_e32 v54, v105
	v_mov_b32_e32 v48, v105
	v_mov_b32_e32 v46, v105
	v_mov_b32_e32 v44, v105
	v_mov_b32_e32 v42, v105
	v_mov_b32_e32 v40, v105
	v_mov_b32_e32 v38, v105
	scratch_store_b64 off, v[0:1], off offset:296 ; 8-byte Folded Spill
	v_mov_b32_e32 v0, v206
	s_mov_b32 s9, 0
	s_sub_co_i32 s13, 0, s36
	s_movk_i32 s10, 0x100
	s_movk_i32 s14, 0x88
	s_mov_b32 s15, 0x4e4c4a48
	s_mov_b32 s28, 0x4040404
	s_clause 0x10                           ; 76-byte Folded Spill
	scratch_store_b32 off, v24, off offset:368
	scratch_store_b32 off, v8, off offset:376
	scratch_store_b32 off, v25, off offset:372
	scratch_store_b32 off, v18, off offset:380
	scratch_store_b32 off, v2, off offset:60
	scratch_store_b64 off, v[0:1], off offset:304
	scratch_store_b64 off, v[102:103], off offset:312
	scratch_store_b32 off, v248, off offset:320
	scratch_store_b32 off, v249, off offset:324
	scratch_store_b32 off, v250, off offset:328
	scratch_store_b32 off, v251, off offset:332
	scratch_store_b32 off, v252, off offset:336
	scratch_store_b32 off, v207, off offset:340
	scratch_store_b32 off, v208, off offset:344
	scratch_store_b32 off, v209, off offset:348
	scratch_store_b32 off, v210, off offset:352
	scratch_store_b32 off, v254, off offset:356
	s_branch .LBB0_16
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=1
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b64 v[1:2], off, off offset:288 th:TH_LOAD_LU
	scratch_load_b64 v[153:154], off, off offset:280 th:TH_LOAD_LU
	v_dual_fmac_f32 v172, v181, v71 :: v_dual_fmac_f32 v171, v187, v74
	v_dual_fmac_f32 v170, v185, v73 :: v_dual_fmac_f32 v169, v191, v76
	v_dual_fmac_f32 v241, v181, v63 :: v_dual_fmac_f32 v238, v191, v68
	v_fmac_f32_e32 v239, v185, v65
	s_clause 0x2                            ; 24-byte Folded Reload
	scratch_load_b64 v[146:147], off, off offset:244 th:TH_LOAD_LU
	scratch_load_b64 v[148:149], off, off offset:252 th:TH_LOAD_LU
	scratch_load_b64 v[150:151], off, off offset:260 th:TH_LOAD_LU
	v_dual_fmac_f32 v169, v192, v167 :: v_dual_fmac_f32 v238, v192, v168
	s_add_co_i32 s12, s12, 1
	s_addk_co_i32 s10, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s13, s12
	s_addk_co_i32 s14, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, 1
	v_fmac_f32_e32 v173, v183, v72
	v_fmac_f32_e32 v242, v183, v64
	v_fmac_f32_e32 v240, v187, v66
	s_clause 0x6                            ; 36-byte Folded Reload
	scratch_load_b32 v68, off, off offset:188 th:TH_LOAD_LU
	scratch_load_b32 v65, off, off offset:176 th:TH_LOAD_LU
	scratch_load_b32 v63, off, off offset:168 th:TH_LOAD_LU
	scratch_load_b32 v66, off, off offset:180 th:TH_LOAD_LU
	scratch_load_b32 v64, off, off offset:172 th:TH_LOAD_LU
	scratch_load_b64 v[72:73], off, off offset:220 th:TH_LOAD_LU
	scratch_load_b64 v[144:145], off, off offset:236 th:TH_LOAD_LU
	v_fmac_f32_e32 v237, v189, v67
	scratch_load_b32 v67, off, off offset:184 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v166, v189, v75
	scratch_load_b64 v[74:75], off, off offset:228 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v240, v188, v168
	v_fmac_f32_e32 v242, v184, v168
	s_wait_dscnt 0x8
	v_fmac_f32_e32 v238, v215, v132
	v_fmac_f32_e32 v172, v182, v167
	v_fmac_f32_e32 v170, v186, v167
	v_fmac_f32_e32 v240, v219, v130
	v_fmac_f32_e32 v242, v223, v128
	v_fmac_f32_e32 v238, v216, v198
	v_fmac_f32_e32 v172, v225, v135
	v_fmac_f32_e32 v170, v221, v137
	v_fmac_f32_e32 v240, v220, v198
	v_fmac_f32_e32 v242, v224, v198
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v172, v226, v197
	v_fmac_f32_e32 v170, v222, v197
	s_wait_loadcnt 0xd
	v_fmac_f32_e32 v174, v1, v69
	s_wait_loadcnt 0xc
	v_fmac_f32_e32 v175, v153, v70
	v_fmac_f32_e32 v243, v1, v61
	v_fmac_f32_e32 v244, v153, v62
	s_clause 0x3                            ; 20-byte Folded Reload
	scratch_load_b32 v69, off, off offset:200 th:TH_LOAD_LU
	scratch_load_b32 v62, off, off offset:164 th:TH_LOAD_LU
	scratch_load_b32 v61, off, off offset:160 th:TH_LOAD_LU
	scratch_load_b64 v[70:71], off, off offset:212 th:TH_LOAD_LU
	v_dual_fmac_f32 v174, v2, v167 :: v_dual_fmac_f32 v243, v2, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v175, v154, v167 :: v_dual_fmac_f32 v174, v229, v133
	v_fmac_f32_e32 v243, v229, v125
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v175, v227, v134 :: v_dual_fmac_f32 v174, v230, v197
	v_fmac_f32_e32 v243, v230, v198
	s_wait_loadcnt 0xc
	v_fmac_f32_e32 v68, v153, v54
	s_wait_loadcnt 0xb
	v_fmac_f32_e32 v65, v187, v58
	scratch_load_b32 v58, off, off offset:148 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0xa
	v_fmac_f32_e32 v66, v181, v55
	s_wait_loadcnt 0x9
	v_fmac_f32_e32 v64, v185, v57
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v57, off, off offset:144 th:TH_LOAD_LU
	scratch_load_b32 v54, off, off offset:120 th:TH_LOAD_LU
	scratch_load_b32 v55, off, off offset:124 th:TH_LOAD_LU
	s_wait_loadcnt 0x9
	v_fmac_f32_e32 v67, v183, v56
	scratch_load_b32 v56, off, off offset:128 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x9
	v_dual_fmac_f32 v164, v72, v38 :: v_dual_fmac_f32 v163, v74, v39
	v_fmac_f32_e32 v162, v144, v40
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v40, off, off offset:76 th:TH_LOAD_LU
	scratch_load_b32 v39, off, off offset:72 th:TH_LOAD_LU
	scratch_load_b32 v38, off, off offset:68 th:TH_LOAD_LU
	v_fmac_f32_e32 v66, v182, v157
	v_dual_fmac_f32 v234, v74, v31 :: v_dual_fmac_f32 v233, v144, v32
	v_dual_fmac_f32 v163, v75, v167 :: v_dual_fmac_f32 v162, v145, v167
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v66, v225, v119
	v_fmac_f32_e32 v234, v75, v168
	v_dual_fmac_f32 v164, v73, v167 :: v_dual_fmac_f32 v173, v184, v167
	v_fmac_f32_e32 v166, v190, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v226, v7
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v162, v207, v104
	v_fmac_f32_e32 v234, v209, v95
	v_fmac_f32_e32 v164, v211, v102
	v_dual_fmac_f32 v173, v223, v136 :: v_dual_fmac_f32 v166, v217, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v175, v228, v197 :: v_dual_fmac_f32 v234, v210, v198
	v_fmac_f32_e32 v162, v208, v197
	v_fmac_f32_e32 v164, v212, v197
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v166, v218, v197
	s_wait_loadcnt 0xb
	v_fmac_f32_e32 v69, v1, v53
	s_wait_loadcnt 0xa
	v_fmac_f32_e32 v62, v189, v59
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v53, off, off offset:116 th:TH_LOAD_LU
	scratch_load_b32 v59, off, off offset:152 th:TH_LOAD_LU
	v_fmac_f32_e32 v63, v191, v60
	scratch_load_b32 v60, off, off offset:156 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0xb
	v_fmac_f32_e32 v165, v70, v37
	v_dual_fmac_f32 v237, v190, v168 :: v_dual_fmac_f32 v62, v190, v157
	v_fmac_f32_e32 v69, v2, v157
	v_dual_fmac_f32 v171, v188, v167 :: v_dual_fmac_f32 v244, v154, v168
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v237, v217, v131
	v_fmac_f32_e32 v241, v182, v168
	v_fmac_f32_e32 v236, v70, v29
	v_dual_fmac_f32 v65, v188, v157 :: v_dual_fmac_f32 v62, v217, v123
	v_fmac_f32_e32 v244, v227, v126
	v_fmac_f32_e32 v171, v219, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v165, v71, v167 :: v_dual_fmac_f32 v236, v71, v168
	v_fmac_f32_e32 v241, v225, v127
	v_dual_fmac_f32 v173, v224, v197 :: v_dual_fmac_f32 v62, v218, v7
	s_wait_loadcnt 0x8
	v_fmac_f32_e32 v54, v183, v48
	scratch_load_b32 v48, off, off offset:108 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v69, v229, v117
	s_wait_loadcnt 0x7
	v_fmac_f32_e32 v56, v153, v46
	s_clause 0x1                            ; 12-byte Folded Reload
	scratch_load_b32 v46, off, off offset:100 th:TH_LOAD_LU
	scratch_load_b64 v[152:153], off, off offset:268 th:TH_LOAD_LU
	v_dual_fmac_f32 v161, v146, v41 :: v_dual_fmac_f32 v160, v148, v42
	v_fmac_f32_e32 v159, v150, v43
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v43, off, off offset:88 th:TH_LOAD_LU
	scratch_load_b32 v42, off, off offset:84 th:TH_LOAD_LU
	scratch_load_b32 v41, off, off offset:80 th:TH_LOAD_LU
	v_dual_fmac_f32 v235, v72, v30 :: v_dual_fmac_f32 v232, v146, v33
	v_fmac_f32_e32 v54, v184, v158
	v_dual_fmac_f32 v63, v192, v157 :: v_dual_fmac_f32 v56, v154, v158
	s_wait_loadcnt 0xb
	v_dual_fmac_f32 v255, v150, v35 :: v_dual_fmac_f32 v40, v148, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v54, v223, v112
	v_dual_fmac_f32 v161, v147, v167 :: v_dual_fmac_f32 v232, v147, v168
	v_dual_fmac_f32 v239, v186, v168 :: v_dual_fmac_f32 v40, v149, v158
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v54, v224, v8
	v_fmac_f32_e32 v56, v227, v110
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v232, v205, v97 :: v_dual_fmac_f32 v231, v148, v34
	s_wait_loadcnt 0xa
	v_fmac_f32_e32 v39, v150, v19
	v_fmac_f32_e32 v160, v149, v167
	v_fmac_f32_e32 v56, v228, v8
	v_dual_fmac_f32 v232, v206, v198 :: v_dual_fmac_f32 v159, v151, v167
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v236, v213, v93 :: v_dual_fmac_f32 v39, v151, v158
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v165, v213, v101 :: v_dual_fmac_f32 v160, v203, v106
	v_fmac_f32_e32 v69, v230, v7
	v_dual_fmac_f32 v171, v220, v197 :: v_dual_fmac_f32 v236, v214, v198
	s_wait_dscnt 0x1
	v_fmac_f32_e32 v39, v201, v83
	v_dual_fmac_f32 v165, v214, v197 :: v_dual_fmac_f32 v160, v204, v197
	v_fmac_f32_e32 v244, v228, v198
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v40, v203, v82 :: v_dual_fmac_f32 v39, v202, v8
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v40, v204, v8 :: v_dual_fmac_f32 v53, v181, v47
	scratch_load_b32 v47, off, off offset:104 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x7
	v_dual_fmac_f32 v60, v72, v22 :: v_dual_fmac_f32 v255, v151, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v235, v73, v168 :: v_dual_fmac_f32 v60, v73, v157
	v_fmac_f32_e32 v163, v209, v103
	v_fmac_f32_e32 v235, v211, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v60, v211, v86
	v_dual_fmac_f32 v235, v212, v198 :: v_dual_fmac_f32 v60, v212, v7
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v48, v185, v49 :: v_dual_fmac_f32 v63, v215, v124
	s_wait_loadcnt 0x5
	v_fmac_f32_e32 v46, v189, v51
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v51, off, off offset:136 th:TH_LOAD_LU
	scratch_load_b32 v49, off, off offset:112 th:TH_LOAD_LU
	v_fmac_f32_e32 v55, v1, v45
	s_wait_loadcnt 0x6
	v_fmac_f32_e32 v245, v152, v44
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v45, off, off offset:96 th:TH_LOAD_LU
	scratch_load_b32 v44, off, off offset:92 th:TH_LOAD_LU
	v_fmac_f32_e32 v3, v152, v36
	s_wait_loadcnt 0x5
	v_dual_fmac_f32 v41, v146, v17 :: v_dual_fmac_f32 v38, v152, v20
	v_dual_fmac_f32 v61, v70, v21 :: v_dual_fmac_f32 v58, v144, v24
	v_dual_fmac_f32 v43, v74, v15 :: v_dual_fmac_f32 v42, v144, v16
	v_fmac_f32_e32 v55, v2, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v41, v147, v158 :: v_dual_fmac_f32 v48, v186, v158
	v_fmac_f32_e32 v46, v190, v158
	v_fmac_f32_e32 v53, v182, v158
	v_fmac_f32_e32 v245, v153, v167
	v_dual_fmac_f32 v233, v145, v168 :: v_dual_fmac_f32 v58, v145, v157
	v_dual_fmac_f32 v43, v75, v158 :: v_dual_fmac_f32 v42, v145, v158
	v_fmac_f32_e32 v161, v205, v105
	v_fmac_f32_e32 v61, v71, v157
	v_dual_fmac_f32 v65, v219, v122 :: v_dual_fmac_f32 v48, v221, v113
	v_fmac_f32_e32 v55, v229, v109
	v_fmac_f32_e32 v46, v217, v115
	v_fmac_f32_e32 v53, v225, v111
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v245, v199, v108
	v_fmac_f32_e32 v58, v207, v88
	v_fmac_f32_e32 v38, v153, v158
	v_dual_fmac_f32 v43, v209, v79 :: v_dual_fmac_f32 v48, v222, v8
	v_dual_fmac_f32 v42, v207, v80 :: v_dual_fmac_f32 v161, v206, v197
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v38, v199, v84
	v_fmac_f32_e32 v53, v226, v8
	v_fmac_f32_e32 v65, v220, v7
	v_dual_fmac_f32 v237, v218, v198 :: v_dual_fmac_f32 v42, v208, v8
	v_dual_fmac_f32 v43, v210, v8 :: v_dual_fmac_f32 v58, v208, v7
	v_fmac_f32_e32 v55, v230, v8
	v_dual_fmac_f32 v46, v218, v8 :: v_dual_fmac_f32 v63, v216, v7
	v_dual_fmac_f32 v245, v200, v197 :: v_dual_fmac_f32 v38, v200, v8
	v_dual_mov_b32 v229, v253 :: v_dual_mov_b32 v230, v0
	s_wait_loadcnt 0x4
	v_fmac_f32_e32 v47, v191, v52
	scratch_load_b32 v52, off, off offset:140 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_dual_fmac_f32 v169, v215, v140 :: v_dual_fmac_f32 v68, v154, v157
	v_fmac_f32_e32 v239, v221, v129
	v_fmac_f32_e32 v47, v192, v158
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v169, v216, v197 :: v_dual_fmac_f32 v68, v227, v118
	v_fmac_f32_e32 v239, v222, v198
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v47, v215, v116 :: v_dual_fmac_f32 v68, v228, v7
	v_fmac_f32_e32 v47, v216, v8
	s_wait_loadcnt 0x4
	v_fmac_f32_e32 v51, v150, v27
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v49, v187, v50
	scratch_load_b32 v50, off, off offset:132 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v57, v146, v25
	v_fmac_f32_e32 v59, v74, v23
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v45, v70, v13
	v_fmac_f32_e32 v49, v188, v158
	v_dual_fmac_f32 v67, v184, v157 :: v_dual_fmac_f32 v64, v186, v157
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v44, v72, v14 :: v_dual_fmac_f32 v233, v207, v96
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v49, v219, v114
	v_dual_fmac_f32 v67, v223, v120 :: v_dual_fmac_f32 v64, v221, v121
	v_fmac_f32_e32 v41, v205, v81
	v_fmac_f32_e32 v51, v151, v157
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v49, v220, v8 :: v_dual_fmac_f32 v44, v73, v158
	v_fmac_f32_e32 v64, v222, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v41, v206, v8
	v_fmac_f32_e32 v51, v201, v91
	v_fmac_f32_e32 v233, v208, v198
	v_fmac_f32_e32 v45, v71, v158
	v_dual_fmac_f32 v61, v213, v85 :: v_dual_fmac_f32 v44, v211, v78
	v_fmac_f32_e32 v241, v226, v198
	v_fmac_f32_e32 v67, v224, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v45, v213, v77
	v_dual_fmac_f32 v51, v202, v7 :: v_dual_fmac_f32 v44, v212, v8
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v45, v214, v8 :: v_dual_fmac_f32 v52, v148, v26
	v_fmac_f32_e32 v59, v75, v157
	v_dual_fmac_f32 v231, v149, v168 :: v_dual_fmac_f32 v52, v149, v157
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v59, v209, v87
	v_fmac_f32_e32 v159, v201, v107
	v_fmac_f32_e32 v231, v203, v98
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v52, v203, v90 :: v_dual_fmac_f32 v59, v210, v7
	v_fmac_f32_e32 v61, v214, v7
	v_fmac_f32_e32 v231, v204, v198
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v159, v202, v197 :: v_dual_fmac_f32 v52, v204, v7
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v50, v152, v28
	v_fmac_f32_e32 v3, v153, v168
	v_dual_fmac_f32 v57, v147, v157 :: v_dual_fmac_f32 v50, v153, v157
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v3, v199, v100
	v_fmac_f32_e32 v57, v205, v89
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v255, v201, v99 :: v_dual_fmac_f32 v50, v199, v92
	v_fmac_f32_e32 v163, v210, v197
	v_fmac_f32_e32 v57, v206, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v255, v202, v198
	v_dual_fmac_f32 v3, v200, v198 :: v_dual_fmac_f32 v50, v200, v7
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_clause 0x2                            ; 16-byte Folded Reload
	scratch_load_b64 v[104:105], off, off offset:192 th:TH_LOAD_LU
	scratch_load_b32 v95, off, off offset:204
	scratch_load_b32 v96, off, off offset:208
	s_wait_loadcnt 0x2
	s_clause 0x1                            ; 12-byte Folded Reload
	scratch_load_b32 v104, off, off offset:276 th:TH_LOAD_LU
	scratch_load_b64 v[205:206], off, off offset:296
	s_wait_loadcnt 0x0
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b64 v[206:207], off, off offset:304
	scratch_load_b64 v[102:103], off, off offset:312
	s_wait_loadcnt 0x1
	s_clause 0x3                            ; 16-byte Folded Reload
	scratch_load_b32 v207, off, off offset:340
	scratch_load_b32 v208, off, off offset:344
	scratch_load_b32 v209, off, off offset:348
	scratch_load_b32 v210, off, off offset:352
	s_cbranch_scc1 .LBB0_24
.LBB0_16:                               ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s10, 0xffffff00
	v_lshlrev_b64_e32 v[7:8], 2, v[104:105]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[16:17], s[8:9]
	s_clause 0x1f                           ; 128-byte Folded Spill
	scratch_store_b32 off, v69, off offset:200
	scratch_store_b32 off, v68, off offset:188
	scratch_store_b32 off, v67, off offset:184
	scratch_store_b32 off, v66, off offset:180
	scratch_store_b32 off, v65, off offset:176
	scratch_store_b32 off, v64, off offset:172
	scratch_store_b32 off, v63, off offset:168
	scratch_store_b32 off, v62, off offset:164
	scratch_store_b32 off, v61, off offset:160
	scratch_store_b32 off, v60, off offset:156
	scratch_store_b32 off, v59, off offset:152
	scratch_store_b32 off, v58, off offset:148
	scratch_store_b32 off, v57, off offset:144
	scratch_store_b32 off, v52, off offset:140
	scratch_store_b32 off, v51, off offset:136
	scratch_store_b32 off, v50, off offset:132
	scratch_store_b32 off, v56, off offset:128
	scratch_store_b32 off, v55, off offset:124
	scratch_store_b32 off, v54, off offset:120
	scratch_store_b32 off, v53, off offset:116
	scratch_store_b32 off, v49, off offset:112
	scratch_store_b32 off, v48, off offset:108
	scratch_store_b32 off, v47, off offset:104
	scratch_store_b32 off, v46, off offset:100
	scratch_store_b32 off, v45, off offset:96
	scratch_store_b32 off, v44, off offset:92
	scratch_store_b32 off, v43, off offset:88
	scratch_store_b32 off, v42, off offset:84
	scratch_store_b32 off, v41, off offset:80
	scratch_store_b32 off, v40, off offset:76
	scratch_store_b32 off, v39, off offset:72
	scratch_store_b32 off, v38, off offset:68
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s8, s38, v205
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s39, 0, s8
	v_add_co_u32 v89, s8, s38, v206
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v90, null, s39, 0, s8
	s_add_co_i32 s8, s14, 0xffffff78
	s_wait_loadcnt 0x3
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v91, vcc_lo, v207, s8
	s_wait_loadcnt 0x2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v92, null, 0, v208, vcc_lo
	s_wait_loadcnt 0x1
	v_add_co_u32 v93, vcc_lo, v209, s8
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v94, null, 0, v210, vcc_lo
	v_add_co_u32 v7, vcc_lo, s18, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s19, v8, vcc_lo
	v_add_co_u32 v13, vcc_lo, v102, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, 0, v103, vcc_lo
	s_clause 0x1
	global_load_b128 v[149:152], v[1:2], off offset:128
	global_load_b128 v[145:148], v[89:90], off offset:128
	global_load_b64 v[195:196], v[91:92], off offset:72
	global_load_b64 v[193:194], v[93:94], off offset:72
	global_load_b32 v211, v[13:14], off offset:4
	global_load_b32 v107, v[7:8], off offset:4
	ds_load_2addr_b64 v[77:80], v246 offset1:144
	ds_load_2addr_b64 v[13:16], v176 offset1:144
	ds_load_2addr_b64 v[81:84], v230 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[13:14], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[13:14], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[15:16], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[15:16], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[81:82], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[81:82], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[83:84], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[83:84], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[77:80], v246 offset0:36 offset1:180
	ds_load_2addr_b64 v[81:84], v176 offset0:36 offset1:180
	ds_load_2addr_b64 v[85:88], v230 offset0:68 offset1:212
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[81:82], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[81:82], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[83:84], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[83:84], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[85:86], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[85:86], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[87:88], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[87:88], v[13:20]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[77:80], v246 offset0:72 offset1:216
	ds_load_2addr_b64 v[81:84], v176 offset0:72 offset1:216
	ds_load_2addr_b64 v[85:88], v230 offset0:104 offset1:248
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[81:82], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[81:82], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[83:84], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[83:84], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[85:86], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[85:86], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[87:88], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[87:88], v[13:20]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v0, 0xc00, v176
	ds_load_2addr_b64 v[77:80], v246 offset0:108 offset1:252
	ds_load_2addr_b64 v[81:84], v176 offset0:108 offset1:252
	ds_load_2addr_b64 v[85:88], v0 offset0:12 offset1:156
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[81:82], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[81:82], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[83:84], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[83:84], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[85:86], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[85:86], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[87:88], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[87:88], v[13:20]
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v7, 0, v9, s5
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v11, 0, v11, s2
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v77, 0xf0f0f0f, v7
	v_cndmask_b32_e64 v7, 0, v4, s3
	v_cndmask_b32_e64 v4, 0, v6, s3
	v_cndmask_b32_e64 v6, 0, v10, s5
	v_and_b32_e32 v78, 0xf0f0f0f, v8
	v_lshrrev_b32_e32 v86, 4, v11
	v_cndmask_b32_e64 v12, 0, v12, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_perm_b32 v79, v78, v77, 0x5010400
	v_perm_b32 v81, v78, v77, 0x7030602
	v_lshrrev_b32_e32 v77, 4, v6
	v_and_b32_e32 v6, 0xf0f0f0f, v6
	v_cndmask_b32_e64 v8, 0, v96, s3
	v_and_b32_e32 v80, 0x7070707, v79
	v_lshrrev_b32_e32 v78, 1, v79
	v_and_b32_e32 v82, 0xf0f0f0f, v77
	v_and_b32_e32 v84, 0x7070707, v81
	v_cndmask_b32_e64 v5, 0, v5, s3
	v_perm_b32 v79, s15, 0x44403800, v80
	v_or_b32_e32 v80, 0x50505050, v80
	v_and_or_b32 v83, v78, s28, 0x3020100
	v_perm_b32 v85, v82, v6, 0x5010400
	v_perm_b32 v6, v82, v6, 0x7030602
	v_cndmask_b32_e64 v78, 0, v95, s4
	s_barrier_wait -1
	v_perm_b32 v79, v80, v79, v83
	v_lshrrev_b32_e32 v80, 1, v81
	v_perm_b32 v81, s15, 0x44403800, v84
	v_or_b32_e32 v83, 0x50505050, v84
	v_and_b32_e32 v84, 0x7070707, v85
	v_lshrrev_b32_e32 v85, 1, v85
	v_and_or_b32 v80, v80, s28, 0x3020100
	global_inv scope:SCOPE_SE
	v_cndmask_b32_e64 v10, 0, v142, s4
	v_perm_b32 v82, s15, 0x44403800, v84
	v_or_b32_e32 v84, 0x50505050, v84
	v_and_or_b32 v85, v85, s28, 0x3020100
	v_perm_b32 v80, v83, v81, v80
	v_and_b32_e32 v83, 0xf0f0f0f, v86
	v_and_b32_e32 v86, 0x7070707, v6
	v_and_b32_e32 v81, 0xf0f0f0f, v11
	v_perm_b32 v11, v84, v82, v85
	v_lshrrev_b32_e32 v6, 1, v6
	v_cndmask_b32_e64 v9, 0, v141, s4
	v_perm_b32 v84, s15, 0x44403800, v86
	v_or_b32_e32 v85, 0x50505050, v86
	v_lshrrev_b32_e32 v86, 4, v12
	v_perm_b32 v82, v83, v81, 0x5010400
	v_perm_b32 v81, v83, v81, 0x7030602
	v_and_b32_e32 v12, 0xf0f0f0f, v12
	v_and_or_b32 v6, v6, s28, 0x3020100
	v_and_b32_e32 v83, 0xf0f0f0f, v86
	v_and_b32_e32 v87, 0x7070707, v82
	v_and_b32_e32 v88, 0x7070707, v81
	v_lshrrev_b32_e32 v81, 1, v81
	v_lshrrev_b32_e32 v82, 1, v82
	v_perm_b32 v95, v83, v12, 0x5010400
	v_perm_b32 v12, v83, v12, 0x7030602
	scratch_load_b32 v106, off, off offset:4 ; 4-byte Folded Reload
	v_and_or_b32 v97, v81, s28, 0x3020100
	v_perm_b32 v86, s15, 0x44403800, v87
	v_and_b32_e32 v83, 0x7070707, v95
	v_lshrrev_b32_e32 v81, 1, v95
	v_and_b32_e32 v95, 0x7070707, v12
	v_lshrrev_b32_e32 v12, 1, v12
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v82, v82, s28, 0x3020100
	v_perm_b32 v96, s15, 0x44403800, v88
	v_perm_b32 v100, s15, 0x44403800, v95
	v_or_b32_e32 v95, 0x50505050, v95
	v_and_or_b32 v101, v12, s28, 0x3020100
	v_perm_b32 v12, v85, v84, v6
	v_or_b32_e32 v88, 0x50505050, v88
	v_perm_b32 v98, s15, 0x44403800, v83
	v_or_b32_e32 v83, 0x50505050, v83
	v_perm_b32 v84, v95, v100, v101
	scratch_load_b32 v101, off, off         ; 4-byte Folded Reload
	v_and_or_b32 v99, v81, s28, 0x3020100
	v_perm_b32 v81, v87, v86, v82
	v_perm_b32 v82, v88, v96, v97
	v_cndmask_b32_e64 v77, 0, v143, s4
	ds_store_2addr_b64 v251, v[7:8], v[4:5] offset1:16
	ds_store_2addr_b64 v252, v[9:10], v[77:78] offset1:16
	v_perm_b32 v83, v83, v98, v99
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v101, v[79:80], v[11:12] offset0:128 offset1:144
	ds_store_2addr_b64 v106, v[81:82], v[83:84] offset0:128 offset1:144
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_load_b128 v[141:144], v[89:90], off offset:192
	global_load_b64 v[9:10], v[91:92], off offset:104
	global_load_b64 v[11:12], v[93:94], off offset:104
	global_load_b128 v[4:7], v[1:2], off offset:192
	ds_load_2addr_b64 v[77:80], v246 offset1:144
	ds_load_2addr_b64 v[81:84], v176 offset1:144
	ds_load_2addr_b64 v[85:88], v230 offset0:32 offset1:176
	v_mov_b32_e32 v2, v107
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[81:82], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[81:82], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[83:84], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[83:84], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[85:86], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[85:86], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[87:88], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[87:88], v[13:20]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[77:80], v246 offset0:36 offset1:180
	ds_load_2addr_b64 v[81:84], v176 offset0:36 offset1:180
	ds_load_2addr_b64 v[85:88], v230 offset0:68 offset1:212
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[81:82], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[81:82], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[83:84], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[83:84], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[85:86], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[85:86], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[87:88], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[87:88], v[13:20]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[77:80], v246 offset0:72 offset1:216
	ds_load_2addr_b64 v[81:84], v176 offset0:72 offset1:216
	ds_load_2addr_b64 v[85:88], v230 offset0:104 offset1:248
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[81:82], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[81:82], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[83:84], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[83:84], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[85:86], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[85:86], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[87:88], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[87:88], v[13:20]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[77:80], v246 offset0:108 offset1:252
	ds_load_2addr_b64 v[81:84], v176 offset0:108 offset1:252
	ds_load_2addr_b64 v[85:88], v0 offset0:12 offset1:156
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[81:82], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[81:82], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[83:84], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[83:84], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[85:86], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[85:86], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[87:88], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[87:88], v[13:20]
	; sched_barrier mask(0x00000000)
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v8, off, off offset:8
	scratch_load_b32 v80, off, off offset:48
	scratch_load_b32 v0, off, off offset:64
	ds_load_2addr_b32 v[77:78], v254 offset1:1
	s_cmp_lt_i32 s12, s36
	v_mov_b32_e32 v79, v148
	s_cselect_b32 s8, -1, 0
	s_cmp_ge_i32 s12, s36
	s_wait_loadcnt 0x2
	ds_load_2addr_b32 v[181:182], v8 offset1:1
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[80:81], v80 offset1:1
	scratch_load_b32 v8, off, off offset:12 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[80:81], off offset:244 ; 8-byte Folded Spill
	scratch_load_b32 v80, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[183:184], v8 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[80:81], v80 offset1:1
	scratch_load_b32 v8, off, off offset:16 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[80:81], off offset:252 ; 8-byte Folded Spill
	scratch_load_b32 v80, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[185:186], v8 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[80:81], v80 offset1:1
	scratch_load_b32 v8, off, off offset:20 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[80:81], off offset:260 ; 8-byte Folded Spill
	scratch_load_b32 v80, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[187:188], v8 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[80:81], v80 offset1:1
	scratch_load_b32 v8, off, off offset:24 ; 4-byte Folded Reload
	ds_load_2addr_b32 v[167:168], v0 offset1:16
	ds_load_2addr_b32 v[157:158], v0 offset0:32 offset1:48
	ds_load_2addr_b32 v[0:1], v229 offset1:1
	s_wait_dscnt 0x3
	scratch_store_b64 off, v[80:81], off offset:268 ; 8-byte Folded Spill
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[189:190], v8 offset1:1
	scratch_load_b32 v8, off, off offset:28 ; 4-byte Folded Reload
	scratch_store_b64 off, v[77:78], off offset:280 ; 8-byte Folded Spill
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[191:192], v8 offset1:1
	scratch_load_b32 v8, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[77:78], v8 offset1:1
	scratch_load_b32 v8, off, off offset:36 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[77:78], off offset:212 ; 8-byte Folded Spill
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[77:78], v8 offset1:1
	scratch_load_b32 v8, off, off offset:40 ; 4-byte Folded Reload
	scratch_store_b64 off, v[0:1], off offset:288 ; 8-byte Folded Spill
	v_add_nc_u32_e32 v0, 2, v104
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[77:78], off offset:220 ; 8-byte Folded Spill
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[77:78], v8 offset1:1
	scratch_load_b32 v8, off, off offset:44 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[77:78], off offset:228 ; 8-byte Folded Spill
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[77:78], v8 offset1:1
	v_mov_b32_e32 v8, v150
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[77:78], off offset:236 ; 8-byte Folded Spill
	v_dual_mov_b32 v77, v152 :: v_dual_mov_b32 v78, v146
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cndmask_b32_e64 v82, 0, v195, s5
	v_cndmask_b32_e64 v81, 0, v8, s3
	v_cndmask_b32_e64 v80, 0, v149, s3
	v_cndmask_b32_e64 v83, 0, v77, s3
	v_lshrrev_b32_e32 v99, v249, v211
	v_lshrrev_b32_e32 v84, 4, v82
	v_and_b32_e32 v85, 0xf0f0f0f, v82
	v_cndmask_b32_e64 v82, 0, v151, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v84, 0xf0f0f0f, v84
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v251, v[80:81], v[82:83] offset1:16
	v_cndmask_b32_e64 v82, 0, v196, s5
	v_perm_b32 v86, v84, v85, 0x5010400
	v_perm_b32 v84, v84, v85, 0x7030602
	v_cndmask_b32_e64 v81, 0, v78, s4
	v_cndmask_b32_e64 v80, 0, v145, s4
	v_lshrrev_b32_e32 v88, 4, v82
	v_and_b32_e32 v87, 0x7070707, v86
	v_lshrrev_b32_e32 v85, 1, v86
	v_and_b32_e32 v89, 0xf0f0f0f, v82
	v_and_b32_e32 v90, 0x7070707, v84
	v_and_b32_e32 v88, 0xf0f0f0f, v88
	v_perm_b32 v86, s15, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v85, v85, s28, 0x3020100
	v_lshrrev_b32_e32 v92, 1, v84
	v_perm_b32 v91, v88, v89, 0x5010400
	v_perm_b32 v88, v88, v89, 0x7030602
	v_cndmask_b32_e64 v83, 0, v79, s4
	v_perm_b32 v84, v87, v86, v85
	v_perm_b32 v85, s15, 0x44403800, v90
	v_and_b32_e32 v87, 0x7070707, v91
	v_lshrrev_b32_e32 v91, 1, v91
	v_or_b32_e32 v86, 0x50505050, v90
	v_and_or_b32 v90, v92, s28, 0x3020100
	v_cndmask_b32_e64 v92, 0, v193, s2
	v_perm_b32 v89, s15, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v91, v91, s28, 0x3020100
	v_perm_b32 v85, v86, v85, v90
	v_lshrrev_b32_e32 v93, 4, v92
	v_and_b32_e32 v90, 0x7070707, v88
	v_and_b32_e32 v92, 0xf0f0f0f, v92
	v_perm_b32 v86, v87, v89, v91
	v_cndmask_b32_e64 v87, 0, v194, s2
	v_and_b32_e32 v93, 0xf0f0f0f, v93
	v_lshrrev_b32_e32 v88, 1, v88
	v_perm_b32 v89, s15, 0x44403800, v90
	v_or_b32_e32 v90, 0x50505050, v90
	v_lshrrev_b32_e32 v94, 4, v87
	v_perm_b32 v91, v93, v92, 0x5010400
	v_and_or_b32 v88, v88, s28, 0x3020100
	v_and_b32_e32 v95, 0xf0f0f0f, v87
	v_perm_b32 v92, v93, v92, 0x7030602
	v_and_b32_e32 v94, 0xf0f0f0f, v94
	v_and_b32_e32 v93, 0x7070707, v91
	v_perm_b32 v87, v90, v89, v88
	v_lshrrev_b32_e32 v88, 1, v91
	v_and_b32_e32 v90, 0x7070707, v92
	v_perm_b32 v91, v94, v95, 0x5010400
	v_perm_b32 v94, v94, v95, 0x7030602
	v_lshrrev_b32_e32 v92, 1, v92
	v_perm_b32 v89, s15, 0x44403800, v93
	v_or_b32_e32 v93, 0x50505050, v93
	v_and_b32_e32 v97, 0x7070707, v91
	v_lshrrev_b32_e32 v91, 1, v91
	v_and_b32_e32 v98, 0x7070707, v94
	v_lshrrev_b32_e32 v94, 1, v94
	v_and_or_b32 v88, v88, s28, 0x3020100
	v_perm_b32 v96, s15, 0x44403800, v90
	v_or_b32_e32 v90, 0x50505050, v90
	v_and_or_b32 v92, v92, s28, 0x3020100
	v_perm_b32 v95, s15, 0x44403800, v97
	v_or_b32_e32 v97, 0x50505050, v97
	v_and_or_b32 v91, v91, s28, 0x3020100
	v_perm_b32 v100, s15, 0x44403800, v98
	v_or_b32_e32 v98, 0x50505050, v98
	v_and_or_b32 v94, v94, s28, 0x3020100
	v_perm_b32 v88, v93, v89, v88
	v_cvt_f32_f16_e32 v93, v99.l
	v_cndmask_b32_e64 v82, 0, v147, s4
	v_perm_b32 v89, v90, v96, v92
	v_perm_b32 v90, v97, v95, v91
	v_perm_b32 v91, v98, v100, v94
	v_cndmask_b32_e64 v92, 0, v2, s6
	v_cndmask_b32_e64 v93, 0, v93, s7
	ds_store_2addr_b64 v252, v[80:81], v[82:83] offset1:16
	ds_store_2addr_b64 v101, v[84:85], v[86:87] offset0:128 offset1:144
	ds_store_2addr_b64 v106, v[88:89], v[90:91] offset0:128 offset1:144
	ds_store_b32 v248, v92 offset:18432
	ds_store_b32 v250, v93 offset:18944
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	scratch_store_b32 off, v0, off offset:276 ; 4-byte Folded Spill
	s_barrier_wait -1
	s_wait_storecnt 0x0
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=1
	v_add_nc_u32_e32 v104, 2, v104
	s_mov_b32 s11, s9
	v_add_co_u32 v81, vcc_lo, v207, s14
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[16:17], s[10:11]
	v_lshlrev_b64_e32 v[83:84], 2, v[104:105]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v77, s11, s38, v205
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v78, null, s39, 0, s11
	v_add_co_u32 v79, s11, s38, v206
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, 0, v208, vcc_lo
	v_add_co_u32 v85, vcc_lo, v209, s14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v80, null, s39, 0, s11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v86, null, 0, v210, vcc_lo
	v_add_co_u32 v83, vcc_lo, s18, v83
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, s19, v84, vcc_lo
	v_add_co_u32 v87, vcc_lo, v102, s14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v88, null, 0, v103, vcc_lo
	s_clause 0x1
	global_load_b128 v[149:152], v[77:78], off
	global_load_b128 v[145:148], v[79:80], off
	global_load_b64 v[195:196], v[81:82], off offset:8
	global_load_b64 v[193:194], v[85:86], off offset:8
	global_load_b32 v2, v[83:84], off
	global_load_b32 v211, v[87:88], off
	v_mov_b32_e32 v8, v105
	scratch_store_b64 off, v[7:8], off offset:192 ; 8-byte Folded Spill
	s_branch .LBB0_19
.LBB0_18:                               ;   in Loop: Header=BB0_16 Depth=1
	scratch_store_b64 off, v[104:105], off offset:192 ; 8-byte Folded Spill
	v_mov_b32_e32 v150, v8
	v_mov_b32_e32 v152, v77
	v_mov_b32_e32 v146, v78
	v_mov_b32_e32 v148, v79
.LBB0_19:                               ;   in Loop: Header=BB0_16 Depth=1
	ds_load_2addr_b64 v[197:200], v246 offset1:144
	ds_load_2addr_b64 v[77:80], v176 offset1:144
	ds_load_2addr_b64 v[201:204], v230 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[197:198], v[77:78], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[199:200], v[77:78], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[197:198], v[79:80], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[199:200], v[79:80], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[197:198], v[201:202], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[199:200], v[201:202], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[197:198], v[203:204], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[199:200], v[203:204], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[197:200], v246 offset0:36 offset1:180
	ds_load_2addr_b64 v[201:204], v176 offset0:36 offset1:180
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[197:198], v[201:202], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[199:200], v[201:202], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[197:198], v[203:204], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[199:200], v[203:204], v[93:100]
	ds_load_2addr_b64 v[201:204], v230 offset0:68 offset1:212
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[197:198], v[201:202], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[199:200], v[201:202], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[197:198], v[203:204], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[199:200], v[203:204], v[77:84]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[197:200], v246 offset0:72 offset1:216
	ds_load_2addr_b64 v[201:204], v176 offset0:72 offset1:216
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[197:198], v[201:202], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[199:200], v[201:202], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[197:198], v[203:204], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[199:200], v[203:204], v[93:100]
	ds_load_2addr_b64 v[201:204], v230 offset0:104 offset1:248
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[197:198], v[201:202], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[199:200], v[201:202], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[197:198], v[203:204], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[199:200], v[203:204], v[77:84]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[197:200], v246 offset0:108 offset1:252
	ds_load_2addr_b64 v[201:204], v176 offset0:108 offset1:252
	v_add_nc_u32_e32 v0, 0xc00, v176
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[197:198], v[201:202], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[199:200], v[201:202], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[197:198], v[203:204], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[199:200], v[203:204], v[93:100]
	ds_load_2addr_b64 v[201:204], v0 offset0:12 offset1:156
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[197:198], v[201:202], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[199:200], v[201:202], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[197:198], v[203:204], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[199:200], v[203:204], v[77:84]
	; sched_barrier mask(0x00000000)
	v_dual_mov_b32 v201, v5 :: v_dual_mov_b32 v200, v144
	v_mov_b32_e32 v5, v7
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cndmask_b32_e64 v7, 0, v9, s5
	v_cndmask_b32_e64 v156, 0, v10, s5
	v_cndmask_b32_e64 v154, 0, v5, s3
	v_cndmask_b32_e64 v153, 0, v6, s3
	v_mov_b32_e32 v1, v211
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v144, 0xf0f0f0f, v7
	v_cndmask_b32_e64 v7, 0, v4, s3
	v_lshrrev_b32_e32 v178, 4, v156
	v_cndmask_b32_e64 v197, 0, v12, s2
	v_and_b32_e32 v155, 0xf0f0f0f, v8
	v_cndmask_b32_e64 v8, 0, v201, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off           ; 4-byte Folded Reload
	v_perm_b32 v177, v155, v144, 0x5010400
	v_perm_b32 v144, v155, v144, 0x7030602
	ds_store_2addr_b64 v251, v[7:8], v[153:154] offset1:16
	v_and_b32_e32 v154, 0xf0f0f0f, v156
	v_and_b32_e32 v155, 0xf0f0f0f, v178
	v_and_b32_e32 v179, 0x7070707, v177
	v_lshrrev_b32_e32 v7, 1, v177
	v_and_b32_e32 v153, 0x7070707, v144
	v_lshrrev_b32_e32 v144, 1, v144
	v_perm_b32 v178, v155, v154, 0x5010400
	v_perm_b32 v8, s15, 0x44403800, v179
	v_or_b32_e32 v156, 0x50505050, v179
	v_and_or_b32 v7, v7, s28, 0x3020100
	v_perm_b32 v177, s15, 0x44403800, v153
	v_or_b32_e32 v153, 0x50505050, v153
	v_and_or_b32 v144, v144, s28, 0x3020100
	v_cndmask_b32_e64 v179, 0, v11, s2
	v_perm_b32 v7, v156, v8, v7
	v_and_b32_e32 v156, 0x7070707, v178
	v_perm_b32 v154, v155, v154, 0x7030602
	v_perm_b32 v8, v153, v177, v144
	v_lshrrev_b32_e32 v144, 4, v179
	v_lshrrev_b32_e32 v153, 1, v178
	v_perm_b32 v155, s15, 0x44403800, v156
	v_and_b32_e32 v177, 0x7070707, v154
	v_and_b32_e32 v178, 0xf0f0f0f, v179
	v_and_b32_e32 v144, 0xf0f0f0f, v144
	v_or_b32_e32 v156, 0x50505050, v156
	v_lshrrev_b32_e32 v154, 1, v154
	v_and_or_b32 v153, v153, s28, 0x3020100
	v_perm_b32 v179, s15, 0x44403800, v177
	v_perm_b32 v180, v144, v178, 0x5010400
	v_or_b32_e32 v177, 0x50505050, v177
	v_and_or_b32 v154, v154, s28, 0x3020100
	v_perm_b32 v153, v156, v155, v153
	v_perm_b32 v144, v144, v178, 0x7030602
	v_lshrrev_b32_e32 v156, 4, v197
	v_and_b32_e32 v155, 0x7070707, v180
	v_perm_b32 v154, v177, v179, v154
	v_lshrrev_b32_e32 v177, 1, v180
	v_and_b32_e32 v179, 0x7070707, v144
	v_and_b32_e32 v180, 0xf0f0f0f, v197
	v_and_b32_e32 v156, 0xf0f0f0f, v156
	v_lshrrev_b32_e32 v144, 1, v144
	v_perm_b32 v178, s15, 0x44403800, v155
	v_or_b32_e32 v155, 0x50505050, v155
	v_and_or_b32 v177, v177, s28, 0x3020100
	v_perm_b32 v197, s15, 0x44403800, v179
	v_perm_b32 v198, v156, v180, 0x5010400
	v_or_b32_e32 v179, 0x50505050, v179
	v_and_or_b32 v144, v144, s28, 0x3020100
	v_perm_b32 v180, v156, v180, 0x7030602
	v_perm_b32 v155, v155, v178, v177
	v_and_b32_e32 v199, 0x7070707, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v156, v179, v197, v144
	v_lshrrev_b32_e32 v144, 1, v198
	v_and_b32_e32 v177, 0x7070707, v180
	v_lshrrev_b32_e32 v179, 1, v180
	v_perm_b32 v178, s15, 0x44403800, v199
	v_or_b32_e32 v180, 0x50505050, v199
	v_and_or_b32 v144, v144, s28, 0x3020100
	v_perm_b32 v197, s15, 0x44403800, v177
	v_or_b32_e32 v198, 0x50505050, v177
	v_and_or_b32 v199, v179, s28, 0x3020100
	v_cndmask_b32_e64 v179, 0, v141, s4
	v_perm_b32 v177, v180, v178, v144
	v_cndmask_b32_e64 v180, 0, v142, s4
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v178, v198, v197, v199
	v_cndmask_b32_e64 v198, 0, v200, s4
	v_cndmask_b32_e64 v197, 0, v143, s4
	ds_store_2addr_b64 v252, v[179:180], v[197:198] offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v0, v[7:8], v[153:154] offset0:128 offset1:144
	scratch_load_b32 v0, off, off offset:4  ; 4-byte Folded Reload
	v_cndmask_b32_e64 v8, 0, 1, s8
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v0, v[155:156], v[177:178] offset0:128 offset1:144
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_16 Depth=1
	s_mov_b32 s11, s9
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[16:17], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, s8, s38, v205
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s39, 0, s8
	v_add_co_u32 v9, s8, s38, v206
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s39, 0, s8
	s_clause 0x1
	global_load_b128 v[4:7], v[4:5], off offset:64
	global_load_b128 v[141:144], v[9:10], off offset:64
	v_add_co_u32 v9, vcc_lo, v207, s14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, 0, v208, vcc_lo
	v_add_co_u32 v11, vcc_lo, v209, s14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, 0, v210, vcc_lo
	global_load_b64 v[9:10], v[9:10], off offset:40
	global_load_b64 v[11:12], v[11:12], off offset:40
	s_wait_loadcnt 0x3
	v_dual_mov_b32 v0, v5 :: v_dual_mov_b32 v5, v7
	scratch_store_b32 off, v0, off offset:208 ; 4-byte Folded Spill
	s_wait_loadcnt 0x2
	v_mov_b32_e32 v0, v144
	scratch_store_b32 off, v0, off offset:204 ; 4-byte Folded Spill
	s_branch .LBB0_22
.LBB0_21:                               ;   in Loop: Header=BB0_16 Depth=1
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v201, off offset:208
	scratch_store_b32 off, v200, off offset:204
.LBB0_22:                               ;   in Loop: Header=BB0_16 Depth=1
	ds_load_2addr_b64 v[197:200], v246 offset1:144
	ds_load_2addr_b64 v[201:204], v176 offset1:144
	ds_load_2addr_b64 v[205:208], v230 offset0:32 offset1:176
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[209:212], v246 offset0:36 offset1:180
	ds_load_2addr_b64 v[213:216], v176 offset0:36 offset1:180
	ds_load_2addr_b64 v[217:220], v230 offset0:68 offset1:212
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[221:224], v246 offset0:72 offset1:216
	ds_load_2addr_b64 v[225:228], v176 offset0:72 offset1:216
	ds_load_2addr_b64 v[247:250], v230 offset0:104 offset1:248
	v_mov_b32_e32 v0, v230
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v7, 0xc00, v176
	ds_load_2addr_b64 v[251:254], v246 offset0:108 offset1:252
	ds_load_2addr_b64 v[153:156], v176 offset0:108 offset1:252
	ds_load_2addr_b64 v[177:180], v7 offset0:12 offset1:156
	; sched_barrier mask(0x00000000)
	v_mov_b32_e32 v144, v149
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v7, off, off offset:64
	scratch_load_b32 v149, off, off offset:8
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[199:200], v[201:202], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[199:200], v[203:204], v[93:100]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[199:200], v[205:206], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[199:200], v[207:208], v[77:84]
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[197:198], v[201:202], v[133:140]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[211:212], v[213:214], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[211:212], v[215:216], v[93:100]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[211:212], v[217:218], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[211:212], v[219:220], v[77:84]
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[209:210], v[213:214], v[133:140]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[223:224], v[225:226], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[223:224], v[227:228], v[93:100]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[223:224], v[247:248], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[223:224], v[249:250], v[77:84]
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[221:222], v[225:226], v[133:140]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[253:254], v[153:154], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[253:254], v[155:156], v[93:100]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[253:254], v[177:178], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[253:254], v[179:180], v[77:84]
	v_mov_b32_e32 v253, v229
	ds_load_2addr_b32 v[229:230], v229 offset1:1
	scratch_load_b32 v254, off, off offset:356 ; 4-byte Folded Reload
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[197:198], v[203:204], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[197:198], v[205:206], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[197:198], v[207:208], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[251:252], v[153:154], v[133:140]
	v_cmp_ne_u32_e32 vcc_lo, 1, v8
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[209:210], v[215:216], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[209:210], v[217:218], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[209:210], v[219:220], v[109:116]
	s_wait_loadcnt 0x2
	ds_load_2addr_b32 v[197:198], v7 offset1:16
	ds_load_2addr_b32 v[7:8], v7 offset0:32 offset1:48
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[225:226], v149 offset1:1
	scratch_load_b32 v149, off, off offset:12 ; 4-byte Folded Reload
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[221:222], v[227:228], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[221:222], v[247:248], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[221:222], v[249:250], v[109:116]
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[251:252], v[155:156], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[251:252], v[177:178], v[117:124]
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[251:252], v[179:180], v[109:116]
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[227:228], v254 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[223:224], v149 offset1:1
	scratch_load_b32 v149, off, off offset:16 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[221:222], v149 offset1:1
	scratch_load_b32 v149, off, off offset:20 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[219:220], v149 offset1:1
	scratch_load_b32 v149, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[217:218], v149 offset1:1
	scratch_load_b32 v149, off, off offset:28 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[215:216], v149 offset1:1
	scratch_load_b32 v149, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[213:214], v149 offset1:1
	scratch_load_b32 v149, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[211:212], v149 offset1:1
	scratch_load_b32 v149, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[209:210], v149 offset1:1
	scratch_load_b32 v149, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[207:208], v149 offset1:1
	scratch_load_b32 v149, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[205:206], v149 offset1:1
	scratch_load_b32 v149, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[203:204], v149 offset1:1
	scratch_load_b32 v149, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[201:202], v149 offset1:1
	scratch_load_b32 v149, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[199:200], v149 offset1:1
	v_mov_b32_e32 v149, v145
	v_mov_b32_e32 v145, v147
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_clause 0x4                            ; 20-byte Folded Reload
	scratch_load_b32 v248, off, off offset:320
	scratch_load_b32 v249, off, off offset:324
	scratch_load_b32 v250, off, off offset:328
	scratch_load_b32 v251, off, off offset:332
	scratch_load_b32 v252, off, off offset:336
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_15
; %bb.23:                               ;   in Loop: Header=BB0_16 Depth=1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v147, v150 :: v_dual_mov_b32 v150, v152
	v_mov_b32_e32 v154, v146
	v_cndmask_b32_e64 v146, 0, v144, s3
	v_cndmask_b32_e64 v151, 0, v151, s3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v147, 0, v147, s3
	v_cndmask_b32_e64 v152, 0, v150, s3
	v_cndmask_b32_e64 v153, 0, v195, s5
	v_lshrrev_b32_e32 v195, v249, v1
	v_lshrrev_b32_e32 v155, 4, v153
	v_and_b32_e32 v144, 0xf0f0f0f, v153
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v251, v[146:147], v[151:152] offset1:16
	v_cndmask_b32_e64 v151, 0, v196, s5
	v_cndmask_b32_e64 v146, 0, v149, s4
	v_cndmask_b32_e64 v149, 0, v148, s4
	v_and_b32_e32 v150, 0xf0f0f0f, v155
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v148, 4, v151
	v_and_b32_e32 v151, 0xf0f0f0f, v151
	v_cndmask_b32_e64 v147, 0, v154, s4
	v_perm_b32 v153, v150, v144, 0x5010400
	v_perm_b32 v144, v150, v144, 0x7030602
	v_and_b32_e32 v154, 0xf0f0f0f, v148
	v_cndmask_b32_e64 v148, 0, v145, s4
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v152, 0x7070707, v153
	v_lshrrev_b32_e32 v150, 1, v153
	v_and_b32_e32 v155, 0x7070707, v144
	ds_store_2addr_b64 v252, v[146:147], v[148:149] offset1:16
	scratch_load_b32 v146, off, off         ; 4-byte Folded Reload
	v_perm_b32 v153, s15, 0x44403800, v152
	v_or_b32_e32 v152, 0x50505050, v152
	v_and_or_b32 v150, v150, s28, 0x3020100
	v_perm_b32 v145, v154, v151, 0x5010400
	v_lshrrev_b32_e32 v156, 1, v144
	v_perm_b32 v151, v154, v151, 0x7030602
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v144, v152, v153, v150
	v_perm_b32 v150, s15, 0x44403800, v155
	v_or_b32_e32 v152, 0x50505050, v155
	v_and_b32_e32 v153, 0x7070707, v145
	v_and_or_b32 v155, v156, s28, 0x3020100
	v_lshrrev_b32_e32 v145, 1, v145
	v_cndmask_b32_e64 v156, 0, v193, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_perm_b32 v154, s15, 0x44403800, v153
	v_or_b32_e32 v153, 0x50505050, v153
	v_and_or_b32 v178, v145, s28, 0x3020100
	v_perm_b32 v145, v152, v150, v155
	v_and_b32_e32 v152, 0x7070707, v151
	v_lshrrev_b32_e32 v151, 1, v151
	v_lshrrev_b32_e32 v177, 4, v156
	v_perm_b32 v150, v153, v154, v178
	v_cndmask_b32_e64 v153, 0, v194, s2
	v_perm_b32 v154, s15, 0x44403800, v152
	v_or_b32_e32 v152, 0x50505050, v152
	v_and_or_b32 v151, v151, s28, 0x3020100
	v_and_b32_e32 v155, 0xf0f0f0f, v156
	v_lshrrev_b32_e32 v178, 4, v153
	v_and_b32_e32 v156, 0xf0f0f0f, v177
	v_and_b32_e32 v153, 0xf0f0f0f, v153
	v_perm_b32 v151, v152, v154, v151
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v178, 0xf0f0f0f, v178
	v_perm_b32 v177, v156, v155, 0x5010400
	v_perm_b32 v155, v156, v155, 0x7030602
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_perm_b32 v179, v178, v153, 0x5010400
	v_perm_b32 v153, v178, v153, 0x7030602
	v_and_b32_e32 v156, 0x7070707, v177
	v_lshrrev_b32_e32 v152, 1, v177
	v_and_b32_e32 v177, 0x7070707, v155
	v_lshrrev_b32_e32 v155, 1, v155
	v_and_b32_e32 v193, 0x7070707, v179
	v_lshrrev_b32_e32 v178, 1, v179
	v_and_b32_e32 v194, 0x7070707, v153
	v_lshrrev_b32_e32 v153, 1, v153
	v_perm_b32 v154, s15, 0x44403800, v156
	v_or_b32_e32 v156, 0x50505050, v156
	v_and_or_b32 v152, v152, s28, 0x3020100
	v_perm_b32 v180, s15, 0x44403800, v177
	v_or_b32_e32 v177, 0x50505050, v177
	v_and_or_b32 v155, v155, s28, 0x3020100
	v_perm_b32 v179, s15, 0x44403800, v193
	v_or_b32_e32 v193, 0x50505050, v193
	v_and_or_b32 v178, v178, s28, 0x3020100
	v_perm_b32 v196, s15, 0x44403800, v194
	v_or_b32_e32 v194, 0x50505050, v194
	v_and_or_b32 v247, v153, s28, 0x3020100
	v_perm_b32 v152, v156, v154, v152
	v_cvt_f32_f16_e64 v156, v195.l
	v_perm_b32 v153, v177, v180, v155
	v_perm_b32 v154, v193, v179, v178
	v_perm_b32 v155, v194, v196, v247
	v_cndmask_b32_e64 v177, 0, v2, s6
	v_cndmask_b32_e64 v156, 0, v156, s7
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v146, v[144:145], v[150:151] offset0:128 offset1:144
	scratch_load_b32 v144, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v144, v[152:153], v[154:155] offset0:128 offset1:144
	ds_store_b32 v248, v177 offset:18432
	ds_store_b32 v250, v156 offset:18944
	s_branch .LBB0_15
.LBB0_24:
	s_clause 0x5                            ; 24-byte Folded Reload
	scratch_load_b32 v7, off, off offset:380 th:TH_LOAD_LU
	scratch_load_b32 v0, off, off offset:360
	scratch_load_b32 v23, off, off offset:364
	scratch_load_b32 v24, off, off offset:368
	scratch_load_b32 v25, off, off offset:372
	scratch_load_b32 v8, off, off offset:376
.LBB0_25:
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v1, 6, v0
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v2, v7, v8
	s_clause 0x1
	s_load_b128 s[8:11], s[0:1], 0x40
	s_load_b64 s[6:7], s[0:1], 0x50
	v_and_b32_e32 v1, 0x3800, v1
	v_add_nc_u32_e32 v4, 17, v2
	v_add_nc_u32_e32 v5, 18, v2
	v_add_nc_u32_e32 v12, 19, v2
	v_add_nc_u32_e32 v13, 20, v2
	v_add_nc_u32_e32 v6, 0, v1
	v_and_b32_e32 v1, 31, v2
	v_and_b32_e32 v4, 31, v4
	v_and_b32_e32 v5, 31, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v7, v8, 7, v6
	v_xor_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v10, v4, 2, v7
	v_lshl_add_u32 v9, v1, 2, v7
	v_add_nc_u32_e32 v1, 21, v2
	v_add_nc_u32_e32 v4, 22, v2
	v_lshl_add_u32 v8, v2, 2, v7
	v_add_nc_u32_e32 v2, 23, v2
	v_lshl_add_u32 v11, v5, 2, v7
	v_and_b32_e32 v5, 31, v12
	v_and_b32_e32 v12, 31, v13
	v_and_b32_e32 v1, 31, v1
	v_and_b32_e32 v4, 31, v4
	v_and_b32_e32 v2, 31, v2
	v_lshl_add_u32 v18, v5, 2, v7
	v_lshl_add_u32 v19, v12, 2, v7
	v_lshl_add_u32 v20, v1, 2, v7
	v_lshl_add_u32 v21, v4, 2, v7
	v_lshl_add_u32 v22, v2, 2, v7
	ds_store_2addr_b32 v8, v174, v175 offset1:1
	ds_store_2addr_b32 v8, v172, v173 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v170, v171 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v166, v169 offset0:6 offset1:7
	ds_store_b32 v9, v165
	ds_store_b32 v10, v164
	ds_store_b32 v11, v163
	ds_store_b32 v18, v162
	ds_store_b32 v19, v161
	ds_store_b32 v20, v160
	ds_store_b32 v21, v159
	ds_store_b32 v22, v245
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_or3_b32 v7, s33, v25, v23
	v_add_nc_u32_e32 v4, s35, v24
	v_lshl_add_u32 v12, v23, 2, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s3, s34, v7
	v_cmp_gt_i32_e64 s4, s29, v4
	v_cmp_le_i32_e64 s2, s34, v7
	v_cmp_gt_i32_e64 s0, s30, v7
	v_cmp_gt_i32_e64 s1, s31, v7
	v_ashrrev_i32_e32 v5, 31, v4
	s_and_b32 s3, s4, s3
	v_cmp_gt_i32_e32 vcc_lo, s24, v7
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	ds_load_b32 v14, v12
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v15, s10 :: v_dual_mov_b32 v16, s11
	v_add_co_u32 v1, s3, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v15, s6, v15, s1
	v_cndmask_b32_e64 v16, s7, v16, s1
	global_load_b32 v13, v[1:2], off
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v15, v15, s8, s0
	v_cndmask_b32_e64 v16, v16, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v15, v15, s22, vcc_lo
	v_cndmask_b32_e64 v16, v16, s23, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v2, v2, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	v_mad_co_u64_u32 v[1:2], null, v2, v4, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v2, 0 :: v_dual_mul_f32 v13, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, s3, v15, v1
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v16, v2, s3
	global_store_b32 v[1:2], v13, off
.LBB0_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v14, 1, v4
	v_add_nc_u32_e32 v13, 1, v0
	s_xor_b32 s4, s2, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s3, s29, v14
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s3, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v17, s10 :: v_dual_and_b32 v16, 31, v13
	v_mov_b32_e32 v23, s11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v16, v16, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v16, v16 offset:128
	v_cndmask_b32_e64 v17, s6, v17, s1
	global_load_b32 v15, v[1:2], off offset:4
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v17, v17, s8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v17, v17, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v14, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v14, s7, v23, s1
	v_cndmask_b32_e64 v14, v14, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cndmask_b32_e64 v14, v14, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v17, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v14, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v15, v15, v16
	global_store_b32 v[1:2], v15, off
.LBB0_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v15, 2, v4
	v_add_nc_u32_e32 v14, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v15
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v24, s11 :: v_dual_and_b32 v17, 31, v14
	v_mov_b32_e32 v23, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v17, v17, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v17, v17 offset:256
	global_load_b32 v16, v[1:2], off offset:8
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v2, v2, s25, s0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	v_mad_co_u64_u32 v[1:2], null, v2, v15, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v23, s6, v23, s1
	v_cndmask_b32_e64 v15, s7, v24, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v23, v23, s8, s0
	v_cndmask_b32_e64 v15, v15, s9, s0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v23, v23, s22, vcc_lo
	v_cndmask_b32_e64 v15, v15, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v23, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v15, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v16, v16, v17
	global_store_b32 v[1:2], v16, off
.LBB0_31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v16, 3, v4
	v_add_nc_u32_e32 v15, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v16
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_33
; %bb.32:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v24, s10 :: v_dual_and_b32 v23, 31, v15
	v_mov_b32_e32 v25, s11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v23, v23, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v23, v23 offset:384
	v_cndmask_b32_e64 v24, s6, v24, s1
	global_load_b32 v17, v[1:2], off offset:12
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v24, v24, s8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v24, v24, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v16, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v16, s7, v25, s1
	v_cndmask_b32_e64 v16, v16, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cndmask_b32_e64 v16, v16, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v24, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v16, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v17, v17, v23
	global_store_b32 v[1:2], v17, off
.LBB0_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v17, 4, v4
	v_add_nc_u32_e32 v16, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v17
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_35
; %bb.34:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v25, s10 :: v_dual_and_b32 v24, 31, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v24, v24, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v24, v24 offset:512
	v_cndmask_b32_e64 v25, s6, v25, s1
	global_load_b32 v23, v[1:2], off offset:16
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_mov_b32_e32 v26, s11
	v_cndmask_b32_e64 v25, v25, s8, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v25, v25, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v17, v[1:2]
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, v25, v1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v23, v23, v24
	v_cndmask_b32_e64 v17, s7, v26, s1
	v_cndmask_b32_e64 v17, v17, s9, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v17, v17, s23, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v17, v2, s2
	global_store_b32 v[1:2], v23, off
.LBB0_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v23, 5, v4
	v_add_nc_u32_e32 v17, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v23
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_37
; %bb.36:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v26, s10 :: v_dual_and_b32 v25, 31, v17
	v_mov_b32_e32 v27, s11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v25, v25, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v25, v25 offset:640
	v_cndmask_b32_e64 v26, s6, v26, s1
	global_load_b32 v24, v[1:2], off offset:20
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v26, v26, s8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v23, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v23, s7, v27, s1
	v_cndmask_b32_e64 v23, v23, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cndmask_b32_e64 v23, v23, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v26, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v23, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v24, v24, v25
	global_store_b32 v[1:2], v24, off
.LBB0_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v24, 6, v4
	v_add_nc_u32_e32 v23, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v24
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_39
; %bb.38:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v27, s10 :: v_dual_and_b32 v26, 31, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v26, v26, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v26, v26 offset:768
	v_cndmask_b32_e64 v27, s6, v27, s1
	global_load_b32 v25, v[1:2], off offset:24
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_mov_b32_e32 v28, s11
	v_cndmask_b32_e64 v27, v27, s8, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v27, v27, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v24, v[1:2]
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, v27, v1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v25, v25, v26
	v_cndmask_b32_e64 v24, s7, v28, s1
	v_cndmask_b32_e64 v24, v24, s9, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, s23, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v24, v2, s2
	global_store_b32 v[1:2], v25, off
.LBB0_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v25, 7, v4
	v_add_nc_u32_e32 v24, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v25
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_41
; %bb.40:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v28, s10 :: v_dual_and_b32 v27, 31, v24
	v_mov_b32_e32 v29, s11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v27, v27, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v27, v27 offset:896
	v_cndmask_b32_e64 v28, s6, v28, s1
	global_load_b32 v26, v[1:2], off offset:28
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v28, v28, s8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v28, v28, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v25, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v25, s7, v29, s1
	v_cndmask_b32_e64 v25, v25, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cndmask_b32_e64 v25, v25, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v28, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v25, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v26, v26, v27
	global_store_b32 v[1:2], v26, off
.LBB0_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v26, 8, v4
	v_add_nc_u32_e32 v25, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v26
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_43
; %bb.42:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v29, s10 :: v_dual_and_b32 v28, 31, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v28, v28, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v28, v28 offset:1024
	v_cndmask_b32_e64 v29, s6, v29, s1
	global_load_b32 v27, v[1:2], off offset:32
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_mov_b32_e32 v30, s11
	v_cndmask_b32_e64 v29, v29, s8, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v29, v29, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v26, v[1:2]
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, v29, v1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	v_cndmask_b32_e64 v26, s7, v30, s1
	v_cndmask_b32_e64 v26, v26, s9, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, s23, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v26, v2, s2
	global_store_b32 v[1:2], v27, off
.LBB0_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v27, 9, v4
	v_add_nc_u32_e32 v26, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v27
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_45
; %bb.44:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v30, s10 :: v_dual_and_b32 v29, 31, v26
	v_mov_b32_e32 v31, s11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v29, v29, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v29, v29 offset:1152
	v_cndmask_b32_e64 v30, s6, v30, s1
	global_load_b32 v28, v[1:2], off offset:36
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v30, v30, s8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v30, v30, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v27, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v27, s7, v31, s1
	v_cndmask_b32_e64 v27, v27, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cndmask_b32_e64 v27, v27, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v30, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v27, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v28, v28, v29
	global_store_b32 v[1:2], v28, off
.LBB0_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v28, 10, v4
	v_add_nc_u32_e32 v27, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v28
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_47
; %bb.46:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v31, s10 :: v_dual_and_b32 v30, 31, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v30, v30, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v30, v30 offset:1280
	v_cndmask_b32_e64 v31, s6, v31, s1
	global_load_b32 v29, v[1:2], off offset:40
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_mov_b32_e32 v32, s11
	v_cndmask_b32_e64 v31, v31, s8, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v31, v31, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v28, v[1:2]
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, v31, v1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v29, v29, v30
	v_cndmask_b32_e64 v28, s7, v32, s1
	v_cndmask_b32_e64 v28, v28, s9, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, s23, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v28, v2, s2
	global_store_b32 v[1:2], v29, off
.LBB0_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v29, 11, v4
	v_add_nc_u32_e32 v28, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v29
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_49
; %bb.48:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v32, s10 :: v_dual_and_b32 v31, 31, v28
	v_mov_b32_e32 v33, s11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v31, v31, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v31, v31 offset:1408
	v_cndmask_b32_e64 v32, s6, v32, s1
	global_load_b32 v30, v[1:2], off offset:44
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v32, v32, s8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v32, v32, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v29, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v29, s7, v33, s1
	v_cndmask_b32_e64 v29, v29, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cndmask_b32_e64 v29, v29, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v32, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v29, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[1:2], v30, off
.LBB0_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v30, 12, v4
	v_add_nc_u32_e32 v29, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v30
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_51
; %bb.50:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v33, s10 :: v_dual_and_b32 v32, 31, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v32, v32 offset:1536
	v_cndmask_b32_e64 v33, s6, v33, s1
	global_load_b32 v31, v[1:2], off offset:48
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_mov_b32_e32 v34, s11
	v_cndmask_b32_e64 v33, v33, s8, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, v33, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v30, v[1:2]
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, v33, v1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v31, v31, v32
	v_cndmask_b32_e64 v30, s7, v34, s1
	v_cndmask_b32_e64 v30, v30, s9, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, s23, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v30, v2, s2
	global_store_b32 v[1:2], v31, off
.LBB0_51:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v31, 13, v4
	v_add_nc_u32_e32 v30, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v31
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_53
; %bb.52:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v34, s10 :: v_dual_and_b32 v33, 31, v30
	v_mov_b32_e32 v35, s11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v33, v33, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v33, v33 offset:1664
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v32, v[1:2], off offset:52
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v31, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v31, s7, v35, s1
	v_cndmask_b32_e64 v31, v31, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cndmask_b32_e64 v31, v31, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v34, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v31, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v33
	global_store_b32 v[1:2], v32, off
.LBB0_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v32, 14, v4
	v_add_nc_u32_e32 v31, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v32
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_55
; %bb.54:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s10 :: v_dual_and_b32 v34, 31, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v34, v34, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v34, v34 offset:1792
	v_cndmask_b32_e64 v35, s6, v35, s1
	global_load_b32 v33, v[1:2], off offset:56
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_mov_b32_e32 v36, s11
	v_cndmask_b32_e64 v35, v35, s8, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v35, v35, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v32, v[1:2]
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, v35, v1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v33, v33, v34
	v_cndmask_b32_e64 v32, s7, v36, s1
	v_cndmask_b32_e64 v32, v32, s9, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, s23, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v32, v2, s2
	global_store_b32 v[1:2], v33, off
.LBB0_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v33, 15, v4
	v_add_nc_u32_e32 v32, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s29, v33
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_57
; %bb.56:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v36, s10 :: v_dual_and_b32 v35, 31, v32
	v_mov_b32_e32 v37, s11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v35, v35, 2, v6
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	ds_load_b32 v35, v35 offset:1920
	v_cndmask_b32_e64 v36, s6, v36, s1
	global_load_b32 v34, v[1:2], off offset:60
	v_dual_mov_b32 v1, s30 :: v_dual_mov_b32 v2, s26
	v_cndmask_b32_e64 v36, v36, s8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s31, v1, s1
	v_cndmask_b32_e64 v2, s27, v2, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v36, v36, s22, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s25, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v2, s24, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[1:2], null, v2, v33, v[1:2]
	v_mov_b32_e32 v2, 0
	v_cndmask_b32_e64 v33, s7, v37, s1
	v_cndmask_b32_e64 v33, v33, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_cndmask_b32_e64 v33, v33, s23, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, v36, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, v33, v2, s2
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[1:2], v34, off
.LBB0_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 16, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v243, v244 offset1:1
	ds_store_2addr_b32 v8, v241, v242 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v239, v240 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v237, v238 offset0:6 offset1:7
	ds_store_b32 v9, v236
	ds_store_b32 v10, v235
	ds_store_b32 v11, v234
	ds_store_b32 v18, v233
	ds_store_b32 v19, v232
	ds_store_b32 v20, v231
	ds_store_b32 v21, v255
	ds_store_b32 v22, v3
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_59
; %bb.58:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	ds_load_b32 v33, v12
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:64
	v_mov_b32_e32 v1, s30
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 17, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_61
; %bb.60:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v13
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:68
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:128
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_61:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 18, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_63
; %bb.62:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v14
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:72
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:256
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 19, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_65
; %bb.64:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v15
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:76
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:384
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 20, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_67
; %bb.66:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v16
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:80
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:512
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 21, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_69
; %bb.68:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v17
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:84
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:640
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 22, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_71
; %bb.70:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v23
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:88
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:768
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 23, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_73
; %bb.72:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v24
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:92
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:896
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 24, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_75
; %bb.74:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v25
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:96
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1024
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 25, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_77
; %bb.76:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v26
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:100
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1152
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 26, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_79
; %bb.78:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v27
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:104
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1280
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 27, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_81
; %bb.80:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v28
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:108
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1408
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 28, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_83
; %bb.82:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v29
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:112
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1536
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 29, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_85
; %bb.84:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v30
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:116
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1664
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 30, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_87
; %bb.86:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v31
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:120
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1792
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 31, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_89
; %bb.88:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v32
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:124
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1920
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 32, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v69, v68 offset1:1
	ds_store_2addr_b32 v8, v66, v67 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v64, v65 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v62, v63 offset0:6 offset1:7
	ds_store_b32 v9, v61
	ds_store_b32 v10, v60
	ds_store_b32 v11, v59
	ds_store_b32 v18, v58
	ds_store_b32 v19, v57
	ds_store_b32 v20, v52
	ds_store_b32 v21, v51
	ds_store_b32 v22, v50
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_91
; %bb.90:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	ds_load_b32 v33, v12
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:128
	v_mov_b32_e32 v1, s30
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 33, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_93
; %bb.92:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v13
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:132
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:128
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 34, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_95
; %bb.94:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v14
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:136
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:256
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 35, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_97
; %bb.96:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v15
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:140
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:384
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 36, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_99
; %bb.98:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v16
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:144
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:512
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 37, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_101
; %bb.100:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v17
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:148
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:640
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 38, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_103
; %bb.102:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v23
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:152
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:768
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 39, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_105
; %bb.104:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v24
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:156
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:896
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 40, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_107
; %bb.106:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v25
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:160
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1024
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_107:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 41, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_109
; %bb.108:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v26
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:164
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1152
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_109:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 42, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_111
; %bb.110:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v27
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:168
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1280
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_111:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 43, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_113
; %bb.112:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v28
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:172
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1408
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_113:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 44, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_115
; %bb.114:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v29
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:176
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1536
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_115:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 45, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_117
; %bb.116:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v30
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:180
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1664
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 46, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_119
; %bb.118:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v31
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:184
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1792
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_119:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 47, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_121
; %bb.120:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_and_b32_e32 v33, 31, v32
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v35, s11 :: v_dual_mov_b32 v34, s10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_lshl_add_u32 v33, v33, 2, v6
	v_cndmask_b32_e64 v34, s6, v34, s1
	global_load_b32 v2, v[1:2], off offset:188
	v_mov_b32_e32 v1, s30
	ds_load_b32 v33, v33 offset:1920
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v34, v34, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v35, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v34, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_121:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 48, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v55, v56 offset1:1
	ds_store_2addr_b32 v8, v53, v54 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v48, v49 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v46, v47 offset0:6 offset1:7
	ds_store_b32 v9, v45
	ds_store_b32 v10, v44
	ds_store_b32 v11, v43
	ds_store_b32 v18, v42
	ds_store_b32 v19, v41
	ds_store_b32 v20, v40
	ds_store_b32 v21, v39
	ds_store_b32 v22, v38
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_123
; %bb.122:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	ds_load_b32 v8, v12
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v9, s10 :: v_dual_mov_b32 v10, s11
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, s6, v9, s1
	global_load_b32 v2, v[1:2], off offset:192
	v_mov_b32_e32 v1, s30
	v_mov_b32_e32 v3, s26
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v3, s27, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v3, v3, s25, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	v_cndmask_b32_e64 v3, s7, v10, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v2, v2, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s9, s0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_123:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 49, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_125
; %bb.124:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v13
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:196
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:128
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_127
; %bb.126:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v14
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:200
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:256
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 51, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_129
; %bb.128:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v15
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:204
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:384
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_129:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 52, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_131
; %bb.130:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:208
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:512
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_131:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 53, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_133
; %bb.132:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v17
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:212
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:640
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_133:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 54, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_135
; %bb.134:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v23
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:216
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:768
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_135:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 55, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_137
; %bb.136:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v24
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:220
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:896
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_137:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 56, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_139
; %bb.138:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v25
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:224
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:1024
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_139:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 57, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_141
; %bb.140:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v26
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:228
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:1152
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_141:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 58, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_143
; %bb.142:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v27
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:232
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:1280
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_143:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 59, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_145
; %bb.144:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v28
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:236
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:1408
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_145:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_147
; %bb.146:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v29
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:240
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:1536
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_147:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 61, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_149
; %bb.148:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:244
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:1664
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_149:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 62, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_151
; %bb.150:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v3, s26 :: v_dual_mov_b32 v10, s11
	v_and_b32_e32 v8, 31, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	v_cndmask_b32_e64 v3, s27, v3, s1
	v_lshl_add_u32 v8, v8, 2, v6
	global_load_b32 v2, v[1:2], off offset:248
	v_mov_b32_e32 v1, s30
	v_cndmask_b32_e64 v3, v3, s25, s0
	ds_load_b32 v8, v8 offset:1792
	v_mov_b32_e32 v9, s10
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, s24, s0
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v8
	v_cndmask_b32_e64 v9, s6, v9, s1
	v_cndmask_b32_e64 v3, s7, v10, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v9, v9, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v9, v9, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, s2, v9, v0
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, s2
	global_store_b32 v[0:1], v2, off
.LBB0_151:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v0, 63, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s29, v0
	s_and_b32 s2, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_153
; %bb.152:
	v_lshlrev_b64_e32 v[1:2], 2, v[4:5]
	v_dual_mov_b32 v3, s26 :: v_dual_and_b32 v4, 31, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, s2, s20, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, s2
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_add_u32 v4, v4, 2, v6
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v6, s11
	v_cndmask_b32_e64 v3, s27, v3, s1
	global_load_b32 v2, v[1:2], off offset:252
	v_mov_b32_e32 v1, s30
	ds_load_b32 v4, v4 offset:1920
	v_mov_b32_e32 v5, s10
	v_cndmask_b32_e64 v3, v3, s25, s0
	v_cndmask_b32_e64 v1, s31, v1, s1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s24, vcc_lo
	v_cndmask_b32_e64 v1, v1, s24, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	v_sub_nc_u32_e32 v1, v7, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[0:1], null, v3, v0, v[1:2]
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, v2, v4
	v_cndmask_b32_e64 v5, s6, v5, s1
	v_cndmask_b32_e64 v3, s7, v6, s1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v5, v5, s8, s0
	v_cndmask_b32_e64 v3, v3, s9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v5, v5, s22, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v3, s23, vcc_lo
	v_add_co_u32 v0, vcc_lo, v5, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
.LBB0_153:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end0:
	.size	qkvza_r2, .Lfunc_end0-qkvza_r2
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel qkvza_r2
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 388
		.amdhsa_kernarg_size 112
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 40
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-qkvza_r2)<<4)&4080)>>4
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
	.set .Lqkvza_r2.num_vgpr, 256
	.set .Lqkvza_r2.num_agpr, 0
	.set .Lqkvza_r2.numbered_sgpr, 40
	.set .Lqkvza_r2.num_named_barrier, 0
	.set .Lqkvza_r2.private_seg_size, 388
	.set .Lqkvza_r2.uses_vcc, 1
	.set .Lqkvza_r2.uses_flat_scratch, 1
	.set .Lqkvza_r2.has_dyn_sized_stack, 0
	.set .Lqkvza_r2.has_recursion, 0
	.set .Lqkvza_r2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 33296
; TotalNumSgprs: 42
; NumVgprs: 256
; ScratchSize: 388
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 42
; NumVGPRsForWavesPerEU: 256
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.type	__hip_cuid_d4ad2e959ff7ee1d,@object ; @__hip_cuid_d4ad2e959ff7ee1d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d4ad2e959ff7ee1d
__hip_cuid_d4ad2e959ff7ee1d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d4ad2e959ff7ee1d, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_d4ad2e959ff7ee1d
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
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .offset:         108
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           qkvza_r2
    .private_segment_fixed_size: 388
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         qkvza_r2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 100
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
