	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	qkv_r2                  ; -- Begin function qkv_r2
	.globl	qkv_r2
	.p2align	8
	.type	qkv_r2,@function
qkv_r2:                                 ; @qkv_r2
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[24:27], s[0:1], 0x48
	s_load_b32 s28, s[0:1], 0x58
	s_load_b512 s[8:23], s[0:1], 0x0
	v_lshrrev_b32_e32 v3, 2, v0
	s_lshl_b32 s31, ttmp9, 7
	v_and_b32_e32 v1, 3, v0
	s_lshl_b32 s30, ttmp7, 7
	v_and_b32_e32 v7, 0x7f, v0
	v_or_b32_e32 v5, s31, v3
	v_or_b32_e32 v4, s30, v3
	v_lshrrev_b32_e32 v48, 1, v0
	v_mov_b32_e32 v47, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v6, 64, v5
	v_or_b32_e32 v15, s30, v48
	s_wait_kmcnt 0x0
	s_add_co_i32 s29, s25, s24
	s_ashr_i32 s4, s27, 31
	s_add_co_i32 s33, s29, s26
	v_mov_b32_e32 v10, s8
	s_add_co_i32 s7, s33, -1
	v_dual_mov_b32 v11, s9 :: v_dual_lshlrev_b32 v2, 4, v1
	v_min_i32_e32 v9, s7, v5
	s_add_co_i32 s35, s28, -1
	s_lshr_b32 s4, s4, 24
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_u32_u24 v8, 0x48, v3, v2
	v_or_b32_e32 v3, 64, v4
	v_cmp_gt_i32_e32 vcc_lo, s24, v9
	v_cmp_gt_i32_e64 s5, s29, v9
	v_cmp_gt_i32_e64 s2, s28, v4
	v_min_i32_e32 v4, s35, v4
	v_cmp_gt_i32_e64 s3, s28, v3
	v_cndmask_b32_e64 v12, s24, 0, vcc_lo
	v_min_i32_e32 v3, s35, v3
	s_add_co_i32 s6, s27, s4
	v_cmp_gt_i32_e64 s4, s33, v5
	v_lshlrev_b32_e32 v1, 3, v1
	v_cndmask_b32_e64 v5, s29, v12, s5
	v_min_i32_e32 v12, s7, v6
	v_mad_co_u64_u32 v[209:210], null, v4, s27, v[2:3]
	v_cndmask_b32_e32 v4, s11, v11, vcc_lo
	v_mad_co_u64_u32 v[210:211], null, v3, s27, v[2:3]
	v_sub_nc_u32_e32 v5, v9, v5
	v_cndmask_b32_e32 v2, s10, v10, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s24, v12
	v_or_b32_e32 v9, s31, v7
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s34, s6, 8
	v_cndmask_b32_e64 v3, s13, v4, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s8, s34, 0x88
	v_cmp_gt_i32_e64 s6, s29, v12
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v4, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, s24, 0, vcc_lo
	v_min_i32_e32 v13, s7, v9
	v_cndmask_b32_e64 v2, s12, v2, s5
	v_cndmask_b32_e32 v14, s11, v11, vcc_lo
	v_add_nc_u32_e32 v254, 0, v8
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v5, s29, v5, s6
	v_cmp_gt_i32_e64 s7, s24, v13
	v_add_co_u32 v2, s5, v2, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, s5
	v_sub_nc_u32_e32 v4, v12, v5
	v_cndmask_b32_e32 v5, s10, v10, vcc_lo
	v_cndmask_b32_e64 v12, s24, 0, s7
	v_cmp_gt_i32_e32 vcc_lo, s29, v13
	v_lshlrev_b32_e32 v7, 3, v7
	v_mul_lo_u32 v4, s8, v4
	v_cndmask_b32_e64 v5, s12, v5, s6
	v_cmp_gt_i32_e64 s5, s33, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, s29, v12, vcc_lo
	v_cndmask_b32_e64 v6, s13, v14, s6
	v_min_i32_e32 v14, s35, v15
	v_cndmask_b32_e64 v10, s10, v10, s7
	v_cndmask_b32_e64 v11, s11, v11, s7
	v_sub_nc_u32_e32 v12, v13, v12
	v_add_co_u32 v4, s6, v5, v4
	v_lshrrev_b32_e32 v13, 7, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, 0, v6, s6
	v_mul_lo_u32 v12, s8, v12
	v_mul_lo_u32 v6, v14, s34
	v_cndmask_b32_e32 v10, s12, v10, vcc_lo
	v_lshlrev_b32_e32 v14, 2, v48
	v_cmp_gt_i32_e64 s7, s33, v9
	v_dual_cndmask_b32 v9, s13, v11 :: v_dual_lshlrev_b32 v248, 4, v13
	v_lshl_or_b32 v7, v13, 2, v7
	v_add_co_u32 v101, vcc_lo, v10, v12
	v_cmp_gt_i32_e64 s6, s28, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v102, null, 0, v9, vcc_lo
	v_lshlrev_b32_e32 v103, 1, v6
	v_add_nc_u32_e32 v253, 0, v14
	v_add_nc_u32_e32 v252, 0, v7
	s_cmp_gt_i32 s27, 0xff
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s27, 0x100
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	v_add_co_u32 v15, vcc_lo, v2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, 0, v3, vcc_lo
	v_add_co_u32 v17, vcc_lo, v4, v1
	global_load_b32 v25, v[101:102], off
	s_clause 0x1
	global_load_b128 v[7:10], v209, s[14:15]
	global_load_b128 v[11:14], v210, s[14:15]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, 0, v5, vcc_lo
	v_dual_mov_b32 v20, 0 :: v_dual_lshlrev_b32 v19, 1, v6
	global_load_b64 v[21:22], v[15:16], off offset:8
	global_load_b64 v[23:24], v[17:18], off offset:8
	s_clause 0x1
	global_load_b128 v[133:136], v209, s[14:15] offset:64
	global_load_b128 v[129:132], v210, s[14:15] offset:64
	v_add_nc_u32_e32 v26, 0x1200, v254
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]
	v_add_nc_u32_e32 v27, 0x2400, v254
	s_mov_b32 s9, 0x4e4c4a48
	s_mov_b32 s10, 0x4040404
	v_add_nc_u32_e32 v28, 0x3600, v254
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v19, vcc_lo, s16, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, s17, v20, vcc_lo
	global_load_b32 v6, v[19:20], off
	global_load_b64 v[85:86], v[15:16], off offset:40
	global_load_b64 v[91:92], v[17:18], off offset:40
	s_wait_loadcnt 0x9
	v_lshrrev_b32_e32 v15, v248, v25
	s_wait_loadcnt 0x8
	v_cndmask_b32_e64 v8, 0, v8, s2
	s_wait_loadcnt 0x7
	v_cndmask_b32_e64 v12, 0, v12, s3
	v_cndmask_b32_e64 v11, 0, v11, s3
	v_cndmask_b32_e64 v14, 0, v14, s3
	v_cvt_f32_f16_e32 v15, v15.l
	s_wait_loadcnt 0x6
	v_cndmask_b32_e64 v16, 0, v21, s4
	v_cndmask_b32_e64 v17, 0, v22, s4
	s_wait_loadcnt 0x5
	v_cndmask_b32_e64 v18, 0, v23, s5
	v_cndmask_b32_e64 v19, 0, v24, s5
	s_wait_loadcnt 0x4
	v_dual_mov_b32 v87, v134 :: v_dual_mov_b32 v134, v136
	s_wait_loadcnt 0x3
	v_mov_b32_e32 v136, v130
	v_cndmask_b32_e64 v23, 0, v15, s7
	v_and_b32_e32 v15, 0xf0f0f0f, v16
	v_lshrrev_b32_e32 v16, 4, v16
	v_and_b32_e32 v20, 0xf0f0f0f, v17
	v_lshrrev_b32_e32 v17, 4, v17
	v_dual_mov_b32 v88, v132 :: v_dual_and_b32 v21, 0xf0f0f0f, v18
	v_lshrrev_b32_e32 v18, 4, v18
	v_lshrrev_b32_e32 v22, 4, v19
	v_and_b32_e32 v16, 0xf0f0f0f, v16
	v_and_b32_e32 v17, 0xf0f0f0f, v17
	v_and_b32_e32 v19, 0xf0f0f0f, v19
	v_and_b32_e32 v18, 0xf0f0f0f, v18
	v_and_b32_e32 v22, 0xf0f0f0f, v22
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v6, 0, v6, s6
	v_perm_b32 v24, v16, v15, 0x5010400
	v_perm_b32 v15, v16, v15, 0x7030602
	v_perm_b32 v16, v17, v20, 0x5010400
	v_perm_b32 v17, v17, v20, 0x7030602
	v_perm_b32 v20, v18, v21, 0x5010400
	v_perm_b32 v18, v18, v21, 0x7030602
	v_perm_b32 v21, v22, v19, 0x5010400
	v_perm_b32 v19, v22, v19, 0x7030602
	ds_store_b32 v253, v6 offset:18432
	v_and_b32_e32 v6, 0x7070707, v24
	v_lshrrev_b32_e32 v22, 1, v24
	v_and_b32_e32 v24, 0x7070707, v15
	v_lshrrev_b32_e32 v15, 1, v15
	v_and_b32_e32 v25, 0x7070707, v16
	v_lshrrev_b32_e32 v16, 1, v16
	v_and_b32_e32 v29, 0x7070707, v17
	v_lshrrev_b32_e32 v17, 1, v17
	v_and_b32_e32 v30, 0x7070707, v20
	v_lshrrev_b32_e32 v20, 1, v20
	v_and_b32_e32 v31, 0x7070707, v18
	v_lshrrev_b32_e32 v18, 1, v18
	v_and_b32_e32 v32, 0x7070707, v21
	v_lshrrev_b32_e32 v21, 1, v21
	v_and_b32_e32 v33, 0x7070707, v19
	v_lshrrev_b32_e32 v19, 1, v19
	s_wait_alu depctr_sa_sdst(0)
	v_perm_b32 v34, s9, 0x44403800, v6
	v_or_b32_e32 v6, 0x50505050, v6
	v_and_or_b32 v22, v22, s10, 0x3020100
	v_perm_b32 v35, s9, 0x44403800, v24
	v_or_b32_e32 v24, 0x50505050, v24
	v_and_or_b32 v36, v15, s10, 0x3020100
	v_perm_b32 v37, s9, 0x44403800, v25
	v_or_b32_e32 v25, 0x50505050, v25
	v_and_or_b32 v38, v16, s10, 0x3020100
	v_perm_b32 v39, s9, 0x44403800, v29
	v_or_b32_e32 v29, 0x50505050, v29
	v_and_or_b32 v40, v17, s10, 0x3020100
	v_perm_b32 v41, s9, 0x44403800, v30
	v_or_b32_e32 v30, 0x50505050, v30
	v_and_or_b32 v20, v20, s10, 0x3020100
	v_perm_b32 v42, s9, 0x44403800, v31
	v_or_b32_e32 v31, 0x50505050, v31
	v_and_or_b32 v43, v18, s10, 0x3020100
	v_perm_b32 v44, s9, 0x44403800, v32
	v_or_b32_e32 v32, 0x50505050, v32
	v_and_or_b32 v21, v21, s10, 0x3020100
	v_perm_b32 v45, s9, 0x44403800, v33
	v_or_b32_e32 v33, 0x50505050, v33
	v_and_or_b32 v46, v19, s10, 0x3020100
	v_cndmask_b32_e64 v13, 0, v13, s3
	v_perm_b32 v15, v6, v34, v22
	v_perm_b32 v16, v24, v35, v36
	v_perm_b32 v17, v25, v37, v38
	v_perm_b32 v18, v29, v39, v40
	v_cndmask_b32_e64 v7, 0, v7, s2
	v_cndmask_b32_e64 v10, 0, v10, s2
	v_cndmask_b32_e64 v9, 0, v9, s2
	v_perm_b32 v19, v30, v41, v20
	v_perm_b32 v20, v31, v42, v43
	v_perm_b32 v21, v32, v44, v21
	v_perm_b32 v22, v33, v45, v46
	ds_store_b32 v252, v23 offset:18944
	ds_store_2addr_b64 v26, v[11:12], v[13:14] offset1:1
	ds_store_2addr_b64 v27, v[15:16], v[17:18] offset1:1
	ds_store_2addr_b64 v254, v[7:8], v[9:10] offset1:1
	ds_store_2addr_b64 v28, v[19:20], v[21:22] offset1:1
	s_branch .LBB0_3
.LBB0_2:
	v_mov_b32_e32 v85, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v86, v85 :: v_dual_mov_b32 v91, v85
	v_dual_mov_b32 v92, v85 :: v_dual_mov_b32 v133, v85
	v_dual_mov_b32 v87, v85 :: v_dual_mov_b32 v134, v85
	v_dual_mov_b32 v135, v85 :: v_dual_mov_b32 v136, v85
	v_dual_mov_b32 v129, v85 :: v_dual_mov_b32 v88, v85
	v_mov_b32_e32 v131, v85
.LBB0_3:
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v37, 0 :: v_dual_and_b32 v6, 15, v0
	v_dual_mov_b32 v36, 0 :: v_dual_and_b32 v19, 0x60, v0
	v_dual_mov_b32 v39, 0 :: v_dual_and_b32 v20, 64, v48
	v_dual_mov_b32 v41, 0 :: v_dual_and_b32 v10, 8, v48
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v233, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v235, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v237, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v239, 0
	v_dual_mov_b32 v232, 0 :: v_dual_mov_b32 v241, 0
	v_dual_mov_b32 v234, 0 :: v_dual_mov_b32 v243, 0
	v_dual_mov_b32 v236, 0 :: v_dual_mov_b32 v245, 0
	v_dual_mov_b32 v238, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v240, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v242, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v244, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v246, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v173, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v174, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_15
; %bb.4:
	scratch_store_b32 off, v0, off offset:356 ; 4-byte Folded Spill
	v_dual_mov_b32 v104, 0 :: v_dual_add_nc_u32 v7, 0, v10
	v_and_b32_e32 v0, 0x6f, v0
	v_or_b32_e32 v8, v19, v10
	v_add_co_u32 v215, vcc_lo, v2, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v173, v104
	v_mad_u32_u24 v0, 0x48, v0, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v169, v104 :: v_dual_lshlrev_b32 v8, 3, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v216, null, 0, v3, vcc_lo
	v_add_co_u32 v217, vcc_lo, v4, v1
	v_add_nc_u32_e32 v0, v0, v10
	v_dual_mov_b32 v174, v104 :: v_dual_add_nc_u32 v1, 0, v8
	v_dual_mov_b32 v167, v104 :: v_dual_add_nc_u32 v2, 0x2400, v254
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v20, off offset:364
	scratch_store_b32 off, v6, off offset:368
	v_or_b32_e32 v6, v20, v6
	v_dual_mov_b32 v170, v104 :: v_dual_add_nc_u32 v3, 0x4a00, v1
	v_dual_mov_b32 v159, v104 :: v_dual_add_nc_u32 v152, 0x2400, v0
	v_dual_mov_b32 v245, v104 :: v_dual_add_nc_u32 v0, 0x1000, v254
	scratch_store_b32 off, v2, off          ; 4-byte Folded Spill
	v_dual_mov_b32 v163, v104 :: v_dual_add_nc_u32 v2, 0x3600, v254
	v_mul_u32_u24_e32 v9, 0x48, v6
	v_dual_mov_b32 v171, v104 :: v_dual_lshlrev_b32 v6, 2, v6
	s_clause 0x2                            ; 12-byte Folded Spill
	scratch_store_b32 off, v2, off offset:4
	scratch_store_b32 off, v19, off offset:360
	scratch_store_b32 off, v3, off offset:12
	v_dual_mov_b32 v168, v104 :: v_dual_add_nc_u32 v3, 0x4a08, v1
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v10, off offset:372
	scratch_store_b32 off, v0, off offset:8
	v_dual_mov_b32 v161, v104 :: v_dual_add_nc_u32 v2, 0, v6
	scratch_store_b32 off, v3, off offset:16 ; 4-byte Folded Spill
	v_dual_mov_b32 v164, v104 :: v_dual_add_nc_u32 v3, 0x4a10, v1
	v_dual_mov_b32 v243, v104 :: v_dual_add_nc_u32 v0, 0x4800, v2
	v_mov_b32_e32 v162, v104
	scratch_store_b32 off, v3, off offset:20 ; 4-byte Folded Spill
	v_dual_mov_b32 v160, v104 :: v_dual_add_nc_u32 v3, 0x4a18, v1
	v_mov_b32_e32 v158, v104
	v_mov_b32_e32 v246, v104
	v_mov_b32_e32 v244, v104
	scratch_store_b32 off, v3, off offset:24 ; 4-byte Folded Spill
	v_dual_mov_b32 v242, v104 :: v_dual_add_nc_u32 v3, 0x4a20, v1
	v_dual_mov_b32 v172, v104 :: v_dual_add_nc_u32 v151, v7, v9
	v_mov_b32_e32 v240, v104
	scratch_store_b32 off, v3, off offset:28 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a28, v1
	scratch_store_b32 off, v0, off offset:76 ; 4-byte Folded Spill
	v_dual_mov_b32 v45, v104 :: v_dual_mov_b32 v0, v209
	v_mov_b32_e32 v238, v104
	scratch_store_b32 off, v3, off offset:32 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a30, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v218, null, 0, v5, vcc_lo
	v_mov_b32_e32 v236, v104
	v_dual_mov_b32 v157, v104 :: v_dual_add_nc_u32 v228, 0x800, v151
	scratch_store_b32 off, v3, off offset:36 ; 4-byte Folded Spill
	v_dual_mov_b32 v234, v104 :: v_dual_add_nc_u32 v3, 0x4a38, v1
	v_dual_mov_b32 v241, v104 :: v_dual_mov_b32 v62, v104
	v_dual_mov_b32 v239, v104 :: v_dual_mov_b32 v64, v104
	scratch_store_b32 off, v3, off offset:40 ; 4-byte Folded Spill
	v_dual_mov_b32 v232, v104 :: v_dual_add_nc_u32 v3, 0x4a80, v1
	v_dual_mov_b32 v237, v104 :: v_dual_mov_b32 v60, v104
	v_dual_mov_b32 v235, v104 :: v_dual_mov_b32 v58, v104
	scratch_store_b32 off, v3, off offset:44 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a88, v1
	v_dual_mov_b32 v233, v104 :: v_dual_mov_b32 v56, v104
	v_mov_b32_e32 v154, v104
	v_mov_b32_e32 v150, v104
	scratch_store_b32 off, v3, off offset:48 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a90, v1
	v_dual_mov_b32 v153, v104 :: v_dual_mov_b32 v54, v104
	v_dual_mov_b32 v149, v104 :: v_dual_mov_b32 v52, v104
	scratch_store_b32 off, v3, off offset:52 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a98, v1
	v_dual_mov_b32 v63, v104 :: v_dual_mov_b32 v50, v104
	v_dual_mov_b32 v61, v104 :: v_dual_mov_b32 v48, v104
	scratch_store_b32 off, v3, off offset:56 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4aa0, v1
	v_dual_mov_b32 v59, v104 :: v_dual_mov_b32 v46, v104
	v_dual_mov_b32 v57, v104 :: v_dual_mov_b32 v44, v104
	scratch_store_b32 off, v3, off offset:60 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4aa8, v1
	v_dual_mov_b32 v55, v104 :: v_dual_mov_b32 v42, v104
	v_dual_mov_b32 v53, v104 :: v_dual_mov_b32 v40, v104
	scratch_store_b32 off, v3, off offset:64 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4ab0, v1
	v_add_nc_u32_e32 v1, 0x4ab8, v1
	v_dual_mov_b32 v51, v104 :: v_dual_mov_b32 v38, v104
	v_dual_mov_b32 v49, v104 :: v_dual_mov_b32 v36, v104
	s_clause 0x2                            ; 16-byte Folded Spill
	scratch_store_b32 off, v3, off offset:68
	scratch_store_b32 off, v1, off offset:72
	scratch_store_b64 off, v[0:1], off offset:300
	v_dual_mov_b32 v43, v104 :: v_dual_mov_b32 v0, v210
	v_mov_b32_e32 v41, v104
	v_mov_b32_e32 v39, v104
	v_mov_b32_e32 v37, v104
	v_mov_b32_e32 v47, v104
	s_mov_b32 s9, 0
	s_sub_co_i32 s12, 0, s34
	s_mov_b32 s13, 1
	s_movk_i32 s10, 0x100
	s_movk_i32 s27, 0x88
	s_mov_b32 s35, 0x4e4c4a48
	s_mov_b32 s36, 0x4040404
	s_clause 0x9                            ; 48-byte Folded Spill
	scratch_store_b64 off, v[0:1], off offset:308
	scratch_store_b64 off, v[101:102], off offset:316
	scratch_store_b32 off, v253, off offset:324
	scratch_store_b32 off, v248, off offset:328
	scratch_store_b32 off, v252, off offset:332
	scratch_store_b32 off, v254, off offset:336
	scratch_store_b32 off, v215, off offset:340
	scratch_store_b32 off, v216, off offset:344
	scratch_store_b32 off, v217, off offset:348
	scratch_store_b32 off, v218, off offset:352
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	v_dual_fmac_f32 v171, v179, v59 :: v_dual_fmac_f32 v172, v181, v60
	v_dual_fmac_f32 v169, v183, v61 :: v_dual_fmac_f32 v170, v185, v62
	v_dual_fmac_f32 v167, v187, v63 :: v_dual_fmac_f32 v168, v189, v64
	v_dual_fmac_f32 v245, v175, v49 :: v_dual_fmac_f32 v246, v177, v50
	v_dual_fmac_f32 v243, v179, v51 :: v_dual_fmac_f32 v244, v181, v52
	s_clause 0xa                            ; 56-byte Folded Reload
	scratch_load_b32 v62, off, off offset:184 th:TH_LOAD_LU
	scratch_load_b32 v63, off, off offset:212 th:TH_LOAD_LU
	scratch_load_b32 v60, off, off offset:176 th:TH_LOAD_LU
	scratch_load_b64 v[137:138], off, off offset:232 th:TH_LOAD_LU
	scratch_load_b32 v61, off, off offset:180 th:TH_LOAD_LU
	scratch_load_b32 v50, off, off offset:136 th:TH_LOAD_LU
	scratch_load_b32 v51, off, off offset:140 th:TH_LOAD_LU
	scratch_load_b32 v49, off, off offset:132 th:TH_LOAD_LU
	scratch_load_b32 v64, off, off offset:292 th:TH_LOAD_LU
	scratch_load_b64 v[139:140], off, off offset:248 th:TH_LOAD_LU
	scratch_load_b64 v[143:144], off, off offset:280 th:TH_LOAD_LU
	v_fmac_f32_e32 v149, v181, v44
	v_dual_fmac_f32 v173, v175, v57 :: v_dual_fmac_f32 v174, v177, v58
	v_dual_fmac_f32 v241, v183, v53 :: v_dual_fmac_f32 v242, v185, v54
	v_dual_fmac_f32 v239, v187, v55 :: v_dual_fmac_f32 v240, v189, v56
	v_dual_fmac_f32 v150, v175, v41 :: v_dual_fmac_f32 v153, v177, v42
	s_clause 0x9                            ; 40-byte Folded Reload
	scratch_load_b32 v59, off, off offset:172 th:TH_LOAD_LU
	scratch_load_b32 v58, off, off offset:168 th:TH_LOAD_LU
	scratch_load_b32 v57, off, off offset:164 th:TH_LOAD_LU
	scratch_load_b32 v56, off, off offset:160 th:TH_LOAD_LU
	scratch_load_b32 v55, off, off offset:156 th:TH_LOAD_LU
	scratch_load_b32 v54, off, off offset:152 th:TH_LOAD_LU
	scratch_load_b32 v53, off, off offset:148 th:TH_LOAD_LU
	scratch_load_b32 v52, off, off offset:144 th:TH_LOAD_LU
	scratch_load_b32 v42, off, off offset:108 th:TH_LOAD_LU
	scratch_load_b32 v41, off, off offset:104 th:TH_LOAD_LU
	v_dual_fmac_f32 v173, v176, v165 :: v_dual_fmac_f32 v244, v182, v166
	v_dual_fmac_f32 v239, v188, v166 :: v_dual_fmac_f32 v246, v178, v166
	v_fmac_f32_e32 v241, v184, v166
	s_add_co_i32 s13, s13, 1
	s_wait_dscnt 0x9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v244, v223, v116 :: v_dual_fmac_f32 v239, v217, v119
	v_dual_fmac_f32 v246, v227, v114 :: v_dual_fmac_f32 v241, v221, v117
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s12, s13
	s_addk_co_i32 s10, 0x100
	s_addk_co_i32 s27, 0x88
	v_dual_fmac_f32 v246, v228, v198 :: v_dual_fmac_f32 v241, v222, v198
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, 1
	scratch_load_b64 v[141:142], off, off offset:264 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_dual_fmac_f32 v245, v176, v166 :: v_dual_fmac_f32 v242, v186, v166
	v_fmac_f32_e32 v153, v178, v155
	v_fmac_f32_e32 v239, v218, v198
	v_dual_fmac_f32 v171, v180, v165 :: v_dual_fmac_f32 v168, v190, v165
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v245, v229, v113 :: v_dual_fmac_f32 v242, v219, v118
	v_fmac_f32_e32 v153, v227, v106
	s_wait_dscnt 0x8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v171, v225, v123 :: v_dual_fmac_f32 v168, v215, v128
	v_dual_fmac_f32 v243, v180, v166 :: v_dual_fmac_f32 v240, v190, v166
	v_dual_fmac_f32 v242, v220, v198 :: v_dual_fmac_f32 v149, v182, v155
	v_dual_fmac_f32 v172, v182, v165 :: v_dual_fmac_f32 v243, v225, v115
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v240, v215, v120 :: v_dual_fmac_f32 v169, v184, v165
	v_dual_fmac_f32 v170, v186, v165 :: v_dual_fmac_f32 v167, v188, v165
	v_dual_fmac_f32 v174, v178, v165 :: v_dual_fmac_f32 v149, v223, v108
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v173, v229, v121 :: v_dual_fmac_f32 v172, v223, v124
	v_dual_fmac_f32 v169, v221, v125 :: v_dual_fmac_f32 v170, v219, v126
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v167, v217, v127 :: v_dual_fmac_f32 v174, v227, v122
	v_fmac_f32_e32 v153, v228, v195
	v_fmac_f32_e32 v171, v226, v197
	v_fmac_f32_e32 v173, v230, v197
	v_fmac_f32_e32 v245, v230, v198
	v_fmac_f32_e32 v149, v224, v195
	v_dual_fmac_f32 v170, v220, v197 :: v_dual_fmac_f32 v167, v218, v197
	v_dual_fmac_f32 v174, v228, v197 :: v_dual_fmac_f32 v169, v222, v197
	s_wait_loadcnt 0x15
	v_dual_fmac_f32 v243, v226, v198 :: v_dual_fmac_f32 v62, v179, v43
	scratch_load_b32 v43, off, off offset:112 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x13
	v_dual_fmac_f32 v60, v187, v47 :: v_dual_fmac_f32 v163, v137, v26
	s_wait_loadcnt 0x11
	v_dual_fmac_f32 v61, v189, v48 :: v_dual_fmac_f32 v50, v175, v33
	s_wait_loadcnt 0x10
	v_fmac_f32_e32 v51, v177, v34
	s_clause 0x1                            ; 12-byte Folded Reload
	scratch_load_b32 v48, off, off offset:128 th:TH_LOAD_LU
	scratch_load_b64 v[33:34], off, off offset:224 th:TH_LOAD_LU
	s_wait_loadcnt 0xf
	v_fmac_f32_e32 v161, v139, v28
	v_fmac_f32_e32 v49, v181, v36
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v36, off, off offset:84 th:TH_LOAD_LU
	scratch_load_b32 v47, off, off offset:80 th:TH_LOAD_LU
	s_wait_loadcnt 0x10
	v_fmac_f32_e32 v157, v143, v32
	v_dual_fmac_f32 v237, v137, v18 :: v_dual_fmac_f32 v154, v143, v24
	v_dual_fmac_f32 v235, v139, v20 :: v_dual_fmac_f32 v150, v176, v155
	v_dual_fmac_f32 v61, v190, v155 :: v_dual_fmac_f32 v62, v180, v155
	v_fmac_f32_e32 v60, v188, v155
	s_wait_loadcnt 0xe
	v_dual_fmac_f32 v58, v137, v10 :: v_dual_fmac_f32 v161, v140, v165
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v56, v139, v12 :: v_dual_fmac_f32 v157, v144, v165
	v_dual_fmac_f32 v150, v229, v105 :: v_dual_fmac_f32 v61, v215, v112
	s_wait_loadcnt 0x8
	v_fmac_f32_e32 v52, v143, v16
	v_dual_fmac_f32 v62, v225, v107 :: v_dual_fmac_f32 v51, v178, v156
	v_dual_fmac_f32 v60, v217, v111 :: v_dual_fmac_f32 v49, v182, v156
	s_wait_dscnt 0x4
	v_fmac_f32_e32 v161, v207, v92
	v_fmac_f32_e32 v237, v138, v166
	v_fmac_f32_e32 v235, v140, v166
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v157, v199, v96 :: v_dual_fmac_f32 v62, v226, v195
	v_fmac_f32_e32 v61, v216, v195
	v_dual_fmac_f32 v50, v176, v156 :: v_dual_fmac_f32 v51, v227, v98
	v_dual_fmac_f32 v49, v223, v100 :: v_dual_fmac_f32 v150, v230, v195
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v157, v200, v197
	v_dual_fmac_f32 v50, v229, v97 :: v_dual_fmac_f32 v51, v228, v196
	s_wait_loadcnt 0x5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v49, v224, v196 :: v_dual_fmac_f32 v54, v141, v14
	v_dual_fmac_f32 v233, v141, v22 :: v_dual_fmac_f32 v50, v230, v196
	v_dual_mov_b32 v228, v0 :: v_dual_fmac_f32 v237, v211, v82
	v_fmac_f32_e32 v235, v207, v84
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v233, v142, v166
	v_dual_fmac_f32 v60, v218, v195 :: v_dual_fmac_f32 v161, v208, v197
	v_fmac_f32_e32 v237, v212, v198
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v235, v208, v198
	v_fmac_f32_e32 v233, v203, v86
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v233, v204, v198 :: v_dual_fmac_f32 v164, v33, v25
	scratch_load_b64 v[25:26], off, off offset:240 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v42, v33, v1
	v_fmac_f32_e32 v59, v33, v9
	s_wait_loadcnt 0x1
	v_dual_fmac_f32 v47, v143, v8 :: v_dual_fmac_f32 v164, v34, v165
	v_fmac_f32_e32 v52, v144, v155
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v154, v144, v166 :: v_dual_fmac_f32 v59, v34, v155
	v_fmac_f32_e32 v47, v144, v156
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v52, v199, v80
	v_dual_fmac_f32 v154, v199, v88 :: v_dual_fmac_f32 v59, v213, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v47, v199, v72 :: v_dual_fmac_f32 v52, v200, v195
	v_dual_fmac_f32 v154, v200, v198 :: v_dual_fmac_f32 v59, v214, v195
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v47, v200, v196 :: v_dual_fmac_f32 v162, v25, v27
	scratch_load_b64 v[27:28], off, off offset:256 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_dual_fmac_f32 v63, v183, v45 :: v_dual_fmac_f32 v64, v185, v46
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v45, off, off offset:120 th:TH_LOAD_LU
	scratch_load_b32 v46, off, off offset:124 th:TH_LOAD_LU
	scratch_load_b32 v44, off, off offset:116 th:TH_LOAD_LU
	v_fmac_f32_e32 v48, v179, v35
	v_fmac_f32_e32 v43, v187, v39
	v_fmac_f32_e32 v159, v141, v30
	scratch_load_b32 v39, off, off offset:96 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v236, v25, v19
	v_dual_fmac_f32 v63, v184, v155 :: v_dual_fmac_f32 v64, v186, v155
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v63, v221, v109 :: v_dual_fmac_f32 v64, v219, v110
	v_dual_fmac_f32 v172, v224, v197 :: v_dual_fmac_f32 v63, v222, v195
	s_wait_loadcnt 0x4
	v_fmac_f32_e32 v160, v27, v29
	scratch_load_b64 v[29:30], off, off offset:272 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v55, v27, v13
	s_wait_loadcnt 0x3
	v_dual_fmac_f32 v45, v183, v37 :: v_dual_fmac_f32 v46, v185, v38
	s_wait_loadcnt 0x2
	v_fmac_f32_e32 v44, v189, v40
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v40, off, off offset:100 th:TH_LOAD_LU
	scratch_load_b32 v38, off, off offset:92 th:TH_LOAD_LU
	scratch_load_b32 v37, off, off offset:88 th:TH_LOAD_LU
	v_fmac_f32_e32 v57, v25, v11
	v_dual_fmac_f32 v45, v184, v156 :: v_dual_fmac_f32 v46, v186, v156
	v_fmac_f32_e32 v48, v180, v156
	v_dual_fmac_f32 v234, v27, v21 :: v_dual_fmac_f32 v43, v188, v156
	v_fmac_f32_e32 v44, v190, v156
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v238, v33, v17 :: v_dual_fmac_f32 v39, v139, v4
	v_fmac_f32_e32 v41, v137, v2
	v_fmac_f32_e32 v45, v221, v101
	v_dual_fmac_f32 v48, v225, v99 :: v_dual_fmac_f32 v163, v138, v165
	v_dual_fmac_f32 v162, v26, v165 :: v_dual_fmac_f32 v55, v28, v155
	v_fmac_f32_e32 v58, v138, v155
	v_fmac_f32_e32 v44, v215, v104
	v_dual_fmac_f32 v160, v28, v165 :: v_dual_fmac_f32 v159, v142, v165
	v_fmac_f32_e32 v234, v28, v166
	v_dual_fmac_f32 v163, v211, v90 :: v_dual_fmac_f32 v162, v209, v91
	v_dual_fmac_f32 v56, v140, v155 :: v_dual_fmac_f32 v39, v140, v156
	v_fmac_f32_e32 v55, v205, v77
	v_fmac_f32_e32 v41, v138, v156
	v_fmac_f32_e32 v42, v34, v156
	v_dual_fmac_f32 v46, v219, v102 :: v_dual_fmac_f32 v43, v217, v103
	v_fmac_f32_e32 v160, v205, v93
	v_fmac_f32_e32 v238, v34, v166
	v_dual_fmac_f32 v164, v213, v89 :: v_dual_fmac_f32 v159, v203, v94
	v_dual_fmac_f32 v41, v211, v66 :: v_dual_fmac_f32 v42, v213, v65
	v_dual_fmac_f32 v39, v207, v68 :: v_dual_fmac_f32 v162, v210, v197
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v238, v213, v81
	v_dual_fmac_f32 v64, v220, v195 :: v_dual_fmac_f32 v43, v218, v196
	v_dual_fmac_f32 v240, v216, v198 :: v_dual_fmac_f32 v55, v206, v195
	v_dual_fmac_f32 v48, v226, v196 :: v_dual_fmac_f32 v163, v212, v197
	v_dual_fmac_f32 v244, v224, v198 :: v_dual_fmac_f32 v45, v222, v196
	v_fmac_f32_e32 v46, v220, v196
	v_dual_fmac_f32 v164, v214, v197 :: v_dual_fmac_f32 v41, v212, v196
	v_fmac_f32_e32 v238, v214, v198
	v_dual_fmac_f32 v42, v214, v196 :: v_dual_fmac_f32 v39, v208, v196
	s_wait_loadcnt 0x3
	v_dual_fmac_f32 v159, v204, v197 :: v_dual_fmac_f32 v232, v29, v23
	v_fmac_f32_e32 v158, v29, v31
	v_fmac_f32_e32 v53, v29, v15
	v_fmac_f32_e32 v36, v29, v7
	v_fmac_f32_e32 v234, v205, v85
	s_wait_loadcnt 0x2
	v_fmac_f32_e32 v40, v25, v3
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v38, v27, v5 :: v_dual_fmac_f32 v37, v141, v6
	v_dual_fmac_f32 v236, v26, v166 :: v_dual_fmac_f32 v57, v26, v155
	v_fmac_f32_e32 v232, v30, v166
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v54, v142, v155 :: v_dual_fmac_f32 v37, v142, v156
	v_fmac_f32_e32 v40, v26, v156
	v_fmac_f32_e32 v158, v30, v165
	v_fmac_f32_e32 v236, v209, v83
	v_fmac_f32_e32 v232, v201, v87
	v_dual_fmac_f32 v58, v211, v74 :: v_dual_fmac_f32 v57, v209, v75
	v_dual_fmac_f32 v37, v203, v70 :: v_dual_fmac_f32 v40, v209, v67
	v_fmac_f32_e32 v158, v201, v95
	v_dual_fmac_f32 v53, v30, v155 :: v_dual_fmac_f32 v168, v216, v197
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v57, v210, v195 :: v_dual_fmac_f32 v236, v210, v198
	v_fmac_f32_e32 v40, v210, v196
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v56, v207, v76 :: v_dual_fmac_f32 v53, v201, v79
	v_fmac_f32_e32 v44, v216, v196
	v_fmac_f32_e32 v38, v28, v156
	v_fmac_f32_e32 v36, v30, v156
	v_fmac_f32_e32 v54, v203, v78
	v_fmac_f32_e32 v58, v212, v195
	v_fmac_f32_e32 v56, v208, v195
	v_fmac_f32_e32 v38, v205, v69
	v_fmac_f32_e32 v36, v201, v71
	v_dual_fmac_f32 v160, v206, v197 :: v_dual_fmac_f32 v37, v204, v196
	v_fmac_f32_e32 v234, v206, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v38, v206, v196
	v_fmac_f32_e32 v54, v204, v195
	v_fmac_f32_e32 v158, v202, v197
	v_dual_fmac_f32 v232, v202, v198 :: v_dual_fmac_f32 v53, v202, v195
	v_fmac_f32_e32 v36, v202, v196
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b64 v[103:104], off, off offset:188 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x0
	s_clause 0x1                            ; 12-byte Folded Reload
	scratch_load_b32 v103, off, off offset:288 th:TH_LOAD_LU
	scratch_load_b64 v[209:210], off, off offset:300
	s_wait_loadcnt 0x0
	s_clause 0x9                            ; 56-byte Folded Reload
	scratch_load_b64 v[210:211], off, off offset:308
	scratch_load_b64 v[101:102], off, off offset:316
	scratch_load_b64 v[91:92], off, off offset:196
	scratch_load_b64 v[85:86], off, off offset:204
	scratch_load_b32 v215, off, off offset:340
	scratch_load_b32 v216, off, off offset:344
	scratch_load_b32 v217, off, off offset:348
	scratch_load_b32 v218, off, off offset:352
	scratch_load_b32 v87, off, off offset:216
	scratch_load_b32 v88, off, off offset:220
	s_cbranch_scc1 .LBB0_14
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s10, 0xffffff00
	v_lshlrev_b64_e32 v[1:2], 2, v[103:104]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[14:15], s[8:9]
	s_clause 0x1c                           ; 116-byte Folded Spill
	scratch_store_b32 off, v64, off offset:292
	scratch_store_b32 off, v63, off offset:212
	scratch_store_b32 off, v62, off offset:184
	scratch_store_b32 off, v61, off offset:180
	scratch_store_b32 off, v60, off offset:176
	scratch_store_b32 off, v59, off offset:172
	scratch_store_b32 off, v58, off offset:168
	scratch_store_b32 off, v57, off offset:164
	scratch_store_b32 off, v56, off offset:160
	scratch_store_b32 off, v55, off offset:156
	scratch_store_b32 off, v54, off offset:152
	scratch_store_b32 off, v53, off offset:148
	scratch_store_b32 off, v52, off offset:144
	scratch_store_b32 off, v51, off offset:140
	scratch_store_b32 off, v50, off offset:136
	scratch_store_b32 off, v49, off offset:132
	scratch_store_b32 off, v48, off offset:128
	scratch_store_b32 off, v46, off offset:124
	scratch_store_b32 off, v45, off offset:120
	scratch_store_b32 off, v44, off offset:116
	scratch_store_b32 off, v43, off offset:112
	scratch_store_b32 off, v42, off offset:108
	scratch_store_b32 off, v41, off offset:104
	scratch_store_b32 off, v40, off offset:100
	scratch_store_b32 off, v39, off offset:96
	scratch_store_b32 off, v38, off offset:92
	scratch_store_b32 off, v37, off offset:88
	scratch_store_b32 off, v36, off offset:84
	scratch_store_b32 off, v47, off offset:80
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v77, s8, s38, v209
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v78, null, s39, 0, s8
	s_wait_loadcnt 0x9
	v_add_co_u32 v79, s8, s38, v210
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v80, null, s39, 0, s8
	s_add_co_i32 s8, s27, 0xffffff78
	s_wait_loadcnt 0x5
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v81, vcc_lo, v215, s8
	s_wait_loadcnt 0x4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, 0, v216, vcc_lo
	s_wait_loadcnt 0x3
	v_add_co_u32 v83, vcc_lo, v217, s8
	s_wait_loadcnt 0x2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, 0, v218, vcc_lo
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	v_add_co_u32 v3, vcc_lo, v101, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v102, vcc_lo
	s_clause 0x1
	global_load_b128 v[141:144], v[77:78], off offset:128
	global_load_b128 v[137:140], v[79:80], off offset:128
	global_load_b64 v[193:194], v[81:82], off offset:72
	global_load_b64 v[191:192], v[83:84], off offset:72
	global_load_b32 v255, v[3:4], off offset:4
	global_load_b32 v105, v[1:2], off offset:4
	ds_load_2addr_b64 v[65:68], v152 offset1:144
	ds_load_2addr_b64 v[1:4], v151 offset1:144
	ds_load_2addr_b64 v[69:72], v228 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[1:2], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[67:68], v[1:2], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[3:4], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[67:68], v[3:4], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[69:70], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[67:68], v[69:70], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[71:72], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[67:68], v[71:72], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[65:68], v152 offset0:2 offset1:146
	ds_load_2addr_b64 v[69:72], v151 offset0:2 offset1:146
	ds_load_2addr_b64 v[73:76], v228 offset0:34 offset1:178
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[69:70], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[67:68], v[69:70], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[71:72], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[67:68], v[71:72], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[73:74], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[67:68], v[73:74], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[75:76], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[67:68], v[75:76], v[1:8]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[65:68], v152 offset0:4 offset1:148
	ds_load_2addr_b64 v[69:72], v151 offset0:4 offset1:148
	ds_load_2addr_b64 v[73:76], v228 offset0:36 offset1:180
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[69:70], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[67:68], v[69:70], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[71:72], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[67:68], v[71:72], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[73:74], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[67:68], v[73:74], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[75:76], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[67:68], v[75:76], v[1:8]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[65:68], v152 offset0:6 offset1:150
	ds_load_2addr_b64 v[69:72], v151 offset0:6 offset1:150
	ds_load_2addr_b64 v[73:76], v228 offset0:38 offset1:182
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[69:70], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[67:68], v[69:70], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[71:72], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[67:68], v[71:72], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[73:74], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[67:68], v[73:74], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[75:76], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[67:68], v[75:76], v[1:8]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x7
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x6
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v65, 0, v85, s4
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v74, 0, v86, s4
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v89, 0, v91, s5
	v_lshrrev_b32_e32 v66, 4, v65
	v_and_b32_e32 v71, 0xf0f0f0f, v65
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_lshrrev_b32_e32 v91, 4, v89
	v_and_b32_e32 v72, 0xf0f0f0f, v66
	v_cndmask_b32_e64 v66, 0, v87, s2
	s_barrier_signal -1
	v_cndmask_b32_e64 v65, 0, v133, s2
	v_cndmask_b32_e64 v68, 0, v134, s2
	v_perm_b32 v73, v72, v71, 0x5010400
	v_perm_b32 v76, v72, v71, 0x7030602
	v_lshrrev_b32_e32 v71, 4, v74
	v_and_b32_e32 v74, 0xf0f0f0f, v74
	v_cndmask_b32_e64 v67, 0, v135, s2
	v_and_b32_e32 v75, 0x7070707, v73
	v_lshrrev_b32_e32 v72, 1, v73
	v_and_b32_e32 v85, 0xf0f0f0f, v71
	v_and_b32_e32 v87, 0x7070707, v76
	v_cndmask_b32_e64 v70, 0, v136, s3
	v_perm_b32 v73, s35, 0x44403800, v75
	v_or_b32_e32 v75, 0x50505050, v75
	v_and_or_b32 v86, v72, s36, 0x3020100
	v_cndmask_b32_e64 v72, 0, v88, s3
	v_perm_b32 v88, v85, v74, 0x5010400
	v_perm_b32 v85, v85, v74, 0x7030602
	s_barrier_wait -1
	v_perm_b32 v73, v75, v73, v86
	v_lshrrev_b32_e32 v75, 1, v76
	v_perm_b32 v76, s35, 0x44403800, v87
	v_or_b32_e32 v86, 0x50505050, v87
	v_and_b32_e32 v87, 0x7070707, v88
	v_lshrrev_b32_e32 v88, 1, v88
	v_and_or_b32 v75, v75, s36, 0x3020100
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off offset:8  ; 4-byte Folded Reload
	v_perm_b32 v90, s35, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v88, v88, s36, 0x3020100
	v_perm_b32 v74, v86, v76, v75
	v_and_b32_e32 v76, 0xf0f0f0f, v89
	v_and_b32_e32 v86, 0xf0f0f0f, v91
	v_and_b32_e32 v89, 0x7070707, v85
	v_perm_b32 v75, v87, v90, v88
	v_cndmask_b32_e64 v87, 0, v92, s5
	v_lshrrev_b32_e32 v85, 1, v85
	v_perm_b32 v88, v86, v76, 0x5010400
	v_perm_b32 v76, v86, v76, 0x7030602
	v_perm_b32 v90, s35, 0x44403800, v89
	v_lshrrev_b32_e32 v91, 4, v87
	v_and_b32_e32 v86, 0xf0f0f0f, v87
	v_and_b32_e32 v92, 0x7070707, v88
	v_lshrrev_b32_e32 v88, 1, v88
	v_and_b32_e32 v93, 0x7070707, v76
	v_and_b32_e32 v87, 0xf0f0f0f, v91
	v_lshrrev_b32_e32 v76, 1, v76
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v85, v85, s36, 0x3020100
	v_perm_b32 v91, s35, 0x44403800, v92
	v_perm_b32 v94, v87, v86, 0x5010400
	v_or_b32_e32 v92, 0x50505050, v92
	v_and_or_b32 v88, v88, s36, 0x3020100
	v_and_or_b32 v96, v76, s36, 0x3020100
	v_perm_b32 v86, v87, v86, 0x7030602
	v_lshrrev_b32_e32 v76, 1, v94
	v_and_b32_e32 v87, 0x7070707, v94
	v_perm_b32 v95, s35, 0x44403800, v93
	v_or_b32_e32 v93, 0x50505050, v93
	v_and_b32_e32 v94, 0x7070707, v86
	v_and_or_b32 v98, v76, s36, 0x3020100
	v_perm_b32 v76, v89, v90, v85
	v_perm_b32 v85, v92, v91, v88
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v90, off, off
	scratch_load_b32 v91, off, off offset:4
	v_lshrrev_b32_e32 v86, 1, v86
	v_perm_b32 v97, s35, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_perm_b32 v99, s35, 0x44403800, v94
	v_or_b32_e32 v94, 0x50505050, v94
	v_and_or_b32 v100, v86, s36, 0x3020100
	v_cndmask_b32_e64 v69, 0, v129, s3
	v_cndmask_b32_e64 v71, 0, v131, s3
	v_perm_b32 v86, v93, v95, v96
	v_perm_b32 v87, v87, v97, v98
	v_perm_b32 v88, v94, v99, v100
	ds_store_2addr_b64 v254, v[65:66], v[67:68] offset1:1
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v0, v[69:70], v[71:72] offset0:64 offset1:65
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v90, v[73:74], v[75:76] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v91, v[85:86], v[87:88] offset1:1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_clause 0x1
	global_load_b128 v[133:136], v[77:78], off offset:192
	global_load_b128 v[129:132], v[79:80], off offset:192
	global_load_b64 v[213:214], v[81:82], off offset:104
	global_load_b64 v[211:212], v[83:84], off offset:104
	ds_load_2addr_b64 v[65:68], v152 offset1:144
	ds_load_2addr_b64 v[69:72], v151 offset1:144
	ds_load_2addr_b64 v[73:76], v228 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[69:70], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[67:68], v[69:70], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[71:72], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[67:68], v[71:72], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[73:74], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[67:68], v[73:74], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[75:76], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[67:68], v[75:76], v[1:8]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[65:68], v152 offset0:2 offset1:146
	ds_load_2addr_b64 v[69:72], v151 offset0:2 offset1:146
	ds_load_2addr_b64 v[73:76], v228 offset0:34 offset1:178
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[69:70], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[67:68], v[69:70], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[71:72], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[67:68], v[71:72], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[73:74], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[67:68], v[73:74], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[75:76], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[67:68], v[75:76], v[1:8]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[65:68], v152 offset0:4 offset1:148
	ds_load_2addr_b64 v[69:72], v151 offset0:4 offset1:148
	ds_load_2addr_b64 v[73:76], v228 offset0:36 offset1:180
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[69:70], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[67:68], v[69:70], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[71:72], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[67:68], v[71:72], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[73:74], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[67:68], v[73:74], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[75:76], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[67:68], v[75:76], v[1:8]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[65:68], v152 offset0:6 offset1:150
	ds_load_2addr_b64 v[69:72], v151 offset0:6 offset1:150
	ds_load_2addr_b64 v[73:76], v228 offset0:38 offset1:182
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[69:70], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[67:68], v[69:70], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[71:72], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[67:68], v[71:72], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[73:74], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[67:68], v[73:74], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[75:76], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[67:68], v[75:76], v[1:8]
	; sched_barrier mask(0x00000000)
	scratch_load_b32 v65, off, off offset:76 ; 4-byte Folded Reload
	v_dual_mov_b32 v67, v138 :: v_dual_mov_b32 v68, v140
	s_cmp_lt_i32 s13, s34
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[165:166], v65 offset1:16
	ds_load_2addr_b32 v[155:156], v65 offset0:32 offset1:48
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v65, off, off offset:12
	scratch_load_b32 v69, off, off offset:60
	s_cselect_b32 s8, -1, 0
	s_cmp_ge_i32 s13, s34
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[175:176], v65 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[69:70], v69 offset1:1
	scratch_load_b32 v65, off, off offset:16 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[69:70], off offset:256 ; 8-byte Folded Spill
	scratch_load_b32 v69, off, off offset:64 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[177:178], v65 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[69:70], v69 offset1:1
	scratch_load_b32 v65, off, off offset:20 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[69:70], off offset:264 ; 8-byte Folded Spill
	scratch_load_b32 v69, off, off offset:68 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[179:180], v65 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[69:70], v69 offset1:1
	scratch_load_b32 v65, off, off offset:24 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[69:70], off offset:272 ; 8-byte Folded Spill
	scratch_load_b32 v69, off, off offset:72 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[181:182], v65 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[69:70], v69 offset1:1
	scratch_load_b32 v65, off, off offset:28 ; 4-byte Folded Reload
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[69:70], off offset:280 ; 8-byte Folded Spill
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[183:184], v65 offset1:1
	scratch_load_b32 v65, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[185:186], v65 offset1:1
	scratch_load_b32 v65, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[187:188], v65 offset1:1
	scratch_load_b32 v65, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[189:190], v65 offset1:1
	scratch_load_b32 v65, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:224 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:232 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:240 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:248 ; 8-byte Folded Spill
	v_dual_mov_b32 v65, v142 :: v_dual_mov_b32 v66, v144
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
	v_cndmask_b32_e64 v71, 0, v193, s4
	v_cndmask_b32_e64 v70, 0, v65, s2
	v_cndmask_b32_e64 v69, 0, v141, s2
	v_cndmask_b32_e64 v72, 0, v66, s2
	v_lshrrev_b32_e32 v88, v248, v255
	v_lshrrev_b32_e32 v73, 4, v71
	v_and_b32_e32 v74, 0xf0f0f0f, v71
	v_cndmask_b32_e64 v71, 0, v143, s2
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v73, 0xf0f0f0f, v73
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v254, v[69:70], v[71:72] offset1:1
	v_cndmask_b32_e64 v71, 0, v194, s4
	v_perm_b32 v75, v73, v74, 0x5010400
	v_perm_b32 v73, v73, v74, 0x7030602
	v_cndmask_b32_e64 v70, 0, v67, s3
	v_cndmask_b32_e64 v69, 0, v137, s3
	v_lshrrev_b32_e32 v77, 4, v71
	v_and_b32_e32 v76, 0x7070707, v75
	v_lshrrev_b32_e32 v74, 1, v75
	v_and_b32_e32 v78, 0xf0f0f0f, v71
	v_and_b32_e32 v79, 0x7070707, v73
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_perm_b32 v75, s35, 0x44403800, v76
	v_or_b32_e32 v76, 0x50505050, v76
	v_and_or_b32 v74, v74, s36, 0x3020100
	v_lshrrev_b32_e32 v81, 1, v73
	v_perm_b32 v80, v77, v78, 0x5010400
	v_perm_b32 v77, v77, v78, 0x7030602
	v_cndmask_b32_e64 v72, 0, v68, s3
	v_perm_b32 v73, v76, v75, v74
	v_perm_b32 v74, s35, 0x44403800, v79
	v_and_b32_e32 v76, 0x7070707, v80
	v_lshrrev_b32_e32 v80, 1, v80
	v_or_b32_e32 v75, 0x50505050, v79
	v_and_or_b32 v79, v81, s36, 0x3020100
	v_cndmask_b32_e64 v81, 0, v191, s5
	v_perm_b32 v78, s35, 0x44403800, v76
	v_or_b32_e32 v76, 0x50505050, v76
	v_and_or_b32 v80, v80, s36, 0x3020100
	v_perm_b32 v74, v75, v74, v79
	v_lshrrev_b32_e32 v82, 4, v81
	v_and_b32_e32 v79, 0x7070707, v77
	v_and_b32_e32 v81, 0xf0f0f0f, v81
	v_perm_b32 v75, v76, v78, v80
	v_cndmask_b32_e64 v76, 0, v192, s5
	v_and_b32_e32 v82, 0xf0f0f0f, v82
	v_lshrrev_b32_e32 v77, 1, v77
	v_perm_b32 v78, s35, 0x44403800, v79
	v_or_b32_e32 v79, 0x50505050, v79
	v_lshrrev_b32_e32 v83, 4, v76
	v_perm_b32 v80, v82, v81, 0x5010400
	v_and_or_b32 v77, v77, s36, 0x3020100
	v_and_b32_e32 v84, 0xf0f0f0f, v76
	v_perm_b32 v81, v82, v81, 0x7030602
	v_and_b32_e32 v83, 0xf0f0f0f, v83
	v_and_b32_e32 v82, 0x7070707, v80
	v_perm_b32 v76, v79, v78, v77
	v_lshrrev_b32_e32 v77, 1, v80
	v_and_b32_e32 v79, 0x7070707, v81
	v_perm_b32 v80, v83, v84, 0x5010400
	v_perm_b32 v83, v83, v84, 0x7030602
	v_lshrrev_b32_e32 v81, 1, v81
	v_perm_b32 v78, s35, 0x44403800, v82
	v_or_b32_e32 v82, 0x50505050, v82
	v_and_b32_e32 v86, 0x7070707, v80
	v_lshrrev_b32_e32 v80, 1, v80
	v_and_b32_e32 v87, 0x7070707, v83
	v_lshrrev_b32_e32 v83, 1, v83
	v_and_or_b32 v77, v77, s36, 0x3020100
	v_perm_b32 v85, s35, 0x44403800, v79
	v_or_b32_e32 v79, 0x50505050, v79
	v_and_or_b32 v81, v81, s36, 0x3020100
	v_perm_b32 v84, s35, 0x44403800, v86
	v_or_b32_e32 v86, 0x50505050, v86
	v_and_or_b32 v80, v80, s36, 0x3020100
	v_perm_b32 v89, s35, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v83, v83, s36, 0x3020100
	v_perm_b32 v77, v82, v78, v77
	v_cvt_f32_f16_e32 v82, v88.l
	v_cndmask_b32_e64 v71, 0, v139, s3
	v_perm_b32 v78, v79, v85, v81
	v_perm_b32 v79, v86, v84, v80
	v_perm_b32 v80, v87, v89, v83
	v_cndmask_b32_e64 v81, 0, v105, s6
	v_cndmask_b32_e64 v82, 0, v82, s7
	ds_store_2addr_b64 v0, v[69:70], v[71:72] offset0:64 offset1:65
	ds_store_2addr_b64 v90, v[73:74], v[75:76] offset1:1
	ds_store_2addr_b64 v91, v[77:78], v[79:80] offset1:1
	ds_store_b32 v253, v81 offset:18432
	ds_store_b32 v252, v82 offset:18944
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v0, 2, v103
	scratch_store_b32 off, v0, off offset:288 ; 4-byte Folded Spill
	s_barrier_wait -1
	s_wait_storecnt 0x0
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_6 Depth=1
	v_add_co_u32 v69, vcc_lo, v215, s27
	v_add_nc_u32_e32 v103, 2, v103
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, 0, v216, vcc_lo
	v_mov_b32_e32 v71, v104
	s_mov_b32 s11, s9
	v_add_co_u32 v73, vcc_lo, v217, s27
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[14:15], s[10:11]
	scratch_store_b64 off, v[70:71], off offset:188 ; 8-byte Folded Spill
	v_lshlrev_b64_e32 v[71:72], 2, v[103:104]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, 0, v218, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v65, s11, s38, v209
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v66, null, s39, 0, s11
	v_add_co_u32 v71, vcc_lo, s16, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, s17, v72, vcc_lo
	v_add_co_u32 v75, vcc_lo, v101, s27
	v_add_co_u32 v67, s11, s38, v210
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, 0, v102, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v68, null, s39, 0, s11
	s_clause 0x1
	global_load_b128 v[141:144], v[65:66], off
	global_load_b128 v[137:140], v[67:68], off
	global_load_b64 v[193:194], v[69:70], off offset:8
	global_load_b64 v[191:192], v[73:74], off offset:8
	global_load_b32 v0, v[71:72], off
	global_load_b32 v255, v[75:76], off
	s_wait_loadcnt 0x1
	scratch_store_b32 off, v0, off offset:296 ; 4-byte Folded Spill
	s_branch .LBB0_9
.LBB0_8:                                ;   in Loop: Header=BB0_6 Depth=1
	s_clause 0x1                            ; 12-byte Folded Spill
	scratch_store_b32 off, v105, off offset:296
	scratch_store_b64 off, v[103:104], off offset:188
	v_mov_b32_e32 v142, v65
	v_mov_b32_e32 v144, v66
	v_mov_b32_e32 v138, v67
	v_mov_b32_e32 v140, v68
.LBB0_9:                                ;   in Loop: Header=BB0_6 Depth=1
	ds_load_2addr_b64 v[195:198], v152 offset1:144
	ds_load_2addr_b64 v[65:68], v151 offset1:144
	ds_load_2addr_b64 v[199:202], v228 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[195:196], v[65:66], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[197:198], v[65:66], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[195:196], v[67:68], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[197:198], v[67:68], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[195:196], v[199:200], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[197:198], v[199:200], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[195:196], v[201:202], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[197:198], v[201:202], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[195:198], v152 offset0:2 offset1:146
	ds_load_2addr_b64 v[199:202], v151 offset0:2 offset1:146
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[195:196], v[199:200], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[197:198], v[199:200], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[195:196], v[201:202], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[197:198], v[201:202], v[81:88]
	ds_load_2addr_b64 v[199:202], v228 offset0:34 offset1:178
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[195:196], v[199:200], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[197:198], v[199:200], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[195:196], v[201:202], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[197:198], v[201:202], v[65:72]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[195:198], v152 offset0:4 offset1:148
	ds_load_2addr_b64 v[199:202], v151 offset0:4 offset1:148
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[195:196], v[199:200], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[197:198], v[199:200], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[195:196], v[201:202], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[197:198], v[201:202], v[81:88]
	ds_load_2addr_b64 v[199:202], v228 offset0:36 offset1:180
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[195:196], v[199:200], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[197:198], v[199:200], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[195:196], v[201:202], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[197:198], v[201:202], v[65:72]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[195:198], v152 offset0:6 offset1:150
	ds_load_2addr_b64 v[199:202], v151 offset0:6 offset1:150
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[195:196], v[199:200], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[197:198], v[199:200], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[195:196], v[201:202], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[197:198], v[201:202], v[81:88]
	ds_load_2addr_b64 v[199:202], v228 offset0:38 offset1:182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[195:196], v[199:200], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[197:198], v[199:200], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[195:196], v[201:202], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[197:198], v[201:202], v[65:72]
	; sched_barrier mask(0x00000000)
	v_dual_mov_b32 v219, v134 :: v_dual_mov_b32 v134, v136
	v_mov_b32_e32 v136, v130
	v_mov_b32_e32 v220, v132
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
	v_cndmask_b32_e64 v130, 0, v213, s4
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v147, 0, v214, s4
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v132, 4, v130
	v_and_b32_e32 v130, 0xf0f0f0f, v130
	v_lshrrev_b32_e32 v195, 4, v147
	v_and_b32_e32 v147, 0xf0f0f0f, v147
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_and_b32_e32 v132, 0xf0f0f0f, v132
	s_barrier_signal -1
	v_and_b32_e32 v196, 0xf0f0f0f, v195
	v_cndmask_b32_e64 v146, 0, v219, s2
	v_cndmask_b32_e64 v145, 0, v133, s2
	v_perm_b32 v148, v132, v130, 0x5010400
	v_perm_b32 v130, v132, v130, 0x7030602
	v_perm_b32 v199, v196, v147, 0x5010400
	v_perm_b32 v147, v196, v147, 0x7030602
	v_cndmask_b32_e64 v206, 0, v220, s3
	v_and_b32_e32 v132, 0x7070707, v148
	v_lshrrev_b32_e32 v148, 1, v148
	v_and_b32_e32 v197, 0x7070707, v130
	v_lshrrev_b32_e32 v130, 1, v130
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	v_perm_b32 v195, s35, 0x44403800, v132
	v_or_b32_e32 v132, 0x50505050, v132
	v_and_or_b32 v198, v148, s36, 0x3020100
	v_perm_b32 v200, s35, 0x44403800, v197
	v_and_or_b32 v130, v130, s36, 0x3020100
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_perm_b32 v195, v132, v195, v198
	v_or_b32_e32 v132, 0x50505050, v197
	v_and_b32_e32 v197, 0x7070707, v199
	v_cndmask_b32_e64 v198, 0, v211, s5
	v_lshrrev_b32_e32 v199, 1, v199
	scratch_load_b32 v0, off, off offset:8  ; 4-byte Folded Reload
	v_perm_b32 v196, v132, v200, v130
	v_perm_b32 v201, s35, 0x44403800, v197
	v_lshrrev_b32_e32 v202, 4, v198
	v_or_b32_e32 v197, 0x50505050, v197
	v_and_or_b32 v199, v199, s36, 0x3020100
	v_and_b32_e32 v130, 0xf0f0f0f, v198
	v_and_b32_e32 v198, 0x7070707, v147
	v_and_b32_e32 v132, 0xf0f0f0f, v202
	v_lshrrev_b32_e32 v147, 1, v147
	v_perm_b32 v197, v197, v201, v199
	v_cndmask_b32_e64 v199, 0, v212, s5
	v_perm_b32 v201, s35, 0x44403800, v198
	v_perm_b32 v200, v132, v130, 0x5010400
	v_or_b32_e32 v198, 0x50505050, v198
	v_and_or_b32 v147, v147, s36, 0x3020100
	v_lshrrev_b32_e32 v202, 4, v199
	v_perm_b32 v130, v132, v130, 0x7030602
	v_and_b32_e32 v203, 0x7070707, v200
	v_and_b32_e32 v132, 0xf0f0f0f, v199
	v_lshrrev_b32_e32 v200, 1, v200
	v_and_b32_e32 v199, 0xf0f0f0f, v202
	v_perm_b32 v198, v198, v201, v147
	v_perm_b32 v202, s35, 0x44403800, v203
	v_and_b32_e32 v147, 0x7070707, v130
	v_or_b32_e32 v201, 0x50505050, v203
	v_perm_b32 v203, v199, v132, 0x5010400
	v_lshrrev_b32_e32 v130, 1, v130
	v_and_or_b32 v200, v200, s36, 0x3020100
	v_perm_b32 v204, s35, 0x44403800, v147
	v_perm_b32 v132, v199, v132, 0x7030602
	v_or_b32_e32 v147, 0x50505050, v147
	v_and_or_b32 v130, v130, s36, 0x3020100
	v_and_b32_e32 v205, 0x7070707, v203
	v_lshrrev_b32_e32 v203, 1, v203
	v_perm_b32 v199, v201, v202, v200
	v_and_b32_e32 v201, 0x7070707, v132
	v_perm_b32 v200, v147, v204, v130
	v_perm_b32 v130, s35, 0x44403800, v205
	v_or_b32_e32 v202, 0x50505050, v205
	v_and_or_b32 v203, v203, s36, 0x3020100
	v_perm_b32 v207, s35, 0x44403800, v201
	v_or_b32_e32 v208, 0x50505050, v201
	v_lshrrev_b32_e32 v132, 1, v132
	v_cndmask_b32_e64 v148, 0, v134, s2
	v_perm_b32 v201, v202, v130, v203
	scratch_load_b32 v130, off, off         ; 4-byte Folded Reload
	v_cndmask_b32_e64 v147, 0, v135, s2
	v_and_or_b32 v132, v132, s36, 0x3020100
	v_cndmask_b32_e64 v204, 0, v136, s3
	v_cndmask_b32_e64 v203, 0, v129, s3
	v_cndmask_b32_e64 v205, 0, v131, s3
	ds_store_2addr_b64 v254, v[145:146], v[147:148] offset1:1
	v_perm_b32 v202, v208, v207, v132
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v0, v[203:204], v[205:206] offset0:64 offset1:65
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v130, v[195:196], v[197:198] offset1:1
	scratch_load_b32 v130, off, off offset:4 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v195, 0, 1, s8
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v130, v[199:200], v[201:202] offset1:1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_co_u32 v145, vcc_lo, v215, s27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v146, null, 0, v216, vcc_lo
	s_mov_b32 s11, s9
	v_add_co_u32 v147, vcc_lo, v217, s27
	global_load_b64 v[145:146], v[145:146], off offset:40
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[14:15], s[10:11]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v148, null, 0, v218, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v129, s8, s38, v209
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, s39, 0, s8
	v_add_co_u32 v131, s8, s38, v210
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v132, null, s39, 0, s8
	s_clause 0x1
	global_load_b128 v[133:136], v[129:130], off offset:64
	global_load_b128 v[129:132], v[131:132], off offset:64
	s_wait_loadcnt 0x2
	scratch_store_b64 off, v[145:146], off offset:204 ; 8-byte Folded Spill
	global_load_b64 v[145:146], v[147:148], off offset:40
	s_wait_loadcnt 0x2
	v_mov_b32_e32 v0, v134
	v_mov_b32_e32 v134, v136
	s_wait_loadcnt 0x1
	v_mov_b32_e32 v136, v130
	scratch_store_b32 off, v0, off offset:216 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, v132
	scratch_store_b32 off, v0, off offset:220 ; 4-byte Folded Spill
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[145:146], off offset:196 ; 8-byte Folded Spill
	s_branch .LBB0_12
.LBB0_11:                               ;   in Loop: Header=BB0_6 Depth=1
	s_clause 0x3                            ; 24-byte Folded Spill
	scratch_store_b32 off, v220, off offset:220
	scratch_store_b32 off, v219, off offset:216
	scratch_store_b64 off, v[213:214], off offset:204
	scratch_store_b64 off, v[211:212], off offset:196
.LBB0_12:                               ;   in Loop: Header=BB0_6 Depth=1
	ds_load_2addr_b64 v[196:199], v152 offset1:144
	ds_load_2addr_b64 v[200:203], v151 offset1:144
	ds_load_2addr_b64 v[204:207], v228 offset0:32 offset1:176
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[208:211], v152 offset0:2 offset1:146
	ds_load_2addr_b64 v[212:215], v151 offset0:2 offset1:146
	ds_load_2addr_b64 v[216:219], v228 offset0:34 offset1:178
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[220:223], v152 offset0:4 offset1:148
	ds_load_2addr_b64 v[224:227], v151 offset0:4 offset1:148
	ds_load_2addr_b64 v[247:250], v228 offset0:36 offset1:180
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[251:254], v152 offset0:6 offset1:150
	ds_load_2addr_b64 v[145:148], v151 offset0:6 offset1:150
	v_mov_b32_e32 v0, v228
	ds_load_2addr_b64 v[228:231], v228 offset0:38 offset1:182
	; sched_barrier mask(0x00000000)
	scratch_load_b32 v132, off, off offset:76 ; 4-byte Folded Reload
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[196:197], v[200:201], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[198:199], v[200:201], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[196:197], v[202:203], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[198:199], v[202:203], v[81:88]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[196:197], v[204:205], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[198:199], v[204:205], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[196:197], v[206:207], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[198:199], v[206:207], v[65:72]
	v_cmp_ne_u32_e32 vcc_lo, 1, v195
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[208:209], v[216:217], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[210:211], v[216:217], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[208:209], v[218:219], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[210:211], v[218:219], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[208:209], v[214:215], v[113:120]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[220:221], v[247:248], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[222:223], v[247:248], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[220:221], v[249:250], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[222:223], v[249:250], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[210:211], v[214:215], v[81:88]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[251:252], v[228:229], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[253:254], v[228:229], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[251:252], v[230:231], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[253:254], v[230:231], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[220:221], v[226:227], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[222:223], v[226:227], v[81:88]
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[208:209], v[212:213], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[210:211], v[212:213], v[89:96]
	v_mov_b32_e32 v130, v142
	v_mov_b32_e32 v142, v144
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[251:252], v[147:148], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[220:221], v[224:225], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[222:223], v[224:225], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[253:254], v[147:148], v[81:88]
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[197:198], v132 offset1:16
	ds_load_2addr_b32 v[195:196], v132 offset0:32 offset1:48
	scratch_load_b32 v132, off, off offset:12 ; 4-byte Folded Reload
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[251:252], v[145:146], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[253:254], v[145:146], v[89:96]
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[229:230], v132 offset1:1
	scratch_load_b32 v132, off, off offset:16 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[227:228], v132 offset1:1
	scratch_load_b32 v132, off, off offset:20 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[225:226], v132 offset1:1
	scratch_load_b32 v132, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[223:224], v132 offset1:1
	scratch_load_b32 v132, off, off offset:28 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[221:222], v132 offset1:1
	scratch_load_b32 v132, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[219:220], v132 offset1:1
	scratch_load_b32 v132, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[217:218], v132 offset1:1
	scratch_load_b32 v132, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[215:216], v132 offset1:1
	scratch_load_b32 v132, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[213:214], v132 offset1:1
	scratch_load_b32 v132, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[211:212], v132 offset1:1
	scratch_load_b32 v132, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[209:210], v132 offset1:1
	scratch_load_b32 v132, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[207:208], v132 offset1:1
	scratch_load_b32 v132, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[205:206], v132 offset1:1
	scratch_load_b32 v132, off, off offset:64 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[203:204], v132 offset1:1
	scratch_load_b32 v132, off, off offset:68 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[201:202], v132 offset1:1
	scratch_load_b32 v132, off, off offset:72 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[199:200], v132 offset1:1
	v_mov_b32_e32 v132, v140
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
	scratch_load_b32 v251, off, off offset:296 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_clause 0x3                            ; 16-byte Folded Reload
	scratch_load_b32 v253, off, off offset:324
	scratch_load_b32 v248, off, off offset:328
	scratch_load_b32 v252, off, off offset:332
	scratch_load_b32 v254, off, off offset:336
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_5
; %bb.13:                               ;   in Loop: Header=BB0_6 Depth=1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cndmask_b32_e64 v144, 0, v193, s4
	v_mov_b32_e32 v145, v142
                                        ; kill: def $vgpr140 killed $vgpr140
	v_cndmask_b32_e64 v142, 0, v143, s2
	v_cndmask_b32_e64 v140, 0, v141, s2
	v_cndmask_b32_e64 v141, 0, v130, s2
	v_lshrrev_b32_e32 v146, 4, v144
	v_and_b32_e32 v130, 0xf0f0f0f, v144
	v_cndmask_b32_e64 v143, 0, v145, s2
	v_cndmask_b32_e64 v147, 0, v191, s5
	v_cndmask_b32_e64 v138, 0, v138, s3
	v_and_b32_e32 v144, 0xf0f0f0f, v146
	v_cndmask_b32_e64 v137, 0, v137, s3
	v_cndmask_b32_e64 v139, 0, v139, s3
	v_lshrrev_b32_e32 v148, 4, v147
	v_and_b32_e32 v147, 0xf0f0f0f, v147
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v254, v[140:141], v[142:143] offset1:1
	v_cndmask_b32_e64 v141, 0, v194, s4
	v_perm_b32 v145, v144, v130, 0x5010400
	v_cndmask_b32_e64 v140, 0, v132, s3
	v_perm_b32 v130, v144, v130, 0x7030602
	v_and_b32_e32 v148, 0xf0f0f0f, v148
	v_lshrrev_b32_e32 v132, 4, v141
	v_and_b32_e32 v142, 0x7070707, v145
	v_lshrrev_b32_e32 v143, 1, v145
	v_and_b32_e32 v145, 0xf0f0f0f, v141
	v_lshrrev_b32_e32 v255, v248, v255
	v_and_b32_e32 v132, 0xf0f0f0f, v132
	v_perm_b32 v144, s35, 0x44403800, v142
	v_or_b32_e32 v141, 0x50505050, v142
	v_and_or_b32 v142, v143, s36, 0x3020100
	v_and_b32_e32 v143, 0x7070707, v130
	v_perm_b32 v146, v132, v145, 0x5010400
	v_lshrrev_b32_e32 v130, 1, v130
	v_perm_b32 v132, v132, v145, 0x7030602
	v_perm_b32 v141, v141, v144, v142
	v_perm_b32 v142, s35, 0x44403800, v143
	v_and_b32_e32 v144, 0x7070707, v146
	v_lshrrev_b32_e32 v146, 1, v146
	v_or_b32_e32 v143, 0x50505050, v143
	v_and_or_b32 v130, v130, s36, 0x3020100
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_perm_b32 v145, s35, 0x44403800, v144
	v_or_b32_e32 v144, 0x50505050, v144
	v_and_or_b32 v146, v146, s36, 0x3020100
	v_perm_b32 v142, v143, v142, v130
	v_and_b32_e32 v130, 0x7070707, v132
	v_lshrrev_b32_e32 v132, 1, v132
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_perm_b32 v143, v144, v145, v146
	v_cndmask_b32_e64 v144, 0, v192, s5
	v_perm_b32 v145, s35, 0x44403800, v130
	v_perm_b32 v146, v148, v147, 0x5010400
	v_or_b32_e32 v130, 0x50505050, v130
	v_and_or_b32 v132, v132, s36, 0x3020100
	v_lshrrev_b32_e32 v191, 4, v144
	v_perm_b32 v147, v148, v147, 0x7030602
	v_and_b32_e32 v192, 0xf0f0f0f, v144
	v_and_b32_e32 v148, 0x7070707, v146
	v_perm_b32 v144, v130, v145, v132
	v_and_b32_e32 v191, 0xf0f0f0f, v191
	v_lshrrev_b32_e32 v130, 1, v146
	v_and_b32_e32 v145, 0x7070707, v147
	v_perm_b32 v132, s35, 0x44403800, v148
	v_or_b32_e32 v148, 0x50505050, v148
	v_perm_b32 v146, v191, v192, 0x5010400
	v_perm_b32 v191, v191, v192, 0x7030602
	v_perm_b32 v193, s35, 0x44403800, v145
	v_or_b32_e32 v192, 0x50505050, v145
	v_and_or_b32 v130, v130, s36, 0x3020100
	v_and_b32_e32 v194, 0x7070707, v146
	v_lshrrev_b32_e32 v145, 1, v146
	v_and_b32_e32 v146, 0x7070707, v191
	v_lshrrev_b32_e32 v191, 1, v191
	v_lshrrev_b32_e32 v147, 1, v147
	v_perm_b32 v231, s35, 0x44403800, v194
	v_and_or_b32 v247, v145, s36, 0x3020100
	v_perm_b32 v249, s35, 0x44403800, v146
	v_or_b32_e32 v250, 0x50505050, v146
	v_and_or_b32 v191, v191, s36, 0x3020100
	v_perm_b32 v145, v148, v132, v130
	v_and_or_b32 v147, v147, s36, 0x3020100
	v_or_b32_e32 v194, 0x50505050, v194
	v_cvt_f32_f16_e64 v130, v255.l
	v_perm_b32 v148, v250, v249, v191
	scratch_load_b32 v191, off, off offset:8 ; 4-byte Folded Reload
	v_perm_b32 v146, v192, v193, v147
	v_perm_b32 v147, v194, v231, v247
	v_cndmask_b32_e64 v132, 0, v251, s6
	v_cndmask_b32_e64 v130, 0, v130, s7
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v191, v[137:138], v[139:140] offset0:64 offset1:65
	scratch_load_b32 v137, off, off         ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v137, v[141:142], v[143:144] offset1:1
	scratch_load_b32 v137, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v137, v[145:146], v[147:148] offset1:1
	ds_store_b32 v253, v132 offset:18432
	ds_store_b32 v252, v130 offset:18944
	s_branch .LBB0_5
.LBB0_14:
	s_clause 0x4                            ; 20-byte Folded Reload
	scratch_load_b32 v0, off, off offset:356
	scratch_load_b32 v19, off, off offset:360
	scratch_load_b32 v20, off, off offset:364
	scratch_load_b32 v6, off, off offset:368
	scratch_load_b32 v10, off, off offset:372
.LBB0_15:
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v1, 6, v0
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v2, v10, v6
	s_load_b64 s[4:5], s[0:1], 0x40
	v_and_b32_e32 v9, 31, v0
	v_and_b32_e32 v1, 0x3800, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v4, 17, v2
	v_add_nc_u32_e32 v5, 18, v2
	v_add_nc_u32_e32 v11, 19, v2
	v_add_nc_u32_e32 v12, 20, v2
	v_add_nc_u32_e32 v3, 0, v1
	v_xor_b32_e32 v1, 16, v2
	v_and_b32_e32 v4, 31, v4
	v_add_nc_u32_e32 v13, 21, v2
	v_and_b32_e32 v8, 31, v5
	v_lshl_add_u32 v10, v6, 7, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v6, v1, 2, v10
	v_add_nc_u32_e32 v1, 22, v2
	v_lshl_add_u32 v5, v2, 2, v10
	v_add_nc_u32_e32 v2, 23, v2
	v_lshl_add_u32 v7, v4, 2, v10
	v_and_b32_e32 v4, 31, v11
	v_and_b32_e32 v11, 31, v12
	v_and_b32_e32 v12, 31, v13
	v_and_b32_e32 v1, 31, v1
	v_and_b32_e32 v2, 31, v2
	v_lshl_add_u32 v14, v4, 2, v10
	v_lshl_add_u32 v15, v11, 2, v10
	v_lshl_add_u32 v16, v12, 2, v10
	v_lshl_add_u32 v17, v1, 2, v10
	v_lshl_add_u32 v18, v2, 2, v10
	v_lshl_add_u32 v8, v8, 2, v10
	ds_store_2addr_b32 v5, v173, v174 offset1:1
	ds_store_2addr_b32 v5, v171, v172 offset0:2 offset1:3
	ds_store_2addr_b32 v5, v169, v170 offset0:4 offset1:5
	ds_store_2addr_b32 v5, v167, v168 offset0:6 offset1:7
	ds_store_b32 v6, v164
	ds_store_b32 v7, v163
	ds_store_b32 v8, v162
	ds_store_b32 v14, v161
	ds_store_b32 v15, v160
	ds_store_b32 v16, v159
	ds_store_b32 v17, v158
	ds_store_b32 v18, v157
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_or3_b32 v4, s31, v19, v9
	v_add_nc_u32_e32 v1, s30, v20
	v_lshl_add_u32 v9, v9, 2, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s2, s33, v4
	v_cmp_gt_i32_e64 s3, s28, v1
	v_cmp_le_i32_e64 s1, s33, v4
	v_cmp_gt_i32_e64 s0, s29, v4
	v_ashrrev_i32_e32 v2, 31, v1
	v_cmp_gt_i32_e32 vcc_lo, s24, v4
	s_and_b32 s2, s3, s2
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_17
; %bb.16:
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	ds_load_b32 v13, v9
	v_dual_mov_b32 v19, s22 :: v_dual_mov_b32 v20, s23
	v_add_co_u32 v10, s2, s18, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s19, v11, s2
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v19, s4, v19, s0
	v_cndmask_b32_e64 v20, s5, v20, s0
	global_load_b32 v12, v[10:11], off
	v_dual_mov_b32 v10, s24 :: v_dual_mov_b32 v11, s25
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v19, v19, s20, vcc_lo
	v_cndmask_b32_e64 v20, v20, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v10, s29, v10, s0
	v_cndmask_b32_e64 v11, s26, v11, s0
	v_cndmask_b32_e64 v10, v10, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v11, v11, s24, vcc_lo
	v_sub_nc_u32_e32 v10, v4, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[10:11], null, v11, v1, v[10:11]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v11, 0 :: v_dual_mul_f32 v12, v12, v13
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s2, v19, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v20, v11, s2
	global_store_b32 v[10:11], v12, off
.LBB0_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 1, v1
	v_add_nc_u32_e32 v10, 1, v0
	s_xor_b32 s3, s1, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s2, s28, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s2, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_19
; %bb.18:
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_dual_mov_b32 v20, s25 :: v_dual_mov_b32 v21, s22
	v_and_b32_e32 v19, 31, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v12, s1, s18, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s19, v13, s1
	v_cndmask_b32_e64 v20, s26, v20, s0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v21, s4, v21, s0
	v_lshl_add_u32 v19, v19, 2, v3
	global_load_b32 v13, v[12:13], off offset:4
	v_mov_b32_e32 v12, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v20, v20, s24, vcc_lo
	v_cndmask_b32_e64 v21, v21, s20, vcc_lo
	ds_load_b32 v19, v19 offset:128
	v_cndmask_b32_e64 v12, s29, v12, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, 0, vcc_lo
	v_sub_nc_u32_e32 v12, v4, v12
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[11:12], null, v20, v11, v[12:13]
	s_wait_dscnt 0x0
	v_dual_mov_b32 v20, s23 :: v_dual_mul_f32 v13, v13, v19
	v_mov_b32_e32 v12, 0
	v_cndmask_b32_e64 v20, s5, v20, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_cndmask_b32_e64 v20, v20, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v21, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v20, v12, s1
	global_store_b32 v[11:12], v13, off
.LBB0_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v12, 2, v1
	v_add_nc_u32_e32 v11, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v12
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_21
; %bb.20:
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_mov_b32_e32 v13, s24
	v_dual_mov_b32 v21, s25 :: v_dual_mov_b32 v22, s22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v19, s1, s18, v19
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v20, null, s19, v20, s1
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v21, s26, v21, s0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v22, s4, v22, s0
	global_load_b32 v19, v[19:20], off offset:8
	v_and_b32_e32 v20, 31, v11
	v_cndmask_b32_e64 v13, s29, v13, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v21, v21, s24, vcc_lo
	v_cndmask_b32_e64 v22, v22, s20, vcc_lo
	v_lshl_add_u32 v20, v20, 2, v3
	v_cndmask_b32_e64 v13, v13, 0, vcc_lo
	ds_load_b32 v20, v20 offset:256
	v_sub_nc_u32_e32 v13, v4, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[12:13], null, v21, v12, v[13:14]
	v_mov_b32_e32 v21, s23
	v_mov_b32_e32 v13, 0
	v_cndmask_b32_e64 v21, s5, v21, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_cndmask_b32_e64 v21, v21, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v22, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v21, v13, s1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v19, v19, v20
	global_store_b32 v[12:13], v19, off
.LBB0_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v13, 3, v1
	v_add_nc_u32_e32 v12, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v13
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_23
; %bb.22:
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_mov_b32_e32 v22, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v22, s26, v22, s0
	v_add_co_u32 v19, s1, s18, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v20, null, s19, v20, s1
	global_load_b32 v21, v[19:20], off offset:12
	v_dual_mov_b32 v19, s24 :: v_dual_and_b32 v20, 31, v12
	v_cndmask_b32_e64 v19, s29, v19, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v20, v20, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v19, v19, 0, vcc_lo
	ds_load_b32 v23, v20 offset:384
	v_cndmask_b32_e64 v20, v22, s24, vcc_lo
	v_mov_b32_e32 v22, s22
	v_sub_nc_u32_e32 v19, v4, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[19:20], null, v20, v13, v[19:20]
	v_dual_mov_b32 v13, s23 :: v_dual_mov_b32 v20, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v13, s5, v13, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]
	v_cndmask_b32_e64 v13, v13, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v21, v21, v23
	v_cndmask_b32_e64 v22, s4, v22, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v22, v22, s20, vcc_lo
	v_add_co_u32 v19, s1, v22, v19
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v20, null, v13, v20, s1
	global_store_b32 v[19:20], v21, off
.LBB0_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v19, 4, v1
	v_add_nc_u32_e32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v19
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_25
; %bb.24:
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_dual_mov_b32 v23, s25 :: v_dual_and_b32 v22, 31, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v20, s1, s18, v20
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v21, null, s19, v21, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v23, s26, v23, s0
	global_load_b32 v21, v[20:21], off offset:16
	v_mov_b32_e32 v20, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v23, v23, s24, vcc_lo
	v_cndmask_b32_e64 v20, s29, v20, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v20, v20, 0, vcc_lo
	v_sub_nc_u32_e32 v20, v4, v20
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[19:20], null, v23, v19, v[20:21]
	v_mov_b32_e32 v23, s23
	v_lshl_add_u32 v22, v22, 2, v3
	v_mov_b32_e32 v20, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v23, s5, v23, s0
	ds_load_b32 v22, v22 offset:512
	v_mov_b32_e32 v24, s22
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]
	v_cndmask_b32_e64 v23, v23, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v21, v21, v22
	v_cndmask_b32_e64 v24, s4, v24, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, s20, vcc_lo
	v_add_co_u32 v19, s1, v24, v19
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v20, null, v23, v20, s1
	global_store_b32 v[19:20], v21, off
.LBB0_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v20, 5, v1
	v_add_nc_u32_e32 v19, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v20
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_lshlrev_b64_e32 v[21:22], 2, v[1:2]
	v_dual_mov_b32 v24, s25 :: v_dual_and_b32 v23, 31, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v21, s1, s18, v21
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v22, null, s19, v22, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v24, s26, v24, s0
	global_load_b32 v22, v[21:22], off offset:20
	v_mov_b32_e32 v21, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v24, v24, s24, vcc_lo
	v_cndmask_b32_e64 v21, s29, v21, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v21, v21, 0, vcc_lo
	v_sub_nc_u32_e32 v21, v4, v21
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[20:21], null, v24, v20, v[21:22]
	v_mov_b32_e32 v24, s23
	v_lshl_add_u32 v23, v23, 2, v3
	v_mov_b32_e32 v21, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v24, s5, v24, s0
	ds_load_b32 v23, v23 offset:640
	v_mov_b32_e32 v25, s22
	v_lshlrev_b64_e32 v[20:21], 2, v[20:21]
	v_cndmask_b32_e64 v24, v24, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v22, v22, v23
	v_cndmask_b32_e64 v25, s4, v25, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v25, v25, s20, vcc_lo
	v_add_co_u32 v20, s1, v25, v20
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v21, null, v24, v21, s1
	global_store_b32 v[20:21], v22, off
.LBB0_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v21, 6, v1
	v_add_nc_u32_e32 v20, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v21
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_lshlrev_b64_e32 v[22:23], 2, v[1:2]
	v_dual_mov_b32 v25, s25 :: v_dual_and_b32 v24, 31, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v22, s1, s18, v22
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v23, null, s19, v23, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v25, s26, v25, s0
	global_load_b32 v23, v[22:23], off offset:24
	v_mov_b32_e32 v22, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v25, v25, s24, vcc_lo
	v_cndmask_b32_e64 v22, s29, v22, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v22, v22, 0, vcc_lo
	v_sub_nc_u32_e32 v22, v4, v22
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[21:22], null, v25, v21, v[22:23]
	v_mov_b32_e32 v25, s23
	v_lshl_add_u32 v24, v24, 2, v3
	v_mov_b32_e32 v22, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v25, s5, v25, s0
	ds_load_b32 v24, v24 offset:768
	v_mov_b32_e32 v26, s22
	v_lshlrev_b64_e32 v[21:22], 2, v[21:22]
	v_cndmask_b32_e64 v25, v25, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v23, v23, v24
	v_cndmask_b32_e64 v26, s4, v26, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, s20, vcc_lo
	v_add_co_u32 v21, s1, v26, v21
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v22, null, v25, v22, s1
	global_store_b32 v[21:22], v23, off
.LBB0_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v22, 7, v1
	v_add_nc_u32_e32 v21, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v22
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_lshlrev_b64_e32 v[23:24], 2, v[1:2]
	v_dual_mov_b32 v26, s25 :: v_dual_and_b32 v25, 31, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v23, s1, s18, v23
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v24, null, s19, v24, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v26, s26, v26, s0
	global_load_b32 v24, v[23:24], off offset:28
	v_mov_b32_e32 v23, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v26, v26, s24, vcc_lo
	v_cndmask_b32_e64 v23, s29, v23, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v23, v23, 0, vcc_lo
	v_sub_nc_u32_e32 v23, v4, v23
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[22:23], null, v26, v22, v[23:24]
	v_mov_b32_e32 v26, s23
	v_lshl_add_u32 v25, v25, 2, v3
	v_mov_b32_e32 v23, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v26, s5, v26, s0
	ds_load_b32 v25, v25 offset:896
	v_mov_b32_e32 v27, s22
	v_lshlrev_b64_e32 v[22:23], 2, v[22:23]
	v_cndmask_b32_e64 v26, v26, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v24, v24, v25
	v_cndmask_b32_e64 v27, s4, v27, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v27, v27, s20, vcc_lo
	v_add_co_u32 v22, s1, v27, v22
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v23, null, v26, v23, s1
	global_store_b32 v[22:23], v24, off
.LBB0_31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v23, 8, v1
	v_add_nc_u32_e32 v22, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v23
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_33
; %bb.32:
	v_lshlrev_b64_e32 v[24:25], 2, v[1:2]
	v_dual_mov_b32 v27, s25 :: v_dual_and_b32 v26, 31, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v24, s1, s18, v24
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v25, null, s19, v25, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v27, s26, v27, s0
	global_load_b32 v25, v[24:25], off offset:32
	v_mov_b32_e32 v24, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v27, v27, s24, vcc_lo
	v_cndmask_b32_e64 v24, s29, v24, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v24, v24, 0, vcc_lo
	v_sub_nc_u32_e32 v24, v4, v24
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[23:24], null, v27, v23, v[24:25]
	v_mov_b32_e32 v27, s23
	v_lshl_add_u32 v26, v26, 2, v3
	v_mov_b32_e32 v24, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v27, s5, v27, s0
	ds_load_b32 v26, v26 offset:1024
	v_mov_b32_e32 v28, s22
	v_lshlrev_b64_e32 v[23:24], 2, v[23:24]
	v_cndmask_b32_e64 v27, v27, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v25, v25, v26
	v_cndmask_b32_e64 v28, s4, v28, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, s20, vcc_lo
	v_add_co_u32 v23, s1, v28, v23
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v24, null, v27, v24, s1
	global_store_b32 v[23:24], v25, off
.LBB0_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v24, 9, v1
	v_add_nc_u32_e32 v23, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v24
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_35
; %bb.34:
	v_lshlrev_b64_e32 v[25:26], 2, v[1:2]
	v_dual_mov_b32 v28, s25 :: v_dual_and_b32 v27, 31, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v25, s1, s18, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, s19, v26, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v28, s26, v28, s0
	global_load_b32 v26, v[25:26], off offset:36
	v_mov_b32_e32 v25, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v28, v28, s24, vcc_lo
	v_cndmask_b32_e64 v25, s29, v25, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v25, v25, 0, vcc_lo
	v_sub_nc_u32_e32 v25, v4, v25
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[24:25], null, v28, v24, v[25:26]
	v_mov_b32_e32 v28, s23
	v_lshl_add_u32 v27, v27, 2, v3
	v_mov_b32_e32 v25, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v28, s5, v28, s0
	ds_load_b32 v27, v27 offset:1152
	v_mov_b32_e32 v29, s22
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	v_cndmask_b32_e64 v28, v28, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v26, v26, v27
	v_cndmask_b32_e64 v29, s4, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, s20, vcc_lo
	v_add_co_u32 v24, s1, v29, v24
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v25, null, v28, v25, s1
	global_store_b32 v[24:25], v26, off
.LBB0_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v25, 10, v1
	v_add_nc_u32_e32 v24, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v25
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_37
; %bb.36:
	v_lshlrev_b64_e32 v[26:27], 2, v[1:2]
	v_dual_mov_b32 v29, s25 :: v_dual_and_b32 v28, 31, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v26, s1, s18, v26
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v27, null, s19, v27, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v29, s26, v29, s0
	global_load_b32 v27, v[26:27], off offset:40
	v_mov_b32_e32 v26, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v29, v29, s24, vcc_lo
	v_cndmask_b32_e64 v26, s29, v26, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 0, vcc_lo
	v_sub_nc_u32_e32 v26, v4, v26
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[25:26], null, v29, v25, v[26:27]
	v_mov_b32_e32 v29, s23
	v_lshl_add_u32 v28, v28, 2, v3
	v_mov_b32_e32 v26, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v29, s5, v29, s0
	ds_load_b32 v28, v28 offset:1280
	v_mov_b32_e32 v30, s22
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_cndmask_b32_e64 v29, v29, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v27, v27, v28
	v_cndmask_b32_e64 v30, s4, v30, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, s20, vcc_lo
	v_add_co_u32 v25, s1, v30, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, v29, v26, s1
	global_store_b32 v[25:26], v27, off
.LBB0_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v26, 11, v1
	v_add_nc_u32_e32 v25, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v26
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_39
; %bb.38:
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	v_dual_mov_b32 v30, s25 :: v_dual_and_b32 v29, 31, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v27, s1, s18, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v28, null, s19, v28, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v30, s26, v30, s0
	global_load_b32 v28, v[27:28], off offset:44
	v_mov_b32_e32 v27, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, s24, vcc_lo
	v_cndmask_b32_e64 v27, s29, v27, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v27, v27, 0, vcc_lo
	v_sub_nc_u32_e32 v27, v4, v27
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[26:27], null, v30, v26, v[27:28]
	v_mov_b32_e32 v30, s23
	v_lshl_add_u32 v29, v29, 2, v3
	v_mov_b32_e32 v27, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v30, s5, v30, s0
	ds_load_b32 v29, v29 offset:1408
	v_mov_b32_e32 v31, s22
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	v_cndmask_b32_e64 v30, v30, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v28, v28, v29
	v_cndmask_b32_e64 v31, s4, v31, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v31, v31, s20, vcc_lo
	v_add_co_u32 v26, s1, v31, v26
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v27, null, v30, v27, s1
	global_store_b32 v[26:27], v28, off
.LBB0_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v27, 12, v1
	v_add_nc_u32_e32 v26, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v27
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_41
; %bb.40:
	v_lshlrev_b64_e32 v[28:29], 2, v[1:2]
	v_dual_mov_b32 v31, s25 :: v_dual_and_b32 v30, 31, v26
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v28, s1, s18, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, s19, v29, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v31, s26, v31, s0
	global_load_b32 v29, v[28:29], off offset:48
	v_mov_b32_e32 v28, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v31, v31, s24, vcc_lo
	v_cndmask_b32_e64 v28, s29, v28, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v28, v28, 0, vcc_lo
	v_sub_nc_u32_e32 v28, v4, v28
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[27:28], null, v31, v27, v[28:29]
	v_mov_b32_e32 v31, s23
	v_lshl_add_u32 v30, v30, 2, v3
	v_mov_b32_e32 v28, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v31, s5, v31, s0
	ds_load_b32 v30, v30 offset:1536
	v_mov_b32_e32 v32, s22
	v_lshlrev_b64_e32 v[27:28], 2, v[27:28]
	v_cndmask_b32_e64 v31, v31, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v29, v29, v30
	v_cndmask_b32_e64 v32, s4, v32, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v32, v32, s20, vcc_lo
	v_add_co_u32 v27, s1, v32, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v28, null, v31, v28, s1
	global_store_b32 v[27:28], v29, off
.LBB0_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v28, 13, v1
	v_add_nc_u32_e32 v27, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v28
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_43
; %bb.42:
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	v_dual_mov_b32 v32, s25 :: v_dual_and_b32 v31, 31, v27
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v29, s1, s18, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, s19, v30, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v32, s26, v32, s0
	global_load_b32 v30, v[29:30], off offset:52
	v_mov_b32_e32 v29, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v32, v32, s24, vcc_lo
	v_cndmask_b32_e64 v29, s29, v29, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, v29, 0, vcc_lo
	v_sub_nc_u32_e32 v29, v4, v29
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[28:29], null, v32, v28, v[29:30]
	v_mov_b32_e32 v32, s23
	v_lshl_add_u32 v31, v31, 2, v3
	v_mov_b32_e32 v29, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v32, s5, v32, s0
	ds_load_b32 v31, v31 offset:1664
	v_mov_b32_e32 v33, s22
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_cndmask_b32_e64 v32, v32, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v28, s1, v33, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v32, v29, s1
	global_store_b32 v[28:29], v30, off
.LBB0_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v29, 14, v1
	v_add_nc_u32_e32 v28, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v29
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_45
; %bb.44:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_dual_mov_b32 v33, s25 :: v_dual_and_b32 v32, 31, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, s26, v33, s0
	global_load_b32 v31, v[30:31], off offset:56
	v_mov_b32_e32 v30, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v33, v33, s24, vcc_lo
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	v_sub_nc_u32_e32 v30, v4, v30
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[29:30], null, v33, v29, v[30:31]
	v_mov_b32_e32 v33, s23
	v_lshl_add_u32 v32, v32, 2, v3
	v_mov_b32_e32 v30, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v33, s5, v33, s0
	ds_load_b32 v32, v32 offset:1792
	v_mov_b32_e32 v34, s22
	v_lshlrev_b64_e32 v[29:30], 2, v[29:30]
	v_cndmask_b32_e64 v33, v33, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v31, v31, v32
	v_cndmask_b32_e64 v34, s4, v34, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, s20, vcc_lo
	v_add_co_u32 v29, s1, v34, v29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v30, null, v33, v30, s1
	global_store_b32 v[29:30], v31, off
.LBB0_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v30, 15, v1
	v_add_nc_u32_e32 v29, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s28, v30
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_47
; %bb.46:
	v_lshlrev_b64_e32 v[31:32], 2, v[1:2]
	v_dual_mov_b32 v34, s25 :: v_dual_and_b32 v33, 31, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v31, s1, s18, v31
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v32, null, s19, v32, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v34, s26, v34, s0
	global_load_b32 v32, v[31:32], off offset:60
	v_mov_b32_e32 v31, s24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v34, v34, s24, vcc_lo
	v_cndmask_b32_e64 v31, s29, v31, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v31, v31, 0, vcc_lo
	v_sub_nc_u32_e32 v31, v4, v31
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[30:31], null, v34, v30, v[31:32]
	v_mov_b32_e32 v34, s23
	v_lshl_add_u32 v33, v33, 2, v3
	v_mov_b32_e32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v34, s5, v34, s0
	ds_load_b32 v33, v33 offset:1920
	v_mov_b32_e32 v35, s22
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v34, v34, s21, vcc_lo
	s_wait_dscnt 0x0
	v_mul_f32_e32 v32, v32, v33
	v_cndmask_b32_e64 v35, s4, v35, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v35, v35, s20, vcc_lo
	v_add_co_u32 v30, s1, v35, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v34, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v245, v246 offset1:1
	ds_store_2addr_b32 v5, v243, v244 offset0:2 offset1:3
	ds_store_2addr_b32 v5, v241, v242 offset0:4 offset1:5
	ds_store_2addr_b32 v5, v239, v240 offset0:6 offset1:7
	ds_store_b32 v6, v238
	ds_store_b32 v7, v237
	ds_store_b32 v8, v236
	ds_store_b32 v14, v235
	ds_store_b32 v15, v234
	ds_store_b32 v16, v233
	ds_store_b32 v17, v232
	ds_store_b32 v18, v154
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_49
; %bb.48:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	ds_load_b32 v33, v9
	v_mov_b32_e32 v34, s22
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:64
	v_dual_mov_b32 v30, s24 :: v_dual_mov_b32 v31, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v30, s29, v30, s0
	v_cndmask_b32_e64 v31, s26, v31, s0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	v_cndmask_b32_e64 v31, v31, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v30, v4, v30
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v34, s4, v34, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s5, v0, s0
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v34, v34, s20, vcc_lo
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v30, s1, v34, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v33
	global_store_b32 v[30:31], v32, off
.LBB0_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 17, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_51
; %bb.50:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:68
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v10
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:128
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_51:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_53
; %bb.52:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:72
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v11
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:256
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 19, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_55
; %bb.54:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:76
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v12
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:384
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 20, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_57
; %bb.56:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:80
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v13
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:512
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 21, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_59
; %bb.58:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:84
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v19
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:640
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 22, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_61
; %bb.60:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:88
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v20
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:768
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_61:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 23, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_63
; %bb.62:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:92
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v21
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:896
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 24, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_65
; %bb.64:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:96
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v22
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1024
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 25, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_67
; %bb.66:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:100
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v23
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1152
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 26, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_69
; %bb.68:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:104
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v24
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1280
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 27, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_71
; %bb.70:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:108
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v25
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1408
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 28, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_73
; %bb.72:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:112
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v26
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1536
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 29, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_75
; %bb.74:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:116
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v27
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1664
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 30, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_77
; %bb.76:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:120
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v28
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1792
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 31, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_79
; %bb.78:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:124
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v29
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1920
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v150, v153 offset1:1
	ds_store_2addr_b32 v5, v62, v149 offset0:2 offset1:3
	ds_store_2addr_b32 v5, v63, v64 offset0:4 offset1:5
	ds_store_2addr_b32 v5, v60, v61 offset0:6 offset1:7
	ds_store_b32 v6, v59
	ds_store_b32 v7, v58
	ds_store_b32 v8, v57
	ds_store_b32 v14, v56
	ds_store_b32 v15, v55
	ds_store_b32 v16, v54
	ds_store_b32 v17, v53
	ds_store_b32 v18, v52
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_81
; %bb.80:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	ds_load_b32 v33, v9
	v_mov_b32_e32 v34, s22
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:128
	v_dual_mov_b32 v30, s24 :: v_dual_mov_b32 v31, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v30, s29, v30, s0
	v_cndmask_b32_e64 v31, s26, v31, s0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	v_cndmask_b32_e64 v31, v31, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v30, v4, v30
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v34, s4, v34, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, s5, v0, s0
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v34, v34, s20, vcc_lo
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v30, s1, v34, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v33
	global_store_b32 v[30:31], v32, off
.LBB0_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 33, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_83
; %bb.82:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:132
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v10
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:128
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 34, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_85
; %bb.84:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:136
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v11
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:256
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 35, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_87
; %bb.86:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:140
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v12
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:384
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 36, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_89
; %bb.88:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:144
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v13
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:512
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 37, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_91
; %bb.90:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:148
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v19
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:640
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 38, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_93
; %bb.92:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:152
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v20
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:768
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 39, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_95
; %bb.94:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:156
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v21
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:896
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 40, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_97
; %bb.96:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:160
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v22
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1024
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 41, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_99
; %bb.98:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:164
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v23
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1152
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 42, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_101
; %bb.100:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:168
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v24
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1280
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 43, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_103
; %bb.102:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:172
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v25
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1408
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 44, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_105
; %bb.104:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:176
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v26
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1536
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 45, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_107
; %bb.106:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:180
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v27
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1664
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_107:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 46, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_109
; %bb.108:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:184
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v28
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1792
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_109:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 47, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_111
; %bb.110:
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	v_mov_b32_e32 v33, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, s26, v33, s0
	v_add_co_u32 v30, s1, s18, v30
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v31, null, s19, v31, s1
	global_load_b32 v32, v[30:31], off offset:188
	v_dual_mov_b32 v30, s24 :: v_dual_and_b32 v31, 31, v29
	v_cndmask_b32_e64 v30, s29, v30, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v31, v31, 2, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v30, v30, 0, vcc_lo
	ds_load_b32 v34, v31 offset:1920
	v_cndmask_b32_e64 v31, v33, s24, vcc_lo
	v_mov_b32_e32 v33, s22
	v_sub_nc_u32_e32 v30, v4, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v31, v0, v[30:31]
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v31, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v32, v32, v34
	v_cndmask_b32_e64 v33, s4, v33, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, s20, vcc_lo
	v_add_co_u32 v30, s1, v33, v30
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, v0, v31, s1
	global_store_b32 v[30:31], v32, off
.LBB0_111:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v50, v51 offset1:1
	ds_store_2addr_b32 v5, v48, v49 offset0:2 offset1:3
	ds_store_2addr_b32 v5, v45, v46 offset0:4 offset1:5
	ds_store_2addr_b32 v5, v43, v44 offset0:6 offset1:7
	ds_store_b32 v6, v42
	ds_store_b32 v7, v41
	ds_store_b32 v8, v40
	ds_store_b32 v14, v39
	ds_store_b32 v15, v38
	ds_store_b32 v16, v37
	ds_store_b32 v17, v36
	ds_store_b32 v18, v47
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_113
; %bb.112:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	ds_load_b32 v8, v9
	v_mov_b32_e32 v9, s22
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:192
	v_dual_mov_b32 v5, s24 :: v_dual_mov_b32 v6, s25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_cndmask_b32_e64 v6, s26, v6, s0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	v_cndmask_b32_e64 v6, v6, s24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v5, v4, v5
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v9, s4, v9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, s5, v0, s0
	v_cndmask_b32_e64 v9, v9, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v8
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, s1, v9, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_113:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 49, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_115
; %bb.114:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:196
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:128
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_115:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_117
; %bb.116:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:200
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:256
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 51, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_119
; %bb.118:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:204
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:384
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_119:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 52, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_121
; %bb.120:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:208
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:512
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_121:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 53, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_123
; %bb.122:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:212
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:640
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_123:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 54, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_125
; %bb.124:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:216
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:768
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 55, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_127
; %bb.126:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:220
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:896
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 56, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_129
; %bb.128:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:224
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:1024
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_129:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 57, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_131
; %bb.130:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:228
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:1152
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_131:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 58, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_133
; %bb.132:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:232
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:1280
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_133:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 59, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_135
; %bb.134:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:236
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:1408
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_135:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_137
; %bb.136:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:240
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:1536
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_137:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 61, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_139
; %bb.138:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:244
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:1664
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_139:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 62, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_141
; %bb.140:
	v_lshlrev_b64_e32 v[5:6], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s18, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s19, v6, s1
	global_load_b32 v7, v[5:6], off offset:248
	v_dual_mov_b32 v5, s24 :: v_dual_and_b32 v6, 31, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, s29, v5, s0
	v_lshl_add_u32 v6, v6, 2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, 0, vcc_lo
	ds_load_b32 v9, v6 offset:1792
	v_mov_b32_e32 v8, s25
	v_sub_nc_u32_e32 v5, v4, v5
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v8, s26, v8, s0
	v_cndmask_b32_e64 v6, v8, s24, vcc_lo
	v_mov_b32_e32 v8, s22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[5:6], null, v6, v0, v[5:6]
	v_mov_b32_e32 v0, s23
	v_mov_b32_e32 v6, 0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v8, s4, v8, s0
	v_cndmask_b32_e64 v0, s5, v0, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_cndmask_b32_e64 v8, v8, s20, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v0, v0, s21, vcc_lo
	v_add_co_u32 v5, s1, v8, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v0, v6, s1
	global_store_b32 v[5:6], v7, off
.LBB0_141:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s28, v0
	s_and_b32 s1, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_143
; %bb.142:
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_dual_mov_b32 v6, s25 :: v_dual_and_b32 v5, 31, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, s1, s18, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s19, v2, s1
	global_load_b32 v2, v[1:2], off offset:252
	v_mov_b32_e32 v1, s24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v1, s29, v1, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v1, v1, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_sub_nc_u32_e32 v1, v4, v1
	v_mov_b32_e32 v4, s22
	v_lshl_add_u32 v3, v5, 2, v3
	v_cndmask_b32_e64 v5, s26, v6, s0
	s_wait_kmcnt 0x0
	v_cndmask_b32_e64 v4, s4, v4, s0
	ds_load_b32 v3, v3 offset:1920
	v_cndmask_b32_e64 v5, v5, s24, vcc_lo
	v_cndmask_b32_e64 v4, v4, s20, vcc_lo
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[0:1], null, v5, v0, v[1:2]
	s_wait_dscnt 0x0
	v_dual_mov_b32 v5, s23 :: v_dual_mul_f32 v2, v2, v3
	v_mov_b32_e32 v1, 0
	v_cndmask_b32_e64 v5, s5, v5, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_cndmask_b32_e64 v5, v5, s21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
.LBB0_143:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end0:
	.size	qkv_r2, .Lfunc_end0-qkv_r2
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel qkv_r2
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 380
		.amdhsa_kernarg_size 92
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-qkv_r2)<<4)&4080)>>4
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
	.set .Lqkv_r2.num_vgpr, 256
	.set .Lqkv_r2.num_agpr, 0
	.set .Lqkv_r2.numbered_sgpr, 40
	.set .Lqkv_r2.num_named_barrier, 0
	.set .Lqkv_r2.private_seg_size, 380
	.set .Lqkv_r2.uses_vcc, 1
	.set .Lqkv_r2.uses_flat_scratch, 1
	.set .Lqkv_r2.has_dyn_sized_stack, 0
	.set .Lqkv_r2.has_recursion, 0
	.set .Lqkv_r2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 30388
; TotalNumSgprs: 42
; NumVgprs: 256
; ScratchSize: 380
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
	.type	__hip_cuid_d7aa9a4147706fbb,@object ; @__hip_cuid_d7aa9a4147706fbb
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d7aa9a4147706fbb
__hip_cuid_d7aa9a4147706fbb:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d7aa9a4147706fbb, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_d7aa9a4147706fbb
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
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 92
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           qkv_r2
    .private_segment_fixed_size: 380
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         qkv_r2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 103
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
