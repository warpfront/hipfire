	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gu_r2                   ; -- Begin function gu_r2
	.globl	gu_r2
	.p2align	8
	.type	gu_r2,@function
gu_r2:                                  ; @gu_r2
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[24:27], s[0:1], 0x38
	s_load_b256 s[16:23], s[0:1], 0x0
	v_lshrrev_b32_e32 v3, 2, v0
	s_lshl_b32 s30, ttmp9, 7
	v_and_b32_e32 v1, 3, v0
	s_lshl_b32 s28, ttmp7, 7
	s_load_b256 s[8:15], s[0:1], 0x20
	v_or_b32_e32 v5, s30, v3
	v_or_b32_e32 v4, s28, v3
	v_and_b32_e32 v7, 0x7f, v0
	v_lshrrev_b32_e32 v48, 1, v0
	v_mov_b32_e32 v47, 0
	v_or_b32_e32 v6, 64, v5
	s_wait_kmcnt 0x0
	s_add_co_i32 s29, s25, s24
	v_mov_b32_e32 v10, s17
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s29, -1
	v_dual_mov_b32 v11, s16 :: v_dual_lshlrev_b32 v2, 4, v1
	v_min_i32_e32 v9, s3, v5
	s_ashr_i32 s2, s26, 31
	s_add_co_i32 s5, s27, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_u32_u24 v8, 0x48, v3, v2
	v_or_b32_e32 v3, 64, v4
	v_cmp_gt_i32_e32 vcc_lo, s24, v9
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s4, s2, 24
	v_cmp_gt_i32_e64 s0, s27, v4
	v_min_i32_e32 v4, s5, v4
	v_cmp_gt_i32_e64 s1, s27, v3
	v_cndmask_b32_e64 v12, s24, 0, vcc_lo
	v_min_i32_e32 v3, s5, v3
	s_add_co_i32 s4, s26, s4
	v_cmp_gt_i32_e64 s2, s29, v5
	v_min_i32_e32 v5, s3, v6
	v_sub_nc_u32_e32 v9, v9, v12
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s16, s4, 8
	v_mad_co_u64_u32 v[209:210], null, v4, s26, v[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s6, s16, 0x88
	v_mad_co_u64_u32 v[210:211], null, v3, s26, v[2:3]
	v_mul_lo_u32 v2, s6, v9
	v_cmp_gt_i32_e64 s4, s24, v5
	v_or_b32_e32 v9, s30, v7
	v_dual_cndmask_b32 v12, s18, v11 :: v_dual_lshlrev_b32 v1, 3, v1
	v_cndmask_b32_e32 v4, s19, v10, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, s24, 0, s4
	v_min_i32_e32 v13, s3, v9
	v_add_co_u32 v2, vcc_lo, v12, v2
	v_or_b32_e32 v12, s28, v48
	s_delay_alu instid0(VALU_DEP_4)
	v_sub_nc_u32_e32 v5, v5, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v4, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s24, v13
	v_cmp_gt_i32_e64 s3, s29, v6
	v_mul_lo_u32 v4, s6, v5
	v_cndmask_b32_e64 v6, s18, v11, s4
	v_cndmask_b32_e64 v14, s19, v10, s4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, s24, 0, vcc_lo
	v_min_i32_e32 v15, s5, v12
	v_dual_cndmask_b32 v10, s19, v10 :: v_dual_lshlrev_b32 v7, 3, v7
	v_cndmask_b32_e32 v11, s18, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_sub_nc_u32_e32 v13, v13, v5
	v_add_co_u32 v4, s4, v6, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, 0, v14, s4
	v_mul_lo_u32 v13, s6, v13
	v_lshrrev_b32_e32 v14, 7, v0
	v_mul_lo_u32 v6, v15, s16
	v_cmp_gt_i32_e64 s4, s27, v12
	v_lshlrev_b32_e32 v12, 2, v48
	v_cmp_gt_i32_e64 s5, s29, v9
	v_lshl_or_b32 v7, v14, 2, v7
	v_lshlrev_b32_e32 v234, 4, v14
	v_add_co_u32 v101, vcc_lo, v11, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v102, null, 0, v10, vcc_lo
	v_lshlrev_b32_e32 v103, 1, v6
	v_add_nc_u32_e32 v187, 0, v12
	v_add_nc_u32_e32 v238, 0, v7
	v_add_nc_u32_e32 v239, 0, v8
	s_cmp_gt_i32 s26, 0xff
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s26, 0x100
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	v_add_co_u32 v15, vcc_lo, v2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, 0, v3, vcc_lo
	v_add_co_u32 v17, vcc_lo, v4, v1
	global_load_b32 v25, v[101:102], off
	s_clause 0x1
	global_load_b128 v[7:10], v209, s[20:21]
	global_load_b128 v[11:14], v210, s[20:21]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, 0, v5, vcc_lo
	v_dual_mov_b32 v20, 0 :: v_dual_lshlrev_b32 v19, 1, v6
	global_load_b64 v[21:22], v[15:16], off offset:8
	global_load_b64 v[23:24], v[17:18], off offset:8
	s_clause 0x1
	global_load_b128 v[133:136], v209, s[20:21] offset:64
	global_load_b128 v[129:132], v210, s[20:21] offset:64
	v_add_nc_u32_e32 v26, 0x1200, v239
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]
	v_add_nc_u32_e32 v27, 0x2400, v239
	s_mov_b32 s7, 0x4e4c4a48
	s_mov_b32 s14, 0x4040404
	v_add_nc_u32_e32 v28, 0x3600, v239
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v19, vcc_lo, s22, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, s23, v20, vcc_lo
	global_load_b32 v6, v[19:20], off
	global_load_b64 v[85:86], v[15:16], off offset:40
	global_load_b64 v[91:92], v[17:18], off offset:40
	s_wait_loadcnt 0x9
	v_lshrrev_b32_e32 v15, v234, v25
	s_wait_loadcnt 0x8
	v_cndmask_b32_e64 v8, 0, v8, s0
	s_wait_loadcnt 0x7
	v_cndmask_b32_e64 v12, 0, v12, s1
	v_cndmask_b32_e64 v11, 0, v11, s1
	v_cndmask_b32_e64 v14, 0, v14, s1
	v_cvt_f32_f16_e32 v15, v15.l
	s_wait_loadcnt 0x6
	v_cndmask_b32_e64 v16, 0, v21, s2
	v_cndmask_b32_e64 v17, 0, v22, s2
	s_wait_loadcnt 0x5
	v_cndmask_b32_e64 v18, 0, v23, s3
	v_cndmask_b32_e64 v19, 0, v24, s3
	s_wait_loadcnt 0x4
	v_dual_mov_b32 v87, v134 :: v_dual_mov_b32 v134, v136
	s_wait_loadcnt 0x3
	v_mov_b32_e32 v136, v130
	v_cndmask_b32_e64 v23, 0, v15, s5
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
	v_cndmask_b32_e64 v6, 0, v6, s4
	v_perm_b32 v24, v16, v15, 0x5010400
	v_perm_b32 v15, v16, v15, 0x7030602
	v_perm_b32 v16, v17, v20, 0x5010400
	v_perm_b32 v17, v17, v20, 0x7030602
	v_perm_b32 v20, v18, v21, 0x5010400
	v_perm_b32 v18, v18, v21, 0x7030602
	v_perm_b32 v21, v22, v19, 0x5010400
	v_perm_b32 v19, v22, v19, 0x7030602
	ds_store_b32 v187, v6 offset:18432
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
	v_perm_b32 v34, s7, 0x44403800, v6
	v_or_b32_e32 v6, 0x50505050, v6
	v_and_or_b32 v22, v22, s14, 0x3020100
	v_perm_b32 v35, s7, 0x44403800, v24
	v_or_b32_e32 v24, 0x50505050, v24
	v_and_or_b32 v36, v15, s14, 0x3020100
	v_perm_b32 v37, s7, 0x44403800, v25
	v_or_b32_e32 v25, 0x50505050, v25
	v_and_or_b32 v38, v16, s14, 0x3020100
	v_perm_b32 v39, s7, 0x44403800, v29
	v_or_b32_e32 v29, 0x50505050, v29
	v_and_or_b32 v40, v17, s14, 0x3020100
	v_perm_b32 v41, s7, 0x44403800, v30
	v_or_b32_e32 v30, 0x50505050, v30
	v_and_or_b32 v20, v20, s14, 0x3020100
	v_perm_b32 v42, s7, 0x44403800, v31
	v_or_b32_e32 v31, 0x50505050, v31
	v_and_or_b32 v43, v18, s14, 0x3020100
	v_perm_b32 v44, s7, 0x44403800, v32
	v_or_b32_e32 v32, 0x50505050, v32
	v_and_or_b32 v21, v21, s14, 0x3020100
	v_perm_b32 v45, s7, 0x44403800, v33
	v_or_b32_e32 v33, 0x50505050, v33
	v_and_or_b32 v46, v19, s14, 0x3020100
	v_cndmask_b32_e64 v13, 0, v13, s1
	v_perm_b32 v15, v6, v34, v22
	v_perm_b32 v16, v24, v35, v36
	v_perm_b32 v17, v25, v37, v38
	v_perm_b32 v18, v29, v39, v40
	v_cndmask_b32_e64 v7, 0, v7, s0
	v_cndmask_b32_e64 v10, 0, v10, s0
	v_cndmask_b32_e64 v9, 0, v9, s0
	v_perm_b32 v19, v30, v41, v20
	v_perm_b32 v20, v31, v42, v43
	v_perm_b32 v21, v32, v44, v21
	v_perm_b32 v22, v33, v45, v46
	ds_store_b32 v238, v23 offset:18944
	ds_store_2addr_b64 v26, v[11:12], v[13:14] offset1:1
	ds_store_2addr_b64 v27, v[15:16], v[17:18] offset1:1
	ds_store_2addr_b64 v239, v[7:8], v[9:10] offset1:1
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
	v_dual_mov_b32 v36, 0 :: v_dual_and_b32 v7, 15, v0
	v_dual_mov_b32 v37, 0 :: v_dual_and_b32 v8, 0x60, v0
	v_dual_mov_b32 v38, 0 :: v_dual_and_b32 v17, 64, v48
	v_dual_mov_b32 v39, 0 :: v_dual_and_b32 v10, 8, v48
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v248, 0
	v_dual_mov_b32 v247, 0 :: v_dual_mov_b32 v250, 0
	v_dual_mov_b32 v249, 0 :: v_dual_mov_b32 v252, 0
	v_dual_mov_b32 v251, 0 :: v_dual_mov_b32 v254, 0
	v_dual_mov_b32 v253, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v255, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v245, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v231, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v184, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v185, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_15
; %bb.4:
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v0, off offset:364
	scratch_store_b32 off, v8, off offset:376
	v_or_b32_e32 v8, v8, v10
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v17, off offset:368
	scratch_store_b32 off, v7, off offset:372
	v_or_b32_e32 v6, v17, v7
	v_dual_mov_b32 v104, 0 :: v_dual_add_nc_u32 v7, 0, v10
	v_and_b32_e32 v0, 0x6f, v0
	v_lshlrev_b32_e32 v8, 3, v8
	v_add_co_u32 v215, vcc_lo, v2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v216, null, 0, v3, vcc_lo
	v_add_co_u32 v217, vcc_lo, v4, v1
	v_mov_b32_e32 v185, v104
	v_mad_u32_u24 v0, 0x48, v0, 0
	v_mov_b32_e32 v181, v104
	v_dual_mov_b32 v184, v104 :: v_dual_add_nc_u32 v1, 0, v8
	v_mov_b32_e32 v180, v104
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v0, v0, v10
	v_mov_b32_e32 v245, v104
	v_dual_mov_b32 v178, v104 :: v_dual_add_nc_u32 v3, 0x4a08, v1
	v_mov_b32_e32 v176, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v231, v104 :: v_dual_add_nc_u32 v244, 0x2400, v0
	v_add_nc_u32_e32 v0, 0x1000, v239
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v10, off offset:380
	scratch_store_b32 off, v3, off offset:4
	v_dual_mov_b32 v154, v104 :: v_dual_add_nc_u32 v3, 0x4a10, v1
	v_mul_u32_u24_e32 v9, 0x48, v6
	v_dual_mov_b32 v183, v104 :: v_dual_lshlrev_b32 v6, 2, v6
	scratch_store_b32 off, v3, off offset:8 ; 4-byte Folded Spill
	v_dual_mov_b32 v152, v104 :: v_dual_add_nc_u32 v3, 0x4a18, v1
	v_dual_mov_b32 v179, v104 :: v_dual_add_nc_u32 v190, v7, v9
	v_mov_b32_e32 v172, v104
	scratch_store_b32 off, v3, off offset:12 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a20, v1
	v_dual_mov_b32 v175, v104 :: v_dual_add_nc_u32 v2, 0, v6
	v_mov_b32_e32 v170, v104
	v_dual_mov_b32 v153, v104 :: v_dual_add_nc_u32 v228, 0x4a00, v1
	scratch_store_b32 off, v3, off offset:16 ; 4-byte Folded Spill
	v_dual_mov_b32 v168, v104 :: v_dual_add_nc_u32 v3, 0x4a28, v1
	scratch_store_b32 off, v0, off          ; 4-byte Folded Spill
	v_dual_mov_b32 v145, v104 :: v_dual_mov_b32 v0, v209
	scratch_store_b32 off, v3, off offset:20 ; 4-byte Folded Spill
	v_dual_mov_b32 v166, v104 :: v_dual_add_nc_u32 v3, 0x4a30, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v218, null, 0, v5, vcc_lo
	v_mov_b32_e32 v164, v104
	scratch_store_b32 off, v3, off offset:24 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a38, v1
	v_dual_mov_b32 v182, v104 :: v_dual_add_nc_u32 v241, 0x2400, v239
	v_mov_b32_e32 v162, v104
	v_dual_mov_b32 v177, v104 :: v_dual_add_nc_u32 v242, 0x3600, v239
	scratch_store_b32 off, v3, off offset:28 ; 4-byte Folded Spill
	v_dual_mov_b32 v160, v104 :: v_dual_add_nc_u32 v3, 0x4a80, v1
	v_dual_mov_b32 v146, v104 :: v_dual_add_nc_u32 v229, 0x800, v190
	v_dual_mov_b32 v151, v104 :: v_dual_add_nc_u32 v230, 0x4800, v2
	scratch_store_b32 off, v3, off offset:32 ; 4-byte Folded Spill
	v_dual_mov_b32 v158, v104 :: v_dual_add_nc_u32 v3, 0x4a88, v1
	v_dual_mov_b32 v171, v104 :: v_dual_mov_b32 v254, v104
	v_dual_mov_b32 v169, v104 :: v_dual_mov_b32 v252, v104
	scratch_store_b32 off, v3, off offset:36 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a90, v1
	v_dual_mov_b32 v167, v104 :: v_dual_mov_b32 v250, v104
	v_dual_mov_b32 v165, v104 :: v_dual_mov_b32 v248, v104
	scratch_store_b32 off, v3, off offset:40 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a98, v1
	v_dual_mov_b32 v163, v104 :: v_dual_mov_b32 v48, v104
	v_dual_mov_b32 v161, v104 :: v_dual_mov_b32 v46, v104
	scratch_store_b32 off, v3, off offset:44 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4aa0, v1
	v_dual_mov_b32 v159, v104 :: v_dual_mov_b32 v44, v104
	v_dual_mov_b32 v150, v104 :: v_dual_mov_b32 v157, v104
	v_mov_b32_e32 v42, v104
	scratch_store_b32 off, v3, off offset:48 ; 4-byte Folded Spill
	v_dual_mov_b32 v148, v104 :: v_dual_add_nc_u32 v3, 0x4aa8, v1
	v_dual_mov_b32 v149, v104 :: v_dual_mov_b32 v40, v104
	v_dual_mov_b32 v147, v104 :: v_dual_mov_b32 v38, v104
	scratch_store_b32 off, v3, off offset:52 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4ab0, v1
	v_add_nc_u32_e32 v1, 0x4ab8, v1
	v_dual_mov_b32 v255, v104 :: v_dual_mov_b32 v36, v104
	v_mov_b32_e32 v253, v104
	s_clause 0x2                            ; 16-byte Folded Spill
	scratch_store_b32 off, v3, off offset:56
	scratch_store_b32 off, v1, off offset:60
	scratch_store_b64 off, v[0:1], off offset:296
	v_mov_b32_e32 v251, v104
	v_mov_b32_e32 v249, v104
	v_mov_b32_e32 v247, v104
	v_mov_b32_e32 v51, v104
	v_mov_b32_e32 v63, v104
	v_mov_b32_e32 v49, v104
	v_mov_b32_e32 v45, v104
	v_mov_b32_e32 v43, v104
	v_mov_b32_e32 v41, v104
	v_mov_b32_e32 v39, v104
	v_mov_b32_e32 v37, v104
	v_dual_mov_b32 v47, v104 :: v_dual_mov_b32 v0, v210
	s_mov_b32 s7, 0
	s_sub_co_i32 s17, 0, s16
	s_mov_b32 s18, 1
	s_movk_i32 s14, 0x100
	s_movk_i32 s19, 0x88
	s_mov_b32 s26, 0x4e4c4a48
	s_mov_b32 s31, 0x4040404
	s_clause 0xc                            ; 60-byte Folded Spill
	scratch_store_b64 off, v[0:1], off offset:304
	scratch_store_b64 off, v[101:102], off offset:312
	scratch_store_b32 off, v187, off offset:320
	scratch_store_b32 off, v234, off offset:324
	scratch_store_b32 off, v238, off offset:328
	scratch_store_b32 off, v239, off offset:332
	scratch_store_b32 off, v215, off offset:336
	scratch_store_b32 off, v216, off offset:340
	scratch_store_b32 off, v217, off offset:344
	scratch_store_b32 off, v218, off offset:348
	scratch_store_b32 off, v241, off offset:352
	scratch_store_b32 off, v242, off offset:356
	scratch_store_b32 off, v230, off offset:360
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_clause 0x3                            ; 32-byte Folded Reload
	scratch_load_b64 v[137:138], off, off offset:220 th:TH_LOAD_LU
	scratch_load_b64 v[139:140], off, off offset:228 th:TH_LOAD_LU
	scratch_load_b64 v[141:142], off, off offset:236 th:TH_LOAD_LU
	scratch_load_b64 v[143:144], off, off offset:244 th:TH_LOAD_LU
	s_add_co_i32 s18, s18, 1
	s_addk_co_i32 s14, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s17, s18
	s_addk_co_i32 s19, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, 1
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b64 v[188:189], off, off offset:260 th:TH_LOAD_LU
	scratch_load_b64 v[191:192], off, off offset:276 th:TH_LOAD_LU
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v172, v137, v49 :: v_dual_fmac_f32 v151, v139, v50
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v182, v141, v59 :: v_dual_fmac_f32 v183, v143, v60
	scratch_load_b64 v[59:60], off, off offset:252 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_dual_fmac_f32 v170, v141, v51 :: v_dual_fmac_f32 v171, v143, v52
	scratch_load_b32 v51, off, off offset:144 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_dual_fmac_f32 v148, v141, v43 :: v_dual_fmac_f32 v149, v143, v44
	scratch_load_b32 v43, off, off offset:96 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_dual_fmac_f32 v184, v137, v57 :: v_dual_fmac_f32 v185, v139, v58
	v_dual_fmac_f32 v150, v137, v41 :: v_dual_fmac_f32 v157, v139, v42
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v42, off, off offset:92 th:TH_LOAD_LU
	scratch_load_b32 v41, off, off offset:88 th:TH_LOAD_LU
	s_wait_loadcnt 0x6
	v_fmac_f32_e32 v181, v188, v62
	v_fmac_f32_e32 v147, v188, v46
	s_wait_loadcnt 0x5
	v_fmac_f32_e32 v145, v191, v48
	scratch_load_b32 v48, off, off offset:112 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v169, v188, v54
	v_dual_fmac_f32 v181, v189, v173 :: v_dual_fmac_f32 v170, v142, v174
	v_dual_fmac_f32 v150, v138, v155 :: v_dual_fmac_f32 v167, v191, v56
	v_dual_fmac_f32 v179, v191, v64 :: v_dual_fmac_f32 v184, v138, v173
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v169, v189, v174 :: v_dual_fmac_f32 v148, v142, v155
	s_wait_dscnt 0xd
	v_dual_fmac_f32 v167, v192, v174 :: v_dual_fmac_f32 v170, v225, v115
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v150, v229, v105 :: v_dual_fmac_f32 v169, v219, v118
	v_fmac_f32_e32 v148, v225, v107
	scratch_load_b64 v[56:57], off, off offset:196 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v184, v229, v121
	v_fmac_f32_e32 v172, v138, v174
	v_fmac_f32_e32 v181, v219, v126
	v_dual_fmac_f32 v170, v226, v198 :: v_dual_fmac_f32 v169, v220, v198
	v_fmac_f32_e32 v182, v142, v173
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v172, v229, v113
	v_dual_fmac_f32 v181, v220, v197 :: v_dual_fmac_f32 v150, v230, v195
	v_fmac_f32_e32 v184, v230, v197
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v182, v225, v123
	v_fmac_f32_e32 v172, v230, v198
	v_fmac_f32_e32 v148, v226, v195
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v182, v226, v197
	s_wait_loadcnt 0x6
	v_fmac_f32_e32 v180, v59, v61
	scratch_load_b64 v[61:62], off, off offset:268 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v168, v59, v53
	v_fmac_f32_e32 v146, v59, v45
	s_clause 0x4                            ; 24-byte Folded Reload
	scratch_load_b32 v45, off, off offset:104 th:TH_LOAD_LU
	scratch_load_b64 v[52:53], off, off offset:164 th:TH_LOAD_LU
	scratch_load_b32 v49, off, off offset:116 th:TH_LOAD_LU
	scratch_load_b32 v46, off, off offset:108 th:TH_LOAD_LU
	scratch_load_b32 v44, off, off offset:100 th:TH_LOAD_LU
	s_wait_loadcnt 0xb
	v_dual_fmac_f32 v51, v137, v33 :: v_dual_fmac_f32 v168, v60, v174
	v_fmac_f32_e32 v151, v140, v174
	s_wait_dscnt 0x8
	v_dual_fmac_f32 v146, v60, v155 :: v_dual_fmac_f32 v167, v215, v120
	v_fmac_f32_e32 v180, v60, v173
	v_fmac_f32_e32 v168, v221, v117
	s_wait_loadcnt 0x7
	v_dual_fmac_f32 v48, v141, v35 :: v_dual_fmac_f32 v151, v227, v114
	v_dual_fmac_f32 v146, v221, v109 :: v_dual_fmac_f32 v167, v216, v198
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v157, v140, v155 :: v_dual_fmac_f32 v48, v142, v156
	v_fmac_f32_e32 v180, v221, v125
	v_fmac_f32_e32 v168, v222, v198
	v_dual_fmac_f32 v146, v222, v195 :: v_dual_fmac_f32 v157, v227, v106
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v48, v225, v99
	v_fmac_f32_e32 v180, v222, v197
	s_wait_loadcnt 0x6
	v_fmac_f32_e32 v160, v56, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v48, v226, v196
	v_fmac_f32_e32 v160, v57, v174
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v160, v203, v86
	v_fmac_f32_e32 v160, v204, v198
	s_wait_loadcnt 0x5
	v_fmac_f32_e32 v178, v61, v63
	scratch_load_b32 v63, off, off offset:288 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x5
	v_dual_fmac_f32 v166, v61, v55 :: v_dual_fmac_f32 v45, v59, v37
	s_wait_loadcnt 0x3
	v_dual_fmac_f32 v176, v52, v26 :: v_dual_fmac_f32 v49, v143, v36
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b64 v[54:55], off, off offset:180 th:TH_LOAD_LU
	scratch_load_b64 v[58:59], off, off offset:212 th:TH_LOAD_LU
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v255, v61, v47 :: v_dual_fmac_f32 v46, v188, v38
	v_dual_fmac_f32 v43, v61, v39 :: v_dual_fmac_f32 v164, v52, v18
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v44, v191, v40
	s_clause 0x5                            ; 24-byte Folded Reload
	scratch_load_b32 v40, off, off offset:84 th:TH_LOAD_LU
	scratch_load_b32 v39, off, off offset:80 th:TH_LOAD_LU
	scratch_load_b32 v38, off, off offset:76 th:TH_LOAD_LU
	scratch_load_b32 v37, off, off offset:72 th:TH_LOAD_LU
	scratch_load_b32 v36, off, off offset:68 th:TH_LOAD_LU
	scratch_load_b32 v47, off, off offset:64 th:TH_LOAD_LU
	v_dual_fmac_f32 v185, v140, v173 :: v_dual_fmac_f32 v178, v62, v173
	v_fmac_f32_e32 v183, v144, v173
	v_dual_fmac_f32 v149, v144, v155 :: v_dual_fmac_f32 v46, v189, v156
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v41, v52, v2 :: v_dual_fmac_f32 v178, v217, v127
	v_fmac_f32_e32 v183, v223, v124
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v149, v223, v108 :: v_dual_fmac_f32 v176, v53, v173
	v_dual_fmac_f32 v179, v192, v173 :: v_dual_fmac_f32 v166, v62, v174
	v_dual_fmac_f32 v147, v189, v155 :: v_dual_fmac_f32 v44, v192, v156
	v_dual_fmac_f32 v255, v62, v155 :: v_dual_fmac_f32 v46, v219, v102
	v_dual_fmac_f32 v145, v192, v155 :: v_dual_fmac_f32 v164, v53, v174
	v_dual_fmac_f32 v51, v138, v156 :: v_dual_fmac_f32 v176, v211, v90
	v_fmac_f32_e32 v179, v215, v128
	v_dual_fmac_f32 v171, v144, v174 :: v_dual_fmac_f32 v166, v217, v119
	v_fmac_f32_e32 v44, v215, v104
	v_dual_fmac_f32 v41, v53, v156 :: v_dual_fmac_f32 v178, v218, v197
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v176, v212, v197 :: v_dual_fmac_f32 v171, v223, v116
	v_fmac_f32_e32 v164, v211, v82
	v_dual_fmac_f32 v41, v211, v66 :: v_dual_fmac_f32 v46, v220, v196
	v_dual_fmac_f32 v166, v218, v198 :: v_dual_fmac_f32 v179, v216, v197
	v_dual_fmac_f32 v44, v216, v196 :: v_dual_fmac_f32 v185, v227, v122
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v164, v212, v198 :: v_dual_fmac_f32 v41, v212, v196
	s_wait_loadcnt 0x8
	v_fmac_f32_e32 v63, v139, v34
	scratch_load_b64 v[33:34], off, off offset:156 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v231, v56, v30
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v253, v52, v10 :: v_dual_fmac_f32 v154, v54, v28
	s_wait_loadcnt 0x7
	v_fmac_f32_e32 v158, v58, v24
	v_fmac_f32_e32 v162, v54, v20
	v_fmac_f32_e32 v49, v144, v156
	v_fmac_f32_e32 v43, v62, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v63, v140, v156 :: v_dual_fmac_f32 v158, v59, v174
	s_wait_loadcnt 0x5
	v_dual_fmac_f32 v162, v55, v174 :: v_dual_fmac_f32 v39, v54, v4
	v_dual_fmac_f32 v255, v217, v111 :: v_dual_fmac_f32 v154, v55, v173
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v37, v56, v6
	s_wait_loadcnt 0x1
	v_fmac_f32_e32 v47, v58, v8
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v158, v199, v88 :: v_dual_fmac_f32 v43, v217, v103
	v_dual_fmac_f32 v162, v207, v84 :: v_dual_fmac_f32 v51, v229, v97
	v_fmac_f32_e32 v154, v207, v92
	v_fmac_f32_e32 v47, v59, v156
	v_fmac_f32_e32 v39, v55, v156
	v_fmac_f32_e32 v255, v218, v195
	v_dual_fmac_f32 v43, v218, v196 :: v_dual_fmac_f32 v158, v200, v198
	v_dual_fmac_f32 v51, v230, v196 :: v_dual_fmac_f32 v154, v208, v197
	v_fmac_f32_e32 v247, v58, v16
	v_dual_fmac_f32 v39, v207, v68 :: v_dual_fmac_f32 v162, v208, v198
	v_add_nc_u32_e32 v229, 0x800, v190
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v247, v59, v155
	v_fmac_f32_e32 v39, v208, v196
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v247, v199, v80
	v_fmac_f32_e32 v247, v200, v195
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v177, v33, v25
	scratch_load_b64 v[25:26], off, off offset:172 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v245, v58, v32
	v_dual_fmac_f32 v251, v54, v12 :: v_dual_fmac_f32 v42, v33, v1
	v_fmac_f32_e32 v177, v34, v173
	v_dual_fmac_f32 v254, v33, v9 :: v_dual_fmac_f32 v63, v227, v98
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v165, v33, v17 :: v_dual_fmac_f32 v42, v34, v156
	v_dual_fmac_f32 v177, v213, v89 :: v_dual_fmac_f32 v254, v34, v155
	v_fmac_f32_e32 v253, v53, v155
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v42, v213, v65
	v_fmac_f32_e32 v185, v228, v197
	v_fmac_f32_e32 v254, v213, v73
	v_fmac_f32_e32 v37, v57, v156
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v177, v214, v197 :: v_dual_fmac_f32 v42, v214, v196
	v_dual_fmac_f32 v254, v214, v195 :: v_dual_fmac_f32 v37, v203, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v37, v204, v196
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v175, v25, v27
	scratch_load_b64 v[27:28], off, off offset:188 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v163, v25, v19
	v_fmac_f32_e32 v252, v25, v11
	v_dual_fmac_f32 v40, v25, v3 :: v_dual_fmac_f32 v175, v26, v173
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v49, v223, v100 :: v_dual_fmac_f32 v252, v26, v155
	v_dual_fmac_f32 v40, v26, v156 :: v_dual_fmac_f32 v175, v209, v91
	v_fmac_f32_e32 v245, v59, v173
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v49, v224, v196 :: v_dual_fmac_f32 v252, v209, v75
	v_dual_fmac_f32 v40, v209, v67 :: v_dual_fmac_f32 v47, v199, v72
	v_fmac_f32_e32 v151, v228, v198
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v175, v210, v197 :: v_dual_fmac_f32 v252, v210, v195
	v_dual_fmac_f32 v40, v210, v196 :: v_dual_fmac_f32 v47, v200, v196
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v153, v27, v29
	scratch_load_b64 v[29:30], off, off offset:204 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_dual_fmac_f32 v250, v27, v13 :: v_dual_fmac_f32 v163, v26, v174
	v_fmac_f32_e32 v161, v27, v21
	v_dual_fmac_f32 v38, v27, v5 :: v_dual_fmac_f32 v249, v56, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v153, v28, v173 :: v_dual_fmac_f32 v250, v28, v155
	v_dual_fmac_f32 v163, v209, v83 :: v_dual_fmac_f32 v38, v28, v156
	v_fmac_f32_e32 v251, v55, v155
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v153, v205, v93
	v_dual_fmac_f32 v245, v199, v96 :: v_dual_fmac_f32 v250, v205, v77
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v163, v210, v198 :: v_dual_fmac_f32 v38, v205, v69
	v_fmac_f32_e32 v251, v207, v76
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v157, v228, v195 :: v_dual_fmac_f32 v250, v206, v195
	v_dual_fmac_f32 v63, v228, v196 :: v_dual_fmac_f32 v38, v206, v196
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v251, v208, v195
	v_fmac_f32_e32 v153, v206, v197
	v_dual_fmac_f32 v245, v200, v197 :: v_dual_mov_b32 v228, v243
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v152, v29, v31 :: v_dual_fmac_f32 v45, v60, v156
	v_fmac_f32_e32 v159, v29, v23
	v_fmac_f32_e32 v248, v29, v15
	v_fmac_f32_e32 v231, v57, v173
	v_dual_fmac_f32 v36, v29, v7 :: v_dual_fmac_f32 v147, v219, v110
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v152, v30, v173 :: v_dual_fmac_f32 v159, v30, v174
	v_fmac_f32_e32 v145, v215, v112
	v_dual_fmac_f32 v161, v28, v174 :: v_dual_fmac_f32 v36, v30, v156
	v_dual_fmac_f32 v248, v30, v155 :: v_dual_fmac_f32 v231, v203, v94
	v_fmac_f32_e32 v45, v221, v101
	v_dual_fmac_f32 v152, v201, v95 :: v_dual_fmac_f32 v165, v34, v174
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v248, v201, v79
	v_fmac_f32_e32 v161, v205, v85
	v_dual_fmac_f32 v253, v211, v74 :: v_dual_fmac_f32 v36, v201, v71
	v_fmac_f32_e32 v145, v216, v195
	v_fmac_f32_e32 v165, v213, v81
	v_fmac_f32_e32 v159, v201, v87
	v_fmac_f32_e32 v249, v57, v155
	v_dual_fmac_f32 v183, v224, v197 :: v_dual_fmac_f32 v152, v202, v197
	v_dual_fmac_f32 v171, v224, v198 :: v_dual_fmac_f32 v248, v202, v195
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v249, v203, v78
	v_dual_fmac_f32 v149, v224, v195 :: v_dual_fmac_f32 v36, v202, v196
	v_fmac_f32_e32 v45, v222, v196
	v_fmac_f32_e32 v147, v220, v195
	v_fmac_f32_e32 v165, v214, v198
	v_fmac_f32_e32 v253, v212, v195
	v_fmac_f32_e32 v161, v206, v198
	v_fmac_f32_e32 v231, v204, v197
	v_fmac_f32_e32 v249, v204, v195
	v_fmac_f32_e32 v159, v202, v198
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b64 v[103:104], off, off offset:120 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x0
	s_clause 0x1                            ; 12-byte Folded Reload
	scratch_load_b32 v103, off, off offset:284 th:TH_LOAD_LU
	scratch_load_b64 v[209:210], off, off offset:296
	s_wait_loadcnt 0x0
	s_clause 0xa                            ; 60-byte Folded Reload
	scratch_load_b64 v[210:211], off, off offset:304
	scratch_load_b64 v[101:102], off, off offset:312
	scratch_load_b64 v[91:92], off, off offset:128
	scratch_load_b64 v[85:86], off, off offset:136
	scratch_load_b32 v215, off, off offset:336
	scratch_load_b32 v216, off, off offset:340
	scratch_load_b32 v217, off, off offset:344
	scratch_load_b32 v218, off, off offset:348
	scratch_load_b32 v230, off, off offset:360
	scratch_load_b32 v87, off, off offset:148
	scratch_load_b32 v88, off, off offset:152
	s_cbranch_scc1 .LBB0_14
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s14, 0xffffff00
	v_lshlrev_b64_e32 v[1:2], 2, v[103:104]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[20:21], s[6:7]
	s_clause 0xf                            ; 64-byte Folded Spill
	scratch_store_b32 off, v63, off offset:288
	scratch_store_b32 off, v51, off offset:144
	scratch_store_b32 off, v49, off offset:116
	scratch_store_b32 off, v48, off offset:112
	scratch_store_b32 off, v46, off offset:108
	scratch_store_b32 off, v45, off offset:104
	scratch_store_b32 off, v44, off offset:100
	scratch_store_b32 off, v43, off offset:96
	scratch_store_b32 off, v42, off offset:92
	scratch_store_b32 off, v41, off offset:88
	scratch_store_b32 off, v40, off offset:84
	scratch_store_b32 off, v39, off offset:80
	scratch_store_b32 off, v38, off offset:76
	scratch_store_b32 off, v37, off offset:72
	scratch_store_b32 off, v36, off offset:68
	scratch_store_b32 off, v47, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v77, s6, s34, v209
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v78, null, s35, 0, s6
	s_wait_loadcnt 0xa
	v_add_co_u32 v79, s6, s34, v210
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v80, null, s35, 0, s6
	s_add_co_i32 s6, s19, 0xffffff78
	s_wait_loadcnt 0x6
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v81, vcc_lo, v215, s6
	s_wait_loadcnt 0x5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, 0, v216, vcc_lo
	s_wait_loadcnt 0x4
	v_add_co_u32 v83, vcc_lo, v217, s6
	s_wait_loadcnt 0x3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, 0, v218, vcc_lo
	v_add_co_u32 v1, vcc_lo, s22, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s23, v2, vcc_lo
	v_add_co_u32 v3, vcc_lo, v101, s6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v102, vcc_lo
	s_clause 0x1
	global_load_b128 v[141:144], v[77:78], off offset:128
	global_load_b128 v[137:140], v[79:80], off offset:128
	global_load_b64 v[193:194], v[81:82], off offset:72
	global_load_b64 v[191:192], v[83:84], off offset:72
	global_load_b32 v246, v[3:4], off offset:4
	global_load_b32 v105, v[1:2], off offset:4
	ds_load_2addr_b64 v[65:68], v244 offset1:144
	ds_load_2addr_b64 v[1:4], v190 offset1:144
	ds_load_2addr_b64 v[69:72], v229 offset0:32 offset1:176
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
	ds_load_2addr_b64 v[65:68], v244 offset0:2 offset1:146
	ds_load_2addr_b64 v[69:72], v190 offset0:2 offset1:146
	ds_load_2addr_b64 v[73:76], v229 offset0:34 offset1:178
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
	ds_load_2addr_b64 v[65:68], v244 offset0:4 offset1:148
	ds_load_2addr_b64 v[69:72], v190 offset0:4 offset1:148
	ds_load_2addr_b64 v[73:76], v229 offset0:36 offset1:180
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
	ds_load_2addr_b64 v[65:68], v244 offset0:6 offset1:150
	ds_load_2addr_b64 v[69:72], v190 offset0:6 offset1:150
	ds_load_2addr_b64 v[73:76], v229 offset0:38 offset1:182
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
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cndmask_b32_e64 v65, 0, v85, s2
	v_cndmask_b32_e64 v74, 0, v86, s2
	v_cndmask_b32_e64 v89, 0, v91, s3
	v_cndmask_b32_e64 v68, 0, v134, s0
	v_cndmask_b32_e64 v67, 0, v135, s0
	v_lshrrev_b32_e32 v66, 4, v65
	v_and_b32_e32 v71, 0xf0f0f0f, v65
	v_lshrrev_b32_e32 v91, 4, v89
	v_cndmask_b32_e64 v65, 0, v133, s0
	v_cndmask_b32_e64 v70, 0, v136, s1
	v_and_b32_e32 v72, 0xf0f0f0f, v66
	v_cndmask_b32_e64 v66, 0, v87, s0
	v_cndmask_b32_e64 v69, 0, v129, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off           ; 4-byte Folded Reload
	v_perm_b32 v73, v72, v71, 0x5010400
	v_perm_b32 v76, v72, v71, 0x7030602
	v_lshrrev_b32_e32 v71, 4, v74
	v_and_b32_e32 v74, 0xf0f0f0f, v74
	ds_store_2addr_b64 v239, v[65:66], v[67:68] offset1:1
	v_and_b32_e32 v75, 0x7070707, v73
	v_lshrrev_b32_e32 v72, 1, v73
	v_and_b32_e32 v85, 0xf0f0f0f, v71
	v_and_b32_e32 v87, 0x7070707, v76
	v_cndmask_b32_e64 v71, 0, v131, s1
	v_perm_b32 v73, s26, 0x44403800, v75
	v_or_b32_e32 v75, 0x50505050, v75
	v_and_or_b32 v86, v72, s31, 0x3020100
	v_cndmask_b32_e64 v72, 0, v88, s1
	v_perm_b32 v88, v85, v74, 0x5010400
	v_perm_b32 v85, v85, v74, 0x7030602
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v73, v75, v73, v86
	v_lshrrev_b32_e32 v75, 1, v76
	v_perm_b32 v76, s26, 0x44403800, v87
	v_or_b32_e32 v86, 0x50505050, v87
	v_and_b32_e32 v87, 0x7070707, v88
	v_lshrrev_b32_e32 v88, 1, v88
	v_and_or_b32 v75, v75, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_perm_b32 v90, s26, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v88, v88, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v74, v86, v76, v75
	v_and_b32_e32 v76, 0xf0f0f0f, v89
	v_and_b32_e32 v86, 0xf0f0f0f, v91
	v_and_b32_e32 v89, 0x7070707, v85
	v_perm_b32 v75, v87, v90, v88
	v_cndmask_b32_e64 v87, 0, v92, s3
	v_lshrrev_b32_e32 v85, 1, v85
	v_perm_b32 v88, v86, v76, 0x5010400
	v_perm_b32 v76, v86, v76, 0x7030602
	v_perm_b32 v90, s26, 0x44403800, v89
	v_lshrrev_b32_e32 v91, 4, v87
	v_and_b32_e32 v86, 0xf0f0f0f, v87
	v_and_b32_e32 v92, 0x7070707, v88
	v_and_b32_e32 v93, 0x7070707, v76
	v_lshrrev_b32_e32 v76, 1, v76
	v_and_b32_e32 v87, 0xf0f0f0f, v91
	v_lshrrev_b32_e32 v88, 1, v88
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v85, v85, s31, 0x3020100
	v_and_or_b32 v96, v76, s31, 0x3020100
	v_perm_b32 v94, v87, v86, 0x5010400
	v_perm_b32 v86, v87, v86, 0x7030602
	v_perm_b32 v91, s26, 0x44403800, v92
	v_or_b32_e32 v92, 0x50505050, v92
	v_and_or_b32 v88, v88, s31, 0x3020100
	v_and_b32_e32 v87, 0x7070707, v94
	v_lshrrev_b32_e32 v76, 1, v94
	v_and_b32_e32 v94, 0x7070707, v86
	v_lshrrev_b32_e32 v86, 1, v86
	v_perm_b32 v95, s26, 0x44403800, v93
	v_or_b32_e32 v93, 0x50505050, v93
	v_perm_b32 v97, s26, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v98, v76, s31, 0x3020100
	v_perm_b32 v99, s26, 0x44403800, v94
	v_or_b32_e32 v94, 0x50505050, v94
	v_and_or_b32 v100, v86, s31, 0x3020100
	v_perm_b32 v76, v89, v90, v85
	v_perm_b32 v85, v92, v91, v88
	v_perm_b32 v86, v93, v95, v96
	v_perm_b32 v87, v87, v97, v98
	v_perm_b32 v88, v94, v99, v100
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v0, v[69:70], v[71:72] offset0:64 offset1:65
	ds_store_2addr_b64 v241, v[73:74], v[75:76] offset1:1
	ds_store_2addr_b64 v242, v[85:86], v[87:88] offset1:1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_clause 0x1
	global_load_b128 v[133:136], v[77:78], off offset:192
	global_load_b128 v[129:132], v[79:80], off offset:192
	global_load_b64 v[213:214], v[81:82], off offset:104
	global_load_b64 v[211:212], v[83:84], off offset:104
	ds_load_2addr_b64 v[65:68], v244 offset1:144
	ds_load_2addr_b64 v[69:72], v190 offset1:144
	ds_load_2addr_b64 v[73:76], v229 offset0:32 offset1:176
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
	ds_load_2addr_b64 v[65:68], v244 offset0:2 offset1:146
	ds_load_2addr_b64 v[69:72], v190 offset0:2 offset1:146
	ds_load_2addr_b64 v[73:76], v229 offset0:34 offset1:178
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
	ds_load_2addr_b64 v[65:68], v244 offset0:4 offset1:148
	ds_load_2addr_b64 v[69:72], v190 offset0:4 offset1:148
	ds_load_2addr_b64 v[73:76], v229 offset0:36 offset1:180
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
	ds_load_2addr_b64 v[65:68], v244 offset0:6 offset1:150
	ds_load_2addr_b64 v[69:72], v190 offset0:6 offset1:150
	ds_load_2addr_b64 v[73:76], v229 offset0:38 offset1:182
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
	ds_load_2addr_b32 v[173:174], v230 offset1:16
	ds_load_2addr_b32 v[155:156], v230 offset0:32 offset1:48
	ds_load_2addr_b32 v[65:66], v228 offset1:1
	v_dual_mov_b32 v67, v138 :: v_dual_mov_b32 v68, v140
	s_cmp_lt_i32 s18, s16
	s_cselect_b32 s6, -1, 0
	s_cmp_ge_i32 s18, s16
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:220 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:228 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:8 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:236 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:12 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:244 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:16 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:252 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:20 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:260 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:268 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:28 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:276 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:156 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:164 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:172 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:180 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:188 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:196 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:204 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:212 ; 8-byte Folded Spill
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
	v_cndmask_b32_e64 v70, 0, v65, s0
	v_cndmask_b32_e64 v69, 0, v141, s0
	v_cndmask_b32_e64 v72, 0, v66, s0
	v_cndmask_b32_e64 v71, 0, v143, s0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v239, v[69:70], v[71:72] offset1:1
	v_cndmask_b32_e64 v70, 0, v67, s1
	v_cndmask_b32_e64 v69, 0, v137, s1
	v_cndmask_b32_e64 v72, 0, v68, s1
	v_cndmask_b32_e64 v71, 0, v139, s1
	ds_store_2addr_b64 v0, v[69:70], v[71:72] offset0:64 offset1:65
	v_cndmask_b32_e64 v69, 0, v193, s2
	v_add_nc_u32_e32 v0, 2, v103
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v70, 0xf0f0f0f, v69
	v_lshrrev_b32_e32 v69, 4, v69
	v_and_b32_e32 v69, 0xf0f0f0f, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v71, v69, v70, 0x5010400
	v_perm_b32 v70, v69, v70, 0x7030602
	v_and_b32_e32 v69, 0x7070707, v71
	v_lshrrev_b32_e32 v71, 1, v71
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v72, s26, 0x44403800, v69
	v_or_b32_e32 v69, 0x50505050, v69
	v_and_or_b32 v71, v71, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v69, v69, v72, v71
	v_and_b32_e32 v71, 0x7070707, v70
	v_lshrrev_b32_e32 v70, 1, v70
	v_perm_b32 v72, s26, 0x44403800, v71
	v_or_b32_e32 v71, 0x50505050, v71
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v70, v70, s31, 0x3020100
	v_perm_b32 v70, v71, v72, v70
	v_cndmask_b32_e64 v71, 0, v194, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v72, 0xf0f0f0f, v71
	v_lshrrev_b32_e32 v71, 4, v71
	v_and_b32_e32 v71, 0xf0f0f0f, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v73, v71, v72, 0x5010400
	v_perm_b32 v72, v71, v72, 0x7030602
	v_and_b32_e32 v71, 0x7070707, v73
	v_lshrrev_b32_e32 v73, 1, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v74, s26, 0x44403800, v71
	v_or_b32_e32 v71, 0x50505050, v71
	v_and_or_b32 v73, v73, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v71, v71, v74, v73
	v_and_b32_e32 v73, 0x7070707, v72
	v_lshrrev_b32_e32 v72, 1, v72
	v_perm_b32 v74, s26, 0x44403800, v73
	v_or_b32_e32 v73, 0x50505050, v73
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v72, v72, s31, 0x3020100
	v_perm_b32 v72, v73, v74, v72
	ds_store_2addr_b64 v241, v[69:70], v[71:72] offset1:1
	v_cndmask_b32_e64 v69, 0, v191, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v70, 0xf0f0f0f, v69
	v_lshrrev_b32_e32 v69, 4, v69
	v_and_b32_e32 v69, 0xf0f0f0f, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v71, v69, v70, 0x5010400
	v_perm_b32 v70, v69, v70, 0x7030602
	v_and_b32_e32 v69, 0x7070707, v71
	v_lshrrev_b32_e32 v71, 1, v71
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v72, s26, 0x44403800, v69
	v_or_b32_e32 v69, 0x50505050, v69
	v_and_or_b32 v71, v71, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v69, v69, v72, v71
	v_and_b32_e32 v71, 0x7070707, v70
	v_lshrrev_b32_e32 v70, 1, v70
	v_perm_b32 v72, s26, 0x44403800, v71
	v_or_b32_e32 v71, 0x50505050, v71
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v70, v70, s31, 0x3020100
	v_perm_b32 v70, v71, v72, v70
	v_cndmask_b32_e64 v71, 0, v192, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v72, 0xf0f0f0f, v71
	v_lshrrev_b32_e32 v71, 4, v71
	v_and_b32_e32 v71, 0xf0f0f0f, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v73, v71, v72, 0x5010400
	v_perm_b32 v72, v71, v72, 0x7030602
	v_and_b32_e32 v71, 0x7070707, v73
	v_lshrrev_b32_e32 v73, 1, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v74, s26, 0x44403800, v71
	v_or_b32_e32 v71, 0x50505050, v71
	v_and_or_b32 v73, v73, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v71, v71, v74, v73
	v_and_b32_e32 v73, 0x7070707, v72
	v_lshrrev_b32_e32 v72, 1, v72
	v_perm_b32 v74, s26, 0x44403800, v73
	v_or_b32_e32 v73, 0x50505050, v73
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v72, v72, s31, 0x3020100
	v_perm_b32 v72, v73, v74, v72
	ds_store_2addr_b64 v242, v[69:70], v[71:72] offset1:1
	v_cndmask_b32_e64 v69, 0, v105, s4
	ds_store_b32 v187, v69 offset:18432
	v_lshrrev_b32_e32 v69, v234, v246
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v69, v69.l
	v_cndmask_b32_e64 v69, 0, v69, s5
	ds_store_b32 v238, v69 offset:18944
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_store_b32 off, v0, off offset:284 ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_6 Depth=1
	v_add_co_u32 v69, vcc_lo, v215, s19
	v_add_nc_u32_e32 v103, 2, v103
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, 0, v216, vcc_lo
	v_mov_b32_e32 v71, v104
	s_mov_b32 s15, s7
	v_add_co_u32 v73, vcc_lo, v217, s19
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[20:21], s[14:15]
	scratch_store_b64 off, v[70:71], off offset:120 ; 8-byte Folded Spill
	v_lshlrev_b64_e32 v[71:72], 2, v[103:104]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, 0, v218, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v65, s15, s34, v209
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v66, null, s35, 0, s15
	v_add_co_u32 v71, vcc_lo, s22, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, s23, v72, vcc_lo
	v_add_co_u32 v75, vcc_lo, v101, s19
	v_add_co_u32 v67, s15, s34, v210
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, 0, v102, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v68, null, s35, 0, s15
	s_clause 0x1
	global_load_b128 v[141:144], v[65:66], off
	global_load_b128 v[137:140], v[67:68], off
	global_load_b64 v[193:194], v[69:70], off offset:8
	global_load_b64 v[191:192], v[73:74], off offset:8
	global_load_b32 v0, v[71:72], off
	global_load_b32 v246, v[75:76], off
	s_wait_loadcnt 0x1
	scratch_store_b32 off, v0, off offset:292 ; 4-byte Folded Spill
	s_branch .LBB0_9
.LBB0_8:                                ;   in Loop: Header=BB0_6 Depth=1
	s_clause 0x1                            ; 12-byte Folded Spill
	scratch_store_b32 off, v105, off offset:292
	scratch_store_b64 off, v[103:104], off offset:120
	v_mov_b32_e32 v142, v65
	v_mov_b32_e32 v144, v66
	v_mov_b32_e32 v138, v67
	v_mov_b32_e32 v140, v68
.LBB0_9:                                ;   in Loop: Header=BB0_6 Depth=1
	ds_load_2addr_b64 v[195:198], v244 offset1:144
	ds_load_2addr_b64 v[65:68], v190 offset1:144
	ds_load_2addr_b64 v[199:202], v229 offset0:32 offset1:176
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
	ds_load_2addr_b64 v[195:198], v244 offset0:2 offset1:146
	ds_load_2addr_b64 v[199:202], v190 offset0:2 offset1:146
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[195:196], v[199:200], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[197:198], v[199:200], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[195:196], v[201:202], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[197:198], v[201:202], v[81:88]
	ds_load_2addr_b64 v[199:202], v229 offset0:34 offset1:178
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[195:196], v[199:200], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[197:198], v[199:200], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[195:196], v[201:202], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[197:198], v[201:202], v[65:72]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[195:198], v244 offset0:4 offset1:148
	ds_load_2addr_b64 v[199:202], v190 offset0:4 offset1:148
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[195:196], v[199:200], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[197:198], v[199:200], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[195:196], v[201:202], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[197:198], v[201:202], v[81:88]
	ds_load_2addr_b64 v[199:202], v229 offset0:36 offset1:180
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[195:196], v[199:200], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[197:198], v[199:200], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[195:196], v[201:202], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[197:198], v[201:202], v[65:72]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[195:198], v244 offset0:6 offset1:150
	ds_load_2addr_b64 v[199:202], v190 offset0:6 offset1:150
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[195:196], v[199:200], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[197:198], v[199:200], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[195:196], v[201:202], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[197:198], v[201:202], v[81:88]
	ds_load_2addr_b64 v[199:202], v229 offset0:38 offset1:182
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
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cndmask_b32_e64 v130, 0, v213, s2
	v_cndmask_b32_e64 v188, 0, v214, s2
	v_cndmask_b32_e64 v187, 0, v219, s0
	v_cndmask_b32_e64 v186, 0, v133, s0
	v_cndmask_b32_e64 v206, 0, v220, s1
	v_lshrrev_b32_e32 v132, 4, v130
	v_and_b32_e32 v130, 0xf0f0f0f, v130
	v_lshrrev_b32_e32 v195, 4, v188
	v_and_b32_e32 v188, 0xf0f0f0f, v188
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	v_and_b32_e32 v132, 0xf0f0f0f, v132
	v_and_b32_e32 v196, 0xf0f0f0f, v195
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off           ; 4-byte Folded Reload
	v_perm_b32 v189, v132, v130, 0x5010400
	v_perm_b32 v130, v132, v130, 0x7030602
	v_perm_b32 v199, v196, v188, 0x5010400
	v_perm_b32 v188, v196, v188, 0x7030602
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v132, 0x7070707, v189
	v_lshrrev_b32_e32 v189, 1, v189
	v_and_b32_e32 v197, 0x7070707, v130
	v_lshrrev_b32_e32 v130, 1, v130
	v_perm_b32 v195, s26, 0x44403800, v132
	v_or_b32_e32 v132, 0x50505050, v132
	v_and_or_b32 v198, v189, s31, 0x3020100
	v_perm_b32 v200, s26, 0x44403800, v197
	v_and_or_b32 v130, v130, s31, 0x3020100
	v_cndmask_b32_e64 v189, 0, v134, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_perm_b32 v195, v132, v195, v198
	v_or_b32_e32 v132, 0x50505050, v197
	v_and_b32_e32 v197, 0x7070707, v199
	v_cndmask_b32_e64 v198, 0, v211, s3
	v_lshrrev_b32_e32 v199, 1, v199
	v_perm_b32 v196, v132, v200, v130
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_perm_b32 v201, s26, 0x44403800, v197
	v_lshrrev_b32_e32 v202, 4, v198
	v_or_b32_e32 v197, 0x50505050, v197
	v_and_or_b32 v199, v199, s31, 0x3020100
	v_and_b32_e32 v130, 0xf0f0f0f, v198
	v_and_b32_e32 v198, 0x7070707, v188
	v_and_b32_e32 v132, 0xf0f0f0f, v202
	v_lshrrev_b32_e32 v188, 1, v188
	v_perm_b32 v197, v197, v201, v199
	v_cndmask_b32_e64 v199, 0, v212, s3
	v_perm_b32 v201, s26, 0x44403800, v198
	v_perm_b32 v200, v132, v130, 0x5010400
	v_or_b32_e32 v198, 0x50505050, v198
	v_and_or_b32 v188, v188, s31, 0x3020100
	v_lshrrev_b32_e32 v202, 4, v199
	v_perm_b32 v130, v132, v130, 0x7030602
	v_and_b32_e32 v203, 0x7070707, v200
	v_and_b32_e32 v132, 0xf0f0f0f, v199
	v_lshrrev_b32_e32 v200, 1, v200
	v_and_b32_e32 v199, 0xf0f0f0f, v202
	v_perm_b32 v198, v198, v201, v188
	v_perm_b32 v202, s26, 0x44403800, v203
	v_and_b32_e32 v188, 0x7070707, v130
	v_or_b32_e32 v201, 0x50505050, v203
	v_perm_b32 v203, v199, v132, 0x5010400
	v_lshrrev_b32_e32 v130, 1, v130
	v_and_or_b32 v200, v200, s31, 0x3020100
	v_perm_b32 v132, v199, v132, 0x7030602
	v_perm_b32 v204, s26, 0x44403800, v188
	v_or_b32_e32 v188, 0x50505050, v188
	v_and_or_b32 v130, v130, s31, 0x3020100
	v_and_b32_e32 v205, 0x7070707, v203
	v_lshrrev_b32_e32 v203, 1, v203
	v_perm_b32 v199, v201, v202, v200
	v_and_b32_e32 v201, 0x7070707, v132
	v_lshrrev_b32_e32 v132, 1, v132
	v_perm_b32 v200, v188, v204, v130
	v_perm_b32 v130, s26, 0x44403800, v205
	v_or_b32_e32 v202, 0x50505050, v205
	v_and_or_b32 v203, v203, s31, 0x3020100
	v_perm_b32 v207, s26, 0x44403800, v201
	v_or_b32_e32 v208, 0x50505050, v201
	v_and_or_b32 v132, v132, s31, 0x3020100
	v_cndmask_b32_e64 v188, 0, v135, s0
	v_perm_b32 v201, v202, v130, v203
	v_cndmask_b32_e64 v204, 0, v136, s1
	v_cndmask_b32_e64 v203, 0, v129, s1
	v_cndmask_b32_e64 v205, 0, v131, s1
	v_perm_b32 v202, v208, v207, v132
	ds_store_2addr_b64 v239, v[186:187], v[188:189] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v0, v[203:204], v[205:206] offset0:64 offset1:65
	ds_store_2addr_b64 v241, v[195:196], v[197:198] offset1:1
	ds_store_2addr_b64 v242, v[199:200], v[201:202] offset1:1
	v_cndmask_b32_e64 v195, 0, 1, s6
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_6 Depth=1
	v_add_co_u32 v186, vcc_lo, v215, s19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v187, null, 0, v216, vcc_lo
	s_mov_b32 s15, s7
	v_add_co_u32 v188, vcc_lo, v217, s19
	global_load_b64 v[186:187], v[186:187], off offset:40
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[20:21], s[14:15]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v189, null, 0, v218, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v129, s6, s34, v209
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, s35, 0, s6
	v_add_co_u32 v131, s6, s34, v210
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v132, null, s35, 0, s6
	s_clause 0x1
	global_load_b128 v[133:136], v[129:130], off offset:64
	global_load_b128 v[129:132], v[131:132], off offset:64
	s_wait_loadcnt 0x2
	scratch_store_b64 off, v[186:187], off offset:136 ; 8-byte Folded Spill
	global_load_b64 v[186:187], v[188:189], off offset:40
	s_wait_loadcnt 0x2
	v_mov_b32_e32 v0, v134
	v_mov_b32_e32 v134, v136
	s_wait_loadcnt 0x1
	v_mov_b32_e32 v136, v130
	scratch_store_b32 off, v0, off offset:148 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, v132
	scratch_store_b32 off, v0, off offset:152 ; 4-byte Folded Spill
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[186:187], off offset:128 ; 8-byte Folded Spill
	s_branch .LBB0_12
.LBB0_11:                               ;   in Loop: Header=BB0_6 Depth=1
	s_clause 0x3                            ; 24-byte Folded Spill
	scratch_store_b32 off, v220, off offset:152
	scratch_store_b32 off, v219, off offset:148
	scratch_store_b64 off, v[213:214], off offset:136
	scratch_store_b64 off, v[211:212], off offset:128
.LBB0_12:                               ;   in Loop: Header=BB0_6 Depth=1
	ds_load_2addr_b64 v[196:199], v244 offset1:144
	ds_load_2addr_b64 v[200:203], v190 offset1:144
	ds_load_2addr_b64 v[204:207], v229 offset0:32 offset1:176
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[208:211], v244 offset0:2 offset1:146
	ds_load_2addr_b64 v[212:215], v190 offset0:2 offset1:146
	ds_load_2addr_b64 v[216:219], v229 offset0:34 offset1:178
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[220:223], v244 offset0:4 offset1:148
	ds_load_2addr_b64 v[224:227], v190 offset0:4 offset1:148
	ds_load_2addr_b64 v[232:235], v229 offset0:36 offset1:180
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[236:239], v244 offset0:6 offset1:150
	ds_load_2addr_b64 v[240:243], v190 offset0:6 offset1:150
	ds_load_2addr_b64 v[186:189], v229 offset0:38 offset1:182
	; sched_barrier mask(0x00000000)
	scratch_load_b32 v132, off, off offset:4 ; 4-byte Folded Reload
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[196:197], v[202:203], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[198:199], v[202:203], v[81:88]
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[196:197], v[200:201], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[198:199], v[200:201], v[89:96]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[196:197], v[204:205], v[105:112]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[208:209], v[214:215], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[210:211], v[214:215], v[81:88]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[198:199], v[204:205], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[196:197], v[206:207], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[198:199], v[206:207], v[65:72]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[220:221], v[226:227], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[222:223], v[226:227], v[81:88]
	v_cmp_ne_u32_e32 vcc_lo, 1, v195
	v_mov_b32_e32 v130, v142
	ds_load_2addr_b32 v[197:198], v230 offset1:16
	ds_load_2addr_b32 v[195:196], v230 offset0:32 offset1:48
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[236:237], v[242:243], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[238:239], v[242:243], v[81:88]
	v_mov_b32_e32 v243, v228
	ds_load_2addr_b32 v[229:230], v228 offset1:1
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[208:209], v[212:213], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[210:211], v[212:213], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[210:211], v[216:217], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[210:211], v[218:219], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[208:209], v[216:217], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[220:221], v[224:225], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[222:223], v[224:225], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[222:223], v[232:233], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[222:223], v[234:235], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[208:209], v[218:219], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[220:221], v[232:233], v[105:112]
	v_mov_b32_e32 v142, v144
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[236:237], v[240:241], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[238:239], v[240:241], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[220:221], v[234:235], v[97:104]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[236:237], v[186:187], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[238:239], v[186:187], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[238:239], v[188:189], v[65:72]
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[236:237], v[188:189], v[97:104]
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[227:228], v132 offset1:1
	scratch_load_b32 v132, off, off offset:8 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[225:226], v132 offset1:1
	scratch_load_b32 v132, off, off offset:12 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[223:224], v132 offset1:1
	scratch_load_b32 v132, off, off offset:16 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[221:222], v132 offset1:1
	scratch_load_b32 v132, off, off offset:20 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[219:220], v132 offset1:1
	scratch_load_b32 v132, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[217:218], v132 offset1:1
	scratch_load_b32 v132, off, off offset:28 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[215:216], v132 offset1:1
	scratch_load_b32 v132, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[213:214], v132 offset1:1
	scratch_load_b32 v132, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[211:212], v132 offset1:1
	scratch_load_b32 v132, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[209:210], v132 offset1:1
	scratch_load_b32 v132, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[207:208], v132 offset1:1
	scratch_load_b32 v132, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[205:206], v132 offset1:1
	scratch_load_b32 v132, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[203:204], v132 offset1:1
	scratch_load_b32 v132, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[201:202], v132 offset1:1
	scratch_load_b32 v132, off, off offset:60 ; 4-byte Folded Reload
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
	scratch_load_b32 v240, off, off offset:292 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_clause 0x5                            ; 24-byte Folded Reload
	scratch_load_b32 v187, off, off offset:320
	scratch_load_b32 v234, off, off offset:324
	scratch_load_b32 v238, off, off offset:328
	scratch_load_b32 v239, off, off offset:332
	scratch_load_b32 v241, off, off offset:352
	scratch_load_b32 v242, off, off offset:356
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_5
; %bb.13:                               ;   in Loop: Header=BB0_6 Depth=1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cndmask_b32_e64 v144, 0, v193, s2
	v_mov_b32_e32 v186, v142
	v_cndmask_b32_e64 v188, 0, v191, s3
                                        ; kill: def $vgpr140 killed $vgpr140
	v_cndmask_b32_e64 v142, 0, v143, s0
	v_cndmask_b32_e64 v140, 0, v141, s0
	v_cndmask_b32_e64 v141, 0, v130, s0
	v_cndmask_b32_e64 v143, 0, v186, s0
	v_lshrrev_b32_e32 v189, 4, v188
	v_mov_b32_e32 v0, v187
	v_lshrrev_b32_e32 v187, 4, v144
	v_and_b32_e32 v188, 0xf0f0f0f, v188
	v_lshrrev_b32_e32 v237, v234, v246
	v_and_b32_e32 v189, 0xf0f0f0f, v189
	v_and_b32_e32 v130, 0xf0f0f0f, v144
	v_and_b32_e32 v144, 0xf0f0f0f, v187
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v239, v[140:141], v[142:143] offset1:1
	v_cndmask_b32_e64 v141, 0, v194, s2
	v_perm_b32 v186, v144, v130, 0x5010400
	v_cndmask_b32_e64 v140, 0, v132, s1
	v_perm_b32 v130, v144, v130, 0x7030602
	v_cndmask_b32_e64 v138, 0, v138, s1
	v_lshrrev_b32_e32 v132, 4, v141
	v_and_b32_e32 v142, 0x7070707, v186
	v_lshrrev_b32_e32 v143, 1, v186
	v_and_b32_e32 v186, 0xf0f0f0f, v141
	v_cndmask_b32_e64 v137, 0, v137, s1
	v_and_b32_e32 v132, 0xf0f0f0f, v132
	v_perm_b32 v144, s26, 0x44403800, v142
	v_or_b32_e32 v141, 0x50505050, v142
	v_and_or_b32 v142, v143, s31, 0x3020100
	v_and_b32_e32 v143, 0x7070707, v130
	v_perm_b32 v187, v132, v186, 0x5010400
	v_lshrrev_b32_e32 v130, 1, v130
	v_perm_b32 v132, v132, v186, 0x7030602
	v_perm_b32 v141, v141, v144, v142
	v_perm_b32 v142, s26, 0x44403800, v143
	v_and_b32_e32 v144, 0x7070707, v187
	v_lshrrev_b32_e32 v187, 1, v187
	v_or_b32_e32 v143, 0x50505050, v143
	v_and_or_b32 v130, v130, s31, 0x3020100
	v_cndmask_b32_e64 v139, 0, v139, s1
	v_perm_b32 v186, s26, 0x44403800, v144
	v_or_b32_e32 v144, 0x50505050, v144
	v_and_or_b32 v187, v187, s31, 0x3020100
	v_perm_b32 v142, v143, v142, v130
	v_and_b32_e32 v130, 0x7070707, v132
	v_lshrrev_b32_e32 v132, 1, v132
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_perm_b32 v143, v144, v186, v187
	v_cndmask_b32_e64 v144, 0, v192, s3
	v_perm_b32 v186, s26, 0x44403800, v130
	v_perm_b32 v187, v189, v188, 0x5010400
	v_or_b32_e32 v130, 0x50505050, v130
	v_and_or_b32 v132, v132, s31, 0x3020100
	v_lshrrev_b32_e32 v191, 4, v144
	v_perm_b32 v188, v189, v188, 0x7030602
	v_and_b32_e32 v192, 0xf0f0f0f, v144
	v_and_b32_e32 v189, 0x7070707, v187
	v_perm_b32 v144, v130, v186, v132
	v_and_b32_e32 v191, 0xf0f0f0f, v191
	v_lshrrev_b32_e32 v130, 1, v187
	v_and_b32_e32 v186, 0x7070707, v188
	v_perm_b32 v132, s26, 0x44403800, v189
	v_or_b32_e32 v189, 0x50505050, v189
	v_perm_b32 v187, v191, v192, 0x5010400
	v_perm_b32 v191, v191, v192, 0x7030602
	v_perm_b32 v193, s26, 0x44403800, v186
	v_or_b32_e32 v192, 0x50505050, v186
	v_and_or_b32 v130, v130, s31, 0x3020100
	v_and_b32_e32 v194, 0x7070707, v187
	v_lshrrev_b32_e32 v186, 1, v187
	v_and_b32_e32 v187, 0x7070707, v191
	v_lshrrev_b32_e32 v191, 1, v191
	v_lshrrev_b32_e32 v188, 1, v188
	v_perm_b32 v232, s26, 0x44403800, v194
	v_and_or_b32 v233, v186, s31, 0x3020100
	v_perm_b32 v235, s26, 0x44403800, v187
	v_or_b32_e32 v236, 0x50505050, v187
	v_and_or_b32 v191, v191, s31, 0x3020100
	v_perm_b32 v186, v189, v132, v130
	v_and_or_b32 v188, v188, s31, 0x3020100
	v_or_b32_e32 v194, 0x50505050, v194
	v_cvt_f32_f16_e64 v130, v237.l
	v_perm_b32 v189, v236, v235, v191
	scratch_load_b32 v191, off, off         ; 4-byte Folded Reload
	v_perm_b32 v187, v192, v193, v188
	v_perm_b32 v188, v194, v232, v233
	v_cndmask_b32_e64 v132, 0, v240, s4
	v_cndmask_b32_e64 v130, 0, v130, s5
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v191, v[137:138], v[139:140] offset0:64 offset1:65
	ds_store_2addr_b64 v241, v[141:142], v[143:144] offset1:1
	ds_store_2addr_b64 v242, v[186:187], v[188:189] offset1:1
	v_mov_b32_e32 v187, v0
	ds_store_b32 v0, v132 offset:18432
	ds_store_b32 v238, v130 offset:18944
	s_branch .LBB0_5
.LBB0_14:
	s_clause 0x4                            ; 20-byte Folded Reload
	scratch_load_b32 v0, off, off offset:364
	scratch_load_b32 v17, off, off offset:368
	scratch_load_b32 v7, off, off offset:372
	scratch_load_b32 v8, off, off offset:376
	scratch_load_b32 v10, off, off offset:380
.LBB0_15:
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v1, 6, v0
	v_and_b32_e32 v11, 31, v0
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v2, v10, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v1, 0x3800, v1
	v_or3_b32 v3, s30, v8, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v1, s11 :: v_dual_add_nc_u32 v4, 0, v1
	v_add_nc_u32_e32 v5, 17, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s0, s24, v3
	v_add_nc_u32_e32 v6, 18, v2
	v_add_nc_u32_e32 v12, 19, v2
	v_add_nc_u32_e32 v13, 22, v2
	v_and_b32_e32 v5, 31, v5
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v50, s13, v1, s0
	v_mov_b32_e32 v1, s10
	v_lshl_add_u32 v16, v7, 7, v4
	v_and_b32_e32 v6, 31, v6
	v_xor_b32_e32 v8, 16, v2
	v_and_b32_e32 v12, 31, v12
	v_and_b32_e32 v15, 31, v13
	v_lshl_add_u32 v9, v5, 2, v16
	v_add_nc_u32_e32 v5, 20, v2
	v_lshl_add_u32 v10, v6, 2, v16
	v_add_nc_u32_e32 v6, 21, v2
	v_lshl_add_u32 v7, v2, 2, v16
	v_add_nc_u32_e32 v2, 23, v2
	v_and_b32_e32 v5, 31, v5
	v_lshl_add_u32 v12, v12, 2, v16
	v_and_b32_e32 v6, 31, v6
	v_lshl_add_u32 v15, v15, 2, v16
	v_and_b32_e32 v2, 31, v2
	v_lshl_add_u32 v13, v5, 2, v16
	v_lshl_add_u32 v8, v8, 2, v16
	v_lshl_add_u32 v14, v6, 2, v16
	ds_store_2addr_b32 v7, v184, v185 offset1:1
	ds_store_2addr_b32 v7, v182, v183 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v180, v181 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v178, v179 offset0:6 offset1:7
	ds_store_b32 v8, v177
	ds_store_b32 v9, v176
	ds_store_b32 v10, v175
	v_lshl_add_u32 v16, v2, 2, v16
	ds_store_b32 v12, v154
	ds_store_b32 v13, v153
	ds_store_b32 v14, v231
	ds_store_b32 v15, v152
	ds_store_b32 v16, v245
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v2, s28, v17
	v_cndmask_b32_e64 v6, s24, 0, s0
	v_mov_b32_e32 v17, s24
	v_cmp_gt_i32_e64 s1, s29, v3
	v_cndmask_b32_e64 v5, s12, v1, s0
	v_cmp_le_i32_e32 vcc_lo, s29, v3
	v_sub_nc_u32_e32 v1, v3, v6
	v_cndmask_b32_e64 v6, s25, v17, s0
	v_cmp_gt_i32_e64 s0, s27, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u32 v11, v11, 2, v4
	s_and_b32 s0, s0, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_17
; %bb.16:
	v_lshlrev_b64_e32 v[17:18], 2, v[2:3]
	ds_load_b32 v20, v11
	v_add_co_u32 v17, s0, s8, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, s9, v18, s0
	global_load_b32 v19, v[17:18], off
	v_mad_co_u64_u32 v[17:18], null, v2, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v18, 0 :: v_dual_mul_f32 v19, v19, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s0, v5, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v50, v18, s0
	global_store_b32 v[17:18], v19, off
.LBB0_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v18, 1, v2
	v_add_nc_u32_e32 v17, 1, v0
	s_xor_b32 s1, vcc_lo, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s0, s27, v18
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_19
; %bb.18:
	v_lshlrev_b64_e32 v[19:20], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v19, vcc_lo, s8, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, s9, v20, vcc_lo
	global_load_b32 v20, v[19:20], off offset:4
	v_and_b32_e32 v19, 31, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v19, v19, 2, v4
	ds_load_b32 v21, v19 offset:128
	v_mad_co_u64_u32 v[18:19], null, v18, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v19, 0 :: v_dual_mul_f32 v20, v20, v21
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc_lo, v5, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v50, v19, vcc_lo
	global_store_b32 v[18:19], v20, off
.LBB0_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v19, 2, v2
	v_add_nc_u32_e32 v18, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v19
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_21
; %bb.20:
	v_lshlrev_b64_e32 v[20:21], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, s8, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s9, v21, vcc_lo
	global_load_b32 v21, v[20:21], off offset:8
	v_and_b32_e32 v20, 31, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v20, v20, 2, v4
	ds_load_b32 v22, v20 offset:256
	v_mad_co_u64_u32 v[19:20], null, v19, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v20, 0 :: v_dual_mul_f32 v21, v21, v22
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v19, vcc_lo, v5, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v50, v20, vcc_lo
	global_store_b32 v[19:20], v21, off
.LBB0_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v20, 3, v2
	v_add_nc_u32_e32 v19, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v20
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_23
; %bb.22:
	v_lshlrev_b64_e32 v[21:22], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v21, vcc_lo, s8, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, s9, v22, vcc_lo
	global_load_b32 v22, v[21:22], off offset:12
	v_and_b32_e32 v21, 31, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v21, v21, 2, v4
	ds_load_b32 v23, v21 offset:384
	v_mad_co_u64_u32 v[20:21], null, v20, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v21, 0 :: v_dual_mul_f32 v22, v22, v23
	v_lshlrev_b64_e32 v[20:21], 2, v[20:21]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, v5, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, v50, v21, vcc_lo
	global_store_b32 v[20:21], v22, off
.LBB0_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v21, 4, v2
	v_add_nc_u32_e32 v20, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v21
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_25
; %bb.24:
	v_lshlrev_b64_e32 v[22:23], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v22, vcc_lo, s8, v22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v23, null, s9, v23, vcc_lo
	global_load_b32 v23, v[22:23], off offset:16
	v_and_b32_e32 v22, 31, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v22, v22, 2, v4
	ds_load_b32 v24, v22 offset:512
	v_mad_co_u64_u32 v[21:22], null, v21, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v22, 0 :: v_dual_mul_f32 v23, v23, v24
	v_lshlrev_b64_e32 v[21:22], 2, v[21:22]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v21, vcc_lo, v5, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, v50, v22, vcc_lo
	global_store_b32 v[21:22], v23, off
.LBB0_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v22, 5, v2
	v_add_nc_u32_e32 v21, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v22
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_lshlrev_b64_e32 v[23:24], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v23, vcc_lo, s8, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, s9, v24, vcc_lo
	global_load_b32 v24, v[23:24], off offset:20
	v_and_b32_e32 v23, 31, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v23, v23, 2, v4
	ds_load_b32 v25, v23 offset:640
	v_mad_co_u64_u32 v[22:23], null, v22, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v23, 0 :: v_dual_mul_f32 v24, v24, v25
	v_lshlrev_b64_e32 v[22:23], 2, v[22:23]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v22, vcc_lo, v5, v22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v23, null, v50, v23, vcc_lo
	global_store_b32 v[22:23], v24, off
.LBB0_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v23, 6, v2
	v_add_nc_u32_e32 v22, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v23
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_lshlrev_b64_e32 v[24:25], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v24, vcc_lo, s8, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, s9, v25, vcc_lo
	global_load_b32 v25, v[24:25], off offset:24
	v_and_b32_e32 v24, 31, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v24, v24, 2, v4
	ds_load_b32 v26, v24 offset:768
	v_mad_co_u64_u32 v[23:24], null, v23, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v24, 0 :: v_dual_mul_f32 v25, v25, v26
	v_lshlrev_b64_e32 v[23:24], 2, v[23:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v23, vcc_lo, v5, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, v50, v24, vcc_lo
	global_store_b32 v[23:24], v25, off
.LBB0_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v24, 7, v2
	v_add_nc_u32_e32 v23, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v24
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_lshlrev_b64_e32 v[25:26], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s8, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s9, v26, vcc_lo
	global_load_b32 v26, v[25:26], off offset:28
	v_and_b32_e32 v25, 31, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v25, v25, 2, v4
	ds_load_b32 v27, v25 offset:896
	v_mad_co_u64_u32 v[24:25], null, v24, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v25, 0 :: v_dual_mul_f32 v26, v26, v27
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v24, vcc_lo, v5, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v50, v25, vcc_lo
	global_store_b32 v[24:25], v26, off
.LBB0_31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 8, v2
	v_add_nc_u32_e32 v24, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_33
; %bb.32:
	v_lshlrev_b64_e32 v[26:27], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v27, v[26:27], off offset:32
	v_and_b32_e32 v26, 31, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v28, v26 offset:1024
	v_mad_co_u64_u32 v[25:26], null, v25, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v26, 0 :: v_dual_mul_f32 v27, v27, v28
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v5, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v50, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 9, v2
	v_add_nc_u32_e32 v25, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v26
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_35
; %bb.34:
	v_lshlrev_b64_e32 v[27:28], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v27, vcc_lo, s8, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s9, v28, vcc_lo
	global_load_b32 v28, v[27:28], off offset:36
	v_and_b32_e32 v27, 31, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v27, v27, 2, v4
	ds_load_b32 v29, v27 offset:1152
	v_mad_co_u64_u32 v[26:27], null, v26, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v27, 0 :: v_dual_mul_f32 v28, v28, v29
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, v5, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v50, v27, vcc_lo
	global_store_b32 v[26:27], v28, off
.LBB0_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v27, 10, v2
	v_add_nc_u32_e32 v26, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v27
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_37
; %bb.36:
	v_lshlrev_b64_e32 v[28:29], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, s8, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s9, v29, vcc_lo
	global_load_b32 v29, v[28:29], off offset:40
	v_and_b32_e32 v28, 31, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v28, v28, 2, v4
	ds_load_b32 v30, v28 offset:1280
	v_mad_co_u64_u32 v[27:28], null, v27, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v28, 0 :: v_dual_mul_f32 v29, v29, v30
	v_lshlrev_b64_e32 v[27:28], 2, v[27:28]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v27, vcc_lo, v5, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, v50, v28, vcc_lo
	global_store_b32 v[27:28], v29, off
.LBB0_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 11, v2
	v_add_nc_u32_e32 v27, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_39
; %bb.38:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s8, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s9, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:44
	v_and_b32_e32 v29, 31, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1408
	v_mad_co_u64_u32 v[28:29], null, v28, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v29, 0 :: v_dual_mul_f32 v30, v30, v31
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v5, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v50, v29, vcc_lo
	global_store_b32 v[28:29], v30, off
.LBB0_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v29, 12, v2
	v_add_nc_u32_e32 v28, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v29
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_41
; %bb.40:
	v_lshlrev_b64_e32 v[30:31], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v31, v[30:31], off offset:48
	v_and_b32_e32 v30, 31, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v30, 2, v4
	ds_load_b32 v32, v30 offset:1536
	v_mad_co_u64_u32 v[29:30], null, v29, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v30, 0 :: v_dual_mul_f32 v31, v31, v32
	v_lshlrev_b64_e32 v[29:30], 2, v[29:30]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, v5, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v50, v30, vcc_lo
	global_store_b32 v[29:30], v31, off
.LBB0_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v30, 13, v2
	v_add_nc_u32_e32 v29, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v30
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_43
; %bb.42:
	v_lshlrev_b64_e32 v[31:32], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v31, vcc_lo, s8, v31
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v32, null, s9, v32, vcc_lo
	global_load_b32 v32, v[31:32], off offset:52
	v_and_b32_e32 v31, 31, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v31, v31, 2, v4
	ds_load_b32 v33, v31 offset:1664
	v_mad_co_u64_u32 v[30:31], null, v30, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v31, 0 :: v_dual_mul_f32 v32, v32, v33
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v30, vcc_lo, v5, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, v50, v31, vcc_lo
	global_store_b32 v[30:31], v32, off
.LBB0_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v31, 14, v2
	v_add_nc_u32_e32 v30, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v31
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_45
; %bb.44:
	v_lshlrev_b64_e32 v[32:33], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, s8, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, s9, v33, vcc_lo
	global_load_b32 v33, v[32:33], off offset:56
	v_and_b32_e32 v32, 31, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v32, v32, 2, v4
	ds_load_b32 v34, v32 offset:1792
	v_mad_co_u64_u32 v[31:32], null, v31, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v32, 0 :: v_dual_mul_f32 v33, v33, v34
	v_lshlrev_b64_e32 v[31:32], 2, v[31:32]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v31, vcc_lo, v5, v31
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v32, null, v50, v32, vcc_lo
	global_store_b32 v[31:32], v33, off
.LBB0_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 15, v2
	v_add_nc_u32_e32 v31, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_47
; %bb.46:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:60
	v_and_b32_e32 v33, 31, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1920
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v32, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v172, v151 offset1:1
	ds_store_2addr_b32 v7, v170, v171 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v168, v169 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v166, v167 offset0:6 offset1:7
	ds_store_b32 v8, v165
	ds_store_b32 v9, v164
	ds_store_b32 v10, v163
	ds_store_b32 v12, v162
	ds_store_b32 v13, v161
	ds_store_b32 v14, v160
	ds_store_b32 v15, v159
	ds_store_b32 v16, v158
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_49
; %bb.48:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	ds_load_b32 v35, v11
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:64
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 17, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_51
; %bb.50:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:68
	v_and_b32_e32 v33, 31, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:128
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_51:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 18, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_53
; %bb.52:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:72
	v_and_b32_e32 v33, 31, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:256
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 19, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_55
; %bb.54:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:76
	v_and_b32_e32 v33, 31, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:384
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 20, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_57
; %bb.56:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:80
	v_and_b32_e32 v33, 31, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:512
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 21, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_59
; %bb.58:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:84
	v_and_b32_e32 v33, 31, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:640
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 22, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_61
; %bb.60:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:88
	v_and_b32_e32 v33, 31, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:768
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_61:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 23, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_63
; %bb.62:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:92
	v_and_b32_e32 v33, 31, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:896
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 24, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_65
; %bb.64:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:96
	v_and_b32_e32 v33, 31, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1024
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 25, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_67
; %bb.66:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:100
	v_and_b32_e32 v33, 31, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1152
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 26, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_69
; %bb.68:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:104
	v_and_b32_e32 v33, 31, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1280
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 27, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_71
; %bb.70:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:108
	v_and_b32_e32 v33, 31, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1408
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 28, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_73
; %bb.72:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:112
	v_and_b32_e32 v33, 31, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1536
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 29, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_75
; %bb.74:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:116
	v_and_b32_e32 v33, 31, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1664
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 30, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_77
; %bb.76:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:120
	v_and_b32_e32 v33, 31, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1792
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 31, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_79
; %bb.78:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:124
	v_and_b32_e32 v33, 31, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1920
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v32, 32, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v150, v157 offset1:1
	ds_store_2addr_b32 v7, v148, v149 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v146, v147 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v255, v145 offset0:6 offset1:7
	ds_store_b32 v8, v254
	ds_store_b32 v9, v253
	ds_store_b32 v10, v252
	ds_store_b32 v12, v251
	ds_store_b32 v13, v250
	ds_store_b32 v14, v249
	ds_store_b32 v15, v248
	ds_store_b32 v16, v247
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_81
; %bb.80:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	ds_load_b32 v35, v11
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:128
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 33, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_83
; %bb.82:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:132
	v_and_b32_e32 v33, 31, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:128
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 34, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_85
; %bb.84:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:136
	v_and_b32_e32 v33, 31, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:256
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 35, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_87
; %bb.86:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:140
	v_and_b32_e32 v33, 31, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:384
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 36, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_89
; %bb.88:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:144
	v_and_b32_e32 v33, 31, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:512
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 37, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_91
; %bb.90:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:148
	v_and_b32_e32 v33, 31, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:640
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 38, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_93
; %bb.92:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:152
	v_and_b32_e32 v33, 31, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:768
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 39, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_95
; %bb.94:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:156
	v_and_b32_e32 v33, 31, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:896
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 40, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_97
; %bb.96:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:160
	v_and_b32_e32 v33, 31, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1024
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 41, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_99
; %bb.98:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:164
	v_and_b32_e32 v33, 31, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1152
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 42, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_101
; %bb.100:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:168
	v_and_b32_e32 v33, 31, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1280
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 43, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_103
; %bb.102:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:172
	v_and_b32_e32 v33, 31, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1408
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 44, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_105
; %bb.104:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:176
	v_and_b32_e32 v33, 31, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1536
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 45, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_107
; %bb.106:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:180
	v_and_b32_e32 v33, 31, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1664
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_107:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 46, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_109
; %bb.108:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:184
	v_and_b32_e32 v33, 31, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1792
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	v_mov_b32_e32 v33, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_109:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 47, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_111
; %bb.110:
	v_lshlrev_b64_e32 v[33:34], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, s9, v34, vcc_lo
	global_load_b32 v34, v[33:34], off offset:188
	v_and_b32_e32 v33, 31, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v35, v33 offset:1920
	v_mad_co_u64_u32 v[32:33], null, v32, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v33, 0 :: v_dual_mul_f32 v34, v34, v35
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v32, vcc_lo, v5, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, v50, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_111:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v51, v63 offset1:1
	ds_store_2addr_b32 v7, v48, v49 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v45, v46 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v43, v44 offset0:6 offset1:7
	ds_store_b32 v8, v42
	ds_store_b32 v9, v41
	ds_store_b32 v10, v40
	ds_store_b32 v12, v39
	ds_store_b32 v13, v38
	ds_store_b32 v14, v37
	ds_store_b32 v15, v36
	ds_store_b32 v16, v47
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 48, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_113
; %bb.112:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	ds_load_b32 v10, v11
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:192
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mul_f32 v9, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_113:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 49, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_115
; %bb.114:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:196
	v_and_b32_e32 v8, 31, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:128
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_115:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 50, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_117
; %bb.116:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:200
	v_and_b32_e32 v8, 31, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:256
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mul_f32 v9, v9, v10
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 51, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_119
; %bb.118:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:204
	v_and_b32_e32 v8, 31, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:384
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_119:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 52, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_121
; %bb.120:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:208
	v_and_b32_e32 v8, 31, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:512
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_121:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 53, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_123
; %bb.122:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:212
	v_and_b32_e32 v8, 31, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:640
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_123:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 54, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_125
; %bb.124:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:216
	v_and_b32_e32 v8, 31, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:768
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mul_f32 v9, v9, v10
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 55, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_127
; %bb.126:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:220
	v_and_b32_e32 v8, 31, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:896
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 56, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_129
; %bb.128:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:224
	v_and_b32_e32 v8, 31, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:1024
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_129:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 57, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_131
; %bb.130:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:228
	v_and_b32_e32 v8, 31, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:1152
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_131:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 58, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_133
; %bb.132:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:232
	v_and_b32_e32 v8, 31, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:1280
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mul_f32 v9, v9, v10
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_133:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 59, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_135
; %bb.134:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:236
	v_and_b32_e32 v8, 31, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:1408
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_135:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 60, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_137
; %bb.136:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:240
	v_and_b32_e32 v8, 31, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:1536
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_137:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 61, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_139
; %bb.138:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:244
	v_and_b32_e32 v8, 31, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:1664
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_139:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 62, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_141
; %bb.140:
	v_lshlrev_b64_e32 v[8:9], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s8, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s9, v9, vcc_lo
	global_load_b32 v9, v[8:9], off offset:248
	v_and_b32_e32 v8, 31, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v10, v8 offset:1792
	v_mad_co_u64_u32 v[7:8], null, v7, v6, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v8, 0 :: v_dual_mul_f32 v9, v9, v10
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v50, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_141:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 63, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_143
; %bb.142:
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s8, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s9, v3, vcc_lo
	global_load_b32 v3, v[2:3], off offset:252
	v_and_b32_e32 v2, 31, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v2, v2, 2, v4
	ds_load_b32 v4, v2 offset:1920
	v_mad_co_u64_u32 v[1:2], null, v7, v6, v[1:2]
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v5, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v50, v2, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v3, v3, v4
	global_store_b32 v[1:2], v3, off
.LBB0_143:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end0:
	.size	gu_r2, .Lfunc_end0-gu_r2
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gu_r2
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 388
		.amdhsa_kernarg_size 72
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
		.amdhsa_next_free_sgpr 36
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gu_r2)<<4)&4080)>>4
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
	.set .Lgu_r2.num_vgpr, 256
	.set .Lgu_r2.num_agpr, 0
	.set .Lgu_r2.numbered_sgpr, 36
	.set .Lgu_r2.num_named_barrier, 0
	.set .Lgu_r2.private_seg_size, 388
	.set .Lgu_r2.uses_vcc, 1
	.set .Lgu_r2.uses_flat_scratch, 1
	.set .Lgu_r2.has_dyn_sized_stack, 0
	.set .Lgu_r2.has_recursion, 0
	.set .Lgu_r2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 23260
; TotalNumSgprs: 38
; NumVgprs: 256
; ScratchSize: 388
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 38
; NumVGPRsForWavesPerEU: 256
; Occupancy: 5
; WaveLimiterHint : 0
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
	.type	__hip_cuid_511352a1cefe48af,@object ; @__hip_cuid_511352a1cefe48af
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_511352a1cefe48af
__hip_cuid_511352a1cefe48af:
	.byte	0                               ; 0x0
	.size	__hip_cuid_511352a1cefe48af, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_511352a1cefe48af
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
    .name:           gu_r2
    .private_segment_fixed_size: 388
    .sgpr_count:     38
    .sgpr_spill_count: 0
    .symbol:         gu_r2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 105
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
