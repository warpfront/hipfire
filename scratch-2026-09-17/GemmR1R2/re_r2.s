	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	re_r2                   ; -- Begin function re_r2
	.globl	re_r2
	.p2align	8
	.type	re_r2,@function
re_r2:                                  ; @re_r2
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[16:18], s[0:1], 0x28
	v_lshrrev_b32_e32 v4, 2, v0
	s_lshl_b32 s19, ttmp7, 7
	v_and_b32_e32 v2, 3, v0
	v_lshlrev_b32_e32 v1, 1, v0
	v_lshrrev_b32_e32 v3, 4, v0
	v_or_b32_e32 v7, s19, v4
	s_lshl_b32 s22, ttmp9, 7
	s_load_b256 s[8:15], s[0:1], 0x0
	v_and_b32_e32 v5, 0x78, v1
	v_lshlrev_b32_e32 v1, 4, v2
	v_or_b32_e32 v8, 64, v7
	v_or_b32_e32 v4, s22, v4
	v_lshrrev_b32_e32 v57, 1, v0
	s_wait_kmcnt 0x0
	s_add_co_i32 s6, s18, -1
	v_cmp_gt_i32_e64 s2, s18, v7
	v_min_i32_e32 v7, s6, v7
	v_min_i32_e32 v9, s6, v8
	s_ashr_i32 s3, s17, 31
	s_add_co_i32 s7, s16, -1
	s_lshr_b32 s3, s3, 24
	v_mad_co_u64_u32 v[203:204], null, v7, s17, v[1:2]
	v_mad_co_u64_u32 v[204:205], null, v9, s17, v[1:2]
	v_mov_b32_e32 v9, 0
	v_and_or_b32 v6, v3, 12, v2
	v_or_b32_e32 v3, 16, v3
	s_add_co_i32 s4, s17, s3
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v7, s7, v4
	s_ashr_i32 s23, s4, 8
	v_mad_u32_u24 v6, 0x120, v6, v5
	v_and_or_b32 v3, v3, 28, v2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s20, s23, 0x88
	v_or_b32_e32 v1, 64, v4
	v_cmp_gt_i32_e64 s3, s18, v8
	v_lshlrev_b32_e32 v13, 3, v2
	v_mad_u32_u24 v3, 0x120, v3, v5
	v_mul_lo_u32 v5, s20, v7
	v_and_b32_e32 v7, 0x7f, v0
	v_min_i32_e32 v2, s7, v1
	v_cmp_gt_i32_e64 s4, s16, v4
	v_or_b32_e32 v4, s19, v57
	s_cmp_lt_i32 s17, 0x100
	v_or_b32_e32 v8, s22, v7
	v_mul_lo_u32 v2, s20, v2
	v_add_co_u32 v14, s5, s8, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v15, null, s9, 0, s5
	v_cmp_gt_i32_e64 s5, s16, v1
	v_min_i32_e32 v1, s6, v4
	v_min_i32_e32 v10, s7, v8
	v_add_co_u32 v16, s6, s8, v2
	v_lshrrev_b32_e32 v2, 7, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v5, v1, s23
	v_mul_lo_u32 v1, s20, v10
	v_lshlrev_b32_e32 v7, 3, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s9, 0, s6
	v_cmp_gt_i32_e64 s6, s18, v4
	v_lshlrev_b32_e32 v4, 2, v57
	v_cmp_gt_i32_e64 s7, s16, v8
	v_lshlrev_b32_e32 v104, 1, v5
	v_add_co_u32 v102, s8, s8, v1
	v_lshl_or_b32 v1, v2, 2, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v103, null, s9, 0, s8
	v_add_nc_u32_e32 v253, 0, v4
	v_lshlrev_b32_e32 v189, 4, v2
	v_add_nc_u32_e32 v252, 0, v1
	v_add_nc_u32_e32 v254, 0, v6
	v_add_nc_u32_e32 v205, 0, v3
	s_cselect_b32 s20, -1, 0
	s_mov_b32 s17, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	v_add_co_u32 v26, vcc_lo, v14, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v27, null, 0, v15, vcc_lo
	v_add_co_u32 v6, vcc_lo, v16, v13
	global_load_b32 v34, v[102:103], off
	s_clause 0x1
	global_load_b128 v[18:21], v203, s[10:11]
	global_load_b128 v[22:25], v204, s[10:11]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v17, vcc_lo
	v_lshlrev_b32_e32 v8, 1, v5
	s_clause 0x1
	global_load_b64 v[28:29], v[26:27], off offset:8
	global_load_b64 v[30:31], v[6:7], off offset:8
	global_load_b128 v[1:4], v204, s[10:11] offset:64
	s_mov_b32 s8, 0x4e4c4a48
	v_lshlrev_b64_e32 v[32:33], 2, v[8:9]
	global_load_b128 v[9:12], v203, s[10:11] offset:64
	v_add_nc_u32_e32 v35, 0x2000, v254
	s_mov_b32 s9, 0x4040404
	v_add_nc_u32_e32 v36, 0x2000, v205
	v_add_co_u32 v32, vcc_lo, s12, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, s13, v33, vcc_lo
	global_load_b32 v32, v[32:33], off
	s_clause 0x1
	global_load_b64 v[7:8], v[6:7], off offset:40
	global_load_b64 v[5:6], v[26:27], off offset:40
	s_wait_loadcnt 0x9
	v_lshrrev_b32_e32 v26, v189, v34
	s_wait_loadcnt 0x8
	v_cndmask_b32_e64 v19, 0, v19, s2
	v_cndmask_b32_e64 v18, 0, v18, s2
	v_cndmask_b32_e64 v21, 0, v21, s2
	v_cndmask_b32_e64 v20, 0, v20, s2
	v_cvt_f32_f16_e32 v26, v26.l
	s_wait_loadcnt 0x6
	v_cndmask_b32_e64 v27, 0, v28, s4
	v_cndmask_b32_e64 v28, 0, v29, s4
	s_wait_loadcnt 0x5
	v_cndmask_b32_e64 v29, 0, v30, s5
	v_cndmask_b32_e64 v30, 0, v31, s5
	v_cndmask_b32_e64 v23, 0, v23, s3
	s_wait_loadcnt 0x3
	v_mov_b32_e32 v142, v10
	v_mov_b32_e32 v144, v12
	v_cndmask_b32_e64 v10, 0, v26, s7
	v_and_b32_e32 v12, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v26, 4, v27
	v_and_b32_e32 v27, 0xf0f0f0f, v28
	v_lshrrev_b32_e32 v28, 4, v28
	v_and_b32_e32 v31, 0xf0f0f0f, v29
	v_lshrrev_b32_e32 v29, 4, v29
	v_lshrrev_b32_e32 v33, 4, v30
	v_and_b32_e32 v26, 0xf0f0f0f, v26
	v_and_b32_e32 v28, 0xf0f0f0f, v28
	v_and_b32_e32 v30, 0xf0f0f0f, v30
	v_and_b32_e32 v29, 0xf0f0f0f, v29
	v_and_b32_e32 v33, 0xf0f0f0f, v33
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v32, 0, v32, s6
	v_perm_b32 v34, v26, v12, 0x5010400
	v_perm_b32 v12, v26, v12, 0x7030602
	v_perm_b32 v26, v28, v27, 0x5010400
	v_perm_b32 v27, v28, v27, 0x7030602
	v_perm_b32 v28, v29, v31, 0x5010400
	v_perm_b32 v29, v29, v31, 0x7030602
	v_perm_b32 v31, v33, v30, 0x5010400
	v_perm_b32 v30, v33, v30, 0x7030602
	ds_store_b32 v253, v32 offset:18432
	v_and_b32_e32 v32, 0x7070707, v34
	v_lshrrev_b32_e32 v33, 1, v34
	v_and_b32_e32 v34, 0x7070707, v12
	v_lshrrev_b32_e32 v12, 1, v12
	v_and_b32_e32 v37, 0x7070707, v26
	v_lshrrev_b32_e32 v26, 1, v26
	v_and_b32_e32 v38, 0x7070707, v27
	v_lshrrev_b32_e32 v27, 1, v27
	v_and_b32_e32 v39, 0x7070707, v28
	v_lshrrev_b32_e32 v28, 1, v28
	v_and_b32_e32 v40, 0x7070707, v29
	v_lshrrev_b32_e32 v29, 1, v29
	v_and_b32_e32 v41, 0x7070707, v31
	v_lshrrev_b32_e32 v31, 1, v31
	v_and_b32_e32 v42, 0x7070707, v30
	v_lshrrev_b32_e32 v30, 1, v30
	s_wait_alu depctr_sa_sdst(0)
	v_perm_b32 v43, s8, 0x44403800, v32
	v_or_b32_e32 v32, 0x50505050, v32
	v_and_or_b32 v33, v33, s9, 0x3020100
	v_perm_b32 v44, s8, 0x44403800, v34
	v_or_b32_e32 v34, 0x50505050, v34
	v_and_or_b32 v12, v12, s9, 0x3020100
	v_perm_b32 v45, s8, 0x44403800, v37
	v_or_b32_e32 v37, 0x50505050, v37
	v_and_or_b32 v46, v26, s9, 0x3020100
	v_perm_b32 v47, s8, 0x44403800, v38
	v_or_b32_e32 v38, 0x50505050, v38
	v_and_or_b32 v48, v27, s9, 0x3020100
	v_perm_b32 v49, s8, 0x44403800, v39
	v_or_b32_e32 v39, 0x50505050, v39
	v_and_or_b32 v50, v28, s9, 0x3020100
	v_perm_b32 v51, s8, 0x44403800, v40
	v_or_b32_e32 v40, 0x50505050, v40
	v_and_or_b32 v52, v29, s9, 0x3020100
	v_perm_b32 v53, s8, 0x44403800, v41
	v_or_b32_e32 v41, 0x50505050, v41
	v_and_or_b32 v54, v31, s9, 0x3020100
	v_perm_b32 v55, s8, 0x44403800, v42
	v_or_b32_e32 v42, 0x50505050, v42
	v_and_or_b32 v56, v30, s9, 0x3020100
	v_cndmask_b32_e64 v22, 0, v22, s3
	v_cndmask_b32_e64 v25, 0, v25, s3
	v_cndmask_b32_e64 v24, 0, v24, s3
	v_perm_b32 v26, v32, v43, v33
	v_perm_b32 v27, v34, v44, v12
	v_perm_b32 v28, v37, v45, v46
	v_perm_b32 v29, v38, v47, v48
	v_perm_b32 v30, v39, v49, v50
	v_perm_b32 v31, v40, v51, v52
	v_perm_b32 v32, v41, v53, v54
	v_perm_b32 v33, v42, v55, v56
	ds_store_b32 v252, v10 offset:18944
	ds_store_2addr_b64 v254, v[18:19], v[20:21] offset1:16
	ds_store_2addr_b64 v205, v[22:23], v[24:25] offset1:16
	ds_store_2addr_b64 v35, v[26:27], v[28:29] offset0:128 offset1:144
	ds_store_2addr_b64 v36, v[30:31], v[32:33] offset0:128 offset1:144
	s_branch .LBB0_3
.LBB0_2:
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v11, v9
	v_mov_b32_e32 v12, v9
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v1, v5
	v_dual_mov_b32 v2, v6 :: v_dual_mov_b32 v3, v7
	v_dual_mov_b32 v4, v8 :: v_dual_mov_b32 v5, v9
	v_dual_mov_b32 v6, v10 :: v_dual_mov_b32 v1, 0
	v_dual_mov_b32 v7, v11 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v8, v12 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v4, 0
.LBB0_3:
	s_load_b64 s[8:9], s[0:1], 0x20
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_bfe_u32 v12, v0, 4, 1
	s_and_not1_b32 vcc_lo, exec_lo, s20
                                        ; implicit-def: $vgpr10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 3, v12
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_5
; %bb.4:
	v_lshlrev_b32_e32 v10, 3, v12
	s_mov_b32 s17, 0
.LBB0_5:
	v_dual_mov_b32 v35, 0 :: v_dual_and_b32 v12, 15, v0
	v_dual_mov_b32 v37, 0 :: v_dual_and_b32 v18, 64, v57
	v_dual_mov_b32 v36, 0 :: v_dual_and_b32 v25, 0x60, v0
	v_dual_mov_b32 v38, 0 :: v_dual_and_b32 v19, 31, v0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v194, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v255, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v236, 0
	v_dual_mov_b32 v235, 0 :: v_dual_mov_b32 v238, 0
	v_dual_mov_b32 v193, 0 :: v_dual_mov_b32 v240, 0
	v_dual_mov_b32 v237, 0 :: v_dual_mov_b32 v242, 0
	v_dual_mov_b32 v239, 0 :: v_dual_mov_b32 v244, 0
	v_dual_mov_b32 v241, 0 :: v_dual_mov_b32 v246, 0
	v_dual_mov_b32 v243, 0 :: v_dual_mov_b32 v248, 0
	v_dual_mov_b32 v245, 0 :: v_dual_mov_b32 v250, 0
	v_dual_mov_b32 v247, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v249, 0 :: v_dual_mov_b32 v182, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_mov_b32 s17, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_16
; %bb.6:
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v0, off offset:360
	scratch_store_b32 off, v19, off offset:364
	v_lshrrev_b32_e32 v0, 2, v18
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v18, off offset:368
	scratch_store_b32 off, v12, off offset:376
	v_or_b32_e32 v12, v18, v12
	v_or_b32_e32 v18, v20, v25
	v_lshrrev_b32_e32 v10, 2, v25
	v_mov_b32_e32 v105, 0
	v_add_co_u32 v206, vcc_lo, v14, v13
	v_lshlrev_b32_e32 v12, 2, v12
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v181, v105 :: v_dual_lshlrev_b32 v18, 3, v18
	v_mad_u32_u24 v0, 0x120, v0, 0
	v_mad_u32_u24 v10, 0x120, v10, 0
	v_dual_mov_b32 v182, v105 :: v_dual_lshlrev_b32 v19, 3, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v207, null, 0, v15, vcc_lo
	v_add_co_u32 v208, vcc_lo, v16, v13
	v_dual_mov_b32 v180, v105 :: v_dual_add_nc_u32 v13, 0, v18
	v_dual_mov_b32 v179, v105 :: v_dual_add_nc_u32 v10, v10, v19
	v_dual_mov_b32 v250, v105 :: v_dual_add_nc_u32 v183, v0, v19
	v_dual_mov_b32 v249, v105 :: v_dual_add_nc_u32 v0, 0, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v247, v105 :: v_dual_add_nc_u32 v12, 0x4a00, v13
	v_dual_mov_b32 v157, v105 :: v_dual_add_nc_u32 v184, 0x2400, v10
	v_dual_mov_b32 v155, v105 :: v_dual_add_nc_u32 v10, 0x2000, v254
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v255, v105 :: v_dual_add_nc_u32 v0, 0x4800, v0
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v12, off offset:8
	scratch_store_b32 off, v10, off
	v_dual_mov_b32 v245, v105 :: v_dual_add_nc_u32 v12, 0x4a08, v13
	v_dual_mov_b32 v153, v105 :: v_dual_add_nc_u32 v10, 0x2000, v205
	scratch_store_b32 off, v0, off offset:72 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, v203
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v12, off offset:12
	scratch_store_b32 off, v10, off offset:4
	v_dual_mov_b32 v243, v105 :: v_dual_add_nc_u32 v12, 0x4a10, v13
	v_mov_b32_e32 v241, v105
	v_mov_b32_e32 v239, v105
	v_mov_b32_e32 v237, v105
	scratch_store_b32 off, v12, off offset:16 ; 4-byte Folded Spill
	v_dual_mov_b32 v193, v105 :: v_dual_add_nc_u32 v12, 0x4a18, v13
	v_mov_b32_e32 v235, v105
	v_mov_b32_e32 v177, v105
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v209, null, 0, v17, vcc_lo
	scratch_store_b32 off, v12, off offset:20 ; 4-byte Folded Spill
	v_dual_mov_b32 v175, v105 :: v_dual_add_nc_u32 v12, 0x4a20, v13
	v_dual_mov_b32 v248, v105 :: v_dual_add_nc_u32 v231, 0x800, v183
	v_dual_mov_b32 v246, v105 :: v_dual_mov_b32 v43, v105
	scratch_store_b32 off, v12, off offset:24 ; 4-byte Folded Spill
	v_dual_mov_b32 v173, v105 :: v_dual_add_nc_u32 v12, 0x4a28, v13
	v_dual_mov_b32 v244, v105 :: v_dual_mov_b32 v49, v105
	v_mov_b32_e32 v171, v105
	scratch_store_b32 off, v12, off offset:28 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v12, 0x4a30, v13
	v_dual_mov_b32 v242, v105 :: v_dual_mov_b32 v55, v105
	v_dual_mov_b32 v167, v105 :: v_dual_mov_b32 v240, v105
	v_mov_b32_e32 v53, v105
	scratch_store_b32 off, v12, off offset:32 ; 4-byte Folded Spill
	v_dual_mov_b32 v165, v105 :: v_dual_add_nc_u32 v12, 0x4a38, v13
	v_dual_mov_b32 v238, v105 :: v_dual_mov_b32 v47, v105
	v_dual_mov_b32 v236, v105 :: v_dual_mov_b32 v45, v105
	scratch_store_b32 off, v12, off offset:36 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v12, 0x4a80, v13
	v_dual_mov_b32 v178, v105 :: v_dual_mov_b32 v41, v105
	v_dual_mov_b32 v176, v105 :: v_dual_mov_b32 v39, v105
	scratch_store_b32 off, v12, off offset:40 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v12, 0x4a88, v13
	v_dual_mov_b32 v174, v105 :: v_dual_mov_b32 v37, v105
	v_dual_mov_b32 v172, v105 :: v_dual_mov_b32 v35, v105
	scratch_store_b32 off, v12, off offset:44 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v12, 0x4a90, v13
	v_mov_b32_e32 v168, v105
	v_mov_b32_e32 v166, v105
	v_mov_b32_e32 v164, v105
	v_mov_b32_e32 v162, v105
	scratch_store_b32 off, v12, off offset:48 ; 4-byte Folded Spill
	v_dual_mov_b32 v163, v105 :: v_dual_add_nc_u32 v12, 0x4a98, v13
	v_dual_mov_b32 v158, v105 :: v_dual_mov_b32 v161, v105
	v_mov_b32_e32 v156, v105
	scratch_store_b32 off, v12, off offset:52 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v12, 0x4aa0, v13
	v_mov_b32_e32 v154, v105
	v_mov_b32_e32 v44, v105
	v_mov_b32_e32 v194, v105
	v_mov_b32_e32 v56, v105
	scratch_store_b32 off, v12, off offset:56 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v12, 0x4aa8, v13
	v_mov_b32_e32 v54, v105
	v_mov_b32_e32 v48, v105
	v_mov_b32_e32 v46, v105
	v_mov_b32_e32 v42, v105
	scratch_store_b32 off, v12, off offset:60 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v12, 0x4ab0, v13
	v_mov_b32_e32 v40, v105
	v_mov_b32_e32 v38, v105
	v_mov_b32_e32 v36, v105
	s_clause 0x1                            ; 12-byte Folded Spill
	scratch_store_b64 off, v[0:1], off offset:304
	scratch_store_b32 off, v12, off offset:64
	v_add_nc_u32_e32 v12, 0x4ab8, v13
	v_mov_b32_e32 v0, v204
	s_mov_b32 s1, 0
	s_sub_co_i32 s24, 0, s23
	s_movk_i32 s20, 0x100
	s_movk_i32 s25, 0x88
	s_mov_b32 s26, 0x4e4c4a48
	s_mov_b32 s27, 0x4040404
	s_clause 0xd                            ; 64-byte Folded Spill
	scratch_store_b32 off, v25, off offset:372
	scratch_store_b32 off, v20, off offset:380
	scratch_store_b32 off, v12, off offset:68
	scratch_store_b64 off, v[0:1], off offset:312
	scratch_store_b64 off, v[102:103], off offset:320
	scratch_store_b32 off, v253, off offset:328
	scratch_store_b32 off, v189, off offset:332
	scratch_store_b32 off, v252, off offset:336
	scratch_store_b32 off, v254, off offset:340
	scratch_store_b32 off, v205, off offset:300
	scratch_store_b32 off, v206, off offset:344
	scratch_store_b32 off, v207, off offset:348
	scratch_store_b32 off, v208, off offset:352
	scratch_store_b32 off, v209, off offset:356
	s_branch .LBB0_8
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=1
	s_clause 0x3                            ; 32-byte Folded Reload
	scratch_load_b64 v[145:146], off, off offset:228 th:TH_LOAD_LU
	scratch_load_b64 v[147:148], off, off offset:236 th:TH_LOAD_LU
	scratch_load_b64 v[149:150], off, off offset:244 th:TH_LOAD_LU
	scratch_load_b64 v[151:152], off, off offset:260 th:TH_LOAD_LU
	s_add_co_i32 s17, s17, 1
	s_addk_co_i32 s20, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s24, s17
	s_addk_co_i32 s25, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 1
	scratch_load_b64 v[185:186], off, off offset:276 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x3
	v_dual_fmac_f32 v181, v145, v69 :: v_dual_fmac_f32 v182, v147, v70
	s_wait_loadcnt 0x2
	v_fmac_f32_e32 v179, v149, v71
	scratch_load_b64 v[70:71], off, off offset:252 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x2
	v_fmac_f32_e32 v249, v151, v73
	v_dual_fmac_f32 v164, v145, v53 :: v_dual_fmac_f32 v165, v147, v54
	v_fmac_f32_e32 v162, v149, v55
	scratch_load_b32 v53, off, off offset:124 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v193, v149, v63
	scratch_load_b32 v55, off, off offset:132 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_dual_fmac_f32 v237, v145, v61 :: v_dual_fmac_f32 v238, v147, v62
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v247, v185, v75
	v_fmac_f32_e32 v164, v146, v159
	scratch_load_b64 v[61:62], off, off offset:184 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v182, v148, v169
	v_fmac_f32_e32 v238, v148, v170
	s_wait_dscnt 0xf
	v_fmac_f32_e32 v164, v233, v117
	s_wait_dscnt 0xe
	v_fmac_f32_e32 v182, v231, v134
	v_fmac_f32_e32 v238, v231, v126
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v164, v234, v199
	v_fmac_f32_e32 v182, v232, v201
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v238, v232, v202
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v180, v70, v72
	scratch_load_b64 v[72:73], off, off offset:268 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_dual_fmac_f32 v163, v70, v56 :: v_dual_fmac_f32 v158, v151, v57
	scratch_load_b32 v56, off, off offset:136 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v236, v70, v64
	v_fmac_f32_e32 v178, v151, v65
	scratch_load_b64 v[63:64], off, off offset:192 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_dual_fmac_f32 v180, v71, v169 :: v_dual_fmac_f32 v165, v148, v159
	v_fmac_f32_e32 v162, v150, v159
	s_wait_loadcnt 0x3
	v_dual_fmac_f32 v236, v71, v170 :: v_dual_fmac_f32 v243, v61, v40
	scratch_load_b32 v40, off, off offset:96 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_dscnt 0xc
	v_dual_fmac_f32 v180, v227, v136 :: v_dual_fmac_f32 v193, v150, v170
	v_fmac_f32_e32 v236, v227, v128
	v_dual_fmac_f32 v172, v61, v32 :: v_dual_fmac_f32 v165, v231, v118
	v_fmac_f32_e32 v162, v229, v119
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v180, v228, v201
	v_fmac_f32_e32 v236, v228, v202
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v172, v62, v170
	v_fmac_f32_e32 v162, v230, v199
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v172, v211, v96
	v_fmac_f32_e32 v172, v212, v202
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v250, v72, v74
	scratch_load_b64 v[74:75], off, off offset:284 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v161, v72, v58
	s_clause 0x1                            ; 12-byte Folded Reload
	scratch_load_b64 v[57:58], off, off offset:168 th:TH_LOAD_LU
	scratch_load_b32 v54, off, off offset:128 th:TH_LOAD_LU
	v_dual_fmac_f32 v235, v72, v66 :: v_dual_fmac_f32 v176, v185, v67
	v_fmac_f32_e32 v53, v149, v47
	scratch_load_b64 v[65:66], off, off offset:200 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v156, v185, v59
	scratch_load_b32 v47, off, off offset:116 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x7
	v_dual_fmac_f32 v56, v147, v46 :: v_dual_fmac_f32 v249, v152, v169
	v_dual_fmac_f32 v237, v146, v170 :: v_dual_fmac_f32 v178, v152, v170
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v193, v229, v127 :: v_dual_fmac_f32 v56, v148, v160
	v_fmac_f32_e32 v249, v225, v137
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v237, v233, v125
	v_fmac_f32_e32 v178, v225, v129
	v_dual_fmac_f32 v181, v146, v169 :: v_dual_fmac_f32 v250, v73, v169
	v_fmac_f32_e32 v56, v231, v110
	v_dual_fmac_f32 v163, v71, v159 :: v_dual_fmac_f32 v158, v152, v159
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v181, v233, v133 :: v_dual_fmac_f32 v250, v223, v138
	v_dual_fmac_f32 v165, v232, v199 :: v_dual_fmac_f32 v178, v226, v202
	v_dual_fmac_f32 v163, v227, v120 :: v_dual_fmac_f32 v158, v225, v121
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v181, v234, v201 :: v_dual_fmac_f32 v250, v224, v201
	v_add_nc_u32_e32 v231, 0x800, v183
	v_dual_fmac_f32 v163, v228, v199 :: v_dual_fmac_f32 v158, v226, v199
	v_fmac_f32_e32 v56, v232, v200
	s_wait_loadcnt 0x4
	v_fmac_f32_e32 v177, v74, v68
	scratch_load_b64 v[67:68], off, off offset:208 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v157, v74, v60
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v54, v70, v48
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b64 v[69:70], off, off offset:216 th:TH_LOAD_LU
	scratch_load_b64 v[59:60], off, off offset:176 th:TH_LOAD_LU
	v_dual_fmac_f32 v245, v57, v38 :: v_dual_fmac_f32 v242, v63, v41
	s_wait_loadcnt 0x4
	v_fmac_f32_e32 v241, v65, v42
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v42, off, off offset:104 th:TH_LOAD_LU
	scratch_load_b32 v41, off, off offset:100 th:TH_LOAD_LU
	scratch_load_b32 v38, off, off offset:88 th:TH_LOAD_LU
	v_fmac_f32_e32 v248, v74, v76
	v_fmac_f32_e32 v247, v186, v169
	v_fmac_f32_e32 v177, v75, v170
	v_dual_fmac_f32 v171, v63, v33 :: v_dual_fmac_f32 v168, v65, v34
	v_fmac_f32_e32 v154, v57, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v247, v221, v139
	v_fmac_f32_e32 v177, v219, v132
	v_dual_fmac_f32 v157, v75, v159 :: v_dual_fmac_f32 v174, v57, v30
	v_fmac_f32_e32 v168, v66, v170
	v_dual_fmac_f32 v245, v58, v169 :: v_dual_fmac_f32 v242, v64, v169
	v_fmac_f32_e32 v154, v58, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v174, v58, v170
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v168, v207, v98
	v_dual_fmac_f32 v245, v215, v102 :: v_dual_fmac_f32 v242, v209, v105
	v_fmac_f32_e32 v154, v215, v86
	v_fmac_f32_e32 v174, v215, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v168, v208, v202 :: v_dual_fmac_f32 v179, v150, v169
	v_dual_fmac_f32 v248, v75, v169 :: v_dual_fmac_f32 v235, v73, v170
	v_dual_fmac_f32 v176, v186, v170 :: v_dual_fmac_f32 v161, v73, v159
	v_fmac_f32_e32 v156, v186, v159
	v_dual_fmac_f32 v54, v71, v160 :: v_dual_fmac_f32 v179, v229, v135
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v248, v219, v140
	v_dual_fmac_f32 v235, v223, v130 :: v_dual_fmac_f32 v176, v221, v131
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v161, v223, v122 :: v_dual_fmac_f32 v156, v221, v123
	v_dual_fmac_f32 v54, v227, v112 :: v_dual_fmac_f32 v237, v234, v202
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v193, v230, v202 :: v_dual_fmac_f32 v248, v220, v201
	v_dual_fmac_f32 v249, v226, v201 :: v_dual_fmac_f32 v174, v216, v202
	v_fmac_f32_e32 v54, v228, v200
	v_dual_fmac_f32 v235, v224, v202 :: v_dual_fmac_f32 v176, v222, v202
	v_dual_fmac_f32 v161, v224, v199 :: v_dual_fmac_f32 v156, v222, v199
	v_dual_fmac_f32 v247, v222, v201 :: v_dual_fmac_f32 v154, v216, v199
	v_fmac_f32_e32 v242, v210, v201
	s_wait_loadcnt 0x5
	v_fmac_f32_e32 v240, v67, v43
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v43, off, off offset:140 th:TH_LOAD_LU
	scratch_load_b32 v48, off, off offset:120 th:TH_LOAD_LU
	scratch_load_b32 v46, off, off offset:112 th:TH_LOAD_LU
	v_fmac_f32_e32 v55, v145, v45
	scratch_load_b32 v45, off, off offset:108 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x8
	v_fmac_f32_e32 v239, v69, v44
	scratch_load_b32 v44, off, off offset:144 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v47, v151, v49
	s_wait_loadcnt 0x8
	v_fmac_f32_e32 v244, v59, v39
	v_dual_fmac_f32 v167, v67, v35 :: v_dual_fmac_f32 v166, v69, v36
	s_clause 0x3                            ; 16-byte Folded Reload
	scratch_load_b32 v49, off, off offset:156 th:TH_LOAD_LU
	scratch_load_b32 v39, off, off offset:92 th:TH_LOAD_LU
	scratch_load_b32 v36, off, off offset:80 th:TH_LOAD_LU
	scratch_load_b32 v35, off, off offset:76 th:TH_LOAD_LU
	v_fmac_f32_e32 v255, v61, v24
	v_fmac_f32_e32 v173, v59, v31
	v_fmac_f32_e32 v194, v67, v27
	v_dual_fmac_f32 v40, v59, v15 :: v_dual_fmac_f32 v243, v62, v169
	v_dual_fmac_f32 v240, v68, v169 :: v_dual_fmac_f32 v239, v70, v169
	v_dual_fmac_f32 v171, v64, v170 :: v_dual_fmac_f32 v166, v70, v170
	v_dual_fmac_f32 v157, v219, v124 :: v_dual_fmac_f32 v244, v60, v169
	v_fmac_f32_e32 v241, v66, v169
	v_fmac_f32_e32 v173, v60, v170
	v_dual_fmac_f32 v167, v68, v170 :: v_dual_fmac_f32 v194, v68, v159
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v240, v205, v107 :: v_dual_fmac_f32 v239, v203, v108
	v_dual_fmac_f32 v171, v209, v97 :: v_dual_fmac_f32 v166, v203, v100
	v_fmac_f32_e32 v40, v60, v160
	v_dual_fmac_f32 v244, v213, v103 :: v_dual_fmac_f32 v241, v207, v106
	v_fmac_f32_e32 v173, v213, v95
	v_fmac_f32_e32 v167, v205, v99
	v_dual_fmac_f32 v194, v205, v91 :: v_dual_fmac_f32 v243, v211, v104
	v_dual_fmac_f32 v40, v213, v79 :: v_dual_fmac_f32 v239, v204, v201
	v_fmac_f32_e32 v166, v204, v202
	v_dual_fmac_f32 v241, v208, v201 :: v_dual_fmac_f32 v240, v206, v201
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v167, v206, v202 :: v_dual_fmac_f32 v194, v206, v199
	v_fmac_f32_e32 v55, v146, v160
	v_dual_fmac_f32 v173, v214, v202 :: v_dual_fmac_f32 v40, v214, v200
	v_fmac_f32_e32 v243, v212, v201
	v_fmac_f32_e32 v171, v210, v202
	v_fmac_f32_e32 v55, v233, v109
	v_dual_fmac_f32 v245, v216, v201 :: v_dual_fmac_f32 v244, v214, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v55, v234, v200
	s_wait_loadcnt 0x8
	v_fmac_f32_e32 v43, v65, v26
	s_wait_loadcnt 0x7
	v_fmac_f32_e32 v48, v72, v50
	s_wait_loadcnt 0x6
	v_fmac_f32_e32 v46, v74, v52
	v_fmac_f32_e32 v53, v150, v160
	s_wait_loadcnt 0x5
	v_fmac_f32_e32 v45, v185, v51
	scratch_load_b64 v[50:51], off, off offset:160 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v153, v59, v23
	s_wait_loadcnt 0x5
	v_dual_fmac_f32 v44, v63, v25 :: v_dual_fmac_f32 v255, v62, v159
	v_fmac_f32_e32 v45, v186, v160
	v_dual_fmac_f32 v47, v152, v160 :: v_dual_fmac_f32 v46, v75, v160
	s_wait_loadcnt 0x4
	v_fmac_f32_e32 v49, v69, v28
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v39, v61, v16 :: v_dual_fmac_f32 v36, v67, v19
	s_wait_loadcnt 0x1
	v_dual_fmac_f32 v35, v69, v20 :: v_dual_fmac_f32 v44, v64, v159
	v_fmac_f32_e32 v49, v70, v159
	v_fmac_f32_e32 v255, v211, v88
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v68, v160 :: v_dual_fmac_f32 v35, v70, v160
	v_dual_fmac_f32 v44, v209, v89 :: v_dual_fmac_f32 v49, v203, v92
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v39, v62, v160 :: v_dual_fmac_f32 v36, v205, v83
	v_fmac_f32_e32 v35, v203, v84
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v48, v73, v160 :: v_dual_fmac_f32 v49, v204, v199
	v_dual_fmac_f32 v53, v229, v111 :: v_dual_fmac_f32 v36, v206, v200
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v35, v204, v200
	v_dual_fmac_f32 v47, v225, v113 :: v_dual_fmac_f32 v46, v219, v116
	v_fmac_f32_e32 v48, v223, v114
	v_fmac_f32_e32 v39, v211, v80
	v_fmac_f32_e32 v179, v230, v201
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v53, v230, v200 :: v_dual_fmac_f32 v46, v220, v200
	v_dual_fmac_f32 v47, v226, v200 :: v_dual_fmac_f32 v48, v224, v200
	v_dual_fmac_f32 v255, v212, v199 :: v_dual_fmac_f32 v44, v210, v199
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v246, v50, v37
	scratch_load_b32 v37, off, off offset:84 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_fmac_f32_e32 v155, v50, v21
	v_fmac_f32_e32 v175, v50, v29
	v_dual_fmac_f32 v42, v50, v13 :: v_dual_fmac_f32 v153, v60, v159
	v_dual_fmac_f32 v41, v57, v14 :: v_dual_fmac_f32 v38, v63, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v155, v51, v159
	v_fmac_f32_e32 v175, v51, v170
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v43, v66, v159 :: v_dual_fmac_f32 v42, v51, v160
	v_fmac_f32_e32 v153, v213, v87
	v_dual_fmac_f32 v45, v221, v115 :: v_dual_fmac_f32 v246, v51, v169
	v_fmac_f32_e32 v175, v217, v93
	v_dual_fmac_f32 v155, v217, v85 :: v_dual_fmac_f32 v38, v64, v160
	v_dual_fmac_f32 v43, v207, v90 :: v_dual_fmac_f32 v42, v217, v77
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v175, v218, v202 :: v_dual_fmac_f32 v246, v217, v101
	v_dual_fmac_f32 v41, v58, v160 :: v_dual_fmac_f32 v38, v209, v81
	v_fmac_f32_e32 v45, v222, v200
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v43, v208, v199
	v_dual_fmac_f32 v177, v220, v202 :: v_dual_fmac_f32 v246, v218, v201
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v41, v215, v78
	v_dual_fmac_f32 v157, v220, v199 :: v_dual_fmac_f32 v42, v218, v200
	v_fmac_f32_e32 v155, v218, v199
	v_fmac_f32_e32 v153, v214, v199
	v_fmac_f32_e32 v41, v216, v200
	v_dual_fmac_f32 v39, v212, v200 :: v_dual_fmac_f32 v38, v210, v200
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v37, v65, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v66, v160
	v_fmac_f32_e32 v37, v207, v82
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v37, v208, v200
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b64 v[104:105], off, off offset:148 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x0
	s_clause 0x1                            ; 12-byte Folded Reload
	scratch_load_b32 v104, off, off offset:224 th:TH_LOAD_LU
	scratch_load_b64 v[203:204], off, off offset:304
	s_wait_loadcnt 0x0
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b64 v[204:205], off, off offset:312
	scratch_load_b64 v[102:103], off, off offset:320
	s_wait_loadcnt 0x1
	s_clause 0x4                            ; 20-byte Folded Reload
	scratch_load_b32 v205, off, off offset:300
	scratch_load_b32 v206, off, off offset:344
	scratch_load_b32 v207, off, off offset:348
	scratch_load_b32 v208, off, off offset:352
	scratch_load_b32 v209, off, off offset:356
	s_cbranch_scc1 .LBB0_15
.LBB0_8:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s20, 0xffffff00
	v_lshlrev_b64_e32 v[12:13], 2, v[104:105]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[28:29], s[10:11], s[0:1]
	s_clause 0x12                           ; 76-byte Folded Spill
	scratch_store_b32 off, v49, off offset:156
	scratch_store_b32 off, v44, off offset:144
	scratch_store_b32 off, v43, off offset:140
	scratch_store_b32 off, v56, off offset:136
	scratch_store_b32 off, v55, off offset:132
	scratch_store_b32 off, v54, off offset:128
	scratch_store_b32 off, v53, off offset:124
	scratch_store_b32 off, v48, off offset:120
	scratch_store_b32 off, v47, off offset:116
	scratch_store_b32 off, v46, off offset:112
	scratch_store_b32 off, v45, off offset:108
	scratch_store_b32 off, v42, off offset:104
	scratch_store_b32 off, v41, off offset:100
	scratch_store_b32 off, v40, off offset:96
	scratch_store_b32 off, v39, off offset:92
	scratch_store_b32 off, v38, off offset:88
	scratch_store_b32 off, v37, off offset:84
	scratch_store_b32 off, v36, off offset:80
	scratch_store_b32 off, v35, off offset:76
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v89, s0, s28, v203
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v90, null, s29, 0, s0
	v_add_co_u32 v91, s0, s28, v204
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v92, null, s29, 0, s0
	s_add_co_i32 s0, s25, 0xffffff78
	s_wait_loadcnt 0x3
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v93, vcc_lo, v206, s0
	s_wait_loadcnt 0x2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v94, null, 0, v207, vcc_lo
	s_wait_loadcnt 0x1
	v_add_co_u32 v95, vcc_lo, v208, s0
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v96, null, 0, v209, vcc_lo
	v_add_co_u32 v12, vcc_lo, s12, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, vcc_lo
	v_add_co_u32 v14, vcc_lo, v102, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, 0, v103, vcc_lo
	s_clause 0x1
	global_load_b128 v[149:152], v[89:90], off offset:128
	global_load_b128 v[145:148], v[91:92], off offset:128
	s_clause 0x2
	global_load_b64 v[197:198], v[93:94], off offset:72
	global_load_b64 v[195:196], v[95:96], off offset:72
	global_load_b32 v106, v[14:15], off offset:4
	global_load_b32 v107, v[12:13], off offset:4
	ds_load_2addr_b64 v[77:80], v184 offset1:144
	ds_load_2addr_b64 v[12:15], v183 offset1:144
	ds_load_2addr_b64 v[81:84], v231 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[77:78], v[12:13], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[79:80], v[12:13], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[77:78], v[14:15], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[79:80], v[14:15], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[77:78], v[81:82], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[79:80], v[81:82], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[77:78], v[83:84], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[79:80], v[83:84], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[77:80], v184 offset0:36 offset1:180
	ds_load_2addr_b64 v[81:84], v183 offset0:36 offset1:180
	ds_load_2addr_b64 v[85:88], v231 offset0:68 offset1:212
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
	ds_load_2addr_b64 v[77:80], v184 offset0:72 offset1:216
	ds_load_2addr_b64 v[81:84], v183 offset0:72 offset1:216
	ds_load_2addr_b64 v[85:88], v231 offset0:104 offset1:248
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
	v_add_nc_u32_e32 v0, 0xc00, v183
	ds_load_2addr_b64 v[77:80], v184 offset0:108 offset1:252
	ds_load_2addr_b64 v[81:84], v183 offset0:108 offset1:252
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
	v_cndmask_b32_e64 v5, 0, v5, s4
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v6, 0, v6, s4
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v7, 0, v7, s5
	v_lshrrev_b32_e32 v10, 4, v5
	v_and_b32_e32 v5, 0xf0f0f0f, v5
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v8, 0, v8, s5
	v_lshrrev_b32_e32 v84, 4, v7
	v_and_b32_e32 v77, 0xf0f0f0f, v10
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cndmask_b32_e64 v10, 0, v142, s2
	v_cndmask_b32_e64 v9, 0, v9, s2
	v_perm_b32 v78, v77, v5, 0x5010400
	v_perm_b32 v77, v77, v5, 0x7030602
	v_lshrrev_b32_e32 v5, 4, v6
	v_and_b32_e32 v6, 0xf0f0f0f, v6
	v_cndmask_b32_e64 v12, 0, v144, s2
	v_and_b32_e32 v79, 0x7070707, v78
	v_lshrrev_b32_e32 v78, 1, v78
	v_and_b32_e32 v81, 0xf0f0f0f, v5
	v_cndmask_b32_e64 v11, 0, v11, s2
	v_cndmask_b32_e64 v2, 0, v2, s3
	v_perm_b32 v80, s26, 0x44403800, v79
	v_or_b32_e32 v79, 0x50505050, v79
	v_and_or_b32 v5, v78, s27, 0x3020100
	v_perm_b32 v82, v81, v6, 0x5010400
	v_and_b32_e32 v78, 0x7070707, v77
	v_lshrrev_b32_e32 v77, 1, v77
	v_perm_b32 v81, v81, v6, 0x7030602
	v_perm_b32 v5, v79, v80, v5
	v_and_b32_e32 v80, 0x7070707, v82
	v_lshrrev_b32_e32 v82, 1, v82
	v_perm_b32 v79, s26, 0x44403800, v78
	v_or_b32_e32 v78, 0x50505050, v78
	v_and_or_b32 v77, v77, s27, 0x3020100
	v_perm_b32 v83, s26, 0x44403800, v80
	v_or_b32_e32 v80, 0x50505050, v80
	v_and_or_b32 v82, v82, s27, 0x3020100
	s_barrier_wait -1
	v_perm_b32 v6, v78, v79, v77
	v_and_b32_e32 v77, 0xf0f0f0f, v7
	v_and_b32_e32 v78, 0xf0f0f0f, v84
	v_perm_b32 v7, v80, v83, v82
	v_lshrrev_b32_e32 v83, 4, v8
	v_and_b32_e32 v79, 0x7070707, v81
	v_lshrrev_b32_e32 v80, 1, v81
	v_perm_b32 v81, v78, v77, 0x5010400
	v_perm_b32 v77, v78, v77, 0x7030602
	v_and_b32_e32 v8, 0xf0f0f0f, v8
	v_and_b32_e32 v78, 0xf0f0f0f, v83
	v_perm_b32 v82, s26, 0x44403800, v79
	v_or_b32_e32 v79, 0x50505050, v79
	v_and_b32_e32 v85, 0x7070707, v77
	v_lshrrev_b32_e32 v77, 1, v77
	v_perm_b32 v86, v78, v8, 0x5010400
	v_perm_b32 v8, v78, v8, 0x7030602
	v_and_or_b32 v80, v80, s27, 0x3020100
	global_inv scope:SCOPE_SE
	v_and_or_b32 v88, v77, s27, 0x3020100
	v_and_b32_e32 v78, 0x7070707, v86
	v_lshrrev_b32_e32 v77, 1, v86
	v_and_b32_e32 v86, 0x7070707, v8
	v_lshrrev_b32_e32 v8, 1, v8
	v_and_b32_e32 v84, 0x7070707, v81
	v_perm_b32 v97, s26, 0x44403800, v78
	v_or_b32_e32 v98, 0x50505050, v78
	v_and_or_b32 v99, v77, s27, 0x3020100
	v_and_or_b32 v101, v8, s27, 0x3020100
	v_perm_b32 v8, v79, v82, v80
	v_lshrrev_b32_e32 v81, 1, v81
	v_perm_b32 v83, s26, 0x44403800, v84
	v_perm_b32 v79, v98, v97, v99
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v97, off, off
	scratch_load_b32 v98, off, off offset:4
	v_or_b32_e32 v84, 0x50505050, v84
	v_and_or_b32 v81, v81, s27, 0x3020100
	v_perm_b32 v87, s26, 0x44403800, v85
	v_or_b32_e32 v85, 0x50505050, v85
	v_perm_b32 v100, s26, 0x44403800, v86
	v_or_b32_e32 v86, 0x50505050, v86
	v_perm_b32 v77, v84, v83, v81
	v_cndmask_b32_e64 v1, 0, v1, s3
	v_perm_b32 v78, v85, v87, v88
	v_cndmask_b32_e64 v4, 0, v4, s3
	v_perm_b32 v80, v86, v100, v101
	v_cndmask_b32_e64 v3, 0, v3, s3
	ds_store_2addr_b64 v254, v[9:10], v[11:12] offset1:16
	ds_store_2addr_b64 v205, v[1:2], v[3:4] offset1:16
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v97, v[5:6], v[7:8] offset0:128 offset1:144
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v98, v[77:78], v[79:80] offset0:128 offset1:144
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_clause 0x1
	global_load_b128 v[141:144], v[89:90], off offset:192
	global_load_b128 v[1:4], v[91:92], off offset:192
	s_clause 0x1
	global_load_b64 v[5:6], v[93:94], off offset:104
	global_load_b64 v[7:8], v[95:96], off offset:104
	ds_load_2addr_b64 v[9:12], v184 offset1:144
	ds_load_2addr_b64 v[77:80], v183 offset1:144
	ds_load_2addr_b64 v[81:84], v231 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[9:10], v[77:78], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[11:12], v[77:78], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[9:10], v[79:80], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[11:12], v[79:80], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[9:10], v[81:82], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[11:12], v[81:82], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[9:10], v[83:84], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[11:12], v[83:84], v[13:20]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[9:12], v184 offset0:36 offset1:180
	ds_load_2addr_b64 v[77:80], v183 offset0:36 offset1:180
	ds_load_2addr_b64 v[81:84], v231 offset0:68 offset1:212
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[9:10], v[77:78], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[11:12], v[77:78], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[9:10], v[79:80], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[11:12], v[79:80], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[9:10], v[81:82], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[11:12], v[81:82], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[9:10], v[83:84], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[11:12], v[83:84], v[13:20]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[9:12], v184 offset0:72 offset1:216
	ds_load_2addr_b64 v[77:80], v183 offset0:72 offset1:216
	ds_load_2addr_b64 v[81:84], v231 offset0:104 offset1:248
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[9:10], v[77:78], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[11:12], v[77:78], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[9:10], v[79:80], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[11:12], v[79:80], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[9:10], v[81:82], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[11:12], v[81:82], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[9:10], v[83:84], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[11:12], v[83:84], v[13:20]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[9:12], v184 offset0:108 offset1:252
	ds_load_2addr_b64 v[77:80], v183 offset0:108 offset1:252
	ds_load_2addr_b64 v[81:84], v0 offset0:12 offset1:156
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[69:76], v[9:10], v[77:78], v[69:76]
	v_wmma_f32_16x16x16_fp8_fp8 v[37:44], v[11:12], v[77:78], v[37:44]
	v_wmma_f32_16x16x16_fp8_fp8 v[61:68], v[9:10], v[79:80], v[61:68]
	v_wmma_f32_16x16x16_fp8_fp8 v[29:36], v[11:12], v[79:80], v[29:36]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[53:60], v[9:10], v[81:82], v[53:60]
	v_wmma_f32_16x16x16_fp8_fp8 v[21:28], v[11:12], v[81:82], v[21:28]
	v_wmma_f32_16x16x16_fp8_fp8 v[45:52], v[9:10], v[83:84], v[45:52]
	v_wmma_f32_16x16x16_fp8_fp8 v[13:20], v[11:12], v[83:84], v[13:20]
	; sched_barrier mask(0x00000000)
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v0, off, off offset:72
	scratch_load_b32 v9, off, off offset:8
	s_cmp_lt_i32 s17, s23
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[169:170], v0 offset1:16
	ds_load_2addr_b32 v[159:160], v0 offset0:32 offset1:48
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	v_add_nc_u32_e32 v0, 2, v104
	s_cselect_b32 s0, -1, 0
	s_cmp_ge_i32 s17, s23
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:228 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:12 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:236 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:16 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:244 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:20 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:252 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:260 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:28 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:268 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:276 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:284 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:160 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:168 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:176 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:184 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:192 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:200 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:64 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:208 ; 8-byte Folded Spill
	scratch_load_b32 v9, off, off offset:68 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[9:10], v9 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[9:10], off offset:216 ; 8-byte Folded Spill
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
	v_cndmask_b32_e64 v11, 0, v197, s4
	v_cndmask_b32_e64 v10, 0, v150, s2
	v_cndmask_b32_e64 v9, 0, v149, s2
	v_cndmask_b32_e64 v12, 0, v152, s2
	v_lshrrev_b32_e32 v92, v189, v106
	v_lshrrev_b32_e32 v77, 4, v11
	v_and_b32_e32 v78, 0xf0f0f0f, v11
	v_cndmask_b32_e64 v11, 0, v151, s2
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v254, v[9:10], v[11:12] offset1:16
	v_cndmask_b32_e64 v11, 0, v198, s4
	v_perm_b32 v79, v77, v78, 0x5010400
	v_perm_b32 v77, v77, v78, 0x7030602
	v_cndmask_b32_e64 v10, 0, v146, s3
	v_cndmask_b32_e64 v9, 0, v145, s3
	v_lshrrev_b32_e32 v81, 4, v11
	v_and_b32_e32 v80, 0x7070707, v79
	v_lshrrev_b32_e32 v78, 1, v79
	v_and_b32_e32 v82, 0xf0f0f0f, v11
	v_and_b32_e32 v83, 0x7070707, v77
	v_and_b32_e32 v81, 0xf0f0f0f, v81
	v_perm_b32 v79, s26, 0x44403800, v80
	v_or_b32_e32 v80, 0x50505050, v80
	v_and_or_b32 v78, v78, s27, 0x3020100
	v_lshrrev_b32_e32 v85, 1, v77
	v_perm_b32 v84, v81, v82, 0x5010400
	v_perm_b32 v81, v81, v82, 0x7030602
	v_cndmask_b32_e64 v12, 0, v148, s3
	v_perm_b32 v77, v80, v79, v78
	v_perm_b32 v78, s26, 0x44403800, v83
	v_and_b32_e32 v80, 0x7070707, v84
	v_lshrrev_b32_e32 v84, 1, v84
	v_or_b32_e32 v79, 0x50505050, v83
	v_and_or_b32 v83, v85, s27, 0x3020100
	v_cndmask_b32_e64 v85, 0, v195, s5
	v_perm_b32 v82, s26, 0x44403800, v80
	v_or_b32_e32 v80, 0x50505050, v80
	v_and_or_b32 v84, v84, s27, 0x3020100
	v_perm_b32 v78, v79, v78, v83
	v_lshrrev_b32_e32 v86, 4, v85
	v_and_b32_e32 v83, 0x7070707, v81
	v_and_b32_e32 v85, 0xf0f0f0f, v85
	v_perm_b32 v79, v80, v82, v84
	v_cndmask_b32_e64 v80, 0, v196, s5
	v_and_b32_e32 v86, 0xf0f0f0f, v86
	v_lshrrev_b32_e32 v81, 1, v81
	v_perm_b32 v82, s26, 0x44403800, v83
	v_or_b32_e32 v83, 0x50505050, v83
	v_lshrrev_b32_e32 v87, 4, v80
	v_perm_b32 v84, v86, v85, 0x5010400
	v_and_or_b32 v81, v81, s27, 0x3020100
	v_and_b32_e32 v88, 0xf0f0f0f, v80
	v_perm_b32 v85, v86, v85, 0x7030602
	v_and_b32_e32 v87, 0xf0f0f0f, v87
	v_and_b32_e32 v86, 0x7070707, v84
	v_perm_b32 v80, v83, v82, v81
	v_lshrrev_b32_e32 v81, 1, v84
	v_and_b32_e32 v83, 0x7070707, v85
	v_perm_b32 v84, v87, v88, 0x5010400
	v_perm_b32 v87, v87, v88, 0x7030602
	v_lshrrev_b32_e32 v85, 1, v85
	v_perm_b32 v82, s26, 0x44403800, v86
	v_or_b32_e32 v86, 0x50505050, v86
	v_and_b32_e32 v90, 0x7070707, v84
	v_lshrrev_b32_e32 v84, 1, v84
	v_and_b32_e32 v91, 0x7070707, v87
	v_lshrrev_b32_e32 v87, 1, v87
	v_and_or_b32 v81, v81, s27, 0x3020100
	v_perm_b32 v89, s26, 0x44403800, v83
	v_or_b32_e32 v83, 0x50505050, v83
	v_and_or_b32 v85, v85, s27, 0x3020100
	v_perm_b32 v88, s26, 0x44403800, v90
	v_or_b32_e32 v90, 0x50505050, v90
	v_and_or_b32 v84, v84, s27, 0x3020100
	v_perm_b32 v93, s26, 0x44403800, v91
	v_or_b32_e32 v91, 0x50505050, v91
	v_and_or_b32 v87, v87, s27, 0x3020100
	v_perm_b32 v81, v86, v82, v81
	v_cvt_f32_f16_e32 v86, v92.l
	v_cndmask_b32_e64 v11, 0, v147, s3
	v_perm_b32 v82, v83, v89, v85
	v_perm_b32 v83, v90, v88, v84
	v_perm_b32 v84, v91, v93, v87
	v_cndmask_b32_e64 v85, 0, v107, s6
	v_cndmask_b32_e64 v86, 0, v86, s7
	ds_store_2addr_b64 v205, v[9:10], v[11:12] offset1:16
	ds_store_2addr_b64 v97, v[77:78], v[79:80] offset0:128 offset1:144
	ds_store_2addr_b64 v98, v[81:82], v[83:84] offset0:128 offset1:144
	ds_store_b32 v253, v85 offset:18432
	ds_store_b32 v252, v86 offset:18944
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	scratch_store_b32 off, v0, off offset:224 ; 4-byte Folded Spill
	s_barrier_wait -1
	s_wait_storecnt 0x0
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_8 Depth=1
	v_add_co_u32 v77, vcc_lo, v206, s25
	v_dual_mov_b32 v79, v105 :: v_dual_add_nc_u32 v104, 2, v104
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, 0, v207, vcc_lo
	s_mov_b32 s21, s1
	v_add_co_u32 v81, vcc_lo, v208, s25
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[28:29], s[10:11], s[20:21]
	scratch_store_b64 off, v[78:79], off offset:148 ; 8-byte Folded Spill
	v_lshlrev_b64_e32 v[79:80], 2, v[104:105]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v9, s21, s28, v203
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s29, 0, s21
	v_add_co_u32 v11, s21, s28, v204
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s29, 0, s21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, 0, v209, vcc_lo
	v_add_co_u32 v79, vcc_lo, s12, v79
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v80, null, s13, v80, vcc_lo
	s_clause 0x1
	global_load_b128 v[149:152], v[9:10], off
	global_load_b128 v[145:148], v[11:12], off
	s_clause 0x1
	global_load_b64 v[197:198], v[77:78], off offset:8
	global_load_b64 v[195:196], v[81:82], off offset:8
	global_load_b32 v0, v[79:80], off
	v_add_co_u32 v83, vcc_lo, v102, s25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, 0, v103, vcc_lo
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:296 ; 4-byte Folded Spill
	global_load_b32 v0, v[83:84], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:292 ; 4-byte Folded Spill
	s_branch .LBB0_11
.LBB0_10:                               ;   in Loop: Header=BB0_8 Depth=1
	s_clause 0x2                            ; 16-byte Folded Spill
	scratch_store_b32 off, v107, off offset:296
	scratch_store_b32 off, v106, off offset:292
	scratch_store_b64 off, v[104:105], off offset:148
.LBB0_11:                               ;   in Loop: Header=BB0_8 Depth=1
	ds_load_2addr_b64 v[9:12], v184 offset1:144
	ds_load_2addr_b64 v[77:80], v183 offset1:144
	ds_load_2addr_b64 v[199:202], v231 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[9:10], v[77:78], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[11:12], v[77:78], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[9:10], v[79:80], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[11:12], v[79:80], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[9:10], v[199:200], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[11:12], v[199:200], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[9:10], v[201:202], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[11:12], v[201:202], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[9:12], v184 offset0:36 offset1:180
	ds_load_2addr_b64 v[199:202], v183 offset0:36 offset1:180
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[9:10], v[199:200], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[11:12], v[199:200], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[9:10], v[201:202], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[11:12], v[201:202], v[93:100]
	ds_load_2addr_b64 v[199:202], v231 offset0:68 offset1:212
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[9:10], v[199:200], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[11:12], v[199:200], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[9:10], v[201:202], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[11:12], v[201:202], v[77:84]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[9:12], v184 offset0:72 offset1:216
	ds_load_2addr_b64 v[199:202], v183 offset0:72 offset1:216
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[9:10], v[199:200], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[11:12], v[199:200], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[9:10], v[201:202], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[11:12], v[201:202], v[93:100]
	ds_load_2addr_b64 v[199:202], v231 offset0:104 offset1:248
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[9:10], v[199:200], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[11:12], v[199:200], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[9:10], v[201:202], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[11:12], v[201:202], v[77:84]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[9:12], v184 offset0:108 offset1:252
	ds_load_2addr_b64 v[199:202], v183 offset0:108 offset1:252
	v_add_nc_u32_e32 v0, 0xc00, v183
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[9:10], v[199:200], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[11:12], v[199:200], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[9:10], v[201:202], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[11:12], v[201:202], v[93:100]
	ds_load_2addr_b64 v[199:202], v0 offset0:12 offset1:156
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[9:10], v[199:200], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[11:12], v[199:200], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[9:10], v[201:202], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[11:12], v[201:202], v[77:84]
	; sched_barrier mask(0x00000000)
	v_mov_b32_e32 v9, v141
	v_mov_b32_e32 v11, v143
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
	v_cndmask_b32_e64 v10, 0, v5, s4
	v_cndmask_b32_e64 v141, 0, v6, s4
	v_cndmask_b32_e64 v186, 0, v142, s2
	v_cndmask_b32_e64 v185, 0, v9, s2
	v_cndmask_b32_e64 v188, 0, v144, s2
	v_lshrrev_b32_e32 v12, 4, v10
	v_and_b32_e32 v10, 0xf0f0f0f, v10
	v_cndmask_b32_e64 v187, 0, v11, s2
	v_lshrrev_b32_e32 v189, 4, v141
	v_and_b32_e32 v141, 0xf0f0f0f, v141
	v_and_b32_e32 v12, 0xf0f0f0f, v12
	v_cndmask_b32_e64 v200, 0, v2, s3
	v_cndmask_b32_e64 v199, 0, v1, s3
	v_cndmask_b32_e64 v202, 0, v4, s3
	v_cndmask_b32_e64 v201, 0, v3, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off           ; 4-byte Folded Reload
	v_perm_b32 v143, v12, v10, 0x5010400
	v_perm_b32 v10, v12, v10, 0x7030602
	ds_store_2addr_b64 v254, v[185:186], v[187:188] offset1:16
	v_and_b32_e32 v186, 0xf0f0f0f, v189
	ds_store_2addr_b64 v205, v[199:200], v[201:202] offset1:16
	v_and_b32_e32 v190, 0x7070707, v143
	v_lshrrev_b32_e32 v12, 1, v143
	v_and_b32_e32 v185, 0x7070707, v10
	v_lshrrev_b32_e32 v10, 1, v10
	v_perm_b32 v189, v186, v141, 0x5010400
	v_perm_b32 v143, s26, 0x44403800, v190
	v_or_b32_e32 v187, 0x50505050, v190
	v_and_or_b32 v12, v12, s27, 0x3020100
	v_perm_b32 v188, s26, 0x44403800, v185
	v_or_b32_e32 v190, 0x50505050, v185
	v_and_or_b32 v10, v10, s27, 0x3020100
	v_perm_b32 v141, v186, v141, 0x7030602
	v_perm_b32 v185, v187, v143, v12
	v_and_b32_e32 v12, 0x7070707, v189
	v_lshrrev_b32_e32 v143, 1, v189
	v_perm_b32 v186, v190, v188, v10
	v_and_b32_e32 v188, 0x7070707, v141
	v_lshrrev_b32_e32 v141, 1, v141
	v_perm_b32 v187, s26, 0x44403800, v12
	v_or_b32_e32 v12, 0x50505050, v12
	v_and_or_b32 v143, v143, s27, 0x3020100
	v_perm_b32 v190, s26, 0x44403800, v188
	v_or_b32_e32 v188, 0x50505050, v188
	v_and_or_b32 v141, v141, s27, 0x3020100
	v_cndmask_b32_e64 v191, 0, v7, s5
	v_perm_b32 v187, v12, v187, v143
	v_cndmask_b32_e64 v143, 0, v8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	v_perm_b32 v188, v188, v190, v141
	v_lshrrev_b32_e32 v10, 4, v191
	v_and_b32_e32 v189, 0xf0f0f0f, v191
	v_lshrrev_b32_e32 v190, 4, v143
	v_and_b32_e32 v143, 0xf0f0f0f, v143
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v10, 0xf0f0f0f, v10
	v_and_b32_e32 v190, 0xf0f0f0f, v190
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v191, v10, v189, 0x5010400
	v_perm_b32 v10, v10, v189, 0x7030602
	v_and_b32_e32 v12, 0x7070707, v191
	v_lshrrev_b32_e32 v141, 1, v191
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v191, 0x7070707, v10
	v_lshrrev_b32_e32 v10, 1, v10
	v_perm_b32 v189, s26, 0x44403800, v12
	v_or_b32_e32 v12, 0x50505050, v12
	v_and_or_b32 v141, v141, s27, 0x3020100
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v10, v10, s27, 0x3020100
	v_perm_b32 v189, v12, v189, v141
	v_perm_b32 v12, s26, 0x44403800, v191
	v_perm_b32 v141, v190, v143, 0x5010400
	v_or_b32_e32 v191, 0x50505050, v191
	v_perm_b32 v143, v190, v143, 0x7030602
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v192, 0x7070707, v141
	v_perm_b32 v190, v191, v12, v10
	v_lshrrev_b32_e32 v10, 1, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v12, 0x7070707, v143
	v_lshrrev_b32_e32 v143, 1, v143
	v_perm_b32 v141, s26, 0x44403800, v192
	v_or_b32_e32 v191, 0x50505050, v192
	v_and_or_b32 v10, v10, s27, 0x3020100
	v_perm_b32 v192, s26, 0x44403800, v12
	v_or_b32_e32 v12, 0x50505050, v12
	v_and_or_b32 v143, v143, s27, 0x3020100
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v191, v191, v141, v10
	v_cndmask_b32_e64 v141, 0, 1, s0
	v_perm_b32 v192, v12, v192, v143
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v0, v[185:186], v[187:188] offset0:128 offset1:144
	scratch_load_b32 v0, off, off offset:4  ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v0, v[189:190], v[191:192] offset0:128 offset1:144
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_8 Depth=1
	s_mov_b32 s21, s1
	v_add_co_u32 v5, vcc_lo, v206, s25
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[28:29], s[10:11], s[20:21]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v207, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s0, s28, v203
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s29, 0, s0
	v_add_co_u32 v3, s0, s28, v204
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s29, 0, s0
	v_add_co_u32 v7, vcc_lo, v208, s25
	s_clause 0x1
	global_load_b128 v[9:12], v[1:2], off offset:64
	global_load_b128 v[1:4], v[3:4], off offset:64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v209, vcc_lo
	s_clause 0x1
	global_load_b64 v[5:6], v[5:6], off offset:40
	global_load_b64 v[7:8], v[7:8], off offset:40
	s_wait_loadcnt 0x3
	v_mov_b32_e32 v142, v10
	v_mov_b32_e32 v144, v12
.LBB0_13:                               ;   in Loop: Header=BB0_8 Depth=1
	ds_load_2addr_b64 v[199:202], v184 offset1:144
	ds_load_2addr_b64 v[203:206], v183 offset1:144
	ds_load_2addr_b64 v[207:210], v231 offset0:32 offset1:176
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[211:214], v184 offset0:36 offset1:180
	ds_load_2addr_b64 v[215:218], v183 offset0:36 offset1:180
	ds_load_2addr_b64 v[219:222], v231 offset0:68 offset1:212
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[223:226], v184 offset0:72 offset1:216
	ds_load_2addr_b64 v[227:230], v183 offset0:72 offset1:216
	ds_load_2addr_b64 v[231:234], v231 offset0:104 offset1:248
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v10, 0xc00, v183
	ds_load_2addr_b64 v[185:188], v184 offset0:108 offset1:252
	ds_load_2addr_b64 v[189:192], v183 offset0:108 offset1:252
	ds_load_2addr_b64 v[251:254], v10 offset0:12 offset1:156
	; sched_barrier mask(0x00000000)
	scratch_load_b32 v10, off, off offset:72 ; 4-byte Folded Reload
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[199:200], v[203:204], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[201:202], v[203:204], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[199:200], v[205:206], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[201:202], v[205:206], v[93:100]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[199:200], v[207:208], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[201:202], v[207:208], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[199:200], v[209:210], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[201:202], v[209:210], v[77:84]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[211:212], v[217:218], v[125:132]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[211:212], v[219:220], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[213:214], v[219:220], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[211:212], v[221:222], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[213:214], v[221:222], v[77:84]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[213:214], v[217:218], v[93:100]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[223:224], v[231:232], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[225:226], v[231:232], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[223:224], v[233:234], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[225:226], v[233:234], v[77:84]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[223:224], v[229:230], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[225:226], v[229:230], v[93:100]
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[211:212], v[215:216], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[213:214], v[215:216], v[101:108]
	v_cmp_ne_u32_e32 vcc_lo, 1, v141
	v_dual_mov_b32 v141, v150 :: v_dual_mov_b32 v12, v146
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[223:224], v[227:228], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[225:226], v[227:228], v[101:108]
	v_mov_b32_e32 v143, v152
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[117:124], v[185:186], v[251:252], v[117:124]
	v_wmma_f32_16x16x16_fp8_fp8 v[85:92], v[187:188], v[251:252], v[85:92]
	v_wmma_f32_16x16x16_fp8_fp8 v[133:140], v[185:186], v[189:190], v[133:140]
	v_wmma_f32_16x16x16_fp8_fp8 v[125:132], v[185:186], v[191:192], v[125:132]
	v_wmma_f32_16x16x16_fp8_fp8 v[109:116], v[185:186], v[253:254], v[109:116]
	v_wmma_f32_16x16x16_fp8_fp8 v[101:108], v[187:188], v[189:190], v[101:108]
	v_wmma_f32_16x16x16_fp8_fp8 v[77:84], v[187:188], v[253:254], v[77:84]
	v_wmma_f32_16x16x16_fp8_fp8 v[93:100], v[187:188], v[191:192], v[93:100]
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[201:202], v10 offset1:16
	ds_load_2addr_b32 v[199:200], v10 offset0:32 offset1:48
	scratch_load_b32 v10, off, off offset:8 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[233:234], v10 offset1:1
	scratch_load_b32 v10, off, off offset:12 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[231:232], v10 offset1:1
	scratch_load_b32 v10, off, off offset:16 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[229:230], v10 offset1:1
	scratch_load_b32 v10, off, off offset:20 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[227:228], v10 offset1:1
	scratch_load_b32 v10, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[225:226], v10 offset1:1
	scratch_load_b32 v10, off, off offset:28 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[223:224], v10 offset1:1
	scratch_load_b32 v10, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[221:222], v10 offset1:1
	scratch_load_b32 v10, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[219:220], v10 offset1:1
	scratch_load_b32 v10, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[217:218], v10 offset1:1
	scratch_load_b32 v10, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[215:216], v10 offset1:1
	scratch_load_b32 v10, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[213:214], v10 offset1:1
	scratch_load_b32 v10, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[211:212], v10 offset1:1
	scratch_load_b32 v10, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[209:210], v10 offset1:1
	scratch_load_b32 v10, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[207:208], v10 offset1:1
	scratch_load_b32 v10, off, off offset:64 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[205:206], v10 offset1:1
	scratch_load_b32 v10, off, off offset:68 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[203:204], v10 offset1:1
	v_mov_b32_e32 v10, v148
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
	scratch_load_b32 v185, off, off offset:292 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_clause 0x3                            ; 16-byte Folded Reload
	scratch_load_b32 v253, off, off offset:328
	scratch_load_b32 v189, off, off offset:332
	scratch_load_b32 v252, off, off offset:336
	scratch_load_b32 v254, off, off offset:340
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_7
; %bb.14:                               ;   in Loop: Header=BB0_8 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cndmask_b32_e64 v146, 0, v197, s4
	v_cndmask_b32_e64 v150, 0, v141, s2
	v_cndmask_b32_e64 v149, 0, v149, s2
	v_cndmask_b32_e64 v152, 0, v143, s2
	v_cndmask_b32_e64 v151, 0, v151, s2
	v_lshrrev_b32_e32 v148, 4, v146
	v_and_b32_e32 v141, 0xf0f0f0f, v146
	v_mov_b32_e32 v197, v185
	v_cndmask_b32_e64 v146, 0, v12, s3
	v_cndmask_b32_e64 v145, 0, v145, s3
	v_and_b32_e32 v143, 0xf0f0f0f, v148
	v_cndmask_b32_e64 v148, 0, v10, s3
	v_cndmask_b32_e64 v147, 0, v147, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_perm_b32 v185, v143, v141, 0x5010400
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v254, v[149:150], v[151:152] offset1:16
	v_cndmask_b32_e64 v149, 0, v198, s4
	v_and_b32_e32 v12, 0x7070707, v185
	v_perm_b32 v141, v143, v141, 0x7030602
	v_lshrrev_b32_e32 v143, 1, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b32_e32 v10, 4, v149
	v_and_b32_e32 v151, 0xf0f0f0f, v149
	v_perm_b32 v150, s26, 0x44403800, v12
	v_or_b32_e32 v12, 0x50505050, v12
	v_and_or_b32 v143, v143, s27, 0x3020100
	v_and_b32_e32 v10, 0xf0f0f0f, v10
	v_and_b32_e32 v152, 0x7070707, v141
	v_lshrrev_b32_e32 v141, 1, v141
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_perm_b32 v149, v12, v150, v143
	v_perm_b32 v185, v10, v151, 0x5010400
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v12, s26, 0x44403800, v152
	v_or_b32_e32 v143, 0x50505050, v152
	v_and_or_b32 v141, v141, s27, 0x3020100
	v_cndmask_b32_e64 v152, 0, v195, s5
	v_and_b32_e32 v150, 0x7070707, v185
	v_lshrrev_b32_e32 v185, 1, v185
	v_perm_b32 v10, v10, v151, 0x7030602
	v_lshrrev_b32_e32 v195, v189, v197
	v_lshrrev_b32_e32 v187, 4, v152
	v_perm_b32 v151, s26, 0x44403800, v150
	v_or_b32_e32 v186, 0x50505050, v150
	v_perm_b32 v150, v143, v12, v141
	v_and_b32_e32 v141, 0xf0f0f0f, v152
	v_cndmask_b32_e64 v152, 0, v196, s5
	v_and_or_b32 v185, v185, s27, 0x3020100
	v_and_b32_e32 v12, 0x7070707, v10
	v_and_b32_e32 v143, 0xf0f0f0f, v187
	v_lshrrev_b32_e32 v10, 1, v10
	v_lshrrev_b32_e32 v187, 4, v152
	v_perm_b32 v151, v186, v151, v185
	v_perm_b32 v185, s26, 0x44403800, v12
	v_perm_b32 v186, v143, v141, 0x5010400
	v_or_b32_e32 v12, 0x50505050, v12
	v_and_or_b32 v10, v10, s27, 0x3020100
	v_perm_b32 v141, v143, v141, 0x7030602
	v_and_b32_e32 v188, 0xf0f0f0f, v152
	v_and_b32_e32 v187, 0xf0f0f0f, v187
	v_and_b32_e32 v143, 0x7070707, v186
	v_perm_b32 v152, v12, v185, v10
	v_lshrrev_b32_e32 v10, 1, v186
	v_and_b32_e32 v185, 0x7070707, v141
	v_perm_b32 v186, v187, v188, 0x5010400
	v_lshrrev_b32_e32 v141, 1, v141
	v_perm_b32 v187, v187, v188, 0x7030602
	v_perm_b32 v12, s26, 0x44403800, v143
	v_perm_b32 v0, s26, 0x44403800, v185
	v_and_b32_e32 v190, 0x7070707, v186
	v_or_b32_e32 v188, 0x50505050, v185
	v_and_or_b32 v141, v141, s27, 0x3020100
	v_lshrrev_b32_e32 v185, 1, v186
	v_and_b32_e32 v186, 0x7070707, v187
	v_lshrrev_b32_e32 v187, 1, v187
	v_or_b32_e32 v143, 0x50505050, v143
	v_and_or_b32 v10, v10, s27, 0x3020100
	v_perm_b32 v191, s26, 0x44403800, v190
	v_perm_b32 v196, s26, 0x44403800, v186
	v_or_b32_e32 v197, 0x50505050, v186
	v_perm_b32 v186, v188, v0, v141
	scratch_load_b32 v141, off, off offset:300 ; 4-byte Folded Reload
	v_or_b32_e32 v190, 0x50505050, v190
	v_and_or_b32 v192, v185, s27, 0x3020100
	v_and_or_b32 v198, v187, s27, 0x3020100
	v_perm_b32 v185, v143, v12, v10
	v_cvt_f32_f16_e64 v10, v195.l
	v_cndmask_b32_e64 v12, 0, v251, s6
	v_perm_b32 v187, v190, v191, v192
	v_perm_b32 v188, v197, v196, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v10, 0, v10, s7
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v141, v[145:146], v[147:148] offset1:16
	scratch_load_b32 v141, off, off         ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v141, v[149:150], v[151:152] offset0:128 offset1:144
	scratch_load_b32 v141, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v141, v[185:186], v[187:188] offset0:128 offset1:144
	ds_store_b32 v253, v12 offset:18432
	ds_store_b32 v252, v10 offset:18944
	s_branch .LBB0_7
.LBB0_15:
	s_clause 0x5                            ; 24-byte Folded Reload
	scratch_load_b32 v10, off, off offset:380 th:TH_LOAD_LU
	scratch_load_b32 v0, off, off offset:360
	scratch_load_b32 v19, off, off offset:364
	scratch_load_b32 v18, off, off offset:368
	scratch_load_b32 v25, off, off offset:372
	scratch_load_b32 v12, off, off offset:376
.LBB0_16:
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v1, 6, v0
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v2, v10, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 0x3800, v1
	v_and_b32_e32 v3, 31, v2
	v_add_nc_u32_e32 v5, 18, v2
	v_add_nc_u32_e32 v6, 19, v2
	v_add_nc_u32_e32 v8, 20, v2
	v_add_nc_u32_e32 v4, 0, v1
	v_add_nc_u32_e32 v1, 17, v2
	v_xor_b32_e32 v3, 16, v3
	v_add_nc_u32_e32 v13, 21, v2
	v_and_b32_e32 v5, 31, v5
	v_lshl_add_u32 v12, v12, 7, v4
	v_and_b32_e32 v1, 31, v1
	v_add_nc_u32_e32 v14, 22, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v7, v2, 2, v12
	v_lshl_add_u32 v10, v1, 2, v12
	v_add_nc_u32_e32 v1, 23, v2
	v_and_b32_e32 v2, 31, v6
	v_lshl_add_u32 v9, v3, 2, v12
	v_and_b32_e32 v3, 31, v8
	v_lshl_add_u32 v11, v5, 2, v12
	v_and_b32_e32 v5, 31, v13
	v_and_b32_e32 v6, 31, v14
	v_and_b32_e32 v1, 31, v1
	v_lshl_add_u32 v20, v2, 2, v12
	v_lshl_add_u32 v21, v3, 2, v12
	v_lshl_add_u32 v22, v5, 2, v12
	v_lshl_add_u32 v23, v6, 2, v12
	v_lshl_add_u32 v24, v1, 2, v12
	ds_store_2addr_b32 v7, v181, v182 offset1:1
	ds_store_2addr_b32 v7, v179, v180 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v249, v250 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v247, v248 offset0:6 offset1:7
	ds_store_b32 v9, v246
	ds_store_b32 v10, v245
	ds_store_b32 v11, v244
	ds_store_b32 v20, v243
	ds_store_b32 v21, v242
	ds_store_b32 v22, v241
	ds_store_b32 v23, v240
	ds_store_b32 v24, v239
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_or3_b32 v3, s22, v25, v19
	v_add_nc_u32_e32 v1, s19, v18
	v_lshl_add_u32 v5, v19, 2, v4
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
	s_cbranch_execz .LBB0_18
; %bb.17:
	v_mad_co_u64_u32 v[12:13], null, v1, s16, v[3:4]
	v_mov_b32_e32 v13, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v14, s0, s14, v14
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s15, v15, s0
	s_wait_kmcnt 0x0
	v_add_co_u32 v12, s0, s8, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s9, v13, s0
	global_load_b32 v6, v[14:15], off
	global_load_b32 v8, v[12:13], off
	ds_load_b32 v14, v5
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v6, v14
	global_store_b32 v[12:13], v8, off
.LBB0_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v8, 1, v1
	v_add_nc_u32_e32 v6, 1, v0
	s_xor_b32 s1, vcc_lo, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s0, s18, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_mad_co_u64_u32 v[12:13], null, v8, s16, v[3:4]
	v_mov_b32_e32 v13, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v14, vcc_lo, s14, v14
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s15, v15, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v12, vcc_lo, s8, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s9, v13, vcc_lo
	global_load_b32 v8, v[14:15], off offset:4
	global_load_b32 v14, v[12:13], off
	v_and_b32_e32 v15, 31, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v15, v15, 2, v4
	ds_load_b32 v15, v15 offset:128
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v14, v8, v15
	global_store_b32 v[12:13], v14, off
.LBB0_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v12, 2, v1
	v_add_nc_u32_e32 v8, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v12
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_mad_co_u64_u32 v[12:13], null, v12, s16, v[3:4]
	v_dual_mov_b32 v13, 0 :: v_dual_and_b32 v16, 31, v8
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v16, v16, 2, v4
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v14, vcc_lo, s14, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s15, v15, vcc_lo
	ds_load_b32 v16, v16 offset:256
	s_wait_kmcnt 0x0
	v_add_co_u32 v12, vcc_lo, s8, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s9, v13, vcc_lo
	global_load_b32 v14, v[14:15], off offset:8
	global_load_b32 v15, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v15, v14, v16
	global_store_b32 v[12:13], v15, off
.LBB0_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v13, 3, v1
	v_add_nc_u32_e32 v12, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v13
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_mad_co_u64_u32 v[13:14], null, v13, s16, v[3:4]
	v_dual_mov_b32 v14, 0 :: v_dual_and_b32 v17, 31, v12
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v17, v17, 2, v4
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v15, vcc_lo, s14, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s15, v16, vcc_lo
	ds_load_b32 v17, v17 offset:384
	s_wait_kmcnt 0x0
	v_add_co_u32 v13, vcc_lo, s8, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s9, v14, vcc_lo
	global_load_b32 v15, v[15:16], off offset:12
	global_load_b32 v16, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v16, v15, v17
	global_store_b32 v[13:14], v16, off
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v14, 4, v1
	v_add_nc_u32_e32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v14
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_mad_co_u64_u32 v[14:15], null, v14, s16, v[3:4]
	v_dual_mov_b32 v15, 0 :: v_dual_and_b32 v18, 31, v13
	v_lshlrev_b64_e32 v[16:17], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v18, v18, 2, v4
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v16, vcc_lo, s14, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s15, v17, vcc_lo
	ds_load_b32 v18, v18 offset:512
	s_wait_kmcnt 0x0
	v_add_co_u32 v14, vcc_lo, s8, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s9, v15, vcc_lo
	global_load_b32 v16, v[16:17], off offset:16
	global_load_b32 v17, v[14:15], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v17, v16, v18
	global_store_b32 v[14:15], v17, off
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v15, 5, v1
	v_add_nc_u32_e32 v14, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v15
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_mad_co_u64_u32 v[15:16], null, v15, s16, v[3:4]
	v_dual_mov_b32 v16, 0 :: v_dual_and_b32 v19, 31, v14
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v19, v19, 2, v4
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v17, vcc_lo, s14, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s15, v18, vcc_lo
	ds_load_b32 v19, v19 offset:640
	s_wait_kmcnt 0x0
	v_add_co_u32 v15, vcc_lo, s8, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s9, v16, vcc_lo
	global_load_b32 v17, v[17:18], off offset:20
	global_load_b32 v18, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v18, v17, v19
	global_store_b32 v[15:16], v18, off
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v16, 6, v1
	v_add_nc_u32_e32 v15, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v16
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_mad_co_u64_u32 v[16:17], null, v16, s16, v[3:4]
	v_mov_b32_e32 v17, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	v_and_b32_e32 v25, 31, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v25, v25, 2, v4
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v18, vcc_lo, s14, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s15, v19, vcc_lo
	ds_load_b32 v25, v25 offset:768
	s_wait_kmcnt 0x0
	v_add_co_u32 v16, vcc_lo, s8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s9, v17, vcc_lo
	global_load_b32 v18, v[18:19], off offset:24
	global_load_b32 v19, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v19, v18, v25
	global_store_b32 v[16:17], v19, off
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v17, 7, v1
	v_add_nc_u32_e32 v16, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v17
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_mad_co_u64_u32 v[17:18], null, v17, s16, v[3:4]
	v_mov_b32_e32 v18, 0
	v_lshlrev_b64_e32 v[25:26], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v25, vcc_lo, s14, v25
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s15, v26, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v17, vcc_lo, s8, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s9, v18, vcc_lo
	global_load_b32 v19, v[25:26], off offset:28
	global_load_b32 v25, v[17:18], off
	v_and_b32_e32 v26, 31, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v26, v26, 2, v4
	ds_load_b32 v26, v26 offset:896
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v25, v19, v26
	global_store_b32 v[17:18], v25, off
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 8, v1
	v_add_nc_u32_e32 v17, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v18
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_and_b32_e32 v27, 31, v17
	v_mad_co_u64_u32 v[18:19], null, v18, s16, v[3:4]
	v_lshlrev_b64_e32 v[25:26], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v27, v27, 2, v4
	v_add_co_u32 v25, vcc_lo, s14, v25
	ds_load_b32 v27, v27 offset:1024
	v_mov_b32_e32 v19, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s15, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_wait_kmcnt 0x0
	v_add_co_u32 v18, vcc_lo, s8, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v19, null, s9, v19, vcc_lo
	global_load_b32 v25, v[25:26], off offset:32
	global_load_b32 v26, v[18:19], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v26, v25, v27
	global_store_b32 v[18:19], v26, off
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v19, 9, v1
	v_add_nc_u32_e32 v18, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v19
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_mad_co_u64_u32 v[25:26], null, v19, s16, v[3:4]
	v_mov_b32_e32 v26, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v27, vcc_lo, s14, v27
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v25, vcc_lo, s8, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s9, v26, vcc_lo
	global_load_b32 v19, v[27:28], off offset:36
	global_load_b32 v27, v[25:26], off
	v_and_b32_e32 v28, 31, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v28, v28, 2, v4
	ds_load_b32 v28, v28 offset:1152
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v27, v19, v28
	global_store_b32 v[25:26], v27, off
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 10, v1
	v_add_nc_u32_e32 v19, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_mad_co_u64_u32 v[25:26], null, v25, s16, v[3:4]
	v_dual_mov_b32 v26, 0 :: v_dual_and_b32 v29, 31, v19
	v_lshlrev_b64_e32 v[27:28], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v29, v29, 2, v4
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v27, vcc_lo, s14, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s15, v28, vcc_lo
	ds_load_b32 v29, v29 offset:1280
	s_wait_kmcnt 0x0
	v_add_co_u32 v25, vcc_lo, s8, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s9, v26, vcc_lo
	global_load_b32 v27, v[27:28], off offset:40
	global_load_b32 v28, v[25:26], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v28, v27, v29
	global_store_b32 v[25:26], v28, off
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 11, v1
	v_add_nc_u32_e32 v25, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v26
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_mad_co_u64_u32 v[26:27], null, v26, s16, v[3:4]
	v_dual_mov_b32 v27, 0 :: v_dual_and_b32 v30, 31, v25
	v_lshlrev_b64_e32 v[28:29], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v30, v30, 2, v4
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v28, vcc_lo, s14, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s15, v29, vcc_lo
	ds_load_b32 v30, v30 offset:1408
	s_wait_kmcnt 0x0
	v_add_co_u32 v26, vcc_lo, s8, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s9, v27, vcc_lo
	global_load_b32 v28, v[28:29], off offset:44
	global_load_b32 v29, v[26:27], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v29, v28, v30
	global_store_b32 v[26:27], v29, off
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v27, 12, v1
	v_add_nc_u32_e32 v26, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v27
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_mad_co_u64_u32 v[27:28], null, v27, s16, v[3:4]
	v_dual_mov_b32 v28, 0 :: v_dual_and_b32 v31, 31, v26
	v_lshlrev_b64_e32 v[29:30], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v31, v31, 2, v4
	v_lshlrev_b64_e32 v[27:28], 2, v[27:28]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v29, vcc_lo, s14, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s15, v30, vcc_lo
	ds_load_b32 v31, v31 offset:1536
	s_wait_kmcnt 0x0
	v_add_co_u32 v27, vcc_lo, s8, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s9, v28, vcc_lo
	global_load_b32 v29, v[29:30], off offset:48
	global_load_b32 v30, v[27:28], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v30, v29, v31
	global_store_b32 v[27:28], v30, off
.LBB0_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 13, v1
	v_add_nc_u32_e32 v27, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v27
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1664
	s_wait_kmcnt 0x0
	v_add_co_u32 v28, vcc_lo, s8, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s9, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:52
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v29, 14, v1
	v_add_nc_u32_e32 v28, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v29
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_mad_co_u64_u32 v[29:30], null, v29, s16, v[3:4]
	v_dual_mov_b32 v30, 0 :: v_dual_and_b32 v33, 31, v28
	v_lshlrev_b64_e32 v[31:32], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v33, v33, 2, v4
	v_lshlrev_b64_e32 v[29:30], 2, v[29:30]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v31, vcc_lo, s14, v31
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v32, null, s15, v32, vcc_lo
	ds_load_b32 v33, v33 offset:1792
	s_wait_kmcnt 0x0
	v_add_co_u32 v29, vcc_lo, s8, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s9, v30, vcc_lo
	global_load_b32 v31, v[31:32], off offset:56
	global_load_b32 v32, v[29:30], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v31, v33
	global_store_b32 v[29:30], v32, off
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v30, 15, v1
	v_add_nc_u32_e32 v29, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v30
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_mad_co_u64_u32 v[30:31], null, v30, s16, v[3:4]
	v_dual_mov_b32 v31, 0 :: v_dual_and_b32 v34, 31, v29
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v34, v34, 2, v4
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v32, vcc_lo, s14, v32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	ds_load_b32 v34, v34 offset:1920
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v32, v[32:33], off offset:60
	global_load_b32 v33, v[30:31], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v33, v32, v34
	global_store_b32 v[30:31], v33, off
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v237, v238 offset1:1
	ds_store_2addr_b32 v7, v193, v236 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v178, v235 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v176, v177 offset0:6 offset1:7
	ds_store_b32 v9, v175
	ds_store_b32 v10, v174
	ds_store_b32 v11, v173
	ds_store_b32 v20, v172
	ds_store_b32 v21, v171
	ds_store_b32 v22, v168
	ds_store_b32 v23, v167
	ds_store_b32 v24, v166
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:64
	global_load_b32 v32, v[30:31], off
	ds_load_b32 v33, v5
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 17, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:68
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:128
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:72
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:256
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 19, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:76
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:384
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 20, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:80
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:512
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 21, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:84
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:640
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 22, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:88
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:768
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 23, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:92
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:896
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 24, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:96
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1024
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 25, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:100
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1152
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 26, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:104
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1280
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 27, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:108
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1408
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 28, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:112
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1536
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 29, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:116
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1664
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 30, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_78
; %bb.77:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:120
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1792
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 31, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:124
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1920
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v164, v165 offset1:1
	ds_store_2addr_b32 v7, v162, v163 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v158, v161 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v156, v157 offset0:6 offset1:7
	ds_store_b32 v9, v155
	ds_store_b32 v10, v154
	ds_store_b32 v11, v153
	ds_store_b32 v20, v255
	ds_store_b32 v21, v44
	ds_store_b32 v22, v43
	ds_store_b32 v23, v194
	ds_store_b32 v24, v49
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:128
	global_load_b32 v32, v[30:31], off
	ds_load_b32 v33, v5
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 33, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:132
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:128
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 34, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:136
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:256
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 35, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:140
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:384
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 36, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:144
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:512
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 37, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_92
; %bb.91:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:148
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:640
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 38, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:152
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:768
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 39, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_96
; %bb.95:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:156
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:896
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 40, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:160
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1024
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 41, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:164
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1152
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 42, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:168
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1280
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 43, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:172
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1408
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 44, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:176
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1536
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 45, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:180
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1664
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 46, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:184
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1792
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 47, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_mad_co_u64_u32 v[30:31], null, v0, s16, v[3:4]
	v_mov_b32_e32 v31, 0
	v_lshlrev_b64_e32 v[32:33], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v32, vcc_lo, s14, v32
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v33, null, s15, v33, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v30, vcc_lo, s8, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s9, v31, vcc_lo
	global_load_b32 v0, v[32:33], off offset:188
	global_load_b32 v32, v[30:31], off
	v_and_b32_e32 v33, 31, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v33, v33, 2, v4
	ds_load_b32 v33, v33 offset:1920
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v32, v0, v33
	global_store_b32 v[30:31], v32, off
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v55, v56 offset1:1
	ds_store_2addr_b32 v7, v53, v54 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v47, v48 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v45, v46 offset0:6 offset1:7
	ds_store_b32 v9, v42
	ds_store_b32 v10, v41
	ds_store_b32 v11, v40
	ds_store_b32 v20, v39
	ds_store_b32 v21, v38
	ds_store_b32 v22, v37
	ds_store_b32 v23, v36
	ds_store_b32 v24, v35
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_114
; %bb.113:
	v_mad_co_u64_u32 v[9:10], null, v0, s16, v[3:4]
	ds_load_b32 v5, v5
	v_mov_b32_e32 v10, 0
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v20, vcc_lo, s14, v20
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v21, null, s15, v21, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v0, v[20:21], off offset:192
	global_load_b32 v7, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v5
	global_store_b32 v[9:10], v7, off
.LBB0_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 49, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_116
; %bb.115:
	v_and_b32_e32 v6, 31, v6
	v_mad_co_u64_u32 v[9:10], null, v0, s16, v[3:4]
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v6, v6, 2, v4
	v_add_co_u32 v20, vcc_lo, s14, v20
	ds_load_b32 v6, v6 offset:128
	v_mov_b32_e32 v10, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s15, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_wait_kmcnt 0x0
	v_add_co_u32 v9, vcc_lo, s8, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s9, v10, vcc_lo
	global_load_b32 v0, v[20:21], off offset:196
	global_load_b32 v5, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v5, v0, v6
	global_store_b32 v[9:10], v5, off
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_and_b32_e32 v8, 31, v8
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v8, v8, 2, v4
	v_add_co_u32 v9, vcc_lo, s14, v9
	ds_load_b32 v8, v8 offset:256
	v_mov_b32_e32 v6, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s15, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[9:10], off offset:200
	global_load_b32 v7, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 51, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:204
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:384
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 52, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_122
; %bb.121:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:208
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:512
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 53, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_124
; %bb.123:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:212
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:640
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 54, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_126
; %bb.125:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:216
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:768
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 55, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_128
; %bb.127:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:220
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:896
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 56, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_130
; %bb.129:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:224
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:1024
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 57, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_132
; %bb.131:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:228
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:1152
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 58, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_134
; %bb.133:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:232
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:1280
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 59, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_136
; %bb.135:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:236
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:1408
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_138
; %bb.137:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:240
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:1536
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 61, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_140
; %bb.139:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:244
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:1664
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 62, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, s14, v7
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s15, v8, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v5, vcc_lo, s8, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s9, v6, vcc_lo
	global_load_b32 v0, v[7:8], off offset:248
	global_load_b32 v7, v[5:6], off
	v_and_b32_e32 v8, 31, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v8, v8, 2, v4
	ds_load_b32 v8, v8 offset:1792
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v7, v0, v8
	global_store_b32 v[5:6], v7, off
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v0, 63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v0
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_mad_co_u64_u32 v[5:6], null, v0, s16, v[3:4]
	v_mov_b32_e32 v6, 0
	v_lshlrev_b64_e32 v[0:1], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s14, v0
	v_lshlrev_b64_e32 v[2:3], 2, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	v_and_b32_e32 v5, 31, v29
	s_wait_kmcnt 0x0
	v_add_co_u32 v2, vcc_lo, s8, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s9, v3, vcc_lo
	global_load_b32 v0, v[0:1], off offset:252
	global_load_b32 v1, v[2:3], off
	v_lshl_add_u32 v4, v5, 2, v4
	ds_load_b32 v4, v4 offset:1920
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v1, v0, v4
	global_store_b32 v[2:3], v1, off
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end0:
	.size	re_r2, .Lfunc_end0-re_r2
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel re_r2
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 388
		.amdhsa_kernarg_size 52
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
		.amdhsa_next_free_sgpr 30
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-re_r2)<<4)&4080)>>4
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
	.set .Lre_r2.num_vgpr, 256
	.set .Lre_r2.num_agpr, 0
	.set .Lre_r2.numbered_sgpr, 30
	.set .Lre_r2.num_named_barrier, 0
	.set .Lre_r2.private_seg_size, 388
	.set .Lre_r2.uses_vcc, 1
	.set .Lre_r2.uses_flat_scratch, 1
	.set .Lre_r2.has_dyn_sized_stack, 0
	.set .Lre_r2.has_recursion, 0
	.set .Lre_r2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 24160
; TotalNumSgprs: 32
; NumVgprs: 256
; ScratchSize: 388
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 32
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
	.type	__hip_cuid_e924e1524695de99,@object ; @__hip_cuid_e924e1524695de99
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e924e1524695de99
__hip_cuid_e924e1524695de99:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e924e1524695de99, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e924e1524695de99
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
    .name:           re_r2
    .private_segment_fixed_size: 388
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         re_r2.kd
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
