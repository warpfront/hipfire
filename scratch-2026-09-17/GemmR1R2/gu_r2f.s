	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gu_r2f                  ; -- Begin function gu_r2f
	.globl	gu_r2f
	.p2align	8
	.type	gu_r2f,@function
gu_r2f:                                 ; @gu_r2f
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b256 s[16:23], s[0:1], 0x0
	s_load_b128 s[24:27], s[0:1], 0x38
	v_mov_b32_e32 v47, v0
	s_lshl_b32 s30, ttmp9, 7
	s_lshl_b32 s28, ttmp7, 7
	s_load_b256 s[8:15], s[0:1], 0x20
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v88, 0
	v_and_b32_e32 v1, 3, v47
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v190, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v254, 0
	v_dual_mov_b32 v255, 0 :: v_dual_mov_b32 v132, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v86, 0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v10, s17
	v_lshlrev_b32_e32 v2, 4, v1
	v_lshrrev_b32_e32 v3, 2, v47
	s_add_co_i32 s29, s25, s24
	s_ashr_i32 s2, s26, 31
	s_add_co_i32 s3, s29, -1
	s_add_co_i32 s5, s27, -1
	v_or_b32_e32 v5, s30, v3
	v_or_b32_e32 v4, s28, v3
	v_mad_u32_u24 v8, 0x48, v3, v2
	s_lshr_b32 s4, s2, 24
	v_and_b32_e32 v7, 0x7f, v47
	v_min_i32_e32 v9, s3, v5
	v_or_b32_e32 v3, 64, v4
	v_or_b32_e32 v6, 64, v5
	v_cmp_gt_i32_e64 s0, s27, v4
	v_min_i32_e32 v4, s5, v4
	v_cmp_gt_i32_e32 vcc_lo, s24, v9
	v_cmp_gt_i32_e64 s1, s27, v3
	v_min_i32_e32 v3, s5, v3
	s_add_co_i32 s4, s26, s4
	v_cmp_gt_i32_e64 s2, s29, v5
	v_cndmask_b32_e64 v12, s24, 0, vcc_lo
	v_min_i32_e32 v5, s3, v6
	v_lshrrev_b32_e32 v0, 1, v47
	v_mov_b32_e32 v11, s16
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s16, s4, 8
	v_sub_nc_u32_e32 v9, v9, v12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s6, s16, 0x88
	v_mad_co_u64_u32 v[129:130], null, v4, s26, v[2:3]
	v_mad_co_u64_u32 v[130:131], null, v3, s26, v[2:3]
	v_cmp_gt_i32_e64 s4, s24, v5
	v_mul_lo_u32 v2, s6, v9
	v_or_b32_e32 v9, s30, v7
	v_dual_cndmask_b32 v4, s18, v11 :: v_dual_lshlrev_b32 v1, 3, v1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, s24, 0, s4
	v_dual_cndmask_b32 v12, s19, v10 :: v_dual_lshlrev_b32 v7, 3, v7
	v_min_i32_e32 v13, s3, v9
	v_add_co_u32 v2, vcc_lo, v4, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_sub_nc_u32_e32 v5, v5, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s24, v13
	v_or_b32_e32 v12, s28, v0
	v_mul_lo_u32 v4, s6, v5
	v_cmp_gt_i32_e64 s3, s29, v6
	v_cndmask_b32_e64 v6, s18, v11, s4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, s24, 0, vcc_lo
	v_cndmask_b32_e64 v14, s19, v10, s4
	v_min_i32_e32 v15, s5, v12
	v_cmp_gt_i32_e64 s5, s29, v9
	v_add_nc_u32_e32 v134, 0, v8
	v_sub_nc_u32_e32 v13, v13, v5
	v_add_co_u32 v4, s4, v6, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, 0, v14, s4
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_lo_u32 v13, s6, v13
	v_lshrrev_b32_e32 v14, 7, v47
	v_cmp_gt_i32_e64 s4, s27, v12
	v_lshlrev_b32_e32 v12, 2, v0
	v_mul_lo_u32 v6, v15, s16
	v_cndmask_b32_e32 v11, s18, v11, vcc_lo
	v_cndmask_b32_e32 v10, s19, v10, vcc_lo
	v_lshl_or_b32 v7, v14, 2, v7
	v_add_nc_u32_e32 v9, 0, v12
	v_mov_b32_e32 v131, 0
	v_add_co_u32 v101, vcc_lo, v11, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v102, null, 0, v10, vcc_lo
	v_lshlrev_b32_e32 v103, 1, v6
	v_add_nc_u32_e32 v7, 0, v7
	scratch_store_b32 off, v9, off offset:256 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v9, 4, v14
	s_cmp_gt_i32 s26, 0xff
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s26, 0x100
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v9, off offset:260
	scratch_store_b32 off, v7, off offset:264
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	v_dual_mov_b32 v20, 0 :: v_dual_lshlrev_b32 v19, 1, v6
	v_add_co_u32 v15, vcc_lo, v2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]
	v_add_co_u32 v17, vcc_lo, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, 0, v5, vcc_lo
	global_load_b64 v[21:22], v[15:16], off offset:8
	global_load_b64 v[23:24], v[17:18], off offset:8
	v_add_co_u32 v19, vcc_lo, s22, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, s23, v20, vcc_lo
	global_load_b32 v6, v[19:20], off
	global_load_b64 v[254:255], v[15:16], off offset:40
	global_load_b64 v[131:132], v[17:18], off offset:40
	scratch_load_b32 v15, off, off offset:260 ; 4-byte Folded Reload
	global_load_b32 v25, v[101:102], off
	s_clause 0x1
	global_load_b128 v[7:10], v129, s[20:21]
	global_load_b128 v[11:14], v130, s[20:21]
	s_mov_b32 s7, 0x4e4c4a48
	s_mov_b32 s14, 0x4040404
	s_clause 0x1
	global_load_b128 v[90:93], v129, s[20:21] offset:64
	global_load_b128 v[86:89], v130, s[20:21] offset:64
	v_add_nc_u32_e32 v26, 0x1200, v134
	v_add_nc_u32_e32 v27, 0x2400, v134
	v_add_nc_u32_e32 v28, 0x3600, v134
	s_wait_loadcnt 0xa
	v_cndmask_b32_e64 v16, 0, v21, s2
	v_cndmask_b32_e64 v17, 0, v22, s2
	s_wait_loadcnt 0x9
	v_cndmask_b32_e64 v18, 0, v23, s3
	v_cndmask_b32_e64 v19, 0, v24, s3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v20, 0xf0f0f0f, v17
	v_lshrrev_b32_e32 v17, 4, v17
	v_and_b32_e32 v21, 0xf0f0f0f, v18
	v_lshrrev_b32_e32 v18, 4, v18
	s_wait_loadcnt 0x4
	v_lshrrev_b32_e32 v15, v15, v25
	v_lshrrev_b32_e32 v22, 4, v19
	v_and_b32_e32 v19, 0xf0f0f0f, v19
	v_and_b32_e32 v17, 0xf0f0f0f, v17
	v_and_b32_e32 v18, 0xf0f0f0f, v18
	v_cvt_f32_f16_e32 v15, v15.l
	v_and_b32_e32 v22, 0xf0f0f0f, v22
	v_cndmask_b32_e64 v6, 0, v6, s4
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v85, v87 :: v_dual_mov_b32 v190, v89
	v_cndmask_b32_e64 v23, 0, v15, s5
	v_and_b32_e32 v15, 0xf0f0f0f, v16
	v_lshrrev_b32_e32 v16, 4, v16
	v_cndmask_b32_e64 v12, 0, v12, s1
	v_cndmask_b32_e64 v11, 0, v11, s1
	v_cndmask_b32_e64 v14, 0, v14, s1
	v_cndmask_b32_e64 v13, 0, v13, s1
	v_and_b32_e32 v16, 0xf0f0f0f, v16
	v_cndmask_b32_e64 v8, 0, v8, s0
	v_cndmask_b32_e64 v7, 0, v7, s0
	v_cndmask_b32_e64 v10, 0, v10, s0
	v_cndmask_b32_e64 v9, 0, v9, s0
	v_perm_b32 v24, v16, v15, 0x5010400
	v_perm_b32 v15, v16, v15, 0x7030602
	v_perm_b32 v16, v17, v20, 0x5010400
	v_perm_b32 v17, v17, v20, 0x7030602
	v_perm_b32 v20, v18, v21, 0x5010400
	v_perm_b32 v18, v18, v21, 0x7030602
	v_perm_b32 v21, v22, v19, 0x5010400
	v_perm_b32 v19, v22, v19, 0x7030602
	scratch_load_b32 v22, off, off offset:256 ; 4-byte Folded Reload
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
	v_perm_b32 v17, v25, v37, v38
	v_perm_b32 v18, v29, v39, v40
	v_perm_b32 v19, v30, v41, v20
	v_perm_b32 v20, v31, v42, v43
	v_perm_b32 v21, v32, v44, v21
	s_wait_loadcnt 0x0
	ds_store_b32 v22, v6 offset:18432
	v_and_b32_e32 v6, 0x7070707, v24
	v_lshrrev_b32_e32 v22, 1, v24
	v_and_b32_e32 v24, 0x7070707, v15
	v_lshrrev_b32_e32 v15, 1, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_perm_b32 v34, s7, 0x44403800, v6
	v_or_b32_e32 v6, 0x50505050, v6
	v_and_or_b32 v22, v22, s14, 0x3020100
	v_and_or_b32 v36, v15, s14, 0x3020100
	v_perm_b32 v35, s7, 0x44403800, v24
	v_or_b32_e32 v24, 0x50505050, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v15, v6, v34, v22
	scratch_load_b32 v6, off, off offset:264 ; 4-byte Folded Reload
	v_perm_b32 v22, v33, v45, v46
	v_perm_b32 v16, v24, v35, v36
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v23 offset:18944
	ds_store_2addr_b64 v26, v[11:12], v[13:14] offset1:1
	ds_store_2addr_b64 v27, v[15:16], v[17:18] offset1:1
	ds_store_2addr_b64 v134, v[7:8], v[9:10] offset1:1
	ds_store_2addr_b64 v28, v[19:20], v[21:22] offset1:1
.LBB0_2:
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v236, 0 :: v_dual_and_b32 v17, 64, v0
	v_dual_mov_b32 v235, 0 :: v_dual_and_b32 v10, 8, v0
	v_dual_mov_b32 v243, 0 :: v_dual_mov_b32 v0, 0
	v_dual_mov_b32 v234, 0 :: v_dual_and_b32 v7, 15, v47
	v_dual_mov_b32 v233, 0 :: v_dual_and_b32 v8, 0x60, v47
	v_dual_mov_b32 v237, 0 :: v_dual_mov_b32 v238, 0
	v_dual_mov_b32 v239, 0 :: v_dual_mov_b32 v240, 0
	v_dual_mov_b32 v241, 0 :: v_dual_mov_b32 v242, 0
	scratch_store_b32 off, v0, off offset:12 ; 4-byte Folded Spill
	v_dual_mov_b32 v244, 0 :: v_dual_mov_b32 v245, 0
	v_dual_mov_b32 v246, 0 :: v_dual_mov_b32 v187, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v247, 0
	v_dual_mov_b32 v248, 0 :: v_dual_mov_b32 v249, 0
	v_dual_mov_b32 v250, 0 :: v_dual_mov_b32 v251, 0
	v_dual_mov_b32 v252, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v253, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v0, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v231, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v179, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v181, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v183, 0 :: v_dual_mov_b32 v184, 0
	v_mov_b32_e32 v185, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_barrier_wait -1
	s_wait_storecnt 0x0
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_14
; %bb.3:
	scratch_store_b32 off, v8, off offset:368 ; 4-byte Folded Spill
	v_or_b32_e32 v8, v8, v10
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v17, off offset:360
	scratch_store_b32 off, v7, off offset:364
	v_or_b32_e32 v6, v17, v7
	v_dual_mov_b32 v104, 0 :: v_dual_add_nc_u32 v7, 0, v10
	v_and_b32_e32 v0, 0x6f, v47
	v_lshlrev_b32_e32 v8, 3, v8
	v_add_co_u32 v221, vcc_lo, v2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v222, null, 0, v3, vcc_lo
	v_add_co_u32 v223, vcc_lo, v4, v1
	v_mov_b32_e32 v185, v104
	v_mad_u32_u24 v0, 0x48, v0, 0
	v_mov_b32_e32 v181, v104
	v_dual_mov_b32 v184, v104 :: v_dual_add_nc_u32 v1, 0, v8
	v_dual_mov_b32 v179, v104 :: v_dual_add_nc_u32 v2, 0x2400, v134
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v0, v0, v10
	v_dual_mov_b32 v178, v104 :: v_dual_add_nc_u32 v3, 0x4a10, v1
	v_dual_mov_b32 v151, v104 :: v_dual_mov_b32 v176, v104
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v148, v104 :: v_dual_add_nc_u32 v227, 0x2400, v0
	v_add_nc_u32_e32 v0, 0x1000, v134
	s_clause 0x2                            ; 12-byte Folded Spill
	scratch_store_b32 off, v2, off
	scratch_store_b32 off, v47, off offset:356
	scratch_store_b32 off, v3, off offset:16
	v_add_nc_u32_e32 v3, 0x4a18, v1
	scratch_store_b32 off, v10, off offset:372 ; 4-byte Folded Spill
	v_mov_b32_e32 v154, v104
	v_mov_b32_e32 v152, v104
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v0, off offset:8
	scratch_store_b32 off, v3, off offset:20
	v_dual_mov_b32 v0, v104 :: v_dual_add_nc_u32 v3, 0x4a20, v1
	v_dual_mov_b32 v172, v104 :: v_dual_add_nc_u32 v189, 0x4a00, v1
	v_mov_b32_e32 v180, v104
	scratch_store_b32 off, v3, off offset:24 ; 4-byte Folded Spill
	v_dual_mov_b32 v170, v104 :: v_dual_add_nc_u32 v3, 0x4a28, v1
	v_dual_mov_b32 v153, v104 :: v_dual_add_nc_u32 v226, 0x4a08, v1
	v_mul_u32_u24_e32 v9, 0x48, v6
	scratch_store_b32 off, v3, off offset:28 ; 4-byte Folded Spill
	v_dual_mov_b32 v168, v104 :: v_dual_add_nc_u32 v3, 0x4a30, v1
	v_dual_mov_b32 v183, v104 :: v_dual_lshlrev_b32 v6, 2, v6
	v_dual_mov_b32 v177, v104 :: v_dual_add_nc_u32 v2, 0x3600, v134
	scratch_store_b32 off, v3, off offset:32 ; 4-byte Folded Spill
	v_dual_mov_b32 v166, v104 :: v_dual_add_nc_u32 v3, 0x4a38, v1
	v_dual_mov_b32 v182, v104 :: v_dual_add_nc_u32 v225, v7, v9
	v_mov_b32_e32 v164, v104
	scratch_store_b32 off, v3, off offset:36 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a80, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v224, null, 0, v5, vcc_lo
	v_mov_b32_e32 v162, v104
	v_dual_mov_b32 v231, v104 :: v_dual_add_nc_u32 v228, 0x800, v225
	scratch_store_b32 off, v3, off offset:40 ; 4-byte Folded Spill
	v_dual_mov_b32 v160, v104 :: v_dual_add_nc_u32 v3, 0x4a88, v1
	v_dual_mov_b32 v171, v104 :: v_dual_mov_b32 v252, v104
	v_dual_mov_b32 v169, v104 :: v_dual_mov_b32 v250, v104
	scratch_store_b32 off, v3, off offset:44 ; 4-byte Folded Spill
	v_dual_mov_b32 v150, v104 :: v_dual_add_nc_u32 v3, 0x4a90, v1
	v_dual_mov_b32 v158, v104 :: v_dual_mov_b32 v175, v104
	v_dual_mov_b32 v167, v104 :: v_dual_mov_b32 v248, v104
	scratch_store_b32 off, v3, off offset:48 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4a98, v1
	v_dual_mov_b32 v165, v104 :: v_dual_mov_b32 v136, v104
	v_dual_mov_b32 v163, v104 :: v_dual_mov_b32 v246, v104
	scratch_store_b32 off, v3, off offset:52 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4aa0, v1
	v_dual_mov_b32 v161, v104 :: v_dual_mov_b32 v244, v104
	v_dual_mov_b32 v159, v104 :: v_dual_mov_b32 v242, v104
	scratch_store_b32 off, v3, off offset:56 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4aa8, v1
	v_dual_mov_b32 v157, v104 :: v_dual_mov_b32 v240, v104
	v_dual_mov_b32 v149, v104 :: v_dual_mov_b32 v238, v104
	scratch_store_b32 off, v3, off offset:60 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v3, 0x4ab0, v1
	v_dual_mov_b32 v146, v104 :: v_dual_add_nc_u32 v1, 0x4ab8, v1
	v_dual_mov_b32 v147, v104 :: v_dual_mov_b32 v236, v104
	v_dual_mov_b32 v253, v104 :: v_dual_mov_b32 v234, v104
	scratch_store_b32 off, v1, off offset:68 ; 4-byte Folded Spill
	v_mov_b32_e32 v1, v104
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v2, off offset:4
	scratch_store_b32 off, v3, off offset:64
	v_dual_mov_b32 v145, v104 :: v_dual_add_nc_u32 v2, 0, v6
	v_mov_b32_e32 v232, v104
	scratch_store_b32 off, v1, off offset:12 ; 4-byte Folded Spill
	v_mov_b32_e32 v1, v129
	v_add_nc_u32_e32 v229, 0x4800, v2
	v_mov_b32_e32 v251, v104
	v_mov_b32_e32 v249, v104
	v_mov_b32_e32 v247, v104
	v_mov_b32_e32 v187, v104
	v_mov_b32_e32 v245, v104
	v_mov_b32_e32 v243, v104
	v_mov_b32_e32 v241, v104
	v_mov_b32_e32 v239, v104
	v_mov_b32_e32 v237, v104
	v_mov_b32_e32 v235, v104
	v_mov_b32_e32 v233, v104
	scratch_store_b64 off, v[1:2], off offset:268 ; 8-byte Folded Spill
	v_mov_b32_e32 v1, v130
	s_mov_b32 s7, 0
	s_sub_co_i32 s17, 0, s16
	s_mov_b32 s18, 1
	s_movk_i32 s14, 0x100
	s_movk_i32 s19, 0x88
	s_mov_b32 s26, 0x4e4c4a48
	s_mov_b32 s31, 0x4040404
	s_clause 0xb                            ; 56-byte Folded Spill
	scratch_store_b64 off, v[1:2], off offset:276
	scratch_store_b64 off, v[101:102], off offset:288
	scratch_store_b32 off, v221, off offset:296
	scratch_store_b32 off, v222, off offset:300
	scratch_store_b32 off, v223, off offset:304
	scratch_store_b32 off, v224, off offset:308
	scratch_store_b32 off, v189, off offset:284
	scratch_store_b32 off, v225, off offset:312
	scratch_store_b32 off, v227, off offset:316
	scratch_store_b32 off, v228, off offset:320
	scratch_store_b32 off, v229, off offset:324
	scratch_store_b32 off, v226, off offset:328
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_clause 0x3                            ; 32-byte Folded Reload
	scratch_load_b64 v[137:138], off, off offset:188 th:TH_LOAD_LU
	scratch_load_b64 v[139:140], off, off offset:196 th:TH_LOAD_LU
	scratch_load_b64 v[141:142], off, off offset:212 th:TH_LOAD_LU
	scratch_load_b64 v[191:192], off, off offset:244 th:TH_LOAD_LU
	s_add_co_i32 s18, s18, 1
	s_addk_co_i32 s14, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s17, s18
	s_addk_co_i32 s19, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, 1
	scratch_load_b64 v[143:144], off, off offset:228 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x3
	v_dual_fmac_f32 v184, v137, v57 :: v_dual_fmac_f32 v185, v139, v58
	scratch_load_b64 v[57:58], off, off offset:204 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x3
	v_fmac_f32_e32 v183, v141, v60
	v_fmac_f32_e32 v245, v137, v33
	scratch_load_b32 v33, off, off offset:12 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_dual_fmac_f32 v150, v137, v41 :: v_dual_fmac_f32 v157, v139, v42
	v_dual_fmac_f32 v172, v137, v49 :: v_dual_fmac_f32 v151, v139, v50
	s_wait_loadcnt 0x2
	v_fmac_f32_e32 v181, v143, v62
	v_fmac_f32_e32 v243, v143, v38
	v_fmac_f32_e32 v147, v143, v46
	v_fmac_f32_e32 v169, v143, v54
	v_fmac_f32_e32 v246, v139, v34
	v_dual_fmac_f32 v150, v138, v155 :: v_dual_fmac_f32 v157, v140, v155
	v_fmac_f32_e32 v172, v138, v174
	v_fmac_f32_e32 v184, v138, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v246, v140, v156
	s_wait_dscnt 0xf
	v_fmac_f32_e32 v150, v229, v105
	v_fmac_f32_e32 v172, v229, v113
	s_wait_dscnt 0xe
	v_fmac_f32_e32 v246, v227, v98
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v150, v230, v195
	v_fmac_f32_e32 v172, v230, v198
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v246, v228, v196
	s_wait_loadcnt 0x1
	v_fmac_f32_e32 v182, v57, v59
	scratch_load_b64 v[59:60], off, off offset:220 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v244, v57, v35
	scratch_load_b64 v[34:35], off, off offset:124 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v148, v57, v43
	scratch_load_b64 v[42:43], off, off offset:180 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v149, v141, v44
	v_fmac_f32_e32 v244, v58, v156
	v_fmac_f32_e32 v170, v57, v51
	v_fmac_f32_e32 v171, v141, v52
	v_fmac_f32_e32 v151, v140, v174
	v_dual_fmac_f32 v185, v140, v173 :: v_dual_fmac_f32 v182, v58, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v170, v58, v174
	v_dual_fmac_f32 v148, v58, v155 :: v_dual_fmac_f32 v157, v227, v106
	s_wait_dscnt 0xd
	v_fmac_f32_e32 v244, v225, v99
	v_fmac_f32_e32 v182, v225, v123
	v_fmac_f32_e32 v170, v225, v115
	v_fmac_f32_e32 v148, v225, v107
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v157, v228, v195 :: v_dual_fmac_f32 v244, v226, v196
	v_dual_fmac_f32 v184, v229, v121 :: v_dual_fmac_f32 v185, v227, v122
	v_fmac_f32_e32 v148, v226, v195
	v_fmac_f32_e32 v182, v226, v197
	v_fmac_f32_e32 v170, v226, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v184, v230, v197 :: v_dual_fmac_f32 v185, v228, v197
	s_wait_loadcnt 0x2
	v_fmac_f32_e32 v180, v59, v61
	scratch_load_b64 v[61:62], off, off offset:236 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v242, v59, v37
	v_dual_fmac_f32 v146, v59, v45 :: v_dual_fmac_f32 v149, v142, v155
	v_fmac_f32_e32 v168, v59, v53
	s_wait_loadcnt 0x1
	v_fmac_f32_e32 v0, v42, v32
	v_fmac_f32_e32 v242, v60, v156
	v_fmac_f32_e32 v146, v60, v155
	v_dual_fmac_f32 v147, v144, v155 :: v_dual_fmac_f32 v158, v42, v24
	v_dual_fmac_f32 v183, v142, v173 :: v_dual_fmac_f32 v180, v60, v173
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v146, v221, v109 :: v_dual_fmac_f32 v147, v219, v110
	v_dual_fmac_f32 v242, v221, v101 :: v_dual_fmac_f32 v171, v142, v174
	v_fmac_f32_e32 v168, v60, v174
	v_fmac_f32_e32 v136, v42, v16
	v_dual_fmac_f32 v183, v223, v124 :: v_dual_fmac_f32 v180, v221, v125
	v_dual_fmac_f32 v242, v222, v196 :: v_dual_fmac_f32 v151, v227, v114
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v168, v221, v117 :: v_dual_fmac_f32 v149, v223, v108
	v_fmac_f32_e32 v0, v43, v173
	v_fmac_f32_e32 v158, v43, v174
	v_fmac_f32_e32 v151, v228, v198
	v_dual_fmac_f32 v183, v224, v197 :: v_dual_fmac_f32 v180, v222, v197
	v_dual_fmac_f32 v149, v224, v195 :: v_dual_fmac_f32 v146, v222, v195
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v158, v199, v88
	v_fmac_f32_e32 v168, v222, v198
	v_fmac_f32_e32 v232, v42, v8
	v_fmac_f32_e32 v0, v199, v96
	v_fmac_f32_e32 v136, v43, v155
	v_fmac_f32_e32 v158, v200, v198
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v232, v43, v156
	v_fmac_f32_e32 v0, v200, v197
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v136, v199, v80
	v_fmac_f32_e32 v232, v199, v72
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v136, v200, v195
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v232, v200, v196 :: v_dual_fmac_f32 v253, v61, v47
	v_fmac_f32_e32 v240, v61, v39
	scratch_load_b64 v[38:39], off, off offset:148 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v33, v141, v36
	scratch_load_b64 v[36:37], off, off offset:132 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v241, v191, v40
	v_fmac_f32_e32 v177, v34, v25
	scratch_load_b64 v[40:41], off, off offset:164 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v145, v191, v48
	v_dual_fmac_f32 v253, v62, v155 :: v_dual_fmac_f32 v240, v62, v156
	v_dual_fmac_f32 v33, v142, v156 :: v_dual_fmac_f32 v166, v61, v55
	v_fmac_f32_e32 v167, v191, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v253, v217, v111
	v_fmac_f32_e32 v240, v217, v103
	v_dual_fmac_f32 v178, v61, v63 :: v_dual_fmac_f32 v179, v191, v64
	v_dual_fmac_f32 v169, v144, v174 :: v_dual_fmac_f32 v166, v62, v174
	v_fmac_f32_e32 v167, v192, v174
	v_fmac_f32_e32 v165, v34, v17
	v_fmac_f32_e32 v239, v34, v1
	v_dual_fmac_f32 v181, v144, v173 :: v_dual_fmac_f32 v178, v62, v173
	v_fmac_f32_e32 v179, v192, v173
	v_fmac_f32_e32 v169, v219, v118
	v_dual_fmac_f32 v167, v215, v120 :: v_dual_fmac_f32 v252, v34, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v178, v217, v127 :: v_dual_fmac_f32 v179, v215, v128
	v_dual_fmac_f32 v171, v223, v116 :: v_dual_fmac_f32 v166, v217, v119
	v_fmac_f32_e32 v181, v219, v126
	v_dual_fmac_f32 v252, v35, v155 :: v_dual_fmac_f32 v169, v220, v198
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v171, v224, v198
	v_dual_fmac_f32 v147, v220, v195 :: v_dual_fmac_f32 v240, v218, v196
	v_fmac_f32_e32 v252, v213, v73
	v_dual_fmac_f32 v181, v220, v197 :: v_dual_fmac_f32 v178, v218, v197
	v_fmac_f32_e32 v166, v218, v198
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v252, v214, v195
	s_wait_loadcnt 0x2
	v_fmac_f32_e32 v154, v38, v28
	s_wait_loadcnt 0x1
	v_dual_fmac_f32 v245, v138, v156 :: v_dual_fmac_f32 v176, v36, v26
	scratch_load_b64 v[25:26], off, off offset:140 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v145, v192, v155
	s_wait_loadcnt 0x1
	v_dual_fmac_f32 v231, v40, v30 :: v_dual_fmac_f32 v162, v38, v20
	v_fmac_f32_e32 v160, v40, v22
	v_fmac_f32_e32 v247, v40, v14
	v_dual_fmac_f32 v145, v215, v112 :: v_dual_fmac_f32 v234, v40, v6
	v_dual_fmac_f32 v243, v144, v156 :: v_dual_fmac_f32 v176, v37, v173
	v_dual_fmac_f32 v241, v192, v156 :: v_dual_fmac_f32 v154, v39, v173
	v_dual_fmac_f32 v164, v36, v18 :: v_dual_fmac_f32 v245, v229, v97
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v176, v211, v90
	v_dual_fmac_f32 v165, v35, v174 :: v_dual_fmac_f32 v160, v41, v174
	v_dual_fmac_f32 v247, v41, v155 :: v_dual_fmac_f32 v234, v41, v156
	v_dual_fmac_f32 v241, v215, v104 :: v_dual_fmac_f32 v164, v37, v174
	v_fmac_f32_e32 v236, v38, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v154, v207, v92 :: v_dual_fmac_f32 v165, v213, v81
	v_fmac_f32_e32 v160, v203, v86
	v_fmac_f32_e32 v234, v203, v70
	v_dual_fmac_f32 v176, v212, v197 :: v_dual_fmac_f32 v249, v38, v12
	v_dual_fmac_f32 v238, v36, v2 :: v_dual_fmac_f32 v231, v41, v173
	v_fmac_f32_e32 v162, v39, v174
	v_fmac_f32_e32 v239, v35, v156
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v249, v39, v155 :: v_dual_fmac_f32 v238, v37, v156
	v_fmac_f32_e32 v236, v39, v156
	v_fmac_f32_e32 v164, v211, v82
	v_fmac_f32_e32 v162, v207, v84
	v_fmac_f32_e32 v249, v207, v76
	v_fmac_f32_e32 v247, v203, v78
	v_dual_fmac_f32 v239, v213, v65 :: v_dual_fmac_f32 v236, v207, v68
	v_dual_fmac_f32 v238, v211, v66 :: v_dual_fmac_f32 v245, v230, v196
	v_dual_fmac_f32 v164, v212, v198 :: v_dual_fmac_f32 v253, v218, v195
	v_fmac_f32_e32 v241, v216, v196
	v_dual_fmac_f32 v165, v214, v198 :: v_dual_fmac_f32 v162, v208, v198
	v_dual_fmac_f32 v239, v214, v196 :: v_dual_fmac_f32 v236, v208, v196
	v_fmac_f32_e32 v238, v212, v196
	v_dual_fmac_f32 v154, v208, v197 :: v_dual_fmac_f32 v249, v208, v195
	v_dual_fmac_f32 v160, v204, v198 :: v_dual_fmac_f32 v247, v204, v195
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v234, v204, v196 :: v_dual_fmac_f32 v175, v25, v27
	scratch_load_b64 v[27:28], off, off offset:156 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v163, v25, v19
	v_dual_fmac_f32 v251, v36, v10 :: v_dual_fmac_f32 v250, v25, v11
	v_fmac_f32_e32 v175, v26, v173
	v_fmac_f32_e32 v237, v25, v3
	v_fmac_f32_e32 v231, v203, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v163, v26, v174 :: v_dual_fmac_f32 v250, v26, v155
	v_fmac_f32_e32 v175, v209, v91
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v237, v26, v156
	v_fmac_f32_e32 v231, v204, v197
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v163, v209, v83
	v_dual_fmac_f32 v250, v209, v75 :: v_dual_fmac_f32 v179, v216, v197
	v_fmac_f32_e32 v237, v209, v67
	v_fmac_f32_e32 v175, v210, v197
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v163, v210, v198 :: v_dual_fmac_f32 v250, v210, v195
	v_fmac_f32_e32 v237, v210, v196
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v153, v27, v29
	scratch_load_b64 v[29:30], off, off offset:172 th:TH_LOAD_LU ; 8-byte Folded Reload
	v_fmac_f32_e32 v161, v27, v21
	v_fmac_f32_e32 v235, v27, v5
	v_fmac_f32_e32 v248, v27, v13
	v_fmac_f32_e32 v153, v28, v173
	v_fmac_f32_e32 v251, v37, v155
	v_fmac_f32_e32 v161, v28, v174
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v235, v28, v156 :: v_dual_fmac_f32 v248, v28, v155
	v_fmac_f32_e32 v153, v205, v93
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v251, v211, v74
	v_fmac_f32_e32 v161, v205, v85
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v235, v205, v69
	v_dual_fmac_f32 v248, v205, v77 :: v_dual_fmac_f32 v167, v216, v198
	v_fmac_f32_e32 v251, v212, v195
	v_fmac_f32_e32 v153, v206, v197
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v161, v206, v198 :: v_dual_fmac_f32 v248, v206, v195
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v235, v206, v196 :: v_dual_fmac_f32 v152, v29, v31
	v_fmac_f32_e32 v33, v223, v100
	v_fmac_f32_e32 v159, v29, v23
	v_fmac_f32_e32 v233, v29, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v177, v35, v173 :: v_dual_fmac_f32 v152, v30, v173
	v_fmac_f32_e32 v33, v224, v196
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v159, v30, v174
	v_fmac_f32_e32 v233, v30, v156
	v_fmac_f32_e32 v187, v29, v15
	v_dual_fmac_f32 v243, v219, v102 :: v_dual_fmac_f32 v152, v201, v95
	scratch_store_b32 off, v33, off offset:12 ; 4-byte Folded Spill
	v_fmac_f32_e32 v177, v213, v89
	v_fmac_f32_e32 v159, v201, v87
	v_fmac_f32_e32 v233, v201, v71
	v_fmac_f32_e32 v187, v30, v155
	v_fmac_f32_e32 v243, v220, v196
	v_dual_fmac_f32 v145, v216, v195 :: v_dual_fmac_f32 v152, v202, v197
	v_fmac_f32_e32 v177, v214, v197
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v187, v201, v79
	v_fmac_f32_e32 v159, v202, v198
	v_fmac_f32_e32 v233, v202, v196
	v_fmac_f32_e32 v187, v202, v195
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b64 v[103:104], off, off offset:72 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x0
	s_clause 0xd                            ; 84-byte Folded Reload
	scratch_load_b32 v103, off, off offset:252 th:TH_LOAD_LU
	scratch_load_b64 v[101:102], off, off offset:288
	scratch_load_b128 v[86:89], off, off offset:80 th:TH_LOAD_LU
	scratch_load_b128 v[90:93], off, off offset:96 th:TH_LOAD_LU
	scratch_load_b32 v221, off, off offset:296
	scratch_load_b32 v222, off, off offset:300
	scratch_load_b32 v223, off, off offset:304
	scratch_load_b32 v224, off, off offset:308
	scratch_load_b32 v225, off, off offset:312
	scratch_load_b32 v226, off, off offset:328
	scratch_load_b32 v227, off, off offset:316
	scratch_load_b32 v228, off, off offset:320
	scratch_load_b32 v229, off, off offset:324
	scratch_load_b32 v85, off, off offset:112
	s_wait_loadcnt 0xa
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v91, off, off offset:116
	scratch_load_b32 v93, off, off offset:120
	s_cbranch_scc1 .LBB0_13
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s14, 0xffffff00
	v_lshlrev_b64_e32 v[1:2], 2, v[103:104]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[20:21], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v77, s6, s34, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v78, null, s35, 0, s6
	v_add_co_u32 v79, s6, s34, v130
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v80, null, s35, 0, s6
	s_add_co_i32 s6, s19, 0xffffff78
	s_wait_loadcnt 0xb
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v81, vcc_lo, v221, s6
	s_wait_loadcnt 0xa
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, 0, v222, vcc_lo
	s_wait_loadcnt 0x9
	v_add_co_u32 v83, vcc_lo, v223, s6
	s_wait_loadcnt 0x8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, 0, v224, vcc_lo
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
	global_load_b32 v186, v[3:4], off offset:4
	global_load_b32 v135, v[1:2], off offset:4
	s_wait_loadcnt 0xb
	ds_load_2addr_b64 v[65:68], v227 offset1:144
	ds_load_2addr_b64 v[1:4], v225 offset1:144
	s_wait_loadcnt 0xa
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
	ds_load_2addr_b64 v[65:68], v227 offset0:2 offset1:146
	ds_load_2addr_b64 v[69:72], v225 offset0:2 offset1:146
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
	ds_load_2addr_b64 v[65:68], v227 offset0:4 offset1:148
	ds_load_2addr_b64 v[69:72], v225 offset0:4 offset1:148
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
	ds_load_2addr_b64 v[65:68], v227 offset0:6 offset1:150
	ds_load_2addr_b64 v[69:72], v225 offset0:6 offset1:150
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
	s_wait_loadcnt 0x6
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v65, 0, v254, s2
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v74, 0, v255, s2
	v_cndmask_b32_e64 v70, 0, v85, s1
	v_cndmask_b32_e64 v69, 0, v86, s1
	v_lshrrev_b32_e32 v66, 4, v65
	v_and_b32_e32 v71, 0xf0f0f0f, v65
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v65, 0, v90, s0
	v_cndmask_b32_e64 v89, 0, v131, s3
	v_and_b32_e32 v72, 0xf0f0f0f, v66
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v66, 0, v91, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_lshrrev_b32_e32 v91, 4, v89
	v_perm_b32 v73, v72, v71, 0x5010400
	v_perm_b32 v76, v72, v71, 0x7030602
	v_lshrrev_b32_e32 v71, 4, v74
	v_and_b32_e32 v74, 0xf0f0f0f, v74
	s_barrier_signal -1
	v_and_b32_e32 v75, 0x7070707, v73
	v_lshrrev_b32_e32 v72, 1, v73
	v_and_b32_e32 v85, 0xf0f0f0f, v71
	v_and_b32_e32 v87, 0x7070707, v76
	v_cndmask_b32_e64 v71, 0, v88, s1
	v_perm_b32 v73, s26, 0x44403800, v75
	v_or_b32_e32 v75, 0x50505050, v75
	v_and_or_b32 v86, v72, s31, 0x3020100
	v_perm_b32 v88, v85, v74, 0x5010400
	v_perm_b32 v85, v85, v74, 0x7030602
	v_cndmask_b32_e64 v68, 0, v93, s0
	v_cndmask_b32_e64 v67, 0, v92, s0
	v_perm_b32 v73, v75, v73, v86
	v_lshrrev_b32_e32 v75, 1, v76
	v_perm_b32 v76, s26, 0x44403800, v87
	v_or_b32_e32 v86, 0x50505050, v87
	v_and_b32_e32 v87, 0x7070707, v88
	v_lshrrev_b32_e32 v88, 1, v88
	v_and_or_b32 v75, v75, s31, 0x3020100
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_perm_b32 v90, s26, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_and_or_b32 v88, v88, s31, 0x3020100
	v_perm_b32 v74, v86, v76, v75
	v_and_b32_e32 v76, 0xf0f0f0f, v89
	v_and_b32_e32 v86, 0xf0f0f0f, v91
	v_and_b32_e32 v89, 0x7070707, v85
	v_perm_b32 v75, v87, v90, v88
	v_cndmask_b32_e64 v87, 0, v132, s3
	v_lshrrev_b32_e32 v85, 1, v85
	v_perm_b32 v88, v86, v76, 0x5010400
	v_perm_b32 v76, v86, v76, 0x7030602
	v_perm_b32 v90, s26, 0x44403800, v89
	v_lshrrev_b32_e32 v91, 4, v87
	v_and_b32_e32 v86, 0xf0f0f0f, v87
	v_and_b32_e32 v92, 0x7070707, v88
	v_lshrrev_b32_e32 v88, 1, v88
	v_and_b32_e32 v93, 0x7070707, v76
	v_and_b32_e32 v87, 0xf0f0f0f, v91
	v_lshrrev_b32_e32 v76, 1, v76
	v_or_b32_e32 v89, 0x50505050, v89
	v_and_or_b32 v85, v85, s31, 0x3020100
	v_perm_b32 v91, s26, 0x44403800, v92
	v_perm_b32 v94, v87, v86, 0x5010400
	v_or_b32_e32 v92, 0x50505050, v92
	v_and_or_b32 v88, v88, s31, 0x3020100
	v_and_or_b32 v96, v76, s31, 0x3020100
	v_perm_b32 v86, v87, v86, 0x7030602
	v_lshrrev_b32_e32 v76, 1, v94
	v_and_b32_e32 v87, 0x7070707, v94
	v_perm_b32 v95, s26, 0x44403800, v93
	v_or_b32_e32 v93, 0x50505050, v93
	v_and_b32_e32 v94, 0x7070707, v86
	v_and_or_b32 v98, v76, s31, 0x3020100
	v_perm_b32 v76, v89, v90, v85
	v_perm_b32 v85, v92, v91, v88
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v91, off, off offset:8
	scratch_load_b32 v89, off, off
	scratch_load_b32 v90, off, off offset:4
	v_lshrrev_b32_e32 v86, 1, v86
	v_perm_b32 v97, s26, 0x44403800, v87
	v_or_b32_e32 v87, 0x50505050, v87
	v_perm_b32 v99, s26, 0x44403800, v94
	v_or_b32_e32 v94, 0x50505050, v94
	v_and_or_b32 v100, v86, s31, 0x3020100
	v_cndmask_b32_e64 v72, 0, v190, s1
	v_perm_b32 v86, v93, v95, v96
	v_perm_b32 v87, v87, v97, v98
	ds_store_2addr_b64 v134, v[65:66], v[67:68] offset1:1
	v_perm_b32 v88, v94, v99, v100
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v91, v[69:70], v[71:72] offset0:64 offset1:65
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v89, v[73:74], v[75:76] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v90, v[85:86], v[87:88] offset1:1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_clause 0x1
	global_load_b128 v[217:220], v[77:78], off offset:192
	global_load_b128 v[213:216], v[79:80], off offset:192
	global_load_b64 v[254:255], v[81:82], off offset:104
	global_load_b64 v[131:132], v[83:84], off offset:104
	ds_load_2addr_b64 v[65:68], v227 offset1:144
	ds_load_2addr_b64 v[69:72], v225 offset1:144
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
	ds_load_2addr_b64 v[65:68], v227 offset0:2 offset1:146
	ds_load_2addr_b64 v[69:72], v225 offset0:2 offset1:146
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
	ds_load_2addr_b64 v[65:68], v227 offset0:4 offset1:148
	ds_load_2addr_b64 v[69:72], v225 offset0:4 offset1:148
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
	ds_load_2addr_b64 v[65:68], v227 offset0:6 offset1:150
	ds_load_2addr_b64 v[69:72], v225 offset0:6 offset1:150
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
	ds_load_2addr_b32 v[173:174], v229 offset1:16
	ds_load_2addr_b32 v[155:156], v229 offset0:32 offset1:48
	ds_load_2addr_b32 v[65:66], v189 offset1:1
	v_mov_b32_e32 v188, v142
	v_mov_b32_e32 v142, v138
	v_mov_b32_e32 v138, v140
	s_cmp_lt_i32 s18, s16
	s_cselect_b32 s6, -1, 0
	s_cmp_ge_i32 s18, s16
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:188 ; 8-byte Folded Spill
	ds_load_2addr_b32 v[65:66], v226 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:196 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:16 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:204 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:20 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:212 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:220 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:28 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:228 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:236 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:244 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:124 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:132 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:140 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:148 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:156 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:164 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:64 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:172 ; 8-byte Folded Spill
	scratch_load_b32 v65, off, off offset:68 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[65:66], v65 offset1:1
	s_wait_dscnt 0x0
	scratch_store_b64 off, v[65:66], off offset:180 ; 8-byte Folded Spill
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
	v_cndmask_b32_e64 v66, 0, v188, s0
	v_cndmask_b32_e64 v65, 0, v141, s0
	v_cndmask_b32_e64 v68, 0, v144, s0
	v_cndmask_b32_e64 v67, 0, v143, s0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v134, v[65:66], v[67:68] offset1:1
	v_cndmask_b32_e64 v66, 0, v142, s1
	v_cndmask_b32_e64 v65, 0, v137, s1
	v_cndmask_b32_e64 v68, 0, v138, s1
	v_cndmask_b32_e64 v67, 0, v139, s1
	ds_store_2addr_b64 v91, v[65:66], v[67:68] offset0:64 offset1:65
	v_cndmask_b32_e64 v65, 0, v193, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v66, 0xf0f0f0f, v65
	v_lshrrev_b32_e32 v65, 4, v65
	v_and_b32_e32 v65, 0xf0f0f0f, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v67, v65, v66, 0x5010400
	v_perm_b32 v66, v65, v66, 0x7030602
	v_and_b32_e32 v65, 0x7070707, v67
	v_lshrrev_b32_e32 v67, 1, v67
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v68, s26, 0x44403800, v65
	v_or_b32_e32 v65, 0x50505050, v65
	v_and_or_b32 v67, v67, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v65, v65, v68, v67
	v_and_b32_e32 v67, 0x7070707, v66
	v_lshrrev_b32_e32 v66, 1, v66
	v_perm_b32 v68, s26, 0x44403800, v67
	v_or_b32_e32 v67, 0x50505050, v67
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v66, v66, s31, 0x3020100
	v_perm_b32 v66, v67, v68, v66
	v_cndmask_b32_e64 v67, 0, v194, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v68, 0xf0f0f0f, v67
	v_lshrrev_b32_e32 v67, 4, v67
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v69, v67, v68, 0x5010400
	v_perm_b32 v68, v67, v68, 0x7030602
	v_and_b32_e32 v67, 0x7070707, v69
	v_lshrrev_b32_e32 v69, 1, v69
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v70, s26, 0x44403800, v67
	v_or_b32_e32 v67, 0x50505050, v67
	v_and_or_b32 v69, v69, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v67, v67, v70, v69
	v_and_b32_e32 v69, 0x7070707, v68
	v_lshrrev_b32_e32 v68, 1, v68
	v_perm_b32 v70, s26, 0x44403800, v69
	v_or_b32_e32 v69, 0x50505050, v69
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v68, v68, s31, 0x3020100
	v_perm_b32 v68, v69, v70, v68
	ds_store_2addr_b64 v89, v[65:66], v[67:68] offset1:1
	v_cndmask_b32_e64 v65, 0, v191, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v66, 0xf0f0f0f, v65
	v_lshrrev_b32_e32 v65, 4, v65
	v_and_b32_e32 v65, 0xf0f0f0f, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v67, v65, v66, 0x5010400
	v_perm_b32 v66, v65, v66, 0x7030602
	v_and_b32_e32 v65, 0x7070707, v67
	v_lshrrev_b32_e32 v67, 1, v67
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v68, s26, 0x44403800, v65
	v_or_b32_e32 v65, 0x50505050, v65
	v_and_or_b32 v67, v67, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v65, v65, v68, v67
	v_and_b32_e32 v67, 0x7070707, v66
	v_lshrrev_b32_e32 v66, 1, v66
	v_perm_b32 v68, s26, 0x44403800, v67
	v_or_b32_e32 v67, 0x50505050, v67
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v66, v66, s31, 0x3020100
	v_perm_b32 v66, v67, v68, v66
	v_cndmask_b32_e64 v67, 0, v192, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v68, 0xf0f0f0f, v67
	v_lshrrev_b32_e32 v67, 4, v67
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v69, v67, v68, 0x5010400
	v_perm_b32 v68, v67, v68, 0x7030602
	v_and_b32_e32 v67, 0x7070707, v69
	v_lshrrev_b32_e32 v69, 1, v69
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v70, s26, 0x44403800, v67
	v_or_b32_e32 v67, 0x50505050, v67
	v_and_or_b32 v69, v69, s31, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v67, v67, v70, v69
	v_and_b32_e32 v69, 0x7070707, v68
	v_lshrrev_b32_e32 v68, 1, v68
	v_perm_b32 v70, s26, 0x44403800, v69
	v_or_b32_e32 v69, 0x50505050, v69
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v68, v68, s31, 0x3020100
	v_perm_b32 v68, v69, v70, v68
	ds_store_2addr_b64 v90, v[65:66], v[67:68] offset1:1
	scratch_load_b32 v66, off, off offset:256 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v65, 0, v135, s4
	s_wait_loadcnt 0x0
	ds_store_b32 v66, v65 offset:18432
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v65, off, off offset:260
	scratch_load_b32 v66, off, off offset:264
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v65, v65, v186
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v65, v65.l
	v_cndmask_b32_e64 v65, 0, v65, s5
	s_wait_loadcnt 0x0
	ds_store_b32 v66, v65 offset:18944
	v_add_nc_u32_e32 v65, 2, v103
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_store_b32 off, v65, off offset:252 ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s15, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[20:21], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v65, s15, s34, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v66, null, s35, 0, s15
	v_add_co_u32 v67, s15, s34, v130
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v68, null, s35, 0, s15
	s_clause 0x1
	global_load_b128 v[141:144], v[65:66], off
	global_load_b128 v[137:140], v[67:68], off
	v_add_co_u32 v65, vcc_lo, v221, s19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, 0, v222, vcc_lo
	v_add_co_u32 v67, vcc_lo, v223, s19
	v_add_nc_u32_e32 v103, 2, v103
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, 0, v224, vcc_lo
	v_mov_b32_e32 v69, v104
	v_add_co_u32 v71, vcc_lo, v101, s19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, 0, v102, vcc_lo
	s_wait_loadcnt 0x1
	v_mov_b32_e32 v188, v142
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v142, v138
	scratch_store_b64 off, v[68:69], off offset:72 ; 8-byte Folded Spill
	v_lshlrev_b64_e32 v[69:70], 2, v[103:104]
	v_mov_b32_e32 v138, v140
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v69, vcc_lo, s22, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s23, v70, vcc_lo
	global_load_b64 v[193:194], v[65:66], off offset:8
	global_load_b64 v[191:192], v[67:68], off offset:8
	global_load_b32 v186, v[71:72], off
	global_load_b32 v135, v[69:70], off
	s_branch .LBB0_8
.LBB0_7:                                ;   in Loop: Header=BB0_5 Depth=1
	scratch_store_b64 off, v[103:104], off offset:72 ; 8-byte Folded Spill
.LBB0_8:                                ;   in Loop: Header=BB0_5 Depth=1
	ds_load_2addr_b64 v[195:198], v227 offset1:144
	ds_load_2addr_b64 v[65:68], v225 offset1:144
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
	ds_load_2addr_b64 v[195:198], v227 offset0:2 offset1:146
	ds_load_2addr_b64 v[199:202], v225 offset0:2 offset1:146
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
	ds_load_2addr_b64 v[195:198], v227 offset0:4 offset1:148
	ds_load_2addr_b64 v[199:202], v225 offset0:4 offset1:148
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
	ds_load_2addr_b64 v[195:198], v227 offset0:6 offset1:150
	ds_load_2addr_b64 v[199:202], v225 offset0:6 offset1:150
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
	v_mov_b32_e32 v209, v216
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
	v_cndmask_b32_e64 v140, 0, v254, s2
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v190, 0, v255, s2
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v189, 4, v140
	v_and_b32_e32 v140, 0xf0f0f0f, v140
	v_lshrrev_b32_e32 v198, 4, v190
	v_and_b32_e32 v190, 0xf0f0f0f, v190
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_and_b32_e32 v189, 0xf0f0f0f, v189
	s_barrier_signal -1
	v_and_b32_e32 v200, 0xf0f0f0f, v198
	v_cndmask_b32_e64 v196, 0, v218, s0
	v_cndmask_b32_e64 v195, 0, v217, s0
	v_perm_b32 v197, v189, v140, 0x5010400
	v_perm_b32 v140, v189, v140, 0x7030602
	v_perm_b32 v202, v200, v190, 0x5010400
	v_perm_b32 v190, v200, v190, 0x7030602
	v_cndmask_b32_e64 v198, 0, v220, s0
	v_and_b32_e32 v189, 0x7070707, v197
	v_lshrrev_b32_e32 v197, 1, v197
	v_and_b32_e32 v201, 0x7070707, v140
	v_lshrrev_b32_e32 v140, 1, v140
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	v_perm_b32 v199, s26, 0x44403800, v189
	v_or_b32_e32 v189, 0x50505050, v189
	v_and_or_b32 v197, v197, s31, 0x3020100
	v_perm_b32 v203, s26, 0x44403800, v201
	v_and_or_b32 v140, v140, s31, 0x3020100
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_perm_b32 v199, v189, v199, v197
	v_and_b32_e32 v197, 0x7070707, v202
	v_lshrrev_b32_e32 v202, 1, v202
	v_or_b32_e32 v189, 0x50505050, v201
	v_cndmask_b32_e64 v201, 0, v131, s3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_perm_b32 v204, s26, 0x44403800, v197
	v_or_b32_e32 v197, 0x50505050, v197
	v_and_or_b32 v202, v202, s31, 0x3020100
	v_lshrrev_b32_e32 v205, 4, v201
	v_perm_b32 v200, v189, v203, v140
	v_and_b32_e32 v140, 0xf0f0f0f, v201
	v_and_b32_e32 v203, 0x7070707, v190
	v_perm_b32 v201, v197, v204, v202
	v_cndmask_b32_e64 v197, 0, v132, s3
	v_and_b32_e32 v189, 0xf0f0f0f, v205
	v_lshrrev_b32_e32 v190, 1, v190
	v_perm_b32 v204, s26, 0x44403800, v203
	v_or_b32_e32 v203, 0x50505050, v203
	v_lshrrev_b32_e32 v205, 4, v197
	v_perm_b32 v202, v189, v140, 0x5010400
	v_and_or_b32 v190, v190, s31, 0x3020100
	v_perm_b32 v140, v189, v140, 0x7030602
	v_and_b32_e32 v189, 0xf0f0f0f, v197
	v_and_b32_e32 v197, 0xf0f0f0f, v205
	v_and_b32_e32 v206, 0x7070707, v202
	v_lshrrev_b32_e32 v205, 1, v202
	v_perm_b32 v202, v203, v204, v190
	v_and_b32_e32 v190, 0x7070707, v140
	v_perm_b32 v204, v197, v189, 0x5010400
	v_lshrrev_b32_e32 v140, 1, v140
	v_perm_b32 v207, s26, 0x44403800, v206
	v_or_b32_e32 v203, 0x50505050, v206
	v_and_or_b32 v205, v205, s31, 0x3020100
	v_perm_b32 v206, s26, 0x44403800, v190
	v_perm_b32 v189, v197, v189, 0x7030602
	v_or_b32_e32 v190, 0x50505050, v190
	v_and_or_b32 v140, v140, s31, 0x3020100
	v_and_b32_e32 v197, 0x7070707, v204
	v_lshrrev_b32_e32 v208, 1, v204
	v_perm_b32 v203, v203, v207, v205
	v_and_b32_e32 v205, 0x7070707, v189
	v_perm_b32 v204, v190, v206, v140
	v_perm_b32 v140, s26, 0x44403800, v197
	v_or_b32_e32 v190, 0x50505050, v197
	v_and_or_b32 v206, v208, s31, 0x3020100
	v_perm_b32 v211, s26, 0x44403800, v205
	v_or_b32_e32 v212, 0x50505050, v205
	v_cndmask_b32_e64 v208, 0, v214, s1
	v_cndmask_b32_e64 v207, 0, v213, s1
	v_perm_b32 v205, v190, v140, v206
	scratch_load_b32 v140, off, off offset:8 ; 4-byte Folded Reload
	v_mov_b32_e32 v190, v209
	v_cndmask_b32_e64 v209, 0, v215, s1
	v_lshrrev_b32_e32 v189, 1, v189
	v_cndmask_b32_e64 v197, 0, v219, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v210, 0, v190, s1
	v_and_or_b32 v189, v189, s31, 0x3020100
	ds_store_2addr_b64 v134, v[195:196], v[197:198] offset1:1
	v_perm_b32 v206, v212, v211, v189
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v140, v[207:208], v[209:210] offset0:64 offset1:65
	scratch_load_b32 v140, off, off         ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v140, v[199:200], v[201:202] offset1:1
	scratch_load_b32 v140, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v140, v[203:204], v[205:206] offset1:1
	v_cndmask_b32_e64 v140, 0, 1, s6
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_5 Depth=1
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b64 v[129:130], off, off offset:268
	scratch_load_b64 v[131:132], off, off offset:276
	s_mov_b32 s15, s7
	v_add_co_u32 v195, vcc_lo, v221, s19
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[20:21], s[14:15]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v196, null, 0, v222, vcc_lo
	v_add_co_u32 v197, vcc_lo, v223, s19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v198, null, 0, v224, vcc_lo
	s_wait_loadcnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v129, s6, s34, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, s35, 0, s6
	s_wait_loadcnt 0x0
	v_add_co_u32 v131, s6, s34, v131
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v132, null, s35, 0, s6
	global_load_b128 v[203:206], v[129:130], off offset:64
	scratch_load_b64 v[129:130], off, off offset:268 ; 8-byte Folded Reload
	global_load_b128 v[199:202], v[131:132], off offset:64
	s_wait_loadcnt 0x1
	scratch_load_b64 v[130:131], off, off offset:276 ; 8-byte Folded Reload
	global_load_b64 v[254:255], v[195:196], off offset:40
	s_wait_loadcnt 0x1
	global_load_b64 v[131:132], v[197:198], off offset:40
	v_mov_b32_e32 v189, v204
	v_mov_b32_e32 v195, v203
	v_dual_mov_b32 v197, v205 :: v_dual_mov_b32 v190, v202
	scratch_store_b32 off, v189, off offset:116 ; 4-byte Folded Spill
	v_mov_b32_e32 v189, v206
	scratch_store_b128 off, v[195:198], off offset:96 ; 16-byte Folded Spill
	v_mov_b32_e32 v195, v199
	v_mov_b32_e32 v197, v201
	scratch_store_b32 off, v189, off offset:120 ; 4-byte Folded Spill
	v_mov_b32_e32 v189, v200
	s_clause 0x1                            ; 20-byte Folded Spill
	scratch_store_b128 off, v[195:198], off offset:80
	scratch_store_b32 off, v189, off offset:112
	s_branch .LBB0_11
.LBB0_10:                               ;   in Loop: Header=BB0_5 Depth=1
	s_clause 0x4                            ; 44-byte Folded Spill
	scratch_store_b32 off, v220, off offset:120
	scratch_store_b32 off, v218, off offset:116
	scratch_store_b32 off, v214, off offset:112
	scratch_store_b128 off, v[217:220], off offset:96
	scratch_store_b128 off, v[213:216], off offset:80
.LBB0_11:                               ;   in Loop: Header=BB0_5 Depth=1
	ds_load_2addr_b64 v[195:198], v227 offset1:144
	ds_load_2addr_b64 v[199:202], v225 offset1:144
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[195:196], v[199:200], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[197:198], v[199:200], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[195:196], v[201:202], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[197:198], v[201:202], v[81:88]
	ds_load_2addr_b64 v[199:202], v228 offset0:32 offset1:176
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[195:196], v[199:200], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[197:198], v[199:200], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[195:196], v[201:202], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[197:198], v[201:202], v[65:72]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[195:198], v227 offset0:2 offset1:146
	ds_load_2addr_b64 v[199:202], v225 offset0:2 offset1:146
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
	ds_load_2addr_b64 v[195:198], v227 offset0:4 offset1:148
	ds_load_2addr_b64 v[199:202], v225 offset0:4 offset1:148
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
	ds_load_2addr_b64 v[195:198], v227 offset0:6 offset1:150
	ds_load_2addr_b64 v[199:202], v225 offset0:6 offset1:150
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
	v_cmp_ne_u32_e32 vcc_lo, 1, v140
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v189, off, off offset:284
	scratch_load_b32 v140, off, off offset:16
	ds_load_2addr_b32 v[197:198], v229 offset1:16
	ds_load_2addr_b32 v[195:196], v229 offset0:32 offset1:48
	s_wait_loadcnt 0x1
	ds_load_2addr_b32 v[229:230], v189 offset1:1
	ds_load_2addr_b32 v[227:228], v226 offset1:1
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[225:226], v140 offset1:1
	scratch_load_b32 v140, off, off offset:20 ; 4-byte Folded Reload
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[223:224], v140 offset1:1
	scratch_load_b32 v140, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[221:222], v140 offset1:1
	scratch_load_b32 v140, off, off offset:28 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[219:220], v140 offset1:1
	scratch_load_b32 v140, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[217:218], v140 offset1:1
	scratch_load_b32 v140, off, off offset:36 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[215:216], v140 offset1:1
	scratch_load_b32 v140, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[213:214], v140 offset1:1
	scratch_load_b32 v140, off, off offset:44 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[211:212], v140 offset1:1
	scratch_load_b32 v140, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[209:210], v140 offset1:1
	scratch_load_b32 v140, off, off offset:52 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[207:208], v140 offset1:1
	scratch_load_b32 v140, off, off offset:56 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[205:206], v140 offset1:1
	scratch_load_b32 v140, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[203:204], v140 offset1:1
	scratch_load_b32 v140, off, off offset:64 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[201:202], v140 offset1:1
	scratch_load_b32 v140, off, off offset:68 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_load_2addr_b32 v[199:200], v140 offset1:1
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
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_4
; %bb.12:                               ;   in Loop: Header=BB0_5 Depth=1
	v_cndmask_b32_e64 v140, 0, v193, s2
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_clause 0x1                            ; 16-byte Folded Spill
	scratch_store_b64 off, v[213:214], off offset:348
	scratch_store_b64 off, v[131:132], off offset:332
	v_dual_mov_b32 v214, v190 :: v_dual_and_b32 v193, 0xf0f0f0f, v140
	v_lshrrev_b32_e32 v190, 4, v140
	v_cndmask_b32_e64 v189, 0, v188, s0
	v_cndmask_b32_e64 v188, 0, v141, s0
	v_cndmask_b32_e64 v141, 0, v144, s0
	v_cndmask_b32_e64 v144, 0, v194, s2
	v_dual_mov_b32 v133, v135 :: v_dual_and_b32 v190, 0xf0f0f0f, v190
	v_cndmask_b32_e64 v140, 0, v143, s0
	v_cndmask_b32_e64 v143, 0, v142, s1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v194, 4, v144
	v_perm_b32 v132, v190, v193, 0x5010400
	v_cndmask_b32_e64 v142, 0, v137, s1
	v_cndmask_b32_e64 v137, 0, v139, s1
	v_and_b32_e32 v144, 0xf0f0f0f, v144
	v_and_b32_e32 v194, 0xf0f0f0f, v194
	v_perm_b32 v139, v190, v193, 0x7030602
	v_and_b32_e32 v190, 0x7070707, v132
	v_lshrrev_b32_e32 v132, 1, v132
	s_barrier_wait -1
	s_wait_storecnt 0x0
	global_inv scope:SCOPE_SE
	scratch_store_b64 off, v[211:212], off offset:340 ; 8-byte Folded Spill
	v_dual_mov_b32 v213, v210 :: v_dual_mov_b32 v212, v209
	v_mov_b32_e32 v211, v208
	ds_store_2addr_b64 v134, v[188:189], v[140:141] offset1:1
	v_and_or_b32 v132, v132, s31, 0x3020100
	v_perm_b32 v189, v194, v144, 0x5010400
	v_dual_mov_b32 v210, v207 :: v_dual_mov_b32 v209, v206
	v_perm_b32 v140, s26, 0x44403800, v190
	v_or_b32_e32 v141, 0x50505050, v190
	v_and_b32_e32 v188, 0x7070707, v139
	v_lshrrev_b32_e32 v190, 1, v139
	v_dual_mov_b32 v208, v205 :: v_dual_mov_b32 v207, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v139, v141, v140, v132
	v_and_b32_e32 v141, 0x7070707, v189
	v_lshrrev_b32_e32 v189, 1, v189
	v_perm_b32 v132, s26, 0x44403800, v188
	v_or_b32_e32 v140, 0x50505050, v188
	v_and_or_b32 v188, v190, s31, 0x3020100
	v_cndmask_b32_e64 v190, 0, v191, s3
	v_perm_b32 v191, s26, 0x44403800, v141
	v_or_b32_e32 v141, 0x50505050, v141
	v_and_or_b32 v189, v189, s31, 0x3020100
	v_perm_b32 v144, v194, v144, 0x7030602
	v_lshrrev_b32_e32 v193, 4, v190
	v_perm_b32 v140, v140, v132, v188
	v_and_b32_e32 v190, 0xf0f0f0f, v190
	v_perm_b32 v188, v141, v191, v189
	v_cndmask_b32_e64 v141, 0, v192, s3
	v_and_b32_e32 v132, 0x7070707, v144
	v_dual_mov_b32 v206, v203 :: v_dual_and_b32 v193, 0xf0f0f0f, v193
	v_lshrrev_b32_e32 v144, 1, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_lshrrev_b32_e32 v192, 4, v141
	v_mov_b32_e32 v205, v202
	v_perm_b32 v189, s26, 0x44403800, v132
	v_or_b32_e32 v132, 0x50505050, v132
	v_and_or_b32 v144, v144, s31, 0x3020100
	v_dual_mov_b32 v204, v201 :: v_dual_and_b32 v141, 0xf0f0f0f, v141
	v_dual_mov_b32 v203, v200 :: v_dual_and_b32 v192, 0xf0f0f0f, v192
	v_perm_b32 v191, v193, v190, 0x5010400
	v_perm_b32 v190, v193, v190, 0x7030602
	v_perm_b32 v189, v132, v189, v144
	v_dual_mov_b32 v202, v199 :: v_dual_mov_b32 v199, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v193, 0x7070707, v191
	v_lshrrev_b32_e32 v132, 1, v191
	v_and_b32_e32 v191, 0x7070707, v190
	v_perm_b32 v194, v192, v141, 0x5010400
	v_lshrrev_b32_e32 v190, 1, v190
	v_dual_mov_b32 v185, v184 :: v_dual_mov_b32 v184, v183
	v_dual_mov_b32 v183, v182 :: v_dual_mov_b32 v182, v181
	v_dual_mov_b32 v181, v180 :: v_dual_mov_b32 v180, v179
	v_dual_mov_b32 v179, v178 :: v_dual_mov_b32 v178, v177
	v_dual_mov_b32 v177, v176 :: v_dual_mov_b32 v176, v175
	v_dual_mov_b32 v175, v154 :: v_dual_mov_b32 v154, v153
	v_mov_b32_e32 v153, v231
	v_dual_mov_b32 v231, v152 :: v_dual_mov_b32 v152, v0
	v_mov_b32_e32 v0, v172
	v_dual_mov_b32 v172, v151 :: v_dual_mov_b32 v151, v171
	v_dual_mov_b32 v171, v170 :: v_dual_mov_b32 v170, v169
	v_dual_mov_b32 v169, v168 :: v_dual_mov_b32 v168, v167
	v_dual_mov_b32 v167, v166 :: v_dual_mov_b32 v166, v165
	v_dual_mov_b32 v165, v164 :: v_dual_mov_b32 v164, v163
	v_dual_mov_b32 v163, v162 :: v_dual_mov_b32 v162, v161
	v_dual_mov_b32 v161, v160 :: v_dual_mov_b32 v160, v159
	v_dual_mov_b32 v159, v158 :: v_dual_mov_b32 v158, v157
	v_dual_mov_b32 v157, v150 :: v_dual_mov_b32 v150, v149
	v_dual_mov_b32 v149, v148 :: v_dual_mov_b32 v148, v147
	v_dual_mov_b32 v147, v146 :: v_dual_mov_b32 v146, v253
	v_dual_mov_b32 v253, v250 :: v_dual_mov_b32 v250, v247
	v_dual_mov_b32 v247, v134 :: v_dual_and_b32 v134, 0x7070707, v194
	v_perm_b32 v141, v192, v141, 0x7030602
	v_and_or_b32 v192, v190, s31, 0x3020100
	v_lshrrev_b32_e32 v190, 1, v194
	v_mov_b32_e32 v200, v254
	v_perm_b32 v130, s26, 0x44403800, v191
	v_or_b32_e32 v191, 0x50505050, v191
	v_mov_b32_e32 v201, v255
	v_perm_b32 v194, s26, 0x44403800, v134
	v_dual_mov_b32 v254, v251 :: v_dual_mov_b32 v129, v244
	v_dual_mov_b32 v251, v248 :: v_dual_mov_b32 v248, v136
	v_and_b32_e32 v136, 0x7070707, v141
	v_or_b32_e32 v134, 0x50505050, v134
	v_dual_mov_b32 v244, v241 :: v_dual_mov_b32 v131, v236
	v_dual_mov_b32 v241, v238 :: v_dual_mov_b32 v238, v235
	v_mov_b32_e32 v255, v145
	v_mov_b32_e32 v235, v232
	v_and_or_b32 v232, v190, s31, 0x3020100
	v_perm_b32 v191, v191, v130, v192
	v_cndmask_b32_e64 v138, 0, v138, s1
	scratch_load_b32 v135, off, off offset:260 ; 4-byte Folded Reload
	v_perm_b32 v144, s26, 0x44403800, v193
	v_perm_b32 v192, v134, v194, v232
	v_dual_mov_b32 v232, v235 :: v_dual_mov_b32 v235, v238
	v_dual_mov_b32 v238, v241 :: v_dual_mov_b32 v241, v244
	v_mov_b32_e32 v244, v129
	scratch_load_b32 v129, off, off offset:8 ; 4-byte Folded Reload
	v_or_b32_e32 v193, 0x50505050, v193
	v_and_or_b32 v132, v132, s31, 0x3020100
	v_dual_mov_b32 v236, v233 :: v_dual_mov_b32 v145, v252
	v_lshrrev_b32_e32 v141, 1, v141
	v_mov_b32_e32 v134, v247
	s_delay_alu instid0(VALU_DEP_4)
	v_perm_b32 v190, v193, v144, v132
	v_cndmask_b32_e64 v130, 0, v133, s4
	v_mov_b32_e32 v252, v145
	v_and_or_b32 v141, v141, s31, 0x3020100
	v_dual_mov_b32 v247, v250 :: v_dual_mov_b32 v250, v253
	v_dual_mov_b32 v253, v146 :: v_dual_mov_b32 v146, v147
	v_dual_mov_b32 v147, v148 :: v_dual_mov_b32 v148, v149
	v_dual_mov_b32 v149, v150 :: v_dual_mov_b32 v150, v157
	v_dual_mov_b32 v157, v158 :: v_dual_mov_b32 v158, v159
	v_dual_mov_b32 v159, v160 :: v_dual_mov_b32 v160, v161
	v_dual_mov_b32 v161, v162 :: v_dual_mov_b32 v162, v163
	v_dual_mov_b32 v163, v164 :: v_dual_mov_b32 v164, v165
	v_dual_mov_b32 v165, v166 :: v_dual_mov_b32 v166, v167
	v_dual_mov_b32 v167, v168 :: v_dual_mov_b32 v168, v169
	v_dual_mov_b32 v169, v170 :: v_dual_mov_b32 v170, v171
	v_mov_b32_e32 v171, v151
	v_dual_mov_b32 v151, v172 :: v_dual_mov_b32 v172, v0
	v_mov_b32_e32 v0, v152
	v_dual_mov_b32 v152, v231 :: v_dual_mov_b32 v231, v153
	v_dual_mov_b32 v153, v154 :: v_dual_mov_b32 v154, v175
	v_dual_mov_b32 v175, v176 :: v_dual_mov_b32 v176, v177
	v_dual_mov_b32 v177, v178 :: v_dual_mov_b32 v178, v179
	v_dual_mov_b32 v179, v180 :: v_dual_mov_b32 v180, v181
	v_dual_mov_b32 v181, v182 :: v_dual_mov_b32 v182, v183
	v_dual_mov_b32 v183, v184 :: v_dual_mov_b32 v184, v185
	v_mov_b32_e32 v145, v255
	v_mov_b32_e32 v185, v199
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v129, v[142:143], v[137:138] offset0:64 offset1:65
	scratch_load_b32 v129, off, off         ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v233, v135, v186
	v_mov_b32_e32 v186, v246
	v_dual_mov_b32 v246, v243 :: v_dual_mov_b32 v243, v240
	v_mov_b32_e32 v240, v237
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_f16_e64 v132, v233.l
	v_dual_mov_b32 v233, v236 :: v_dual_mov_b32 v236, v131
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v131, off, off offset:256
	scratch_load_b32 v135, off, off offset:264
	v_mov_b32_e32 v237, v234
	v_perm_b32 v234, s26, 0x44403800, v136
	v_or_b32_e32 v136, 0x50505050, v136
	v_cndmask_b32_e64 v132, 0, v132, s5
	s_delay_alu instid0(VALU_DEP_2)
	v_perm_b32 v193, v136, v234, v141
	v_mov_b32_e32 v136, v248
	v_dual_mov_b32 v248, v251 :: v_dual_mov_b32 v251, v254
	v_dual_mov_b32 v234, v237 :: v_dual_mov_b32 v255, v201
	v_dual_mov_b32 v237, v240 :: v_dual_mov_b32 v240, v243
	v_dual_mov_b32 v254, v200 :: v_dual_mov_b32 v199, v202
	v_dual_mov_b32 v243, v246 :: v_dual_mov_b32 v246, v186
	v_dual_mov_b32 v200, v203 :: v_dual_mov_b32 v201, v204
	v_dual_mov_b32 v202, v205 :: v_dual_mov_b32 v203, v206
	v_dual_mov_b32 v204, v207 :: v_dual_mov_b32 v205, v208
	v_dual_mov_b32 v206, v209 :: v_dual_mov_b32 v207, v210
	v_dual_mov_b32 v208, v211 :: v_dual_mov_b32 v209, v212
	v_mov_b32_e32 v210, v213
	scratch_load_b64 v[211:212], off, off offset:340 ; 8-byte Folded Reload
	s_wait_loadcnt 0x3
	ds_store_2addr_b64 v129, v[139:140], v[188:189] offset1:1
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v189, off, off offset:284
	scratch_load_b32 v129, off, off offset:4
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v129, v[190:191], v[192:193] offset1:1
	ds_store_b32 v131, v130 offset:18432
	scratch_load_b64 v[129:130], off, off offset:268 ; 8-byte Folded Reload
	v_mov_b32_e32 v190, v214
	scratch_load_b64 v[213:214], off, off offset:348 ; 8-byte Folded Reload
	s_wait_loadcnt 0x1
	scratch_load_b64 v[130:131], off, off offset:276 ; 8-byte Folded Reload
	ds_store_b32 v135, v132 offset:18944
	s_wait_loadcnt 0x0
	scratch_load_b64 v[131:132], off, off offset:332 ; 8-byte Folded Reload
	s_branch .LBB0_4
.LBB0_13:
	s_clause 0x4                            ; 20-byte Folded Reload
	scratch_load_b32 v47, off, off offset:356
	scratch_load_b32 v17, off, off offset:360
	scratch_load_b32 v7, off, off offset:364
	scratch_load_b32 v8, off, off offset:368
	scratch_load_b32 v10, off, off offset:372
.LBB0_14:
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v1, 6, v47
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v2, v10, v7
	v_and_b32_e32 v11, 31, v47
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 0x3800, v1
	v_or3_b32 v3, s30, v8, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v1, s11 :: v_dual_add_nc_u32 v4, 0, v1
	v_add_nc_u32_e32 v5, 17, v2
	v_add_nc_u32_e32 v6, 18, v2
	v_add_nc_u32_e32 v12, 19, v2
	v_lshl_add_u32 v16, v7, 7, v4
	v_add_nc_u32_e32 v13, 22, v2
	v_and_b32_e32 v5, 31, v5
	v_and_b32_e32 v6, 31, v6
	v_xor_b32_e32 v8, 16, v2
	v_lshl_add_u32 v7, v2, 2, v16
	v_and_b32_e32 v12, 31, v12
	v_lshl_add_u32 v9, v5, 2, v16
	v_add_nc_u32_e32 v5, 20, v2
	v_lshl_add_u32 v10, v6, 2, v16
	v_add_nc_u32_e32 v6, 21, v2
	v_add_nc_u32_e32 v2, 23, v2
	v_and_b32_e32 v15, 31, v13
	v_and_b32_e32 v5, 31, v5
	v_lshl_add_u32 v12, v12, 2, v16
	v_and_b32_e32 v6, 31, v6
	v_and_b32_e32 v2, 31, v2
	v_lshl_add_u32 v15, v15, 2, v16
	v_lshl_add_u32 v13, v5, 2, v16
	v_cmp_gt_i32_e64 s0, s24, v3
	v_lshl_add_u32 v14, v6, 2, v16
	v_lshl_add_u32 v8, v8, 2, v16
	v_lshl_add_u32 v16, v2, 2, v16
	ds_store_2addr_b32 v7, v184, v185 offset1:1
	ds_store_2addr_b32 v7, v182, v183 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v180, v181 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v178, v179 offset0:6 offset1:7
	ds_store_b32 v8, v177
	ds_store_b32 v9, v176
	ds_store_b32 v10, v175
	ds_store_b32 v12, v154
	ds_store_b32 v13, v153
	ds_store_b32 v14, v231
	ds_store_b32 v15, v152
	ds_store_b32 v16, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v2, s28, v17
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v0, s13, v1, s0
	v_mov_b32_e32 v1, s10
	v_cndmask_b32_e64 v6, s24, 0, s0
	v_mov_b32_e32 v17, s24
	v_cmp_gt_i32_e64 s1, s29, v3
	v_cmp_le_i32_e32 vcc_lo, s29, v3
	v_cndmask_b32_e64 v5, s12, v1, s0
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
	s_cbranch_execz .LBB0_16
; %bb.15:
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
	v_add_co_ci_u32_e64 v18, null, v0, v18, s0
	global_store_b32 v[17:18], v19, off
.LBB0_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v18, 1, v2
	v_add_nc_u32_e32 v17, 1, v47
	s_xor_b32 s1, vcc_lo, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s0, s27, v18
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_18
; %bb.17:
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
	v_add_co_ci_u32_e64 v19, null, v0, v19, vcc_lo
	global_store_b32 v[18:19], v20, off
.LBB0_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v19, 2, v2
	v_add_nc_u32_e32 v18, 2, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v19
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_20
; %bb.19:
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
	v_add_co_ci_u32_e64 v20, null, v0, v20, vcc_lo
	global_store_b32 v[19:20], v21, off
.LBB0_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v20, 3, v2
	v_add_nc_u32_e32 v19, 3, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v20
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_22
; %bb.21:
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
	v_add_co_ci_u32_e64 v21, null, v0, v21, vcc_lo
	global_store_b32 v[20:21], v22, off
.LBB0_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v21, 4, v2
	v_add_nc_u32_e32 v20, 4, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v21
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_24
; %bb.23:
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
	v_add_co_ci_u32_e64 v22, null, v0, v22, vcc_lo
	global_store_b32 v[21:22], v23, off
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v22, 5, v2
	v_add_nc_u32_e32 v21, 5, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v22
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_26
; %bb.25:
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
	v_add_co_ci_u32_e64 v23, null, v0, v23, vcc_lo
	global_store_b32 v[22:23], v24, off
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v23, 6, v2
	v_add_nc_u32_e32 v22, 6, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v23
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_28
; %bb.27:
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
	v_add_co_ci_u32_e64 v24, null, v0, v24, vcc_lo
	global_store_b32 v[23:24], v25, off
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v24, 7, v2
	v_add_nc_u32_e32 v23, 7, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v24
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_30
; %bb.29:
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
	v_add_co_ci_u32_e64 v25, null, v0, v25, vcc_lo
	global_store_b32 v[24:25], v26, off
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v25, 8, v2
	v_add_nc_u32_e32 v24, 8, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v25
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_32
; %bb.31:
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
	v_add_co_ci_u32_e64 v26, null, v0, v26, vcc_lo
	global_store_b32 v[25:26], v27, off
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 9, v2
	v_add_nc_u32_e32 v25, 9, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v26
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_34
; %bb.33:
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
	v_add_co_ci_u32_e64 v27, null, v0, v27, vcc_lo
	global_store_b32 v[26:27], v28, off
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v27, 10, v2
	v_add_nc_u32_e32 v26, 10, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v27
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_36
; %bb.35:
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
	v_add_co_ci_u32_e64 v28, null, v0, v28, vcc_lo
	global_store_b32 v[27:28], v29, off
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 11, v2
	v_add_nc_u32_e32 v27, 11, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_38
; %bb.37:
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
	v_add_co_ci_u32_e64 v29, null, v0, v29, vcc_lo
	global_store_b32 v[28:29], v30, off
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v29, 12, v2
	v_add_nc_u32_e32 v28, 12, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v29
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_40
; %bb.39:
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
	v_add_co_ci_u32_e64 v30, null, v0, v30, vcc_lo
	global_store_b32 v[29:30], v31, off
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v30, 13, v2
	v_add_nc_u32_e32 v29, 13, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v30
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_42
; %bb.41:
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
	v_add_co_ci_u32_e64 v31, null, v0, v31, vcc_lo
	global_store_b32 v[30:31], v32, off
.LBB0_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v31, 14, v2
	v_add_nc_u32_e32 v30, 14, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v31
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_44
; %bb.43:
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
	v_add_co_ci_u32_e64 v32, null, v0, v32, vcc_lo
	global_store_b32 v[31:32], v33, off
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 15, v2
	v_add_nc_u32_e32 v31, 15, v47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_46
; %bb.45:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_46:
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
	s_cbranch_execz .LBB0_48
; %bb.47:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 17, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_50
; %bb.49:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 18, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_52
; %bb.51:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 19, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_54
; %bb.53:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 20, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_56
; %bb.55:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 21, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_58
; %bb.57:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 22, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_60
; %bb.59:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 23, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_62
; %bb.61:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 24, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_64
; %bb.63:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 25, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_66
; %bb.65:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 26, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_68
; %bb.67:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 27, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_70
; %bb.69:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 28, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_72
; %bb.71:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 29, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_74
; %bb.73:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 30, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_76
; %bb.75:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 31, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_78
; %bb.77:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_78:
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
	ds_store_2addr_b32 v7, v253, v145 offset0:6 offset1:7
	ds_store_b32 v8, v252
	ds_store_b32 v9, v251
	ds_store_b32 v10, v250
	ds_store_b32 v12, v249
	ds_store_b32 v13, v248
	ds_store_b32 v14, v247
	ds_store_b32 v15, v187
	ds_store_b32 v16, v136
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_80
; %bb.79:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 33, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_82
; %bb.81:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 34, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_84
; %bb.83:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 35, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_86
; %bb.85:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 36, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_88
; %bb.87:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 37, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_90
; %bb.89:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 38, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_92
; %bb.91:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 39, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_94
; %bb.93:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 40, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_96
; %bb.95:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 41, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_98
; %bb.97:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 42, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_100
; %bb.99:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 43, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_102
; %bb.101:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 44, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_104
; %bb.103:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 45, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_106
; %bb.105:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 46, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_108
; %bb.107:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v34, v34, v35
	global_store_b32 v[32:33], v34, off
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 47, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v32
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_110
; %bb.109:
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
	v_add_co_ci_u32_e64 v33, null, v0, v33, vcc_lo
	global_store_b32 v[32:33], v34, off
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v32, off, off offset:12 th:TH_LOAD_LU ; 4-byte Folded Reload
	ds_store_2addr_b32 v7, v245, v246 offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v7, v244, v32 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v242, v243 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v240, v241 offset0:6 offset1:7
	ds_store_b32 v8, v239
	ds_store_b32 v9, v238
	ds_store_b32 v10, v237
	ds_store_b32 v12, v236
	ds_store_b32 v13, v235
	ds_store_b32 v14, v234
	ds_store_b32 v15, v233
	ds_store_b32 v16, v232
	v_or_b32_e32 v7, 48, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_112
; %bb.111:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 49, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_114
; %bb.113:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 50, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_116
; %bb.115:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 51, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_118
; %bb.117:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 52, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_120
; %bb.119:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 53, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_122
; %bb.121:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 54, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_124
; %bb.123:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 55, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_126
; %bb.125:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 56, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_128
; %bb.127:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 57, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_130
; %bb.129:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 58, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_132
; %bb.131:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 59, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_134
; %bb.133:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 60, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_136
; %bb.135:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 61, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_138
; %bb.137:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v9, v9, v10
	global_store_b32 v[7:8], v9, off
.LBB0_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 62, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_140
; %bb.139:
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
	v_add_co_ci_u32_e64 v8, null, v0, v8, vcc_lo
	global_store_b32 v[7:8], v9, off
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 63, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s27, v7
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_142
; %bb.141:
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
	v_add_co_ci_u32_e64 v2, null, v0, v2, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v3, v3, v4
	global_store_b32 v[1:2], v3, off
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end0:
	.size	gu_r2f, .Lfunc_end0-gu_r2f
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gu_r2f
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 380
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gu_r2f)<<4)&4080)>>4
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
	.set .Lgu_r2f.num_vgpr, 256
	.set .Lgu_r2f.num_agpr, 0
	.set .Lgu_r2f.numbered_sgpr, 36
	.set .Lgu_r2f.num_named_barrier, 0
	.set .Lgu_r2f.private_seg_size, 380
	.set .Lgu_r2f.uses_vcc, 1
	.set .Lgu_r2f.uses_flat_scratch, 1
	.set .Lgu_r2f.has_dyn_sized_stack, 0
	.set .Lgu_r2f.has_recursion, 0
	.set .Lgu_r2f.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 23804
; TotalNumSgprs: 38
; NumVgprs: 256
; ScratchSize: 380
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
	.type	__hip_cuid_f43f54cf8fde3738,@object ; @__hip_cuid_f43f54cf8fde3738
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_f43f54cf8fde3738
__hip_cuid_f43f54cf8fde3738:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f43f54cf8fde3738, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f43f54cf8fde3738
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
    .name:           gu_r2f
    .private_segment_fixed_size: 380
    .sgpr_count:     38
    .sgpr_spill_count: 0
    .symbol:         gu_r2f.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 109
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
