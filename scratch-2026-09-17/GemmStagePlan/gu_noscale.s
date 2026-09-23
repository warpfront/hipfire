	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gu_noscale              ; -- Begin function gu_noscale
	.globl	gu_noscale
	.p2align	8
	.type	gu_noscale,@function
gu_noscale:                             ; @gu_noscale
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[16:19], s[0:1], 0x38
	v_bfe_u32 v2, v0, 4, 1
	v_lshrrev_b32_e32 v1, 1, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v161, 3, v2
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s18, 0xff
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	v_lshlrev_b32_e32 v2, 3, v2
	s_mov_b32 s2, 0
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b32 s2, -1
                                        ; implicit-def: $vgpr2
.LBB0_3:
	v_and_b32_e32 v159, 15, v0
	v_and_b32_e32 v160, 0x60, v0
	v_and_b32_e32 v158, 64, v1
	s_lshl_b32 s23, ttmp7, 7
	s_add_co_i32 s22, s17, s16
	s_lshl_b32 s24, ttmp9, 7
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_mov_b32 s21, 0
	s_cbranch_vccnz .LBB0_20
; %bb.4:
	s_load_b256 s[8:15], s[0:1], 0x0
	v_lshrrev_b32_e32 v2, 2, v0
	s_add_co_i32 s5, s22, -1
	s_ashr_i32 s2, s18, 31
	v_and_b32_e32 v4, 0x7f, v0
	s_lshr_b32 s2, s2, 24
	v_add_nc_u32_e32 v3, s24, v2
	s_add_co_i32 s2, s18, s2
	v_or_b32_e32 v7, 64, v2
	s_ashr_i32 s25, s2, 8
	s_add_co_i32 s20, s19, -1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s7, s25, 0x88
	v_mul_u32_u24_e32 v22, 0x48, v2
	v_mul_u32_u24_e32 v21, 0x48, v7
	v_mad_u32_u24 v28, 0x48, v2, 0
                                        ; implicit-def: $vgpr84_vgpr85_vgpr86_vgpr87_vgpr88_vgpr89_vgpr90_vgpr91
                                        ; implicit-def: $vgpr92_vgpr93_vgpr94_vgpr95_vgpr96_vgpr97_vgpr98_vgpr99
                                        ; implicit-def: $vgpr100_vgpr101_vgpr102_vgpr103_vgpr104_vgpr105_vgpr106_vgpr107
                                        ; implicit-def: $vgpr108_vgpr109_vgpr110_vgpr111_vgpr112_vgpr113_vgpr114_vgpr115
                                        ; implicit-def: $vgpr116_vgpr117_vgpr118_vgpr119_vgpr120_vgpr121_vgpr122_vgpr123
                                        ; implicit-def: $vgpr124_vgpr125_vgpr126_vgpr127_vgpr128_vgpr129_vgpr130_vgpr131
                                        ; implicit-def: $vgpr132_vgpr133_vgpr134_vgpr135_vgpr136_vgpr137_vgpr138_vgpr139
                                        ; implicit-def: $vgpr140_vgpr141_vgpr142_vgpr143_vgpr144_vgpr145_vgpr146_vgpr147
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v12, s8
	v_min_i32_e32 v5, s5, v3
	v_add_nc_u32_e32 v9, 64, v3
	v_cmp_gt_i32_e64 s3, s22, v3
	v_mov_b32_e32 v11, s9
	v_or_b32_e32 v14, s24, v4
	v_cmp_gt_i32_e32 vcc_lo, s16, v5
	v_cndmask_b32_e64 v8, s16, 0, vcc_lo
	v_cndmask_b32_e32 v15, s10, v12, vcc_lo
	v_cndmask_b32_e32 v16, s11, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_sub_nc_u32_e32 v3, v5, v8
	v_mov_b32_e32 v8, 0
	v_min_i32_e32 v5, s5, v9
	v_and_b32_e32 v7, 0x6f, v0
	v_add_nc_u32_e32 v6, s23, v2
	v_mul_lo_u32 v3, s7, v3
	v_dual_mov_b32 v2, v8 :: v_dual_lshlrev_b32 v23, 2, v1
	v_cmp_gt_i32_e64 s6, s16, v5
	v_mad_u32_u24 v26, 0x48, v7, 0
	v_lshlrev_b32_e32 v24, 3, v4
	v_or_b32_e32 v4, v161, v160
	v_add_nc_u32_e32 v27, 0, v161
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v13, s16, 0, s6
	v_add_co_u32 v3, vcc_lo, v15, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, 0, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_sub_nc_u32_e32 v5, v5, v13
	v_min_i32_e32 v13, s5, v14
	v_cmp_gt_i32_e64 s5, s22, v9
	v_cndmask_b32_e64 v9, s10, v12, s6
	v_cndmask_b32_e64 v18, s11, v11, s6
	v_mul_lo_u32 v5, s7, v5
	v_cmp_gt_i32_e32 vcc_lo, s16, v13
	v_add_nc_u32_e32 v16, s23, v1
	v_or_b32_e32 v1, v158, v159
	v_lshlrev_b32_e32 v32, 3, v4
	v_mov_b32_e32 v4, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v17, s16, 0, vcc_lo
	v_cndmask_b32_e32 v12, s10, v12, vcc_lo
	v_cndmask_b32_e32 v11, s11, v11, vcc_lo
	v_add_co_u32 v5, s6, v9, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_sub_nc_u32_e32 v13, v13, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, 0, v18, s6
	v_mul_u32_u24_e32 v30, 0x48, v1
	v_lshlrev_b32_e32 v31, 2, v1
	v_mul_lo_u32 v13, s7, v13
	v_cmp_gt_i32_e64 s7, s22, v14
	v_dual_mov_b32 v7, v8 :: v_dual_and_b32 v14, 3, v0
	v_mov_b32_e32 v1, v8
	v_min_i32_e32 v19, s20, v16
	v_mov_b32_e32 v18, v8
	v_cmp_gt_i32_e64 s6, s19, v16
	v_add_co_u32 v163, vcc_lo, v12, v13
	v_lshlrev_b32_e32 v12, 3, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v164, null, 0, v11, vcc_lo
	v_lshrrev_b32_e32 v11, 7, v0
	v_mul_lo_u32 v162, v19, s25
	v_add_co_u32 v166, vcc_lo, v3, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v167, null, 0, v15, vcc_lo
	v_add_co_u32 v168, vcc_lo, v5, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v169, null, 0, v9, vcc_lo
	v_dual_mov_b32 v9, v8 :: v_dual_add_nc_u32 v10, 64, v6
	v_dual_mov_b32 v3, v8 :: v_dual_lshlrev_b32 v20, 4, v14
	v_mov_b32_e32 v5, v8
	v_lshl_add_u32 v25, v11, 2, 0
	v_lshlrev_b32_e32 v165, 4, v11
	v_mov_b32_e32 v11, v8
	v_cmp_gt_i32_e64 s4, s19, v10
	v_min_i32_e32 v10, s20, v10
	v_cmp_gt_i32_e64 s2, s19, v6
	v_min_i32_e32 v6, s20, v6
	v_dual_mov_b32 v12, v8 :: v_dual_add_nc_u32 v29, 0, v20
	v_dual_mov_b32 v19, v8 :: v_dual_mov_b32 v14, v8
	v_mad_co_u64_u32 v[152:153], null, v10, s18, v[20:21]
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[153:154], null, v6, s18, v[20:21]
	v_mov_b32_e32 v13, v8
	v_dual_mov_b32 v15, v8 :: v_dual_mov_b32 v16, v8
	v_dual_mov_b32 v17, v8 :: v_dual_add_nc_u32 v170, 0, v23
	v_add_nc_u32_e32 v171, v25, v24
	v_add_nc_u32_e32 v174, v28, v20
	v_add3_u32 v175, v29, v22, 0x2400
	v_add3_u32 v176, v29, v21, 0x2400
	v_add_nc_u32_e32 v177, v26, v161
	v_add_nc_u32_e32 v178, v27, v30
	v_dual_mov_b32 v27, v19 :: v_dual_add_nc_u32 v172, 0, v31
	v_mov_b32_e32 v25, v17
	v_dual_mov_b32 v24, v16 :: v_dual_add_nc_u32 v173, 0, v32
	v_dual_mov_b32 v22, v14 :: v_dual_mov_b32 v35, v19
	v_dual_mov_b32 v28, v12 :: v_dual_mov_b32 v43, v19
	v_dual_mov_b32 v36, v12 :: v_dual_mov_b32 v51, v19
	v_mov_b32_e32 v10, v8
	v_mov_b32_e32 v6, v8
	v_dual_mov_b32 v44, v12 :: v_dual_mov_b32 v59, v19
	v_dual_mov_b32 v52, v12 :: v_dual_mov_b32 v151, v11
	v_dual_mov_b32 v148, v8 :: v_dual_mov_b32 v67, v19
	v_dual_mov_b32 v60, v12 :: v_dual_mov_b32 v75, v19
	v_dual_mov_b32 v68, v12 :: v_dual_mov_b32 v83, v8
	v_dual_mov_b32 v26, v18 :: v_dual_mov_b32 v23, v15
	v_dual_mov_b32 v21, v13 :: v_dual_mov_b32 v20, v12
	v_dual_mov_b32 v34, v18 :: v_dual_mov_b32 v33, v17
	v_dual_mov_b32 v32, v16 :: v_dual_mov_b32 v31, v15
	v_dual_mov_b32 v30, v14 :: v_dual_mov_b32 v29, v13
	v_dual_mov_b32 v42, v18 :: v_dual_mov_b32 v41, v17
	v_dual_mov_b32 v40, v16 :: v_dual_mov_b32 v39, v15
	v_dual_mov_b32 v38, v14 :: v_dual_mov_b32 v37, v13
	v_dual_mov_b32 v50, v18 :: v_dual_mov_b32 v49, v17
	v_dual_mov_b32 v48, v16 :: v_dual_mov_b32 v47, v15
	v_dual_mov_b32 v46, v14 :: v_dual_mov_b32 v45, v13
	v_dual_mov_b32 v58, v18 :: v_dual_mov_b32 v57, v17
	v_dual_mov_b32 v56, v16 :: v_dual_mov_b32 v55, v15
	v_dual_mov_b32 v54, v14 :: v_dual_mov_b32 v53, v13
	v_dual_mov_b32 v150, v10 :: v_dual_mov_b32 v149, v9
	v_dual_mov_b32 v66, v18 :: v_dual_mov_b32 v65, v17
	v_dual_mov_b32 v64, v16 :: v_dual_mov_b32 v63, v15
	v_dual_mov_b32 v62, v14 :: v_dual_mov_b32 v61, v13
	v_dual_mov_b32 v74, v18 :: v_dual_mov_b32 v73, v17
	v_dual_mov_b32 v72, v16 :: v_dual_mov_b32 v71, v15
	v_dual_mov_b32 v70, v14 :: v_dual_mov_b32 v69, v13
	v_dual_mov_b32 v82, v7 :: v_dual_mov_b32 v81, v6
	v_dual_mov_b32 v80, v5 :: v_dual_mov_b32 v79, v4
	v_dual_mov_b32 v78, v3 :: v_dual_mov_b32 v77, v2
	v_mov_b32_e32 v76, v1
	s_mov_b32 s10, 0x4e4c4a48
	s_mov_b32 s11, 0x4040404
	s_mov_b32 s18, 0
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s18, s25
	s_cbranch_scc1 .LBB0_21
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
                                        ;       Child Loop BB0_10 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	v_add_lshl_u32 v11, s18, v162, 1
	s_lshl_b32 s20, s18, 8
	s_mul_i32 s26, s18, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[12:13], s[20:21]
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
	v_or_b32_e32 v7, s29, v11
	s_xor_b32 s27, s20, -1
	s_lshl_b32 s20, s29, 2
	s_mov_b32 s28, -1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s20, s20, s26
	v_lshlrev_b64_e32 v[1:2], 2, v[7:8]
	v_add_nc_u32_e32 v7, 0x1200, v174
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s14, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s15, v2, vcc_lo
	global_load_b32 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v1, 0, v1, s6
	ds_store_b32 v170, v1 offset:18432
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v163, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v164, vcc_lo
	s_lshl_b32 s20, s29, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[8:9], s[20:21]
	global_load_b32 v1, v[1:2], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s20, s30, v153
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s31, 0, s20
	v_add_co_u32 v9, s20, s30, v152
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s31, 0, s20
	s_lshl_b32 s20, s29, 6
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s20, s26, s20
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v154, vcc_lo, v166, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, 0, v167, vcc_lo
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v165, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s7
	ds_store_b32 v171, v1 offset:18944
	global_load_b128 v[1:4], v[5:6], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v2, 0, v2, s2
	v_cndmask_b32_e64 v1, 0, v1, s2
	v_cndmask_b32_e64 v4, 0, v4, s2
	v_cndmask_b32_e64 v3, 0, v3, s2
	ds_store_2addr_b64 v174, v[1:2], v[3:4] offset1:1
	global_load_b128 v[1:4], v[9:10], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v2, 0, v2, s4
	v_cndmask_b32_e64 v1, 0, v1, s4
	v_cndmask_b32_e64 v4, 0, v4, s4
	v_cndmask_b32_e64 v3, 0, v3, s4
	ds_store_2addr_b64 v7, v[1:2], v[3:4] offset1:1
	global_load_b64 v[1:2], v[154:155], off offset:8
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v1, 0, v1, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v4, v1, v3, 0x5010400
	v_perm_b32 v1, v1, v3, 0x7030602
	v_and_b32_e32 v3, 0x7070707, v4
	v_lshrrev_b32_e32 v4, 1, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v156, s10, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v4, v4, s11, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v3, v156, v4
	v_and_b32_e32 v4, 0x7070707, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_perm_b32 v156, s10, 0x44403800, v4
	v_or_b32_e32 v4, 0x50505050, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, v1, s11, 0x3020100
	v_perm_b32 v4, v4, v156, v1
	v_cndmask_b32_e64 v1, 0, v2, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v156, v1, v2, 0x5010400
	v_perm_b32 v2, v1, v2, 0x7030602
	v_and_b32_e32 v1, 0x7070707, v156
	v_lshrrev_b32_e32 v156, 1, v156
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v157, s10, 0x44403800, v1
	v_or_b32_e32 v1, 0x50505050, v1
	v_and_or_b32 v156, v156, s11, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v1, v1, v157, v156
	v_and_b32_e32 v156, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	v_perm_b32 v157, s10, 0x44403800, v156
	v_or_b32_e32 v156, 0x50505050, v156
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, v2, s11, 0x3020100
	v_perm_b32 v2, v156, v157, v2
	v_add_co_u32 v156, vcc_lo, v168, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v157, null, 0, v169, vcc_lo
	ds_store_2addr_b64 v175, v[3:4], v[1:2] offset1:1
	global_load_b64 v[1:2], v[156:157], off offset:8
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v1, 0, v1, s5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v3, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v4, v1, v3, 0x5010400
	v_perm_b32 v1, v1, v3, 0x7030602
	v_and_b32_e32 v3, 0x7070707, v4
	v_lshrrev_b32_e32 v4, 1, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v179, s10, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v4, v4, s11, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v3, v179, v4
	v_and_b32_e32 v4, 0x7070707, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_perm_b32 v179, s10, 0x44403800, v4
	v_or_b32_e32 v4, 0x50505050, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, v1, s11, 0x3020100
	v_perm_b32 v4, v4, v179, v1
	v_cndmask_b32_e64 v1, 0, v2, s5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v179, v1, v2, 0x5010400
	v_perm_b32 v2, v1, v2, 0x7030602
	v_and_b32_e32 v1, 0x7070707, v179
	v_lshrrev_b32_e32 v179, 1, v179
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v180, s10, 0x44403800, v1
	v_or_b32_e32 v1, 0x50505050, v1
	v_and_or_b32 v179, v179, s11, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v1, v1, v180, v179
	v_and_b32_e32 v179, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	v_perm_b32 v180, s10, 0x44403800, v179
	v_or_b32_e32 v179, 0x50505050, v179
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, v2, s11, 0x3020100
	v_perm_b32 v2, v179, v180, v2
	ds_store_2addr_b64 v176, v[3:4], v[1:2] offset1:1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_10
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=3
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s29, -1
	s_mov_b32 s28, 0
	s_and_b32 vcc_lo, exec_lo, s20
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_7
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_8 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_xor_b32 s20, s28, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=3
	s_clause 0x1
	global_load_b128 v[1:4], v[5:6], off offset:64
	global_load_b128 v[80:83], v[9:10], off offset:64
	global_load_b64 v[148:149], v[154:155], off offset:40
	global_load_b64 v[150:151], v[156:157], off offset:40
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v77, 0, v2, s2
	v_cndmask_b32_e64 v76, 0, v1, s2
	v_cndmask_b32_e64 v79, 0, v4, s2
	v_cndmask_b32_e64 v78, 0, v3, s2
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v81, 0, v81, s4
	v_cndmask_b32_e64 v80, 0, v80, s4
	v_cndmask_b32_e64 v83, 0, v83, s4
	v_cndmask_b32_e64 v82, 0, v82, s4
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v148, 0, v148, s3
	v_cndmask_b32_e64 v149, 0, v149, s3
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v150, 0, v150, s5
	v_cndmask_b32_e64 v151, 0, v151, s5
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=3
	v_add_nc_u32_e32 v180, 0x2400, v177
	v_add_nc_u32_e32 v179, 0x800, v178
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_mov_b32 s30, -1
	ds_load_2addr_b64 v[1:4], v180 offset1:144
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_16
; %bb.13:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s30
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_17
.LBB0_14:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_18
.LBB0_15:                               ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
	s_branch .LBB0_19
.LBB0_16:                               ;   in Loop: Header=BB0_10 Depth=3
	ds_load_2addr_b64 v[181:184], v178 offset1:144
	ds_load_2addr_b64 v[185:188], v179 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[181:182], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[181:182], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[183:184], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[183:184], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[181:184], v180 offset0:2 offset1:146
	ds_load_2addr_b64 v[185:188], v178 offset0:2 offset1:146
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[181:182], v[185:186], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[183:184], v[185:186], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[181:182], v[187:188], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[183:184], v[187:188], v[116:123]
	ds_load_2addr_b64 v[185:188], v179 offset0:34 offset1:178
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[181:182], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[183:184], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[181:182], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[183:184], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[181:184], v180 offset0:4 offset1:148
	ds_load_2addr_b64 v[185:188], v178 offset0:4 offset1:148
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[181:182], v[185:186], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[183:184], v[185:186], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[181:182], v[187:188], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[183:184], v[187:188], v[116:123]
	ds_load_2addr_b64 v[185:188], v179 offset0:36 offset1:180
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[181:182], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[183:184], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[181:182], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[183:184], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[181:184], v180 offset0:6 offset1:150
	ds_load_2addr_b64 v[185:188], v178 offset0:6 offset1:150
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[181:182], v[185:186], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[183:184], v[185:186], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[181:182], v[187:188], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[183:184], v[187:188], v[116:123]
	ds_load_2addr_b64 v[185:188], v179 offset0:38 offset1:182
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[181:182], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[183:184], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[181:182], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[183:184], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	s_cbranch_execnz .LBB0_14
.LBB0_17:                               ;   in Loop: Header=BB0_10 Depth=3
	ds_load_2addr_b64 v[84:87], v178 offset1:144
	ds_load_2addr_b64 v[181:184], v179 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_nop
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[84:85], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[84:85], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[86:87], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[86:87], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[181:182], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[181:182], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[183:184], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[183:184], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v180 offset0:2 offset1:146
	ds_load_2addr_b64 v[181:184], v178 offset0:2 offset1:146
	ds_load_2addr_b64 v[185:188], v179 offset0:34 offset1:178
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[181:182], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[181:182], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[183:184], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[183:184], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v180 offset0:4 offset1:148
	ds_load_2addr_b64 v[181:184], v178 offset0:4 offset1:148
	ds_load_2addr_b64 v[185:188], v179 offset0:36 offset1:180
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[181:182], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[181:182], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[183:184], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[183:184], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[185:186], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[185:186], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[187:188], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[187:188], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v180 offset0:6 offset1:150
	ds_load_2addr_b64 v[180:183], v178 offset0:6 offset1:150
	ds_load_2addr_b64 v[184:187], v179 offset0:38 offset1:182
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[180:181], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[180:181], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[182:183], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[182:183], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[184:185], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[184:185], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[186:187], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[186:187], v[84:91]
	; sched_barrier mask(0x00000000)
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_15
.LBB0_18:                               ;   in Loop: Header=BB0_10 Depth=3
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v2, 4, v148
	v_and_b32_e32 v1, 0xf0f0f0f, v148
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0xf0f0f0f, v2
	v_perm_b32 v3, v2, v1, 0x5010400
	v_perm_b32 v2, v2, v1, 0x7030602
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 0x7070707, v3
	v_lshrrev_b32_e32 v3, 1, v3
	s_barrier_wait -1
	v_perm_b32 v4, s10, 0x44403800, v1
	v_or_b32_e32 v1, 0x50505050, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_and_or_b32 v3, v3, s11, 0x3020100
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v174, v[76:77], v[78:79] offset1:1
	ds_store_2addr_b64 v7, v[80:81], v[82:83] offset1:1
	v_perm_b32 v1, v1, v4, v3
	v_and_b32_e32 v3, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v4, s10, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v2, v2, s11, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v2, v3, v4, v2
	v_lshrrev_b32_e32 v4, 4, v149
	v_and_b32_e32 v3, 0xf0f0f0f, v149
	v_and_b32_e32 v4, 0xf0f0f0f, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v179, v4, v3, 0x5010400
	v_perm_b32 v4, v4, v3, 0x7030602
	v_and_b32_e32 v3, 0x7070707, v179
	v_lshrrev_b32_e32 v179, 1, v179
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v180, s10, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v179, v179, s11, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v3, v180, v179
	v_and_b32_e32 v179, 0x7070707, v4
	v_lshrrev_b32_e32 v4, 1, v4
	v_perm_b32 v180, s10, 0x44403800, v179
	v_or_b32_e32 v179, 0x50505050, v179
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v4, v4, s11, 0x3020100
	v_perm_b32 v4, v179, v180, v4
	ds_store_2addr_b64 v175, v[1:2], v[3:4] offset1:1
	v_lshrrev_b32_e32 v2, 4, v150
	v_and_b32_e32 v1, 0xf0f0f0f, v150
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0xf0f0f0f, v2
	v_perm_b32 v3, v2, v1, 0x5010400
	v_perm_b32 v2, v2, v1, 0x7030602
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 0x7070707, v3
	v_lshrrev_b32_e32 v3, 1, v3
	v_perm_b32 v4, s10, 0x44403800, v1
	v_or_b32_e32 v1, 0x50505050, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v3, v3, s11, 0x3020100
	v_perm_b32 v1, v1, v4, v3
	v_and_b32_e32 v3, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v4, s10, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v2, v2, s11, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v2, v3, v4, v2
	v_lshrrev_b32_e32 v4, 4, v151
	v_and_b32_e32 v3, 0xf0f0f0f, v151
	v_and_b32_e32 v4, 0xf0f0f0f, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v179, v4, v3, 0x5010400
	v_perm_b32 v4, v4, v3, 0x7030602
	v_and_b32_e32 v3, 0x7070707, v179
	v_lshrrev_b32_e32 v179, 1, v179
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v180, s10, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v179, v179, s11, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v3, v180, v179
	v_and_b32_e32 v179, 0x7070707, v4
	v_lshrrev_b32_e32 v4, 1, v4
	v_perm_b32 v180, s10, 0x44403800, v179
	v_or_b32_e32 v179, 0x50505050, v179
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v4, v4, s11, 0x3020100
	v_perm_b32 v4, v179, v180, v4
	ds_store_2addr_b64 v176, v[1:2], v[3:4] offset1:1
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
.LBB0_19:                               ;   in Loop: Header=BB0_10 Depth=3
	v_add_nc_u32_e32 v179, 0x4a10, v173
	v_add_nc_u32_e32 v183, 0x4800, v172
	ds_load_2addr_b32 v[179:180], v179 offset1:1
	ds_load_2addr_b32 v[181:182], v183 offset1:16
	ds_load_2addr_b32 v[183:184], v183 offset0:32 offset1:48
	s_wait_dscnt 0x3
	v_add_nc_u32_e32 v1, 0x4a00, v173
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_add_nc_u32_e32 v3, 0x4a08, v173
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	s_wait_dscnt 0x4
	v_fma_f32 v54, v179, v126, v54
	v_add_nc_u32_e32 v185, 0x4a18, v173
	v_fma_f32 v38, v179, v110, v38
	v_add_nc_u32_e32 v187, 0x4a20, v173
	v_fma_f32 v70, v179, v142, v70
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v54, v180, v182
	v_fma_f32 v22, v179, v94, v22
	s_wait_dscnt 0x2
	v_fmac_f32_e32 v38, v180, v183
	ds_load_2addr_b32 v[185:186], v185 offset1:1
	ds_load_2addr_b32 v[187:188], v187 offset1:1
	s_wait_dscnt 0x3
	v_fma_f32 v68, v1, v140, v68
	v_fma_f32 v52, v1, v124, v52
	v_fma_f32 v36, v1, v108, v36
	v_fma_f32 v20, v1, v92, v20
	v_dual_fmac_f32 v22, v180, v184 :: v_dual_add_nc_u32 v1, 0x4a28, v173
	s_wait_dscnt 0x2
	v_fma_f32 v69, v3, v141, v69
	v_fma_f32 v53, v3, v125, v53
	v_fma_f32 v37, v3, v109, v37
	v_fma_f32 v21, v3, v93, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v68, v2, v181 :: v_dual_fmac_f32 v69, v4, v181
	v_fmac_f32_e32 v53, v4, v182
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v37, v4, v183
	v_dual_fmac_f32 v21, v4, v184 :: v_dual_fmac_f32 v52, v2, v182
	v_fmac_f32_e32 v36, v2, v183
	v_fmac_f32_e32 v20, v2, v184
	s_wait_dscnt 0x1
	v_fma_f32 v71, v185, v143, v71
	v_fma_f32 v55, v185, v127, v55
	v_fma_f32 v39, v185, v111, v39
	v_fma_f32 v23, v185, v95, v23
	s_wait_dscnt 0x0
	v_fma_f32 v72, v187, v144, v72
	v_fma_f32 v56, v187, v128, v56
	v_add_nc_u32_e32 v3, 0x4a30, v173
	v_fma_f32 v40, v187, v112, v40
	v_fma_f32 v24, v187, v96, v24
	v_add_nc_u32_e32 v185, 0x4a90, v173
	v_add_nc_u32_e32 v179, 0x4a38, v173
	v_add_nc_u32_e32 v187, 0x4a98, v173
	v_dual_fmac_f32 v70, v180, v181 :: v_dual_fmac_f32 v71, v186, v181
	v_dual_fmac_f32 v55, v186, v182 :: v_dual_fmac_f32 v72, v188, v181
	v_dual_fmac_f32 v39, v186, v183 :: v_dual_fmac_f32 v24, v188, v184
	v_dual_fmac_f32 v23, v186, v184 :: v_dual_fmac_f32 v56, v188, v182
	v_fmac_f32_e32 v40, v188, v183
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	ds_load_2addr_b32 v[179:180], v179 offset1:1
	ds_load_2addr_b32 v[185:186], v185 offset1:1
	ds_load_2addr_b32 v[187:188], v187 offset1:1
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	s_wait_dscnt 0x4
	v_fma_f32 v74, v3, v146, v74
	v_fma_f32 v58, v3, v130, v58
	v_fma_f32 v42, v3, v114, v42
	v_fma_f32 v26, v3, v98, v26
	s_wait_dscnt 0x2
	v_fma_f32 v46, v185, v118, v46
	v_add_nc_u32_e32 v3, 0x4a88, v173
	v_fmac_f32_e32 v74, v4, v181
	v_fmac_f32_e32 v58, v4, v182
	v_fmac_f32_e32 v42, v4, v183
	v_fmac_f32_e32 v26, v4, v184
	v_fmac_f32_e32 v46, v186, v182
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	s_wait_dscnt 0x1
	v_fma_f32 v73, v1, v145, v73
	v_fma_f32 v57, v1, v129, v57
	v_fma_f32 v41, v1, v113, v41
	v_fma_f32 v25, v1, v97, v25
	v_add_nc_u32_e32 v1, 0x4a80, v173
	v_fmac_f32_e32 v73, v2, v181
	v_fmac_f32_e32 v57, v2, v182
	v_fmac_f32_e32 v41, v2, v183
	v_fmac_f32_e32 v25, v2, v184
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_fmac_f32_e32 v27, v179, v99
	v_fma_f32 v62, v185, v134, v62
	v_fma_f32 v30, v185, v102, v30
	v_fma_f32 v14, v185, v86, v14
	v_fmac_f32_e32 v75, v179, v147
	v_add_nc_u32_e32 v185, 0x4ab8, v173
	v_dual_fmac_f32 v27, v180, v184 :: v_dual_fmac_f32 v62, v186, v181
	v_fmac_f32_e32 v30, v186, v183
	s_wait_dscnt 0x1
	v_fma_f32 v61, v3, v133, v61
	v_fma_f32 v45, v3, v117, v45
	v_fma_f32 v29, v3, v101, v29
	v_fma_f32 v13, v3, v85, v13
	v_add_nc_u32_e32 v3, 0x4aa8, v173
	v_fmac_f32_e32 v61, v4, v181
	v_fmac_f32_e32 v45, v4, v182
	v_fmac_f32_e32 v29, v4, v183
	v_fmac_f32_e32 v13, v4, v184
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	s_wait_dscnt 0x1
	v_fma_f32 v60, v1, v132, v60
	v_fma_f32 v44, v1, v116, v44
	v_fma_f32 v28, v1, v100, v28
	v_fma_f32 v12, v1, v84, v12
	v_add_nc_u32_e32 v1, 0x4aa0, v173
	v_fmac_f32_e32 v60, v2, v181
	v_fmac_f32_e32 v44, v2, v182
	v_fmac_f32_e32 v28, v2, v183
	v_fmac_f32_e32 v12, v2, v184
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_dual_fmac_f32 v14, v186, v184 :: v_dual_fmac_f32 v75, v180, v181
	v_fma_f32 v63, v187, v135, v63
	v_fma_f32 v47, v187, v119, v47
	v_fma_f32 v31, v187, v103, v31
	v_fma_f32 v15, v187, v87, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v63, v188, v181
	v_fmac_f32_e32 v47, v188, v182
	s_wait_dscnt 0x1
	v_fma_f32 v65, v3, v137, v65
	v_fma_f32 v49, v3, v121, v49
	v_fma_f32 v33, v3, v105, v33
	v_fma_f32 v17, v3, v89, v17
	v_fmac_f32_e32 v31, v188, v183
	v_fmac_f32_e32 v15, v188, v184
	v_fmac_f32_e32 v65, v4, v181
	v_fmac_f32_e32 v49, v4, v182
	v_fmac_f32_e32 v33, v4, v183
	v_fmac_f32_e32 v17, v4, v184
	s_wait_dscnt 0x0
	v_fma_f32 v48, v1, v120, v48
	v_fmac_f32_e32 v59, v179, v131
	v_fma_f32 v16, v1, v88, v16
	v_fmac_f32_e32 v43, v179, v115
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v48, v2, v182 :: v_dual_add_nc_u32 v179, 0x4ab0, v173
	v_dual_fmac_f32 v59, v180, v182 :: v_dual_fmac_f32 v16, v2, v184
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v43, v180, v183
	ds_load_2addr_b32 v[179:180], v179 offset1:1
	ds_load_2addr_b32 v[185:186], v185 offset1:1
	v_fma_f32 v64, v1, v136, v64
	v_fma_f32 v32, v1, v104, v32
	v_fmac_f32_e32 v64, v2, v181
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v32, v2, v183
	s_wait_dscnt 0x1
	v_fma_f32 v66, v179, v138, v66
	v_fma_f32 v50, v179, v122, v50
	v_fma_f32 v34, v179, v106, v34
	v_fma_f32 v18, v179, v90, v18
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v67, v185, v139
	v_fmac_f32_e32 v51, v185, v123
	v_fmac_f32_e32 v35, v185, v107
	v_dual_fmac_f32 v19, v185, v91 :: v_dual_fmac_f32 v66, v180, v181
	v_fmac_f32_e32 v50, v180, v182
	v_fmac_f32_e32 v34, v180, v183
	v_dual_fmac_f32 v18, v180, v184 :: v_dual_fmac_f32 v67, v186, v181
	v_fmac_f32_e32 v51, v186, v182
	v_fmac_f32_e32 v35, v186, v183
	v_fmac_f32_e32 v19, v186, v184
	s_branch .LBB0_9
.LBB0_20:
	v_mov_b32_e32 v68, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v69, v68 :: v_dual_mov_b32 v70, v68
	v_dual_mov_b32 v71, v68 :: v_dual_mov_b32 v72, v68
	v_dual_mov_b32 v73, v68 :: v_dual_mov_b32 v74, v68
	v_mov_b32_e32 v75, v68
	v_dual_mov_b32 v60, v68 :: v_dual_mov_b32 v61, v69
	v_dual_mov_b32 v52, v68 :: v_dual_mov_b32 v53, v69
	v_dual_mov_b32 v44, v68 :: v_dual_mov_b32 v45, v69
	v_dual_mov_b32 v36, v68 :: v_dual_mov_b32 v37, v69
	v_dual_mov_b32 v28, v68 :: v_dual_mov_b32 v29, v69
	v_dual_mov_b32 v20, v68 :: v_dual_mov_b32 v21, v69
	v_dual_mov_b32 v12, v68 :: v_dual_mov_b32 v13, v69
	v_dual_mov_b32 v62, v70 :: v_dual_mov_b32 v63, v71
	v_dual_mov_b32 v64, v72 :: v_dual_mov_b32 v65, v73
	v_dual_mov_b32 v66, v74 :: v_dual_mov_b32 v67, v75
	v_dual_mov_b32 v54, v70 :: v_dual_mov_b32 v55, v71
	v_dual_mov_b32 v56, v72 :: v_dual_mov_b32 v57, v73
	v_dual_mov_b32 v58, v74 :: v_dual_mov_b32 v59, v75
	v_dual_mov_b32 v46, v70 :: v_dual_mov_b32 v47, v71
	v_dual_mov_b32 v48, v72 :: v_dual_mov_b32 v49, v73
	v_dual_mov_b32 v50, v74 :: v_dual_mov_b32 v51, v75
	v_dual_mov_b32 v38, v70 :: v_dual_mov_b32 v39, v71
	v_dual_mov_b32 v40, v72 :: v_dual_mov_b32 v41, v73
	v_dual_mov_b32 v42, v74 :: v_dual_mov_b32 v43, v75
	v_dual_mov_b32 v30, v70 :: v_dual_mov_b32 v31, v71
	v_dual_mov_b32 v32, v72 :: v_dual_mov_b32 v33, v73
	v_dual_mov_b32 v34, v74 :: v_dual_mov_b32 v35, v75
	v_dual_mov_b32 v22, v70 :: v_dual_mov_b32 v23, v71
	v_dual_mov_b32 v24, v72 :: v_dual_mov_b32 v25, v73
	v_dual_mov_b32 v26, v74 :: v_dual_mov_b32 v27, v75
	v_dual_mov_b32 v14, v70 :: v_dual_mov_b32 v15, v71
	v_dual_mov_b32 v16, v72 :: v_dual_mov_b32 v17, v73
	v_dual_mov_b32 v18, v74 :: v_dual_mov_b32 v19, v75
	s_branch .LBB0_22
.LBB0_21:
	v_mov_b32_e32 v2, v161
.LBB0_22:
	s_load_b128 s[4:7], s[0:1], 0x28
	v_lshlrev_b32_e32 v1, 6, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v3, v2, v159
	v_and_b32_e32 v76, 31, v0
	v_and_b32_e32 v4, 31, v3
	v_add_nc_u32_e32 v5, 17, v3
	v_add_nc_u32_e32 v6, 18, v3
	v_add_nc_u32_e32 v11, 19, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_xor_b32_e32 v4, 16, v4
	v_and_b32_e32 v5, 31, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_and_b32_e32 v6, 31, v6
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v78, s5 :: v_dual_and_b32 v11, 31, v11
	v_and_b32_e32 v1, 0x3800, v1
	v_add_nc_u32_e32 v2, 0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v77, v159, 7, v2
	v_lshl_add_u32 v8, v4, 2, v77
	v_add_nc_u32_e32 v4, 20, v3
	v_lshl_add_u32 v9, v5, 2, v77
	v_add_nc_u32_e32 v5, 21, v3
	v_lshl_add_u32 v10, v6, 2, v77
	v_add_nc_u32_e32 v6, 22, v3
	v_lshl_add_u32 v7, v3, 2, v77
	v_add_nc_u32_e32 v3, 23, v3
	v_and_b32_e32 v4, 31, v4
	v_and_b32_e32 v5, 31, v5
	v_and_b32_e32 v6, 31, v6
	v_lshl_add_u32 v11, v11, 2, v77
	v_and_b32_e32 v3, 31, v3
	ds_store_2addr_b32 v7, v68, v69 offset1:1
	ds_store_2addr_b32 v7, v70, v71 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v72, v73 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v74, v75 offset0:6 offset1:7
	ds_store_b32 v8, v60
	ds_store_b32 v9, v61
	ds_store_b32 v10, v62
	v_lshl_add_u32 v60, v4, 2, v77
	v_lshl_add_u32 v61, v5, 2, v77
	v_lshl_add_u32 v62, v6, 2, v77
	v_lshl_add_u32 v68, v3, 2, v77
	ds_store_b32 v11, v63
	ds_store_b32 v60, v64
	ds_store_b32 v61, v65
	ds_store_b32 v62, v66
	ds_store_b32 v68, v67
	v_mov_b32_e32 v63, s16
	v_or3_b32 v1, s24, v160, v76
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v6, s23, v158
	v_mov_b32_e32 v4, s4
	v_cmp_gt_i32_e64 s0, s16, v1
	v_cmp_gt_i32_e64 s1, s22, v1
	v_cmp_le_i32_e32 vcc_lo, s22, v1
	v_cmp_gt_i32_e64 s2, s19, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v5, s16, 0, s0
	v_cndmask_b32_e64 v3, s7, v78, s0
	v_cndmask_b32_e64 v4, s6, v4, s0
	v_sub_nc_u32_e32 v1, v1, v5
	v_cndmask_b32_e64 v5, s17, v63, s0
	v_lshl_add_u32 v63, v76, 2, v2
	s_and_b32 s0, s2, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_mad_co_u64_u32 v[64:65], null, v6, v5, v[1:2]
	ds_load_b32 v66, v63
	v_mov_b32_e32 v65, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	v_add_co_u32 v64, s0, v4, v64
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v65, null, v3, v65, s0
	s_wait_dscnt 0x0
	global_store_b32 v[64:65], v66, off
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v65, 1, v6
	v_add_nc_u32_e32 v64, 1, v0
	s_xor_b32 s1, vcc_lo, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s0, s19, v65
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_and_b32_e32 v66, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v67, v66, 2, v2
	v_mad_co_u64_u32 v[65:66], null, v65, v5, v[1:2]
	v_mov_b32_e32 v66, 0
	ds_load_b32 v67, v67 offset:128
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	v_add_co_u32 v65, vcc_lo, v4, v65
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v66, null, v3, v66, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[65:66], v67, off
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v66, 2, v6
	v_add_nc_u32_e32 v65, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v66
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_and_b32_e32 v67, 31, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v69, v67, 2, v2
	v_mad_co_u64_u32 v[66:67], null, v66, v5, v[1:2]
	v_mov_b32_e32 v67, 0
	ds_load_b32 v69, v69 offset:256
	v_lshlrev_b64_e32 v[66:67], 2, v[66:67]
	v_add_co_u32 v66, vcc_lo, v4, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v67, null, v3, v67, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[66:67], v69, off
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v67, 3, v6
	v_add_nc_u32_e32 v66, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v67
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_and_b32_e32 v69, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v71, v69, 2, v2
	v_mad_co_u64_u32 v[69:70], null, v67, v5, v[1:2]
	v_mov_b32_e32 v70, 0
	ds_load_b32 v67, v71 offset:384
	v_lshlrev_b64_e32 v[69:70], 2, v[69:70]
	v_add_co_u32 v69, vcc_lo, v4, v69
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v70, null, v3, v70, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[69:70], v67, off
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v69, 4, v6
	v_add_nc_u32_e32 v67, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v69
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_and_b32_e32 v70, 31, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v71, v70, 2, v2
	v_mad_co_u64_u32 v[69:70], null, v69, v5, v[1:2]
	v_mov_b32_e32 v70, 0
	ds_load_b32 v71, v71 offset:512
	v_lshlrev_b64_e32 v[69:70], 2, v[69:70]
	v_add_co_u32 v69, vcc_lo, v4, v69
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v70, null, v3, v70, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[69:70], v71, off
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v70, 5, v6
	v_add_nc_u32_e32 v69, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v70
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_and_b32_e32 v71, 31, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v72, v71, 2, v2
	v_mad_co_u64_u32 v[70:71], null, v70, v5, v[1:2]
	v_mov_b32_e32 v71, 0
	ds_load_b32 v72, v72 offset:640
	v_lshlrev_b64_e32 v[70:71], 2, v[70:71]
	v_add_co_u32 v70, vcc_lo, v4, v70
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v71, null, v3, v71, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[70:71], v72, off
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v71, 6, v6
	v_add_nc_u32_e32 v70, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v71
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_and_b32_e32 v72, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v73, v72, 2, v2
	v_mad_co_u64_u32 v[71:72], null, v71, v5, v[1:2]
	v_mov_b32_e32 v72, 0
	ds_load_b32 v73, v73 offset:768
	v_lshlrev_b64_e32 v[71:72], 2, v[71:72]
	v_add_co_u32 v71, vcc_lo, v4, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v72, null, v3, v72, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[71:72], v73, off
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v72, 7, v6
	v_add_nc_u32_e32 v71, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v72
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_and_b32_e32 v73, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v74, v73, 2, v2
	v_mad_co_u64_u32 v[72:73], null, v72, v5, v[1:2]
	v_mov_b32_e32 v73, 0
	ds_load_b32 v74, v74 offset:896
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	v_add_co_u32 v72, vcc_lo, v4, v72
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v73, null, v3, v73, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[72:73], v74, off
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v73, 8, v6
	v_add_nc_u32_e32 v72, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v73
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_and_b32_e32 v74, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v75, v74, 2, v2
	v_mad_co_u64_u32 v[73:74], null, v73, v5, v[1:2]
	v_mov_b32_e32 v74, 0
	ds_load_b32 v75, v75 offset:1024
	v_lshlrev_b64_e32 v[73:74], 2, v[73:74]
	v_add_co_u32 v73, vcc_lo, v4, v73
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v74, null, v3, v74, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[73:74], v75, off
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v74, 9, v6
	v_add_nc_u32_e32 v73, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v74
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_and_b32_e32 v75, 31, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v76, v75, 2, v2
	v_mad_co_u64_u32 v[74:75], null, v74, v5, v[1:2]
	v_mov_b32_e32 v75, 0
	ds_load_b32 v76, v76 offset:1152
	v_lshlrev_b64_e32 v[74:75], 2, v[74:75]
	v_add_co_u32 v74, vcc_lo, v4, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v75, null, v3, v75, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[74:75], v76, off
.LBB0_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v75, 10, v6
	v_add_nc_u32_e32 v74, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v75
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_and_b32_e32 v76, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v77, v76, 2, v2
	v_mad_co_u64_u32 v[75:76], null, v75, v5, v[1:2]
	v_mov_b32_e32 v76, 0
	ds_load_b32 v77, v77 offset:1280
	v_lshlrev_b64_e32 v[75:76], 2, v[75:76]
	v_add_co_u32 v75, vcc_lo, v4, v75
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v76, null, v3, v76, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[75:76], v77, off
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v76, 11, v6
	v_add_nc_u32_e32 v75, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v76
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_and_b32_e32 v77, 31, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v78, v77, 2, v2
	v_mad_co_u64_u32 v[76:77], null, v76, v5, v[1:2]
	v_mov_b32_e32 v77, 0
	ds_load_b32 v78, v78 offset:1408
	v_lshlrev_b64_e32 v[76:77], 2, v[76:77]
	v_add_co_u32 v76, vcc_lo, v4, v76
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, v3, v77, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[76:77], v78, off
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v77, 12, v6
	v_add_nc_u32_e32 v76, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v77
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_and_b32_e32 v78, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v79, v78, 2, v2
	v_mad_co_u64_u32 v[77:78], null, v77, v5, v[1:2]
	v_mov_b32_e32 v78, 0
	ds_load_b32 v79, v79 offset:1536
	v_lshlrev_b64_e32 v[77:78], 2, v[77:78]
	v_add_co_u32 v77, vcc_lo, v4, v77
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v78, null, v3, v78, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[77:78], v79, off
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v78, 13, v6
	v_add_nc_u32_e32 v77, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v78
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_and_b32_e32 v79, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v80, v79, 2, v2
	v_mad_co_u64_u32 v[78:79], null, v78, v5, v[1:2]
	v_mov_b32_e32 v79, 0
	ds_load_b32 v80, v80 offset:1664
	v_lshlrev_b64_e32 v[78:79], 2, v[78:79]
	v_add_co_u32 v78, vcc_lo, v4, v78
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v79, null, v3, v79, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[78:79], v80, off
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v79, 14, v6
	v_add_nc_u32_e32 v78, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v79
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_and_b32_e32 v80, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v81, v80, 2, v2
	v_mad_co_u64_u32 v[79:80], null, v79, v5, v[1:2]
	v_mov_b32_e32 v80, 0
	ds_load_b32 v81, v81 offset:1792
	v_lshlrev_b64_e32 v[79:80], 2, v[79:80]
	v_add_co_u32 v79, vcc_lo, v4, v79
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v80, null, v3, v80, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[79:80], v81, off
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v79, 15, v6
	v_add_nc_u32_e32 v0, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s19, v79
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_and_b32_e32 v80, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v81, v80, 2, v2
	v_mad_co_u64_u32 v[79:80], null, v79, v5, v[1:2]
	v_mov_b32_e32 v80, 0
	ds_load_b32 v81, v81 offset:1920
	v_lshlrev_b64_e32 v[79:80], 2, v[79:80]
	v_add_co_u32 v79, vcc_lo, v4, v79
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v80, null, v3, v80, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[79:80], v81, off
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v52, v53 offset1:1
	ds_store_2addr_b32 v7, v54, v55 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v56, v57 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v58, v59 offset0:6 offset1:7
	ds_store_b32 v8, v44
	ds_store_b32 v9, v45
	ds_store_b32 v10, v46
	ds_store_b32 v11, v47
	ds_store_b32 v60, v48
	ds_store_b32 v61, v49
	ds_store_b32 v62, v50
	ds_store_b32 v68, v51
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v44, 16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	ds_load_b32 v46, v63
	v_mov_b32_e32 v45, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 17, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_and_b32_e32 v45, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:128
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 18, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_and_b32_e32 v45, 31, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:256
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 19, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_and_b32_e32 v45, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:384
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 20, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_and_b32_e32 v45, 31, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:512
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 21, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_and_b32_e32 v45, 31, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:640
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 22, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_and_b32_e32 v45, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:768
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 23, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_and_b32_e32 v45, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:896
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 24, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_and_b32_e32 v45, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:1024
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 25, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_and_b32_e32 v45, 31, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:1152
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 26, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_and_b32_e32 v45, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:1280
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 27, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_78
; %bb.77:
	v_and_b32_e32 v45, 31, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:1408
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 28, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_and_b32_e32 v45, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:1536
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 29, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_and_b32_e32 v45, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:1664
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 30, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_and_b32_e32 v45, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:1792
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 31, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_and_b32_e32 v45, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v46, v45, 2, v2
	v_mad_co_u64_u32 v[44:45], null, v44, v5, v[1:2]
	v_mov_b32_e32 v45, 0
	ds_load_b32 v46, v46 offset:1920
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v4, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v3, v45, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[44:45], v46, off
.LBB0_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v36, v37 offset1:1
	ds_store_2addr_b32 v7, v38, v39 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v40, v41 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v42, v43 offset0:6 offset1:7
	ds_store_b32 v8, v28
	ds_store_b32 v9, v29
	ds_store_b32 v10, v30
	ds_store_b32 v11, v31
	ds_store_b32 v60, v32
	ds_store_b32 v61, v33
	ds_store_b32 v62, v34
	ds_store_b32 v68, v35
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v28, 32, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	ds_load_b32 v30, v63
	v_mov_b32_e32 v29, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 33, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_and_b32_e32 v29, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:128
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 34, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_92
; %bb.91:
	v_and_b32_e32 v29, 31, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:256
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 35, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_and_b32_e32 v29, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:384
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 36, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_96
; %bb.95:
	v_and_b32_e32 v29, 31, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:512
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 37, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_and_b32_e32 v29, 31, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:640
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 38, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_and_b32_e32 v29, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:768
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 39, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_and_b32_e32 v29, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:896
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 40, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_and_b32_e32 v29, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:1024
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 41, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_and_b32_e32 v29, 31, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:1152
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 42, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_and_b32_e32 v29, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:1280
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 43, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_and_b32_e32 v29, 31, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:1408
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 44, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_and_b32_e32 v29, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:1536
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 45, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_114
; %bb.113:
	v_and_b32_e32 v29, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:1664
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 46, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_116
; %bb.115:
	v_and_b32_e32 v29, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:1792
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 47, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_and_b32_e32 v29, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v30, v29, 2, v2
	v_mad_co_u64_u32 v[28:29], null, v28, v5, v[1:2]
	v_mov_b32_e32 v29, 0
	ds_load_b32 v30, v30 offset:1920
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v4, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v3, v29, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[28:29], v30, off
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v20, v21 offset1:1
	ds_store_2addr_b32 v7, v22, v23 offset0:2 offset1:3
	ds_store_2addr_b32 v7, v24, v25 offset0:4 offset1:5
	ds_store_2addr_b32 v7, v26, v27 offset0:6 offset1:7
	ds_store_b32 v8, v12
	ds_store_b32 v9, v13
	ds_store_b32 v10, v14
	ds_store_b32 v11, v15
	ds_store_b32 v60, v16
	ds_store_b32 v61, v17
	ds_store_b32 v62, v18
	ds_store_b32 v68, v19
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 48, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	ds_load_b32 v9, v63
	v_mov_b32_e32 v8, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 49, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_122
; %bb.121:
	v_and_b32_e32 v8, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:128
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 50, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_124
; %bb.123:
	v_and_b32_e32 v8, 31, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:256
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 51, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_126
; %bb.125:
	v_and_b32_e32 v8, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:384
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 52, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_128
; %bb.127:
	v_and_b32_e32 v8, 31, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:512
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 53, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_130
; %bb.129:
	v_and_b32_e32 v8, 31, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:640
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 54, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_132
; %bb.131:
	v_and_b32_e32 v8, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:768
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 55, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_134
; %bb.133:
	v_and_b32_e32 v8, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:896
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 56, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_136
; %bb.135:
	v_and_b32_e32 v8, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:1024
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 57, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_138
; %bb.137:
	v_and_b32_e32 v8, 31, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:1152
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 58, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_140
; %bb.139:
	v_and_b32_e32 v8, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:1280
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 59, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_and_b32_e32 v8, 31, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:1408
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 60, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_and_b32_e32 v8, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:1536
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 61, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_146
; %bb.145:
	v_and_b32_e32 v8, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:1664
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 62, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v7
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_148
; %bb.147:
	v_and_b32_e32 v8, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v8, 2, v2
	v_mad_co_u64_u32 v[7:8], null, v7, v5, v[1:2]
	v_mov_b32_e32 v8, 0
	ds_load_b32 v9, v9 offset:1792
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v4, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v3, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v9, off
.LBB0_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v6, 63, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s19, v6
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_150
; %bb.149:
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v2, v0, 2, v2
	v_mad_co_u64_u32 v[0:1], null, v6, v5, v[1:2]
	ds_load_b32 v2, v2 offset:1920
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, v4, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v2, off
.LBB0_150:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end0:
	.size	gu_noscale, .Lfunc_end0-gu_noscale
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gu_noscale
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
		.amdhsa_next_free_vgpr 189
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gu_noscale)<<4)&4080)>>4
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
	.set .Lgu_noscale.num_vgpr, 189
	.set .Lgu_noscale.num_agpr, 0
	.set .Lgu_noscale.numbered_sgpr, 32
	.set .Lgu_noscale.num_named_barrier, 0
	.set .Lgu_noscale.private_seg_size, 0
	.set .Lgu_noscale.uses_vcc, 1
	.set .Lgu_noscale.uses_flat_scratch, 0
	.set .Lgu_noscale.has_dyn_sized_stack, 0
	.set .Lgu_noscale.has_recursion, 0
	.set .Lgu_noscale.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13844
; TotalNumSgprs: 34
; NumVgprs: 189
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 189
; Occupancy: 8
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
	.type	__hip_cuid_7110c422939a33fc,@object ; @__hip_cuid_7110c422939a33fc
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7110c422939a33fc
__hip_cuid_7110c422939a33fc:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7110c422939a33fc, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7110c422939a33fc
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
    .name:           gu_noscale
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         gu_noscale.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     189
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
