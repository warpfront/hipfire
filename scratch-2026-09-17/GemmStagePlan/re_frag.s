	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	re_frag                 ; -- Begin function re_frag
	.globl	re_frag
	.p2align	8
	.type	re_frag,@function
re_frag:                                ; @re_frag
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
	v_lshlrev_b32_e32 v164, 3, v2
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
	v_and_b32_e32 v163, 15, v0
	v_and_b32_e32 v161, 64, v1
	v_and_b32_e32 v162, 0x60, v0
	v_and_b32_e32 v160, 31, v0
	s_lshl_b32 s19, ttmp7, 7
	s_lshl_b32 s22, ttmp9, 7
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s21, 0
	s_cbranch_vccnz .LBB0_20
; %bb.4:
	v_lshrrev_b32_e32 v2, 2, v161
	v_lshrrev_b32_e32 v3, 2, v0
	v_lshrrev_b32_e32 v4, 2, v162
	s_ashr_i32 s0, s17, 31
	s_add_co_i32 s4, s16, -1
	v_mad_u32_u24 v20, 0x120, v2, 0
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v5, s19, v3
	v_add_nc_u32_e32 v2, s22, v3
	v_and_b32_e32 v9, 0x7f, v0
	s_lshr_b32 s0, s0, 24
	v_mad_u32_u24 v21, 0x120, v4, 0
	s_add_co_i32 s0, s17, s0
	s_add_co_i32 s20, s18, -1
	v_or_b32_e32 v10, s22, v9
	v_dual_mov_b32 v17, v8 :: v_dual_lshlrev_b32 v26, 3, v9
	v_dual_mov_b32 v9, v8 :: v_dual_add_nc_u32 v4, 64, v2
	v_min_i32_e32 v6, s4, v2
	s_ashr_i32 s23, s0, 8
	v_cmp_gt_i32_e64 s2, s16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s5, s23, 0x88
	v_min_i32_e32 v7, s4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v6, s5, v6
	v_min_i32_e32 v12, s4, v10
	v_add_nc_u32_e32 v3, 64, v5
	v_cmp_gt_i32_e64 s0, s18, v5
	v_mul_lo_u32 v2, s5, v7
	v_add_nc_u32_e32 v7, s19, v1
	v_min_i32_e32 v5, s20, v5
	v_lshrrev_b32_e32 v15, 7, v0
	v_add_co_u32 v6, s3, s8, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, s9, 0, s3
	v_cmp_gt_i32_e64 s3, s16, v4
	v_min_i32_e32 v4, s20, v7
	v_add_co_u32 v13, s4, s8, v2
	v_mul_lo_u32 v2, s5, v12
	v_lshrrev_b32_e32 v12, 4, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s9, 0, s4
	v_mul_lo_u32 v165, v4, s23
	v_cmp_gt_i32_e64 s4, s18, v7
	v_cmp_gt_i32_e64 s5, s16, v10
	v_and_b32_e32 v4, 3, v0
	v_or_b32_e32 v7, 16, v12
	v_lshlrev_b32_e32 v10, 1, v0
	v_add_co_u32 v166, s8, s8, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_or_b32 v12, v12, 12, v4
	v_and_or_b32 v7, v7, 28, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v10, 0x78, v10
	v_lshlrev_b32_e32 v2, 4, v4
	v_lshlrev_b32_e32 v25, 2, v1
	v_or_b32_e32 v1, v161, v163
	v_lshl_add_u32 v27, v15, 2, 0
	v_mad_u32_u24 v23, 0x120, v7, v10
	v_mov_b32_e32 v7, v8
	v_mad_u32_u24 v24, 0x120, v12, v10
	v_mov_b32_e32 v10, v8
	v_cmp_gt_i32_e64 s1, s18, v3
	v_min_i32_e32 v3, s20, v3
	v_dual_mov_b32 v19, v8 :: v_dual_lshlrev_b32 v168, 4, v15
	v_dual_mov_b32 v15, v8 :: v_dual_lshlrev_b32 v22, 3, v160
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[152:153], null, v3, s17, v[2:3]
	v_mad_co_u64_u32 v[153:154], null, v5, s17, v[2:3]
	v_dual_mov_b32 v3, v8 :: v_dual_lshlrev_b32 v4, 3, v4
	v_or_b32_e32 v2, v164, v162
	v_mov_b32_e32 v12, v8
	v_mov_b32_e32 v16, v8
	v_dual_mov_b32 v18, v8 :: v_dual_add_nc_u32 v177, 0, v24
	v_add_co_u32 v169, vcc_lo, v6, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v170, null, 0, v11, vcc_lo
	v_add_co_u32 v171, vcc_lo, v13, v4
	v_lshlrev_b32_e32 v29, 3, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v172, null, 0, v14, vcc_lo
	v_dual_mov_b32 v6, v8 :: v_dual_mov_b32 v13, v8
	v_dual_mov_b32 v14, v8 :: v_dual_add_nc_u32 v173, 0, v25
	v_dual_mov_b32 v1, v8 :: v_dual_lshlrev_b32 v28, 2, v1
	v_add_nc_u32_e32 v174, v27, v26
	v_add_nc_u32_e32 v178, 0, v23
	v_add_nc_u32_e32 v179, v21, v22
	v_dual_mov_b32 v27, v19 :: v_dual_add_nc_u32 v180, v20, v22
	v_add_nc_u32_e32 v175, 0, v28
	v_dual_mov_b32 v23, v15 :: v_dual_add_nc_u32 v176, 0, v29
	v_mov_b32_e32 v35, v19
	v_dual_mov_b32 v11, v8 :: v_dual_mov_b32 v34, v18
	v_dual_mov_b32 v43, v19 :: v_dual_mov_b32 v2, v8
	v_dual_mov_b32 v4, v8 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v42, v18 :: v_dual_mov_b32 v51, v19
	v_dual_mov_b32 v50, v18 :: v_dual_mov_b32 v59, v19
	v_dual_mov_b32 v54, v14 :: v_dual_mov_b32 v67, v19
	v_dual_mov_b32 v66, v18 :: v_dual_mov_b32 v75, v19
	v_dual_mov_b32 v58, v18 :: v_dual_mov_b32 v151, v11
	v_dual_mov_b32 v74, v18 :: v_dual_mov_b32 v83, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v167, null, s9, 0, s8
	v_dual_mov_b32 v26, v18 :: v_dual_mov_b32 v25, v17
	v_dual_mov_b32 v24, v16 :: v_dual_mov_b32 v33, v17
	v_dual_mov_b32 v22, v14 :: v_dual_mov_b32 v31, v15
	v_dual_mov_b32 v21, v13 :: v_dual_mov_b32 v20, v12
	v_dual_mov_b32 v29, v13 :: v_dual_mov_b32 v32, v16
	v_dual_mov_b32 v41, v17 :: v_dual_mov_b32 v30, v14
	v_dual_mov_b32 v39, v15 :: v_dual_mov_b32 v28, v12
	v_dual_mov_b32 v37, v13 :: v_dual_mov_b32 v40, v16
	v_dual_mov_b32 v49, v17 :: v_dual_mov_b32 v38, v14
	v_dual_mov_b32 v47, v15 :: v_dual_mov_b32 v36, v12
	v_dual_mov_b32 v45, v13 :: v_dual_mov_b32 v48, v16
	v_dual_mov_b32 v57, v17 :: v_dual_mov_b32 v46, v14
	v_dual_mov_b32 v55, v15 :: v_dual_mov_b32 v44, v12
	v_dual_mov_b32 v53, v13 :: v_dual_mov_b32 v56, v16
	v_dual_mov_b32 v149, v9 :: v_dual_mov_b32 v52, v12
	v_dual_mov_b32 v65, v17 :: v_dual_mov_b32 v150, v10
	v_dual_mov_b32 v63, v15 :: v_dual_mov_b32 v148, v8
	v_dual_mov_b32 v61, v13 :: v_dual_mov_b32 v64, v16
	v_dual_mov_b32 v73, v17 :: v_dual_mov_b32 v62, v14
	v_dual_mov_b32 v71, v15 :: v_dual_mov_b32 v60, v12
	v_dual_mov_b32 v69, v13 :: v_dual_mov_b32 v72, v16
	v_dual_mov_b32 v81, v6 :: v_dual_mov_b32 v70, v14
	v_dual_mov_b32 v79, v4 :: v_dual_mov_b32 v68, v12
	v_dual_mov_b32 v77, v2 :: v_dual_mov_b32 v82, v7
	v_mov_b32_e32 v80, v5
	v_mov_b32_e32 v78, v3
	v_mov_b32_e32 v76, v1
	s_mov_b32 s17, 0x4e4c4a48
	s_mov_b32 s24, 0x4040404
	s_mov_b32 s25, 0
                                        ; implicit-def: $vgpr84_vgpr85_vgpr86_vgpr87_vgpr88_vgpr89_vgpr90_vgpr91
                                        ; implicit-def: $vgpr92_vgpr93_vgpr94_vgpr95_vgpr96_vgpr97_vgpr98_vgpr99
                                        ; implicit-def: $vgpr100_vgpr101_vgpr102_vgpr103_vgpr104_vgpr105_vgpr106_vgpr107
                                        ; implicit-def: $vgpr108_vgpr109_vgpr110_vgpr111_vgpr112_vgpr113_vgpr114_vgpr115
                                        ; implicit-def: $vgpr116_vgpr117_vgpr118_vgpr119_vgpr120_vgpr121_vgpr122_vgpr123
                                        ; implicit-def: $vgpr124_vgpr125_vgpr126_vgpr127_vgpr128_vgpr129_vgpr130_vgpr131
                                        ; implicit-def: $vgpr132_vgpr133_vgpr134_vgpr135_vgpr136_vgpr137_vgpr138_vgpr139
                                        ; implicit-def: $vgpr140_vgpr141_vgpr142_vgpr143_vgpr144_vgpr145_vgpr146_vgpr147
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_add_co_i32 s25, s25, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s25, s23
	s_cbranch_scc1 .LBB0_21
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
                                        ;       Child Loop BB0_10 Depth 3
	v_add_lshl_u32 v11, s25, v165, 1
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
	v_or_b32_e32 v7, s29, v11
	s_xor_b32 s27, s20, -1
	s_lshl_b32 s20, s29, 2
	v_add_nc_u32_e32 v181, 0x2000, v178
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s20, s20, s26
	v_lshlrev_b64_e32 v[1:2], 2, v[7:8]
	s_mov_b32 s28, -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s13, v2, vcc_lo
	global_load_b32 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v1, 0, v1, s4
	ds_store_b32 v173, v1 offset:18432
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, v166, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v167, vcc_lo
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
	v_add_co_u32 v154, vcc_lo, v169, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, 0, v170, vcc_lo
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v168, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s5
	ds_store_b32 v174, v1 offset:18944
	global_load_b128 v[1:4], v[5:6], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	ds_store_2addr_b64 v177, v[1:2], v[3:4] offset1:16
	global_load_b128 v[1:4], v[9:10], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v2, 0, v2, s1
	v_cndmask_b32_e64 v1, 0, v1, s1
	v_cndmask_b32_e64 v4, 0, v4, s1
	v_cndmask_b32_e64 v3, 0, v3, s1
	ds_store_2addr_b64 v178, v[1:2], v[3:4] offset1:16
	global_load_b64 v[1:2], v[154:155], off offset:8
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v1, 0, v1, s2
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
	v_perm_b32 v7, s17, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v4, v4, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v3, v7, v4
	v_and_b32_e32 v4, 0x7070707, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_perm_b32 v7, s17, 0x44403800, v4
	v_or_b32_e32 v4, 0x50505050, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, v1, s24, 0x3020100
	v_perm_b32 v4, v4, v7, v1
	v_cndmask_b32_e64 v1, 0, v2, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v7, v1, v2, 0x5010400
	v_perm_b32 v2, v1, v2, 0x7030602
	v_and_b32_e32 v1, 0x7070707, v7
	v_lshrrev_b32_e32 v7, 1, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v156, s17, 0x44403800, v1
	v_or_b32_e32 v1, 0x50505050, v1
	v_and_or_b32 v7, v7, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v1, v1, v156, v7
	v_and_b32_e32 v7, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	v_perm_b32 v156, s17, 0x44403800, v7
	v_or_b32_e32 v7, 0x50505050, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, v2, s24, 0x3020100
	v_perm_b32 v2, v7, v156, v2
	v_add_co_u32 v156, vcc_lo, v171, s20
	v_add_nc_u32_e32 v7, 0x2000, v177
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v157, null, 0, v172, vcc_lo
	ds_store_2addr_b64 v7, v[3:4], v[1:2] offset0:128 offset1:144
	global_load_b64 v[1:2], v[156:157], off offset:8
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
	v_perm_b32 v158, s17, 0x44403800, v3
	v_or_b32_e32 v3, 0x50505050, v3
	v_and_or_b32 v4, v4, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v3, v158, v4
	v_and_b32_e32 v4, 0x7070707, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_perm_b32 v158, s17, 0x44403800, v4
	v_or_b32_e32 v4, 0x50505050, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v1, v1, s24, 0x3020100
	v_perm_b32 v4, v4, v158, v1
	v_cndmask_b32_e64 v1, 0, v2, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v158, v1, v2, 0x5010400
	v_perm_b32 v2, v1, v2, 0x7030602
	v_and_b32_e32 v1, 0x7070707, v158
	v_lshrrev_b32_e32 v158, 1, v158
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v159, s17, 0x44403800, v1
	v_or_b32_e32 v1, 0x50505050, v1
	v_and_or_b32 v158, v158, s24, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v1, v1, v159, v158
	v_and_b32_e32 v158, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	v_perm_b32 v159, s17, 0x44403800, v158
	v_or_b32_e32 v158, 0x50505050, v158
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, v2, s24, 0x3020100
	v_perm_b32 v2, v158, v159, v2
	ds_store_2addr_b64 v181, v[3:4], v[1:2] offset0:128 offset1:144
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
	s_clause 0x1
	global_load_b64 v[148:149], v[154:155], off offset:40
	global_load_b64 v[150:151], v[156:157], off offset:40
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v77, 0, v2, s0
	v_cndmask_b32_e64 v76, 0, v1, s0
	v_cndmask_b32_e64 v79, 0, v4, s0
	v_cndmask_b32_e64 v78, 0, v3, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v81, 0, v81, s1
	v_cndmask_b32_e64 v80, 0, v80, s1
	v_cndmask_b32_e64 v83, 0, v83, s1
	v_cndmask_b32_e64 v82, 0, v82, s1
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v148, 0, v148, s2
	v_cndmask_b32_e64 v149, 0, v149, s2
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v150, 0, v150, s3
	v_cndmask_b32_e64 v151, 0, v151, s3
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=3
	v_add_nc_u32_e32 v183, 0x2400, v179
	v_add_nc_u32_e32 v185, 0x400, v180
	v_add_nc_u32_e32 v184, 0x800, v180
	v_add_nc_u32_e32 v182, 0xc00, v180
	s_and_not1_b32 vcc_lo, exec_lo, s20
	ds_load_2addr_b64 v[1:4], v183 offset1:144
	ds_load_b64 v[158:159], v180
	s_mov_b32 s30, -1
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
	ds_load_2addr_b64 v[186:189], v185 offset0:16 offset1:160
	ds_load_b64 v[190:191], v180 offset:3456
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[158:159], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[158:159], v[132:139]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[186:187], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[186:187], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[188:189], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[188:189], v[100:107]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[190:191], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[190:191], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[186:189], v183 offset0:36 offset1:180
	ds_load_2addr_b64 v[190:193], v180 offset0:36 offset1:180
	ds_load_2addr_b64 v[194:197], v184 offset0:68 offset1:212
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[186:187], v[190:191], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[188:189], v[190:191], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[186:187], v[192:193], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[188:189], v[192:193], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[186:187], v[194:195], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[188:189], v[194:195], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[186:187], v[196:197], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[188:189], v[196:197], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[186:189], v183 offset0:72 offset1:216
	ds_load_2addr_b64 v[190:193], v180 offset0:72 offset1:216
	ds_load_2addr_b64 v[194:197], v184 offset0:104 offset1:248
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[186:187], v[190:191], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[188:189], v[190:191], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[186:187], v[192:193], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[188:189], v[192:193], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[186:187], v[194:195], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[188:189], v[194:195], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[186:187], v[196:197], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[188:189], v[196:197], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[186:189], v183 offset0:108 offset1:252
	ds_load_2addr_b64 v[190:193], v180 offset0:108 offset1:252
	ds_load_2addr_b64 v[194:197], v182 offset0:12 offset1:156
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[186:187], v[190:191], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[188:189], v[190:191], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[186:187], v[192:193], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[188:189], v[192:193], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[186:187], v[194:195], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[188:189], v[194:195], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[186:187], v[196:197], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[188:189], v[196:197], v[84:91]
	; sched_barrier mask(0x00000000)
	s_cbranch_execnz .LBB0_14
.LBB0_17:                               ;   in Loop: Header=BB0_10 Depth=3
	ds_load_2addr_b64 v[84:87], v185 offset0:16 offset1:160
	ds_load_b64 v[185:186], v180 offset:3456
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[158:159], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[158:159], 0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[84:85], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[84:85], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[86:87], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[86:87], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[185:186], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[185:186], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v183 offset0:36 offset1:180
	ds_load_2addr_b64 v[185:188], v180 offset0:36 offset1:180
	ds_load_2addr_b64 v[189:192], v184 offset0:68 offset1:212
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[185:186], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[185:186], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[187:188], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[187:188], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[189:190], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[189:190], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[191:192], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[191:192], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v183 offset0:72 offset1:216
	ds_load_2addr_b64 v[185:188], v180 offset0:72 offset1:216
	ds_load_2addr_b64 v[189:192], v184 offset0:104 offset1:248
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[185:186], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[185:186], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[187:188], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[187:188], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[189:190], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[189:190], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[191:192], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[191:192], v[84:91]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[1:4], v183 offset0:108 offset1:252
	ds_load_2addr_b64 v[183:186], v180 offset0:108 offset1:252
	ds_load_2addr_b64 v[187:190], v182 offset0:12 offset1:156
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[183:184], v[140:147]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[3:4], v[183:184], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[185:186], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[3:4], v[185:186], v[116:123]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[187:188], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[3:4], v[187:188], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[189:190], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[3:4], v[189:190], v[84:91]
	; sched_barrier mask(0x00000000)
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_15
.LBB0_18:                               ;   in Loop: Header=BB0_10 Depth=3
	s_wait_dscnt 0x1
	v_lshrrev_b32_e32 v1, 4, v148
	v_lshrrev_b32_e32 v2, 4, v149
	v_and_b32_e32 v3, 0xf0f0f0f, v148
	v_and_b32_e32 v4, 0xf0f0f0f, v149
	s_wait_loadcnt_dscnt 0x0
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	v_and_b32_e32 v2, 0xf0f0f0f, v2
	s_barrier_signal -1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v158, v1, v3, 0x5010400
	v_perm_b32 v1, v1, v3, 0x7030602
	v_perm_b32 v3, v2, v4, 0x5010400
	v_perm_b32 v4, v2, v4, 0x7030602
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v2, 0x7070707, v158
	v_lshrrev_b32_e32 v158, 1, v158
	v_and_b32_e32 v159, 0x7070707, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_and_b32_e32 v182, 0x7070707, v3
	v_perm_b32 v183, s17, 0x44403800, v2
	v_or_b32_e32 v2, 0x50505050, v2
	v_and_or_b32 v158, v158, s24, 0x3020100
	v_lshrrev_b32_e32 v3, 1, v3
	v_perm_b32 v184, s17, 0x44403800, v159
	v_or_b32_e32 v159, 0x50505050, v159
	v_and_or_b32 v185, v1, s24, 0x3020100
	v_perm_b32 v1, v2, v183, v158
	v_lshrrev_b32_e32 v2, 4, v150
	v_perm_b32 v158, s17, 0x44403800, v182
	v_or_b32_e32 v182, 0x50505050, v182
	v_and_or_b32 v3, v3, s24, 0x3020100
	v_and_b32_e32 v183, 0xf0f0f0f, v150
	v_and_b32_e32 v186, 0xf0f0f0f, v2
	v_perm_b32 v2, v159, v184, v185
	v_lshrrev_b32_e32 v184, 4, v151
	v_perm_b32 v3, v182, v158, v3
	v_and_b32_e32 v187, 0x7070707, v4
	v_perm_b32 v158, v186, v183, 0x5010400
	v_perm_b32 v183, v186, v183, 0x7030602
	v_and_b32_e32 v186, 0xf0f0f0f, v151
	v_and_b32_e32 v184, 0xf0f0f0f, v184
	v_lshrrev_b32_e32 v4, 1, v4
	v_and_b32_e32 v185, 0x7070707, v158
	v_lshrrev_b32_e32 v158, 1, v158
	v_and_b32_e32 v188, 0x7070707, v183
	v_perm_b32 v189, v184, v186, 0x5010400
	v_perm_b32 v184, v184, v186, 0x7030602
	v_lshrrev_b32_e32 v183, 1, v183
	v_perm_b32 v159, s17, 0x44403800, v187
	v_or_b32_e32 v182, 0x50505050, v187
	v_and_b32_e32 v186, 0x7070707, v189
	v_lshrrev_b32_e32 v189, 1, v189
	v_and_b32_e32 v191, 0x7070707, v184
	v_lshrrev_b32_e32 v184, 1, v184
	v_and_or_b32 v4, v4, s24, 0x3020100
	v_perm_b32 v187, s17, 0x44403800, v185
	v_or_b32_e32 v185, 0x50505050, v185
	v_and_or_b32 v158, v158, s24, 0x3020100
	v_perm_b32 v190, s17, 0x44403800, v188
	v_or_b32_e32 v188, 0x50505050, v188
	v_and_or_b32 v183, v183, s24, 0x3020100
	v_perm_b32 v192, s17, 0x44403800, v186
	v_or_b32_e32 v186, 0x50505050, v186
	v_and_or_b32 v189, v189, s24, 0x3020100
	v_perm_b32 v193, s17, 0x44403800, v191
	v_or_b32_e32 v191, 0x50505050, v191
	v_and_or_b32 v184, v184, s24, 0x3020100
	v_perm_b32 v4, v182, v159, v4
	v_perm_b32 v158, v185, v187, v158
	v_perm_b32 v159, v188, v190, v183
	v_perm_b32 v182, v186, v192, v189
	v_perm_b32 v183, v191, v193, v184
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v177, v[76:77], v[78:79] offset1:16
	ds_store_2addr_b64 v178, v[80:81], v[82:83] offset1:16
	ds_store_2addr_b64 v7, v[1:2], v[3:4] offset0:128 offset1:144
	ds_store_2addr_b64 v181, v[158:159], v[182:183] offset0:128 offset1:144
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
.LBB0_19:                               ;   in Loop: Header=BB0_10 Depth=3
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v158, 0x4a10, v176
	v_add_nc_u32_e32 v184, 0x4800, v175
	ds_load_2addr_b32 v[158:159], v158 offset1:1
	ds_load_2addr_b32 v[182:183], v184 offset1:16
	ds_load_2addr_b32 v[184:185], v184 offset0:32 offset1:48
	v_add_nc_u32_e32 v1, 0x4a00, v176
	v_add_nc_u32_e32 v188, 0x4a20, v176
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_add_nc_u32_e32 v3, 0x4a08, v176
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	s_wait_dscnt 0x4
	v_fma_f32 v70, v158, v142, v70
	v_fma_f32 v54, v158, v126, v54
	v_fma_f32 v38, v158, v110, v38
	v_fma_f32 v22, v158, v94, v22
	s_wait_dscnt 0x3
	v_fmac_f32_e32 v70, v159, v182
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v22, v159, v185
	s_wait_dscnt 0x1
	v_fma_f32 v20, v1, v92, v20
	v_fma_f32 v68, v1, v140, v68
	v_fma_f32 v52, v1, v124, v52
	v_fma_f32 v36, v1, v108, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v20, v2, v185 :: v_dual_add_nc_u32 v1, 0x4a28, v176
	s_wait_dscnt 0x0
	v_fma_f32 v53, v3, v125, v53
	v_add_nc_u32_e32 v186, 0x4a18, v176
	v_fma_f32 v69, v3, v141, v69
	v_fma_f32 v37, v3, v109, v37
	v_fma_f32 v21, v3, v93, v21
	v_fmac_f32_e32 v53, v4, v183
	ds_load_2addr_b32 v[186:187], v186 offset1:1
	ds_load_2addr_b32 v[188:189], v188 offset1:1
	v_dual_fmac_f32 v68, v2, v182 :: v_dual_add_nc_u32 v3, 0x4a30, v176
	v_fmac_f32_e32 v52, v2, v183
	v_dual_fmac_f32 v36, v2, v184 :: v_dual_fmac_f32 v37, v4, v184
	v_fmac_f32_e32 v69, v4, v182
	v_fmac_f32_e32 v21, v4, v185
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_fmac_f32_e32 v54, v159, v183
	v_fmac_f32_e32 v38, v159, v184
	s_wait_dscnt 0x2
	v_fma_f32 v55, v186, v127, v55
	v_add_nc_u32_e32 v158, 0x4a38, v176
	s_wait_dscnt 0x1
	v_fma_f32 v40, v188, v112, v40
	v_fma_f32 v24, v188, v96, v24
	v_fma_f32 v71, v186, v143, v71
	v_fmac_f32_e32 v55, v187, v183
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	ds_load_2addr_b32 v[158:159], v158 offset1:1
	v_fma_f32 v39, v186, v111, v39
	v_fma_f32 v23, v186, v95, v23
	v_fma_f32 v72, v188, v144, v72
	v_fma_f32 v56, v188, v128, v56
	v_fmac_f32_e32 v71, v187, v182
	v_fmac_f32_e32 v39, v187, v184
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v23, v187, v185 :: v_dual_fmac_f32 v72, v189, v182
	v_fmac_f32_e32 v56, v189, v183
	v_add_nc_u32_e32 v186, 0x4a90, v176
	s_wait_dscnt 0x1
	v_fma_f32 v74, v3, v146, v74
	v_fma_f32 v58, v3, v130, v58
	v_fma_f32 v42, v3, v114, v42
	v_fma_f32 v26, v3, v98, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v74, v4, v182 :: v_dual_add_nc_u32 v3, 0x4a88, v176
	s_wait_dscnt 0x0
	v_dual_fmac_f32 v75, v158, v147 :: v_dual_fmac_f32 v42, v4, v184
	v_fmac_f32_e32 v58, v4, v183
	v_dual_fmac_f32 v26, v4, v185 :: v_dual_fmac_f32 v59, v158, v131
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	v_fma_f32 v73, v1, v145, v73
	v_fma_f32 v57, v1, v129, v57
	v_fma_f32 v41, v1, v113, v41
	v_fma_f32 v25, v1, v97, v25
	v_add_nc_u32_e32 v1, 0x4a80, v176
	v_dual_fmac_f32 v43, v158, v115 :: v_dual_add_nc_u32 v188, 0x4a98, v176
	v_dual_fmac_f32 v59, v159, v183 :: v_dual_fmac_f32 v40, v189, v184
	v_dual_fmac_f32 v41, v2, v184 :: v_dual_fmac_f32 v24, v189, v185
	v_fmac_f32_e32 v25, v2, v185
	v_fmac_f32_e32 v73, v2, v182
	v_fmac_f32_e32 v57, v2, v183
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	ds_load_2addr_b32 v[186:187], v186 offset1:1
	ds_load_2addr_b32 v[188:189], v188 offset1:1
	v_fmac_f32_e32 v75, v159, v182
	s_wait_dscnt 0x3
	v_fma_f32 v61, v3, v133, v61
	v_fma_f32 v45, v3, v117, v45
	v_fma_f32 v29, v3, v101, v29
	v_fma_f32 v13, v3, v85, v13
	v_add_nc_u32_e32 v3, 0x4aa8, v176
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v13, v4, v185
	s_wait_dscnt 0x1
	v_fma_f32 v46, v186, v118, v46
	v_fmac_f32_e32 v61, v4, v182
	v_fma_f32 v30, v186, v102, v30
	v_fmac_f32_e32 v45, v4, v183
	v_fma_f32 v14, v186, v86, v14
	v_fmac_f32_e32 v29, v4, v184
	ds_load_2addr_b32 v[3:4], v3 offset1:1
	v_fma_f32 v12, v1, v84, v12
	v_fma_f32 v60, v1, v132, v60
	v_fma_f32 v44, v1, v116, v44
	v_fma_f32 v28, v1, v100, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v12, v2, v185 :: v_dual_add_nc_u32 v1, 0x4aa0, v176
	v_fma_f32 v62, v186, v134, v62
	v_dual_fmac_f32 v27, v158, v99 :: v_dual_add_nc_u32 v158, 0x4ab0, v176
	s_wait_dscnt 0x1
	v_fma_f32 v47, v188, v119, v47
	v_add_nc_u32_e32 v186, 0x4ab8, v176
	v_dual_fmac_f32 v43, v159, v184 :: v_dual_fmac_f32 v60, v2, v182
	v_fmac_f32_e32 v44, v2, v183
	v_fmac_f32_e32 v62, v187, v182
	v_fmac_f32_e32 v27, v159, v185
	v_fmac_f32_e32 v46, v187, v183
	v_fmac_f32_e32 v30, v187, v184
	v_fmac_f32_e32 v14, v187, v185
	v_fma_f32 v63, v188, v135, v63
	v_fmac_f32_e32 v28, v2, v184
	ds_load_2addr_b32 v[1:2], v1 offset1:1
	v_fmac_f32_e32 v47, v189, v183
	ds_load_2addr_b32 v[158:159], v158 offset1:1
	ds_load_2addr_b32 v[186:187], v186 offset1:1
	v_fma_f32 v31, v188, v103, v31
	v_fma_f32 v15, v188, v87, v15
	s_wait_dscnt 0x3
	v_fma_f32 v65, v3, v137, v65
	v_fma_f32 v49, v3, v121, v49
	v_fma_f32 v33, v3, v105, v33
	v_fma_f32 v17, v3, v89, v17
	v_fmac_f32_e32 v63, v189, v182
	v_fmac_f32_e32 v65, v4, v182
	s_wait_dscnt 0x2
	v_fma_f32 v64, v1, v136, v64
	v_fma_f32 v48, v1, v120, v48
	v_fma_f32 v32, v1, v104, v32
	v_fma_f32 v16, v1, v88, v16
	s_wait_dscnt 0x1
	v_fma_f32 v66, v158, v138, v66
	v_fma_f32 v50, v158, v122, v50
	v_fma_f32 v34, v158, v106, v34
	v_fma_f32 v18, v158, v90, v18
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v67, v186, v139
	v_fmac_f32_e32 v51, v186, v123
	v_fmac_f32_e32 v35, v186, v107
	v_fmac_f32_e32 v19, v186, v91
	v_dual_fmac_f32 v31, v189, v184 :: v_dual_fmac_f32 v48, v2, v183
	v_dual_fmac_f32 v15, v189, v185 :: v_dual_fmac_f32 v32, v2, v184
	v_fmac_f32_e32 v64, v2, v182
	v_dual_fmac_f32 v16, v2, v185 :: v_dual_fmac_f32 v49, v4, v183
	v_dual_fmac_f32 v66, v159, v182 :: v_dual_fmac_f32 v33, v4, v184
	v_dual_fmac_f32 v50, v159, v183 :: v_dual_fmac_f32 v17, v4, v185
	v_fmac_f32_e32 v34, v159, v184
	v_fmac_f32_e32 v18, v159, v185
	v_fmac_f32_e32 v67, v187, v182
	v_fmac_f32_e32 v51, v187, v183
	v_fmac_f32_e32 v35, v187, v184
	v_fmac_f32_e32 v19, v187, v185
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
	v_mov_b32_e32 v2, v164
.LBB0_22:
	v_lshlrev_b32_e32 v1, 6, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v2, v2, v163
	v_and_b32_e32 v1, 0x3800, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v3, 31, v2
	v_add_nc_u32_e32 v5, 18, v2
	v_add_nc_u32_e32 v7, 19, v2
	v_add_nc_u32_e32 v8, 20, v2
	v_add_nc_u32_e32 v4, 0, v1
	v_add_nc_u32_e32 v1, 17, v2
	v_xor_b32_e32 v3, 16, v3
	v_add_nc_u32_e32 v77, 21, v2
	v_and_b32_e32 v5, 31, v5
	v_lshl_add_u32 v76, v163, 7, v4
	v_and_b32_e32 v1, 31, v1
	v_add_nc_u32_e32 v78, 22, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v6, v2, 2, v76
	v_lshl_add_u32 v10, v1, 2, v76
	v_add_nc_u32_e32 v1, 23, v2
	v_and_b32_e32 v2, 31, v7
	v_lshl_add_u32 v9, v3, 2, v76
	v_and_b32_e32 v3, 31, v8
	v_lshl_add_u32 v11, v5, 2, v76
	v_and_b32_e32 v5, 31, v77
	v_and_b32_e32 v7, 31, v78
	v_and_b32_e32 v1, 31, v1
	ds_store_2addr_b32 v6, v68, v69 offset1:1
	ds_store_2addr_b32 v6, v70, v71 offset0:2 offset1:3
	ds_store_2addr_b32 v6, v72, v73 offset0:4 offset1:5
	ds_store_2addr_b32 v6, v74, v75 offset0:6 offset1:7
	ds_store_b32 v9, v60
	ds_store_b32 v10, v61
	ds_store_b32 v11, v62
	v_lshl_add_u32 v68, v2, 2, v76
	v_lshl_add_u32 v69, v3, 2, v76
	v_lshl_add_u32 v70, v5, 2, v76
	v_lshl_add_u32 v71, v7, 2, v76
	v_lshl_add_u32 v72, v1, 2, v76
	ds_store_b32 v68, v63
	ds_store_b32 v69, v64
	ds_store_b32 v70, v65
	ds_store_b32 v71, v66
	ds_store_b32 v72, v67
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or3_b32 v3, s22, v162, v160
	v_add_nc_u32_e32 v1, s19, v161
	v_lshl_add_u32 v5, v160, 2, v4
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
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_mad_co_u64_u32 v[7:8], null, v1, s16, v[3:4]
	ds_load_b32 v62, v5
	v_mov_b32_e32 v8, 0
	v_lshlrev_b64_e32 v[60:61], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v60, s0, s14, v60
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v61, null, s15, v61, s0
	v_add_co_u32 v7, s0, s6, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s7, v8, s0
	global_load_b32 v60, v[60:61], off
	global_load_b32 v61, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v61, v60, v62
	global_store_b32 v[7:8], v61, off
.LBB0_24:
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
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_mad_co_u64_u32 v[60:61], null, v8, s16, v[3:4]
	v_mov_b32_e32 v61, 0
	v_lshlrev_b64_e32 v[62:63], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v62, vcc_lo, s14, v62
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s15, v63, vcc_lo
	v_add_co_u32 v60, vcc_lo, s6, v60
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v61, null, s7, v61, vcc_lo
	global_load_b32 v8, v[62:63], off offset:4
	global_load_b32 v62, v[60:61], off
	v_and_b32_e32 v63, 31, v7
	v_lshl_add_u32 v63, v63, 2, v4
	ds_load_b32 v63, v63 offset:128
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v62, v8, v63
	global_store_b32 v[60:61], v62, off
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v60, 2, v1
	v_add_nc_u32_e32 v8, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v60
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_mad_co_u64_u32 v[60:61], null, v60, s16, v[3:4]
	v_dual_mov_b32 v61, 0 :: v_dual_and_b32 v64, 31, v8
	v_lshlrev_b64_e32 v[62:63], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v64, v64, 2, v4
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v62, vcc_lo, s14, v62
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, s15, v63, vcc_lo
	ds_load_b32 v64, v64 offset:256
	v_add_co_u32 v60, vcc_lo, s6, v60
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v61, null, s7, v61, vcc_lo
	global_load_b32 v62, v[62:63], off offset:8
	global_load_b32 v63, v[60:61], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v63, v62, v64
	global_store_b32 v[60:61], v63, off
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v61, 3, v1
	v_add_nc_u32_e32 v60, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v61
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_mad_co_u64_u32 v[61:62], null, v61, s16, v[3:4]
	v_dual_mov_b32 v62, 0 :: v_dual_and_b32 v65, 31, v60
	v_lshlrev_b64_e32 v[63:64], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v65, v65, 2, v4
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v63, vcc_lo, s14, v63
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v64, null, s15, v64, vcc_lo
	ds_load_b32 v65, v65 offset:384
	v_add_co_u32 v61, vcc_lo, s6, v61
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v62, null, s7, v62, vcc_lo
	global_load_b32 v63, v[63:64], off offset:12
	global_load_b32 v64, v[61:62], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v64, v63, v65
	global_store_b32 v[61:62], v64, off
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v62, 4, v1
	v_add_nc_u32_e32 v61, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v62
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_mad_co_u64_u32 v[62:63], null, v62, s16, v[3:4]
	v_dual_mov_b32 v63, 0 :: v_dual_and_b32 v66, 31, v61
	v_lshlrev_b64_e32 v[64:65], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v66, v66, 2, v4
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v64, vcc_lo, s14, v64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, s15, v65, vcc_lo
	ds_load_b32 v66, v66 offset:512
	v_add_co_u32 v62, vcc_lo, s6, v62
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, s7, v63, vcc_lo
	global_load_b32 v64, v[64:65], off offset:16
	global_load_b32 v65, v[62:63], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v65, v64, v66
	global_store_b32 v[62:63], v65, off
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v63, 5, v1
	v_add_nc_u32_e32 v62, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v63
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_mad_co_u64_u32 v[63:64], null, v63, s16, v[3:4]
	v_dual_mov_b32 v64, 0 :: v_dual_and_b32 v67, 31, v62
	v_lshlrev_b64_e32 v[65:66], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v67, v67, 2, v4
	v_lshlrev_b64_e32 v[63:64], 2, v[63:64]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v65, vcc_lo, s14, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, s15, v66, vcc_lo
	ds_load_b32 v67, v67 offset:640
	v_add_co_u32 v63, vcc_lo, s6, v63
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v64, null, s7, v64, vcc_lo
	global_load_b32 v65, v[65:66], off offset:20
	global_load_b32 v66, v[63:64], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v66, v65, v67
	global_store_b32 v[63:64], v66, off
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v64, 6, v1
	v_add_nc_u32_e32 v63, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v64
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_mad_co_u64_u32 v[64:65], null, v64, s16, v[3:4]
	v_mov_b32_e32 v65, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[1:2]
	v_and_b32_e32 v73, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v73, v73, 2, v4
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v66, vcc_lo, s14, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, s15, v67, vcc_lo
	ds_load_b32 v73, v73 offset:768
	v_add_co_u32 v64, vcc_lo, s6, v64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, s7, v65, vcc_lo
	global_load_b32 v66, v[66:67], off offset:24
	global_load_b32 v67, v[64:65], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v67, v66, v73
	global_store_b32 v[64:65], v67, off
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v65, 7, v1
	v_add_nc_u32_e32 v64, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v65
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_mad_co_u64_u32 v[65:66], null, v65, s16, v[3:4]
	v_mov_b32_e32 v66, 0
	v_lshlrev_b64_e32 v[73:74], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v73, vcc_lo, s14, v73
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v74, null, s15, v74, vcc_lo
	v_add_co_u32 v65, vcc_lo, s6, v65
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v66, null, s7, v66, vcc_lo
	global_load_b32 v67, v[73:74], off offset:28
	global_load_b32 v73, v[65:66], off
	v_and_b32_e32 v74, 31, v64
	v_lshl_add_u32 v74, v74, 2, v4
	ds_load_b32 v74, v74 offset:896
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v73, v67, v74
	global_store_b32 v[65:66], v73, off
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v66, 8, v1
	v_add_nc_u32_e32 v65, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v66
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_and_b32_e32 v75, 31, v65
	v_mad_co_u64_u32 v[66:67], null, v66, s16, v[3:4]
	v_lshlrev_b64_e32 v[73:74], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_add_u32 v75, v75, 2, v4
	v_add_co_u32 v73, vcc_lo, s14, v73
	ds_load_b32 v75, v75 offset:1024
	v_mov_b32_e32 v67, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, s15, v74, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[66:67], 2, v[66:67]
	v_add_co_u32 v66, vcc_lo, s6, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v67, null, s7, v67, vcc_lo
	global_load_b32 v73, v[73:74], off offset:32
	global_load_b32 v74, v[66:67], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v74, v73, v75
	global_store_b32 v[66:67], v74, off
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v67, 9, v1
	v_add_nc_u32_e32 v66, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v67
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_mad_co_u64_u32 v[73:74], null, v67, s16, v[3:4]
	v_mov_b32_e32 v74, 0
	v_lshlrev_b64_e32 v[75:76], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v75, vcc_lo, s14, v75
	v_lshlrev_b64_e32 v[73:74], 2, v[73:74]
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v76, null, s15, v76, vcc_lo
	v_add_co_u32 v73, vcc_lo, s6, v73
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v74, null, s7, v74, vcc_lo
	global_load_b32 v67, v[75:76], off offset:36
	global_load_b32 v75, v[73:74], off
	v_and_b32_e32 v76, 31, v66
	v_lshl_add_u32 v76, v76, 2, v4
	ds_load_b32 v76, v76 offset:1152
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v75, v67, v76
	global_store_b32 v[73:74], v75, off
.LBB0_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v73, 10, v1
	v_add_nc_u32_e32 v67, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v73
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_mad_co_u64_u32 v[73:74], null, v73, s16, v[3:4]
	v_dual_mov_b32 v74, 0 :: v_dual_and_b32 v77, 31, v67
	v_lshlrev_b64_e32 v[75:76], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v77, v77, 2, v4
	v_lshlrev_b64_e32 v[73:74], 2, v[73:74]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v75, vcc_lo, s14, v75
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, s15, v76, vcc_lo
	ds_load_b32 v77, v77 offset:1280
	v_add_co_u32 v73, vcc_lo, s6, v73
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, s7, v74, vcc_lo
	global_load_b32 v75, v[75:76], off offset:40
	global_load_b32 v76, v[73:74], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v76, v75, v77
	global_store_b32 v[73:74], v76, off
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v74, 11, v1
	v_add_nc_u32_e32 v73, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v74
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_mad_co_u64_u32 v[74:75], null, v74, s16, v[3:4]
	v_dual_mov_b32 v75, 0 :: v_dual_and_b32 v78, 31, v73
	v_lshlrev_b64_e32 v[76:77], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v78, v78, 2, v4
	v_lshlrev_b64_e32 v[74:75], 2, v[74:75]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v76, vcc_lo, s14, v76
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s15, v77, vcc_lo
	ds_load_b32 v78, v78 offset:1408
	v_add_co_u32 v74, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, s7, v75, vcc_lo
	global_load_b32 v76, v[76:77], off offset:44
	global_load_b32 v77, v[74:75], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v77, v76, v78
	global_store_b32 v[74:75], v77, off
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v75, 12, v1
	v_add_nc_u32_e32 v74, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v75
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_mad_co_u64_u32 v[75:76], null, v75, s16, v[3:4]
	v_dual_mov_b32 v76, 0 :: v_dual_and_b32 v79, 31, v74
	v_lshlrev_b64_e32 v[77:78], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v79, v79, 2, v4
	v_lshlrev_b64_e32 v[75:76], 2, v[75:76]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v77, vcc_lo, s14, v77
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, s15, v78, vcc_lo
	ds_load_b32 v79, v79 offset:1536
	v_add_co_u32 v75, vcc_lo, s6, v75
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, s7, v76, vcc_lo
	global_load_b32 v77, v[77:78], off offset:48
	global_load_b32 v78, v[75:76], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v78, v77, v79
	global_store_b32 v[75:76], v78, off
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v76, 13, v1
	v_add_nc_u32_e32 v75, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v76
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_mad_co_u64_u32 v[76:77], null, v76, s16, v[3:4]
	v_dual_mov_b32 v77, 0 :: v_dual_and_b32 v80, 31, v75
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
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v77, 14, v1
	v_add_nc_u32_e32 v76, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v77
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_52
; %bb.51:
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
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v77, 15, v1
	v_add_nc_u32_e32 v0, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s18, v77
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_54
; %bb.53:
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
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v52, v53 offset1:1
	ds_store_2addr_b32 v6, v54, v55 offset0:2 offset1:3
	ds_store_2addr_b32 v6, v56, v57 offset0:4 offset1:5
	ds_store_2addr_b32 v6, v58, v59 offset0:6 offset1:7
	ds_store_b32 v9, v44
	ds_store_b32 v10, v45
	ds_store_b32 v11, v46
	ds_store_b32 v68, v47
	ds_store_b32 v69, v48
	ds_store_b32 v70, v49
	ds_store_b32 v71, v50
	ds_store_b32 v72, v51
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v44, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	ds_load_b32 v48, v5
	v_add_co_u32 v46, vcc_lo, s14, v46
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:64
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 17, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v7
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:128
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:68
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v8
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:256
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:72
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 19, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v60
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:384
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:76
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 20, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v61
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:512
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:80
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 21, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v62
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:640
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:84
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 22, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v63
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:768
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:88
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 23, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v64
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:896
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:92
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 24, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v65
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:1024
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:96
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 25, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v66
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:1152
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:100
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 26, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v67
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:1280
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:104
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 27, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_78
; %bb.77:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v73
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:1408
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:108
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 28, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v74
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:1536
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:112
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 29, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v75
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:1664
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:116
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 30, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v76
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:1792
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:120
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 31, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_mad_co_u64_u32 v[44:45], null, v44, s16, v[3:4]
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v48, 31, v0
	v_lshlrev_b64_e32 v[46:47], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v48, v48, 2, v4
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v46, vcc_lo, s14, v46
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v47, null, s15, v47, vcc_lo
	ds_load_b32 v48, v48 offset:1920
	v_add_co_u32 v44, vcc_lo, s6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, s7, v45, vcc_lo
	global_load_b32 v46, v[46:47], off offset:124
	global_load_b32 v47, v[44:45], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v47, v46, v48
	global_store_b32 v[44:45], v47, off
.LBB0_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v36, v37 offset1:1
	ds_store_2addr_b32 v6, v38, v39 offset0:2 offset1:3
	ds_store_2addr_b32 v6, v40, v41 offset0:4 offset1:5
	ds_store_2addr_b32 v6, v42, v43 offset0:6 offset1:7
	ds_store_b32 v9, v28
	ds_store_b32 v10, v29
	ds_store_b32 v11, v30
	ds_store_b32 v68, v31
	ds_store_b32 v69, v32
	ds_store_b32 v70, v33
	ds_store_b32 v71, v34
	ds_store_b32 v72, v35
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v28, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	ds_load_b32 v32, v5
	v_add_co_u32 v30, vcc_lo, s14, v30
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:128
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 33, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v7
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:128
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:132
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 34, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_92
; %bb.91:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v8
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:256
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:136
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 35, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v60
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:384
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:140
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 36, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_96
; %bb.95:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v61
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:512
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:144
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 37, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v62
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:640
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:148
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 38, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v63
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:768
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:152
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 39, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v64
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:896
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:156
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 40, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v65
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1024
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:160
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 41, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v66
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1152
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:164
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 42, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v67
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1280
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:168
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 43, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v73
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1408
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:172
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 44, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v74
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1536
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:176
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 45, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_114
; %bb.113:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v75
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1664
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:180
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 46, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_116
; %bb.115:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v76
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1792
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:184
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 47, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_mad_co_u64_u32 v[28:29], null, v28, s16, v[3:4]
	v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v32, 31, v0
	v_lshlrev_b64_e32 v[30:31], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v32, v32, 2, v4
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v30, vcc_lo, s14, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, s15, v31, vcc_lo
	ds_load_b32 v32, v32 offset:1920
	v_add_co_u32 v28, vcc_lo, s6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v29, vcc_lo
	global_load_b32 v30, v[30:31], off offset:188
	global_load_b32 v31, v[28:29], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v31, v30, v32
	global_store_b32 v[28:29], v31, off
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v20, v21 offset1:1
	ds_store_2addr_b32 v6, v22, v23 offset0:2 offset1:3
	ds_store_2addr_b32 v6, v24, v25 offset0:4 offset1:5
	ds_store_2addr_b32 v6, v26, v27 offset0:6 offset1:7
	ds_store_b32 v9, v12
	ds_store_b32 v10, v13
	ds_store_b32 v11, v14
	ds_store_b32 v68, v15
	ds_store_b32 v69, v16
	ds_store_b32 v70, v17
	ds_store_b32 v71, v18
	ds_store_b32 v72, v19
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
	s_cbranch_execz .LBB0_120
; %bb.119:
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
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 49, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_122
; %bb.121:
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
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_124
; %bb.123:
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
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 51, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_126
; %bb.125:
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
	ds_load_b32 v9, v9 offset:384
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:204
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 52, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_128
; %bb.127:
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
	ds_load_b32 v9, v9 offset:512
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:208
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 53, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_130
; %bb.129:
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
	ds_load_b32 v9, v9 offset:640
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:212
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 54, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_132
; %bb.131:
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
	ds_load_b32 v9, v9 offset:768
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:216
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 55, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_134
; %bb.133:
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
	ds_load_b32 v9, v9 offset:896
	v_add_co_u32 v5, vcc_lo, s6, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s7, v6, vcc_lo
	global_load_b32 v7, v[7:8], off offset:220
	global_load_b32 v8, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v8, v7, v9
	global_store_b32 v[5:6], v8, off
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 56, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_136
; %bb.135:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v65
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
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 57, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_138
; %bb.137:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v66
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
.LBB0_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 58, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_140
; %bb.139:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v67
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
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 59, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v73
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
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v74
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
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 61, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_146
; %bb.145:
	v_mad_co_u64_u32 v[5:6], null, v5, s16, v[3:4]
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v9, 31, v75
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
.LBB0_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 62, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_148
; %bb.147:
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
.LBB0_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 63, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s18, v5
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_150
; %bb.149:
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
	.size	re_frag, .Lfunc_end0-re_frag
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel re_frag
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
		.amdhsa_next_free_vgpr 198
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-re_frag)<<4)&4080)>>4
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
	.set .Lre_frag.num_vgpr, 198
	.set .Lre_frag.num_agpr, 0
	.set .Lre_frag.numbered_sgpr, 32
	.set .Lre_frag.num_named_barrier, 0
	.set .Lre_frag.private_seg_size, 0
	.set .Lre_frag.uses_vcc, 1
	.set .Lre_frag.uses_flat_scratch, 0
	.set .Lre_frag.has_dyn_sized_stack, 0
	.set .Lre_frag.has_recursion, 0
	.set .Lre_frag.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16996
; TotalNumSgprs: 34
; NumVgprs: 198
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 198
; Occupancy: 7
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
	.type	__hip_cuid_1fe34f1e0c791613,@object ; @__hip_cuid_1fe34f1e0c791613
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1fe34f1e0c791613
__hip_cuid_1fe34f1e0c791613:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1fe34f1e0c791613, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_1fe34f1e0c791613
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
    .name:           re_frag
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         re_frag.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     198
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
