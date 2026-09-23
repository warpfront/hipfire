	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gate_symfold            ; -- Begin function gate_symfold
	.globl	gate_symfold
	.p2align	8
	.type	gate_symfold,@function
gate_symfold:                           ; @gate_symfold
	.cfi_startproc
; %bb.0:                                ; %.preheader495
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[8:11], s[0:1], 0x38
	v_bfe_u32 v1, v0, 4, 1
	v_lshrrev_b32_e32 v2, 1, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v159, 3, v1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s10, 0xff
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.preheader495..preheader480_crit_edge
	v_lshlrev_b32_e32 v1, 3, v1
	s_mov_b32 s2, 0
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b32 s2, -1
                                        ; implicit-def: $vgpr1
.LBB0_3:                                ; %Flow1112
	v_and_b32_e32 v157, 15, v0
	v_and_b32_e32 v158, 0x60, v0
	v_and_b32_e32 v156, 64, v2
	s_lshl_b32 s21, ttmp7, 7
	s_add_co_i32 s20, s9, s8
	s_lshl_b32 s22, ttmp9, 7
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_mov_b32 s17, 0
	s_cbranch_vccnz .LBB0_20
; %bb.4:                                ; %.preheader493.lr.ph
	v_lshrrev_b32_e32 v1, 2, v0
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x10
	s_add_co_i32 s5, s20, -1
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v3, s22, v0
	v_add_nc_u32_e32 v2, s22, v1
	v_add_nc_u32_e32 v5, s21, v1
	s_ashr_i32 s2, s10, 31
	s_delay_alu instid0(VALU_DEP_3)
	v_min_i32_e32 v13, s5, v3
	s_lshr_b32 s2, s2, 24
	v_min_i32_e32 v4, s5, v2
	v_cmp_gt_i32_e64 s3, s20, v2
	v_add_nc_u32_e32 v10, 64, v5
	s_add_co_i32 s2, s10, s2
	v_or_b32_e32 v6, 64, v1
	v_cmp_gt_i32_e32 vcc_lo, s8, v4
	v_add_nc_u32_e32 v9, 64, v2
	s_ashr_i32 s7, s2, 8
	v_lshlrev_b32_e32 v23, 2, v0
	s_mul_i32 s16, s7, 0x88
	v_cndmask_b32_e64 v7, s8, 0, vcc_lo
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v11, s12
	v_mul_u32_u24_e32 v21, 0x48, v6
	v_and_b32_e32 v6, 0x6f, v0
	s_add_co_i32 s12, s11, -1
	v_sub_nc_u32_e32 v2, v4, v7
	v_mov_b32_e32 v7, s13
	v_cndmask_b32_e32 v15, s14, v11, vcc_lo
	v_mad_u32_u24 v24, 0x48, v6, 0
	v_cmp_gt_i32_e64 s4, s11, v10
	v_mul_lo_u32 v2, s16, v2
	v_cndmask_b32_e32 v14, s15, v7, vcc_lo
	v_min_i32_e32 v4, s5, v9
	v_cmp_gt_i32_e32 vcc_lo, s8, v13
	v_mul_u32_u24_e32 v22, 0x48, v1
	v_dual_mov_b32 v16, v8 :: v_dual_add_nc_u32 v25, 0, v159
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s6, s8, v4
	v_add_co_u32 v2, s5, v15, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, 0, v14, s5
	v_cndmask_b32_e64 v12, s8, 0, s6
	v_cmp_gt_i32_e64 s5, s20, v9
	v_cndmask_b32_e64 v9, s14, v11, s6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, s14, v11, vcc_lo
	v_mad_u32_u24 v26, 0x48, v1, 0
	v_sub_nc_u32_e32 v4, v4, v12
	v_cndmask_b32_e64 v12, s8, 0, vcc_lo
	v_mov_b32_e32 v19, v8
	v_mov_b32_e32 v17, v8
	s_mov_b32 s14, 0x4e4c4a48
	v_mul_lo_u32 v4, s16, v4
	v_sub_nc_u32_e32 v12, v13, v12
	v_cndmask_b32_e64 v13, s15, v7, s6
	v_cndmask_b32_e32 v7, s15, v7, vcc_lo
	s_mov_b32 s15, 0x4040404
	s_mov_b32 s23, 0
	v_mul_lo_u32 v12, s16, v12
                                        ; implicit-def: $vgpr76_vgpr77_vgpr78_vgpr79_vgpr80_vgpr81_vgpr82_vgpr83
                                        ; implicit-def: $vgpr84_vgpr85_vgpr86_vgpr87_vgpr88_vgpr89_vgpr90_vgpr91
                                        ; implicit-def: $vgpr92_vgpr93_vgpr94_vgpr95_vgpr96_vgpr97_vgpr98_vgpr99
                                        ; implicit-def: $vgpr100_vgpr101_vgpr102_vgpr103_vgpr104_vgpr105_vgpr106_vgpr107
                                        ; implicit-def: $vgpr108_vgpr109_vgpr110_vgpr111_vgpr112_vgpr113_vgpr114_vgpr115
                                        ; implicit-def: $vgpr116_vgpr117_vgpr118_vgpr119_vgpr120_vgpr121_vgpr122_vgpr123
                                        ; implicit-def: $vgpr124_vgpr125_vgpr126_vgpr127_vgpr128_vgpr129_vgpr130_vgpr131
                                        ; implicit-def: $vgpr132_vgpr133_vgpr134_vgpr135_vgpr136_vgpr137_vgpr138_vgpr139
	v_mov_b32_e32 v6, v8
	v_add_co_u32 v4, s6, v9, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, 0, v13, s6
	v_and_b32_e32 v13, 3, v0
	v_cmp_gt_i32_e64 s6, s20, v3
	v_add_co_u32 v160, vcc_lo, v11, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v161, null, 0, v7, vcc_lo
	v_dual_mov_b32 v12, v8 :: v_dual_lshlrev_b32 v7, 3, v13
	v_lshlrev_b32_e32 v20, 4, v13
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v3, s12, v10
	v_mov_b32_e32 v10, v8
	v_cmp_gt_i32_e64 s2, s11, v5
	v_min_i32_e32 v5, s12, v5
	v_add_co_u32 v162, vcc_lo, v2, v7
	v_mad_co_u64_u32 v[148:149], null, v3, s10, v[20:21]
	v_or_b32_e32 v3, v156, v157
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v163, null, 0, v14, vcc_lo
	v_dual_mov_b32 v14, v8 :: v_dual_mov_b32 v11, v8
	v_add_nc_u32_e32 v166, 0, v23
	v_mad_co_u64_u32 v[149:150], null, v5, s10, v[20:21]
	v_or_b32_e32 v5, v159, v158
	v_add_co_u32 v164, vcc_lo, v4, v7
	v_dual_mov_b32 v18, v8 :: v_dual_add_nc_u32 v27, 0, v20
	v_mul_u32_u24_e32 v28, 0x48, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v165, null, 0, v9, vcc_lo
	v_mov_b32_e32 v9, v8
	v_mov_b32_e32 v143, v11
	v_dual_mov_b32 v13, v8 :: v_dual_mov_b32 v142, v10
	v_dual_mov_b32 v15, v8 :: v_dual_mov_b32 v140, v8
	v_lshlrev_b32_e32 v29, 2, v5
	v_dual_mov_b32 v1, v8 :: v_dual_add_nc_u32 v168, v26, v20
	v_dual_mov_b32 v3, v8 :: v_dual_add_nc_u32 v172, v25, v28
	v_add3_u32 v169, v27, v22, 0x2400
	v_add3_u32 v170, v27, v21, 0x2400
	v_add_nc_u32_e32 v171, v24, v159
	v_dual_mov_b32 v27, v19 :: v_dual_mov_b32 v26, v18
	v_dual_mov_b32 v20, v12 :: v_dual_add_nc_u32 v167, 0, v29
	v_mov_b32_e32 v35, v19
	v_mov_b32_e32 v43, v19
	v_mov_b32_e32 v51, v19
	v_mov_b32_e32 v59, v19
	v_mov_b32_e32 v67, v19
	v_dual_mov_b32 v75, v19 :: v_dual_mov_b32 v2, v8
	v_dual_mov_b32 v4, v8 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v24, v16 :: v_dual_mov_b32 v7, v8
	v_dual_mov_b32 v22, v14 :: v_dual_mov_b32 v25, v17
	v_dual_mov_b32 v30, v14 :: v_dual_mov_b32 v23, v15
	v_dual_mov_b32 v28, v12 :: v_dual_mov_b32 v21, v13
	v_mov_b32_e32 v42, v18
	v_dual_mov_b32 v34, v18 :: v_dual_mov_b32 v33, v17
	v_mov_b32_e32 v38, v14
	v_dual_mov_b32 v32, v16 :: v_dual_mov_b32 v31, v15
	v_dual_mov_b32 v36, v12 :: v_dual_mov_b32 v29, v13
	v_dual_mov_b32 v50, v18 :: v_dual_mov_b32 v41, v17
	v_mov_b32_e32 v46, v14
	v_dual_mov_b32 v40, v16 :: v_dual_mov_b32 v39, v15
	v_dual_mov_b32 v44, v12 :: v_dual_mov_b32 v37, v13
	v_dual_mov_b32 v58, v18 :: v_dual_mov_b32 v49, v17
	v_mov_b32_e32 v54, v14
	v_dual_mov_b32 v48, v16 :: v_dual_mov_b32 v47, v15
	v_dual_mov_b32 v52, v12 :: v_dual_mov_b32 v45, v13
	v_dual_mov_b32 v66, v18 :: v_dual_mov_b32 v57, v17
	v_mov_b32_e32 v62, v14
	v_dual_mov_b32 v56, v16 :: v_dual_mov_b32 v55, v15
	v_dual_mov_b32 v60, v12 :: v_dual_mov_b32 v53, v13
	v_dual_mov_b32 v74, v18 :: v_dual_mov_b32 v65, v17
	v_mov_b32_e32 v70, v14
	v_dual_mov_b32 v64, v16 :: v_dual_mov_b32 v63, v15
	v_dual_mov_b32 v68, v12 :: v_dual_mov_b32 v61, v13
	v_mov_b32_e32 v141, v9
	v_dual_mov_b32 v73, v17 :: v_dual_mov_b32 v72, v16
	v_mov_b32_e32 v71, v15
	v_mov_b32_e32 v69, v13
	s_mov_b32 s10, 0xb8c0c4c8
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_add_co_i32 s23, s23, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s23, s7
	s_cbranch_scc1 .LBB0_21
.LBB0_6:                                ; %.preheader493
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
                                        ;       Child Loop BB0_10 Depth 3
	s_lshl_b32 s16, s23, 8
	s_mul_i32 s24, s23, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[18:19], s[16:17]
	s_mov_b32 s16, -1
	s_mov_b32 s27, 0
	s_branch .LBB0_8
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=2
	s_mov_b32 s27, 1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s16, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_5
.LBB0_8:                                ; %.preheader492
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_10 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s25, s16, -1
	s_lshl_b32 s16, s27, 2
	v_add_nc_u32_e32 v11, 0x1200, v168
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s16, s16, s24
	s_mov_b32 s26, -1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v9, vcc_lo, v160, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, 0, v161, vcc_lo
	s_lshl_b32 s16, s27, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[28:29], s[12:13], s[16:17]
	global_load_d16_b16 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v9, v9.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, 0, v9, s6
	ds_store_b32 v166, v9 offset:18432
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v9, s16, s28, v149
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s29, 0, s16
	v_add_co_u32 v150, s16, s28, v148
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v151, null, s29, 0, s16
	global_load_b128 v[144:147], v[9:10], off
	s_lshl_b32 s16, s27, 6
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s16, s24, s16
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v152, vcc_lo, v162, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, 0, v163, vcc_lo
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v145, 0, v145, s2
	v_cndmask_b32_e64 v144, 0, v144, s2
	v_cndmask_b32_e64 v147, 0, v147, s2
	v_cndmask_b32_e64 v146, 0, v146, s2
	ds_store_2addr_b64 v168, v[144:145], v[146:147] offset1:1
	global_load_b128 v[144:147], v[150:151], off
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v145, 0, v145, s4
	v_cndmask_b32_e64 v144, 0, v144, s4
	v_cndmask_b32_e64 v147, 0, v147, s4
	v_cndmask_b32_e64 v146, 0, v146, s4
	ds_store_2addr_b64 v11, v[144:145], v[146:147] offset1:1
	global_load_b64 v[144:145], v[152:153], off offset:8
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v11, 0, v144, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v144, 0xf0f0f0f, v11
	v_lshrrev_b32_e32 v11, 4, v11
	v_and_b32_e32 v11, 0xf0f0f0f, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v146, v11, v144, 0x5010400
	v_perm_b32 v11, v11, v144, 0x7030602
	v_and_b32_e32 v144, 0x7070707, v146
	v_lshrrev_b32_e32 v146, 1, v146
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v147, s10, 0xcaccced0, v144
	v_perm_b32 v144, s14, 0x44403800, v144
	v_and_or_b32 v146, v146, s15, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v146, v144, v147, v146
	v_and_b32_e32 v144, 0x7070707, v11
	v_lshrrev_b32_e32 v11, 1, v11
	v_perm_b32 v147, s10, 0xcaccced0, v144
	v_perm_b32 v144, s14, 0x44403800, v144
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v11, v11, s15, 0x3020100
	v_perm_b32 v147, v144, v147, v11
	v_cndmask_b32_e64 v11, 0, v145, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v144, 0xf0f0f0f, v11
	v_lshrrev_b32_e32 v11, 4, v11
	v_and_b32_e32 v11, 0xf0f0f0f, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v145, v11, v144, 0x5010400
	v_perm_b32 v11, v11, v144, 0x7030602
	v_and_b32_e32 v144, 0x7070707, v145
	v_lshrrev_b32_e32 v145, 1, v145
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v154, s10, 0xcaccced0, v144
	v_perm_b32 v144, s14, 0x44403800, v144
	v_and_or_b32 v145, v145, s15, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v144, v144, v154, v145
	v_and_b32_e32 v145, 0x7070707, v11
	v_lshrrev_b32_e32 v11, 1, v11
	v_perm_b32 v154, s10, 0xcaccced0, v145
	v_perm_b32 v145, s14, 0x44403800, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v11, v11, s15, 0x3020100
	v_perm_b32 v145, v145, v154, v11
	v_add_co_u32 v154, vcc_lo, v164, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v155, null, 0, v165, vcc_lo
	ds_store_2addr_b64 v169, v[146:147], v[144:145] offset1:1
	global_load_b64 v[144:145], v[154:155], off offset:8
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v11, 0, v144, s5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v144, 0xf0f0f0f, v11
	v_lshrrev_b32_e32 v11, 4, v11
	v_and_b32_e32 v11, 0xf0f0f0f, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v146, v11, v144, 0x5010400
	v_perm_b32 v11, v11, v144, 0x7030602
	v_and_b32_e32 v144, 0x7070707, v146
	v_lshrrev_b32_e32 v146, 1, v146
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v147, s10, 0xcaccced0, v144
	v_perm_b32 v144, s14, 0x44403800, v144
	v_and_or_b32 v146, v146, s15, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v146, v144, v147, v146
	v_and_b32_e32 v144, 0x7070707, v11
	v_lshrrev_b32_e32 v11, 1, v11
	v_perm_b32 v147, s10, 0xcaccced0, v144
	v_perm_b32 v144, s14, 0x44403800, v144
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v11, v11, s15, 0x3020100
	v_perm_b32 v147, v144, v147, v11
	v_cndmask_b32_e64 v11, 0, v145, s5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v144, 0xf0f0f0f, v11
	v_lshrrev_b32_e32 v11, 4, v11
	v_and_b32_e32 v11, 0xf0f0f0f, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v145, v11, v144, 0x5010400
	v_perm_b32 v11, v11, v144, 0x7030602
	v_and_b32_e32 v144, 0x7070707, v145
	v_lshrrev_b32_e32 v145, 1, v145
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v173, s10, 0xcaccced0, v144
	v_perm_b32 v144, s14, 0x44403800, v144
	v_and_or_b32 v145, v145, s15, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v144, v144, v173, v145
	v_and_b32_e32 v145, 0x7070707, v11
	v_lshrrev_b32_e32 v11, 1, v11
	v_perm_b32 v173, s10, 0xcaccced0, v145
	v_perm_b32 v145, s14, 0x44403800, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v11, v11, s15, 0x3020100
	v_perm_b32 v145, v145, v173, v11
	ds_store_2addr_b64 v170, v[146:147], v[144:145] offset1:1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_10
.LBB0_9:                                ; %.loopexit
                                        ;   in Loop: Header=BB0_10 Depth=3
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s27, -1
	s_mov_b32 s26, 0
	s_and_b32 vcc_lo, exec_lo, s16
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_7
.LBB0_10:                               ;   Parent Loop BB0_6 Depth=1
                                        ;     Parent Loop BB0_8 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_xor_b32 s16, s26, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s16
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
; %bb.11:                               ; %.preheader568
                                        ;   in Loop: Header=BB0_10 Depth=3
	s_clause 0x1
	global_load_b128 v[1:4], v[9:10], off offset:64
	global_load_b128 v[5:8], v[150:151], off offset:64
	global_load_b64 v[140:141], v[152:153], off offset:40
	global_load_b64 v[142:143], v[154:155], off offset:40
.LBB0_12:                               ; %.loopexit491
                                        ;   in Loop: Header=BB0_10 Depth=3
	v_add_nc_u32_e32 v173, 0x2400, v171
	v_add_nc_u32_e32 v11, 0x800, v172
	s_and_not1_b32 vcc_lo, exec_lo, s16
	s_mov_b32 s28, -1
	ds_load_2addr_b64 v[144:147], v173 offset1:144
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_16
; %bb.13:                               ; %Flow
                                        ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_17
.LBB0_14:                               ; %.loopexit488
                                        ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_18
.LBB0_15:                               ; %.critedge
                                        ;   in Loop: Header=BB0_10 Depth=3
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
	s_branch .LBB0_19
.LBB0_16:                               ; %.preheader489.preheader
                                        ;   in Loop: Header=BB0_10 Depth=3
	ds_load_2addr_b64 v[174:177], v172 offset1:144
	ds_load_2addr_b64 v[178:181], v11 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[174:175], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[146:147], v[174:175], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[144:145], v[176:177], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[146:147], v[176:177], v[108:115]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[144:145], v[178:179], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[146:147], v[178:179], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[144:145], v[180:181], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[76:83], v[146:147], v[180:181], v[76:83]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[174:177], v173 offset0:2 offset1:146
	ds_load_2addr_b64 v[178:181], v172 offset0:2 offset1:146
	ds_load_2addr_b64 v[182:185], v11 offset0:34 offset1:178
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[174:175], v[178:179], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[176:177], v[178:179], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[174:175], v[180:181], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[176:177], v[180:181], v[108:115]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[174:175], v[182:183], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[176:177], v[182:183], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[174:175], v[184:185], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[76:83], v[176:177], v[184:185], v[76:83]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[174:177], v173 offset0:4 offset1:148
	ds_load_2addr_b64 v[178:181], v172 offset0:4 offset1:148
	ds_load_2addr_b64 v[182:185], v11 offset0:36 offset1:180
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[174:175], v[178:179], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[176:177], v[178:179], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[174:175], v[180:181], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[176:177], v[180:181], v[108:115]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[174:175], v[182:183], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[176:177], v[182:183], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[174:175], v[184:185], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[76:83], v[176:177], v[184:185], v[76:83]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[174:177], v173 offset0:6 offset1:150
	ds_load_2addr_b64 v[178:181], v172 offset0:6 offset1:150
	ds_load_2addr_b64 v[182:185], v11 offset0:38 offset1:182
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[174:175], v[178:179], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[176:177], v[178:179], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[174:175], v[180:181], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[176:177], v[180:181], v[108:115]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[174:175], v[182:183], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[176:177], v[182:183], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[174:175], v[184:185], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[76:83], v[176:177], v[184:185], v[76:83]
	; sched_barrier mask(0x00000000)
	s_cbranch_execnz .LBB0_14
.LBB0_17:                               ; %.loopexit488.loopexit
                                        ;   in Loop: Header=BB0_10 Depth=3
	ds_load_2addr_b64 v[76:79], v172 offset1:144
	ds_load_2addr_b64 v[174:177], v11 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_nop
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[76:77], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[146:147], v[76:77], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[144:145], v[78:79], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[146:147], v[78:79], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[144:145], v[174:175], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[146:147], v[174:175], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[144:145], v[176:177], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[76:83], v[146:147], v[176:177], 0
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[144:147], v173 offset0:2 offset1:146
	ds_load_2addr_b64 v[174:177], v172 offset0:2 offset1:146
	ds_load_2addr_b64 v[178:181], v11 offset0:34 offset1:178
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[174:175], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[146:147], v[174:175], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[144:145], v[176:177], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[146:147], v[176:177], v[108:115]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[144:145], v[178:179], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[146:147], v[178:179], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[144:145], v[180:181], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[76:83], v[146:147], v[180:181], v[76:83]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[144:147], v173 offset0:4 offset1:148
	ds_load_2addr_b64 v[174:177], v172 offset0:4 offset1:148
	ds_load_2addr_b64 v[178:181], v11 offset0:36 offset1:180
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[174:175], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[146:147], v[174:175], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[144:145], v[176:177], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[146:147], v[176:177], v[108:115]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[144:145], v[178:179], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[146:147], v[178:179], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[144:145], v[180:181], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[76:83], v[146:147], v[180:181], v[76:83]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[144:147], v173 offset0:6 offset1:150
	ds_load_2addr_b64 v[173:176], v172 offset0:6 offset1:150
	ds_load_2addr_b64 v[177:180], v11 offset0:38 offset1:182
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[173:174], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[146:147], v[173:174], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[144:145], v[175:176], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[146:147], v[175:176], v[108:115]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[144:145], v[177:178], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[146:147], v[177:178], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[144:145], v[179:180], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[76:83], v[146:147], v[179:180], v[76:83]
	; sched_barrier mask(0x00000000)
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_15
.LBB0_18:                               ; %.preheader487.preheader
                                        ;   in Loop: Header=BB0_10 Depth=3
	s_wait_loadcnt 0x3
	v_mov_b32_e32 v173, v1
	v_mov_b32_e32 v175, v3
	s_wait_loadcnt 0x2
	v_mov_b32_e32 v177, v5
	v_mov_b32_e32 v179, v7
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v174, v2
	v_mov_b32_e32 v176, v4
	v_cndmask_b32_e64 v2, 0, v2, s2
	v_cndmask_b32_e64 v1, 0, v173, s2
	v_cndmask_b32_e64 v4, 0, v4, s2
	v_cndmask_b32_e64 v3, 0, v175, s2
	v_mov_b32_e32 v178, v6
	v_dual_mov_b32 v180, v8 :: v_dual_add_nc_u32 v5, 0x1000, v168
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b64 v168, v[1:2], v[3:4] offset1:1
	v_cndmask_b32_e64 v2, 0, v6, s4
	v_cndmask_b32_e64 v1, 0, v177, s4
	v_cndmask_b32_e64 v4, 0, v8, s4
	v_cndmask_b32_e64 v3, 0, v179, s4
	ds_store_2addr_b64 v5, v[1:2], v[3:4] offset0:64 offset1:65
	v_cndmask_b32_e64 v1, 0, v140, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v1, v2, 0x5010400
	v_perm_b32 v2, v1, v2, 0x7030602
	v_and_b32_e32 v1, 0x7070707, v3
	v_lshrrev_b32_e32 v3, 1, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v4, s10, 0xcaccced0, v1
	v_perm_b32 v1, s14, 0x44403800, v1
	v_and_or_b32 v3, v3, s15, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v1, v1, v4, v3
	v_and_b32_e32 v3, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	v_perm_b32 v4, s10, 0xcaccced0, v3
	v_perm_b32 v3, s14, 0x44403800, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, v2, s15, 0x3020100
	v_perm_b32 v2, v3, v4, v2
	v_cndmask_b32_e64 v3, 0, v141, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v4, 0xf0f0f0f, v3
	v_lshrrev_b32_e32 v3, 4, v3
	v_and_b32_e32 v3, 0xf0f0f0f, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v5, v3, v4, 0x5010400
	v_perm_b32 v4, v3, v4, 0x7030602
	v_and_b32_e32 v3, 0x7070707, v5
	v_lshrrev_b32_e32 v5, 1, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v6, s10, 0xcaccced0, v3
	v_perm_b32 v3, s14, 0x44403800, v3
	v_and_or_b32 v5, v5, s15, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v3, v6, v5
	v_and_b32_e32 v5, 0x7070707, v4
	v_lshrrev_b32_e32 v4, 1, v4
	v_perm_b32 v6, s10, 0xcaccced0, v5
	v_perm_b32 v5, s14, 0x44403800, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v4, v4, s15, 0x3020100
	v_perm_b32 v4, v5, v6, v4
	ds_store_2addr_b64 v169, v[1:2], v[3:4] offset1:1
	v_cndmask_b32_e64 v1, 0, v142, s5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, 0xf0f0f0f, v1
	v_lshrrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v1, 0xf0f0f0f, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v1, v2, 0x5010400
	v_perm_b32 v2, v1, v2, 0x7030602
	v_and_b32_e32 v1, 0x7070707, v3
	v_lshrrev_b32_e32 v3, 1, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v4, s10, 0xcaccced0, v1
	v_perm_b32 v1, s14, 0x44403800, v1
	v_and_or_b32 v3, v3, s15, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v1, v1, v4, v3
	v_and_b32_e32 v3, 0x7070707, v2
	v_lshrrev_b32_e32 v2, 1, v2
	v_perm_b32 v4, s10, 0xcaccced0, v3
	v_perm_b32 v3, s14, 0x44403800, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v2, v2, s15, 0x3020100
	v_perm_b32 v2, v3, v4, v2
	v_cndmask_b32_e64 v3, 0, v143, s5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_and_b32_e32 v4, 0xf0f0f0f, v3
	v_lshrrev_b32_e32 v3, 4, v3
	v_and_b32_e32 v3, 0xf0f0f0f, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v5, v3, v4, 0x5010400
	v_perm_b32 v4, v3, v4, 0x7030602
	v_and_b32_e32 v3, 0x7070707, v5
	v_lshrrev_b32_e32 v5, 1, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v6, s10, 0xcaccced0, v3
	v_perm_b32 v3, s14, 0x44403800, v3
	v_and_or_b32 v5, v5, s15, 0x3020100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_perm_b32 v3, v3, v6, v5
	v_and_b32_e32 v5, 0x7070707, v4
	v_lshrrev_b32_e32 v4, 1, v4
	v_perm_b32 v6, s10, 0xcaccced0, v5
	v_perm_b32 v5, s14, 0x44403800, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v4, v4, s15, 0x3020100
	v_perm_b32 v4, v5, v6, v4
	ds_store_2addr_b64 v170, v[1:2], v[3:4] offset1:1
	v_dual_mov_b32 v1, v173 :: v_dual_mov_b32 v2, v174
	v_dual_mov_b32 v3, v175 :: v_dual_mov_b32 v4, v176
	v_dual_mov_b32 v5, v177 :: v_dual_mov_b32 v6, v178
	v_dual_mov_b32 v7, v179 :: v_dual_mov_b32 v8, v180
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_9
.LBB0_19:                               ; %.preheader481.preheader
                                        ;   in Loop: Header=BB0_10 Depth=3
	v_add_nc_u32_e32 v11, 0x4800, v167
	v_add_nc_u32_e32 v175, 0x4818, v167
	s_wait_dscnt 0x0
	ds_load_2addr_b32 v[144:145], v11 offset1:1
	ds_load_2addr_b32 v[175:176], v175 offset1:1
	v_add_nc_u32_e32 v11, 0x4808, v167
	ds_load_2addr_b32 v[146:147], v11 offset1:1
	s_wait_dscnt 0x2
	v_fma_f32 v68, v144, v132, v68
	s_wait_dscnt 0x1
	v_fma_f32 v74, v175, v138, v74
	v_fma_f32 v58, v175, v122, v58
	v_fma_f32 v42, v175, v106, v42
	v_fma_f32 v26, v175, v90, v26
	v_add_nc_u32_e32 v175, 0x4858, v167
	v_fmac_f32_e32 v75, v176, v139
	v_fmac_f32_e32 v59, v176, v123
	v_fmac_f32_e32 v43, v176, v107
	v_fmac_f32_e32 v27, v176, v91
	ds_load_2addr_b32 v[175:176], v175 offset1:1
	v_add_nc_u32_e32 v11, 0x4810, v167
	v_fma_f32 v52, v144, v116, v52
	v_fma_f32 v36, v144, v100, v36
	v_fma_f32 v20, v144, v84, v20
	v_fma_f32 v69, v145, v133, v69
	ds_load_2addr_b32 v[173:174], v11 offset1:1
	v_add_nc_u32_e32 v11, 0x4840, v167
	v_fma_f32 v53, v145, v117, v53
	v_fma_f32 v37, v145, v101, v37
	v_fma_f32 v21, v145, v85, v21
	s_wait_dscnt 0x2
	v_fma_f32 v70, v146, v134, v70
	ds_load_2addr_b32 v[144:145], v11 offset1:1
	v_add_nc_u32_e32 v11, 0x4848, v167
	v_fma_f32 v54, v146, v118, v54
	v_fma_f32 v38, v146, v102, v38
	v_fma_f32 v22, v146, v86, v22
	v_fma_f32 v71, v147, v135, v71
	v_fma_f32 v55, v147, v119, v55
	v_fma_f32 v39, v147, v103, v39
	v_fma_f32 v23, v147, v87, v23
	ds_load_2addr_b32 v[146:147], v11 offset1:1
	v_add_nc_u32_e32 v11, 0x4850, v167
	s_wait_dscnt 0x3
	v_fma_f32 v66, v175, v130, v66
	v_fma_f32 v50, v175, v114, v50
	v_fma_f32 v34, v175, v98, v34
	s_wait_dscnt 0x2
	v_fma_f32 v72, v173, v136, v72
	v_fma_f32 v56, v173, v120, v56
	v_fma_f32 v40, v173, v104, v40
	v_fma_f32 v24, v173, v88, v24
	v_fma_f32 v73, v174, v137, v73
	v_fma_f32 v57, v174, v121, v57
	v_fma_f32 v41, v174, v105, v41
	v_fma_f32 v25, v174, v89, v25
	ds_load_2addr_b32 v[173:174], v11 offset1:1
	s_wait_dscnt 0x2
	v_fma_f32 v60, v144, v124, v60
	v_fma_f32 v44, v144, v108, v44
	v_fma_f32 v28, v144, v92, v28
	v_fma_f32 v12, v144, v76, v12
	v_fma_f32 v61, v145, v125, v61
	v_fma_f32 v45, v145, v109, v45
	v_fma_f32 v29, v145, v93, v29
	v_fma_f32 v13, v145, v77, v13
	s_wait_dscnt 0x1
	v_fma_f32 v62, v146, v126, v62
	v_fma_f32 v46, v146, v110, v46
	v_fma_f32 v30, v146, v94, v30
	v_fma_f32 v14, v146, v78, v14
	v_fma_f32 v63, v147, v127, v63
	v_fma_f32 v47, v147, v111, v47
	v_fma_f32 v31, v147, v95, v31
	v_fma_f32 v15, v147, v79, v15
	v_fma_f32 v18, v175, v82, v18
	v_fmac_f32_e32 v67, v176, v131
	v_fmac_f32_e32 v51, v176, v115
	s_wait_dscnt 0x0
	v_fma_f32 v64, v173, v128, v64
	v_fma_f32 v48, v173, v112, v48
	v_fma_f32 v32, v173, v96, v32
	v_fma_f32 v16, v173, v80, v16
	v_fma_f32 v65, v174, v129, v65
	v_fma_f32 v49, v174, v113, v49
	v_fma_f32 v33, v174, v97, v33
	v_fma_f32 v17, v174, v81, v17
	v_fmac_f32_e32 v35, v176, v99
	v_fmac_f32_e32 v19, v176, v83
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
.LBB0_21:                               ; %Flow1110
	v_mov_b32_e32 v1, v159
.LBB0_22:                               ; %Flow1113
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x20
	s_load_b64 s[2:3], s[0:1], 0x30
	v_lshlrev_b32_e32 v2, 6, v0
	v_add_nc_u32_e32 v1, v1, v157
	v_and_b32_e32 v76, 31, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v2, 0x3800, v2
	v_add_nc_u32_e32 v3, 17, v1
	v_add_nc_u32_e32 v5, 18, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v4, 0, v2
	v_and_b32_e32 v2, 31, v1
	v_and_b32_e32 v3, 31, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v5, 31, v5
	v_lshl_add_u32 v6, v157, 7, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v2, 16, v2
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v77, s7
	v_or3_b32 v7, s22, v158, v76
	v_lshl_add_u32 v8, v1, 2, v6
	v_lshl_add_u32 v9, v2, 2, v6
	v_lshl_add_u32 v10, v3, 2, v6
	v_lshl_add_u32 v11, v5, 2, v6
	ds_store_2addr_b32 v8, v68, v69 offset1:1
	ds_store_2addr_b32 v8, v70, v71 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v72, v73 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v74, v75 offset0:6 offset1:7
	ds_store_b32 v9, v60
	ds_store_b32 v10, v61
	ds_store_b32 v11, v62
	v_add_nc_u32_e32 v60, 22, v1
	v_cmp_gt_i32_e64 s0, s8, v7
	v_cmp_le_i32_e32 vcc_lo, s20, v7
	v_cmp_gt_i32_e64 s1, s20, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v60, 31, v60
	v_lshl_add_u32 v69, v60, 2, v6
	v_mov_b32_e32 v60, s8
	v_add_nc_u32_e32 v2, 19, v1
	v_add_nc_u32_e32 v3, 20, v1
	v_add_nc_u32_e32 v5, 21, v1
	v_add_nc_u32_e32 v1, 23, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v2, 31, v2
	v_and_b32_e32 v3, 31, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v5, 31, v5
	v_and_b32_e32 v1, 31, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v61, v2, 2, v6
	v_lshl_add_u32 v62, v3, 2, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_add_u32 v68, v5, 2, v6
	v_lshl_add_u32 v70, v1, 2, v6
	ds_store_b32 v61, v63
	ds_store_b32 v62, v64
	ds_store_b32 v68, v65
	ds_store_b32 v69, v66
	ds_store_b32 v70, v67
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v1, s6 :: v_dual_add_nc_u32 v2, s21, v156
	v_cndmask_b32_e64 v3, s8, 0, s0
	v_cndmask_b32_e64 v5, s3, v77, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v6, s2, v1, s0
	v_sub_nc_u32_e32 v1, v7, v3
	v_cndmask_b32_e64 v7, s9, v60, s0
	v_cmp_gt_i32_e64 s0, s11, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u32 v60, v76, 2, v4
	s_and_b32 s0, s0, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_lshlrev_b64_e32 v[63:64], 2, v[2:3]
	ds_load_b32 v66, v60
	v_add_co_u32 v63, s0, s4, v63
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v64, null, s5, v64, s0
	global_load_b32 v65, v[63:64], off
	v_mad_co_u64_u32 v[63:64], null, v2, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v64, 0 :: v_dual_mul_f32 v65, v65, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[63:64], 2, v[63:64]
	v_add_co_u32 v63, s0, v6, v63
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v64, null, v5, v64, s0
	global_store_b32 v[63:64], v65, off
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v64, 1, v2
	v_add_nc_u32_e32 v63, 1, v0
	s_xor_b32 s1, vcc_lo, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s0, s11, v64
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_lshlrev_b64_e32 v[65:66], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v65, vcc_lo, s4, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, s5, v66, vcc_lo
	global_load_b32 v66, v[65:66], off offset:4
	v_and_b32_e32 v65, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v65, v65, 2, v4
	ds_load_b32 v67, v65 offset:128
	v_mad_co_u64_u32 v[64:65], null, v64, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v65, 0 :: v_dual_mul_f32 v66, v66, v67
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v64, vcc_lo, v6, v64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, v5, v65, vcc_lo
	global_store_b32 v[64:65], v66, off
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v65, 2, v2
	v_add_nc_u32_e32 v64, 2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v65
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_lshlrev_b64_e32 v[66:67], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v66, vcc_lo, s4, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, s5, v67, vcc_lo
	global_load_b32 v67, v[66:67], off offset:8
	v_and_b32_e32 v66, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v66, v66, 2, v4
	ds_load_b32 v71, v66 offset:256
	v_mad_co_u64_u32 v[65:66], null, v65, v7, v[1:2]
	v_mov_b32_e32 v66, 0
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v65, vcc_lo, v6, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v5, v66, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v67, v67, v71
	global_store_b32 v[65:66], v67, off
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v66, 3, v2
	v_add_nc_u32_e32 v65, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v66
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_lshlrev_b64_e32 v[71:72], 2, v[2:3]
	v_and_b32_e32 v67, 31, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v67, v67, 2, v4
	v_add_co_u32 v71, vcc_lo, s4, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v72, null, s5, v72, vcc_lo
	global_load_b32 v71, v[71:72], off offset:12
	ds_load_b32 v72, v67 offset:384
	v_mad_co_u64_u32 v[66:67], null, v66, v7, v[1:2]
	v_mov_b32_e32 v67, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v66, vcc_lo, v6, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, v5, v67, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v71, v71, v72
	global_store_b32 v[66:67], v71, off
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v67, 4, v2
	v_add_nc_u32_e32 v66, 4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v67
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_lshlrev_b64_e32 v[71:72], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v71, vcc_lo, s4, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, s5, v72, vcc_lo
	global_load_b32 v73, v[71:72], off offset:16
	v_and_b32_e32 v71, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v71, v71, 2, v4
	ds_load_b32 v74, v71 offset:512
	v_mad_co_u64_u32 v[71:72], null, v67, v7, v[1:2]
	v_mov_b32_e32 v72, 0
	v_lshlrev_b64_e32 v[71:72], 2, v[71:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v71, vcc_lo, v6, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, v5, v72, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v67, v73, v74
	global_store_b32 v[71:72], v67, off
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v71, 5, v2
	v_add_nc_u32_e32 v67, 5, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v71
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_lshlrev_b64_e32 v[72:73], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v72, vcc_lo, s4, v72
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, s5, v73, vcc_lo
	global_load_b32 v73, v[72:73], off offset:20
	v_and_b32_e32 v72, 31, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v72, v72, 2, v4
	ds_load_b32 v74, v72 offset:640
	v_mad_co_u64_u32 v[71:72], null, v71, v7, v[1:2]
	v_mov_b32_e32 v72, 0
	v_lshlrev_b64_e32 v[71:72], 2, v[71:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v71, vcc_lo, v6, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v72, null, v5, v72, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v73, v73, v74
	global_store_b32 v[71:72], v73, off
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v72, 6, v2
	v_add_nc_u32_e32 v71, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v72
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_lshlrev_b64_e32 v[73:74], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v73, vcc_lo, s4, v73
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, s5, v74, vcc_lo
	global_load_b32 v74, v[73:74], off offset:24
	v_and_b32_e32 v73, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v73, v73, 2, v4
	ds_load_b32 v75, v73 offset:768
	v_mad_co_u64_u32 v[72:73], null, v72, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v73, 0 :: v_dual_mul_f32 v74, v74, v75
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v72, vcc_lo, v6, v72
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, v5, v73, vcc_lo
	global_store_b32 v[72:73], v74, off
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v73, 7, v2
	v_add_nc_u32_e32 v72, 7, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v73
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_lshlrev_b64_e32 v[74:75], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v74, vcc_lo, s4, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, s5, v75, vcc_lo
	global_load_b32 v75, v[74:75], off offset:28
	v_and_b32_e32 v74, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v74, v74, 2, v4
	ds_load_b32 v76, v74 offset:896
	v_mad_co_u64_u32 v[73:74], null, v73, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v74, 0 :: v_dual_mul_f32 v75, v75, v76
	v_lshlrev_b64_e32 v[73:74], 2, v[73:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v73, vcc_lo, v6, v73
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, v5, v74, vcc_lo
	global_store_b32 v[73:74], v75, off
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v74, 8, v2
	v_add_nc_u32_e32 v73, 8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v74
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_lshlrev_b64_e32 v[75:76], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v75, vcc_lo, s4, v75
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, s5, v76, vcc_lo
	global_load_b32 v76, v[75:76], off offset:32
	v_and_b32_e32 v75, 31, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v75, v75, 2, v4
	ds_load_b32 v77, v75 offset:1024
	v_mad_co_u64_u32 v[74:75], null, v74, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v75, 0 :: v_dual_mul_f32 v76, v76, v77
	v_lshlrev_b64_e32 v[74:75], 2, v[74:75]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v74, vcc_lo, v6, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v5, v75, vcc_lo
	global_store_b32 v[74:75], v76, off
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v75, 9, v2
	v_add_nc_u32_e32 v74, 9, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v75
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_lshlrev_b64_e32 v[76:77], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v76, vcc_lo, s4, v76
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s5, v77, vcc_lo
	global_load_b32 v77, v[76:77], off offset:36
	v_and_b32_e32 v76, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v76, v76, 2, v4
	ds_load_b32 v78, v76 offset:1152
	v_mad_co_u64_u32 v[75:76], null, v75, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v76, 0 :: v_dual_mul_f32 v77, v77, v78
	v_lshlrev_b64_e32 v[75:76], 2, v[75:76]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v75, vcc_lo, v6, v75
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, v5, v76, vcc_lo
	global_store_b32 v[75:76], v77, off
.LBB0_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v76, 10, v2
	v_add_nc_u32_e32 v75, 10, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v76
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_lshlrev_b64_e32 v[77:78], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v77, vcc_lo, s4, v77
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, s5, v78, vcc_lo
	global_load_b32 v78, v[77:78], off offset:40
	v_and_b32_e32 v77, 31, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v77, v77, 2, v4
	ds_load_b32 v79, v77 offset:1280
	v_mad_co_u64_u32 v[76:77], null, v76, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v77, 0 :: v_dual_mul_f32 v78, v78, v79
	v_lshlrev_b64_e32 v[76:77], 2, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v76, vcc_lo, v6, v76
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, v5, v77, vcc_lo
	global_store_b32 v[76:77], v78, off
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v77, 11, v2
	v_add_nc_u32_e32 v76, 11, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v77
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_lshlrev_b64_e32 v[78:79], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v78, vcc_lo, s4, v78
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, s5, v79, vcc_lo
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
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v78, 12, v2
	v_add_nc_u32_e32 v77, 12, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v78
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_lshlrev_b64_e32 v[79:80], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v79, vcc_lo, s4, v79
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v80, null, s5, v80, vcc_lo
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
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v79, 13, v2
	v_add_nc_u32_e32 v78, 13, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v79
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_lshlrev_b64_e32 v[80:81], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v80, vcc_lo, s4, v80
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, s5, v81, vcc_lo
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
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v80, 14, v2
	v_add_nc_u32_e32 v79, 14, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v80
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_lshlrev_b64_e32 v[81:82], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v81, vcc_lo, s4, v81
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, s5, v82, vcc_lo
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
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v80, 15, v2
	v_add_nc_u32_e32 v0, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s11, v80
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_lshlrev_b64_e32 v[81:82], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v81, vcc_lo, s4, v81
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, s5, v82, vcc_lo
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
.LBB0_54:                               ; %.preheader479.1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v52, v53 offset1:1
	ds_store_2addr_b32 v8, v54, v55 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v56, v57 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v58, v59 offset0:6 offset1:7
	ds_store_b32 v9, v44
	ds_store_b32 v10, v45
	ds_store_b32 v11, v46
	ds_store_b32 v61, v47
	ds_store_b32 v62, v48
	ds_store_b32 v68, v49
	ds_store_b32 v69, v50
	ds_store_b32 v70, v51
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v44, 16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	ds_load_b32 v47, v60
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:64
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v45, 0 :: v_dual_mul_f32 v46, v46, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	global_store_b32 v[44:45], v46, off
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 17, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:68
	v_and_b32_e32 v45, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:128
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v45, 0 :: v_dual_mul_f32 v46, v46, v47
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	global_store_b32 v[44:45], v46, off
.LBB0_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 18, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:72
	v_and_b32_e32 v45, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:256
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 19, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:76
	v_and_b32_e32 v45, 31, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:384
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 20, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:80
	v_and_b32_e32 v45, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:512
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 21, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:84
	v_and_b32_e32 v45, 31, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:640
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v45, 0 :: v_dual_mul_f32 v46, v46, v47
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	global_store_b32 v[44:45], v46, off
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 22, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:88
	v_and_b32_e32 v45, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:768
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v45, 0 :: v_dual_mul_f32 v46, v46, v47
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	global_store_b32 v[44:45], v46, off
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 23, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:92
	v_and_b32_e32 v45, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:896
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 24, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:96
	v_and_b32_e32 v45, 31, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:1024
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 25, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:100
	v_and_b32_e32 v45, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:1152
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 26, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:104
	v_and_b32_e32 v45, 31, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:1280
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v45, 0 :: v_dual_mul_f32 v46, v46, v47
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	global_store_b32 v[44:45], v46, off
.LBB0_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 27, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_78
; %bb.77:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:108
	v_and_b32_e32 v45, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:1408
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 28, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:112
	v_and_b32_e32 v45, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:1536
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 29, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:116
	v_and_b32_e32 v45, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:1664
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 30, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:120
	v_and_b32_e32 v45, 31, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:1792
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v45, 0 :: v_dual_mul_f32 v46, v46, v47
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	global_store_b32 v[44:45], v46, off
.LBB0_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v44, 31, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v44
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_lshlrev_b64_e32 v[45:46], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v45, vcc_lo, s4, v45
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, s5, v46, vcc_lo
	global_load_b32 v46, v[45:46], off offset:124
	v_and_b32_e32 v45, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v45, v45, 2, v4
	ds_load_b32 v47, v45 offset:1920
	v_mad_co_u64_u32 v[44:45], null, v44, v7, v[1:2]
	v_mov_b32_e32 v45, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v44, vcc_lo, v6, v44
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v45, null, v5, v45, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v46, v46, v47
	global_store_b32 v[44:45], v46, off
.LBB0_86:                               ; %.preheader479.2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v36, v37 offset1:1
	ds_store_2addr_b32 v8, v38, v39 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v40, v41 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v42, v43 offset0:6 offset1:7
	ds_store_b32 v9, v28
	ds_store_b32 v10, v29
	ds_store_b32 v11, v30
	ds_store_b32 v61, v31
	ds_store_b32 v62, v32
	ds_store_b32 v68, v33
	ds_store_b32 v69, v34
	ds_store_b32 v70, v35
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v28, 32, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	ds_load_b32 v31, v60
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:128
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v29, 0 :: v_dual_mul_f32 v30, v30, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	global_store_b32 v[28:29], v30, off
.LBB0_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 33, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:132
	v_and_b32_e32 v29, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:128
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v29, 0 :: v_dual_mul_f32 v30, v30, v31
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	global_store_b32 v[28:29], v30, off
.LBB0_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 34, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_92
; %bb.91:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:136
	v_and_b32_e32 v29, 31, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:256
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 35, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:140
	v_and_b32_e32 v29, 31, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:384
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 36, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_96
; %bb.95:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:144
	v_and_b32_e32 v29, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:512
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 37, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:148
	v_and_b32_e32 v29, 31, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:640
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v29, 0 :: v_dual_mul_f32 v30, v30, v31
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	global_store_b32 v[28:29], v30, off
.LBB0_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 38, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:152
	v_and_b32_e32 v29, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:768
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v29, 0 :: v_dual_mul_f32 v30, v30, v31
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	global_store_b32 v[28:29], v30, off
.LBB0_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 39, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:156
	v_and_b32_e32 v29, 31, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:896
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 40, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:160
	v_and_b32_e32 v29, 31, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1024
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 41, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:164
	v_and_b32_e32 v29, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1152
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 42, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:168
	v_and_b32_e32 v29, 31, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1280
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v29, 0 :: v_dual_mul_f32 v30, v30, v31
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	global_store_b32 v[28:29], v30, off
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 43, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:172
	v_and_b32_e32 v29, 31, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1408
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 44, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:176
	v_and_b32_e32 v29, 31, v77
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1536
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 45, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_114
; %bb.113:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:180
	v_and_b32_e32 v29, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1664
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 46, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_116
; %bb.115:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:184
	v_and_b32_e32 v29, 31, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1792
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v29, 0 :: v_dual_mul_f32 v30, v30, v31
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	global_store_b32 v[28:29], v30, off
.LBB0_116:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v28, 47, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v28
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_lshlrev_b64_e32 v[29:30], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, s4, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s5, v30, vcc_lo
	global_load_b32 v30, v[29:30], off offset:188
	v_and_b32_e32 v29, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v29, v29, 2, v4
	ds_load_b32 v31, v29 offset:1920
	v_mad_co_u64_u32 v[28:29], null, v28, v7, v[1:2]
	v_mov_b32_e32 v29, 0
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v6, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v5, v29, vcc_lo
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v30, v30, v31
	global_store_b32 v[28:29], v30, off
.LBB0_118:                              ; %.preheader479.3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v20, v21 offset1:1
	ds_store_2addr_b32 v8, v22, v23 offset0:2 offset1:3
	ds_store_2addr_b32 v8, v24, v25 offset0:4 offset1:5
	ds_store_2addr_b32 v8, v26, v27 offset0:6 offset1:7
	ds_store_b32 v9, v12
	ds_store_b32 v10, v13
	ds_store_b32 v11, v14
	ds_store_b32 v61, v15
	ds_store_b32 v62, v16
	ds_store_b32 v68, v17
	ds_store_b32 v69, v18
	ds_store_b32 v70, v19
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v8, 48, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	ds_load_b32 v11, v60
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
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
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 49, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_122
; %bb.121:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:196
	v_and_b32_e32 v9, 31, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:128
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mul_f32 v10, v10, v11
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	global_store_b32 v[8:9], v10, off
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 50, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_124
; %bb.123:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:200
	v_and_b32_e32 v9, 31, v64
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
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 51, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_126
; %bb.125:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:204
	v_and_b32_e32 v9, 31, v65
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
.LBB0_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 52, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_128
; %bb.127:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:208
	v_and_b32_e32 v9, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:512
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
	v_or_b32_e32 v8, 53, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_130
; %bb.129:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:212
	v_and_b32_e32 v9, 31, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:640
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mul_f32 v10, v10, v11
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	global_store_b32 v[8:9], v10, off
.LBB0_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 54, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_132
; %bb.131:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:216
	v_and_b32_e32 v9, 31, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:768
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mul_f32 v10, v10, v11
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	global_store_b32 v[8:9], v10, off
.LBB0_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 55, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_134
; %bb.133:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:220
	v_and_b32_e32 v9, 31, v72
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
.LBB0_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 56, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_136
; %bb.135:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:224
	v_and_b32_e32 v9, 31, v73
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
.LBB0_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 57, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_138
; %bb.137:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:228
	v_and_b32_e32 v9, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1152
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
	v_or_b32_e32 v8, 58, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_140
; %bb.139:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
	global_load_b32 v10, v[9:10], off offset:232
	v_and_b32_e32 v9, 31, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v9, v9, 2, v4
	ds_load_b32 v11, v9 offset:1280
	v_mad_co_u64_u32 v[8:9], null, v8, v7, v[1:2]
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v9, 0 :: v_dual_mul_f32 v10, v10, v11
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v5, v9, vcc_lo
	global_store_b32 v[8:9], v10, off
.LBB0_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 59, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
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
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 60, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
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
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 61, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_146
; %bb.145:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
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
.LBB0_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 62, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s2, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_148
; %bb.147:
	v_lshlrev_b64_e32 v[9:10], 2, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s4, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s5, v10, vcc_lo
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
.LBB0_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 63, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s11, v8
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_150
; %bb.149:
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v0, v0, 2, v4
	v_add_co_u32 v2, vcc_lo, s4, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, s5, v3, vcc_lo
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
	.size	gate_symfold, .Lfunc_end0-gate_symfold
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gate_symfold
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
		.amdhsa_next_free_vgpr 186
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gate_symfold)<<4)&4080)>>4
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
	.set .Lgate_symfold.num_vgpr, 186
	.set .Lgate_symfold.num_agpr, 0
	.set .Lgate_symfold.numbered_sgpr, 30
	.set .Lgate_symfold.num_named_barrier, 0
	.set .Lgate_symfold.private_seg_size, 0
	.set .Lgate_symfold.uses_vcc, 1
	.set .Lgate_symfold.uses_flat_scratch, 0
	.set .Lgate_symfold.has_dyn_sized_stack, 0
	.set .Lgate_symfold.has_recursion, 0
	.set .Lgate_symfold.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16192
; TotalNumSgprs: 32
; NumVgprs: 186
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 186
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
	.type	__hip_cuid_45f68a788a77cfd2,@object ; @__hip_cuid_45f68a788a77cfd2
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_45f68a788a77cfd2
__hip_cuid_45f68a788a77cfd2:
	.byte	0                               ; 0x0
	.size	__hip_cuid_45f68a788a77cfd2, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_45f68a788a77cfd2
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
    .name:           gate_symfold
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         gate_symfold.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     186
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
