	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gemm_up_silu_mq4g256v2_iu4_gfx1201 ; -- Begin function gemm_up_silu_mq4g256v2_iu4_gfx1201
	.globl	gemm_up_silu_mq4g256v2_iu4_gfx1201
	.p2align	8
	.type	gemm_up_silu_mq4g256v2_iu4_gfx1201,@function
gemm_up_silu_mq4g256v2_iu4_gfx1201:     ; @gemm_up_silu_mq4g256v2_iu4_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[24:26], s[0:1], 0x38
	s_lshl_b32 s30, ttmp9, 8
	s_lshl_b32 s5, ttmp7, 6
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s30, s24
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s5, s26
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_256
; %bb.1:                                ; %.preheader914.i
	s_clause 0x1
	s_load_b256 s[16:23], s[0:1], 0x0
	s_load_b256 s[8:15], s[0:1], 0x20
	v_cmp_gt_u32_e64 s0, 0x80, v0
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v20, 1, v0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_5
; %bb.2:                                ; %.lr.ph.i
	v_dual_mov_b32 v3, 0 :: v_dual_and_b32 v2, 1, v0
	v_or_b32_e32 v4, s5, v1
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v2, 2, v2
	v_cmpx_gt_i32_e64 s26, v4
	s_cbranch_execz .LBB0_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[3:4], null, 0x48, v4, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc_lo, v3, v2
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	global_load_b32 v3, v[3:4], off
.LBB0_4:                                ; %.preheader909.loopexit.i
	s_or_b32 exec_lo, exec_lo, s2
	v_lshlrev_b32_e32 v4, 3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v2, 0, v4, v2
	s_wait_loadcnt 0x0
	ds_store_b32 v2, v3 offset:18432
.LBB0_5:                                ; %Flow1105
	s_or_b32 exec_lo, exec_lo, s1
	s_ashr_i32 s1, s25, 31
	s_add_co_i32 s3, s24, -1
	s_lshr_b32 s1, s1, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s25, s1
	s_ashr_i32 s31, s1, 8
	s_mov_b32 s1, exec_lo
	s_mul_i32 s6, s31, 0x88
	v_cmpx_gt_u32_e32 0x200, v0
	s_cbranch_execz .LBB0_8
; %bb.6:                                ; %.lr.ph926.i
	v_and_b32_e32 v3, 1, v0
	v_add_nc_u32_e32 v5, s30, v1
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v2, 2, v3
	v_lshlrev_b32_e32 v3, 4, v3
	v_lshl_or_b32 v4, v1, 3, v2
	v_or_b32_e32 v2, 0xffffff00, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v4, 0, v4
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	v_min_i32_e32 v6, s3, v5
	v_cmp_gt_i32_e32 vcc_lo, s24, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v2, s4, 0x100, v2
	s_xor_b32 s4, s4, -1
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[6:7], null, v6, s6, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, exec_lo, s4
	v_add_nc_u32_e32 v5, 0x80, v5
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s4, s2
	global_load_b32 v6, v[6:7], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v6, v3, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v6, v6.l
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	ds_store_b32 v4, v6 offset:19456
	v_add_nc_u32_e32 v4, 0x400, v4
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_7
.LBB0_8:                                ; %Flow1104
	s_or_b32 exec_lo, exec_lo, s1
	v_lshrrev_b32_e32 v14, 2, v0
	v_dual_mov_b32 v67, 0 :: v_dual_and_b32 v2, 3, v0
	s_add_co_i32 s2, s26, -1
	v_lshrrev_b32_e32 v193, 5, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v3, s30, v14
	v_add_nc_u32_e32 v15, s5, v14
	v_lshlrev_b32_e32 v2, 3, v2
	v_bfe_u32 v16, v0, 1, 1
	v_or_b32_e32 v18, 16, v193
	v_add_nc_u32_e32 v4, 64, v3
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v7, s2, v15
	v_add_nc_u32_e32 v5, 0x80, v3
	v_add_nc_u32_e32 v6, 0xc0, v3
	v_min_i32_e32 v8, s3, v3
	v_min_i32_e32 v9, s3, v4
	v_mad_co_u64_u32 v[3:4], null, 0x48, v7, v[2:3]
	v_min_i32_e32 v5, s3, v5
	v_min_i32_e32 v6, s3, v6
	v_or_b32_e32 v19, 24, v193
	v_and_or_b32 v17, v193, 6, v16
	v_and_or_b32 v18, v18, 22, v16
	v_cmp_gt_i32_e64 s1, s26, v15
	v_and_b32_e32 v192, 31, v0
	v_mad_co_u64_u32 v[71:72], null, s6, v8, v[2:3]
	v_mad_co_u64_u32 v[72:73], null, s6, v9, v[2:3]
	v_mad_co_u64_u32 v[73:74], null, s6, v5, v[2:3]
	v_mad_co_u64_u32 v[74:75], null, s6, v6, v[2:3]
	v_lshlrev_b32_e32 v2, 4, v0
	s_wait_kmcnt 0x0
	v_add_co_u32 v75, s4, s18, v3
	global_load_b64 v[4:5], v3, s[18:19] offset:8
	s_clause 0x3
	global_load_b64 v[6:7], v71, s[16:17] offset:8
	global_load_b64 v[8:9], v72, s[16:17] offset:8
	global_load_b64 v[10:11], v73, s[16:17] offset:8
	global_load_b64 v[12:13], v74, s[16:17] offset:8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v76, null, s19, 0, s4
	v_and_b32_e32 v2, 16, v2
	s_mov_b32 s4, -1
	s_cmp_gt_i32 s25, 0xff
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v2, v14, 15, v2
	v_or_b32_e32 v14, 8, v193
	v_lshlrev_b32_e32 v2, 3, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_and_or_b32 v14, v14, 14, v16
	v_and_or_b32 v16, v19, 30, v16
	v_lshl_or_b32 v17, v17, 8, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v14, v14, 8, v2
	v_lshl_or_b32 v15, v18, 8, v2
	v_lshl_or_b32 v2, v16, 8, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v198, 0, v17
	v_add_nc_u32_e32 v199, 0, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v200, 0, v15
	v_add_nc_u32_e32 v201, 0, v2
	s_wait_loadcnt 0x4
	v_cndmask_b32_e64 v5, 0, v5, s1
	v_cndmask_b32_e64 v4, 0, v4, s1
	s_wait_loadcnt 0x3
	ds_store_2addr_stride64_b64 v198, v[4:5], v[6:7] offset1:4
	s_wait_loadcnt 0x2
	ds_store_b64 v199, v[8:9] offset:2048
	s_wait_loadcnt 0x1
	ds_store_b64 v200, v[10:11] offset:2048
	s_wait_loadcnt 0x0
	ds_store_b64 v201, v[12:13] offset:2048
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_10
; %bb.9:                                ; %._crit_edge..preheader903_crit_edge.i
	v_and_b32_e32 v2, 1, v0
	s_ashr_i32 s27, s26, 31
	s_mov_b32 s4, 0
	s_branch .LBB0_11
.LBB0_10:
                                        ; implicit-def: $vgpr2
.LBB0_11:                               ; %Flow1101
	v_dual_mov_b32 v129, 0 :: v_dual_and_b32 v196, 0xe0, v0
	v_dual_mov_b32 v70, 0 :: v_dual_lshlrev_b32 v195, 3, v192
	v_bfe_u32 v197, v0, 4, 1
	v_dual_mov_b32 v131, 0 :: v_dual_lshlrev_b32 v194, 10, v193
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v179, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v181, 0
	v_dual_mov_b32 v178, 0 :: v_dual_mov_b32 v183, 0
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v185, 0
	v_dual_mov_b32 v182, 0 :: v_dual_mov_b32 v187, 0
	v_dual_mov_b32 v184, 0 :: v_dual_mov_b32 v189, 0
	v_dual_mov_b32 v186, 0 :: v_dual_mov_b32 v175, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v190, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_23
; %bb.12:                               ; %.preheader908.lr.ph.i
	v_dual_mov_b32 v175, 0 :: v_dual_lshlrev_b32 v2, 3, v0
	v_or_b32_e32 v3, s5, v1
	v_or_b32_e32 v6, 0x80, v1
	v_dual_mov_b32 v190, 0 :: v_dual_add_nc_u32 v7, s30, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v189, 0 :: v_dual_and_b32 v2, 0x78, v2
	v_min_i32_e32 v5, s2, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v185, 0 :: v_dual_add_nc_u32 v8, s30, v6
	v_cmp_gt_i32_e64 s2, s26, v3
	v_dual_mov_b32 v187, 0 :: v_dual_add_nc_u32 v202, 0, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mad_co_i64_i32 v[2:3], null, 0x48, v5, s[18:19]
	v_min_i32_e32 v9, s3, v7
	v_dual_mov_b32 v186, 0 :: v_dual_lshlrev_b32 v203, 3, v1
	v_min_i32_e32 v1, s3, v8
	v_dual_mov_b32 v188, 0 :: v_dual_lshlrev_b32 v5, 2, v20
	v_mad_co_u64_u32 v[65:66], null, s6, v9, 0
	v_or_b32_e32 v4, v194, v195
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[68:69], null, s6, v1, 0
	v_add_co_u32 v77, vcc_lo, v2, v5
	v_ashrrev_i32_e32 v2, 31, v9
	v_ashrrev_i32_e32 v1, 31, v1
	v_dual_mov_b32 v183, 0 :: v_dual_add_nc_u32 v204, 0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v78, null, 0, v3, vcc_lo
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v3, 6, v197
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v5, 3, v196
	v_mad_co_u64_u32 v[66:67], null, s6, v2, v[66:67]
	v_mad_co_u64_u32 v[69:70], null, s6, v1, v[69:70]
	v_cmp_gt_i32_e64 s3, s24, v7
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v205, 4, v20
	v_cmp_gt_i32_e64 s4, s24, v8
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v206, 3, v6
	v_add3_u32 v207, 0, v3, v5
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v191, 0, v4
	v_dual_mov_b32 v179, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v70, 0
	v_mov_b32_e32 v67, 0
	s_mov_b32 s7, 0
	s_ashr_i32 s27, s26, 31
	s_movk_i32 s25, 0x4c00
	s_movk_i32 s33, 0x4800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s14, s7
	s_branch .LBB0_14
.LBB0_13:                               ;   in Loop: Header=BB0_14 Depth=1
	s_and_b32 vcc_lo, exec_lo, s35
	s_mov_b32 s14, s34
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_22
.LBB0_14:                               ; %.preheader908.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_17 Depth 2
	s_mov_b32 s15, s7
	s_add_co_i32 s34, s14, 1
	s_mul_u64 s[18:19], s[14:15], 0x88
	s_lshl_b32 s15, s14, 1
	s_cmp_eq_u32 s34, s31
	s_mov_b32 s38, s7
	s_cselect_b32 s35, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[18:19], s[16:17], s[18:19]
	s_mov_b32 s36, -1
	s_mov_b32 s37, s7
	s_branch .LBB0_17
.LBB0_15:                               ; %.loopexit906.loopexit.i
                                        ;   in Loop: Header=BB0_17 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s28
	s_add_co_i32 s6, s37, s14
	s_and_b32 s28, s38, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[28:29], s[6:7], 0x88
	s_cselect_b32 s38, s25, 0x5400
	s_lshl_b32 s6, s37, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[28:29], s[16:17], s[28:29]
	s_xor_b32 s6, s6, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[28:29], s[28:29], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v119, vcc_lo, s28, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v120, null, s29, v66, vcc_lo
	v_add_co_u32 v121, vcc_lo, s28, v68
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v122, null, s29, v69, vcc_lo
	s_clause 0x1
	global_load_b32 v119, v[119:120], off
	global_load_b32 v120, v[121:122], off
	v_add_nc_u32_e32 v121, s38, v204
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v122, v121, v203
	v_add_nc_u32_e32 v121, v121, v206
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v119, v205, v119
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v120, v205, v120
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_f16_e32 v119, v119.l
	v_cvt_f32_f16_e32 v120, v120.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v119, 0, v119, s3
	v_cndmask_b32_e64 v120, 0, v120, s4
	ds_store_b32 v122, v119
	ds_store_b32 v121, v120
.LBB0_16:                               ; %.loopexit906.i
                                        ;   in Loop: Header=BB0_17 Depth=2
	s_wait_loadcnt_dscnt 0x103
	v_dual_mul_f32 v119, v117, v101 :: v_dual_mul_f32 v120, v115, v101
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v102, v102
	v_cvt_f32_i32_e32 v59, v59
	v_mul_f32_e32 v121, v116, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v57, v119, v57 :: v_dual_mul_f32 v58, v120, v58
	v_dual_mul_f32 v119, v118, v101 :: v_dual_mul_f32 v120, v113, v101
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_cvt_f32_i32_e32 v62, v62
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v59, v120, v59 :: v_dual_mul_f32 v120, v114, v101
	v_dual_fmac_f32 v57, v119, v102 :: v_dual_fmac_f32 v58, v121, v102
	v_mul_f32_e32 v119, v111, v101
	v_cvt_f32_i32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v59, v120, v102
	v_dual_add_f32 v175, v175, v57 :: v_dual_add_f32 v190, v190, v58
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v119, v60 :: v_dual_mul_f32 v60, v112, v101
	v_dual_mul_f32 v119, v107, v101 :: v_dual_mul_f32 v120, v109, v101
	v_cvt_f32_i32_e32 v50, v50
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	v_mul_f32_e32 v58, v119, v61
	v_add_f32_e32 v188, v188, v59
	v_dual_fmac_f32 v57, v60, v102 :: v_dual_mul_f32 v60, v103, v101
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v119, v105, v101
	v_mul_f32_e32 v59, v120, v62
	v_mul_f32_e32 v62, v108, v101
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v54, v54
	v_cvt_f32_i32_e32 v56, v56
	v_cvt_f32_i32_e32 v42, v42
	v_fmac_f32_e32 v58, v62, v102
	v_mul_f32_e32 v62, v119, v64
	v_mul_f32_e32 v60, v60, v61
	v_mul_f32_e32 v61, v104, v101
	v_cvt_f32_i32_e32 v41, v41
	v_add_f32_e32 v186, v186, v58
	s_wait_dscnt 0x2
	v_dual_mul_f32 v63, v110, v101 :: v_dual_mul_f32 v58, v115, v99
	v_fmac_f32_e32 v60, v61, v102
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v59, v63, v102
	v_dual_mul_f32 v63, v106, v101 :: v_dual_add_f32 v184, v184, v60
	v_add_f32_e32 v189, v189, v57
	v_mul_f32_e32 v57, v117, v99
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v187, v187, v59 :: v_dual_mul_f32 v50, v58, v50
	v_fmac_f32_e32 v62, v63, v102
	v_cvt_f32_i32_e32 v59, v100
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v49, v57, v49
	v_dual_mul_f32 v57, v118, v99 :: v_dual_mul_f32 v58, v113, v99
	v_dual_add_f32 v185, v185, v62 :: v_dual_mul_f32 v60, v116, v99
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v51, v58, v51
	v_mul_f32_e32 v58, v114, v99
	v_dual_fmac_f32 v49, v57, v59 :: v_dual_fmac_f32 v50, v60, v59
	v_mul_f32_e32 v57, v111, v99
	v_cvt_f32_i32_e32 v48, v48
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v173, v173, v49 :: v_dual_add_f32 v174, v174, v50
	v_dual_mul_f32 v49, v57, v52 :: v_dual_mul_f32 v52, v112, v99
	v_mul_f32_e32 v57, v107, v99
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	v_dual_mul_f32 v50, v57, v53 :: v_dual_fmac_f32 v49, v52, v59
	v_mul_f32_e32 v52, v103, v99
	v_cvt_f32_i32_e32 v53, v55
	v_fmac_f32_e32 v51, v58, v59
	v_mul_f32_e32 v58, v109, v99
	v_mul_f32_e32 v57, v105, v99
	v_cvt_f32_i32_e32 v38, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v52, v52, v53 :: v_dual_add_f32 v171, v171, v51
	v_dual_mul_f32 v51, v58, v54 :: v_dual_mul_f32 v54, v108, v99
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_fmac_f32_e32 v50, v54, v59
	v_dual_mul_f32 v55, v110, v99 :: v_dual_add_f32 v172, v172, v49
	s_wait_dscnt 0x1
	v_mul_f32_e32 v49, v117, v93
	v_cvt_f32_i32_e32 v27, v27
	v_dual_add_f32 v169, v169, v50 :: v_dual_mul_f32 v50, v115, v93
	v_dual_mul_f32 v53, v104, v99 :: v_dual_mul_f32 v54, v57, v56
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v42, v50, v42 :: v_dual_fmac_f32 v51, v55, v59
	v_fmac_f32_e32 v52, v53, v59
	v_dual_mul_f32 v55, v106, v99 :: v_dual_mul_f32 v50, v113, v93
	v_cvt_f32_i32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v170, v170, v51 :: v_dual_add_f32 v167, v167, v52
	v_fmac_f32_e32 v54, v55, v59
	v_cvt_f32_i32_e32 v51, v94
	v_mul_f32_e32 v52, v116, v93
	v_mul_f32_e32 v41, v49, v41
	v_mul_f32_e32 v49, v118, v93
	v_mul_f32_e32 v43, v50, v43
	v_mul_f32_e32 v50, v114, v93
	v_fmac_f32_e32 v42, v52, v51
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v41, v49, v51
	v_mul_f32_e32 v49, v111, v93
	v_dual_fmac_f32 v43, v50, v51 :: v_dual_mul_f32 v50, v109, v93
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v158, v158, v42 :: v_dual_add_f32 v157, v157, v41
	v_dual_mul_f32 v41, v49, v44 :: v_dual_mul_f32 v44, v112, v93
	v_mul_f32_e32 v49, v107, v93
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v42, v49, v45
	v_cvt_f32_i32_e32 v45, v47
	v_dual_fmac_f32 v41, v44, v51 :: v_dual_mul_f32 v44, v103, v93
	v_mul_f32_e32 v49, v105, v93
	v_add_f32_e32 v155, v155, v43
	v_mul_f32_e32 v47, v110, v93
	v_cvt_f32_i32_e32 v19, v19
	v_mul_f32_e32 v44, v44, v45
	v_mul_f32_e32 v45, v104, v93
	v_dual_mul_f32 v43, v50, v46 :: v_dual_mul_f32 v46, v108, v93
	s_wait_dscnt 0x0
	v_dual_add_f32 v156, v156, v41 :: v_dual_mul_f32 v41, v117, v79
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v44, v45, v51
	v_cvt_f32_i32_e32 v22, v22
	v_fmac_f32_e32 v42, v46, v51
	v_mul_f32_e32 v46, v49, v48
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v12, v12
	v_dual_add_f32 v153, v153, v42 :: v_dual_mul_f32 v42, v115, v79
	v_fmac_f32_e32 v43, v47, v51
	v_dual_add_f32 v151, v151, v44 :: v_dual_mul_f32 v44, v116, v79
	v_mul_f32_e32 v33, v41, v33
	v_mul_f32_e32 v41, v118, v79
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v154, v154, v43
	v_cvt_f32_i32_e32 v43, v80
	v_mul_f32_e32 v34, v42, v34
	v_mul_f32_e32 v42, v113, v79
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_fmac_f32_e32 v34, v44, v43
	v_mul_f32_e32 v35, v42, v35
	v_mul_f32_e32 v42, v114, v79
	v_fmac_f32_e32 v33, v41, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v111, v79 :: v_dual_add_f32 v142, v142, v34
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v5, v5
	v_add_f32_e32 v141, v141, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v33, v41, v36 :: v_dual_mul_f32 v36, v112, v79
	v_mul_f32_e32 v41, v107, v79
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v3, v3
	v_dual_mul_f32 v34, v41, v37 :: v_dual_fmac_f32 v33, v36, v43
	v_dual_mul_f32 v36, v108, v79 :: v_dual_mul_f32 v41, v105, v79
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v8, v8
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v34, v36, v43 :: v_dual_fmac_f32 v35, v42, v43
	v_add_f32_e32 v140, v140, v33
	v_mul_f32_e32 v42, v109, v79
	s_barrier_signal -1
	v_mul_f32_e32 v47, v106, v93
	v_dual_add_f32 v139, v139, v35 :: v_dual_add_f32 v168, v168, v54
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v35, v42, v38 :: v_dual_mul_f32 v38, v103, v79
	v_fmac_f32_e32 v46, v47, v51
	s_xor_b32 s6, s36, -1
	s_mov_b32 s37, 1
	s_mov_b32 s36, 0
	v_mul_f32_e32 v33, v38, v39
	v_mul_f32_e32 v38, v106, v79
	v_mul_f32_e32 v36, v41, v40
	v_dual_mul_f32 v39, v101, v97 :: v_dual_mul_f32 v40, v101, v95
	v_dual_mul_f32 v37, v110, v79 :: v_dual_add_f32 v152, v152, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v36, v38, v43
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s6
	v_dual_mul_f32 v26, v40, v26 :: v_dual_fmac_f32 v35, v37, v43
	v_add_f32_e32 v137, v137, v34
	v_dual_mul_f32 v25, v39, v25 :: v_dual_mul_f32 v34, v101, v98
	v_add_f32_e32 v136, v136, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v138, v138, v35
	s_mov_b32 s38, -1
	s_barrier_wait -1
	v_fmac_f32_e32 v25, v34, v102
	v_mul_f32_e32 v37, v104, v79
	v_mul_f32_e32 v34, v101, v91
	global_inv scope:SCOPE_SE
	v_dual_add_f32 v182, v182, v25 :: v_dual_fmac_f32 v33, v37, v43
	v_mul_f32_e32 v37, v101, v96
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v135, v135, v33 :: v_dual_fmac_f32 v26, v37, v102
	v_mul_f32_e32 v33, v101, v89
	v_add_f32_e32 v183, v183, v26
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v25, v33, v27
	v_dual_mul_f32 v26, v101, v90 :: v_dual_mul_f32 v27, v34, v28
	v_dual_mul_f32 v28, v101, v87 :: v_dual_mul_f32 v33, v101, v92
	v_dual_mul_f32 v28, v28, v29 :: v_dual_mul_f32 v29, v101, v88
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v25, v26, v102 :: v_dual_mul_f32 v26, v101, v85
	v_fmac_f32_e32 v28, v29, v102
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v180, v180, v25
	v_mul_f32_e32 v29, v101, v81
	v_mul_f32_e32 v25, v26, v30
	v_mul_f32_e32 v26, v101, v86
	v_mul_f32_e32 v30, v101, v83
	v_dual_fmac_f32 v27, v33, v102 :: v_dual_add_f32 v178, v178, v28
	v_mul_f32_e32 v28, v101, v82
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v25, v26, v102 :: v_dual_mul_f32 v26, v29, v31
	v_add_f32_e32 v181, v181, v27
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v29, v101, v84 :: v_dual_fmac_f32 v26, v28, v102
	v_dual_mul_f32 v31, v99, v95 :: v_dual_mul_f32 v28, v99, v96
	v_dual_add_f32 v176, v176, v26 :: v_dual_mul_f32 v27, v30, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v18, v31, v18
	v_mul_f32_e32 v30, v99, v97
	v_add_f32_e32 v179, v179, v25
	v_mul_f32_e32 v26, v99, v92
	v_dual_fmac_f32 v18, v28, v59 :: v_dual_mul_f32 v17, v30, v17
	v_dual_mul_f32 v30, v99, v91 :: v_dual_fmac_f32 v27, v29, v102
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v29, v99, v89 :: v_dual_add_f32 v166, v166, v18
	v_cvt_f32_i32_e32 v18, v21
	v_dual_mul_f32 v20, v30, v20 :: v_dual_mul_f32 v25, v99, v98
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v19, v29, v19
	v_mul_f32_e32 v21, v99, v85
	v_add_f32_e32 v177, v177, v27
	v_dual_fmac_f32 v20, v26, v59 :: v_dual_fmac_f32 v17, v25, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v25, v99, v90 :: v_dual_add_f32 v164, v164, v20
	v_add_f32_e32 v165, v165, v17
	v_dual_mul_f32 v17, v99, v87 :: v_dual_mul_f32 v20, v99, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v17, v17, v18 :: v_dual_mul_f32 v18, v99, v88
	v_fmac_f32_e32 v19, v25, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_add_f32_e32 v163, v163, v19
	v_mul_f32_e32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v22, v99, v86
	v_cvt_f32_i32_e32 v23, v24
	v_dual_mul_f32 v20, v20, v21 :: v_dual_mul_f32 v21, v99, v82
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v20, v21, v59 :: v_dual_fmac_f32 v19, v22, v59
	v_dual_mul_f32 v22, v93, v95 :: v_dual_mul_f32 v21, v93, v97
	v_mul_f32_e32 v10, v22, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v162, v162, v19 :: v_dual_mul_f32 v19, v93, v96
	v_fmac_f32_e32 v10, v19, v51
	v_fmac_f32_e32 v17, v18, v59
	v_dual_mul_f32 v18, v99, v83 :: v_dual_mul_f32 v9, v21, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v21, v93, v91 :: v_dual_add_f32 v150, v150, v10
	v_add_f32_e32 v161, v161, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v99, v84
	v_mul_f32_e32 v12, v21, v12
	v_dual_mul_f32 v19, v93, v87 :: v_dual_mul_f32 v10, v93, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v93, v98
	v_dual_fmac_f32 v9, v18, v51 :: v_dual_mul_f32 v18, v93, v92
	v_dual_add_f32 v159, v159, v20 :: v_dual_mul_f32 v20, v93, v89
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v12, v18, v51
	v_dual_mul_f32 v11, v20, v11 :: v_dual_mul_f32 v20, v93, v85
	v_dual_add_f32 v160, v160, v17 :: v_dual_mul_f32 v17, v93, v90
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_f32 v148, v148, v12 :: v_dual_add_f32 v149, v149, v9
	v_mul_f32_e32 v9, v19, v13
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v93, v88
	v_mul_f32_e32 v12, v93, v83
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v93, v86
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_add_f32_e32 v147, v147, v11
	v_cvt_f32_i32_e32 v11, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v145, v145, v9
	v_dual_mul_f32 v9, v10, v11 :: v_dual_mul_f32 v10, v93, v82
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v10, v51
	v_mul_f32_e32 v10, v79, v95
	v_dual_add_f32 v143, v143, v9 :: v_dual_mul_f32 v2, v10, v2
	v_mul_f32_e32 v9, v79, v96
	v_dual_mul_f32 v11, v12, v14 :: v_dual_mul_f32 v12, v79, v97
	v_mul_f32_e32 v10, v79, v89
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v2, v9, v43 :: v_dual_mul_f32 v9, v79, v92
	v_dual_mul_f32 v1, v12, v1 :: v_dual_mul_f32 v12, v79, v98
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_add_f32 v134, v134, v2 :: v_dual_fmac_f32 v1, v12, v43
	v_mul_f32_e32 v12, v79, v91
	v_fmac_f32_e32 v13, v17, v51
	v_add_f32_e32 v133, v133, v1
	v_mul_f32_e32 v1, v10, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v3, v12, v4 :: v_dual_mul_f32 v4, v79, v90
	v_dual_add_f32 v146, v146, v13 :: v_dual_mul_f32 v13, v93, v84
	v_mul_f32_e32 v10, v79, v87
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v3, v9, v43
	v_mul_f32_e32 v9, v79, v83
	v_dual_fmac_f32 v11, v13, v51 :: v_dual_mul_f32 v2, v10, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v132, v132, v3
	v_fmac_f32_e32 v1, v4, v43
	v_dual_mul_f32 v5, v79, v88 :: v_dual_add_f32 v144, v144, v11
	v_dual_mul_f32 v11, v79, v85 :: v_dual_mul_f32 v10, v79, v86
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v131, v131, v1 :: v_dual_fmac_f32 v2, v5, v43
	v_mul_f32_e32 v4, v11, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v6, v79, v81 :: v_dual_add_f32 v129, v129, v2
	v_fmac_f32_e32 v4, v10, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v6, v6, v7 :: v_dual_mul_f32 v7, v9, v8
	v_dual_mul_f32 v8, v79, v82 :: v_dual_mul_f32 v9, v79, v84
	v_add_f32_e32 v130, v130, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v6, v8, v43 :: v_dual_fmac_f32 v7, v9, v43
	v_dual_add_f32 v70, v70, v6 :: v_dual_add_f32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_13
.LBB0_17:                               ;   Parent Loop BB0_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_b32 s6, s37, s15
	s_lshl_b32 s28, s37, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[6:7], s[26:27]
	s_mov_b32 s29, s7
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[1:2], null, 0x48, s40, v[75:76]
	s_add_nc_u64 s[42:43], s[18:19], s[28:29]
	v_add_nc_u32_e32 v79, 0, v195
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, s29, s42, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s43, 0, s29
	v_add_co_u32 v6, s29, s42, v72
	v_mad_co_u64_u32 v[2:3], null, 0x48, s41, v[2:3]
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s43, 0, s29
	v_add_co_u32 v8, s29, s42, v73
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s43, 0, s29
	v_add_co_u32 v10, s29, s42, v74
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s43, 0, s29
	global_load_b64 v[127:128], v[1:2], off offset:40
	s_clause 0x3
	global_load_b64 v[123:124], v[4:5], off offset:40
	global_load_b64 v[121:122], v[6:7], off offset:40
	global_load_b64 v[119:120], v[8:9], off offset:40
	global_load_b64 v[125:126], v[10:11], off offset:40
	ds_load_2addr_stride64_b64 v[80:83], v191 offset0:4 offset1:5
	ds_load_2addr_stride64_b64 v[1:4], v79 offset1:1
	ds_load_2addr_stride64_b64 v[84:87], v79 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[80:81], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[80:81], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[80:81], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[80:81], v[86:87], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[82:83], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[82:83], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[82:83], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[82:83], v[86:87], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v80, 0x100, v191
	ds_load_2addr_b64 v[81:84], v79 offset0:32 offset1:96
	ds_load_2addr_stride64_b64 v[85:88], v80 offset0:4 offset1:5
	ds_load_2addr_b64 v[89:92], v79 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[85:86], v[81:82], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[85:86], v[83:84], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[85:86], v[89:90], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[85:86], v[91:92], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[87:88], v[81:82], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[87:88], v[83:84], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[87:88], v[89:90], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[87:88], v[91:92], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x4
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v82, 0, v128, s1
	v_cndmask_b32_e64 v81, 0, v127, s1
	s_and_b32 s29, s38, s35
	s_wait_loadcnt 0x3
	ds_store_2addr_stride64_b64 v198, v[123:124], v[81:82] offset0:20 offset1:46
	s_wait_loadcnt 0x2
	ds_store_b64 v199, v[121:122] offset:10240
	s_wait_loadcnt 0x1
	ds_store_b64 v200, v[119:120] offset:10240
	s_wait_loadcnt 0x0
	ds_store_b64 v201, v[125:126] offset:10240
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s29
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_19
; %bb.18:                               ; %.loopexit907.loopexit.i
                                        ;   in Loop: Header=BB0_17 Depth=2
	s_add_co_i32 s40, s6, 1
	s_mov_b32 s41, s7
	s_add_co_i32 s42, s37, s14
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[40:41], s[26:27]
	s_mov_b32 s43, s7
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[81:82], null, 0x48, s40, v[75:76]
	s_mul_u64 s[42:43], s[42:43], 0x88
	s_xor_b32 s44, s28, 64
	s_mov_b32 s45, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[42:43], s[16:17], s[42:43]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[42:43], s[42:43], s[44:45]
	v_mad_co_u64_u32 v[82:83], null, 0x48, s41, v[82:83]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v83, s28, s42, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v84, null, s43, 0, s28
	v_add_co_u32 v85, s28, s42, v72
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v86, null, s43, 0, s28
	v_add_co_u32 v87, s28, s42, v73
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v88, null, s43, 0, s28
	v_add_co_u32 v89, s28, s42, v74
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v90, null, s43, 0, s28
	global_load_b64 v[127:128], v[81:82], off offset:8
	s_clause 0x3
	global_load_b64 v[123:124], v[83:84], off offset:8
	global_load_b64 v[121:122], v[85:86], off offset:8
	global_load_b64 v[119:120], v[87:88], off offset:8
	global_load_b64 v[125:126], v[89:90], off offset:8
.LBB0_19:                               ; %.loopexit907.i
                                        ;   in Loop: Header=BB0_17 Depth=2
	ds_load_2addr_stride64_b64 v[81:84], v191 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[85:88], v79 offset0:46 offset1:47
	ds_load_2addr_stride64_b64 v[89:92], v79 offset0:48 offset1:49
	s_xor_b32 s28, s29, -1
	s_and_b32 s29, s36, exec_lo
	s_cselect_b32 s29, s25, 0x5400
	s_cselect_b32 s39, s33, 0x4a00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[81:82], v[85:86], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[81:82], v[87:88], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[81:82], v[89:90], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[81:82], v[91:92], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[83:84], v[85:86], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[83:84], v[87:88], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[83:84], v[89:90], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[83:84], v[91:92], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v87, 0x100, v79
	ds_load_2addr_stride64_b64 v[79:82], v80 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[83:86], v87 offset0:46 offset1:47
	ds_load_2addr_stride64_b64 v[87:90], v87 offset0:48 offset1:49
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[79:80], v[83:84], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[79:80], v[85:86], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[79:80], v[87:88], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[79:80], v[89:90], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[81:82], v[83:84], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[81:82], v[85:86], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[81:82], v[87:88], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[81:82], v[89:90], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v83, s29, v207
	v_add_nc_u32_e32 v79, s39, v202
	s_and_not1_b32 vcc_lo, exec_lo, s28
	ds_load_2addr_b32 v[117:118], v83 offset1:1
	ds_load_2addr_b32 v[115:116], v83 offset0:2 offset1:3
	ds_load_2addr_b32 v[113:114], v83 offset0:4 offset1:5
	ds_load_2addr_b32 v[111:112], v83 offset0:6 offset1:7
	ds_load_2addr_b32 v[107:108], v83 offset0:8 offset1:9
	ds_load_2addr_b32 v[109:110], v83 offset0:10 offset1:11
	ds_load_2addr_b32 v[103:104], v83 offset0:12 offset1:13
	ds_load_2addr_b32 v[105:106], v83 offset0:14 offset1:15
	ds_load_2addr_b32 v[97:98], v83 offset0:32 offset1:33
	ds_load_2addr_b32 v[95:96], v83 offset0:34 offset1:35
	ds_load_2addr_b32 v[89:90], v83 offset0:36 offset1:37
	ds_load_2addr_b32 v[91:92], v83 offset0:38 offset1:39
	ds_load_2addr_b32 v[87:88], v83 offset0:40 offset1:41
	ds_load_2addr_b32 v[85:86], v83 offset0:42 offset1:43
	ds_load_2addr_b32 v[81:82], v83 offset0:44 offset1:45
	ds_load_2addr_b32 v[83:84], v83 offset0:46 offset1:47
	ds_load_2addr_b32 v[101:102], v79 offset1:1
	ds_load_2addr_b32 v[99:100], v79 offset0:32 offset1:33
	ds_load_2addr_b32 v[93:94], v79 offset0:64 offset1:65
	ds_load_2addr_b32 v[79:80], v79 offset0:96 offset1:97
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_16
; %bb.20:                               ;   in Loop: Header=BB0_17 Depth=2
	s_wait_loadcnt 0x4
	v_cndmask_b32_e64 v128, 0, v128, s1
	v_cndmask_b32_e64 v127, 0, v127, s1
	s_wait_loadcnt 0x3
	ds_store_2addr_stride64_b64 v198, v[127:128], v[123:124] offset1:4
	s_wait_loadcnt 0x2
	ds_store_b64 v199, v[121:122] offset:2048
	s_wait_loadcnt 0x1
	ds_store_b64 v200, v[119:120] offset:2048
	s_wait_loadcnt 0x0
	ds_store_b64 v201, v[125:126] offset:2048
	s_and_saveexec_b32 s28, s0
	s_cbranch_execz .LBB0_15
; %bb.21:                               ;   in Loop: Header=BB0_17 Depth=2
	s_and_b32 s29, s38, exec_lo
	s_cselect_b32 s29, s33, 0x4a00
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[6:7], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[119:120], null, 0x48, s40, v[77:78]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[120:121], null, 0x48, s41, v[120:121]
	global_load_b32 v119, v[119:120], off
	v_add_nc_u32_e32 v120, v204, v203
	v_add_nc_u32_e32 v120, s29, v120
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v119, 0, v119, s2
	ds_store_b32 v120, v119
	s_branch .LBB0_15
.LBB0_22:                               ; %Flow1099
	v_and_b32_e32 v2, 1, v0
.LBB0_23:                               ; %.preheader903.i
	v_lshlrev_b32_e32 v1, 10, v0
	v_lshlrev_b32_e32 v3, 5, v197
	v_lshlrev_b32_e32 v4, 2, v196
	v_and_b32_e32 v5, 8, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 0x1c00, v1
	v_cmp_eq_u32_e64 s2, 0, v5
	v_cmp_ne_u32_e64 s6, 0, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v1, 0, v1
	v_add3_u32 v34, v1, v3, v4
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_25
; %bb.24:                               ; %.preheader902.i
	ds_store_2addr_b32 v34, v175, v190 offset1:1
	ds_store_2addr_b32 v34, v188, v189 offset0:2 offset1:3
	ds_store_2addr_b32 v34, v186, v187 offset0:4 offset1:5
	ds_store_2addr_b32 v34, v184, v185 offset0:6 offset1:7
	ds_store_2addr_b32 v34, v182, v183 offset0:16 offset1:17
	ds_store_2addr_b32 v34, v180, v181 offset0:18 offset1:19
	ds_store_2addr_b32 v34, v178, v179 offset0:20 offset1:21
	ds_store_2addr_b32 v34, v176, v177 offset0:22 offset1:23
.LBB0_25:                               ; %.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v3, s30, v195
	s_ashr_i32 s0, s30, 31
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_lshlrev_b32_e32 v1, 2, v195
	v_ashrrev_i32_e32 v4, 31, v3
	v_sub_co_u32 v5, vcc_lo, v3, s30
	v_cmp_eq_u32_e64 s1, 0, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v7, 0, v1
	s_wait_alu depctr_sa_sdst(0) depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v6, null, s0, v4, vcc_lo
	v_lshlrev_b64_e32 v[27:28], 2, v[3:4]
	v_and_b32_e32 v2, 4, v0
	v_add_co_u32 v21, s0, s8, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[3:4], 2, v[5:6]
	v_and_b32_e32 v5, 2, v0
	v_and_b32_e32 v0, 16, v0
	v_add_co_u32 v25, vcc_lo, s22, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s23, v28, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v22, null, s9, 0, s0
	v_add_co_u32 v23, vcc_lo, s8, v3
	v_add_co_u32 v19, s0, s10, v1
	s_lshl_b32 s14, ttmp9, 1
	v_or_b32_e32 v35, s5, v193
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, s9, v4, vcc_lo
	v_cmp_eq_u32_e64 s5, 0, v5
	v_cmp_eq_u32_e64 s4, 0, v2
	v_cmp_eq_u32_e64 s3, 0, v0
	v_add_co_ci_u32_e64 v20, null, s11, 0, s0
	v_lshrrev_b32_e32 v32, 1, v192
	v_lshlrev_b32_e32 v31, 1, v192
	v_cmp_eq_u32_e64 s0, 0, v192
	v_add_nc_u32_e32 v33, v7, v194
	s_ashr_i32 s15, s14, 31
	s_barrier_wait -1
	s_mul_u64 s[10:11], s[26:27], s[14:15]
	s_mov_b32 s15, exec_lo
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s26, v35
	s_cbranch_execz .LBB0_52
; %bb.26:
	v_mad_co_i64_i32 v[0:1], null, s24, v35, 0
	s_mov_b32 s17, 0
	s_mov_b32 s16, exec_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v0, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v28, vcc_lo
	global_load_b128 v[40:43], v[4:5], off
	global_load_b128 v[0:3], v[25:26], off
	global_load_b128 v[8:11], v[4:5], off offset:16
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v4, 0xbfb8aa3b, v40
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, 0xbfb8aa3b, v40, -v4
	v_rndne_f32_e32 v6, v4
	v_dual_fmac_f32 v5, 0xb2a5705f, v40 :: v_dual_sub_f32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v40
	global_load_b128 v[4:7], v[25:26], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0x7f800000, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v14, 1.0, v12
	v_mul_f32_e32 v12, 0xbfb8aa3b, v41
	v_div_scale_f32 v15, null, v14, v14, v40
	v_div_scale_f32 v29, vcc_lo, v40, v14, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v17, 0xbfb8aa3b, v41, -v12
	v_rndne_f32_e32 v18, v12
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v17, 0xb2a5705f, v41 :: v_dual_sub_f32 v12, v12, v18
	v_cvt_i32_f32_e32 v18, v18
	v_add_f32_e32 v12, v12, v17
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v13, -v15, v16, 1.0
	v_exp_f32_e32 v17, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v13, v16
	v_mul_f32_e32 v30, v29, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_ldexp_f32 v17, v17, v18
	v_fma_f32 v13, -v15, v30, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v13, v16
	ds_load_2addr_b32 v[12:13], v33 offset1:1
	v_fma_f32 v15, -v15, v30, v29
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v15, v16, v30
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v41
	ds_load_2addr_b32 v[44:45], v33 offset0:2 offset1:3
	ds_load_2addr_b32 v[46:47], v33 offset0:4 offset1:5
	ds_load_2addr_b32 v[29:30], v33 offset0:6 offset1:7
	v_div_fixup_f32 v14, v15, v14, v40
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, 0, v17, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v41
	s_wait_dscnt 0x3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v36, v12, v14 :: v_dual_cndmask_b32 v15, 0x7f800000, v16
	v_mul_f32_e32 v16, 0xbfb8aa3b, v42
	s_wait_loadcnt 0x2
	v_div_scale_f32 v14, null, v0, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, 1.0, v15
	v_fma_f32 v37, 0xbfb8aa3b, v42, -v16
	v_rndne_f32_e32 v38, v16
	v_div_scale_f32 v48, vcc_lo, v36, v0, v36
	v_rcp_f32_e32 v17, v14
	v_div_scale_f32 v15, null, v12, v12, v41
	v_fmac_f32_e32 v37, 0xb2a5705f, v42
	v_sub_f32_e32 v16, v16, v38
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v15
	v_add_f32_e32 v16, v16, v37
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v14, v17, 1.0
	v_exp_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v17, v39, v17
	v_fma_f32 v40, -v15, v18, 1.0
	v_div_scale_f32 v39, s7, v41, v12, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v40, v18
	v_dual_mul_f32 v40, v48, v17 :: v_dual_mul_f32 v49, v39, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v37, -v14, v40, v48
	v_fma_f32 v50, -v15, v49, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v40, v37, v17
	v_cvt_i32_f32_e32 v37, v38
	v_fmac_f32_e32 v49, v50, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v14, -v14, v40, v48
	v_ldexp_f32 v16, v16, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v15, -v15, v49, v39
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v14, v17, v40
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v15, v18, v49
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v12, v14, v12, v41
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v16, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v42
	v_mul_f32_e32 v38, v13, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v14, 0x7f800000, v15, vcc_lo
	v_mul_f32_e32 v15, 0xbfb8aa3b, v43
	v_div_scale_f32 v13, null, v1, v1, v38
	v_div_scale_f32 v48, vcc_lo, v38, v1, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v12, 1.0, v14
	v_fma_f32 v18, 0xbfb8aa3b, v43, -v15
	v_rndne_f32_e32 v39, v15
	v_rcp_f32_e32 v16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v14, null, v12, v12, v42
	v_fmac_f32_e32 v18, 0xb2a5705f, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v15, v15, v39
	v_cvt_i32_f32_e32 v39, v39
	v_rcp_f32_e32 v17, v14
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v13, v16, 1.0
	v_dual_add_f32 v15, v15, v18 :: v_dual_fmac_f32 v16, v40, v16
	v_div_scale_f32 v40, s7, v42, v12, v42
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v14, v17, 1.0
	v_exp_f32_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v41, v17
	v_mul_f32_e32 v41, v48, v16
	v_mul_f32_e32 v18, v40, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v49, -v13, v41, v48
	v_ldexp_f32 v15, v15, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v14, v18, v40
	v_fmac_f32_e32 v41, v49, v16
	v_div_fixup_f32 v0, v37, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, v50, v17
	v_fma_f32 v13, -v13, v41, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v14, v18, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v39, v13, v16, v41
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v14, v17, v18
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v12, v13, v12, v42
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v43
	s_wait_dscnt 0x2
	v_mul_f32_e32 v40, v44, v12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, 0xbfb8aa3b, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, 0x7f800000, v14, vcc_lo
	v_div_scale_f32 v42, null, v2, v2, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v16, v12
	v_add_f32_e32 v44, 1.0, v13
	v_fma_f32 v13, 0xbfb8aa3b, v8, -v12
	v_div_scale_f32 v51, vcc_lo, v40, v2, v40
	v_rcp_f32_e32 v49, v42
	v_sub_f32_e32 v12, v12, v16
	v_div_scale_f32 v48, null, v44, v44, v43
	v_fmac_f32_e32 v13, 0xb2a5705f, v8
	v_div_scale_f32 v52, s7, v43, v44, v43
	v_cvt_i32_f32_e32 v56, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v50, v48
	v_add_f32_e32 v17, v12, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v42, v49, 1.0
	v_exp_f32_e32 v55, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v49, v14, v49
	v_fma_f32 v15, -v48, v50, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v53, v51, v49
	v_fmac_f32_e32 v50, v15, v50
	global_load_b128 v[12:15], v[23:24], off offset:4
	v_fma_f32 v18, -v42, v53, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v54, v52, v50 :: v_dual_fmac_f32 v53, v18, v49
	v_div_fixup_f32 v1, v39, v1, v38
	v_fma_f32 v17, -v48, v54, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v42, -v42, v53, v51
	v_ldexp_f32 v51, v55, v56
	v_fmac_f32_e32 v54, v17, v50
	s_clause 0x1
	global_load_b32 v41, v[21:22], off
	global_load_b96 v[16:18], v[23:24], off offset:20
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v42, v42, v49, v53
	s_mov_b32 vcc_lo, s7
	v_fma_f32 v48, -v48, v54, v52
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v48, v48, v50, v54
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v8
	v_div_fixup_f32 v43, v48, v44, v43
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, 0, v51, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v43, v45, v43 :: v_dual_cndmask_b32 v44, 0x7f800000, v49
	v_mul_f32_e32 v49, 0xbfb8aa3b, v9
	v_div_scale_f32 v45, null, v3, v3, v43
	v_div_scale_f32 v56, vcc_lo, v43, v3, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v44, 1.0, v44
	v_fma_f32 v52, 0xbfb8aa3b, v9, -v49
	v_rndne_f32_e32 v53, v49
	v_rcp_f32_e32 v50, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v48, null, v44, v44, v8
	v_fmac_f32_e32 v52, 0xb2a5705f, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v49, v49, v53
	v_cvt_i32_f32_e32 v53, v53
	v_rcp_f32_e32 v51, v48
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v54, -v45, v50, 1.0
	v_dual_add_f32 v49, v49, v52 :: v_dual_fmac_f32 v50, v54, v50
	v_div_scale_f32 v54, s7, v8, v44, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v48, v51, 1.0
	v_exp_f32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v51, v55, v51
	v_dual_mul_f32 v55, v56, v50 :: v_dual_mul_f32 v52, v54, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v57, -v45, v55, v56
	v_ldexp_f32 v49, v49, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v48, v52, v54
	v_dual_fmac_f32 v55, v57, v50 :: v_dual_fmac_f32 v52, v58, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v45, -v45, v55, v56
	v_fma_f32 v48, -v48, v52, v54
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v53, v45, v50, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v48, v51, v52
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v3, v53, v3, v43
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v8, v45, v44, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, 0, v49, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	s_wait_dscnt 0x1
	v_mul_f32_e32 v8, v46, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0x7f800000, v48, vcc_lo
	v_mul_f32_e32 v48, 0xbfb8aa3b, v10
	s_wait_loadcnt 0x3
	v_div_scale_f32 v45, null, v4, v4, v8
	v_div_scale_f32 v56, vcc_lo, v8, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v49, v45
	v_fma_f32 v54, -v45, v49, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v49, v54, v49
	v_div_fixup_f32 v2, v42, v2, v40
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v3, v3, v14 :: v_dual_add_f32 v44, 1.0, v44
	v_mul_f32_e32 v1, v1, v12
	v_fma_f32 v51, 0xbfb8aa3b, v10, -v48
	v_rndne_f32_e32 v52, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v46, null, v44, v44, v9
	v_div_scale_f32 v54, s7, v9, v44, v9
	v_dual_fmac_f32 v51, 0xb2a5705f, v10 :: v_dual_sub_f32 v48, v48, v52
	v_cvt_i32_f32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v50, v46
	v_add_f32_e32 v48, v48, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v48, v48
	v_fma_f32 v55, -v46, v50, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_dual_fmac_f32 v50, v55, v50 :: v_dual_mul_f32 v55, v56, v49
	v_ldexp_f32 v48, v48, v52
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v51, v54, v50
	v_fma_f32 v57, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v58, -v46, v51, v54
	v_fmac_f32_e32 v55, v57, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v51, v58, v50
	v_fma_f32 v45, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v51, v54
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v52, v45, v49, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v46, v50, v51
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v9, v45, v44, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v48, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v9, v47, v9 :: v_dual_cndmask_b32 v44, 0x7f800000, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v55, null, v5, v5, v9
	v_div_scale_f32 v60, vcc_lo, v9, v5, v9
	v_add_f32_e32 v54, 1.0, v44
	v_mul_f32_e32 v44, 0xbfb8aa3b, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v57, v55
	v_div_scale_f32 v56, null, v54, v54, v10
	v_div_scale_f32 v61, s7, v10, v54, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v45, 0xbfb8aa3b, v11, -v44
	v_rndne_f32_e32 v59, v44
	v_rcp_f32_e32 v58, v56
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v55, v57, 1.0
	v_fmac_f32_e32 v45, 0xb2a5705f, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v59
	v_cvt_i32_f32_e32 v59, v59
	v_fmac_f32_e32 v57, v46, v57
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v47, -v56, v58, 1.0
	v_add_f32_e32 v48, v44, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v62, v60, v57
	v_fmac_f32_e32 v58, v47, v58
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v65, v48
	s_clause 0x1
	global_load_b128 v[44:47], v[19:20], off
	global_load_b128 v[48:51], v[19:20], off offset:16
	v_fma_f32 v64, -v55, v62, v60
	v_dual_mul_f32 v63, v61, v58 :: v_dual_fmac_f32 v62, v64, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v66, -v56, v63, v61
	v_ldexp_f32 v59, v65, v59
	v_fma_f32 v55, -v55, v62, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v63, v66, v58
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v55, v55, v57, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v63, v61
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v5, v55, v5, v9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v56, v56, v58, v63
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	s_wait_loadcnt 0x3
	v_fma_f32 v9, v0, v41, v1
	v_fma_f32 v0, v0, v41, -v1
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v5, v5, v16
	v_div_fixup_f32 v10, v56, v54, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v57, 0, v59, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	v_fma_f32 v1, v2, v13, -v3
	s_wait_dscnt 0x0
	v_mul_f32_e32 v10, v29, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v54, 0x7f800000, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v61, vcc_lo, v10, v6, v10
	v_add_f32_e32 v29, 1.0, v54
	v_div_scale_f32 v54, null, v6, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v56, null, v29, v29, v11
	v_rcp_f32_e32 v57, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v58, v56
	v_fma_f32 v59, -v54, v57, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v60, -v56, v58, 1.0
	v_fmac_f32_e32 v57, v59, v57
	v_div_scale_f32 v59, s7, v11, v29, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v58, v60, v58
	v_mul_f32_e32 v60, v61, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v62, v59, v58
	v_fma_f32 v63, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v64, -v56, v62, v59
	v_fmac_f32_e32 v60, v63, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v62, v64, v58
	v_fma_f32 v54, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v62, v59
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v54, v54, v57, v60
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v56, v56, v58, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v6, v54, v6, v10
	v_div_fixup_f32 v11, v56, v29, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v11, v30, v11
	v_div_scale_f32 v29, null, v7, v7, v11
	v_div_scale_f32 v57, vcc_lo, v11, v7, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v30, v29
	v_fma_f32 v56, -v29, v30, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v56, v30
	v_mul_f32_e32 v56, v57, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v29, v56, v57
	v_fmac_f32_e32 v56, v58, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f32 v12, -v29, v56, v57
	v_mbcnt_lo_u32_b32 v29, -1, 0
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v12, v12, v30, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v30, 1, v29
	v_div_fixup_f32 v7, v12, v7, v11
	v_fma_f32 v11, v2, v13, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v7, v7, v18
	v_div_fixup_f32 v4, v52, v4, v8
	v_add_f32_e32 v2, v9, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v8, v6, v17, v7
	v_fma_f32 v3, v4, v15, v5
	v_fma_f32 v4, v4, v15, -v5
	v_fma_f32 v5, v6, v17, -v7
	v_dual_sub_f32 v6, v9, v11 :: v_dual_add_f32 v7, v0, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v9, v3, v8 :: v_dual_sub_f32 v0, v0, v1
	v_add_f32_e32 v1, v4, v5
	v_dual_sub_f32 v3, v3, v8 :: v_dual_sub_f32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v8, v7, v1 :: v_dual_sub_f32 v1, v7, v1
	v_add_f32_e32 v7, v6, v3
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v7, -v7, v7, s1
	v_cndmask_b32_e64 v1, -v1, v1, s1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v7, -v7, v7, s5
	v_cndmask_b32_e64 v1, -v1, v1, s5
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v7, -v7, v7, s4
	v_cndmask_b32_e64 v1, -v1, v1, s4
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v7, -v7, v7, s2
	v_cndmask_b32_e64 v1, -v1, v1, s2
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	v_dual_sub_f32 v3, v6, v3 :: v_dual_lshlrev_b32 v10, 2, v32
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v3, -v3, v3, s1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v3, v14
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v3, -v3, v3, s5
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v3, -v3, v3, s4
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v3, v14
	v_dual_add_f32 v5, v2, v9 :: v_dual_sub_f32 v2, v2, v9
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v3, -v3, v3, s2
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v6, v0, v4
	v_sub_f32_e32 v0, v0, v4
	ds_swizzle_b32 v4, v5 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v14, v1 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s1
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v6, -v6, v6, s1
	v_cndmask_b32_e64 v1, -v1, v1, s3
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v1, v14
	v_add_f32_e32 v4, v5, v4
	s_wait_dscnt 0x0
	v_dual_add_f32 v6, v6, v11 :: v_dual_mul_f32 v1, 0x3d800000, v1
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v6, -v6, v6, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v7 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v7, -v7, v7, s3
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, 0x3d800000, v7
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v8, -v8, v8, s1
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v8, v9
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v4, -v4, v4, s5
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, -v5, v5, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v5, -v5, v5, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, -v5, v5, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v2, -v2, v2, s1
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v0, -v0, v0, s1
	v_xor_b32_e32 v9, 16, v29
	v_mul_f32_e32 v5, 0x3d800000, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v0, -v0, v0, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v0, -v0, v0, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v0, -v0, v0, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v3 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v3, -v3, v3, s3
	ds_swizzle_b32 v16, v0 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v0, -v0, v0, s3
	s_wait_dscnt 0x1
	v_dual_add_f32 v3, v3, v15 :: v_dual_add_f32 v2, v2, v12
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v0, v0, v16 :: v_dual_mul_f32 v3, 0x3d800000, v3
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v2, -v2, v2, s5
	v_add_f32_e32 v4, v4, v8
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v0, 0x3d800000, v0 :: v_dual_mul_f32 v17, v3, v50
	v_mul_f32_e32 v16, v1, v49
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, s4
	v_mul_f32_e32 v18, v0, v51
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v2, -v2, v2, s4
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, s2
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v2, -v2, v2, s2
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, s3
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v6 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v6, -v6, v6, s3
	ds_swizzle_b32 v13, v2 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v2, -v2, v2, s3
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v4, v8
	ds_bpermute_b32 v8, v10, v18
	v_mul_f32_e32 v4, 0x3d800000, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v11, v4, v44
	s_wait_dscnt 0x2
	v_add_f32_e32 v6, v6, v12
	v_mul_f32_e32 v12, v5, v45
	ds_bpermute_b32 v0, v10, v11
	s_wait_dscnt 0x2
	v_add_f32_e32 v2, v2, v13
	v_mul_f32_e32 v6, 0x3d800000, v6
	v_mul_f32_e32 v13, v7, v46
	ds_bpermute_b32 v1, v10, v12
	ds_bpermute_b32 v7, v10, v17
	v_mul_f32_e32 v2, 0x3d800000, v2
	v_mul_f32_e32 v14, v6, v47
	ds_bpermute_b32 v3, v10, v13
	ds_bpermute_b32 v6, v10, v16
	v_mul_f32_e32 v15, v2, v48
	ds_bpermute_b32 v5, v10, v14
	ds_bpermute_b32 v2, v10, v15
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v4, v2, v0, s1
	v_cndmask_b32_e64 v2, v6, v1, s1
	v_cndmask_b32_e64 v1, v7, v3, s1
	v_cndmask_b32_e64 v0, v8, v5, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v29, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v6, |v4|, |v2|, |v1|
	v_max_num_f32_e64 v7, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v5, 2, v3
	v_max_num_f32_e32 v3, v6, v7
	v_xor_b32_e32 v7, 8, v29
	ds_bpermute_b32 v6, v5, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v29, v7 :: v_dual_max_num_f32 v8, v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v3, v3, v8 :: v_dual_lshlrev_b32 v6, 2, v7
	v_xor_b32_e32 v8, 4, v29
	ds_bpermute_b32 v7, v6, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v8, v29, v8 :: v_dual_max_num_f32 v9, v7, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v7, 2, v8
	v_max_num_f32_e32 v3, v3, v9
	v_xor_b32_e32 v9, 2, v29
	ds_bpermute_b32 v8, v7, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v29, v9, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v29, v29, v30 :: v_dual_max_num_f32 v8, v8, v8
	v_max_num_f32_e32 v3, v3, v8
	ds_bpermute_b32 v8, v9, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v30, v8, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v3, v3, v30 :: v_dual_lshlrev_b32 v8, 2, v29
	ds_bpermute_b32 v29, v8, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v29, v29, v29
	v_max_num_f32_e32 v29, v3, v29
	v_mov_b32_e32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v29
	s_cbranch_execz .LBB0_29
; %bb.27:                               ; %.preheader76.i.i.i
	v_div_scale_f32 v3, null, 0x40e00000, 0x40e00000, v29
	v_div_scale_f32 v37, vcc_lo, v29, 0x40e00000, v29
	s_mov_b32 s18, 0x40e00000
	s_mov_b32 s19, 0xc1000000
	v_rcp_f32_e32 v30, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v3, v30, 1.0
	v_fmac_f32_e32 v30, v36, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, v37, v30
	v_fma_f32 v38, -v3, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v38, v30
	v_fma_f32 v3, -v3, v36, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v3, v3, v30, v36
	v_mov_b32_e32 v36, 0x7149f2ca
	v_div_fixup_f32 v30, v3, 0x40e00000, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v3, 1.0 :: v_dual_mul_f32 v30, 0.5, v30
.LBB0_28:                               ; %.preheader.preheader.i.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s7, s17
	s_add_co_i32 s17, s17, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s17, 8
	v_div_scale_f32 v37, null, s18, s18, s7
	v_div_scale_f32 v38, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v39, v37
	v_xor_b32_e32 v37, 0x80000000, v37
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v40, v37, v39, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v39
	v_mul_f32_e32 v40, v38, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v37, v40, v38
	v_fmac_f32_e32 v40, v41, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v37, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v38, v39, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v37, v37, 0x40e00000, s7
	v_add_f32_e32 v37, 1.0, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v30, v37
	v_div_scale_f32 v38, null, v37, v37, v4
	v_div_scale_f32 v40, null, v37, v37, v2
	v_div_scale_f32 v42, null, v37, v37, v1
	v_div_scale_f32 v44, null, v37, v37, v0
	v_div_scale_f32 v39, vcc_lo, v4, v37, v4
	v_rcp_f32_e32 v46, v38
	v_rcp_f32_e32 v47, v40
	v_rcp_f32_e32 v48, v42
	v_rcp_f32_e32 v49, v44
	v_div_scale_f32 v41, s7, v2, v37, v2
	v_div_scale_f32 v43, s8, v1, v37, v1
	v_div_scale_f32 v45, s9, v0, v37, v0
	v_fma_f32 v50, -v38, v46, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v51, -v40, v47, 1.0
	v_fma_f32 v52, -v42, v48, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v53, -v44, v49, 1.0
	v_dual_fmac_f32 v46, v50, v46 :: v_dual_fmac_f32 v47, v51, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v48, v52, v48 :: v_dual_fmac_f32 v49, v53, v49
	v_dual_mul_f32 v50, v39, v46 :: v_dual_mul_f32 v51, v41, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v52, v43, v48 :: v_dual_mul_f32 v53, v45, v49
	v_fma_f32 v54, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v40, v51, v41
	v_fma_f32 v56, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v57, -v44, v53, v45
	v_dual_fmac_f32 v50, v54, v46 :: v_dual_fmac_f32 v51, v55, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v52, v56, v48 :: v_dual_fmac_f32 v53, v57, v49
	v_fma_f32 v38, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v40, v51, v41
	v_fma_f32 v40, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v41, -v44, v53, v45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v38, v46, v50
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v39, v39, v47, v51
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v38, v38, v37, v4
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v40, v40, v48, v52
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v39, v39, v37, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v41, v41, v49, v53
	v_rndne_f32_e32 v38, v38
	v_div_fixup_f32 v40, v40, v37, v1
	v_rndne_f32_e32 v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v41, v41, v37, v0
	v_med3_num_f32 v38, v38, s19, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v40, v40
	v_med3_num_f32 v39, v39, s19, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v41, v41
	v_fma_f32 v38, -v38, v37, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v40, v40, s19, 0x40e00000
	v_fma_f32 v39, -v39, v37, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v41, v41, s19, 0x40e00000
	v_fma_f32 v38, v38, v38, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v40, -v40, v37, v1
	v_fmac_f32_e32 v38, v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v41, v37, v0
	v_fmac_f32_e32 v38, v40, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v39
	ds_bpermute_b32 v39, v5, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v6, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v7, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v9, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v8, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v38, v36
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v36, v36, v38 :: v_dual_cndmask_b32 v3, v3, v37
	s_cbranch_scc1 .LBB0_28
.LBB0_29:                               ; %Flow1095
	s_or_b32 exec_lo, exec_lo, s16
	v_cmp_neq_f32_e64 s7, 0, v29
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_div_scale_f32 v30, null, v3, v3, v4
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v36, v30
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v30, v36, 1.0
	v_fmac_f32_e32 v36, v37, v36
	v_div_scale_f32 v37, vcc_lo, v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v36
	v_fma_f32 v39, -v30, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v36
	v_fma_f32 v30, -v30, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v30, v30, v36, v38
	v_div_fixup_f32 v4, v30, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v30, v4
.LBB0_31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_33
; %bb.32:
	v_div_scale_f32 v4, null, v3, v3, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v29, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v4, v29, 1.0
	v_fmac_f32_e32 v29, v36, v29
	v_div_scale_f32 v36, vcc_lo, v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v36, v29
	v_fma_f32 v38, -v4, v37, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v38, v29
	v_fma_f32 v4, -v4, v37, v36
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v29, v37
	v_div_fixup_f32 v2, v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v29, v2
.LBB0_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_35
; %bb.34:
	v_div_scale_f32 v4, null, v3, v3, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v36, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v4, v36, 1.0
	v_fmac_f32_e32 v36, v37, v36
	v_div_scale_f32 v37, vcc_lo, v1, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v36
	v_fma_f32 v39, -v4, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v36
	v_fma_f32 v4, -v4, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v36, v38
	v_div_fixup_f32 v1, v4, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v4, v1
.LBB0_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_37
; %bb.36:
	v_div_scale_f32 v1, null, v3, v3, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v36, v2
	v_div_scale_f32 v36, vcc_lo, v0, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v36, v2
	v_fma_f32 v38, -v1, v37, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v38, v2
	v_fma_f32 v1, -v1, v37, v36
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v37
	v_div_fixup_f32 v0, v1, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v29, v30
	v_and_b32_e32 v30, 15, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v0, v0, v4, v2
	v_lshl_or_b32 v29, v29, 4, v30
	ds_bpermute_b32 v1, v5, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v6, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v7, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v36, v0, v1
	v_mad_co_i64_i32 v[0:1], null, 0x48, v35, s[12:13]
	ds_bpermute_b32 v37, v9, v36
	v_mad_co_u64_u32 v[0:1], null, 0x48, s10, v[0:1]
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v36, v36, v37
	v_and_b32_e32 v37, 15, v4
	ds_bpermute_b32 v4, v8, v36
	v_lshl_or_b32 v37, v2, 4, v37
	v_mad_co_u64_u32 v[1:2], null, 0x48, s11, v[1:2]
	v_and_b16 v2.h, 0xff, v29.l
	v_add_co_u32 v29, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v37.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[29:30], v2, off offset:8
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB0_39
; %bb.38:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v36, v4
	global_store_b64 v[0:1], v[3:4], off
.LBB0_39:                               ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit.i.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_bpermute_b32 v2, v10, v11 offset:64
	ds_bpermute_b32 v3, v10, v12 offset:64
	s_wait_dscnt 0x2
	ds_bpermute_b32 v4, v10, v13 offset:64
	ds_bpermute_b32 v12, v10, v14 offset:64
	ds_bpermute_b32 v11, v10, v15 offset:64
	ds_bpermute_b32 v13, v10, v16 offset:64
	ds_bpermute_b32 v14, v10, v17 offset:64
	ds_bpermute_b32 v15, v10, v18 offset:64
	s_mov_b32 s17, 0
	s_mov_b32 s16, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v11, v11, v2, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v10, v13, v3, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, v14, v4, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v2, v15, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v3, |v11|, |v10|, |v4|
	v_max_num_f32_e64 v12, |v2|, |v2|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v5, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v6, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v7, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v9, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v8, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_dual_max_num_f32 v12, v3, v12 :: v_dual_mov_b32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_42
; %bb.40:                               ; %.preheader76.i.1.i.i
	v_div_scale_f32 v3, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s18, 0x40e00000
	s_mov_b32 s19, 0xc1000000
	v_rcp_f32_e32 v13, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v3, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v3, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v3, -v3, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v3, v3, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v3, 0x40e00000, v12
	v_mov_b32_e32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_41:                               ; %.preheader.preheader.i.1.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s7, s17
	s_add_co_i32 s17, s17, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s17, 8
	v_div_scale_f32 v15, null, s18, s18, s7
	v_div_scale_f32 v16, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v29, v15, v18, v16
	v_fmac_f32_e32 v18, v29, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s7
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v18, null, v15, v15, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v40, v18
	v_fma_f32 v44, -v18, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v44, v40
	v_div_scale_f32 v16, null, v15, v15, v11
	v_div_scale_f32 v30, null, v15, v15, v4
	v_div_scale_f32 v37, null, v15, v15, v2
	v_div_scale_f32 v17, vcc_lo, v11, v15, v11
	v_div_scale_f32 v29, s7, v10, v15, v10
	v_rcp_f32_e32 v39, v16
	v_rcp_f32_e32 v41, v30
	v_rcp_f32_e32 v42, v37
	v_div_scale_f32 v38, s9, v2, v15, v2
	v_div_scale_f32 v36, s8, v4, v15, v4
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v43, -v16, v39, 1.0
	v_fma_f32 v45, -v30, v41, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v37, v42, 1.0
	v_fmac_f32_e32 v39, v43, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v41, v45, v41 :: v_dual_fmac_f32 v42, v46, v42
	v_mul_f32_e32 v44, v29, v40
	v_mul_f32_e32 v43, v17, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v45, v36, v41 :: v_dual_mul_f32 v46, v38, v42
	v_fma_f32 v48, -v18, v44, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v16, v43, v17
	v_fma_f32 v49, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v37, v46, v38
	v_dual_fmac_f32 v44, v48, v40 :: v_dual_fmac_f32 v43, v47, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v45, v49, v41 :: v_dual_fmac_f32 v46, v50, v42
	v_fma_f32 v16, -v16, v43, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v44, v29
	v_fma_f32 v18, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v29, -v37, v46, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v39, v43
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v40, v44
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v16, v16, v15, v11
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v41, v45
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v17, v17, v15, v10
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v29, v29, v42, v46
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v4
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v29, v29, v15, v2
	v_med3_num_f32 v16, v16, s19, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s19, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v29, v29
	v_fma_f32 v16, -v16, v15, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s19, 0x40e00000
	v_fma_f32 v17, -v17, v15, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v29, v29, s19, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v4
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v29, v15, v2
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v5, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v6, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v14, v16 :: v_dual_cndmask_b32 v3, v3, v15
	s_cbranch_scc1 .LBB0_41
.LBB0_42:                               ; %Flow1093
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s16
	v_cmp_neq_f32_e64 s7, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_div_scale_f32 v13, null, v3, v3, v11
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v11, v3, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v11, v13, v3, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v11, v11
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v11, v11, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v11
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_div_scale_f32 v11, null, v3, v3, v10
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v12, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v11, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v10, v3, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v11, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v11, -v11, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v12, v15
	v_div_fixup_f32 v10, v11, v3, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v10, v10
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v10, v10, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v10
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_div_scale_f32 v11, null, v3, v3, v4
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v11, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v11, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v11, -v11, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v14, v16
	v_div_fixup_f32 v4, v11, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v11, v4
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_div_scale_f32 v4, null, v3, v3, v2
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v10, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v4, v10, 1.0
	v_fmac_f32_e32 v10, v14, v10
	v_div_scale_f32 v14, vcc_lo, v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v10
	v_fma_f32 v16, -v4, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v10
	v_fma_f32 v4, -v4, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v10, v15
	v_div_fixup_f32 v2, v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v10, v2
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v2, v12, v13
	v_mad_co_u64_u32 v[0:1], null, 0x48, s26, v[0:1]
	s_delay_alu instid0(VALU_DEP_2)
	v_add3_u32 v2, v2, v11, v10
	ds_bpermute_b32 v4, v5, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v6, v2
	v_and_b32_e32 v6, 15, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v6, v12, 4, v6
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v7, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v9, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v2, v4
	v_and_b32_e32 v2, 15, v11
	ds_bpermute_b32 v5, v8, v4
	v_lshl_or_b32 v7, v10, 4, v2
	v_mad_co_u64_u32 v[1:2], null, 0x48, s27, v[1:2]
	v_and_b16 v2.h, 0xff, v6.l
	v_add_co_u32 v6, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v7.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[6:7], v2, off offset:8
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_52
; %bb.51:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v4, v5
	global_store_b64 v[0:1], v[3:4], off
.LBB0_52:                               ; %Flow1096
	s_or_b32 exec_lo, exec_lo, s15
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_54
; %bb.53:                               ; %.preheader902.1.i
	ds_store_2addr_b32 v34, v175, v190 offset1:1
	ds_store_2addr_b32 v34, v188, v189 offset0:2 offset1:3
	ds_store_2addr_b32 v34, v186, v187 offset0:4 offset1:5
	ds_store_2addr_b32 v34, v184, v185 offset0:6 offset1:7
	ds_store_2addr_b32 v34, v182, v183 offset0:16 offset1:17
	ds_store_2addr_b32 v34, v180, v181 offset0:18 offset1:19
	ds_store_2addr_b32 v34, v178, v179 offset0:20 offset1:21
	ds_store_2addr_b32 v34, v176, v177 offset0:22 offset1:23
.LBB0_54:                               ; %.loopexit.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_or_b32 s8, s14, 1
	v_add_nc_u32_e32 v0, 8, v35
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s9, s8, 31
	s_mov_b32 s16, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[14:15], s[26:27], s[8:9]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s26, v0
	s_cbranch_execz .LBB0_81
; %bb.55:
	v_mad_co_i64_i32 v[0:1], null, s24, v0, 0
	s_mov_b32 s18, 0
	s_mov_b32 s17, exec_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v0, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v28, vcc_lo
	global_load_b128 v[40:43], v[4:5], off
	global_load_b128 v[0:3], v[25:26], off
	global_load_b128 v[8:11], v[4:5], off offset:16
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v4, 0xbfb8aa3b, v40
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, 0xbfb8aa3b, v40, -v4
	v_rndne_f32_e32 v6, v4
	v_dual_fmac_f32 v5, 0xb2a5705f, v40 :: v_dual_sub_f32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v40
	global_load_b128 v[4:7], v[25:26], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0x7f800000, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v14, 1.0, v12
	v_mul_f32_e32 v12, 0xbfb8aa3b, v41
	v_div_scale_f32 v15, null, v14, v14, v40
	v_div_scale_f32 v29, vcc_lo, v40, v14, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v17, 0xbfb8aa3b, v41, -v12
	v_rndne_f32_e32 v18, v12
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v17, 0xb2a5705f, v41 :: v_dual_sub_f32 v12, v12, v18
	v_cvt_i32_f32_e32 v18, v18
	v_add_f32_e32 v12, v12, v17
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v13, -v15, v16, 1.0
	v_exp_f32_e32 v17, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v13, v16
	v_mul_f32_e32 v30, v29, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_ldexp_f32 v17, v17, v18
	v_fma_f32 v13, -v15, v30, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v13, v16
	ds_load_2addr_b32 v[12:13], v33 offset1:1
	v_fma_f32 v15, -v15, v30, v29
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v15, v16, v30
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v41
	ds_load_2addr_b32 v[44:45], v33 offset0:2 offset1:3
	ds_load_2addr_b32 v[46:47], v33 offset0:4 offset1:5
	ds_load_2addr_b32 v[29:30], v33 offset0:6 offset1:7
	v_div_fixup_f32 v14, v15, v14, v40
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, 0, v17, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v41
	s_wait_dscnt 0x3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v36, v12, v14 :: v_dual_cndmask_b32 v15, 0x7f800000, v16
	v_mul_f32_e32 v16, 0xbfb8aa3b, v42
	s_wait_loadcnt 0x2
	v_div_scale_f32 v14, null, v0, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, 1.0, v15
	v_fma_f32 v37, 0xbfb8aa3b, v42, -v16
	v_rndne_f32_e32 v38, v16
	v_div_scale_f32 v48, vcc_lo, v36, v0, v36
	v_rcp_f32_e32 v17, v14
	v_div_scale_f32 v15, null, v12, v12, v41
	v_fmac_f32_e32 v37, 0xb2a5705f, v42
	v_sub_f32_e32 v16, v16, v38
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v15
	v_add_f32_e32 v16, v16, v37
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v14, v17, 1.0
	v_exp_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v17, v39, v17
	v_fma_f32 v40, -v15, v18, 1.0
	v_div_scale_f32 v39, s7, v41, v12, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v40, v18
	v_dual_mul_f32 v40, v48, v17 :: v_dual_mul_f32 v49, v39, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v37, -v14, v40, v48
	v_fma_f32 v50, -v15, v49, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v40, v37, v17
	v_cvt_i32_f32_e32 v37, v38
	v_fmac_f32_e32 v49, v50, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v14, -v14, v40, v48
	v_ldexp_f32 v16, v16, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v15, -v15, v49, v39
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v14, v17, v40
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v15, v18, v49
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v12, v14, v12, v41
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v16, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v42
	v_mul_f32_e32 v38, v13, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v14, 0x7f800000, v15, vcc_lo
	v_mul_f32_e32 v15, 0xbfb8aa3b, v43
	v_div_scale_f32 v13, null, v1, v1, v38
	v_div_scale_f32 v48, vcc_lo, v38, v1, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v12, 1.0, v14
	v_fma_f32 v18, 0xbfb8aa3b, v43, -v15
	v_rndne_f32_e32 v39, v15
	v_rcp_f32_e32 v16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v14, null, v12, v12, v42
	v_fmac_f32_e32 v18, 0xb2a5705f, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v15, v15, v39
	v_cvt_i32_f32_e32 v39, v39
	v_rcp_f32_e32 v17, v14
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v13, v16, 1.0
	v_dual_add_f32 v15, v15, v18 :: v_dual_fmac_f32 v16, v40, v16
	v_div_scale_f32 v40, s7, v42, v12, v42
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v14, v17, 1.0
	v_exp_f32_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v41, v17
	v_mul_f32_e32 v41, v48, v16
	v_mul_f32_e32 v18, v40, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v49, -v13, v41, v48
	v_ldexp_f32 v15, v15, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v14, v18, v40
	v_fmac_f32_e32 v41, v49, v16
	v_div_fixup_f32 v0, v37, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, v50, v17
	v_fma_f32 v13, -v13, v41, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v14, v18, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v39, v13, v16, v41
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v14, v17, v18
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v12, v13, v12, v42
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v43
	s_wait_dscnt 0x2
	v_mul_f32_e32 v40, v44, v12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, 0xbfb8aa3b, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, 0x7f800000, v14, vcc_lo
	v_div_scale_f32 v42, null, v2, v2, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v16, v12
	v_add_f32_e32 v44, 1.0, v13
	v_fma_f32 v13, 0xbfb8aa3b, v8, -v12
	v_div_scale_f32 v51, vcc_lo, v40, v2, v40
	v_rcp_f32_e32 v49, v42
	v_sub_f32_e32 v12, v12, v16
	v_div_scale_f32 v48, null, v44, v44, v43
	v_fmac_f32_e32 v13, 0xb2a5705f, v8
	v_div_scale_f32 v52, s7, v43, v44, v43
	v_cvt_i32_f32_e32 v56, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v50, v48
	v_add_f32_e32 v17, v12, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v42, v49, 1.0
	v_exp_f32_e32 v55, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v49, v14, v49
	v_fma_f32 v15, -v48, v50, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v53, v51, v49
	v_fmac_f32_e32 v50, v15, v50
	global_load_b128 v[12:15], v[23:24], off offset:4
	v_fma_f32 v18, -v42, v53, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v54, v52, v50 :: v_dual_fmac_f32 v53, v18, v49
	v_div_fixup_f32 v1, v39, v1, v38
	v_fma_f32 v17, -v48, v54, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v42, -v42, v53, v51
	v_ldexp_f32 v51, v55, v56
	v_fmac_f32_e32 v54, v17, v50
	s_clause 0x1
	global_load_b32 v41, v[21:22], off
	global_load_b96 v[16:18], v[23:24], off offset:20
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v42, v42, v49, v53
	s_mov_b32 vcc_lo, s7
	v_fma_f32 v48, -v48, v54, v52
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v48, v48, v50, v54
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v8
	v_div_fixup_f32 v43, v48, v44, v43
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, 0, v51, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v43, v45, v43 :: v_dual_cndmask_b32 v44, 0x7f800000, v49
	v_mul_f32_e32 v49, 0xbfb8aa3b, v9
	v_div_scale_f32 v45, null, v3, v3, v43
	v_div_scale_f32 v56, vcc_lo, v43, v3, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v44, 1.0, v44
	v_fma_f32 v52, 0xbfb8aa3b, v9, -v49
	v_rndne_f32_e32 v53, v49
	v_rcp_f32_e32 v50, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v48, null, v44, v44, v8
	v_fmac_f32_e32 v52, 0xb2a5705f, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v49, v49, v53
	v_cvt_i32_f32_e32 v53, v53
	v_rcp_f32_e32 v51, v48
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v54, -v45, v50, 1.0
	v_dual_add_f32 v49, v49, v52 :: v_dual_fmac_f32 v50, v54, v50
	v_div_scale_f32 v54, s7, v8, v44, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v48, v51, 1.0
	v_exp_f32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v51, v55, v51
	v_dual_mul_f32 v55, v56, v50 :: v_dual_mul_f32 v52, v54, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v57, -v45, v55, v56
	v_ldexp_f32 v49, v49, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v48, v52, v54
	v_dual_fmac_f32 v55, v57, v50 :: v_dual_fmac_f32 v52, v58, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v45, -v45, v55, v56
	v_fma_f32 v48, -v48, v52, v54
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v53, v45, v50, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v48, v51, v52
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v3, v53, v3, v43
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v8, v45, v44, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, 0, v49, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	s_wait_dscnt 0x1
	v_mul_f32_e32 v8, v46, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0x7f800000, v48, vcc_lo
	v_mul_f32_e32 v48, 0xbfb8aa3b, v10
	s_wait_loadcnt 0x3
	v_div_scale_f32 v45, null, v4, v4, v8
	v_div_scale_f32 v56, vcc_lo, v8, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v49, v45
	v_fma_f32 v54, -v45, v49, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v49, v54, v49
	v_div_fixup_f32 v2, v42, v2, v40
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v3, v3, v14 :: v_dual_add_f32 v44, 1.0, v44
	v_mul_f32_e32 v1, v1, v12
	v_fma_f32 v51, 0xbfb8aa3b, v10, -v48
	v_rndne_f32_e32 v52, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v46, null, v44, v44, v9
	v_div_scale_f32 v54, s7, v9, v44, v9
	v_dual_fmac_f32 v51, 0xb2a5705f, v10 :: v_dual_sub_f32 v48, v48, v52
	v_cvt_i32_f32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v50, v46
	v_add_f32_e32 v48, v48, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v48, v48
	v_fma_f32 v55, -v46, v50, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_dual_fmac_f32 v50, v55, v50 :: v_dual_mul_f32 v55, v56, v49
	v_ldexp_f32 v48, v48, v52
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v51, v54, v50
	v_fma_f32 v57, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v58, -v46, v51, v54
	v_fmac_f32_e32 v55, v57, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v51, v58, v50
	v_fma_f32 v45, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v51, v54
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v52, v45, v49, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v46, v50, v51
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v9, v45, v44, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v48, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v9, v47, v9 :: v_dual_cndmask_b32 v44, 0x7f800000, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v55, null, v5, v5, v9
	v_div_scale_f32 v60, vcc_lo, v9, v5, v9
	v_add_f32_e32 v54, 1.0, v44
	v_mul_f32_e32 v44, 0xbfb8aa3b, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v57, v55
	v_div_scale_f32 v56, null, v54, v54, v10
	v_div_scale_f32 v61, s7, v10, v54, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v45, 0xbfb8aa3b, v11, -v44
	v_rndne_f32_e32 v59, v44
	v_rcp_f32_e32 v58, v56
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v55, v57, 1.0
	v_fmac_f32_e32 v45, 0xb2a5705f, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v59
	v_cvt_i32_f32_e32 v59, v59
	v_fmac_f32_e32 v57, v46, v57
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v47, -v56, v58, 1.0
	v_add_f32_e32 v48, v44, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v62, v60, v57
	v_fmac_f32_e32 v58, v47, v58
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v65, v48
	s_clause 0x1
	global_load_b128 v[44:47], v[19:20], off
	global_load_b128 v[48:51], v[19:20], off offset:16
	v_fma_f32 v64, -v55, v62, v60
	v_dual_mul_f32 v63, v61, v58 :: v_dual_fmac_f32 v62, v64, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v66, -v56, v63, v61
	v_ldexp_f32 v59, v65, v59
	v_fma_f32 v55, -v55, v62, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v63, v66, v58
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v55, v55, v57, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v63, v61
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v5, v55, v5, v9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v56, v56, v58, v63
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	s_wait_loadcnt 0x3
	v_fma_f32 v9, v0, v41, v1
	v_fma_f32 v0, v0, v41, -v1
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v5, v5, v16
	v_div_fixup_f32 v10, v56, v54, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v57, 0, v59, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	v_fma_f32 v1, v2, v13, -v3
	s_wait_dscnt 0x0
	v_mul_f32_e32 v10, v29, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v54, 0x7f800000, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v61, vcc_lo, v10, v6, v10
	v_add_f32_e32 v29, 1.0, v54
	v_div_scale_f32 v54, null, v6, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v56, null, v29, v29, v11
	v_rcp_f32_e32 v57, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v58, v56
	v_fma_f32 v59, -v54, v57, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v60, -v56, v58, 1.0
	v_fmac_f32_e32 v57, v59, v57
	v_div_scale_f32 v59, s7, v11, v29, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v58, v60, v58
	v_mul_f32_e32 v60, v61, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v62, v59, v58
	v_fma_f32 v63, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v64, -v56, v62, v59
	v_fmac_f32_e32 v60, v63, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v62, v64, v58
	v_fma_f32 v54, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v62, v59
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v54, v54, v57, v60
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v56, v56, v58, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v6, v54, v6, v10
	v_div_fixup_f32 v11, v56, v29, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v11, v30, v11
	v_div_scale_f32 v29, null, v7, v7, v11
	v_div_scale_f32 v57, vcc_lo, v11, v7, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v30, v29
	v_fma_f32 v56, -v29, v30, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v56, v30
	v_mul_f32_e32 v56, v57, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v29, v56, v57
	v_fmac_f32_e32 v56, v58, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v12, -v29, v56, v57
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v12, v12, v30, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v7, v12, v7, v11
	v_fma_f32 v11, v2, v13, v3
	v_mul_f32_e32 v7, v7, v18
	v_div_fixup_f32 v4, v52, v4, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v9, v11
	v_fma_f32 v8, v6, v17, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v3, v4, v15, v5
	v_fma_f32 v4, v4, v15, -v5
	v_fma_f32 v5, v6, v17, -v7
	v_dual_sub_f32 v6, v9, v11 :: v_dual_add_f32 v7, v0, v1
	v_dual_add_f32 v9, v3, v8 :: v_dual_sub_f32 v0, v0, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v4, v5
	v_sub_f32_e32 v3, v3, v8
	v_dual_add_f32 v8, v7, v1 :: v_dual_sub_f32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v7, v6, v3
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v7, -v7, v7, s1
	v_cndmask_b32_e64 v1, -v1, v1, s1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v7, -v7, v7, s5
	v_cndmask_b32_e64 v1, -v1, v1, s5
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v7, -v7, v7, s4
	v_cndmask_b32_e64 v1, -v1, v1, s4
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v7, -v7, v7, s2
	v_cndmask_b32_e64 v1, -v1, v1, s2
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	v_sub_f32_e32 v3, v6, v3
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v3, -v3, v3, s1
	v_cndmask_b32_e64 v7, -v7, v7, s3
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v4, v4, v5 :: v_dual_add_f32 v7, v7, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_mul_f32_e32 v7, 0x3d800000, v7
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v3, -v3, v3, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v3, v14
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v3, -v3, v3, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	v_dual_add_f32 v5, v2, v9 :: v_dual_sub_f32 v2, v2, v9
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v8, -v8, v8, s1
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v3, -v3, v3, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v6, v0, v4
	v_sub_f32_e32 v0, v0, v4
	ds_swizzle_b32 v4, v5 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v14, v1 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s1
	v_cndmask_b32_e64 v1, -v1, v1, s3
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v6, -v6, v6, s1
	s_wait_dscnt 0x1
	v_add_f32_e32 v1, v1, v14
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v8, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v1, 0x3d800000, v1
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, -v5, v5, s5
	v_cndmask_b32_e64 v4, -v4, v4, s5
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v18, v1, v49
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v5, -v5, v5, s4
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, -v5, v5, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v2, -v2, v2, s1
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v0, -v0, v0, s1
	v_xor_b32_e32 v9, 16, v10
	v_mul_f32_e32 v5, 0x3d800000, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v14, v5, v45
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v0, -v0, v0, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v0, -v0, v0, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v0, -v0, v0, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v3 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v3, -v3, v3, s3
	ds_swizzle_b32 v16, v0 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v0, -v0, v0, s3
	s_wait_dscnt 0x1
	v_dual_add_f32 v3, v3, v15 :: v_dual_add_f32 v2, v2, v12
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v15, v7, v46 :: v_dual_add_f32 v0, v0, v16
	v_mul_f32_e32 v3, 0x3d800000, v3
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,2)
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s5
	v_dual_mul_f32 v0, 0x3d800000, v0 :: v_dual_mul_f32 v29, v3, v50
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, s4
	v_mul_f32_e32 v30, v0, v51
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,4)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s4
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, s2
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,8)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s2
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, s3
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	v_lshlrev_b32_e32 v12, 2, v32
	ds_swizzle_b32 v13, v2 offset:swizzle(SWAP,16)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s3
	ds_bpermute_b32 v1, v12, v14
	ds_bpermute_b32 v7, v12, v29
	ds_bpermute_b32 v8, v12, v30
	v_mul_f32_e32 v4, 0x3d800000, v4
	v_add_f32_e32 v6, v6, v11
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v2, v2, v13 :: v_dual_mul_f32 v13, v4, v44
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v6, -v6, v6, s5
	ds_bpermute_b32 v4, v12, v15
	v_mul_f32_e32 v2, 0x3d800000, v2
	ds_bpermute_b32 v0, v12, v13
	v_mul_f32_e32 v17, v2, v48
	ds_bpermute_b32 v2, v12, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, s4
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v3, v2, v0, s1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v6, -v6, v6, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	v_mul_f32_e32 v6, 0x3d800000, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, v6, v47
	ds_bpermute_b32 v6, v12, v18
	ds_bpermute_b32 v5, v12, v16
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v2, v6, v1, s1
	v_cndmask_b32_e64 v1, v7, v4, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v10, v9, vcc_lo
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v8, v5, s1
	v_max3_num_f32 v5, |v3|, |v2|, |v1|
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v7, 2, v4
	v_max_num_f32_e64 v6, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v4, v5, v6
	v_xor_b32_e32 v6, 8, v10
	ds_bpermute_b32 v5, v7, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v10, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 2, v6
	v_xor_b32_e32 v6, 4, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_cndmask_b32 v6, v10, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v4, v4, v5 :: v_dual_lshlrev_b32 v9, 2, v6
	v_xor_b32_e32 v6, 2, v10
	ds_bpermute_b32 v5, v8, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v10, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v6
	v_xor_b32_e32 v6, 1, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_cndmask_b32 v6, v10, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v4, v4, v5
	v_lshlrev_b32_e32 v10, 2, v6
	ds_bpermute_b32 v5, v9, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v11, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	v_max_num_f32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v10, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v4, v4, v5 :: v_dual_mov_b32 v5, 1.0
	v_cmpx_neq_f32_e32 0, v4
	s_cbranch_execz .LBB0_58
; %bb.56:                               ; %.preheader76.i.i.1.i
	v_div_scale_f32 v5, null, 0x40e00000, 0x40e00000, v4
	v_div_scale_f32 v37, vcc_lo, v4, 0x40e00000, v4
	s_mov_b32 s19, 0x40e00000
	s_mov_b32 s22, 0xc1000000
	v_rcp_f32_e32 v6, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v5, v6, 1.0
	v_fmac_f32_e32 v6, v36, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, v37, v6
	v_fma_f32 v38, -v5, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v38, v6
	v_fma_f32 v5, -v5, v36, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v5, v5, v6, v36
	v_mov_b32_e32 v36, 0x7149f2ca
	v_div_fixup_f32 v6, v5, 0x40e00000, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v5, 1.0 :: v_dual_mul_f32 v6, 0.5, v6
.LBB0_57:                               ; %.preheader.preheader.i.i.1.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s7, s18
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s18, 8
	v_div_scale_f32 v37, null, s19, s19, s7
	v_div_scale_f32 v38, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v39, v37
	v_xor_b32_e32 v37, 0x80000000, v37
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v40, v37, v39, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v39
	v_mul_f32_e32 v40, v38, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v37, v40, v38
	v_fmac_f32_e32 v40, v41, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v37, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v38, v39, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v37, v37, 0x40e00000, s7
	v_add_f32_e32 v37, 1.0, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v6, v37
	v_div_scale_f32 v38, null, v37, v37, v3
	v_div_scale_f32 v40, null, v37, v37, v2
	v_div_scale_f32 v42, null, v37, v37, v1
	v_div_scale_f32 v44, null, v37, v37, v0
	v_div_scale_f32 v39, vcc_lo, v3, v37, v3
	v_rcp_f32_e32 v46, v38
	v_rcp_f32_e32 v47, v40
	v_rcp_f32_e32 v48, v42
	v_rcp_f32_e32 v49, v44
	v_div_scale_f32 v41, s7, v2, v37, v2
	v_div_scale_f32 v43, s8, v1, v37, v1
	v_div_scale_f32 v45, s9, v0, v37, v0
	v_fma_f32 v50, -v38, v46, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v51, -v40, v47, 1.0
	v_fma_f32 v52, -v42, v48, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v53, -v44, v49, 1.0
	v_dual_fmac_f32 v46, v50, v46 :: v_dual_fmac_f32 v47, v51, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v48, v52, v48 :: v_dual_fmac_f32 v49, v53, v49
	v_dual_mul_f32 v50, v39, v46 :: v_dual_mul_f32 v51, v41, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v52, v43, v48 :: v_dual_mul_f32 v53, v45, v49
	v_fma_f32 v54, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v40, v51, v41
	v_fma_f32 v56, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v57, -v44, v53, v45
	v_dual_fmac_f32 v50, v54, v46 :: v_dual_fmac_f32 v51, v55, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v52, v56, v48 :: v_dual_fmac_f32 v53, v57, v49
	v_fma_f32 v38, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v40, v51, v41
	v_fma_f32 v40, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v41, -v44, v53, v45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v38, v46, v50
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v39, v39, v47, v51
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v38, v38, v37, v3
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v40, v40, v48, v52
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v39, v39, v37, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v41, v41, v49, v53
	v_rndne_f32_e32 v38, v38
	v_div_fixup_f32 v40, v40, v37, v1
	v_rndne_f32_e32 v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v41, v41, v37, v0
	v_med3_num_f32 v38, v38, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v40, v40
	v_med3_num_f32 v39, v39, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v41, v41
	v_fma_f32 v38, -v38, v37, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v40, v40, s22, 0x40e00000
	v_fma_f32 v39, -v39, v37, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v41, v41, s22, 0x40e00000
	v_fma_f32 v38, v38, v38, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v40, -v40, v37, v1
	v_fmac_f32_e32 v38, v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v41, v37, v0
	v_fmac_f32_e32 v38, v40, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v39
	ds_bpermute_b32 v39, v7, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v8, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v9, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v11, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v10, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v38, v36
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v36, v36, v38 :: v_dual_cndmask_b32 v5, v5, v37
	s_cbranch_scc1 .LBB0_57
.LBB0_58:                               ; %Flow1089
	s_or_b32 exec_lo, exec_lo, s17
	v_cmp_neq_f32_e64 s7, 0, v4
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v36, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_div_scale_f32 v4, null, v5, v5, v3
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v36, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v4, v36, 1.0
	v_fmac_f32_e32 v36, v37, v36
	v_div_scale_f32 v37, vcc_lo, v3, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v36
	v_fma_f32 v39, -v4, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v36
	v_fma_f32 v4, -v4, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v36, v38
	v_div_fixup_f32 v3, v4, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v3, v3
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v3, v3, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v36, v3
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_div_scale_f32 v3, null, v5, v5, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v6, v4
	v_div_scale_f32 v6, vcc_lo, v2, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v6, v4
	v_fma_f32 v38, -v3, v37, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v38, v4
	v_fma_f32 v3, -v3, v37, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v37
	v_div_fixup_f32 v2, v3, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v2
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v37, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_div_scale_f32 v3, null, v5, v5, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v37, v4
	v_div_scale_f32 v37, vcc_lo, v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v4
	v_fma_f32 v39, -v3, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v4
	v_fma_f32 v3, -v3, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v38
	v_div_fixup_f32 v1, v3, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v37, v1
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_div_scale_f32 v1, null, v5, v5, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v3, v2
	v_div_scale_f32 v3, vcc_lo, v0, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v4, v3, v2
	v_fma_f32 v38, -v1, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v38, v2
	v_fma_f32 v1, -v1, v4, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v4
	v_div_fixup_f32 v0, v1, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v6, v36
	v_mad_co_i64_i32 v[3:4], null, 0x48, v35, s[12:13]
	v_and_b32_e32 v39, 15, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add3_u32 v0, v0, v37, v2
	v_and_b32_e32 v37, 15, v37
	v_lshl_or_b32 v6, v6, 4, v39
	ds_bpermute_b32 v1, v7, v0
	v_lshl_or_b32 v37, v2, 4, v37
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v8, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v9, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v11, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v38, v0, v1
	v_mad_co_u64_u32 v[0:1], null, 0x48, s10, v[3:4]
	ds_bpermute_b32 v36, v10, v38
	v_mad_co_u64_u32 v[1:2], null, 0x48, s11, v[1:2]
	v_lshlrev_b16 v2.l, 8, v37.l
	v_and_b16 v2.h, 0xff, v6.l
	v_add_co_u32 v39, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, 0, v1, vcc_lo
	global_store_b16 v[39:40], v2, off offset:584
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB0_68
; %bb.67:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v38, v36
	global_store_b64 v[0:1], v[5:6], off offset:576
.LBB0_68:                               ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit.i.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_bpermute_b32 v0, v12, v13 offset:64
	ds_bpermute_b32 v1, v12, v14 offset:64
	ds_bpermute_b32 v5, v12, v15 offset:64
	ds_bpermute_b32 v13, v12, v16 offset:64
	ds_bpermute_b32 v2, v12, v17 offset:64
	ds_bpermute_b32 v14, v12, v18 offset:64
	ds_bpermute_b32 v15, v12, v29 offset:64
	ds_bpermute_b32 v12, v12, v30 offset:64
	s_mov_b32 s18, 0
	s_mov_b32 s17, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v6, v2, v0, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v2, v14, v1, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v1, v15, v5, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v12, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v5, |v6|, |v2|, |v1|
	v_max_num_f32_e64 v12, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v7, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v8, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v9, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v11, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v10, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_dual_max_num_f32 v12, v5, v12 :: v_dual_mov_b32 v5, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_71
; %bb.69:                               ; %.preheader76.i.1.i.1.i
	v_div_scale_f32 v5, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s19, 0x40e00000
	s_mov_b32 s22, 0xc1000000
	v_rcp_f32_e32 v13, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v5, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v5, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v5, -v5, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v5, v5, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v5, 0x40e00000, v12
	v_mov_b32_e32 v5, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_70:                               ; %.preheader.preheader.i.1.i.1.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s7, s18
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s18, 8
	v_div_scale_f32 v15, null, s19, s19, s7
	v_div_scale_f32 v16, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v29, v15, v18, v16
	v_fmac_f32_e32 v18, v29, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s7
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v18, null, v15, v15, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v40, v18
	v_fma_f32 v44, -v18, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v44, v40
	v_div_scale_f32 v16, null, v15, v15, v6
	v_div_scale_f32 v30, null, v15, v15, v1
	v_div_scale_f32 v37, null, v15, v15, v0
	v_div_scale_f32 v17, vcc_lo, v6, v15, v6
	v_div_scale_f32 v29, s7, v2, v15, v2
	v_rcp_f32_e32 v39, v16
	v_rcp_f32_e32 v41, v30
	v_rcp_f32_e32 v42, v37
	v_div_scale_f32 v38, s9, v0, v15, v0
	v_div_scale_f32 v36, s8, v1, v15, v1
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v43, -v16, v39, 1.0
	v_fma_f32 v45, -v30, v41, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v37, v42, 1.0
	v_fmac_f32_e32 v39, v43, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v41, v45, v41 :: v_dual_fmac_f32 v42, v46, v42
	v_mul_f32_e32 v44, v29, v40
	v_mul_f32_e32 v43, v17, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v45, v36, v41 :: v_dual_mul_f32 v46, v38, v42
	v_fma_f32 v48, -v18, v44, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v16, v43, v17
	v_fma_f32 v49, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v37, v46, v38
	v_dual_fmac_f32 v44, v48, v40 :: v_dual_fmac_f32 v43, v47, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v45, v49, v41 :: v_dual_fmac_f32 v46, v50, v42
	v_fma_f32 v16, -v16, v43, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v44, v29
	v_fma_f32 v18, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v29, -v37, v46, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v39, v43
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v40, v44
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v16, v16, v15, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v41, v45
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v17, v17, v15, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v29, v29, v42, v46
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v1
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v29, v29, v15, v0
	v_med3_num_f32 v16, v16, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v29, v29
	v_fma_f32 v16, -v16, v15, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s22, 0x40e00000
	v_fma_f32 v17, -v17, v15, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v29, v29, s22, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v1
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v29, v15, v0
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v11, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v10, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v14, v16 :: v_dual_cndmask_b32 v5, v5, v15
	s_cbranch_scc1 .LBB0_70
.LBB0_71:                               ; %Flow1087
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s17
	v_cmp_neq_f32_e64 s7, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_73
; %bb.72:
	v_div_scale_f32 v13, null, v5, v5, v6
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v6, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v6, v13, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v6, v6
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v6, v6, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v6
.LBB0_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_75
; %bb.74:
	v_div_scale_f32 v6, null, v5, v5, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v12, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v2, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v6, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v6, -v6, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v12, v15
	v_div_fixup_f32 v2, v6, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v2
.LBB0_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v6, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_77
; %bb.76:
	v_div_scale_f32 v6, null, v5, v5, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v6, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v6, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v6, -v6, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v14, v16
	v_div_fixup_f32 v1, v6, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v1
.LBB0_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_79
; %bb.78:
	v_div_scale_f32 v1, null, v5, v5, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v14, v2
	v_div_scale_f32 v14, vcc_lo, v0, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v2
	v_fma_f32 v16, -v1, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v2
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v15
	v_div_fixup_f32 v0, v1, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v0, v6, v2
	ds_bpermute_b32 v1, v7, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v8, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v9, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v11, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v0, v1
	v_mad_co_u64_u32 v[0:1], null, 0x48, s14, v[3:4]
	v_and_b32_e32 v4, 15, v6
	v_and_b32_e32 v6, 15, v13
	ds_bpermute_b32 v3, v10, v7
	v_lshl_or_b32 v4, v2, 4, v4
	v_lshl_or_b32 v6, v12, 4, v6
	v_mad_co_u64_u32 v[1:2], null, 0x48, s15, v[1:2]
	v_add_co_u32 v8, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v4.l
	v_and_b16 v2.h, 0xff, v6.l
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, 0, v1, vcc_lo
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[8:9], v2, off offset:584
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_81
; %bb.80:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v7, v3
	global_store_b64 v[0:1], v[5:6], off offset:576
.LBB0_81:                               ; %Flow1090
	s_or_b32 exec_lo, exec_lo, s16
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB0_83
; %bb.82:                               ; %.preheader902.2.i
	ds_store_2addr_b32 v34, v173, v174 offset1:1
	ds_store_2addr_b32 v34, v171, v172 offset0:2 offset1:3
	ds_store_2addr_b32 v34, v169, v170 offset0:4 offset1:5
	ds_store_2addr_b32 v34, v167, v168 offset0:6 offset1:7
	ds_store_2addr_b32 v34, v165, v166 offset0:16 offset1:17
	ds_store_2addr_b32 v34, v163, v164 offset0:18 offset1:19
	ds_store_2addr_b32 v34, v161, v162 offset0:20 offset1:21
	ds_store_2addr_b32 v34, v159, v160 offset0:22 offset1:23
.LBB0_83:                               ; %.loopexit.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v0, 16, v35
	s_mov_b32 s16, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s26, v0
	s_cbranch_execz .LBB0_110
; %bb.84:
	v_mad_co_i64_i32 v[0:1], null, s24, v0, 0
	s_mov_b32 s18, 0
	s_mov_b32 s17, exec_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v0, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v28, vcc_lo
	global_load_b128 v[40:43], v[4:5], off
	global_load_b128 v[0:3], v[25:26], off
	global_load_b128 v[8:11], v[4:5], off offset:16
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v4, 0xbfb8aa3b, v40
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, 0xbfb8aa3b, v40, -v4
	v_rndne_f32_e32 v6, v4
	v_dual_fmac_f32 v5, 0xb2a5705f, v40 :: v_dual_sub_f32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v40
	global_load_b128 v[4:7], v[25:26], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0x7f800000, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v14, 1.0, v12
	v_mul_f32_e32 v12, 0xbfb8aa3b, v41
	v_div_scale_f32 v15, null, v14, v14, v40
	v_div_scale_f32 v29, vcc_lo, v40, v14, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v17, 0xbfb8aa3b, v41, -v12
	v_rndne_f32_e32 v18, v12
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v17, 0xb2a5705f, v41 :: v_dual_sub_f32 v12, v12, v18
	v_cvt_i32_f32_e32 v18, v18
	v_add_f32_e32 v12, v12, v17
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v13, -v15, v16, 1.0
	v_exp_f32_e32 v17, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v13, v16
	v_mul_f32_e32 v30, v29, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_ldexp_f32 v17, v17, v18
	v_fma_f32 v13, -v15, v30, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v13, v16
	ds_load_2addr_b32 v[12:13], v33 offset1:1
	v_fma_f32 v15, -v15, v30, v29
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v15, v16, v30
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v41
	ds_load_2addr_b32 v[44:45], v33 offset0:2 offset1:3
	ds_load_2addr_b32 v[46:47], v33 offset0:4 offset1:5
	ds_load_2addr_b32 v[29:30], v33 offset0:6 offset1:7
	v_div_fixup_f32 v14, v15, v14, v40
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, 0, v17, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v41
	s_wait_dscnt 0x3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v36, v12, v14 :: v_dual_cndmask_b32 v15, 0x7f800000, v16
	v_mul_f32_e32 v16, 0xbfb8aa3b, v42
	s_wait_loadcnt 0x2
	v_div_scale_f32 v14, null, v0, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, 1.0, v15
	v_fma_f32 v37, 0xbfb8aa3b, v42, -v16
	v_rndne_f32_e32 v38, v16
	v_div_scale_f32 v48, vcc_lo, v36, v0, v36
	v_rcp_f32_e32 v17, v14
	v_div_scale_f32 v15, null, v12, v12, v41
	v_fmac_f32_e32 v37, 0xb2a5705f, v42
	v_sub_f32_e32 v16, v16, v38
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v15
	v_add_f32_e32 v16, v16, v37
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v14, v17, 1.0
	v_exp_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v17, v39, v17
	v_fma_f32 v40, -v15, v18, 1.0
	v_div_scale_f32 v39, s7, v41, v12, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v40, v18
	v_dual_mul_f32 v40, v48, v17 :: v_dual_mul_f32 v49, v39, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v37, -v14, v40, v48
	v_fma_f32 v50, -v15, v49, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v40, v37, v17
	v_cvt_i32_f32_e32 v37, v38
	v_fmac_f32_e32 v49, v50, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v14, -v14, v40, v48
	v_ldexp_f32 v16, v16, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v15, -v15, v49, v39
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v14, v17, v40
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v15, v18, v49
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v12, v14, v12, v41
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v16, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v42
	v_mul_f32_e32 v38, v13, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v14, 0x7f800000, v15, vcc_lo
	v_mul_f32_e32 v15, 0xbfb8aa3b, v43
	v_div_scale_f32 v13, null, v1, v1, v38
	v_div_scale_f32 v48, vcc_lo, v38, v1, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v12, 1.0, v14
	v_fma_f32 v18, 0xbfb8aa3b, v43, -v15
	v_rndne_f32_e32 v39, v15
	v_rcp_f32_e32 v16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v14, null, v12, v12, v42
	v_fmac_f32_e32 v18, 0xb2a5705f, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v15, v15, v39
	v_cvt_i32_f32_e32 v39, v39
	v_rcp_f32_e32 v17, v14
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v13, v16, 1.0
	v_dual_add_f32 v15, v15, v18 :: v_dual_fmac_f32 v16, v40, v16
	v_div_scale_f32 v40, s7, v42, v12, v42
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v14, v17, 1.0
	v_exp_f32_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v41, v17
	v_mul_f32_e32 v41, v48, v16
	v_mul_f32_e32 v18, v40, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v49, -v13, v41, v48
	v_ldexp_f32 v15, v15, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v14, v18, v40
	v_fmac_f32_e32 v41, v49, v16
	v_div_fixup_f32 v0, v37, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, v50, v17
	v_fma_f32 v13, -v13, v41, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v14, v18, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v39, v13, v16, v41
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v14, v17, v18
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v12, v13, v12, v42
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v43
	s_wait_dscnt 0x2
	v_mul_f32_e32 v40, v44, v12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, 0xbfb8aa3b, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, 0x7f800000, v14, vcc_lo
	v_div_scale_f32 v42, null, v2, v2, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v16, v12
	v_add_f32_e32 v44, 1.0, v13
	v_fma_f32 v13, 0xbfb8aa3b, v8, -v12
	v_div_scale_f32 v51, vcc_lo, v40, v2, v40
	v_rcp_f32_e32 v49, v42
	v_sub_f32_e32 v12, v12, v16
	v_div_scale_f32 v48, null, v44, v44, v43
	v_fmac_f32_e32 v13, 0xb2a5705f, v8
	v_div_scale_f32 v52, s7, v43, v44, v43
	v_cvt_i32_f32_e32 v56, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v50, v48
	v_add_f32_e32 v17, v12, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v42, v49, 1.0
	v_exp_f32_e32 v55, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v49, v14, v49
	v_fma_f32 v15, -v48, v50, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v53, v51, v49
	v_fmac_f32_e32 v50, v15, v50
	global_load_b128 v[12:15], v[23:24], off offset:4
	v_fma_f32 v18, -v42, v53, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v54, v52, v50 :: v_dual_fmac_f32 v53, v18, v49
	v_div_fixup_f32 v1, v39, v1, v38
	v_fma_f32 v17, -v48, v54, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v42, -v42, v53, v51
	v_ldexp_f32 v51, v55, v56
	v_fmac_f32_e32 v54, v17, v50
	s_clause 0x1
	global_load_b32 v41, v[21:22], off
	global_load_b96 v[16:18], v[23:24], off offset:20
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v42, v42, v49, v53
	s_mov_b32 vcc_lo, s7
	v_fma_f32 v48, -v48, v54, v52
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v48, v48, v50, v54
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v8
	v_div_fixup_f32 v43, v48, v44, v43
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, 0, v51, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v43, v45, v43 :: v_dual_cndmask_b32 v44, 0x7f800000, v49
	v_mul_f32_e32 v49, 0xbfb8aa3b, v9
	v_div_scale_f32 v45, null, v3, v3, v43
	v_div_scale_f32 v56, vcc_lo, v43, v3, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v44, 1.0, v44
	v_fma_f32 v52, 0xbfb8aa3b, v9, -v49
	v_rndne_f32_e32 v53, v49
	v_rcp_f32_e32 v50, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v48, null, v44, v44, v8
	v_fmac_f32_e32 v52, 0xb2a5705f, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v49, v49, v53
	v_cvt_i32_f32_e32 v53, v53
	v_rcp_f32_e32 v51, v48
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v54, -v45, v50, 1.0
	v_dual_add_f32 v49, v49, v52 :: v_dual_fmac_f32 v50, v54, v50
	v_div_scale_f32 v54, s7, v8, v44, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v48, v51, 1.0
	v_exp_f32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v51, v55, v51
	v_dual_mul_f32 v55, v56, v50 :: v_dual_mul_f32 v52, v54, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v57, -v45, v55, v56
	v_ldexp_f32 v49, v49, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v48, v52, v54
	v_dual_fmac_f32 v55, v57, v50 :: v_dual_fmac_f32 v52, v58, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v45, -v45, v55, v56
	v_fma_f32 v48, -v48, v52, v54
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v53, v45, v50, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v48, v51, v52
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v3, v53, v3, v43
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v8, v45, v44, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, 0, v49, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	s_wait_dscnt 0x1
	v_mul_f32_e32 v8, v46, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0x7f800000, v48, vcc_lo
	v_mul_f32_e32 v48, 0xbfb8aa3b, v10
	s_wait_loadcnt 0x3
	v_div_scale_f32 v45, null, v4, v4, v8
	v_div_scale_f32 v56, vcc_lo, v8, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v49, v45
	v_fma_f32 v54, -v45, v49, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v49, v54, v49
	v_div_fixup_f32 v2, v42, v2, v40
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v3, v3, v14 :: v_dual_add_f32 v44, 1.0, v44
	v_mul_f32_e32 v1, v1, v12
	v_fma_f32 v51, 0xbfb8aa3b, v10, -v48
	v_rndne_f32_e32 v52, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v46, null, v44, v44, v9
	v_div_scale_f32 v54, s7, v9, v44, v9
	v_dual_fmac_f32 v51, 0xb2a5705f, v10 :: v_dual_sub_f32 v48, v48, v52
	v_cvt_i32_f32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v50, v46
	v_add_f32_e32 v48, v48, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v48, v48
	v_fma_f32 v55, -v46, v50, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_dual_fmac_f32 v50, v55, v50 :: v_dual_mul_f32 v55, v56, v49
	v_ldexp_f32 v48, v48, v52
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v51, v54, v50
	v_fma_f32 v57, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v58, -v46, v51, v54
	v_fmac_f32_e32 v55, v57, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v51, v58, v50
	v_fma_f32 v45, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v51, v54
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v52, v45, v49, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v46, v50, v51
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v9, v45, v44, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v48, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v9, v47, v9 :: v_dual_cndmask_b32 v44, 0x7f800000, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v55, null, v5, v5, v9
	v_div_scale_f32 v60, vcc_lo, v9, v5, v9
	v_add_f32_e32 v54, 1.0, v44
	v_mul_f32_e32 v44, 0xbfb8aa3b, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v57, v55
	v_div_scale_f32 v56, null, v54, v54, v10
	v_div_scale_f32 v61, s7, v10, v54, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v45, 0xbfb8aa3b, v11, -v44
	v_rndne_f32_e32 v59, v44
	v_rcp_f32_e32 v58, v56
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v55, v57, 1.0
	v_fmac_f32_e32 v45, 0xb2a5705f, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v59
	v_cvt_i32_f32_e32 v59, v59
	v_fmac_f32_e32 v57, v46, v57
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v47, -v56, v58, 1.0
	v_add_f32_e32 v48, v44, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v62, v60, v57
	v_fmac_f32_e32 v58, v47, v58
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v65, v48
	s_clause 0x1
	global_load_b128 v[44:47], v[19:20], off
	global_load_b128 v[48:51], v[19:20], off offset:16
	v_fma_f32 v64, -v55, v62, v60
	v_dual_mul_f32 v63, v61, v58 :: v_dual_fmac_f32 v62, v64, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v66, -v56, v63, v61
	v_ldexp_f32 v59, v65, v59
	v_fma_f32 v55, -v55, v62, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v63, v66, v58
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v55, v55, v57, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v63, v61
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v5, v55, v5, v9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v56, v56, v58, v63
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	s_wait_loadcnt 0x3
	v_fma_f32 v9, v0, v41, v1
	v_fma_f32 v0, v0, v41, -v1
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v5, v5, v16
	v_div_fixup_f32 v10, v56, v54, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v57, 0, v59, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	v_fma_f32 v1, v2, v13, -v3
	s_wait_dscnt 0x0
	v_mul_f32_e32 v10, v29, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v54, 0x7f800000, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v61, vcc_lo, v10, v6, v10
	v_add_f32_e32 v29, 1.0, v54
	v_div_scale_f32 v54, null, v6, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v56, null, v29, v29, v11
	v_rcp_f32_e32 v57, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v58, v56
	v_fma_f32 v59, -v54, v57, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v60, -v56, v58, 1.0
	v_fmac_f32_e32 v57, v59, v57
	v_div_scale_f32 v59, s7, v11, v29, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v58, v60, v58
	v_mul_f32_e32 v60, v61, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v62, v59, v58
	v_fma_f32 v63, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v64, -v56, v62, v59
	v_fmac_f32_e32 v60, v63, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v62, v64, v58
	v_fma_f32 v54, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v62, v59
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v54, v54, v57, v60
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v56, v56, v58, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v6, v54, v6, v10
	v_div_fixup_f32 v11, v56, v29, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v11, v30, v11
	v_div_scale_f32 v29, null, v7, v7, v11
	v_div_scale_f32 v57, vcc_lo, v11, v7, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v30, v29
	v_fma_f32 v56, -v29, v30, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v56, v30
	v_mul_f32_e32 v56, v57, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v29, v56, v57
	v_fmac_f32_e32 v56, v58, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v12, -v29, v56, v57
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v12, v12, v30, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v7, v12, v7, v11
	v_fma_f32 v11, v2, v13, v3
	v_mul_f32_e32 v7, v7, v18
	v_div_fixup_f32 v4, v52, v4, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v9, v11
	v_fma_f32 v8, v6, v17, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v3, v4, v15, v5
	v_fma_f32 v4, v4, v15, -v5
	v_fma_f32 v5, v6, v17, -v7
	v_dual_sub_f32 v6, v9, v11 :: v_dual_add_f32 v7, v0, v1
	v_dual_add_f32 v9, v3, v8 :: v_dual_sub_f32 v0, v0, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v4, v5
	v_sub_f32_e32 v3, v3, v8
	v_dual_add_f32 v8, v7, v1 :: v_dual_sub_f32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v7, v6, v3
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v7, -v7, v7, s1
	v_cndmask_b32_e64 v1, -v1, v1, s1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v7, -v7, v7, s5
	v_cndmask_b32_e64 v1, -v1, v1, s5
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v7, -v7, v7, s4
	v_cndmask_b32_e64 v1, -v1, v1, s4
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v7, -v7, v7, s2
	v_cndmask_b32_e64 v1, -v1, v1, s2
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	v_sub_f32_e32 v3, v6, v3
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v3, -v3, v3, s1
	v_cndmask_b32_e64 v7, -v7, v7, s3
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v4, v4, v5 :: v_dual_add_f32 v7, v7, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_mul_f32_e32 v7, 0x3d800000, v7
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v3, -v3, v3, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v3, v14
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v3, -v3, v3, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	v_dual_add_f32 v5, v2, v9 :: v_dual_sub_f32 v2, v2, v9
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v8, -v8, v8, s1
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v3, -v3, v3, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v6, v0, v4
	v_sub_f32_e32 v0, v0, v4
	ds_swizzle_b32 v4, v5 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v14, v1 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s1
	v_cndmask_b32_e64 v1, -v1, v1, s3
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v6, -v6, v6, s1
	s_wait_dscnt 0x1
	v_add_f32_e32 v1, v1, v14
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v8, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v1, 0x3d800000, v1
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, -v5, v5, s5
	v_cndmask_b32_e64 v4, -v4, v4, s5
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v18, v1, v49
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v5, -v5, v5, s4
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, -v5, v5, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v2, -v2, v2, s1
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v0, -v0, v0, s1
	v_xor_b32_e32 v9, 16, v10
	v_mul_f32_e32 v5, 0x3d800000, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v14, v5, v45
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v0, -v0, v0, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v0, -v0, v0, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v0, -v0, v0, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v3 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v3, -v3, v3, s3
	ds_swizzle_b32 v16, v0 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v0, -v0, v0, s3
	s_wait_dscnt 0x1
	v_dual_add_f32 v3, v3, v15 :: v_dual_add_f32 v2, v2, v12
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v15, v7, v46 :: v_dual_add_f32 v0, v0, v16
	v_mul_f32_e32 v3, 0x3d800000, v3
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,2)
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s5
	v_dual_mul_f32 v0, 0x3d800000, v0 :: v_dual_mul_f32 v29, v3, v50
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, s4
	v_mul_f32_e32 v30, v0, v51
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,4)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s4
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, s2
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,8)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s2
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, s3
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	v_lshlrev_b32_e32 v12, 2, v32
	ds_swizzle_b32 v13, v2 offset:swizzle(SWAP,16)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s3
	ds_bpermute_b32 v1, v12, v14
	ds_bpermute_b32 v7, v12, v29
	ds_bpermute_b32 v8, v12, v30
	v_mul_f32_e32 v4, 0x3d800000, v4
	v_add_f32_e32 v6, v6, v11
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v2, v2, v13 :: v_dual_mul_f32 v13, v4, v44
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v6, -v6, v6, s5
	ds_bpermute_b32 v4, v12, v15
	v_mul_f32_e32 v2, 0x3d800000, v2
	ds_bpermute_b32 v0, v12, v13
	v_mul_f32_e32 v17, v2, v48
	ds_bpermute_b32 v2, v12, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, s4
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v3, v2, v0, s1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v6, -v6, v6, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	v_mul_f32_e32 v6, 0x3d800000, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, v6, v47
	ds_bpermute_b32 v6, v12, v18
	ds_bpermute_b32 v5, v12, v16
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v2, v6, v1, s1
	v_cndmask_b32_e64 v1, v7, v4, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v10, v9, vcc_lo
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v8, v5, s1
	v_max3_num_f32 v5, |v3|, |v2|, |v1|
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v7, 2, v4
	v_max_num_f32_e64 v6, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v4, v5, v6
	v_xor_b32_e32 v6, 8, v10
	ds_bpermute_b32 v5, v7, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v10, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 2, v6
	v_xor_b32_e32 v6, 4, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_cndmask_b32 v6, v10, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v4, v4, v5 :: v_dual_lshlrev_b32 v9, 2, v6
	v_xor_b32_e32 v6, 2, v10
	ds_bpermute_b32 v5, v8, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v10, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v6
	v_xor_b32_e32 v6, 1, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_cndmask_b32 v6, v10, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v4, v4, v5
	v_lshlrev_b32_e32 v10, 2, v6
	ds_bpermute_b32 v5, v9, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v11, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	v_max_num_f32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v10, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v4, v4, v5 :: v_dual_mov_b32 v5, 1.0
	v_cmpx_neq_f32_e32 0, v4
	s_cbranch_execz .LBB0_87
; %bb.85:                               ; %.preheader76.i.i.2.i
	v_div_scale_f32 v5, null, 0x40e00000, 0x40e00000, v4
	v_div_scale_f32 v37, vcc_lo, v4, 0x40e00000, v4
	s_mov_b32 s19, 0x40e00000
	s_mov_b32 s22, 0xc1000000
	v_rcp_f32_e32 v6, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v5, v6, 1.0
	v_fmac_f32_e32 v6, v36, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, v37, v6
	v_fma_f32 v38, -v5, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v38, v6
	v_fma_f32 v5, -v5, v36, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v5, v5, v6, v36
	v_mov_b32_e32 v36, 0x7149f2ca
	v_div_fixup_f32 v6, v5, 0x40e00000, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v5, 1.0 :: v_dual_mul_f32 v6, 0.5, v6
.LBB0_86:                               ; %.preheader.preheader.i.i.2.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s7, s18
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s18, 8
	v_div_scale_f32 v37, null, s19, s19, s7
	v_div_scale_f32 v38, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v39, v37
	v_xor_b32_e32 v37, 0x80000000, v37
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v40, v37, v39, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v39
	v_mul_f32_e32 v40, v38, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v37, v40, v38
	v_fmac_f32_e32 v40, v41, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v37, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v38, v39, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v37, v37, 0x40e00000, s7
	v_add_f32_e32 v37, 1.0, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v6, v37
	v_div_scale_f32 v38, null, v37, v37, v3
	v_div_scale_f32 v40, null, v37, v37, v2
	v_div_scale_f32 v42, null, v37, v37, v1
	v_div_scale_f32 v44, null, v37, v37, v0
	v_div_scale_f32 v39, vcc_lo, v3, v37, v3
	v_rcp_f32_e32 v46, v38
	v_rcp_f32_e32 v47, v40
	v_rcp_f32_e32 v48, v42
	v_rcp_f32_e32 v49, v44
	v_div_scale_f32 v41, s7, v2, v37, v2
	v_div_scale_f32 v43, s8, v1, v37, v1
	v_div_scale_f32 v45, s9, v0, v37, v0
	v_fma_f32 v50, -v38, v46, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v51, -v40, v47, 1.0
	v_fma_f32 v52, -v42, v48, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v53, -v44, v49, 1.0
	v_dual_fmac_f32 v46, v50, v46 :: v_dual_fmac_f32 v47, v51, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v48, v52, v48 :: v_dual_fmac_f32 v49, v53, v49
	v_dual_mul_f32 v50, v39, v46 :: v_dual_mul_f32 v51, v41, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v52, v43, v48 :: v_dual_mul_f32 v53, v45, v49
	v_fma_f32 v54, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v40, v51, v41
	v_fma_f32 v56, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v57, -v44, v53, v45
	v_dual_fmac_f32 v50, v54, v46 :: v_dual_fmac_f32 v51, v55, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v52, v56, v48 :: v_dual_fmac_f32 v53, v57, v49
	v_fma_f32 v38, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v40, v51, v41
	v_fma_f32 v40, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v41, -v44, v53, v45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v38, v46, v50
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v39, v39, v47, v51
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v38, v38, v37, v3
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v40, v40, v48, v52
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v39, v39, v37, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v41, v41, v49, v53
	v_rndne_f32_e32 v38, v38
	v_div_fixup_f32 v40, v40, v37, v1
	v_rndne_f32_e32 v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v41, v41, v37, v0
	v_med3_num_f32 v38, v38, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v40, v40
	v_med3_num_f32 v39, v39, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v41, v41
	v_fma_f32 v38, -v38, v37, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v40, v40, s22, 0x40e00000
	v_fma_f32 v39, -v39, v37, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v41, v41, s22, 0x40e00000
	v_fma_f32 v38, v38, v38, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v40, -v40, v37, v1
	v_fmac_f32_e32 v38, v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v41, v37, v0
	v_fmac_f32_e32 v38, v40, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v39
	ds_bpermute_b32 v39, v7, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v8, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v9, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v11, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v10, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v38, v36
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v36, v36, v38 :: v_dual_cndmask_b32 v5, v5, v37
	s_cbranch_scc1 .LBB0_86
.LBB0_87:                               ; %Flow1083
	s_or_b32 exec_lo, exec_lo, s17
	v_cmp_neq_f32_e64 s7, 0, v4
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v36, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_89
; %bb.88:
	v_div_scale_f32 v4, null, v5, v5, v3
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v36, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v4, v36, 1.0
	v_fmac_f32_e32 v36, v37, v36
	v_div_scale_f32 v37, vcc_lo, v3, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v36
	v_fma_f32 v39, -v4, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v36
	v_fma_f32 v4, -v4, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v36, v38
	v_div_fixup_f32 v3, v4, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v3, v3
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v3, v3, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v36, v3
.LBB0_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_91
; %bb.90:
	v_div_scale_f32 v3, null, v5, v5, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v6, v4
	v_div_scale_f32 v6, vcc_lo, v2, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v6, v4
	v_fma_f32 v38, -v3, v37, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v38, v4
	v_fma_f32 v3, -v3, v37, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v37
	v_div_fixup_f32 v2, v3, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v2
.LBB0_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v37, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_93
; %bb.92:
	v_div_scale_f32 v3, null, v5, v5, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v37, v4
	v_div_scale_f32 v37, vcc_lo, v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v4
	v_fma_f32 v39, -v3, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v4
	v_fma_f32 v3, -v3, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v38
	v_div_fixup_f32 v1, v3, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v37, v1
.LBB0_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_95
; %bb.94:
	v_div_scale_f32 v1, null, v5, v5, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v3, v2
	v_div_scale_f32 v3, vcc_lo, v0, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v4, v3, v2
	v_fma_f32 v38, -v1, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v38, v2
	v_fma_f32 v1, -v1, v4, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v4
	v_div_fixup_f32 v0, v1, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v6, v36
	v_mad_co_i64_i32 v[3:4], null, 0x48, v35, s[12:13]
	v_and_b32_e32 v39, 15, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add3_u32 v0, v0, v37, v2
	v_and_b32_e32 v37, 15, v37
	v_lshl_or_b32 v6, v6, 4, v39
	ds_bpermute_b32 v1, v7, v0
	v_lshl_or_b32 v37, v2, 4, v37
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v8, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v9, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v11, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v38, v0, v1
	v_mad_co_u64_u32 v[0:1], null, 0x48, s10, v[3:4]
	ds_bpermute_b32 v36, v10, v38
	v_mad_co_u64_u32 v[1:2], null, 0x48, s11, v[1:2]
	v_lshlrev_b16 v2.l, 8, v37.l
	v_and_b16 v2.h, 0xff, v6.l
	v_add_co_u32 v39, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, 0, v1, vcc_lo
	global_store_b16 v[39:40], v2, off offset:1160
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB0_97
; %bb.96:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v38, v36
	global_store_b64 v[0:1], v[5:6], off offset:1152
.LBB0_97:                               ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit.i.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_bpermute_b32 v0, v12, v13 offset:64
	ds_bpermute_b32 v1, v12, v14 offset:64
	ds_bpermute_b32 v5, v12, v15 offset:64
	ds_bpermute_b32 v13, v12, v16 offset:64
	ds_bpermute_b32 v2, v12, v17 offset:64
	ds_bpermute_b32 v14, v12, v18 offset:64
	ds_bpermute_b32 v15, v12, v29 offset:64
	ds_bpermute_b32 v12, v12, v30 offset:64
	s_mov_b32 s18, 0
	s_mov_b32 s17, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v6, v2, v0, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v2, v14, v1, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v1, v15, v5, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v12, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v5, |v6|, |v2|, |v1|
	v_max_num_f32_e64 v12, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v7, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v8, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v9, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v11, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v10, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_dual_max_num_f32 v12, v5, v12 :: v_dual_mov_b32 v5, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_100
; %bb.98:                               ; %.preheader76.i.1.i.2.i
	v_div_scale_f32 v5, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s19, 0x40e00000
	s_mov_b32 s22, 0xc1000000
	v_rcp_f32_e32 v13, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v5, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v5, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v5, -v5, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v5, v5, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v5, 0x40e00000, v12
	v_mov_b32_e32 v5, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_99:                               ; %.preheader.preheader.i.1.i.2.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s7, s18
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s18, 8
	v_div_scale_f32 v15, null, s19, s19, s7
	v_div_scale_f32 v16, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v29, v15, v18, v16
	v_fmac_f32_e32 v18, v29, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s7
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v18, null, v15, v15, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v40, v18
	v_fma_f32 v44, -v18, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v44, v40
	v_div_scale_f32 v16, null, v15, v15, v6
	v_div_scale_f32 v30, null, v15, v15, v1
	v_div_scale_f32 v37, null, v15, v15, v0
	v_div_scale_f32 v17, vcc_lo, v6, v15, v6
	v_div_scale_f32 v29, s7, v2, v15, v2
	v_rcp_f32_e32 v39, v16
	v_rcp_f32_e32 v41, v30
	v_rcp_f32_e32 v42, v37
	v_div_scale_f32 v38, s9, v0, v15, v0
	v_div_scale_f32 v36, s8, v1, v15, v1
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v43, -v16, v39, 1.0
	v_fma_f32 v45, -v30, v41, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v37, v42, 1.0
	v_fmac_f32_e32 v39, v43, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v41, v45, v41 :: v_dual_fmac_f32 v42, v46, v42
	v_mul_f32_e32 v44, v29, v40
	v_mul_f32_e32 v43, v17, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v45, v36, v41 :: v_dual_mul_f32 v46, v38, v42
	v_fma_f32 v48, -v18, v44, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v16, v43, v17
	v_fma_f32 v49, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v37, v46, v38
	v_dual_fmac_f32 v44, v48, v40 :: v_dual_fmac_f32 v43, v47, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v45, v49, v41 :: v_dual_fmac_f32 v46, v50, v42
	v_fma_f32 v16, -v16, v43, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v44, v29
	v_fma_f32 v18, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v29, -v37, v46, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v39, v43
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v40, v44
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v16, v16, v15, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v41, v45
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v17, v17, v15, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v29, v29, v42, v46
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v1
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v29, v29, v15, v0
	v_med3_num_f32 v16, v16, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v29, v29
	v_fma_f32 v16, -v16, v15, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s22, 0x40e00000
	v_fma_f32 v17, -v17, v15, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v29, v29, s22, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v1
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v29, v15, v0
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v11, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v10, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v14, v16 :: v_dual_cndmask_b32 v5, v5, v15
	s_cbranch_scc1 .LBB0_99
.LBB0_100:                              ; %Flow1081
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s17
	v_cmp_neq_f32_e64 s7, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_div_scale_f32 v13, null, v5, v5, v6
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v6, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v6, v13, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v6, v6
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v6, v6, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v6
.LBB0_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_div_scale_f32 v6, null, v5, v5, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v12, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v2, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v6, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v6, -v6, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v12, v15
	v_div_fixup_f32 v2, v6, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v2
.LBB0_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v6, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_div_scale_f32 v6, null, v5, v5, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v6, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v6, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v6, -v6, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v14, v16
	v_div_fixup_f32 v1, v6, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v1
.LBB0_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_div_scale_f32 v1, null, v5, v5, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v14, v2
	v_div_scale_f32 v14, vcc_lo, v0, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v2
	v_fma_f32 v16, -v1, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v2
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v15
	v_div_fixup_f32 v0, v1, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v0, v6, v2
	ds_bpermute_b32 v1, v7, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v8, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v9, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v11, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v0, v1
	v_mad_co_u64_u32 v[0:1], null, 0x48, s14, v[3:4]
	v_and_b32_e32 v4, 15, v6
	v_and_b32_e32 v6, 15, v13
	ds_bpermute_b32 v3, v10, v7
	v_lshl_or_b32 v4, v2, 4, v4
	v_lshl_or_b32 v6, v12, 4, v6
	v_mad_co_u64_u32 v[1:2], null, 0x48, s15, v[1:2]
	v_add_co_u32 v8, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v4.l
	v_and_b16 v2.h, 0xff, v6.l
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, 0, v1, vcc_lo
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[8:9], v2, off offset:1160
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_110
; %bb.109:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v7, v3
	global_store_b64 v[0:1], v[5:6], off offset:1152
.LBB0_110:                              ; %Flow1084
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s16
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_112
; %bb.111:                              ; %.preheader902.3.i
	ds_store_2addr_b32 v34, v173, v174 offset1:1
	ds_store_2addr_b32 v34, v171, v172 offset0:2 offset1:3
	ds_store_2addr_b32 v34, v169, v170 offset0:4 offset1:5
	ds_store_2addr_b32 v34, v167, v168 offset0:6 offset1:7
	ds_store_2addr_b32 v34, v165, v166 offset0:16 offset1:17
	ds_store_2addr_b32 v34, v163, v164 offset0:18 offset1:19
	ds_store_2addr_b32 v34, v161, v162 offset0:20 offset1:21
	ds_store_2addr_b32 v34, v159, v160 offset0:22 offset1:23
.LBB0_112:                              ; %.loopexit.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v0, 24, v35
	s_mov_b32 s16, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s26, v0
	s_cbranch_execz .LBB0_139
; %bb.113:
	v_mad_co_i64_i32 v[0:1], null, s24, v0, 0
	s_mov_b32 s18, 0
	s_mov_b32 s17, exec_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v0, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v28, vcc_lo
	global_load_b128 v[40:43], v[4:5], off
	global_load_b128 v[0:3], v[25:26], off
	global_load_b128 v[8:11], v[4:5], off offset:16
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v4, 0xbfb8aa3b, v40
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, 0xbfb8aa3b, v40, -v4
	v_rndne_f32_e32 v6, v4
	v_dual_fmac_f32 v5, 0xb2a5705f, v40 :: v_dual_sub_f32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v40
	global_load_b128 v[4:7], v[25:26], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0x7f800000, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v14, 1.0, v12
	v_mul_f32_e32 v12, 0xbfb8aa3b, v41
	v_div_scale_f32 v15, null, v14, v14, v40
	v_div_scale_f32 v29, vcc_lo, v40, v14, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v17, 0xbfb8aa3b, v41, -v12
	v_rndne_f32_e32 v18, v12
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v17, 0xb2a5705f, v41 :: v_dual_sub_f32 v12, v12, v18
	v_cvt_i32_f32_e32 v18, v18
	v_add_f32_e32 v12, v12, v17
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v13, -v15, v16, 1.0
	v_exp_f32_e32 v17, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v13, v16
	v_mul_f32_e32 v30, v29, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_ldexp_f32 v17, v17, v18
	v_fma_f32 v13, -v15, v30, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v13, v16
	ds_load_2addr_b32 v[12:13], v33 offset1:1
	v_fma_f32 v15, -v15, v30, v29
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v15, v16, v30
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v41
	ds_load_2addr_b32 v[44:45], v33 offset0:2 offset1:3
	ds_load_2addr_b32 v[46:47], v33 offset0:4 offset1:5
	ds_load_2addr_b32 v[29:30], v33 offset0:6 offset1:7
	v_div_fixup_f32 v14, v15, v14, v40
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, 0, v17, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v41
	s_wait_dscnt 0x3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v36, v12, v14 :: v_dual_cndmask_b32 v15, 0x7f800000, v16
	v_mul_f32_e32 v16, 0xbfb8aa3b, v42
	s_wait_loadcnt 0x2
	v_div_scale_f32 v14, null, v0, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, 1.0, v15
	v_fma_f32 v37, 0xbfb8aa3b, v42, -v16
	v_rndne_f32_e32 v38, v16
	v_div_scale_f32 v48, vcc_lo, v36, v0, v36
	v_rcp_f32_e32 v17, v14
	v_div_scale_f32 v15, null, v12, v12, v41
	v_fmac_f32_e32 v37, 0xb2a5705f, v42
	v_sub_f32_e32 v16, v16, v38
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v15
	v_add_f32_e32 v16, v16, v37
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v14, v17, 1.0
	v_exp_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v17, v39, v17
	v_fma_f32 v40, -v15, v18, 1.0
	v_div_scale_f32 v39, s7, v41, v12, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v40, v18
	v_dual_mul_f32 v40, v48, v17 :: v_dual_mul_f32 v49, v39, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v37, -v14, v40, v48
	v_fma_f32 v50, -v15, v49, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v40, v37, v17
	v_cvt_i32_f32_e32 v37, v38
	v_fmac_f32_e32 v49, v50, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v14, -v14, v40, v48
	v_ldexp_f32 v16, v16, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v15, -v15, v49, v39
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v14, v17, v40
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v15, v18, v49
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v12, v14, v12, v41
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v16, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v42
	v_mul_f32_e32 v38, v13, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v14, 0x7f800000, v15, vcc_lo
	v_mul_f32_e32 v15, 0xbfb8aa3b, v43
	v_div_scale_f32 v13, null, v1, v1, v38
	v_div_scale_f32 v48, vcc_lo, v38, v1, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v12, 1.0, v14
	v_fma_f32 v18, 0xbfb8aa3b, v43, -v15
	v_rndne_f32_e32 v39, v15
	v_rcp_f32_e32 v16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v14, null, v12, v12, v42
	v_fmac_f32_e32 v18, 0xb2a5705f, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v15, v15, v39
	v_cvt_i32_f32_e32 v39, v39
	v_rcp_f32_e32 v17, v14
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v13, v16, 1.0
	v_dual_add_f32 v15, v15, v18 :: v_dual_fmac_f32 v16, v40, v16
	v_div_scale_f32 v40, s7, v42, v12, v42
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v14, v17, 1.0
	v_exp_f32_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v41, v17
	v_mul_f32_e32 v41, v48, v16
	v_mul_f32_e32 v18, v40, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v49, -v13, v41, v48
	v_ldexp_f32 v15, v15, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v14, v18, v40
	v_fmac_f32_e32 v41, v49, v16
	v_div_fixup_f32 v0, v37, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, v50, v17
	v_fma_f32 v13, -v13, v41, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v14, v18, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v39, v13, v16, v41
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v14, v17, v18
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v12, v13, v12, v42
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v43
	s_wait_dscnt 0x2
	v_mul_f32_e32 v40, v44, v12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, 0xbfb8aa3b, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, 0x7f800000, v14, vcc_lo
	v_div_scale_f32 v42, null, v2, v2, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v16, v12
	v_add_f32_e32 v44, 1.0, v13
	v_fma_f32 v13, 0xbfb8aa3b, v8, -v12
	v_div_scale_f32 v51, vcc_lo, v40, v2, v40
	v_rcp_f32_e32 v49, v42
	v_sub_f32_e32 v12, v12, v16
	v_div_scale_f32 v48, null, v44, v44, v43
	v_fmac_f32_e32 v13, 0xb2a5705f, v8
	v_div_scale_f32 v52, s7, v43, v44, v43
	v_cvt_i32_f32_e32 v56, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v50, v48
	v_add_f32_e32 v17, v12, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v42, v49, 1.0
	v_exp_f32_e32 v55, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v49, v14, v49
	v_fma_f32 v15, -v48, v50, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v53, v51, v49
	v_fmac_f32_e32 v50, v15, v50
	global_load_b128 v[12:15], v[23:24], off offset:4
	v_fma_f32 v18, -v42, v53, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v54, v52, v50 :: v_dual_fmac_f32 v53, v18, v49
	v_div_fixup_f32 v1, v39, v1, v38
	v_fma_f32 v17, -v48, v54, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v42, -v42, v53, v51
	v_ldexp_f32 v51, v55, v56
	v_fmac_f32_e32 v54, v17, v50
	s_clause 0x1
	global_load_b32 v41, v[21:22], off
	global_load_b96 v[16:18], v[23:24], off offset:20
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v42, v42, v49, v53
	s_mov_b32 vcc_lo, s7
	v_fma_f32 v48, -v48, v54, v52
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v48, v48, v50, v54
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v8
	v_div_fixup_f32 v43, v48, v44, v43
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, 0, v51, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v43, v45, v43 :: v_dual_cndmask_b32 v44, 0x7f800000, v49
	v_mul_f32_e32 v49, 0xbfb8aa3b, v9
	v_div_scale_f32 v45, null, v3, v3, v43
	v_div_scale_f32 v56, vcc_lo, v43, v3, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v44, 1.0, v44
	v_fma_f32 v52, 0xbfb8aa3b, v9, -v49
	v_rndne_f32_e32 v53, v49
	v_rcp_f32_e32 v50, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v48, null, v44, v44, v8
	v_fmac_f32_e32 v52, 0xb2a5705f, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v49, v49, v53
	v_cvt_i32_f32_e32 v53, v53
	v_rcp_f32_e32 v51, v48
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v54, -v45, v50, 1.0
	v_dual_add_f32 v49, v49, v52 :: v_dual_fmac_f32 v50, v54, v50
	v_div_scale_f32 v54, s7, v8, v44, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v48, v51, 1.0
	v_exp_f32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v51, v55, v51
	v_dual_mul_f32 v55, v56, v50 :: v_dual_mul_f32 v52, v54, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v57, -v45, v55, v56
	v_ldexp_f32 v49, v49, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v48, v52, v54
	v_dual_fmac_f32 v55, v57, v50 :: v_dual_fmac_f32 v52, v58, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v45, -v45, v55, v56
	v_fma_f32 v48, -v48, v52, v54
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v53, v45, v50, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v48, v51, v52
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v3, v53, v3, v43
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v8, v45, v44, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, 0, v49, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	s_wait_dscnt 0x1
	v_mul_f32_e32 v8, v46, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0x7f800000, v48, vcc_lo
	v_mul_f32_e32 v48, 0xbfb8aa3b, v10
	s_wait_loadcnt 0x3
	v_div_scale_f32 v45, null, v4, v4, v8
	v_div_scale_f32 v56, vcc_lo, v8, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v49, v45
	v_fma_f32 v54, -v45, v49, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v49, v54, v49
	v_div_fixup_f32 v2, v42, v2, v40
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v3, v3, v14 :: v_dual_add_f32 v44, 1.0, v44
	v_mul_f32_e32 v1, v1, v12
	v_fma_f32 v51, 0xbfb8aa3b, v10, -v48
	v_rndne_f32_e32 v52, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v46, null, v44, v44, v9
	v_div_scale_f32 v54, s7, v9, v44, v9
	v_dual_fmac_f32 v51, 0xb2a5705f, v10 :: v_dual_sub_f32 v48, v48, v52
	v_cvt_i32_f32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v50, v46
	v_add_f32_e32 v48, v48, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v48, v48
	v_fma_f32 v55, -v46, v50, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_dual_fmac_f32 v50, v55, v50 :: v_dual_mul_f32 v55, v56, v49
	v_ldexp_f32 v48, v48, v52
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v51, v54, v50
	v_fma_f32 v57, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v58, -v46, v51, v54
	v_fmac_f32_e32 v55, v57, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v51, v58, v50
	v_fma_f32 v45, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v51, v54
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v52, v45, v49, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v46, v50, v51
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v9, v45, v44, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v48, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v9, v47, v9 :: v_dual_cndmask_b32 v44, 0x7f800000, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v55, null, v5, v5, v9
	v_div_scale_f32 v60, vcc_lo, v9, v5, v9
	v_add_f32_e32 v54, 1.0, v44
	v_mul_f32_e32 v44, 0xbfb8aa3b, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v57, v55
	v_div_scale_f32 v56, null, v54, v54, v10
	v_div_scale_f32 v61, s7, v10, v54, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v45, 0xbfb8aa3b, v11, -v44
	v_rndne_f32_e32 v59, v44
	v_rcp_f32_e32 v58, v56
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v55, v57, 1.0
	v_fmac_f32_e32 v45, 0xb2a5705f, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v59
	v_cvt_i32_f32_e32 v59, v59
	v_fmac_f32_e32 v57, v46, v57
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v47, -v56, v58, 1.0
	v_add_f32_e32 v48, v44, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v62, v60, v57
	v_fmac_f32_e32 v58, v47, v58
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v65, v48
	s_clause 0x1
	global_load_b128 v[44:47], v[19:20], off
	global_load_b128 v[48:51], v[19:20], off offset:16
	v_fma_f32 v64, -v55, v62, v60
	v_dual_mul_f32 v63, v61, v58 :: v_dual_fmac_f32 v62, v64, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v66, -v56, v63, v61
	v_ldexp_f32 v59, v65, v59
	v_fma_f32 v55, -v55, v62, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v63, v66, v58
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v55, v55, v57, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v63, v61
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v5, v55, v5, v9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v56, v56, v58, v63
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	s_wait_loadcnt 0x3
	v_fma_f32 v9, v0, v41, v1
	v_fma_f32 v0, v0, v41, -v1
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v5, v5, v16
	v_div_fixup_f32 v10, v56, v54, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v57, 0, v59, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	v_fma_f32 v1, v2, v13, -v3
	s_wait_dscnt 0x0
	v_mul_f32_e32 v10, v29, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v54, 0x7f800000, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v61, vcc_lo, v10, v6, v10
	v_add_f32_e32 v29, 1.0, v54
	v_div_scale_f32 v54, null, v6, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v56, null, v29, v29, v11
	v_rcp_f32_e32 v57, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v58, v56
	v_fma_f32 v59, -v54, v57, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v60, -v56, v58, 1.0
	v_fmac_f32_e32 v57, v59, v57
	v_div_scale_f32 v59, s7, v11, v29, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v58, v60, v58
	v_mul_f32_e32 v60, v61, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v62, v59, v58
	v_fma_f32 v63, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v64, -v56, v62, v59
	v_fmac_f32_e32 v60, v63, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v62, v64, v58
	v_fma_f32 v54, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v62, v59
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v54, v54, v57, v60
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v56, v56, v58, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v6, v54, v6, v10
	v_div_fixup_f32 v11, v56, v29, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v11, v30, v11
	v_div_scale_f32 v29, null, v7, v7, v11
	v_div_scale_f32 v57, vcc_lo, v11, v7, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v30, v29
	v_fma_f32 v56, -v29, v30, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v56, v30
	v_mul_f32_e32 v56, v57, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v29, v56, v57
	v_fmac_f32_e32 v56, v58, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v12, -v29, v56, v57
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v12, v12, v30, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v7, v12, v7, v11
	v_fma_f32 v11, v2, v13, v3
	v_mul_f32_e32 v7, v7, v18
	v_div_fixup_f32 v4, v52, v4, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v9, v11
	v_fma_f32 v8, v6, v17, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v3, v4, v15, v5
	v_fma_f32 v4, v4, v15, -v5
	v_fma_f32 v5, v6, v17, -v7
	v_dual_sub_f32 v6, v9, v11 :: v_dual_add_f32 v7, v0, v1
	v_dual_add_f32 v9, v3, v8 :: v_dual_sub_f32 v0, v0, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v4, v5
	v_sub_f32_e32 v3, v3, v8
	v_dual_add_f32 v8, v7, v1 :: v_dual_sub_f32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v7, v6, v3
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v7, -v7, v7, s1
	v_cndmask_b32_e64 v1, -v1, v1, s1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v7, -v7, v7, s5
	v_cndmask_b32_e64 v1, -v1, v1, s5
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v7, -v7, v7, s4
	v_cndmask_b32_e64 v1, -v1, v1, s4
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v7, -v7, v7, s2
	v_cndmask_b32_e64 v1, -v1, v1, s2
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	v_sub_f32_e32 v3, v6, v3
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v3, -v3, v3, s1
	v_cndmask_b32_e64 v7, -v7, v7, s3
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v4, v4, v5 :: v_dual_add_f32 v7, v7, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_mul_f32_e32 v7, 0x3d800000, v7
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v3, -v3, v3, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v3, v14
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v3, -v3, v3, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	v_dual_add_f32 v5, v2, v9 :: v_dual_sub_f32 v2, v2, v9
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v8, -v8, v8, s1
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v3, -v3, v3, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v6, v0, v4
	v_sub_f32_e32 v0, v0, v4
	ds_swizzle_b32 v4, v5 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v14, v1 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s1
	v_cndmask_b32_e64 v1, -v1, v1, s3
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v6, -v6, v6, s1
	s_wait_dscnt 0x1
	v_add_f32_e32 v1, v1, v14
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v8, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v1, 0x3d800000, v1
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, -v5, v5, s5
	v_cndmask_b32_e64 v4, -v4, v4, s5
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v18, v1, v49
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v5, -v5, v5, s4
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, -v5, v5, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v2, -v2, v2, s1
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v0, -v0, v0, s1
	v_xor_b32_e32 v9, 16, v10
	v_mul_f32_e32 v5, 0x3d800000, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v14, v5, v45
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v0, -v0, v0, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v0, -v0, v0, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v0, -v0, v0, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v3 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v3, -v3, v3, s3
	ds_swizzle_b32 v16, v0 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v0, -v0, v0, s3
	s_wait_dscnt 0x1
	v_dual_add_f32 v3, v3, v15 :: v_dual_add_f32 v2, v2, v12
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v15, v7, v46 :: v_dual_add_f32 v0, v0, v16
	v_mul_f32_e32 v3, 0x3d800000, v3
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,2)
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s5
	v_dual_mul_f32 v0, 0x3d800000, v0 :: v_dual_mul_f32 v29, v3, v50
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, s4
	v_mul_f32_e32 v30, v0, v51
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,4)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s4
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, s2
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,8)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s2
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, s3
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	v_lshlrev_b32_e32 v12, 2, v32
	ds_swizzle_b32 v13, v2 offset:swizzle(SWAP,16)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s3
	ds_bpermute_b32 v1, v12, v14
	ds_bpermute_b32 v7, v12, v29
	ds_bpermute_b32 v8, v12, v30
	v_mul_f32_e32 v4, 0x3d800000, v4
	v_add_f32_e32 v6, v6, v11
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v2, v2, v13 :: v_dual_mul_f32 v13, v4, v44
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v6, -v6, v6, s5
	ds_bpermute_b32 v4, v12, v15
	v_mul_f32_e32 v2, 0x3d800000, v2
	ds_bpermute_b32 v0, v12, v13
	v_mul_f32_e32 v17, v2, v48
	ds_bpermute_b32 v2, v12, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, s4
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v3, v2, v0, s1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v6, -v6, v6, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	v_mul_f32_e32 v6, 0x3d800000, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, v6, v47
	ds_bpermute_b32 v6, v12, v18
	ds_bpermute_b32 v5, v12, v16
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v2, v6, v1, s1
	v_cndmask_b32_e64 v1, v7, v4, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v10, v9, vcc_lo
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v8, v5, s1
	v_max3_num_f32 v5, |v3|, |v2|, |v1|
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v7, 2, v4
	v_max_num_f32_e64 v6, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v4, v5, v6
	v_xor_b32_e32 v6, 8, v10
	ds_bpermute_b32 v5, v7, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v10, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 2, v6
	v_xor_b32_e32 v6, 4, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_cndmask_b32 v6, v10, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v4, v4, v5 :: v_dual_lshlrev_b32 v9, 2, v6
	v_xor_b32_e32 v6, 2, v10
	ds_bpermute_b32 v5, v8, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v10, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v6
	v_xor_b32_e32 v6, 1, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_cndmask_b32 v6, v10, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v4, v4, v5
	v_lshlrev_b32_e32 v10, 2, v6
	ds_bpermute_b32 v5, v9, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v11, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	v_max_num_f32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v10, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v4, v4, v5 :: v_dual_mov_b32 v5, 1.0
	v_cmpx_neq_f32_e32 0, v4
	s_cbranch_execz .LBB0_116
; %bb.114:                              ; %.preheader76.i.i.3.i
	v_div_scale_f32 v5, null, 0x40e00000, 0x40e00000, v4
	v_div_scale_f32 v37, vcc_lo, v4, 0x40e00000, v4
	s_mov_b32 s19, 0x40e00000
	s_mov_b32 s22, 0xc1000000
	v_rcp_f32_e32 v6, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v5, v6, 1.0
	v_fmac_f32_e32 v6, v36, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, v37, v6
	v_fma_f32 v38, -v5, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v38, v6
	v_fma_f32 v5, -v5, v36, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v5, v5, v6, v36
	v_mov_b32_e32 v36, 0x7149f2ca
	v_div_fixup_f32 v6, v5, 0x40e00000, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v5, 1.0 :: v_dual_mul_f32 v6, 0.5, v6
.LBB0_115:                              ; %.preheader.preheader.i.i.3.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s7, s18
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s18, 8
	v_div_scale_f32 v37, null, s19, s19, s7
	v_div_scale_f32 v38, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v39, v37
	v_xor_b32_e32 v37, 0x80000000, v37
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v40, v37, v39, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v39
	v_mul_f32_e32 v40, v38, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v37, v40, v38
	v_fmac_f32_e32 v40, v41, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v37, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v38, v39, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v37, v37, 0x40e00000, s7
	v_add_f32_e32 v37, 1.0, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v6, v37
	v_div_scale_f32 v38, null, v37, v37, v3
	v_div_scale_f32 v40, null, v37, v37, v2
	v_div_scale_f32 v42, null, v37, v37, v1
	v_div_scale_f32 v44, null, v37, v37, v0
	v_div_scale_f32 v39, vcc_lo, v3, v37, v3
	v_rcp_f32_e32 v46, v38
	v_rcp_f32_e32 v47, v40
	v_rcp_f32_e32 v48, v42
	v_rcp_f32_e32 v49, v44
	v_div_scale_f32 v41, s7, v2, v37, v2
	v_div_scale_f32 v43, s8, v1, v37, v1
	v_div_scale_f32 v45, s9, v0, v37, v0
	v_fma_f32 v50, -v38, v46, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v51, -v40, v47, 1.0
	v_fma_f32 v52, -v42, v48, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v53, -v44, v49, 1.0
	v_dual_fmac_f32 v46, v50, v46 :: v_dual_fmac_f32 v47, v51, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v48, v52, v48 :: v_dual_fmac_f32 v49, v53, v49
	v_dual_mul_f32 v50, v39, v46 :: v_dual_mul_f32 v51, v41, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v52, v43, v48 :: v_dual_mul_f32 v53, v45, v49
	v_fma_f32 v54, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v40, v51, v41
	v_fma_f32 v56, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v57, -v44, v53, v45
	v_dual_fmac_f32 v50, v54, v46 :: v_dual_fmac_f32 v51, v55, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v52, v56, v48 :: v_dual_fmac_f32 v53, v57, v49
	v_fma_f32 v38, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v40, v51, v41
	v_fma_f32 v40, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v41, -v44, v53, v45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v38, v46, v50
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v39, v39, v47, v51
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v38, v38, v37, v3
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v40, v40, v48, v52
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v39, v39, v37, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v41, v41, v49, v53
	v_rndne_f32_e32 v38, v38
	v_div_fixup_f32 v40, v40, v37, v1
	v_rndne_f32_e32 v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v41, v41, v37, v0
	v_med3_num_f32 v38, v38, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v40, v40
	v_med3_num_f32 v39, v39, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v41, v41
	v_fma_f32 v38, -v38, v37, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v40, v40, s22, 0x40e00000
	v_fma_f32 v39, -v39, v37, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v41, v41, s22, 0x40e00000
	v_fma_f32 v38, v38, v38, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v40, -v40, v37, v1
	v_fmac_f32_e32 v38, v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v41, v37, v0
	v_fmac_f32_e32 v38, v40, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v39
	ds_bpermute_b32 v39, v7, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v8, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v9, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v11, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v10, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v38, v36
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v36, v36, v38 :: v_dual_cndmask_b32 v5, v5, v37
	s_cbranch_scc1 .LBB0_115
.LBB0_116:                              ; %Flow1077
	s_or_b32 exec_lo, exec_lo, s17
	v_cmp_neq_f32_e64 s7, 0, v4
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v36, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_118
; %bb.117:
	v_div_scale_f32 v4, null, v5, v5, v3
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v36, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v4, v36, 1.0
	v_fmac_f32_e32 v36, v37, v36
	v_div_scale_f32 v37, vcc_lo, v3, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v36
	v_fma_f32 v39, -v4, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v36
	v_fma_f32 v4, -v4, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v36, v38
	v_div_fixup_f32 v3, v4, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v3, v3
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v3, v3, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v36, v3
.LBB0_118:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_div_scale_f32 v3, null, v5, v5, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v6, v4
	v_div_scale_f32 v6, vcc_lo, v2, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v6, v4
	v_fma_f32 v38, -v3, v37, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v38, v4
	v_fma_f32 v3, -v3, v37, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v37
	v_div_fixup_f32 v2, v3, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v2
.LBB0_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v37, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_122
; %bb.121:
	v_div_scale_f32 v3, null, v5, v5, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v37, v4
	v_div_scale_f32 v37, vcc_lo, v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v4
	v_fma_f32 v39, -v3, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v4
	v_fma_f32 v3, -v3, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v38
	v_div_fixup_f32 v1, v3, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v37, v1
.LBB0_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_124
; %bb.123:
	v_div_scale_f32 v1, null, v5, v5, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v3, v2
	v_div_scale_f32 v3, vcc_lo, v0, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v4, v3, v2
	v_fma_f32 v38, -v1, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v38, v2
	v_fma_f32 v1, -v1, v4, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v4
	v_div_fixup_f32 v0, v1, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v6, v36
	v_mad_co_i64_i32 v[3:4], null, 0x48, v35, s[12:13]
	v_and_b32_e32 v39, 15, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add3_u32 v0, v0, v37, v2
	v_and_b32_e32 v37, 15, v37
	v_lshl_or_b32 v6, v6, 4, v39
	ds_bpermute_b32 v1, v7, v0
	v_lshl_or_b32 v37, v2, 4, v37
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v8, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v9, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v11, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v38, v0, v1
	v_mad_co_u64_u32 v[0:1], null, 0x48, s10, v[3:4]
	ds_bpermute_b32 v36, v10, v38
	v_mad_co_u64_u32 v[1:2], null, 0x48, s11, v[1:2]
	v_lshlrev_b16 v2.l, 8, v37.l
	v_and_b16 v2.h, 0xff, v6.l
	v_add_co_u32 v39, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, 0, v1, vcc_lo
	global_store_b16 v[39:40], v2, off offset:1736
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB0_126
; %bb.125:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v38, v36
	global_store_b64 v[0:1], v[5:6], off offset:1728
.LBB0_126:                              ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit.i.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_bpermute_b32 v0, v12, v13 offset:64
	ds_bpermute_b32 v1, v12, v14 offset:64
	ds_bpermute_b32 v5, v12, v15 offset:64
	ds_bpermute_b32 v13, v12, v16 offset:64
	ds_bpermute_b32 v2, v12, v17 offset:64
	ds_bpermute_b32 v14, v12, v18 offset:64
	ds_bpermute_b32 v15, v12, v29 offset:64
	ds_bpermute_b32 v12, v12, v30 offset:64
	s_mov_b32 s18, 0
	s_mov_b32 s17, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v6, v2, v0, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v2, v14, v1, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v1, v15, v5, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v12, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v5, |v6|, |v2|, |v1|
	v_max_num_f32_e64 v12, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v7, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v8, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v9, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v11, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v10, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_dual_max_num_f32 v12, v5, v12 :: v_dual_mov_b32 v5, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_129
; %bb.127:                              ; %.preheader76.i.1.i.3.i
	v_div_scale_f32 v5, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s19, 0x40e00000
	s_mov_b32 s22, 0xc1000000
	v_rcp_f32_e32 v13, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v5, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v5, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v5, -v5, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v5, v5, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v5, 0x40e00000, v12
	v_mov_b32_e32 v5, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_128:                              ; %.preheader.preheader.i.1.i.3.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s7, s18
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s18, 8
	v_div_scale_f32 v15, null, s19, s19, s7
	v_div_scale_f32 v16, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v29, v15, v18, v16
	v_fmac_f32_e32 v18, v29, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s7
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v18, null, v15, v15, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v40, v18
	v_fma_f32 v44, -v18, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v44, v40
	v_div_scale_f32 v16, null, v15, v15, v6
	v_div_scale_f32 v30, null, v15, v15, v1
	v_div_scale_f32 v37, null, v15, v15, v0
	v_div_scale_f32 v17, vcc_lo, v6, v15, v6
	v_div_scale_f32 v29, s7, v2, v15, v2
	v_rcp_f32_e32 v39, v16
	v_rcp_f32_e32 v41, v30
	v_rcp_f32_e32 v42, v37
	v_div_scale_f32 v38, s9, v0, v15, v0
	v_div_scale_f32 v36, s8, v1, v15, v1
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v43, -v16, v39, 1.0
	v_fma_f32 v45, -v30, v41, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v37, v42, 1.0
	v_fmac_f32_e32 v39, v43, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v41, v45, v41 :: v_dual_fmac_f32 v42, v46, v42
	v_mul_f32_e32 v44, v29, v40
	v_mul_f32_e32 v43, v17, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v45, v36, v41 :: v_dual_mul_f32 v46, v38, v42
	v_fma_f32 v48, -v18, v44, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v16, v43, v17
	v_fma_f32 v49, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v37, v46, v38
	v_dual_fmac_f32 v44, v48, v40 :: v_dual_fmac_f32 v43, v47, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v45, v49, v41 :: v_dual_fmac_f32 v46, v50, v42
	v_fma_f32 v16, -v16, v43, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v44, v29
	v_fma_f32 v18, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v29, -v37, v46, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v39, v43
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v40, v44
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v16, v16, v15, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v41, v45
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v17, v17, v15, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v29, v29, v42, v46
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v1
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v29, v29, v15, v0
	v_med3_num_f32 v16, v16, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v29, v29
	v_fma_f32 v16, -v16, v15, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s22, 0x40e00000
	v_fma_f32 v17, -v17, v15, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v29, v29, s22, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v1
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v29, v15, v0
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v11, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v10, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v14, v16 :: v_dual_cndmask_b32 v5, v5, v15
	s_cbranch_scc1 .LBB0_128
.LBB0_129:                              ; %Flow1075
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s17
	v_cmp_neq_f32_e64 s7, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_131
; %bb.130:
	v_div_scale_f32 v13, null, v5, v5, v6
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v6, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v6, v13, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v6, v6
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v6, v6, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v6
.LBB0_131:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_133
; %bb.132:
	v_div_scale_f32 v6, null, v5, v5, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v12, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v2, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v6, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v6, -v6, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v12, v15
	v_div_fixup_f32 v2, v6, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v2
.LBB0_133:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v6, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_135
; %bb.134:
	v_div_scale_f32 v6, null, v5, v5, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v6, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v6, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v6, -v6, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v14, v16
	v_div_fixup_f32 v1, v6, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v1
.LBB0_135:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_137
; %bb.136:
	v_div_scale_f32 v1, null, v5, v5, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v14, v2
	v_div_scale_f32 v14, vcc_lo, v0, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v2
	v_fma_f32 v16, -v1, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v2
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v15
	v_div_fixup_f32 v0, v1, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_137:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v0, v6, v2
	ds_bpermute_b32 v1, v7, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v8, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v9, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v11, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v0, v1
	v_mad_co_u64_u32 v[0:1], null, 0x48, s14, v[3:4]
	v_and_b32_e32 v4, 15, v6
	v_and_b32_e32 v6, 15, v13
	ds_bpermute_b32 v3, v10, v7
	v_lshl_or_b32 v4, v2, 4, v4
	v_lshl_or_b32 v6, v12, 4, v6
	v_mad_co_u64_u32 v[1:2], null, 0x48, s15, v[1:2]
	v_add_co_u32 v8, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v4.l
	v_and_b16 v2.h, 0xff, v6.l
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, 0, v1, vcc_lo
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[8:9], v2, off offset:1736
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_139
; %bb.138:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v7, v3
	global_store_b64 v[0:1], v[5:6], off offset:1728
.LBB0_139:                              ; %Flow1078
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s16
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB0_141
; %bb.140:                              ; %.preheader902.4.i
	ds_store_2addr_b32 v34, v157, v158 offset1:1
	ds_store_2addr_b32 v34, v155, v156 offset0:2 offset1:3
	ds_store_2addr_b32 v34, v153, v154 offset0:4 offset1:5
	ds_store_2addr_b32 v34, v151, v152 offset0:6 offset1:7
	ds_store_2addr_b32 v34, v149, v150 offset0:16 offset1:17
	ds_store_2addr_b32 v34, v147, v148 offset0:18 offset1:19
	ds_store_2addr_b32 v34, v145, v146 offset0:20 offset1:21
	ds_store_2addr_b32 v34, v143, v144 offset0:22 offset1:23
.LBB0_141:                              ; %.loopexit.4.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, 32, v35
	s_mov_b32 s16, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s26, v0
	s_cbranch_execz .LBB0_168
; %bb.142:
	v_mad_co_i64_i32 v[0:1], null, s24, v0, 0
	s_mov_b32 s18, 0
	s_mov_b32 s17, exec_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v0, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v28, vcc_lo
	global_load_b128 v[40:43], v[4:5], off
	global_load_b128 v[0:3], v[25:26], off
	global_load_b128 v[8:11], v[4:5], off offset:16
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v4, 0xbfb8aa3b, v40
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, 0xbfb8aa3b, v40, -v4
	v_rndne_f32_e32 v6, v4
	v_dual_fmac_f32 v5, 0xb2a5705f, v40 :: v_dual_sub_f32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v40
	global_load_b128 v[4:7], v[25:26], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0x7f800000, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v14, 1.0, v12
	v_mul_f32_e32 v12, 0xbfb8aa3b, v41
	v_div_scale_f32 v15, null, v14, v14, v40
	v_div_scale_f32 v29, vcc_lo, v40, v14, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v17, 0xbfb8aa3b, v41, -v12
	v_rndne_f32_e32 v18, v12
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v17, 0xb2a5705f, v41 :: v_dual_sub_f32 v12, v12, v18
	v_cvt_i32_f32_e32 v18, v18
	v_add_f32_e32 v12, v12, v17
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v13, -v15, v16, 1.0
	v_exp_f32_e32 v17, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v13, v16
	v_mul_f32_e32 v30, v29, v16
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_ldexp_f32 v17, v17, v18
	v_fma_f32 v13, -v15, v30, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v13, v16
	ds_load_2addr_b32 v[12:13], v33 offset1:1
	v_fma_f32 v15, -v15, v30, v29
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v15, v16, v30
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v41
	ds_load_2addr_b32 v[44:45], v33 offset0:2 offset1:3
	ds_load_2addr_b32 v[46:47], v33 offset0:4 offset1:5
	ds_load_2addr_b32 v[29:30], v33 offset0:6 offset1:7
	v_div_fixup_f32 v14, v15, v14, v40
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, 0, v17, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v41
	s_wait_dscnt 0x3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v36, v12, v14 :: v_dual_cndmask_b32 v15, 0x7f800000, v16
	v_mul_f32_e32 v16, 0xbfb8aa3b, v42
	s_wait_loadcnt 0x2
	v_div_scale_f32 v14, null, v0, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, 1.0, v15
	v_fma_f32 v37, 0xbfb8aa3b, v42, -v16
	v_rndne_f32_e32 v38, v16
	v_div_scale_f32 v48, vcc_lo, v36, v0, v36
	v_rcp_f32_e32 v17, v14
	v_div_scale_f32 v15, null, v12, v12, v41
	v_fmac_f32_e32 v37, 0xb2a5705f, v42
	v_sub_f32_e32 v16, v16, v38
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v15
	v_add_f32_e32 v16, v16, v37
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v14, v17, 1.0
	v_exp_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v17, v39, v17
	v_fma_f32 v40, -v15, v18, 1.0
	v_div_scale_f32 v39, s7, v41, v12, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v40, v18
	v_dual_mul_f32 v40, v48, v17 :: v_dual_mul_f32 v49, v39, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v37, -v14, v40, v48
	v_fma_f32 v50, -v15, v49, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v40, v37, v17
	v_cvt_i32_f32_e32 v37, v38
	v_fmac_f32_e32 v49, v50, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v14, -v14, v40, v48
	v_ldexp_f32 v16, v16, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v15, -v15, v49, v39
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v14, v17, v40
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v15, v18, v49
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v12, v14, v12, v41
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v16, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v42
	v_mul_f32_e32 v38, v13, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v14, 0x7f800000, v15, vcc_lo
	v_mul_f32_e32 v15, 0xbfb8aa3b, v43
	v_div_scale_f32 v13, null, v1, v1, v38
	v_div_scale_f32 v48, vcc_lo, v38, v1, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v12, 1.0, v14
	v_fma_f32 v18, 0xbfb8aa3b, v43, -v15
	v_rndne_f32_e32 v39, v15
	v_rcp_f32_e32 v16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v14, null, v12, v12, v42
	v_fmac_f32_e32 v18, 0xb2a5705f, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v15, v15, v39
	v_cvt_i32_f32_e32 v39, v39
	v_rcp_f32_e32 v17, v14
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v13, v16, 1.0
	v_dual_add_f32 v15, v15, v18 :: v_dual_fmac_f32 v16, v40, v16
	v_div_scale_f32 v40, s7, v42, v12, v42
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v14, v17, 1.0
	v_exp_f32_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v41, v17
	v_mul_f32_e32 v41, v48, v16
	v_mul_f32_e32 v18, v40, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v49, -v13, v41, v48
	v_ldexp_f32 v15, v15, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v14, v18, v40
	v_fmac_f32_e32 v41, v49, v16
	v_div_fixup_f32 v0, v37, v0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, v50, v17
	v_fma_f32 v13, -v13, v41, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v14, v18, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v39, v13, v16, v41
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v14, v17, v18
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v12, v13, v12, v42
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v43
	s_wait_dscnt 0x2
	v_mul_f32_e32 v40, v44, v12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, 0xbfb8aa3b, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, 0x7f800000, v14, vcc_lo
	v_div_scale_f32 v42, null, v2, v2, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v16, v12
	v_add_f32_e32 v44, 1.0, v13
	v_fma_f32 v13, 0xbfb8aa3b, v8, -v12
	v_div_scale_f32 v51, vcc_lo, v40, v2, v40
	v_rcp_f32_e32 v49, v42
	v_sub_f32_e32 v12, v12, v16
	v_div_scale_f32 v48, null, v44, v44, v43
	v_fmac_f32_e32 v13, 0xb2a5705f, v8
	v_div_scale_f32 v52, s7, v43, v44, v43
	v_cvt_i32_f32_e32 v56, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v50, v48
	v_add_f32_e32 v17, v12, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v42, v49, 1.0
	v_exp_f32_e32 v55, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v49, v14, v49
	v_fma_f32 v15, -v48, v50, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v53, v51, v49
	v_fmac_f32_e32 v50, v15, v50
	global_load_b128 v[12:15], v[23:24], off offset:4
	v_fma_f32 v18, -v42, v53, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v54, v52, v50 :: v_dual_fmac_f32 v53, v18, v49
	v_div_fixup_f32 v1, v39, v1, v38
	v_fma_f32 v17, -v48, v54, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v42, -v42, v53, v51
	v_ldexp_f32 v51, v55, v56
	v_fmac_f32_e32 v54, v17, v50
	s_clause 0x1
	global_load_b32 v41, v[21:22], off
	global_load_b96 v[16:18], v[23:24], off offset:20
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v42, v42, v49, v53
	s_mov_b32 vcc_lo, s7
	v_fma_f32 v48, -v48, v54, v52
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v48, v48, v50, v54
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v8
	v_div_fixup_f32 v43, v48, v44, v43
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, 0, v51, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v43, v45, v43 :: v_dual_cndmask_b32 v44, 0x7f800000, v49
	v_mul_f32_e32 v49, 0xbfb8aa3b, v9
	v_div_scale_f32 v45, null, v3, v3, v43
	v_div_scale_f32 v56, vcc_lo, v43, v3, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v44, 1.0, v44
	v_fma_f32 v52, 0xbfb8aa3b, v9, -v49
	v_rndne_f32_e32 v53, v49
	v_rcp_f32_e32 v50, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v48, null, v44, v44, v8
	v_fmac_f32_e32 v52, 0xb2a5705f, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v49, v49, v53
	v_cvt_i32_f32_e32 v53, v53
	v_rcp_f32_e32 v51, v48
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v54, -v45, v50, 1.0
	v_dual_add_f32 v49, v49, v52 :: v_dual_fmac_f32 v50, v54, v50
	v_div_scale_f32 v54, s7, v8, v44, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v48, v51, 1.0
	v_exp_f32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v51, v55, v51
	v_dual_mul_f32 v55, v56, v50 :: v_dual_mul_f32 v52, v54, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v57, -v45, v55, v56
	v_ldexp_f32 v49, v49, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v48, v52, v54
	v_dual_fmac_f32 v55, v57, v50 :: v_dual_fmac_f32 v52, v58, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v45, -v45, v55, v56
	v_fma_f32 v48, -v48, v52, v54
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v53, v45, v50, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v48, v51, v52
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v3, v53, v3, v43
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v8, v45, v44, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, 0, v49, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	s_wait_dscnt 0x1
	v_mul_f32_e32 v8, v46, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0x7f800000, v48, vcc_lo
	v_mul_f32_e32 v48, 0xbfb8aa3b, v10
	s_wait_loadcnt 0x3
	v_div_scale_f32 v45, null, v4, v4, v8
	v_div_scale_f32 v56, vcc_lo, v8, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v49, v45
	v_fma_f32 v54, -v45, v49, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v49, v54, v49
	v_div_fixup_f32 v2, v42, v2, v40
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v3, v3, v14 :: v_dual_add_f32 v44, 1.0, v44
	v_mul_f32_e32 v1, v1, v12
	v_fma_f32 v51, 0xbfb8aa3b, v10, -v48
	v_rndne_f32_e32 v52, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v46, null, v44, v44, v9
	v_div_scale_f32 v54, s7, v9, v44, v9
	v_dual_fmac_f32 v51, 0xb2a5705f, v10 :: v_dual_sub_f32 v48, v48, v52
	v_cvt_i32_f32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v50, v46
	v_add_f32_e32 v48, v48, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v48, v48
	v_fma_f32 v55, -v46, v50, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_dual_fmac_f32 v50, v55, v50 :: v_dual_mul_f32 v55, v56, v49
	v_ldexp_f32 v48, v48, v52
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v51, v54, v50
	v_fma_f32 v57, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v58, -v46, v51, v54
	v_fmac_f32_e32 v55, v57, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v51, v58, v50
	v_fma_f32 v45, -v45, v55, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v51, v54
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v52, v45, v49, v55
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v46, v50, v51
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v9, v45, v44, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v48, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v9, v47, v9 :: v_dual_cndmask_b32 v44, 0x7f800000, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v55, null, v5, v5, v9
	v_div_scale_f32 v60, vcc_lo, v9, v5, v9
	v_add_f32_e32 v54, 1.0, v44
	v_mul_f32_e32 v44, 0xbfb8aa3b, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v57, v55
	v_div_scale_f32 v56, null, v54, v54, v10
	v_div_scale_f32 v61, s7, v10, v54, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v45, 0xbfb8aa3b, v11, -v44
	v_rndne_f32_e32 v59, v44
	v_rcp_f32_e32 v58, v56
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v55, v57, 1.0
	v_fmac_f32_e32 v45, 0xb2a5705f, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v59
	v_cvt_i32_f32_e32 v59, v59
	v_fmac_f32_e32 v57, v46, v57
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v47, -v56, v58, 1.0
	v_add_f32_e32 v48, v44, v45
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v62, v60, v57
	v_fmac_f32_e32 v58, v47, v58
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v65, v48
	s_clause 0x1
	global_load_b128 v[44:47], v[19:20], off
	global_load_b128 v[48:51], v[19:20], off offset:16
	v_fma_f32 v64, -v55, v62, v60
	v_dual_mul_f32 v63, v61, v58 :: v_dual_fmac_f32 v62, v64, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v66, -v56, v63, v61
	v_ldexp_f32 v59, v65, v59
	v_fma_f32 v55, -v55, v62, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v63, v66, v58
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v55, v55, v57, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v63, v61
	s_mov_b32 vcc_lo, s7
	v_div_fixup_f32 v5, v55, v5, v9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v56, v56, v58, v63
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	s_wait_loadcnt 0x3
	v_fma_f32 v9, v0, v41, v1
	v_fma_f32 v0, v0, v41, -v1
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v5, v5, v16
	v_div_fixup_f32 v10, v56, v54, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v57, 0, v59, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	v_fma_f32 v1, v2, v13, -v3
	s_wait_dscnt 0x0
	v_mul_f32_e32 v10, v29, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v54, 0x7f800000, v57, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v61, vcc_lo, v10, v6, v10
	v_add_f32_e32 v29, 1.0, v54
	v_div_scale_f32 v54, null, v6, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v56, null, v29, v29, v11
	v_rcp_f32_e32 v57, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v58, v56
	v_fma_f32 v59, -v54, v57, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v60, -v56, v58, 1.0
	v_fmac_f32_e32 v57, v59, v57
	v_div_scale_f32 v59, s7, v11, v29, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v58, v60, v58
	v_mul_f32_e32 v60, v61, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v62, v59, v58
	v_fma_f32 v63, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v64, -v56, v62, v59
	v_fmac_f32_e32 v60, v63, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v62, v64, v58
	v_fma_f32 v54, -v54, v60, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v56, v62, v59
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v54, v54, v57, v60
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v56, v56, v58, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v6, v54, v6, v10
	v_div_fixup_f32 v11, v56, v29, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v11, v30, v11
	v_div_scale_f32 v29, null, v7, v7, v11
	v_div_scale_f32 v57, vcc_lo, v11, v7, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v30, v29
	v_fma_f32 v56, -v29, v30, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v30, v56, v30
	v_mul_f32_e32 v56, v57, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v58, -v29, v56, v57
	v_fmac_f32_e32 v56, v58, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v12, -v29, v56, v57
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v12, v12, v30, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v7, v12, v7, v11
	v_fma_f32 v11, v2, v13, v3
	v_mul_f32_e32 v7, v7, v18
	v_div_fixup_f32 v4, v52, v4, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v9, v11
	v_fma_f32 v8, v6, v17, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v3, v4, v15, v5
	v_fma_f32 v4, v4, v15, -v5
	v_fma_f32 v5, v6, v17, -v7
	v_dual_sub_f32 v6, v9, v11 :: v_dual_add_f32 v7, v0, v1
	v_dual_add_f32 v9, v3, v8 :: v_dual_sub_f32 v0, v0, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v4, v5
	v_sub_f32_e32 v3, v3, v8
	v_dual_add_f32 v8, v7, v1 :: v_dual_sub_f32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v7, v6, v3
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v7, -v7, v7, s1
	v_cndmask_b32_e64 v1, -v1, v1, s1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v7, -v7, v7, s5
	v_cndmask_b32_e64 v1, -v1, v1, s5
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v7, -v7, v7, s4
	v_cndmask_b32_e64 v1, -v1, v1, s4
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v7, -v7, v7, s2
	v_cndmask_b32_e64 v1, -v1, v1, s2
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	v_sub_f32_e32 v3, v6, v3
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v3, -v3, v3, s1
	v_cndmask_b32_e64 v7, -v7, v7, s3
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v4, v4, v5 :: v_dual_add_f32 v7, v7, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_mul_f32_e32 v7, 0x3d800000, v7
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v3, -v3, v3, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v3, v14
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v3, -v3, v3, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	v_dual_add_f32 v5, v2, v9 :: v_dual_sub_f32 v2, v2, v9
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v8, -v8, v8, s1
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v3, -v3, v3, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v6, v0, v4
	v_sub_f32_e32 v0, v0, v4
	ds_swizzle_b32 v4, v5 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v14, v1 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s1
	v_cndmask_b32_e64 v1, -v1, v1, s3
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v6, -v6, v6, s1
	s_wait_dscnt 0x1
	v_add_f32_e32 v1, v1, v14
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v8, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v1, 0x3d800000, v1
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, -v5, v5, s5
	v_cndmask_b32_e64 v4, -v4, v4, s5
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v18, v1, v49
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v5, -v5, v5, s4
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, -v5, v5, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v2, -v2, v2, s1
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v0, -v0, v0, s1
	v_xor_b32_e32 v9, 16, v10
	v_mul_f32_e32 v5, 0x3d800000, v5
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v14, v5, v45
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v0, -v0, v0, s5
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v0, -v0, v0, s4
	s_wait_dscnt 0x0
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v0, -v0, v0, s2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v3 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v3, -v3, v3, s3
	ds_swizzle_b32 v16, v0 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v0, -v0, v0, s3
	s_wait_dscnt 0x1
	v_dual_add_f32 v3, v3, v15 :: v_dual_add_f32 v2, v2, v12
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v15, v7, v46 :: v_dual_add_f32 v0, v0, v16
	v_mul_f32_e32 v3, 0x3d800000, v3
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,2)
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s5
	v_dual_mul_f32 v0, 0x3d800000, v0 :: v_dual_mul_f32 v29, v3, v50
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, s4
	v_mul_f32_e32 v30, v0, v51
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,4)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s4
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, s2
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,8)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s2
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, s3
	s_wait_dscnt 0x1
	v_add_f32_e32 v2, v2, v12
	v_lshlrev_b32_e32 v12, 2, v32
	ds_swizzle_b32 v13, v2 offset:swizzle(SWAP,16)
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v8
	v_cndmask_b32_e64 v2, -v2, v2, s3
	ds_bpermute_b32 v1, v12, v14
	ds_bpermute_b32 v7, v12, v29
	ds_bpermute_b32 v8, v12, v30
	v_mul_f32_e32 v4, 0x3d800000, v4
	v_add_f32_e32 v6, v6, v11
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v2, v2, v13 :: v_dual_mul_f32 v13, v4, v44
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v6, -v6, v6, s5
	ds_bpermute_b32 v4, v12, v15
	v_mul_f32_e32 v2, 0x3d800000, v2
	ds_bpermute_b32 v0, v12, v13
	v_mul_f32_e32 v17, v2, v48
	ds_bpermute_b32 v2, v12, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, s4
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v3, v2, v0, s1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, s2
	s_wait_dscnt 0x0
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v6, -v6, v6, s3
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v11
	v_mul_f32_e32 v6, 0x3d800000, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v16, v6, v47
	ds_bpermute_b32 v6, v12, v18
	ds_bpermute_b32 v5, v12, v16
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v2, v6, v1, s1
	v_cndmask_b32_e64 v1, v7, v4, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v10, v9, vcc_lo
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v8, v5, s1
	v_max3_num_f32 v5, |v3|, |v2|, |v1|
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v7, 2, v4
	v_max_num_f32_e64 v6, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v4, v5, v6
	v_xor_b32_e32 v6, 8, v10
	ds_bpermute_b32 v5, v7, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v10, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 2, v6
	v_xor_b32_e32 v6, 4, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_cndmask_b32 v6, v10, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v4, v4, v5 :: v_dual_lshlrev_b32 v9, 2, v6
	v_xor_b32_e32 v6, 2, v10
	ds_bpermute_b32 v5, v8, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v10, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v6
	v_xor_b32_e32 v6, 1, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_cndmask_b32 v6, v10, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v4, v4, v5
	v_lshlrev_b32_e32 v10, 2, v6
	ds_bpermute_b32 v5, v9, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v11, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	v_max_num_f32_e32 v4, v4, v5
	ds_bpermute_b32 v5, v10, v4
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v4, v4, v5 :: v_dual_mov_b32 v5, 1.0
	v_cmpx_neq_f32_e32 0, v4
	s_cbranch_execz .LBB0_145
; %bb.143:                              ; %.preheader76.i.i.4.i
	v_div_scale_f32 v5, null, 0x40e00000, 0x40e00000, v4
	v_div_scale_f32 v37, vcc_lo, v4, 0x40e00000, v4
	s_mov_b32 s19, 0x40e00000
	s_mov_b32 s22, 0xc1000000
	v_rcp_f32_e32 v6, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v5, v6, 1.0
	v_fmac_f32_e32 v6, v36, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, v37, v6
	v_fma_f32 v38, -v5, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v38, v6
	v_fma_f32 v5, -v5, v36, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v5, v5, v6, v36
	v_mov_b32_e32 v36, 0x7149f2ca
	v_div_fixup_f32 v6, v5, 0x40e00000, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v5, 1.0 :: v_dual_mul_f32 v6, 0.5, v6
.LBB0_144:                              ; %.preheader.preheader.i.i.4.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s7, s18
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s18, 8
	v_div_scale_f32 v37, null, s19, s19, s7
	v_div_scale_f32 v38, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v39, v37
	v_xor_b32_e32 v37, 0x80000000, v37
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v40, v37, v39, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v39
	v_mul_f32_e32 v40, v38, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v37, v40, v38
	v_fmac_f32_e32 v40, v41, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v37, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v37, v38, v39, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v37, v37, 0x40e00000, s7
	v_add_f32_e32 v37, 1.0, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v6, v37
	v_div_scale_f32 v38, null, v37, v37, v3
	v_div_scale_f32 v40, null, v37, v37, v2
	v_div_scale_f32 v42, null, v37, v37, v1
	v_div_scale_f32 v44, null, v37, v37, v0
	v_div_scale_f32 v39, vcc_lo, v3, v37, v3
	v_rcp_f32_e32 v46, v38
	v_rcp_f32_e32 v47, v40
	v_rcp_f32_e32 v48, v42
	v_rcp_f32_e32 v49, v44
	v_div_scale_f32 v41, s7, v2, v37, v2
	v_div_scale_f32 v43, s8, v1, v37, v1
	v_div_scale_f32 v45, s9, v0, v37, v0
	v_fma_f32 v50, -v38, v46, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v51, -v40, v47, 1.0
	v_fma_f32 v52, -v42, v48, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v53, -v44, v49, 1.0
	v_dual_fmac_f32 v46, v50, v46 :: v_dual_fmac_f32 v47, v51, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v48, v52, v48 :: v_dual_fmac_f32 v49, v53, v49
	v_dual_mul_f32 v50, v39, v46 :: v_dual_mul_f32 v51, v41, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v52, v43, v48 :: v_dual_mul_f32 v53, v45, v49
	v_fma_f32 v54, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v55, -v40, v51, v41
	v_fma_f32 v56, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v57, -v44, v53, v45
	v_dual_fmac_f32 v50, v54, v46 :: v_dual_fmac_f32 v51, v55, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v52, v56, v48 :: v_dual_fmac_f32 v53, v57, v49
	v_fma_f32 v38, -v38, v50, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v40, v51, v41
	v_fma_f32 v40, -v42, v52, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v41, -v44, v53, v45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v38, v46, v50
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v39, v39, v47, v51
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v38, v38, v37, v3
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v40, v40, v48, v52
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v39, v39, v37, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v41, v41, v49, v53
	v_rndne_f32_e32 v38, v38
	v_div_fixup_f32 v40, v40, v37, v1
	v_rndne_f32_e32 v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v41, v41, v37, v0
	v_med3_num_f32 v38, v38, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v40, v40
	v_med3_num_f32 v39, v39, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v41, v41
	v_fma_f32 v38, -v38, v37, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v40, v40, s22, 0x40e00000
	v_fma_f32 v39, -v39, v37, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v41, v41, s22, 0x40e00000
	v_fma_f32 v38, v38, v38, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v40, -v40, v37, v1
	v_fmac_f32_e32 v38, v39, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v41, v37, v0
	v_fmac_f32_e32 v38, v40, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v39
	ds_bpermute_b32 v39, v7, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v8, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v9, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v11, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	ds_bpermute_b32 v39, v10, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v38, v38, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v38, v36
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v36, v36, v38 :: v_dual_cndmask_b32 v5, v5, v37
	s_cbranch_scc1 .LBB0_144
.LBB0_145:                              ; %Flow1071
	s_or_b32 exec_lo, exec_lo, s17
	v_cmp_neq_f32_e64 s7, 0, v4
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v36, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_147
; %bb.146:
	v_div_scale_f32 v4, null, v5, v5, v3
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v36, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v4, v36, 1.0
	v_fmac_f32_e32 v36, v37, v36
	v_div_scale_f32 v37, vcc_lo, v3, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v36
	v_fma_f32 v39, -v4, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v36
	v_fma_f32 v4, -v4, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v36, v38
	v_div_fixup_f32 v3, v4, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v3, v3
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v3, v3, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v36, v3
.LBB0_147:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_149
; %bb.148:
	v_div_scale_f32 v3, null, v5, v5, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v6, v4
	v_div_scale_f32 v6, vcc_lo, v2, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v6, v4
	v_fma_f32 v38, -v3, v37, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v38, v4
	v_fma_f32 v3, -v3, v37, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v37
	v_div_fixup_f32 v2, v3, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v2
.LBB0_149:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v37, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_151
; %bb.150:
	v_div_scale_f32 v3, null, v5, v5, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v37, v4
	v_div_scale_f32 v37, vcc_lo, v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v4
	v_fma_f32 v39, -v3, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v4
	v_fma_f32 v3, -v3, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v38
	v_div_fixup_f32 v1, v3, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v37, v1
.LBB0_151:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_153
; %bb.152:
	v_div_scale_f32 v1, null, v5, v5, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v3, v2
	v_div_scale_f32 v3, vcc_lo, v0, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v4, v3, v2
	v_fma_f32 v38, -v1, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v38, v2
	v_fma_f32 v1, -v1, v4, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v4
	v_div_fixup_f32 v0, v1, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_153:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v6, v36
	v_mad_co_i64_i32 v[3:4], null, 0x48, v35, s[12:13]
	v_and_b32_e32 v39, 15, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add3_u32 v0, v0, v37, v2
	v_and_b32_e32 v37, 15, v37
	v_lshl_or_b32 v6, v6, 4, v39
	ds_bpermute_b32 v1, v7, v0
	v_lshl_or_b32 v37, v2, 4, v37
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v8, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v9, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v11, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v38, v0, v1
	v_mad_co_u64_u32 v[0:1], null, 0x48, s10, v[3:4]
	ds_bpermute_b32 v36, v10, v38
	v_mad_co_u64_u32 v[1:2], null, 0x48, s11, v[1:2]
	v_lshlrev_b16 v2.l, 8, v37.l
	v_and_b16 v2.h, 0xff, v6.l
	v_add_co_u32 v39, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, 0, v1, vcc_lo
	global_store_b16 v[39:40], v2, off offset:2312
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB0_155
; %bb.154:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v38, v36
	global_store_b64 v[0:1], v[5:6], off offset:2304
.LBB0_155:                              ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit.i.4.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_bpermute_b32 v0, v12, v13 offset:64
	ds_bpermute_b32 v1, v12, v14 offset:64
	ds_bpermute_b32 v5, v12, v15 offset:64
	ds_bpermute_b32 v13, v12, v16 offset:64
	ds_bpermute_b32 v2, v12, v17 offset:64
	ds_bpermute_b32 v14, v12, v18 offset:64
	ds_bpermute_b32 v15, v12, v29 offset:64
	ds_bpermute_b32 v12, v12, v30 offset:64
	s_mov_b32 s18, 0
	s_mov_b32 s17, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v6, v2, v0, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v2, v14, v1, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v1, v15, v5, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v12, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v5, |v6|, |v2|, |v1|
	v_max_num_f32_e64 v12, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v7, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v8, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v9, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v11, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v5, v5, v12
	ds_bpermute_b32 v12, v10, v5
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_dual_max_num_f32 v12, v5, v12 :: v_dual_mov_b32 v5, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_158
; %bb.156:                              ; %.preheader76.i.1.i.4.i
	v_div_scale_f32 v5, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s19, 0x40e00000
	s_mov_b32 s22, 0xc1000000
	v_rcp_f32_e32 v13, v5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v5, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v5, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v5, -v5, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v5, v5, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v5, 0x40e00000, v12
	v_mov_b32_e32 v5, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_157:                              ; %.preheader.preheader.i.1.i.4.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s7, s18
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s18, 8
	v_div_scale_f32 v15, null, s19, s19, s7
	v_div_scale_f32 v16, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v29, v15, v18, v16
	v_fmac_f32_e32 v18, v29, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s7
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v18, null, v15, v15, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v40, v18
	v_fma_f32 v44, -v18, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v44, v40
	v_div_scale_f32 v16, null, v15, v15, v6
	v_div_scale_f32 v30, null, v15, v15, v1
	v_div_scale_f32 v37, null, v15, v15, v0
	v_div_scale_f32 v17, vcc_lo, v6, v15, v6
	v_div_scale_f32 v29, s7, v2, v15, v2
	v_rcp_f32_e32 v39, v16
	v_rcp_f32_e32 v41, v30
	v_rcp_f32_e32 v42, v37
	v_div_scale_f32 v38, s9, v0, v15, v0
	v_div_scale_f32 v36, s8, v1, v15, v1
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v43, -v16, v39, 1.0
	v_fma_f32 v45, -v30, v41, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v37, v42, 1.0
	v_fmac_f32_e32 v39, v43, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v41, v45, v41 :: v_dual_fmac_f32 v42, v46, v42
	v_mul_f32_e32 v44, v29, v40
	v_mul_f32_e32 v43, v17, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v45, v36, v41 :: v_dual_mul_f32 v46, v38, v42
	v_fma_f32 v48, -v18, v44, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v16, v43, v17
	v_fma_f32 v49, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v37, v46, v38
	v_dual_fmac_f32 v44, v48, v40 :: v_dual_fmac_f32 v43, v47, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v45, v49, v41 :: v_dual_fmac_f32 v46, v50, v42
	v_fma_f32 v16, -v16, v43, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v44, v29
	v_fma_f32 v18, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v29, -v37, v46, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v39, v43
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v40, v44
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v16, v16, v15, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v41, v45
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v17, v17, v15, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v29, v29, v42, v46
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v1
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v29, v29, v15, v0
	v_med3_num_f32 v16, v16, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s22, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v29, v29
	v_fma_f32 v16, -v16, v15, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s22, 0x40e00000
	v_fma_f32 v17, -v17, v15, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v29, v29, s22, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v1
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v29, v15, v0
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v11, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v10, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v14, v16 :: v_dual_cndmask_b32 v5, v5, v15
	s_cbranch_scc1 .LBB0_157
.LBB0_158:                              ; %Flow1069
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s17
	v_cmp_neq_f32_e64 s7, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_160
; %bb.159:
	v_div_scale_f32 v13, null, v5, v5, v6
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v6, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v6, v13, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v6, v6
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v6, v6, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v6
.LBB0_160:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_162
; %bb.161:
	v_div_scale_f32 v6, null, v5, v5, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v12, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v2, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v6, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v6, -v6, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v12, v15
	v_div_fixup_f32 v2, v6, v5, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v2
.LBB0_162:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v6, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_164
; %bb.163:
	v_div_scale_f32 v6, null, v5, v5, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v6, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v6, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v6, -v6, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v14, v16
	v_div_fixup_f32 v1, v6, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v1
.LBB0_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_166
; %bb.165:
	v_div_scale_f32 v1, null, v5, v5, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v14, v2
	v_div_scale_f32 v14, vcc_lo, v0, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v2
	v_fma_f32 v16, -v1, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v2
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v15
	v_div_fixup_f32 v0, v1, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v12, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, v0, v6, v2
	ds_bpermute_b32 v1, v7, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v8, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v9, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v11, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v0, v1
	v_mad_co_u64_u32 v[0:1], null, 0x48, s14, v[3:4]
	v_and_b32_e32 v4, 15, v6
	v_and_b32_e32 v6, 15, v13
	ds_bpermute_b32 v3, v10, v7
	v_lshl_or_b32 v4, v2, 4, v4
	v_lshl_or_b32 v6, v12, 4, v6
	v_mad_co_u64_u32 v[1:2], null, 0x48, s15, v[1:2]
	v_add_co_u32 v8, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v4.l
	v_and_b16 v2.h, 0xff, v6.l
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, 0, v1, vcc_lo
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[8:9], v2, off offset:2312
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_168
; %bb.167:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v6, v7, v3
	global_store_b64 v[0:1], v[5:6], off offset:2304
.LBB0_168:                              ; %Flow1072
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s16
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_170
; %bb.169:                              ; %.preheader902.5.i
	ds_store_2addr_b32 v34, v157, v158 offset1:1
	ds_store_2addr_b32 v34, v155, v156 offset0:2 offset1:3
	ds_store_2addr_b32 v34, v153, v154 offset0:4 offset1:5
	ds_store_2addr_b32 v34, v151, v152 offset0:6 offset1:7
	ds_store_2addr_b32 v34, v149, v150 offset0:16 offset1:17
	ds_store_2addr_b32 v34, v147, v148 offset0:18 offset1:19
	ds_store_2addr_b32 v34, v145, v146 offset0:20 offset1:21
	ds_store_2addr_b32 v34, v143, v144 offset0:22 offset1:23
.LBB0_170:                              ; %.loopexit.5.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v36, 40, v35
	s_mov_b32 s14, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s26, v36
	s_cbranch_execz .LBB0_197
; %bb.171:
	v_mad_co_i64_i32 v[0:1], null, s24, v36, 0
	s_mov_b32 s16, 0
	s_mov_b32 s15, exec_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v0, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v28, vcc_lo
	global_load_b128 v[41:44], v[4:5], off
	global_load_b128 v[0:3], v[25:26], off
	global_load_b128 v[8:11], v[4:5], off offset:16
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v4, 0xbfb8aa3b, v41
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, 0xbfb8aa3b, v41, -v4
	v_rndne_f32_e32 v6, v4
	v_dual_fmac_f32 v5, 0xb2a5705f, v41 :: v_dual_sub_f32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v41
	global_load_b128 v[4:7], v[25:26], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0x7f800000, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v14, 1.0, v12
	v_mul_f32_e32 v12, 0xbfb8aa3b, v42
	v_div_scale_f32 v15, null, v14, v14, v41
	v_div_scale_f32 v29, vcc_lo, v41, v14, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v17, 0xbfb8aa3b, v42, -v12
	v_rndne_f32_e32 v18, v12
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, 0xb2a5705f, v42
	v_sub_f32_e32 v12, v12, v18
	v_cvt_i32_f32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v12, v12, v17
	v_fma_f32 v13, -v15, v16, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v17, v12
	v_fmac_f32_e32 v16, v13, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v30, v29, v16
	v_ldexp_f32 v17, v17, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v15, v30, v29
	v_fmac_f32_e32 v30, v13, v16
	ds_load_2addr_b32 v[12:13], v33 offset1:1
	v_fma_f32 v15, -v15, v30, v29
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v15, v15, v16, v30
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v42
	ds_load_2addr_b32 v[45:46], v33 offset0:2 offset1:3
	ds_load_2addr_b32 v[47:48], v33 offset0:4 offset1:5
	ds_load_2addr_b32 v[29:30], v33 offset0:6 offset1:7
	v_div_fixup_f32 v14, v15, v14, v41
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, 0, v17, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v42
	s_wait_dscnt 0x3
	v_mul_f32_e32 v37, v12, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v16, vcc_lo
	v_mul_f32_e32 v16, 0xbfb8aa3b, v43
	s_wait_loadcnt 0x2
	v_div_scale_f32 v14, null, v0, v0, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, 1.0, v15
	v_fma_f32 v38, 0xbfb8aa3b, v43, -v16
	v_rndne_f32_e32 v39, v16
	v_div_scale_f32 v49, vcc_lo, v37, v0, v37
	v_rcp_f32_e32 v17, v14
	v_div_scale_f32 v15, null, v12, v12, v42
	v_fmac_f32_e32 v38, 0xb2a5705f, v43
	v_sub_f32_e32 v16, v16, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v15
	v_add_f32_e32 v16, v16, v38
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v40, -v14, v17, 1.0
	v_exp_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v17, v40, v17
	v_fma_f32 v41, -v15, v18, 1.0
	v_div_scale_f32 v40, s7, v42, v12, v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v41, v18
	v_dual_mul_f32 v41, v49, v17 :: v_dual_mul_f32 v50, v40, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v38, -v14, v41, v49
	v_fma_f32 v51, -v15, v50, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v41, v38, v17
	v_cvt_i32_f32_e32 v38, v39
	v_fmac_f32_e32 v50, v51, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v14, -v14, v41, v49
	v_ldexp_f32 v16, v16, v38
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v15, -v15, v50, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v14, v17, v41
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v15, v18, v50
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v43
	v_div_fixup_f32 v0, v38, v0, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v12, v14, v12, v42
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v16, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v43
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v39, v13, v12 :: v_dual_cndmask_b32 v14, 0x7f800000, v15
	v_mul_f32_e32 v15, 0xbfb8aa3b, v44
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_div_scale_f32 v13, null, v1, v1, v39
	v_div_scale_f32 v49, vcc_lo, v39, v1, v39
	v_add_f32_e32 v12, 1.0, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v18, 0xbfb8aa3b, v44, -v15
	v_rndne_f32_e32 v40, v15
	v_rcp_f32_e32 v16, v13
	v_div_scale_f32 v14, null, v12, v12, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, 0xb2a5705f, v44
	v_sub_f32_e32 v15, v15, v40
	v_cvt_i32_f32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v17, v14
	v_fma_f32 v41, -v13, v16, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_add_f32 v15, v15, v18 :: v_dual_fmac_f32 v16, v41, v16
	v_div_scale_f32 v41, s7, v43, v12, v43
	v_fma_f32 v42, -v14, v17, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v15, v15
	v_dual_fmac_f32 v17, v42, v17 :: v_dual_mul_f32 v42, v49, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v18, v41, v17
	v_fma_f32 v50, -v13, v42, v49
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v15, v15, v40
	v_fma_f32 v51, -v14, v18, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v42, v50, v16
	v_fmac_f32_e32 v18, v51, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v13, -v13, v42, v49
	v_fma_f32 v14, -v14, v18, v41
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v40, v13, v16, v42
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v14, v17, v18
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v44
	v_div_fixup_f32 v1, v40, v1, v39
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v12, v13, v12, v43
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v44
	s_wait_dscnt 0x2
	v_mul_f32_e32 v41, v45, v12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, 0xbfb8aa3b, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, 0x7f800000, v14, vcc_lo
	v_div_scale_f32 v43, null, v2, v2, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v16, v12
	v_add_f32_e32 v45, 1.0, v13
	v_fma_f32 v13, 0xbfb8aa3b, v8, -v12
	v_div_scale_f32 v52, vcc_lo, v41, v2, v41
	v_rcp_f32_e32 v50, v43
	v_sub_f32_e32 v12, v12, v16
	v_div_scale_f32 v49, null, v45, v45, v44
	v_fmac_f32_e32 v13, 0xb2a5705f, v8
	v_div_scale_f32 v53, s7, v44, v45, v44
	v_cvt_i32_f32_e32 v57, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v51, v49
	v_add_f32_e32 v17, v12, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v43, v50, 1.0
	v_exp_f32_e32 v56, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v50, v14, v50
	v_fma_f32 v15, -v49, v51, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v54, v52, v50 :: v_dual_fmac_f32 v51, v15, v51
	global_load_b128 v[12:15], v[23:24], off offset:4
	v_fma_f32 v18, -v43, v54, v52
	v_dual_mul_f32 v55, v53, v51 :: v_dual_fmac_f32 v54, v18, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v49, v55, v53
	v_fma_f32 v43, -v43, v54, v52
	v_ldexp_f32 v52, v56, v57
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v55, v17, v51
	s_clause 0x1
	global_load_b32 v42, v[21:22], off
	global_load_b96 v[16:18], v[23:24], off offset:20
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v43, v43, v50, v54
	s_mov_b32 vcc_lo, s7
	v_fma_f32 v49, -v49, v55, v53
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v2, v43, v2, v41
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v49, v49, v51, v55
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v44, v49, v45, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v50, 0, v52, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v8
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v44, v46, v44 :: v_dual_cndmask_b32 v45, 0x7f800000, v50
	v_mul_f32_e32 v50, 0xbfb8aa3b, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_div_scale_f32 v46, null, v3, v3, v44
	v_div_scale_f32 v57, vcc_lo, v44, v3, v44
	v_add_f32_e32 v45, 1.0, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v53, 0xbfb8aa3b, v9, -v50
	v_rndne_f32_e32 v54, v50
	v_rcp_f32_e32 v51, v46
	v_div_scale_f32 v49, null, v45, v45, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v53, 0xb2a5705f, v9 :: v_dual_sub_f32 v50, v50, v54
	v_cvt_i32_f32_e32 v54, v54
	v_rcp_f32_e32 v52, v49
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v55, -v46, v51, 1.0
	v_dual_add_f32 v50, v50, v53 :: v_dual_fmac_f32 v51, v55, v51
	v_div_scale_f32 v55, s7, v8, v45, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v56, -v49, v52, 1.0
	v_exp_f32_e32 v50, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v52, v56, v52
	v_dual_mul_f32 v56, v57, v51 :: v_dual_mul_f32 v53, v55, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v58, -v46, v56, v57
	v_ldexp_f32 v50, v50, v54
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v59, -v49, v53, v55
	v_dual_fmac_f32 v56, v58, v51 :: v_dual_fmac_f32 v53, v59, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v56, v57
	v_fma_f32 v49, -v49, v53, v55
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v54, v46, v51, v56
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v46, v49, v52, v53
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v8, v46, v45, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, 0, v50, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	s_wait_dscnt 0x1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v8, v47, v8 :: v_dual_cndmask_b32 v45, 0x7f800000, v49
	v_mul_f32_e32 v49, 0xbfb8aa3b, v10
	s_wait_loadcnt 0x3
	v_div_scale_f32 v46, null, v4, v4, v8
	v_div_scale_f32 v57, vcc_lo, v8, v4, v8
	v_add_f32_e32 v45, 1.0, v45
	v_fma_f32 v52, 0xbfb8aa3b, v10, -v49
	v_rndne_f32_e32 v53, v49
	v_rcp_f32_e32 v50, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v47, null, v45, v45, v9
	v_dual_fmac_f32 v52, 0xb2a5705f, v10 :: v_dual_sub_f32 v49, v49, v53
	v_cvt_i32_f32_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v51, v47
	v_fma_f32 v55, -v46, v50, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_add_f32 v49, v49, v52 :: v_dual_fmac_f32 v50, v55, v50
	v_div_scale_f32 v55, s7, v9, v45, v9
	v_fma_f32 v56, -v47, v51, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v49, v49
	v_dual_fmac_f32 v51, v56, v51 :: v_dual_mul_f32 v56, v57, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v52, v55, v51
	v_fma_f32 v58, -v46, v56, v57
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v49, v49, v53
	v_fma_f32 v59, -v47, v52, v55
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v56, v58, v50
	v_fmac_f32_e32 v52, v59, v51
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v56, v57
	v_fma_f32 v47, -v47, v52, v55
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v53, v46, v50, v56
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v46, v47, v51, v52
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	v_div_fixup_f32 v9, v46, v45, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v47, 0, v49, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v9, v48, v9
	v_div_fixup_f32 v3, v54, v3, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v45, 0x7f800000, v47, vcc_lo
	v_div_scale_f32 v56, null, v5, v5, v9
	v_div_scale_f32 v61, vcc_lo, v9, v5, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v55, 1.0, v45
	v_mul_f32_e32 v45, 0xbfb8aa3b, v11
	v_rcp_f32_e32 v58, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v57, null, v55, v55, v10
	v_fma_f32 v46, 0xbfb8aa3b, v11, -v45
	v_rndne_f32_e32 v60, v45
	v_div_scale_f32 v62, s7, v10, v55, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v59, v57
	v_dual_fmac_f32 v46, 0xb2a5705f, v11 :: v_dual_sub_f32 v45, v45, v60
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v56, v58, 1.0
	v_cvt_i32_f32_e32 v60, v60
	v_add_f32_e32 v49, v45, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v58, v47, v58
	v_fma_f32 v48, -v57, v59, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v66, v49
	global_load_b128 v[49:52], v[19:20], off offset:16
	v_fmac_f32_e32 v59, v48, v59
	global_load_b128 v[45:48], v[19:20], off
	v_mul_f32_e32 v63, v61, v58
	v_div_fixup_f32 v4, v53, v4, v8
	v_mul_f32_e32 v64, v62, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v65, -v56, v63, v61
	v_ldexp_f32 v60, v66, v60
	v_fma_f32 v68, -v57, v64, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v63, v65, v58 :: v_dual_fmac_f32 v64, v68, v59
	v_fma_f32 v56, -v56, v63, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v57, -v57, v64, v62
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v56, v56, v58, v63
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v57, v57, v59, v64
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	v_div_fixup_f32 v5, v56, v5, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v10, v57, v55, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v58, 0, v60, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v1, v1, v12
	v_mul_f32_e32 v3, v3, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v55, 0x7f800000, v58, vcc_lo
	s_wait_loadcnt 0x3
	v_fma_f32 v9, v0, v42, v1
	v_fma_f32 v0, v0, v42, -v1
	v_fma_f32 v1, v2, v13, -v3
	s_wait_dscnt 0x0
	v_dual_mul_f32 v10, v29, v10 :: v_dual_add_f32 v29, 1.0, v55
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v5, v5, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v55, null, v6, v6, v10
	v_div_scale_f32 v57, null, v29, v29, v11
	v_div_scale_f32 v62, vcc_lo, v10, v6, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v58, v55
	v_rcp_f32_e32 v59, v57
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v60, -v55, v58, 1.0
	v_fma_f32 v61, -v57, v59, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v58, v60, v58
	v_div_scale_f32 v60, s7, v11, v29, v11
	v_fmac_f32_e32 v59, v61, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v61, v62, v58
	v_mul_f32_e32 v63, v60, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v64, -v55, v61, v62
	v_fma_f32 v65, -v57, v63, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v61, v64, v58
	v_fmac_f32_e32 v63, v65, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v55, -v55, v61, v62
	v_fma_f32 v57, -v57, v63, v60
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v55, v55, v58, v61
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v57, v57, v59, v63
	v_div_fixup_f32 v6, v55, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v11, v57, v29, v11
	v_mul_f32_e32 v11, v30, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v29, null, v7, v7, v11
	v_div_scale_f32 v58, vcc_lo, v11, v7, v11
	v_rcp_f32_e32 v30, v29
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v57, -v29, v30, 1.0
	v_fmac_f32_e32 v30, v57, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v57, v58, v30
	v_fma_f32 v59, -v29, v57, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v57, v59, v30
	v_fma_f32 v12, -v29, v57, v58
	v_mbcnt_lo_u32_b32 v29, -1, 0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v12, v12, v30, v57
	v_xor_b32_e32 v30, 1, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v7, v12, v7, v11
	v_fma_f32 v11, v2, v13, v3
	v_fma_f32 v3, v4, v15, v5
	v_fma_f32 v4, v4, v15, -v5
	v_dual_mul_f32 v7, v7, v18 :: v_dual_add_f32 v2, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v8, v6, v17, v7
	v_fma_f32 v5, v6, v17, -v7
	v_dual_sub_f32 v6, v9, v11 :: v_dual_add_f32 v7, v0, v1
	v_dual_sub_f32 v0, v0, v1 :: v_dual_add_f32 v9, v3, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v4, v5
	v_dual_sub_f32 v3, v3, v8 :: v_dual_sub_f32 v4, v4, v5
	v_dual_add_f32 v5, v2, v9 :: v_dual_sub_f32 v2, v2, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v8, v7, v1 :: v_dual_sub_f32 v1, v7, v1
	v_add_f32_e32 v7, v6, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v3, v6, v3 :: v_dual_add_f32 v6, v0, v4
	v_sub_f32_e32 v0, v0, v4
	ds_swizzle_b32 v4, v5 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v5, -v5, v5, s1
	v_cndmask_b32_e64 v8, -v8, v8, s1
	v_cndmask_b32_e64 v7, -v7, v7, s1
	v_cndmask_b32_e64 v6, -v6, v6, s1
	v_cndmask_b32_e64 v2, -v2, v2, s1
	v_cndmask_b32_e64 v1, -v1, v1, s1
	v_cndmask_b32_e64 v3, -v3, v3, s1
	v_cndmask_b32_e64 v0, -v0, v0, s1
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v8, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v10 :: v_dual_add_f32 v6, v6, v11
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v1, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v0, v0, v15
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v4, -v4, v4, s5
	v_cndmask_b32_e64 v5, -v5, v5, s5
	v_cndmask_b32_e64 v7, -v7, v7, s5
	v_cndmask_b32_e64 v6, -v6, v6, s5
	v_cndmask_b32_e64 v2, -v2, v2, s5
	v_cndmask_b32_e64 v1, -v1, v1, s5
	v_cndmask_b32_e64 v3, -v3, v3, s5
	v_cndmask_b32_e64 v0, -v0, v0, s5
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v4, v8 :: v_dual_add_f32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v10 :: v_dual_add_f32 v6, v6, v11
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v1, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v0, v0, v15
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, s4
	v_cndmask_b32_e64 v5, -v5, v5, s4
	v_cndmask_b32_e64 v7, -v7, v7, s4
	v_cndmask_b32_e64 v6, -v6, v6, s4
	v_cndmask_b32_e64 v2, -v2, v2, s4
	v_cndmask_b32_e64 v1, -v1, v1, s4
	v_cndmask_b32_e64 v3, -v3, v3, s4
	v_cndmask_b32_e64 v0, -v0, v0, s4
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v4, v8 :: v_dual_add_f32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v10 :: v_dual_add_f32 v6, v6, v11
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v1, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v0, v0, v15
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, s2
	v_cndmask_b32_e64 v5, -v5, v5, s2
	v_cndmask_b32_e64 v7, -v7, v7, s2
	v_cndmask_b32_e64 v6, -v6, v6, s2
	v_cndmask_b32_e64 v2, -v2, v2, s2
	v_cndmask_b32_e64 v1, -v1, v1, s2
	v_cndmask_b32_e64 v3, -v3, v3, s2
	v_cndmask_b32_e64 v0, -v0, v0, s2
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v4, v8 :: v_dual_add_f32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v10 :: v_dual_add_f32 v6, v6, v11
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v1, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v0, v0, v15
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v11, v7 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v12, v6 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v13, v2 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v14, v1 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v15, v3 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v16, v0 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, s3
	v_cndmask_b32_e64 v5, -v5, v5, s3
	v_cndmask_b32_e64 v7, -v7, v7, s3
	v_cndmask_b32_e64 v6, -v6, v6, s3
	v_cndmask_b32_e64 v2, -v2, v2, s3
	v_cndmask_b32_e64 v1, -v1, v1, s3
	v_cndmask_b32_e64 v3, -v3, v3, s3
	v_lshlrev_b32_e32 v10, 2, v32
	v_cndmask_b32_e64 v0, -v0, v0, s3
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v4, v8 :: v_dual_add_f32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v11 :: v_dual_add_f32 v6, v6, v12
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v13 :: v_dual_add_f32 v1, v1, v14
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v15 :: v_dual_add_f32 v0, v0, v16
	v_dual_mul_f32 v5, 0x3d800000, v5 :: v_dual_mul_f32 v4, 0x3d800000, v4
	v_dual_mul_f32 v7, 0x3d800000, v7 :: v_dual_mul_f32 v6, 0x3d800000, v6
	v_dual_mul_f32 v1, 0x3d800000, v1 :: v_dual_mul_f32 v2, 0x3d800000, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v3, 0x3d800000, v3 :: v_dual_mul_f32 v0, 0x3d800000, v0
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v11, v4, v45 :: v_dual_mul_f32 v12, v5, v46
	v_dual_mul_f32 v13, v7, v47 :: v_dual_mul_f32 v14, v6, v48
	v_mul_f32_e32 v17, v3, v51
	v_dual_mul_f32 v15, v2, v49 :: v_dual_mul_f32 v16, v1, v50
	v_xor_b32_e32 v9, 16, v29
	ds_bpermute_b32 v1, v10, v12
	ds_bpermute_b32 v3, v10, v13
	ds_bpermute_b32 v2, v10, v15
	ds_bpermute_b32 v6, v10, v16
	ds_bpermute_b32 v7, v10, v17
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	v_mul_f32_e32 v18, v0, v52
	ds_bpermute_b32 v0, v10, v11
	ds_bpermute_b32 v5, v10, v14
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, v2, v0, s1
	v_cndmask_b32_e64 v2, v6, v1, s1
	v_cndmask_b32_e64 v1, v7, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v29, v9, vcc_lo
	ds_bpermute_b32 v8, v10, v18
	v_max3_num_f32 v6, |v4|, |v2|, |v1|
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v8, v5, s1
	v_lshlrev_b32_e32 v5, 2, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e64 v7, |v0|, |v0|
	v_max_num_f32_e32 v3, v6, v7
	v_xor_b32_e32 v7, 8, v29
	ds_bpermute_b32 v6, v5, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v29, v7 :: v_dual_max_num_f32 v8, v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v3, v3, v8 :: v_dual_lshlrev_b32 v6, 2, v7
	v_xor_b32_e32 v8, 4, v29
	ds_bpermute_b32 v7, v6, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v8, v29, v8 :: v_dual_max_num_f32 v9, v7, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v7, 2, v8
	v_max_num_f32_e32 v3, v3, v9
	v_xor_b32_e32 v9, 2, v29
	ds_bpermute_b32 v8, v7, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v29, v9, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v29, v29, v30 :: v_dual_max_num_f32 v8, v8, v8
	v_max_num_f32_e32 v3, v3, v8
	ds_bpermute_b32 v8, v9, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v30, v8, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v3, v3, v30 :: v_dual_lshlrev_b32 v8, 2, v29
	ds_bpermute_b32 v29, v8, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v29, v29, v29
	v_max_num_f32_e32 v29, v3, v29
	v_mov_b32_e32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v29
	s_cbranch_execz .LBB0_174
; %bb.172:                              ; %.preheader76.i.i.5.i
	v_div_scale_f32 v3, null, 0x40e00000, 0x40e00000, v29
	v_div_scale_f32 v38, vcc_lo, v29, 0x40e00000, v29
	s_mov_b32 s17, 0x40e00000
	s_mov_b32 s18, 0xc1000000
	v_rcp_f32_e32 v30, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v3, v30, 1.0
	v_fmac_f32_e32 v30, v37, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v38, v30
	v_fma_f32 v39, -v3, v37, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v39, v30
	v_fma_f32 v3, -v3, v37, v38
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v3, v3, v30, v37
	v_mov_b32_e32 v37, 0x7149f2ca
	v_div_fixup_f32 v30, v3, 0x40e00000, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v3, 1.0 :: v_dual_mul_f32 v30, 0.5, v30
.LBB0_173:                              ; %.preheader.preheader.i.i.5.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s7, s16
	s_add_co_i32 s16, s16, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s16, 8
	v_div_scale_f32 v38, null, s17, s17, s7
	v_div_scale_f32 v39, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v40, v38
	v_xor_b32_e32 v38, 0x80000000, v38
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v38, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v40, v41, v40
	v_mul_f32_e32 v41, v39, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v42, v38, v41, v39
	v_fmac_f32_e32 v41, v42, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v38, v41
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v39, v40, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v38, v38, 0x40e00000, s7
	v_add_f32_e32 v38, 1.0, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v30, v38
	v_div_scale_f32 v39, null, v38, v38, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v47, v39
	v_fma_f32 v51, -v39, v47, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v51, v47
	v_div_scale_f32 v41, null, v38, v38, v2
	v_div_scale_f32 v45, null, v38, v38, v0
	v_div_scale_f32 v43, null, v38, v38, v1
	v_div_scale_f32 v40, vcc_lo, v4, v38, v4
	v_div_scale_f32 v42, s7, v2, v38, v2
	v_rcp_f32_e32 v48, v41
	v_rcp_f32_e32 v50, v45
	v_rcp_f32_e32 v49, v43
	v_div_scale_f32 v44, s8, v1, v38, v1
	v_div_scale_f32 v46, s9, v0, v38, v0
	v_mul_f32_e32 v51, v40, v47
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v52, -v41, v48, 1.0
	v_fma_f32 v54, -v45, v50, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v53, -v43, v49, 1.0
	v_fma_f32 v55, -v39, v51, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v48, v52, v48
	v_dual_fmac_f32 v50, v54, v50 :: v_dual_fmac_f32 v49, v53, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v51, v55, v47 :: v_dual_mul_f32 v52, v42, v48
	v_dual_mul_f32 v54, v46, v50 :: v_dual_mul_f32 v53, v44, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v39, v51, v40
	v_fma_f32 v56, -v41, v52, v42
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v58, -v45, v54, v46
	v_fma_f32 v57, -v43, v53, v44
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v39, v39, v47, v51
	s_mov_b32 vcc_lo, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v52, v56, v48 :: v_dual_fmac_f32 v53, v57, v49
	v_div_fixup_f32 v39, v39, v38, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v40, -v41, v52, v42
	v_fma_f32 v41, -v43, v53, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v39, v39
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v40, v40, v48, v52
	s_mov_b32 vcc_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v41, v41, v49, v53
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v40, v40, v38, v2
	v_med3_num_f32 v39, v39, s18, 0x40e00000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v41, v41, v38, v1
	v_rndne_f32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v39, v38, v4
	v_rndne_f32_e32 v41, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v40, v40, s18, 0x40e00000
	v_fma_f32 v39, v39, v39, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v41, v41, s18, 0x40e00000
	v_fma_f32 v40, -v40, v38, v2
	v_fmac_f32_e32 v54, v58, v50
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v41, v38, v1
	v_fmac_f32_e32 v39, v40, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v42, -v45, v54, v46
	v_fmac_f32_e32 v39, v41, v41
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v42, v42, v50, v54
	v_div_fixup_f32 v42, v42, v38, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v42, v42
	v_med3_num_f32 v42, v42, s18, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v42, v38, v0
	v_fmac_f32_e32 v39, v40, v40
	ds_bpermute_b32 v40, v5, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	ds_bpermute_b32 v40, v6, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	ds_bpermute_b32 v40, v7, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	ds_bpermute_b32 v40, v9, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	ds_bpermute_b32 v40, v8, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v39, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v37, v39, vcc_lo
	v_cndmask_b32_e32 v3, v3, v38, vcc_lo
	s_cbranch_scc1 .LBB0_173
.LBB0_174:                              ; %Flow1065
	s_or_b32 exec_lo, exec_lo, s15
	v_cmp_neq_f32_e64 s7, 0, v29
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_176
; %bb.175:
	v_div_scale_f32 v30, null, v3, v3, v4
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v37, v30
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v38, -v30, v37, 1.0
	v_fmac_f32_e32 v37, v38, v37
	v_div_scale_f32 v38, vcc_lo, v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v39, v38, v37
	v_fma_f32 v40, -v30, v39, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v37
	v_fma_f32 v30, -v30, v39, v38
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v30, v30, v37, v39
	v_div_fixup_f32 v4, v30, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v30, v4
.LBB0_176:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_178
; %bb.177:
	v_div_scale_f32 v4, null, v3, v3, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v29, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v4, v29, 1.0
	v_fmac_f32_e32 v29, v37, v29
	v_div_scale_f32 v37, vcc_lo, v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v29
	v_fma_f32 v39, -v4, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v29
	v_fma_f32 v4, -v4, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v29, v38
	v_div_fixup_f32 v2, v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v29, v2
.LBB0_178:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_180
; %bb.179:
	v_div_scale_f32 v4, null, v3, v3, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v37, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v38, -v4, v37, 1.0
	v_fmac_f32_e32 v37, v38, v37
	v_div_scale_f32 v38, vcc_lo, v1, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v39, v38, v37
	v_fma_f32 v40, -v4, v39, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v37
	v_fma_f32 v4, -v4, v39, v38
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v37, v39
	v_div_fixup_f32 v1, v4, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v4, v1
.LBB0_180:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_182
; %bb.181:
	v_div_scale_f32 v1, null, v3, v3, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v37, v2
	v_div_scale_f32 v37, vcc_lo, v0, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v2
	v_fma_f32 v39, -v1, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v2
	v_fma_f32 v1, -v1, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v38
	v_div_fixup_f32 v0, v1, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_182:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v29, v30
	v_and_b32_e32 v30, 15, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v0, v0, v4, v2
	v_lshl_or_b32 v29, v29, 4, v30
	ds_bpermute_b32 v1, v5, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v6, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v7, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v37, v0, v1
	v_mad_co_i64_i32 v[0:1], null, 0x48, v36, s[12:13]
	ds_bpermute_b32 v38, v9, v37
	v_mad_co_u64_u32 v[0:1], null, 0x48, s10, v[0:1]
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v36, v37, v38
	v_and_b32_e32 v37, 15, v4
	ds_bpermute_b32 v4, v8, v36
	v_lshl_or_b32 v37, v2, 4, v37
	v_mad_co_u64_u32 v[1:2], null, 0x48, s11, v[1:2]
	v_and_b16 v2.h, 0xff, v29.l
	v_add_co_u32 v29, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v37.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[29:30], v2, off offset:8
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB0_184
; %bb.183:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v36, v4
	global_store_b64 v[0:1], v[3:4], off
.LBB0_184:                              ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit.i.5.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_bpermute_b32 v2, v10, v11 offset:64
	ds_bpermute_b32 v3, v10, v12 offset:64
	s_wait_dscnt 0x2
	ds_bpermute_b32 v4, v10, v13 offset:64
	ds_bpermute_b32 v12, v10, v14 offset:64
	ds_bpermute_b32 v11, v10, v15 offset:64
	ds_bpermute_b32 v13, v10, v16 offset:64
	ds_bpermute_b32 v14, v10, v17 offset:64
	ds_bpermute_b32 v15, v10, v18 offset:64
	s_mov_b32 s16, 0
	s_mov_b32 s15, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v11, v11, v2, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v10, v13, v3, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, v14, v4, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v2, v15, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v3, |v11|, |v10|, |v4|
	v_max_num_f32_e64 v12, |v2|, |v2|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v5, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v6, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v7, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v9, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v8, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_dual_max_num_f32 v12, v3, v12 :: v_dual_mov_b32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_187
; %bb.185:                              ; %.preheader76.i.1.i.5.i
	v_div_scale_f32 v3, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s17, 0x40e00000
	s_mov_b32 s18, 0xc1000000
	v_rcp_f32_e32 v13, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v3, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v3, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v3, -v3, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v3, v3, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v3, 0x40e00000, v12
	v_mov_b32_e32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_186:                              ; %.preheader.preheader.i.1.i.5.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s7, s16
	s_add_co_i32 s16, s16, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s16, 8
	v_div_scale_f32 v15, null, s17, s17, s7
	v_div_scale_f32 v16, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v29, v15, v18, v16
	v_fmac_f32_e32 v18, v29, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s7
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v18, null, v15, v15, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v40, v18
	v_fma_f32 v44, -v18, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v44, v40
	v_div_scale_f32 v16, null, v15, v15, v11
	v_div_scale_f32 v30, null, v15, v15, v4
	v_div_scale_f32 v37, null, v15, v15, v2
	v_div_scale_f32 v17, vcc_lo, v11, v15, v11
	v_div_scale_f32 v29, s7, v10, v15, v10
	v_rcp_f32_e32 v39, v16
	v_rcp_f32_e32 v41, v30
	v_rcp_f32_e32 v42, v37
	v_div_scale_f32 v38, s9, v2, v15, v2
	v_div_scale_f32 v36, s8, v4, v15, v4
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v43, -v16, v39, 1.0
	v_fma_f32 v45, -v30, v41, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v37, v42, 1.0
	v_fmac_f32_e32 v39, v43, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v41, v45, v41 :: v_dual_fmac_f32 v42, v46, v42
	v_mul_f32_e32 v44, v29, v40
	v_mul_f32_e32 v43, v17, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v45, v36, v41 :: v_dual_mul_f32 v46, v38, v42
	v_fma_f32 v48, -v18, v44, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v16, v43, v17
	v_fma_f32 v49, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v37, v46, v38
	v_dual_fmac_f32 v44, v48, v40 :: v_dual_fmac_f32 v43, v47, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v45, v49, v41 :: v_dual_fmac_f32 v46, v50, v42
	v_fma_f32 v16, -v16, v43, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v44, v29
	v_fma_f32 v18, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v29, -v37, v46, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v39, v43
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v40, v44
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v16, v16, v15, v11
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v41, v45
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v17, v17, v15, v10
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v29, v29, v42, v46
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v4
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v29, v29, v15, v2
	v_med3_num_f32 v16, v16, s18, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s18, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v29, v29
	v_fma_f32 v16, -v16, v15, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s18, 0x40e00000
	v_fma_f32 v17, -v17, v15, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v29, v29, s18, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v4
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v29, v15, v2
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v5, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v6, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v14, v16 :: v_dual_cndmask_b32 v3, v3, v15
	s_cbranch_scc1 .LBB0_186
.LBB0_187:                              ; %Flow1063
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_cmp_neq_f32_e64 s7, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_189
; %bb.188:
	v_div_scale_f32 v13, null, v3, v3, v11
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v11, v3, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v11, v13, v3, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v11, v11
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v11, v11, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v11
.LBB0_189:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_191
; %bb.190:
	v_div_scale_f32 v11, null, v3, v3, v10
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v12, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v11, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v10, v3, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v11, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v11, -v11, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v12, v15
	v_div_fixup_f32 v10, v11, v3, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v10, v10
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v10, v10, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v10
.LBB0_191:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_193
; %bb.192:
	v_div_scale_f32 v11, null, v3, v3, v4
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v11, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v11, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v11, -v11, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v14, v16
	v_div_fixup_f32 v4, v11, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v11, v4
.LBB0_193:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_195
; %bb.194:
	v_div_scale_f32 v4, null, v3, v3, v2
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v10, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v4, v10, 1.0
	v_fmac_f32_e32 v10, v14, v10
	v_div_scale_f32 v14, vcc_lo, v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v10
	v_fma_f32 v16, -v4, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v10
	v_fma_f32 v4, -v4, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v10, v15
	v_div_fixup_f32 v2, v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v10, v2
.LBB0_195:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v2, v12, v13
	v_mad_co_u64_u32 v[0:1], null, 0x48, s26, v[0:1]
	s_delay_alu instid0(VALU_DEP_2)
	v_add3_u32 v2, v2, v11, v10
	ds_bpermute_b32 v4, v5, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v6, v2
	v_and_b32_e32 v6, 15, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v6, v12, 4, v6
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v7, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v9, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v2, v4
	v_and_b32_e32 v2, 15, v11
	ds_bpermute_b32 v5, v8, v4
	v_lshl_or_b32 v7, v10, 4, v2
	v_mad_co_u64_u32 v[1:2], null, 0x48, s27, v[1:2]
	v_and_b16 v2.h, 0xff, v6.l
	v_add_co_u32 v6, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v7.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[6:7], v2, off offset:8
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_197
; %bb.196:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v4, v5
	global_store_b64 v[0:1], v[3:4], off
.LBB0_197:                              ; %Flow1066
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB0_199
; %bb.198:                              ; %.preheader902.6.i
	ds_store_2addr_b32 v34, v141, v142 offset1:1
	ds_store_2addr_b32 v34, v139, v140 offset0:2 offset1:3
	ds_store_2addr_b32 v34, v137, v138 offset0:4 offset1:5
	ds_store_2addr_b32 v34, v135, v136 offset0:6 offset1:7
	ds_store_2addr_b32 v34, v133, v134 offset0:16 offset1:17
	ds_store_2addr_b32 v34, v131, v132 offset0:18 offset1:19
	ds_store_2addr_b32 v34, v129, v130 offset0:20 offset1:21
	ds_store_2addr_b32 v34, v70, v67 offset0:22 offset1:23
.LBB0_199:                              ; %.loopexit.6.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v36, 48, v35
	s_mov_b32 s14, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s26, v36
	s_cbranch_execz .LBB0_226
; %bb.200:
	v_mad_co_i64_i32 v[0:1], null, s24, v36, 0
	s_mov_b32 s16, 0
	s_mov_b32 s15, exec_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v0, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v28, vcc_lo
	global_load_b128 v[41:44], v[4:5], off
	global_load_b128 v[0:3], v[25:26], off
	global_load_b128 v[8:11], v[4:5], off offset:16
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v4, 0xbfb8aa3b, v41
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, 0xbfb8aa3b, v41, -v4
	v_rndne_f32_e32 v6, v4
	v_dual_fmac_f32 v5, 0xb2a5705f, v41 :: v_dual_sub_f32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	v_exp_f32_e32 v4, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v41
	global_load_b128 v[4:7], v[25:26], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0x7f800000, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v14, 1.0, v12
	v_mul_f32_e32 v12, 0xbfb8aa3b, v42
	v_div_scale_f32 v15, null, v14, v14, v41
	v_div_scale_f32 v29, vcc_lo, v41, v14, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v17, 0xbfb8aa3b, v42, -v12
	v_rndne_f32_e32 v18, v12
	v_rcp_f32_e32 v16, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, 0xb2a5705f, v42
	v_sub_f32_e32 v12, v12, v18
	v_cvt_i32_f32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v12, v12, v17
	v_fma_f32 v13, -v15, v16, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v17, v12
	v_fmac_f32_e32 v16, v13, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v30, v29, v16
	v_ldexp_f32 v17, v17, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v15, v30, v29
	v_fmac_f32_e32 v30, v13, v16
	ds_load_2addr_b32 v[12:13], v33 offset1:1
	v_fma_f32 v15, -v15, v30, v29
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v15, v15, v16, v30
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v42
	ds_load_2addr_b32 v[45:46], v33 offset0:2 offset1:3
	ds_load_2addr_b32 v[47:48], v33 offset0:4 offset1:5
	ds_load_2addr_b32 v[29:30], v33 offset0:6 offset1:7
	v_div_fixup_f32 v14, v15, v14, v41
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, 0, v17, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v42
	s_wait_dscnt 0x3
	v_mul_f32_e32 v37, v12, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v16, vcc_lo
	v_mul_f32_e32 v16, 0xbfb8aa3b, v43
	s_wait_loadcnt 0x2
	v_div_scale_f32 v14, null, v0, v0, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, 1.0, v15
	v_fma_f32 v38, 0xbfb8aa3b, v43, -v16
	v_rndne_f32_e32 v39, v16
	v_div_scale_f32 v49, vcc_lo, v37, v0, v37
	v_rcp_f32_e32 v17, v14
	v_div_scale_f32 v15, null, v12, v12, v42
	v_fmac_f32_e32 v38, 0xb2a5705f, v43
	v_sub_f32_e32 v16, v16, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v15
	v_add_f32_e32 v16, v16, v38
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v40, -v14, v17, 1.0
	v_exp_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v17, v40, v17
	v_fma_f32 v41, -v15, v18, 1.0
	v_div_scale_f32 v40, s7, v42, v12, v42
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v41, v18
	v_dual_mul_f32 v41, v49, v17 :: v_dual_mul_f32 v50, v40, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v38, -v14, v41, v49
	v_fma_f32 v51, -v15, v50, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v41, v38, v17
	v_cvt_i32_f32_e32 v38, v39
	v_fmac_f32_e32 v50, v51, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v14, -v14, v41, v49
	v_ldexp_f32 v16, v16, v38
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v15, -v15, v50, v40
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v14, v17, v41
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v15, v18, v50
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v43
	v_div_fixup_f32 v0, v38, v0, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v12, v14, v12, v42
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v16, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v43
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v39, v13, v12 :: v_dual_cndmask_b32 v14, 0x7f800000, v15
	v_mul_f32_e32 v15, 0xbfb8aa3b, v44
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_div_scale_f32 v13, null, v1, v1, v39
	v_div_scale_f32 v49, vcc_lo, v39, v1, v39
	v_add_f32_e32 v12, 1.0, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v18, 0xbfb8aa3b, v44, -v15
	v_rndne_f32_e32 v40, v15
	v_rcp_f32_e32 v16, v13
	v_div_scale_f32 v14, null, v12, v12, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, 0xb2a5705f, v44
	v_sub_f32_e32 v15, v15, v40
	v_cvt_i32_f32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v17, v14
	v_fma_f32 v41, -v13, v16, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_add_f32 v15, v15, v18 :: v_dual_fmac_f32 v16, v41, v16
	v_div_scale_f32 v41, s7, v43, v12, v43
	v_fma_f32 v42, -v14, v17, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v15, v15
	v_dual_fmac_f32 v17, v42, v17 :: v_dual_mul_f32 v42, v49, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v18, v41, v17
	v_fma_f32 v50, -v13, v42, v49
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v15, v15, v40
	v_fma_f32 v51, -v14, v18, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v42, v50, v16
	v_fmac_f32_e32 v18, v51, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v13, -v13, v42, v49
	v_fma_f32 v14, -v14, v18, v41
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v40, v13, v16, v42
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v14, v17, v18
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v44
	v_div_fixup_f32 v1, v40, v1, v39
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v12, v13, v12, v43
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v44
	s_wait_dscnt 0x2
	v_mul_f32_e32 v41, v45, v12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, 0xbfb8aa3b, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, 0x7f800000, v14, vcc_lo
	v_div_scale_f32 v43, null, v2, v2, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v16, v12
	v_add_f32_e32 v45, 1.0, v13
	v_fma_f32 v13, 0xbfb8aa3b, v8, -v12
	v_div_scale_f32 v52, vcc_lo, v41, v2, v41
	v_rcp_f32_e32 v50, v43
	v_sub_f32_e32 v12, v12, v16
	v_div_scale_f32 v49, null, v45, v45, v44
	v_fmac_f32_e32 v13, 0xb2a5705f, v8
	v_div_scale_f32 v53, s7, v44, v45, v44
	v_cvt_i32_f32_e32 v57, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v51, v49
	v_add_f32_e32 v17, v12, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v43, v50, 1.0
	v_exp_f32_e32 v56, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v50, v14, v50
	v_fma_f32 v15, -v49, v51, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v54, v52, v50 :: v_dual_fmac_f32 v51, v15, v51
	global_load_b128 v[12:15], v[23:24], off offset:4
	v_fma_f32 v18, -v43, v54, v52
	v_dual_mul_f32 v55, v53, v51 :: v_dual_fmac_f32 v54, v18, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v49, v55, v53
	v_fma_f32 v43, -v43, v54, v52
	v_ldexp_f32 v52, v56, v57
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v55, v17, v51
	s_clause 0x1
	global_load_b32 v42, v[21:22], off
	global_load_b96 v[16:18], v[23:24], off offset:20
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v43, v43, v50, v54
	s_mov_b32 vcc_lo, s7
	v_fma_f32 v49, -v49, v55, v53
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v2, v43, v2, v41
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v49, v49, v51, v55
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v44, v49, v45, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v50, 0, v52, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v8
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v44, v46, v44 :: v_dual_cndmask_b32 v45, 0x7f800000, v50
	v_mul_f32_e32 v50, 0xbfb8aa3b, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_div_scale_f32 v46, null, v3, v3, v44
	v_div_scale_f32 v57, vcc_lo, v44, v3, v44
	v_add_f32_e32 v45, 1.0, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v53, 0xbfb8aa3b, v9, -v50
	v_rndne_f32_e32 v54, v50
	v_rcp_f32_e32 v51, v46
	v_div_scale_f32 v49, null, v45, v45, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v53, 0xb2a5705f, v9 :: v_dual_sub_f32 v50, v50, v54
	v_cvt_i32_f32_e32 v54, v54
	v_rcp_f32_e32 v52, v49
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v55, -v46, v51, 1.0
	v_dual_add_f32 v50, v50, v53 :: v_dual_fmac_f32 v51, v55, v51
	v_div_scale_f32 v55, s7, v8, v45, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v56, -v49, v52, 1.0
	v_exp_f32_e32 v50, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v52, v56, v52
	v_dual_mul_f32 v56, v57, v51 :: v_dual_mul_f32 v53, v55, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v58, -v46, v56, v57
	v_ldexp_f32 v50, v50, v54
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v59, -v49, v53, v55
	v_dual_fmac_f32 v56, v58, v51 :: v_dual_fmac_f32 v53, v59, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v56, v57
	v_fma_f32 v49, -v49, v53, v55
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v54, v46, v51, v56
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v46, v49, v52, v53
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v8, v46, v45, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, 0, v50, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	s_wait_dscnt 0x1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v8, v47, v8 :: v_dual_cndmask_b32 v45, 0x7f800000, v49
	v_mul_f32_e32 v49, 0xbfb8aa3b, v10
	s_wait_loadcnt 0x3
	v_div_scale_f32 v46, null, v4, v4, v8
	v_div_scale_f32 v57, vcc_lo, v8, v4, v8
	v_add_f32_e32 v45, 1.0, v45
	v_fma_f32 v52, 0xbfb8aa3b, v10, -v49
	v_rndne_f32_e32 v53, v49
	v_rcp_f32_e32 v50, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v47, null, v45, v45, v9
	v_dual_fmac_f32 v52, 0xb2a5705f, v10 :: v_dual_sub_f32 v49, v49, v53
	v_cvt_i32_f32_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v51, v47
	v_fma_f32 v55, -v46, v50, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_dual_add_f32 v49, v49, v52 :: v_dual_fmac_f32 v50, v55, v50
	v_div_scale_f32 v55, s7, v9, v45, v9
	v_fma_f32 v56, -v47, v51, 1.0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v49, v49
	v_dual_fmac_f32 v51, v56, v51 :: v_dual_mul_f32 v56, v57, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v52, v55, v51
	v_fma_f32 v58, -v46, v56, v57
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v49, v49, v53
	v_fma_f32 v59, -v47, v52, v55
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v56, v58, v50
	v_fmac_f32_e32 v52, v59, v51
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v46, v56, v57
	v_fma_f32 v47, -v47, v52, v55
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v53, v46, v50, v56
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v46, v47, v51, v52
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	v_div_fixup_f32 v9, v46, v45, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v47, 0, v49, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v9, v48, v9
	v_div_fixup_f32 v3, v54, v3, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v45, 0x7f800000, v47, vcc_lo
	v_div_scale_f32 v56, null, v5, v5, v9
	v_div_scale_f32 v61, vcc_lo, v9, v5, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v55, 1.0, v45
	v_mul_f32_e32 v45, 0xbfb8aa3b, v11
	v_rcp_f32_e32 v58, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_scale_f32 v57, null, v55, v55, v10
	v_fma_f32 v46, 0xbfb8aa3b, v11, -v45
	v_rndne_f32_e32 v60, v45
	v_div_scale_f32 v62, s7, v10, v55, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v59, v57
	v_dual_fmac_f32 v46, 0xb2a5705f, v11 :: v_dual_sub_f32 v45, v45, v60
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v56, v58, 1.0
	v_cvt_i32_f32_e32 v60, v60
	v_add_f32_e32 v49, v45, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v58, v47, v58
	v_fma_f32 v48, -v57, v59, 1.0
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v66, v49
	global_load_b128 v[49:52], v[19:20], off offset:16
	v_fmac_f32_e32 v59, v48, v59
	global_load_b128 v[45:48], v[19:20], off
	v_mul_f32_e32 v63, v61, v58
	v_div_fixup_f32 v4, v53, v4, v8
	v_mul_f32_e32 v64, v62, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v65, -v56, v63, v61
	v_ldexp_f32 v60, v66, v60
	v_fma_f32 v68, -v57, v64, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v63, v65, v58 :: v_dual_fmac_f32 v64, v68, v59
	v_fma_f32 v56, -v56, v63, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v57, -v57, v64, v62
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v56, v56, v58, v63
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v57, v57, v59, v64
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	v_div_fixup_f32 v5, v56, v5, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v10, v57, v55, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v58, 0, v60, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v1, v1, v12
	v_mul_f32_e32 v3, v3, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v55, 0x7f800000, v58, vcc_lo
	s_wait_loadcnt 0x3
	v_fma_f32 v9, v0, v42, v1
	v_fma_f32 v0, v0, v42, -v1
	v_fma_f32 v1, v2, v13, -v3
	s_wait_dscnt 0x0
	v_dual_mul_f32 v10, v29, v10 :: v_dual_add_f32 v29, 1.0, v55
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v5, v5, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v55, null, v6, v6, v10
	v_div_scale_f32 v57, null, v29, v29, v11
	v_div_scale_f32 v62, vcc_lo, v10, v6, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v58, v55
	v_rcp_f32_e32 v59, v57
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v60, -v55, v58, 1.0
	v_fma_f32 v61, -v57, v59, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v58, v60, v58
	v_div_scale_f32 v60, s7, v11, v29, v11
	v_fmac_f32_e32 v59, v61, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v61, v62, v58
	v_mul_f32_e32 v63, v60, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v64, -v55, v61, v62
	v_fma_f32 v65, -v57, v63, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v61, v64, v58
	v_fmac_f32_e32 v63, v65, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v55, -v55, v61, v62
	v_fma_f32 v57, -v57, v63, v60
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v55, v55, v58, v61
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v57, v57, v59, v63
	v_div_fixup_f32 v6, v55, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v11, v57, v29, v11
	v_mul_f32_e32 v11, v30, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v29, null, v7, v7, v11
	v_div_scale_f32 v58, vcc_lo, v11, v7, v11
	v_rcp_f32_e32 v30, v29
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v57, -v29, v30, 1.0
	v_fmac_f32_e32 v30, v57, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v57, v58, v30
	v_fma_f32 v59, -v29, v57, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v57, v59, v30
	v_fma_f32 v12, -v29, v57, v58
	v_mbcnt_lo_u32_b32 v29, -1, 0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v12, v12, v30, v57
	v_xor_b32_e32 v30, 1, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v7, v12, v7, v11
	v_fma_f32 v11, v2, v13, v3
	v_fma_f32 v3, v4, v15, v5
	v_fma_f32 v4, v4, v15, -v5
	v_dual_mul_f32 v7, v7, v18 :: v_dual_add_f32 v2, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v8, v6, v17, v7
	v_fma_f32 v5, v6, v17, -v7
	v_dual_sub_f32 v6, v9, v11 :: v_dual_add_f32 v7, v0, v1
	v_dual_sub_f32 v0, v0, v1 :: v_dual_add_f32 v9, v3, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v4, v5
	v_dual_sub_f32 v3, v3, v8 :: v_dual_sub_f32 v4, v4, v5
	v_dual_add_f32 v5, v2, v9 :: v_dual_sub_f32 v2, v2, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v8, v7, v1 :: v_dual_sub_f32 v1, v7, v1
	v_add_f32_e32 v7, v6, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v3, v6, v3 :: v_dual_add_f32 v6, v0, v4
	v_sub_f32_e32 v0, v0, v4
	ds_swizzle_b32 v4, v5 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v5, -v5, v5, s1
	v_cndmask_b32_e64 v8, -v8, v8, s1
	v_cndmask_b32_e64 v7, -v7, v7, s1
	v_cndmask_b32_e64 v6, -v6, v6, s1
	v_cndmask_b32_e64 v2, -v2, v2, s1
	v_cndmask_b32_e64 v1, -v1, v1, s1
	v_cndmask_b32_e64 v3, -v3, v3, s1
	v_cndmask_b32_e64 v0, -v0, v0, s1
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v8, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v10 :: v_dual_add_f32 v6, v6, v11
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v1, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v0, v0, v15
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v4, -v4, v4, s5
	v_cndmask_b32_e64 v5, -v5, v5, s5
	v_cndmask_b32_e64 v7, -v7, v7, s5
	v_cndmask_b32_e64 v6, -v6, v6, s5
	v_cndmask_b32_e64 v2, -v2, v2, s5
	v_cndmask_b32_e64 v1, -v1, v1, s5
	v_cndmask_b32_e64 v3, -v3, v3, s5
	v_cndmask_b32_e64 v0, -v0, v0, s5
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v4, v8 :: v_dual_add_f32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v10 :: v_dual_add_f32 v6, v6, v11
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v1, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v0, v0, v15
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, s4
	v_cndmask_b32_e64 v5, -v5, v5, s4
	v_cndmask_b32_e64 v7, -v7, v7, s4
	v_cndmask_b32_e64 v6, -v6, v6, s4
	v_cndmask_b32_e64 v2, -v2, v2, s4
	v_cndmask_b32_e64 v1, -v1, v1, s4
	v_cndmask_b32_e64 v3, -v3, v3, s4
	v_cndmask_b32_e64 v0, -v0, v0, s4
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v4, v8 :: v_dual_add_f32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v10 :: v_dual_add_f32 v6, v6, v11
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v1, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v0, v0, v15
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, s2
	v_cndmask_b32_e64 v5, -v5, v5, s2
	v_cndmask_b32_e64 v7, -v7, v7, s2
	v_cndmask_b32_e64 v6, -v6, v6, s2
	v_cndmask_b32_e64 v2, -v2, v2, s2
	v_cndmask_b32_e64 v1, -v1, v1, s2
	v_cndmask_b32_e64 v3, -v3, v3, s2
	v_cndmask_b32_e64 v0, -v0, v0, s2
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v4, v8 :: v_dual_add_f32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v10 :: v_dual_add_f32 v6, v6, v11
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v1, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v0, v0, v15
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v11, v7 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v12, v6 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v13, v2 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v14, v1 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v15, v3 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v16, v0 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, s3
	v_cndmask_b32_e64 v5, -v5, v5, s3
	v_cndmask_b32_e64 v7, -v7, v7, s3
	v_cndmask_b32_e64 v6, -v6, v6, s3
	v_cndmask_b32_e64 v2, -v2, v2, s3
	v_cndmask_b32_e64 v1, -v1, v1, s3
	v_cndmask_b32_e64 v3, -v3, v3, s3
	v_lshlrev_b32_e32 v10, 2, v32
	v_cndmask_b32_e64 v0, -v0, v0, s3
	s_wait_dscnt 0x6
	v_dual_add_f32 v4, v4, v8 :: v_dual_add_f32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v11 :: v_dual_add_f32 v6, v6, v12
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v13 :: v_dual_add_f32 v1, v1, v14
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v15 :: v_dual_add_f32 v0, v0, v16
	v_dual_mul_f32 v5, 0x3d800000, v5 :: v_dual_mul_f32 v4, 0x3d800000, v4
	v_dual_mul_f32 v7, 0x3d800000, v7 :: v_dual_mul_f32 v6, 0x3d800000, v6
	v_dual_mul_f32 v1, 0x3d800000, v1 :: v_dual_mul_f32 v2, 0x3d800000, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v3, 0x3d800000, v3 :: v_dual_mul_f32 v0, 0x3d800000, v0
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v11, v4, v45 :: v_dual_mul_f32 v12, v5, v46
	v_dual_mul_f32 v13, v7, v47 :: v_dual_mul_f32 v14, v6, v48
	v_mul_f32_e32 v17, v3, v51
	v_dual_mul_f32 v15, v2, v49 :: v_dual_mul_f32 v16, v1, v50
	v_xor_b32_e32 v9, 16, v29
	ds_bpermute_b32 v1, v10, v12
	ds_bpermute_b32 v3, v10, v13
	ds_bpermute_b32 v2, v10, v15
	ds_bpermute_b32 v6, v10, v16
	ds_bpermute_b32 v7, v10, v17
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	v_mul_f32_e32 v18, v0, v52
	ds_bpermute_b32 v0, v10, v11
	ds_bpermute_b32 v5, v10, v14
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, v2, v0, s1
	v_cndmask_b32_e64 v2, v6, v1, s1
	v_cndmask_b32_e64 v1, v7, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v29, v9, vcc_lo
	ds_bpermute_b32 v8, v10, v18
	v_max3_num_f32 v6, |v4|, |v2|, |v1|
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v8, v5, s1
	v_lshlrev_b32_e32 v5, 2, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e64 v7, |v0|, |v0|
	v_max_num_f32_e32 v3, v6, v7
	v_xor_b32_e32 v7, 8, v29
	ds_bpermute_b32 v6, v5, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v29, v7 :: v_dual_max_num_f32 v8, v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v3, v3, v8 :: v_dual_lshlrev_b32 v6, 2, v7
	v_xor_b32_e32 v8, 4, v29
	ds_bpermute_b32 v7, v6, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v8, v29, v8 :: v_dual_max_num_f32 v9, v7, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v7, 2, v8
	v_max_num_f32_e32 v3, v3, v9
	v_xor_b32_e32 v9, 2, v29
	ds_bpermute_b32 v8, v7, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v29, v9, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v29, v29, v30 :: v_dual_max_num_f32 v8, v8, v8
	v_max_num_f32_e32 v3, v3, v8
	ds_bpermute_b32 v8, v9, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v30, v8, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v3, v3, v30 :: v_dual_lshlrev_b32 v8, 2, v29
	ds_bpermute_b32 v29, v8, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v29, v29, v29
	v_max_num_f32_e32 v29, v3, v29
	v_mov_b32_e32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v29
	s_cbranch_execz .LBB0_203
; %bb.201:                              ; %.preheader76.i.i.6.i
	v_div_scale_f32 v3, null, 0x40e00000, 0x40e00000, v29
	v_div_scale_f32 v38, vcc_lo, v29, 0x40e00000, v29
	s_mov_b32 s17, 0x40e00000
	s_mov_b32 s18, 0xc1000000
	v_rcp_f32_e32 v30, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v3, v30, 1.0
	v_fmac_f32_e32 v30, v37, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, v38, v30
	v_fma_f32 v39, -v3, v37, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v39, v30
	v_fma_f32 v3, -v3, v37, v38
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v3, v3, v30, v37
	v_mov_b32_e32 v37, 0x7149f2ca
	v_div_fixup_f32 v30, v3, 0x40e00000, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v3, 1.0 :: v_dual_mul_f32 v30, 0.5, v30
.LBB0_202:                              ; %.preheader.preheader.i.i.6.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s7, s16
	s_add_co_i32 s16, s16, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s16, 8
	v_div_scale_f32 v38, null, s17, s17, s7
	v_div_scale_f32 v39, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v40, v38
	v_xor_b32_e32 v38, 0x80000000, v38
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v38, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v40, v41, v40
	v_mul_f32_e32 v41, v39, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v42, v38, v41, v39
	v_fmac_f32_e32 v41, v42, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v38, v41
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v39, v40, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v38, v38, 0x40e00000, s7
	v_add_f32_e32 v38, 1.0, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v30, v38
	v_div_scale_f32 v39, null, v38, v38, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v47, v39
	v_fma_f32 v51, -v39, v47, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v47, v51, v47
	v_div_scale_f32 v41, null, v38, v38, v2
	v_div_scale_f32 v45, null, v38, v38, v0
	v_div_scale_f32 v43, null, v38, v38, v1
	v_div_scale_f32 v40, vcc_lo, v4, v38, v4
	v_div_scale_f32 v42, s7, v2, v38, v2
	v_rcp_f32_e32 v48, v41
	v_rcp_f32_e32 v50, v45
	v_rcp_f32_e32 v49, v43
	v_div_scale_f32 v44, s8, v1, v38, v1
	v_div_scale_f32 v46, s9, v0, v38, v0
	v_mul_f32_e32 v51, v40, v47
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v52, -v41, v48, 1.0
	v_fma_f32 v54, -v45, v50, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v53, -v43, v49, 1.0
	v_fma_f32 v55, -v39, v51, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v48, v52, v48
	v_dual_fmac_f32 v50, v54, v50 :: v_dual_fmac_f32 v49, v53, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v51, v55, v47 :: v_dual_mul_f32 v52, v42, v48
	v_dual_mul_f32 v54, v46, v50 :: v_dual_mul_f32 v53, v44, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v39, v51, v40
	v_fma_f32 v56, -v41, v52, v42
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v58, -v45, v54, v46
	v_fma_f32 v57, -v43, v53, v44
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v39, v39, v47, v51
	s_mov_b32 vcc_lo, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v52, v56, v48 :: v_dual_fmac_f32 v53, v57, v49
	v_div_fixup_f32 v39, v39, v38, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v40, -v41, v52, v42
	v_fma_f32 v41, -v43, v53, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v39, v39
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v40, v40, v48, v52
	s_mov_b32 vcc_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v41, v41, v49, v53
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v40, v40, v38, v2
	v_med3_num_f32 v39, v39, s18, 0x40e00000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v41, v41, v38, v1
	v_rndne_f32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v39, -v39, v38, v4
	v_rndne_f32_e32 v41, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v40, v40, s18, 0x40e00000
	v_fma_f32 v39, v39, v39, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v41, v41, s18, 0x40e00000
	v_fma_f32 v40, -v40, v38, v2
	v_fmac_f32_e32 v54, v58, v50
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v41, v38, v1
	v_fmac_f32_e32 v39, v40, v40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v42, -v45, v54, v46
	v_fmac_f32_e32 v39, v41, v41
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v42, v42, v50, v54
	v_div_fixup_f32 v42, v42, v38, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v42, v42
	v_med3_num_f32 v42, v42, s18, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v42, v38, v0
	v_fmac_f32_e32 v39, v40, v40
	ds_bpermute_b32 v40, v5, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	ds_bpermute_b32 v40, v6, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	ds_bpermute_b32 v40, v7, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	ds_bpermute_b32 v40, v9, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	ds_bpermute_b32 v40, v8, v39
	s_wait_dscnt 0x0
	v_add_f32_e32 v39, v39, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v39, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v37, v39, vcc_lo
	v_cndmask_b32_e32 v3, v3, v38, vcc_lo
	s_cbranch_scc1 .LBB0_202
.LBB0_203:                              ; %Flow1059
	s_or_b32 exec_lo, exec_lo, s15
	v_cmp_neq_f32_e64 s7, 0, v29
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_205
; %bb.204:
	v_div_scale_f32 v30, null, v3, v3, v4
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v37, v30
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v38, -v30, v37, 1.0
	v_fmac_f32_e32 v37, v38, v37
	v_div_scale_f32 v38, vcc_lo, v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v39, v38, v37
	v_fma_f32 v40, -v30, v39, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v37
	v_fma_f32 v30, -v30, v39, v38
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v30, v30, v37, v39
	v_div_fixup_f32 v4, v30, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v30, v4
.LBB0_205:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_207
; %bb.206:
	v_div_scale_f32 v4, null, v3, v3, v2
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v29, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v4, v29, 1.0
	v_fmac_f32_e32 v29, v37, v29
	v_div_scale_f32 v37, vcc_lo, v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v29
	v_fma_f32 v39, -v4, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v29
	v_fma_f32 v4, -v4, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v29, v38
	v_div_fixup_f32 v2, v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v29, v2
.LBB0_207:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_209
; %bb.208:
	v_div_scale_f32 v4, null, v3, v3, v1
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v37, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v38, -v4, v37, 1.0
	v_fmac_f32_e32 v37, v38, v37
	v_div_scale_f32 v38, vcc_lo, v1, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v39, v38, v37
	v_fma_f32 v40, -v4, v39, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v40, v37
	v_fma_f32 v4, -v4, v39, v38
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v37, v39
	v_div_fixup_f32 v1, v4, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v4, v1
.LBB0_209:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_211
; %bb.210:
	v_div_scale_f32 v1, null, v3, v3, v0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v37, v2
	v_div_scale_f32 v37, vcc_lo, v0, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v38, v37, v2
	v_fma_f32 v39, -v1, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v39, v2
	v_fma_f32 v1, -v1, v38, v37
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v38
	v_div_fixup_f32 v0, v1, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_211:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v0, v29, v30
	v_and_b32_e32 v30, 15, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v0, v0, v4, v2
	v_lshl_or_b32 v29, v29, 4, v30
	ds_bpermute_b32 v1, v5, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v6, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v7, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v37, v0, v1
	v_mad_co_i64_i32 v[0:1], null, 0x48, v36, s[12:13]
	ds_bpermute_b32 v38, v9, v37
	v_mad_co_u64_u32 v[0:1], null, 0x48, s10, v[0:1]
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v36, v37, v38
	v_and_b32_e32 v37, 15, v4
	ds_bpermute_b32 v4, v8, v36
	v_lshl_or_b32 v37, v2, 4, v37
	v_mad_co_u64_u32 v[1:2], null, 0x48, s11, v[1:2]
	v_and_b16 v2.h, 0xff, v29.l
	v_add_co_u32 v29, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v37.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[29:30], v2, off offset:8
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB0_213
; %bb.212:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v36, v4
	global_store_b64 v[0:1], v[3:4], off
.LBB0_213:                              ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit.i.6.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_bpermute_b32 v2, v10, v11 offset:64
	ds_bpermute_b32 v3, v10, v12 offset:64
	s_wait_dscnt 0x2
	ds_bpermute_b32 v4, v10, v13 offset:64
	ds_bpermute_b32 v12, v10, v14 offset:64
	ds_bpermute_b32 v11, v10, v15 offset:64
	ds_bpermute_b32 v13, v10, v16 offset:64
	ds_bpermute_b32 v14, v10, v17 offset:64
	ds_bpermute_b32 v15, v10, v18 offset:64
	s_mov_b32 s16, 0
	s_mov_b32 s15, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v11, v11, v2, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v10, v13, v3, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, v14, v4, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v2, v15, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v3, |v11|, |v10|, |v4|
	v_max_num_f32_e64 v12, |v2|, |v2|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v5, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v6, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v7, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v9, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v8, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_dual_max_num_f32 v12, v3, v12 :: v_dual_mov_b32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_216
; %bb.214:                              ; %.preheader76.i.1.i.6.i
	v_div_scale_f32 v3, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s17, 0x40e00000
	s_mov_b32 s18, 0xc1000000
	v_rcp_f32_e32 v13, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v3, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v3, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v3, -v3, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v3, v3, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v3, 0x40e00000, v12
	v_mov_b32_e32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_215:                              ; %.preheader.preheader.i.1.i.6.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s7, s16
	s_add_co_i32 s16, s16, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s16, 8
	v_div_scale_f32 v15, null, s17, s17, s7
	v_div_scale_f32 v16, vcc_lo, s7, 0x40e00000, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v29, v15, v18, v16
	v_fmac_f32_e32 v18, v29, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s7
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v18, null, v15, v15, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v40, v18
	v_fma_f32 v44, -v18, v40, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v44, v40
	v_div_scale_f32 v16, null, v15, v15, v11
	v_div_scale_f32 v30, null, v15, v15, v4
	v_div_scale_f32 v37, null, v15, v15, v2
	v_div_scale_f32 v17, vcc_lo, v11, v15, v11
	v_div_scale_f32 v29, s7, v10, v15, v10
	v_rcp_f32_e32 v39, v16
	v_rcp_f32_e32 v41, v30
	v_rcp_f32_e32 v42, v37
	v_div_scale_f32 v38, s9, v2, v15, v2
	v_div_scale_f32 v36, s8, v4, v15, v4
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v43, -v16, v39, 1.0
	v_fma_f32 v45, -v30, v41, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v37, v42, 1.0
	v_fmac_f32_e32 v39, v43, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v41, v45, v41 :: v_dual_fmac_f32 v42, v46, v42
	v_mul_f32_e32 v44, v29, v40
	v_mul_f32_e32 v43, v17, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v45, v36, v41 :: v_dual_mul_f32 v46, v38, v42
	v_fma_f32 v48, -v18, v44, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v47, -v16, v43, v17
	v_fma_f32 v49, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v50, -v37, v46, v38
	v_dual_fmac_f32 v44, v48, v40 :: v_dual_fmac_f32 v43, v47, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v45, v49, v41 :: v_dual_fmac_f32 v46, v50, v42
	v_fma_f32 v16, -v16, v43, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v44, v29
	v_fma_f32 v18, -v30, v45, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v29, -v37, v46, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v39, v43
	s_mov_b32 vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v40, v44
	s_mov_b32 vcc_lo, s8
	v_div_fixup_f32 v16, v16, v15, v11
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v41, v45
	s_mov_b32 vcc_lo, s9
	v_div_fixup_f32 v17, v17, v15, v10
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v29, v29, v42, v46
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v4
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v29, v29, v15, v2
	v_med3_num_f32 v16, v16, s18, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s18, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v29, v29
	v_fma_f32 v16, -v16, v15, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s18, 0x40e00000
	v_fma_f32 v17, -v17, v15, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v29, v29, s18, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v4
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v29, v15, v2
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v5, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v6, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v14, v16 :: v_dual_cndmask_b32 v3, v3, v15
	s_cbranch_scc1 .LBB0_215
.LBB0_216:                              ; %Flow1057
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_cmp_neq_f32_e64 s7, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_218
; %bb.217:
	v_div_scale_f32 v13, null, v3, v3, v11
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v11, v3, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v11, v13, v3, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v11, v11
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v11, v11, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v11
.LBB0_218:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_220
; %bb.219:
	v_div_scale_f32 v11, null, v3, v3, v10
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v12, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v11, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v10, v3, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v11, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v11, -v11, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v12, v15
	v_div_fixup_f32 v10, v11, v3, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v10, v10
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v10, v10, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v10
.LBB0_220:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_222
; %bb.221:
	v_div_scale_f32 v11, null, v3, v3, v4
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v14, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v11, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v11, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v11, -v11, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v14, v16
	v_div_fixup_f32 v4, v11, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v11, v4
.LBB0_222:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_224
; %bb.223:
	v_div_scale_f32 v4, null, v3, v3, v2
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v10, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v4, v10, 1.0
	v_fmac_f32_e32 v10, v14, v10
	v_div_scale_f32 v14, vcc_lo, v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v10
	v_fma_f32 v16, -v4, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v10
	v_fma_f32 v4, -v4, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v10, v15
	v_div_fixup_f32 v2, v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v10, v2
.LBB0_224:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v2, v12, v13
	v_mad_co_u64_u32 v[0:1], null, 0x48, s26, v[0:1]
	s_delay_alu instid0(VALU_DEP_2)
	v_add3_u32 v2, v2, v11, v10
	ds_bpermute_b32 v4, v5, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v6, v2
	v_and_b32_e32 v6, 15, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v6, v12, 4, v6
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v7, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v9, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v2, v4
	v_and_b32_e32 v2, 15, v11
	ds_bpermute_b32 v5, v8, v4
	v_lshl_or_b32 v7, v10, 4, v2
	v_mad_co_u64_u32 v[1:2], null, 0x48, s27, v[1:2]
	v_and_b16 v2.h, 0xff, v6.l
	v_add_co_u32 v6, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v7.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[6:7], v2, off offset:8
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_226
; %bb.225:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v4, v5
	global_store_b64 v[0:1], v[3:4], off
.LBB0_226:                              ; %Flow1060
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_228
; %bb.227:                              ; %.preheader902.7.i
	ds_store_2addr_b32 v34, v141, v142 offset1:1
	ds_store_2addr_b32 v34, v139, v140 offset0:2 offset1:3
	ds_store_2addr_b32 v34, v137, v138 offset0:4 offset1:5
	ds_store_2addr_b32 v34, v135, v136 offset0:6 offset1:7
	ds_store_2addr_b32 v34, v133, v134 offset0:16 offset1:17
	ds_store_2addr_b32 v34, v131, v132 offset0:18 offset1:19
	ds_store_2addr_b32 v34, v129, v130 offset0:20 offset1:21
	ds_store_2addr_b32 v34, v70, v67 offset0:22 offset1:23
.LBB0_228:                              ; %.loopexit.7.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v29, 56, v35
	s_mov_b32 s7, exec_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s26, v29
	s_cbranch_execz .LBB0_255
; %bb.229:
	v_mad_co_i64_i32 v[0:1], null, s24, v29, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	v_add_co_u32 v4, vcc_lo, v0, v27
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v1, v28, vcc_lo
	global_load_b128 v[34:37], v[4:5], off
	global_load_b128 v[0:3], v[25:26], off
	global_load_b128 v[8:11], v[4:5], off offset:16
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v4, 0xbfb8aa3b, v34
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v5, 0xbfb8aa3b, v34, -v4
	v_rndne_f32_e32 v6, v4
	v_fmac_f32_e32 v5, 0xb2a5705f, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_f32_e32 v4, v4, v6
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v4, v4
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v12, 0, v4, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v34
	global_load_b128 v[4:7], v[25:26], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, 0x7f800000, v12, vcc_lo
	v_add_f32_e32 v14, 1.0, v12
	v_mul_f32_e32 v12, 0xbfb8aa3b, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v15, null, v14, v14, v34
	v_div_scale_f32 v25, vcc_lo, v34, v14, v34
	v_fma_f32 v17, 0xbfb8aa3b, v35, -v12
	v_rndne_f32_e32 v18, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v16, v15
	v_dual_fmac_f32 v17, 0xb2a5705f, v35 :: v_dual_sub_f32 v12, v12, v18
	v_cvt_i32_f32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v12, v12, v17
	v_fma_f32 v13, -v15, v16, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v17, v12
	v_fmac_f32_e32 v16, v13, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v26, v25, v16
	v_ldexp_f32 v17, v17, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v15, v26, v25
	v_fmac_f32_e32 v26, v13, v16
	ds_load_2addr_b32 v[12:13], v33 offset1:1
	v_fma_f32 v15, -v15, v26, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v15, v15, v16, v26
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v35
	ds_load_2addr_b32 v[38:39], v33 offset0:2 offset1:3
	ds_load_2addr_b32 v[40:41], v33 offset0:4 offset1:5
	ds_load_2addr_b32 v[25:26], v33 offset0:6 offset1:7
	v_div_fixup_f32 v14, v15, v14, v34
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v16, 0, v17, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v35
	s_wait_dscnt 0x3
	v_mul_f32_e32 v27, v12, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0x7f800000, v16, vcc_lo
	v_mul_f32_e32 v16, 0xbfb8aa3b, v36
	s_wait_loadcnt 0x2
	v_div_scale_f32 v14, null, v0, v0, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, 1.0, v15
	v_fma_f32 v28, 0xbfb8aa3b, v36, -v16
	v_rndne_f32_e32 v30, v16
	v_div_scale_f32 v42, vcc_lo, v27, v0, v27
	v_rcp_f32_e32 v17, v14
	v_div_scale_f32 v15, null, v12, v12, v35
	v_fmac_f32_e32 v28, 0xb2a5705f, v36
	v_sub_f32_e32 v16, v16, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v18, v15
	v_add_f32_e32 v16, v16, v28
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v33, -v14, v17, 1.0
	v_exp_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v17, v33, v17
	v_fma_f32 v34, -v15, v18, 1.0
	v_div_scale_f32 v33, s6, v35, v12, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v34, v18
	v_dual_mul_f32 v34, v42, v17 :: v_dual_mul_f32 v43, v33, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v28, -v14, v34, v42
	v_fma_f32 v44, -v15, v43, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v34, v28, v17
	v_cvt_i32_f32_e32 v28, v30
	v_fmac_f32_e32 v43, v44, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v14, -v14, v34, v42
	v_ldexp_f32 v16, v16, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v15, -v15, v43, v33
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v28, v14, v17, v34
	s_mov_b32 vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v14, v15, v18, v43
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v36
	v_div_fixup_f32 v0, v28, v0, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v12, v14, v12, v35
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, 0, v16, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v36
	v_mul_f32_e32 v30, v13, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v14, 0x7f800000, v15, vcc_lo
	v_mul_f32_e32 v15, 0xbfb8aa3b, v37
	v_div_scale_f32 v13, null, v1, v1, v30
	v_div_scale_f32 v42, vcc_lo, v30, v1, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v12, 1.0, v14
	v_fma_f32 v18, 0xbfb8aa3b, v37, -v15
	v_rndne_f32_e32 v33, v15
	v_rcp_f32_e32 v16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v14, null, v12, v12, v36
	v_fmac_f32_e32 v18, 0xb2a5705f, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v15, v15, v33
	v_cvt_i32_f32_e32 v33, v33
	v_rcp_f32_e32 v17, v14
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v34, -v13, v16, 1.0
	v_dual_add_f32 v15, v15, v18 :: v_dual_fmac_f32 v16, v34, v16
	v_div_scale_f32 v34, s6, v36, v12, v36
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v35, -v14, v17, 1.0
	v_exp_f32_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v35, v17
	v_mul_f32_e32 v35, v42, v16
	v_mul_f32_e32 v18, v34, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v43, -v13, v35, v42
	v_ldexp_f32 v15, v15, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v44, -v14, v18, v34
	v_dual_fmac_f32 v35, v43, v16 :: v_dual_fmac_f32 v18, v44, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v13, -v13, v35, v42
	v_fma_f32 v14, -v14, v18, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v33, v13, v16, v35
	s_mov_b32 vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v13, v14, v17, v18
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v37
	v_div_fixup_f32 v12, v13, v12, v36
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, 0, v15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v37
	s_wait_dscnt 0x2
	v_mul_f32_e32 v34, v38, v12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, 0xbfb8aa3b, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, 0x7f800000, v14, vcc_lo
	v_div_scale_f32 v36, null, v2, v2, v34
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v16, v12
	v_add_f32_e32 v35, 1.0, v13
	v_fma_f32 v13, 0xbfb8aa3b, v8, -v12
	v_div_scale_f32 v44, vcc_lo, v34, v2, v34
	v_rcp_f32_e32 v42, v36
	v_sub_f32_e32 v12, v12, v16
	v_div_scale_f32 v38, null, v35, v35, v37
	v_fmac_f32_e32 v13, 0xb2a5705f, v8
	v_div_scale_f32 v45, s6, v37, v35, v37
	v_cvt_i32_f32_e32 v49, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v43, v38
	v_add_f32_e32 v17, v12, v13
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v14, -v36, v42, 1.0
	v_exp_f32_e32 v48, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v42, v14, v42
	v_fma_f32 v15, -v38, v43, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v46, v44, v42 :: v_dual_fmac_f32 v43, v15, v43
	global_load_b128 v[12:15], v[23:24], off offset:4
	v_fma_f32 v18, -v36, v46, v44
	v_dual_mul_f32 v47, v45, v43 :: v_dual_fmac_f32 v46, v18, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, -v38, v47, v45
	v_fmac_f32_e32 v47, v17, v43
	s_clause 0x1
	global_load_b32 v21, v[21:22], off
	global_load_b96 v[16:18], v[23:24], off offset:20
	v_fma_f32 v22, -v36, v46, v44
	v_ldexp_f32 v23, v48, v49
	v_fma_f32 v24, -v38, v47, v45
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v22, v22, v42, v46
	s_mov_b32 vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v24, v24, v43, v47
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v8
	v_div_fixup_f32 v24, v24, v35, v37
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v23, 0, v23, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v8
	v_mul_f32_e32 v37, 0xbfb8aa3b, v9
	v_div_fixup_f32 v1, v33, v1, v30
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v35, 0x7f800000, v23, vcc_lo
	v_mul_f32_e32 v23, v39, v24
	v_fma_f32 v42, 0xbfb8aa3b, v9, -v37
	v_rndne_f32_e32 v43, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v24, 1.0, v35
	v_div_scale_f32 v35, null, v3, v3, v23
	v_div_scale_f32 v46, vcc_lo, v23, v3, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v42, 0xb2a5705f, v9 :: v_dual_sub_f32 v37, v37, v43
	v_div_scale_f32 v36, null, v24, v24, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v38, v35
	v_cvt_i32_f32_e32 v43, v43
	v_add_f32_e32 v37, v37, v42
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v39, v36
	v_exp_f32_e32 v37, v37
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v44, -v35, v38, 1.0
	v_fma_f32 v45, -v36, v39, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v38, v44, v38
	v_div_scale_f32 v44, s6, v8, v24, v8
	v_ldexp_f32 v37, v37, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v39, v45, v39
	v_dual_mul_f32 v45, v46, v38 :: v_dual_mul_f32 v42, v44, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v47, -v35, v45, v46
	v_fma_f32 v48, -v36, v42, v44
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v45, v47, v38
	v_div_fixup_f32 v2, v22, v2, v34
	v_fmac_f32_e32 v42, v48, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v35, -v35, v45, v46
	v_fma_f32 v36, -v36, v42, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v43, v35, v38, v45
	s_mov_b32 vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v35, v36, v39, v42
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v3, v43, v3, v23
	s_delay_alu instid0(VALU_DEP_3)
	v_div_fixup_f32 v8, v35, v24, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v36, 0, v37, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	s_wait_dscnt 0x1
	v_dual_mul_f32 v37, 0xbfb8aa3b, v10 :: v_dual_mul_f32 v8, v40, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, 0x7f800000, v36, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v40, 0xbfb8aa3b, v10, -v37
	v_rndne_f32_e32 v42, v37
	s_wait_loadcnt 0x3
	v_div_scale_f32 v35, null, v4, v4, v8
	v_div_scale_f32 v46, vcc_lo, v8, v4, v8
	v_add_f32_e32 v24, 1.0, v24
	v_fmac_f32_e32 v40, 0xb2a5705f, v10
	v_sub_f32_e32 v37, v37, v42
	v_rcp_f32_e32 v38, v35
	v_cvt_i32_f32_e32 v42, v42
	v_div_scale_f32 v36, null, v24, v24, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v37, v37, v40
	v_rcp_f32_e32 v39, v36
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v44, -v35, v38, 1.0
	v_exp_f32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v38, v44, v38
	v_div_scale_f32 v44, s6, v9, v24, v9
	v_fma_f32 v45, -v36, v39, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_ldexp_f32 v37, v37, v42
	v_fmac_f32_e32 v39, v45, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v45, v46, v38 :: v_dual_mul_f32 v40, v44, v39
	v_fma_f32 v47, -v35, v45, v46
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v48, -v36, v40, v44
	v_dual_fmac_f32 v45, v47, v38 :: v_dual_fmac_f32 v40, v48, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v35, -v35, v45, v46
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v1, v1, v12
	v_fma_f32 v36, -v36, v40, v44
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v44, v35, v38, v45
	s_mov_b32 vcc_lo, s6
	v_mul_f32_e32 v3, v3, v14
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v35, v36, v39, v40
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v9, v35, v24, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v36, 0, v37, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	v_mul_f32_e32 v35, 0xbfb8aa3b, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v9, v41, v9 :: v_dual_cndmask_b32 v24, 0x7f800000, v36
	v_fma_f32 v36, 0xbfb8aa3b, v11, -v35
	v_rndne_f32_e32 v49, v35
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_div_scale_f32 v45, null, v5, v5, v9
	v_div_scale_f32 v50, vcc_lo, v9, v5, v9
	v_add_f32_e32 v24, 1.0, v24
	v_dual_fmac_f32 v36, 0xb2a5705f, v11 :: v_dual_sub_f32 v35, v35, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v47, v45
	v_div_scale_f32 v46, null, v24, v24, v10
	v_div_scale_f32 v51, s6, v10, v24, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v39, v35, v36
	v_rcp_f32_e32 v48, v46
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v37, -v45, v47, 1.0
	v_exp_f32_e32 v55, v39
	global_load_b128 v[39:42], v[19:20], off offset:16
	v_fmac_f32_e32 v47, v37, v47
	v_fma_f32 v38, -v46, v48, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v52, v50, v47
	v_fmac_f32_e32 v48, v38, v48
	global_load_b128 v[35:38], v[19:20], off
	v_cvt_i32_f32_e32 v19, v49
	v_fma_f32 v54, -v45, v52, v50
	v_mul_f32_e32 v53, v51, v48
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v19, v55, v19
	v_fmac_f32_e32 v52, v54, v47
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v56, -v46, v53, v51
	v_fma_f32 v20, -v45, v52, v50
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v53, v56, v48
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v20, v20, v47, v52
	s_mov_b32 vcc_lo, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v45, -v46, v53, v51
	v_div_fixup_f32 v5, v20, v5, v9
	s_wait_loadcnt 0x3
	v_fma_f32 v9, v0, v21, v1
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v45, v45, v48, v53
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	v_fma_f32 v0, v0, v21, -v1
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v5, v5, v16
	v_fma_f32 v1, v2, v13, -v3
	v_div_fixup_f32 v10, v45, v24, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, 0, v19, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v10, v25, v10 :: v_dual_cndmask_b32 v19, 0x7f800000, v19
	v_div_fixup_f32 v4, v44, v4, v8
	v_div_scale_f32 v24, null, v6, v6, v10
	v_div_scale_f32 v49, vcc_lo, v10, v6, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v19, 1.0, v19
	v_rcp_f32_e32 v45, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v25, null, v19, v19, v11
	v_rcp_f32_e32 v46, v25
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v47, -v24, v45, 1.0
	v_fmac_f32_e32 v45, v47, v45
	v_div_scale_f32 v47, s6, v11, v19, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v48, -v25, v46, 1.0
	v_fmac_f32_e32 v46, v48, v46
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v48, v49, v45
	v_mul_f32_e32 v50, v47, v46
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v51, -v24, v48, v49
	v_fma_f32 v52, -v25, v50, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v48, v51, v45
	v_fmac_f32_e32 v50, v52, v46
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v24, -v24, v48, v49
	v_fma_f32 v25, -v25, v50, v47
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v24, v24, v45, v48
	s_mov_b32 vcc_lo, s6
	s_mov_b32 s6, 0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v25, v25, v46, v50
	v_div_fixup_f32 v6, v24, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v11, v25, v19, v11
	v_mul_f32_e32 v11, v26, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v19, null, v7, v7, v11
	v_div_scale_f32 v45, vcc_lo, v11, v7, v11
	v_rcp_f32_e32 v25, v19
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v26, -v19, v25, 1.0
	v_fmac_f32_e32 v25, v26, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v26, v45, v25
	v_fma_f32 v46, -v19, v26, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v26, v46, v25
	v_fma_f32 v12, -v19, v26, v45
	v_mbcnt_lo_u32_b32 v19, -1, 0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v12, v12, v25, v26
	v_xor_b32_e32 v20, 1, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v7, v12, v7, v11
	v_fma_f32 v11, v2, v13, v3
	v_fma_f32 v3, v4, v15, v5
	v_fma_f32 v4, v4, v15, -v5
	v_dual_mul_f32 v7, v7, v18 :: v_dual_add_f32 v2, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v8, v6, v17, v7
	v_fma_f32 v5, v6, v17, -v7
	v_dual_sub_f32 v6, v9, v11 :: v_dual_add_f32 v7, v0, v1
	v_dual_sub_f32 v0, v0, v1 :: v_dual_add_f32 v9, v3, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v1, v4, v5
	v_sub_f32_e32 v3, v3, v8
	v_dual_add_f32 v8, v7, v1 :: v_dual_sub_f32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v7, v6, v3
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v7, -v7, v7, s1
	v_cndmask_b32_e64 v1, -v1, v1, s1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v7, -v7, v7, s5
	v_cndmask_b32_e64 v1, -v1, v1, s5
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v7, v7, v10
	v_sub_f32_e32 v3, v6, v3
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v3, -v3, v3, s1
	v_cndmask_b32_e64 v7, -v7, v7, s4
	v_sub_f32_e32 v4, v4, v5
	v_dual_add_f32 v5, v2, v9 :: v_dual_sub_f32 v2, v2, v9
	v_cndmask_b32_e64 v1, -v1, v1, s4
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x1
	v_add_f32_e32 v7, v7, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v3, v14
	ds_swizzle_b32 v13, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v7 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v3, -v3, v3, s5
	v_cndmask_b32_e64 v7, -v7, v7, s2
	v_cndmask_b32_e64 v1, -v1, v1, s2
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v13
	s_wait_dscnt 0x1
	v_add_f32_e32 v7, v7, v10
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v3, v14 :: v_dual_add_f32 v6, v0, v4
	v_sub_f32_e32 v0, v0, v4
	ds_swizzle_b32 v4, v5 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v5, -v5, v5, s1
	v_cndmask_b32_e64 v8, -v8, v8, s1
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, s1
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v3, -v3, v3, s4
	v_cndmask_b32_e64 v0, -v0, v0, s1
	s_wait_dscnt 0x3
	v_dual_add_f32 v4, v5, v4 :: v_dual_add_f32 v5, v8, v9
	s_wait_dscnt 0x1
	v_dual_add_f32 v6, v6, v11 :: v_dual_add_f32 v3, v3, v14
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, -v5, v5, s5
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v14, v3 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, s5
	v_cndmask_b32_e64 v3, -v3, v3, s2
	s_wait_dscnt 0x2
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v2, -v2, v2, s1
	s_wait_dscnt 0x1
	v_dual_add_f32 v6, v6, v11 :: v_dual_add_f32 v3, v3, v14
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v5, -v5, v5, s4
	ds_swizzle_b32 v14, v1 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v1, -v1, v1, s3
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, s4
	s_wait_dscnt 0x2
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v5, v5, v9
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v4, -v4, v4, s5
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v2, -v2, v2, s5
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v5, -v5, v5, s2
	s_wait_dscnt 0x4
	v_dual_add_f32 v0, v0, v15 :: v_dual_add_f32 v1, v1, v14
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v1, 0x3d800000, v1 :: v_dual_add_f32 v4, v4, v8
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v0, -v0, v0, s5
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, s4
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v5, -v5, v5, s3
	v_add_f32_e32 v2, v2, v12
	s_mov_b32 s5, exec_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v0, v0, v15 :: v_dual_add_f32 v5, v5, v9
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v0, -v0, v0, s4
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,4)
	v_dual_add_f32 v6, v6, v11 :: v_dual_mul_f32 v5, 0x3d800000, v5
	v_cndmask_b32_e64 v2, -v2, v2, s4
	v_xor_b32_e32 v9, 16, v19
	ds_swizzle_b32 v11, v6 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, s2
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_wait_dscnt 0x2
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v0 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v0, -v0, v0, s2
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v11
	ds_swizzle_b32 v11, v7 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v7, -v7, v7, s3
	v_add_f32_e32 v4, v4, v8
	s_wait_dscnt 0x1
	v_add_f32_e32 v0, v0, v15
	ds_swizzle_b32 v15, v3 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v3, -v3, v3, s3
	s_wait_dscnt 0x1
	v_dual_add_f32 v2, v2, v12 :: v_dual_add_f32 v7, v7, v11
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, s2
	v_mul_f32_e32 v7, 0x3d800000, v7
	s_wait_dscnt 0x1
	v_add_f32_e32 v3, v3, v15
	ds_swizzle_b32 v12, v2 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v2, -v2, v2, s2
	s_wait_dscnt 0x1
	v_dual_add_f32 v4, v4, v8 :: v_dual_mul_f32 v3, 0x3d800000, v3
	ds_swizzle_b32 v8, v4 offset:swizzle(SWAP,16)
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v17, v3, v41
	ds_swizzle_b32 v16, v0 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, s3
	v_cndmask_b32_e64 v0, -v0, v0, s3
	s_wait_dscnt 0x2
	v_add_f32_e32 v2, v2, v12
	ds_swizzle_b32 v13, v2 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v2, -v2, v2, s3
	s_wait_dscnt 0x1
	v_add_f32_e32 v0, v0, v16
	v_mul_f32_e32 v16, v1, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v0, 0x3d800000, v0
	v_mul_f32_e32 v18, v0, v42
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v2, v13
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v13, v7, v37
	ds_swizzle_b32 v12, v6 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v6, -v6, v6, s3
	v_mul_f32_e32 v2, 0x3d800000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v4, v4, v8 :: v_dual_mul_f32 v15, v2, v39
	v_mul_f32_e32 v4, 0x3d800000, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mul_f32 v11, v4, v35 :: v_dual_lshlrev_b32 v10, 2, v32
	ds_bpermute_b32 v3, v10, v13
	ds_bpermute_b32 v2, v10, v15
	ds_bpermute_b32 v7, v10, v17
	ds_bpermute_b32 v8, v10, v18
	ds_bpermute_b32 v0, v10, v11
	s_wait_dscnt 0x5
	v_add_f32_e32 v6, v6, v12
	v_mul_f32_e32 v12, v5, v36
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v6, 0x3d800000, v6
	ds_bpermute_b32 v1, v10, v12
	v_mul_f32_e32 v14, v6, v38
	ds_bpermute_b32 v6, v10, v16
	ds_bpermute_b32 v5, v10, v14
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v4, v2, v0, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v2, v6, v1, s1
	v_cndmask_b32_e64 v1, v7, v3, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v19, v9, vcc_lo
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v0, v8, v5, s1
	v_max3_num_f32 v6, |v4|, |v2|, |v1|
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v5, 2, v3
	v_max_num_f32_e64 v7, |v0|, |v0|
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v3, v6, v7
	v_xor_b32_e32 v7, 8, v19
	ds_bpermute_b32 v6, v5, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v19, v7 :: v_dual_max_num_f32 v8, v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v3, v3, v8 :: v_dual_lshlrev_b32 v6, 2, v7
	v_xor_b32_e32 v8, 4, v19
	ds_bpermute_b32 v7, v6, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v19, v8, vcc_lo
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v9, v7, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v7, 2, v8
	v_max_num_f32_e32 v3, v3, v9
	v_xor_b32_e32 v9, 2, v19
	ds_bpermute_b32 v8, v7, v3
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v19, v9, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, v19, v20, vcc_lo
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v8, v8, v8
	v_max_num_f32_e32 v3, v3, v8
	ds_bpermute_b32 v8, v9, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v20, v8, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v3, v3, v20 :: v_dual_lshlrev_b32 v8, 2, v19
	ds_bpermute_b32 v19, v8, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v19, v19, v19
	v_max_num_f32_e32 v19, v3, v19
	v_mov_b32_e32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v19
	s_cbranch_execz .LBB0_232
; %bb.230:                              ; %.preheader76.i.i.7.i
	v_div_scale_f32 v3, null, 0x40e00000, 0x40e00000, v19
	v_div_scale_f32 v22, vcc_lo, v19, 0x40e00000, v19
	s_mov_b32 s8, 0x40e00000
	s_mov_b32 s9, 0xc1000000
	v_rcp_f32_e32 v20, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v21, -v3, v20, 1.0
	v_fmac_f32_e32 v20, v21, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v21, v22, v20
	v_fma_f32 v23, -v3, v21, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v23, v20
	v_fma_f32 v3, -v3, v21, v22
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v3, v3, v20, v21
	v_mov_b32_e32 v21, 0x7149f2ca
	v_div_fixup_f32 v20, v3, 0x40e00000, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v3, 1.0 :: v_dual_mul_f32 v20, 0.5, v20
.LBB0_231:                              ; %.preheader.preheader.i.i.7.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s2, s6
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, 8
	v_div_scale_f32 v22, null, s8, s8, s2
	v_div_scale_f32 v23, vcc_lo, s2, 0x40e00000, s2
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v24, v22
	v_xor_b32_e32 v22, 0x80000000, v22
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v25, v22, v24, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v25, v24
	v_mul_f32_e32 v25, v23, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v26, v22, v25, v23
	v_fmac_f32_e32 v25, v26, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v22, v25
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v22, v23, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v22, v22, 0x40e00000, s2
	v_add_f32_e32 v22, 1.0, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v22, v20, v22
	v_div_scale_f32 v23, null, v22, v22, v4
	v_div_scale_f32 v25, null, v22, v22, v2
	v_div_scale_f32 v27, null, v22, v22, v1
	v_div_scale_f32 v30, null, v22, v22, v0
	v_div_scale_f32 v24, vcc_lo, v4, v22, v4
	v_rcp_f32_e32 v33, v23
	v_rcp_f32_e32 v34, v25
	v_rcp_f32_e32 v35, v27
	v_rcp_f32_e32 v36, v30
	v_div_scale_f32 v26, s2, v2, v22, v2
	v_div_scale_f32 v28, s3, v1, v22, v1
	v_div_scale_f32 v32, s4, v0, v22, v0
	v_fma_f32 v37, -v23, v33, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v38, -v25, v34, 1.0
	v_fma_f32 v39, -v27, v35, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v40, -v30, v36, 1.0
	v_dual_fmac_f32 v33, v37, v33 :: v_dual_fmac_f32 v34, v38, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v35, v39, v35 :: v_dual_fmac_f32 v36, v40, v36
	v_dual_mul_f32 v37, v24, v33 :: v_dual_mul_f32 v38, v26, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v23, v37, v24
	v_mul_f32_e32 v40, v32, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v42, -v25, v38, v26
	v_mul_f32_e32 v39, v28, v35
	v_fmac_f32_e32 v37, v41, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v44, -v30, v40, v32
	v_fmac_f32_e32 v38, v42, v34
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v43, -v27, v39, v28
	v_fma_f32 v23, -v23, v37, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v40, v44, v36
	v_fma_f32 v24, -v25, v38, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v39, v43, v35
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v23, v23, v33, v37
	s_mov_b32 vcc_lo, s2
	v_fma_f32 v26, -v30, v40, v32
	v_fma_f32 v25, -v27, v39, v28
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v24, v24, v34, v38
	v_div_fixup_f32 v23, v23, v22, v4
	s_mov_b32 vcc_lo, s3
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v25, v25, v35, v39
	v_div_fixup_f32 v24, v24, v22, v2
	v_rndne_f32_e32 v23, v23
	s_mov_b32 vcc_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v26, v26, v36, v40
	v_div_fixup_f32 v25, v25, v22, v1
	v_rndne_f32_e32 v24, v24
	v_med3_num_f32 v23, v23, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v26, v26, v22, v0
	v_rndne_f32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v24, v24, s9, 0x40e00000
	v_fma_f32 v23, -v23, v22, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v26, v26
	v_med3_num_f32 v25, v25, s9, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v24, -v24, v22, v2
	v_fma_f32 v23, v23, v23, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v26, v26, s9, 0x40e00000
	v_fma_f32 v25, -v25, v22, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v23, v24, v24
	v_fma_f32 v24, -v26, v22, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v25, v25
	v_fmac_f32_e32 v23, v24, v24
	ds_bpermute_b32 v24, v5, v23
	s_wait_dscnt 0x0
	v_add_f32_e32 v23, v23, v24
	ds_bpermute_b32 v24, v6, v23
	s_wait_dscnt 0x0
	v_add_f32_e32 v23, v23, v24
	ds_bpermute_b32 v24, v7, v23
	s_wait_dscnt 0x0
	v_add_f32_e32 v23, v23, v24
	ds_bpermute_b32 v24, v9, v23
	s_wait_dscnt 0x0
	v_add_f32_e32 v23, v23, v24
	ds_bpermute_b32 v24, v8, v23
	s_wait_dscnt 0x0
	v_add_f32_e32 v23, v23, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v23, v21
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v21, v21, v23, vcc_lo
	v_cndmask_b32_e32 v3, v3, v22, vcc_lo
	s_cbranch_scc1 .LBB0_231
.LBB0_232:                              ; %Flow1053
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_cmp_neq_f32_e64 s2, 0, v19
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_234
; %bb.233:
	v_div_scale_f32 v20, null, v3, v3, v4
	s_mov_b32 s4, 0xc1000000
	v_rcp_f32_e32 v21, v20
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v22, -v20, v21, 1.0
	v_fmac_f32_e32 v21, v22, v21
	v_div_scale_f32 v22, vcc_lo, v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v23, v22, v21
	v_fma_f32 v24, -v20, v23, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v24, v21
	v_fma_f32 v20, -v20, v23, v22
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v20, v20, v21, v23
	v_div_fixup_f32 v4, v20, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s4, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v20, v4
.LBB0_234:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_236
; %bb.235:
	v_div_scale_f32 v4, null, v3, v3, v2
	s_mov_b32 s4, 0xc1000000
	v_rcp_f32_e32 v19, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v21, -v4, v19, 1.0
	v_fmac_f32_e32 v19, v21, v19
	v_div_scale_f32 v21, vcc_lo, v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v22, v21, v19
	v_fma_f32 v23, -v4, v22, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v22, v23, v19
	v_fma_f32 v4, -v4, v22, v21
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v19, v22
	v_div_fixup_f32 v2, v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s4, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v19, v2
.LBB0_236:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v4, 0
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_238
; %bb.237:
	v_div_scale_f32 v4, null, v3, v3, v1
	s_mov_b32 s4, 0xc1000000
	v_rcp_f32_e32 v21, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v22, -v4, v21, 1.0
	v_fmac_f32_e32 v21, v22, v21
	v_div_scale_f32 v22, vcc_lo, v1, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v23, v22, v21
	v_fma_f32 v24, -v4, v23, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v24, v21
	v_fma_f32 v4, -v4, v23, v22
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v21, v23
	v_div_fixup_f32 v1, v4, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s4, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v4, v1
.LBB0_238:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_240
; %bb.239:
	v_div_scale_f32 v1, null, v3, v3, v0
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v21, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v21, v2
	v_div_scale_f32 v21, vcc_lo, v0, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v22, v21, v2
	v_fma_f32 v23, -v1, v22, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v22, v23, v2
	v_fma_f32 v1, -v1, v22, v21
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v22
	v_div_fixup_f32 v0, v1, v3, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v0, v0
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v0, v0, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v0
.LBB0_240:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v0, v19, v20
	v_and_b32_e32 v20, 15, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v0, v0, v4, v2
	v_lshl_or_b32 v19, v19, 4, v20
	ds_bpermute_b32 v1, v5, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v6, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v0, v0, v1
	ds_bpermute_b32 v1, v7, v0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v21, v0, v1
	v_mad_co_i64_i32 v[0:1], null, 0x48, v29, s[12:13]
	ds_bpermute_b32 v22, v9, v21
	v_mad_co_u64_u32 v[0:1], null, 0x48, s10, v[0:1]
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v21, v21, v22
	v_and_b32_e32 v22, 15, v4
	ds_bpermute_b32 v4, v8, v21
	v_lshl_or_b32 v22, v2, 4, v22
	v_mad_co_u64_u32 v[1:2], null, 0x48, s11, v[1:2]
	v_and_b16 v2.h, 0xff, v19.l
	v_add_co_u32 v19, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v22.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[19:20], v2, off offset:8
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_242
; %bb.241:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v21, v4
	global_store_b64 v[0:1], v[3:4], off
.LBB0_242:                              ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit.i.7.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	ds_bpermute_b32 v2, v10, v11 offset:64
	ds_bpermute_b32 v3, v10, v12 offset:64
	s_wait_dscnt 0x2
	ds_bpermute_b32 v4, v10, v13 offset:64
	ds_bpermute_b32 v12, v10, v14 offset:64
	ds_bpermute_b32 v11, v10, v15 offset:64
	ds_bpermute_b32 v13, v10, v16 offset:64
	ds_bpermute_b32 v14, v10, v17 offset:64
	ds_bpermute_b32 v15, v10, v18 offset:64
	s_mov_b32 s5, 0
	s_mov_b32 s4, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v11, v11, v2, s1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v10, v13, v3, s1
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, v14, v4, s1
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v2, v15, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v3, |v11|, |v10|, |v4|
	v_max_num_f32_e64 v12, |v2|, |v2|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v5, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v6, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v7, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v9, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v3, v3, v12
	ds_bpermute_b32 v12, v8, v3
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_dual_max_num_f32 v12, v3, v12 :: v_dual_mov_b32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_245
; %bb.243:                              ; %.preheader76.i.1.i.7.i
	v_div_scale_f32 v3, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s6, 0x40e00000
	s_mov_b32 s8, 0xc1000000
	v_rcp_f32_e32 v13, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v3, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v3, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v3, -v3, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v3, v3, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v3, 0x40e00000, v12
	v_mov_b32_e32 v3, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_244:                              ; %.preheader.preheader.i.1.i.7.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s1, s5
	s_add_co_i32 s5, s5, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s5, 8
	v_div_scale_f32 v15, null, s6, s6, s1
	v_div_scale_f32 v16, vcc_lo, s1, 0x40e00000, s1
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v19, v15, v18, v16
	v_fmac_f32_e32 v18, v19, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s1
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v16, null, v15, v15, v11
	v_div_scale_f32 v18, null, v15, v15, v10
	v_div_scale_f32 v20, null, v15, v15, v4
	v_div_scale_f32 v22, null, v15, v15, v2
	v_div_scale_f32 v17, vcc_lo, v11, v15, v11
	v_rcp_f32_e32 v24, v16
	v_rcp_f32_e32 v25, v18
	v_rcp_f32_e32 v26, v20
	v_rcp_f32_e32 v27, v22
	v_div_scale_f32 v19, s1, v10, v15, v10
	v_div_scale_f32 v21, s2, v4, v15, v4
	v_div_scale_f32 v23, s3, v2, v15, v2
	v_fma_f32 v28, -v16, v24, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v29, -v18, v25, 1.0
	v_fma_f32 v30, -v20, v26, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v32, -v22, v27, 1.0
	v_dual_fmac_f32 v24, v28, v24 :: v_dual_fmac_f32 v25, v29, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v26, v30, v26 :: v_dual_fmac_f32 v27, v32, v27
	v_dual_mul_f32 v28, v17, v24 :: v_dual_mul_f32 v29, v19, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v32, v23, v27
	v_fma_f32 v33, -v16, v28, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v34, -v18, v29, v19
	v_mul_f32_e32 v30, v21, v26
	v_fma_f32 v36, -v22, v32, v23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v33, v24 :: v_dual_fmac_f32 v29, v34, v25
	v_fma_f32 v35, -v20, v30, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v32, v36, v27
	v_fma_f32 v16, -v16, v28, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v17, -v18, v29, v19
	v_fmac_f32_e32 v30, v35, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v19, -v22, v32, v23
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v24, v28
	s_mov_b32 vcc_lo, s1
	v_fma_f32 v18, -v20, v30, v21
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v25, v29
	s_mov_b32 vcc_lo, s2
	v_div_fixup_f32 v16, v16, v15, v11
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v26, v30
	v_div_fixup_f32 v17, v17, v15, v10
	s_mov_b32 vcc_lo, s3
	v_rndne_f32_e32 v16, v16
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v19, v19, v27, v32
	v_div_fixup_f32 v18, v18, v15, v4
	v_rndne_f32_e32 v17, v17
	v_med3_num_f32 v16, v16, s8, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v19, v19, v15, v2
	v_rndne_f32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v17, v17, s8, 0x40e00000
	v_fma_f32 v16, -v16, v15, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v19, v19
	v_med3_num_f32 v18, v18, s8, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v17, -v17, v15, v10
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v19, v19, s8, 0x40e00000
	v_fma_f32 v18, -v18, v15, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v16, v17, v17
	v_fma_f32 v17, -v19, v15, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v18, v18
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v5, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v6, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v14, v14, v16 :: v_dual_cndmask_b32 v3, v3, v15
	s_cbranch_scc1 .LBB0_244
.LBB0_245:                              ; %Flow1051
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_cmp_neq_f32_e64 s1, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_247
; %bb.246:
	v_div_scale_f32 v13, null, v3, v3, v11
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v11, v3, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v11, v13, v3, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v11, v11
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v11, v11, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v11
.LBB0_247:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_249
; %bb.248:
	v_div_scale_f32 v11, null, v3, v3, v10
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v12, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v11, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v10, v3, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v11, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v11, -v11, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v12, v15
	v_div_fixup_f32 v10, v11, v3, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v10, v10
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v10, v10, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v10
.LBB0_249:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_251
; %bb.250:
	v_div_scale_f32 v11, null, v3, v3, v4
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v14, v11
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v11, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v11, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v11, -v11, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v11, v11, v14, v16
	v_div_fixup_f32 v4, v11, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v11, v4
.LBB0_251:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_253
; %bb.252:
	v_div_scale_f32 v4, null, v3, v3, v2
	s_mov_b32 s1, 0xc1000000
	v_rcp_f32_e32 v10, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v4, v10, 1.0
	v_fmac_f32_e32 v10, v14, v10
	v_div_scale_f32 v14, vcc_lo, v2, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v10
	v_fma_f32 v16, -v4, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v10
	v_fma_f32 v4, -v4, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v10, v15
	v_div_fixup_f32 v2, v4, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s1, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v10, v2
.LBB0_253:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v2, v12, v13
	v_mad_co_u64_u32 v[0:1], null, 0x48, s26, v[0:1]
	s_delay_alu instid0(VALU_DEP_2)
	v_add3_u32 v2, v2, v11, v10
	ds_bpermute_b32 v4, v5, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v6, v2
	v_and_b32_e32 v6, 15, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v6, v12, 4, v6
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v7, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v4
	ds_bpermute_b32 v4, v9, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v2, v4
	v_and_b32_e32 v2, 15, v11
	ds_bpermute_b32 v5, v8, v4
	v_lshl_or_b32 v7, v10, 4, v2
	v_mad_co_u64_u32 v[1:2], null, 0x48, s27, v[1:2]
	v_and_b16 v2.h, 0xff, v6.l
	v_add_co_u32 v6, vcc_lo, v0, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v2.l, 8, v7.l
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v2.l, v2.h, v2.l
	global_store_b16 v[6:7], v2, off offset:8
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB0_255
; %bb.254:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v4, v4, v5
	global_store_b64 v[0:1], v[3:4], off
.LBB0_255:                              ; %Flow1054
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_256:                              ; %_ZL39gemm_up_silu_mq4g256v2_iu4_gfx1201_bodyPKcPK12block_i4_128PKfS5_S5_S5_PS1_iii.exit
	s_endpgm
.Lfunc_end0:
	.size	gemm_up_silu_mq4g256v2_iu4_gfx1201, .Lfunc_end0-gemm_up_silu_mq4g256v2_iu4_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_up_silu_mq4g256v2_iu4_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 68
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
		.amdhsa_next_free_vgpr 208
		.amdhsa_next_free_sgpr 46
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gemm_up_silu_mq4g256v2_iu4_gfx1201)<<4)&4080)>>4
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
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.num_vgpr, 208
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.num_agpr, 0
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.numbered_sgpr, 46
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.num_named_barrier, 0
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.private_seg_size, 0
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.uses_vcc, 1
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.uses_flat_scratch, 0
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.has_dyn_sized_stack, 0
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.has_recursion, 0
	.set .Lgemm_up_silu_mq4g256v2_iu4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 71184
; TotalNumSgprs: 48
; NumVgprs: 208
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 25
; NumSGPRsForWavesPerEU: 48
; NumVGPRsForWavesPerEU: 208
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
	.type	__hip_cuid_1b6ba6dae9e499e4,@object ; @__hip_cuid_1b6ba6dae9e499e4
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1b6ba6dae9e499e4
__hip_cuid_1b6ba6dae9e499e4:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1b6ba6dae9e499e4, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_1b6ba6dae9e499e4
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           gemm_up_silu_mq4g256v2_iu4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         gemm_up_silu_mq4g256v2_iu4_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     208
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
