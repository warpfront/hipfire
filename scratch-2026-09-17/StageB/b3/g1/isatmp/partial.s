attention_fp8_e4m3_fa2_gqa_partial_gfx1201: ; @attention_fp8_e4m3_fa2_gqa_partial_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x28
	s_load_b64 s[16:17], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 24
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s5, 4
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lg_u32 s6, 0x100
	s_cselect_b32 s3, -1, 0
	s_add_co_i32 s4, s17, -9
	s_or_b32 s2, s2, s3
	s_cmp_lt_u32 s4, -8
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_93
; %bb.1:
	s_and_b32 s3, ttmp7, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s3, 3
	s_cbranch_scc1 .LBB3_93
; %bb.2:
	s_lshl_b32 s4, ttmp9, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s4, s7
	s_cbranch_scc1 .LBB3_93
; %bb.3:
	s_lshr_b32 s18, ttmp7, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s18, s17
	s_cbranch_scc1 .LBB3_93
; %bb.4:
	v_cmp_gt_u32_e64 s2, 0x60, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_lshrrev_b32_e32 v157, 4, v0
	v_lshrrev_b32_e32 v158, 3, v0
	s_mov_b32 s19, 0
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB3_6
; %bb.5:
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_or_b32 v2, v0, 7, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 6, v1
	v_cmp_gt_i32_e32 vcc_lo, s7, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[3:4], null, s3, 6, v[1:2]
	v_lshrrev_b32_e32 v1, 3, v0
	v_mul_lo_u32 v4, v2, 24
	s_and_b32 s19, vcc_lo, exec_lo
	v_and_or_b32 v5, v1, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v5, v4, 8
	v_mov_b32_e32 v4, 0
.LBB3_6:
	s_or_b32 exec_lo, exec_lo, s5
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x20
	v_and_b32_e32 v9, 31, v0
	v_mov_b32_e32 v7, -1
	v_bfrev_b32_e32 v8, -2
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 8, v9
	s_cbranch_execz .LBB3_10
; %bb.7:
	v_or_b32_e32 v6, s4, v9
	v_bfrev_b32_e32 v8, -2
	v_mov_b32_e32 v7, -1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s7, v6
	s_cbranch_execz .LBB3_9
; %bb.8:
	v_ashrrev_i32_e32 v7, 31, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_wait_kmcnt 0x0
	v_add_co_u32 v6, vcc_lo, s0, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	global_load_b32 v7, v[6:7], off
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v8, v7
.LBB3_9:                                ; %Flow296
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_10:                               ; %.lr.ph.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_mbcnt_lo_u32_b32 v10, -1, 0
	v_lshrrev_b32_e32 v13, 1, v0
	v_lshrrev_b32_e32 v160, 4, v9
	v_dual_mov_b32 v154, 0xff800000 :: v_dual_and_b32 v161, 15, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_xor_b32_e32 v1, 16, v10
	v_xor_b32_e32 v12, 8, v10
	v_lshl_add_u32 v162, v9, 3, 0
	v_cndmask_b32_e64 v9, 0, v4, s19
	v_lshrrev_b32_e32 v163, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
	v_xor_b32_e32 v15, 1, v10
	s_cvt_f32_u32 s4, s17
	s_add_co_i32 s6, s17, 0x1ff
	s_mov_b32 s5, 0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, v10, v1 :: v_dual_lshlrev_b32 v14, 3, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s20, s4
	s_and_b32 s6, s6, 0xffff
	v_dual_mov_b32 v156, 1.0 :: v_dual_lshlrev_b32 v159, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s6, s6
	v_and_b32_e32 v166, 8, v13
	v_lshlrev_b32_e32 v164, 3, v160
	ds_bpermute_b32 v6, v159, v7
	ds_bpermute_b32 v11, v159, v8
	s_wait_alu depctr_sa_sdst(0)
	s_mul_f32 s20, s6, s20
	s_delay_alu instid0(SALU_CYCLE_3)
	s_trunc_f32 s20, s20
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[145:146], null, v2, 24, v[5:6]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v10, v12, vcc_lo
	v_mov_b32_e32 v1, 0
	v_max_i32_e32 v4, v7, v6
	v_xor_b32_e32 v7, 4, v10
	v_cndmask_b32_e64 v12, 0, v3, s19
	v_lshlrev_b32_e32 v0, 2, v5
	s_wait_dscnt 0x0
	v_min_i32_e32 v5, v8, v11
	v_xor_b32_e32 v11, 2, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	v_ashrrev_i32_e32 v3, 31, v2
	ds_bpermute_b32 v6, v0, v4
	ds_bpermute_b32 v0, v0, v5
	v_and_b32_e32 v13, 0xf8, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v10, v7, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	v_lshlrev_b64_e32 v[129:130], 2, v[2:3]
	v_mov_b32_e32 v3, v1
	v_dual_mov_b32 v2, v1 :: v_dual_lshlrev_b32 v165, 7, v161
	v_lshlrev_b32_e32 v7, 2, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, v10, v11, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	v_lshlrev_b32_e32 v133, 4, v161
	v_dual_mov_b32 v8, v1 :: v_dual_lshlrev_b32 v167, 4, v163
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v146, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v134, v10, v15, vcc_lo
	v_add_co_u32 v136, vcc_lo, v12, v164
	s_wait_dscnt 0x1
	v_max_i32_e32 v41, v4, v6
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v5, v0
	v_mov_b32_e32 v6, v1
	v_lshl_or_b32 v135, v163, 8, v13
	v_mov_b32_e32 v4, v1
	ds_bpermute_b32 v42, v7, v41
	ds_bpermute_b32 v43, v7, v0
	v_mov_b32_e32 v5, v1
	v_dual_mov_b32 v7, v1 :: v_dual_add_nc_u32 v168, 0, v135
	v_lshlrev_b32_e32 v65, 2, v44
	v_lshlrev_b64_e32 v[131:132], 2, v[145:146]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v137, null, 0, v9, vcc_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v146, vcc_lo, s0, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v147, null, s1, v130, vcc_lo
	v_lshlrev_b32_e32 v129, 2, v134
	s_xor_b32 s0, s20, 0x80000000
	v_add_co_u32 v135, vcc_lo, s8, v136
	s_wait_alu depctr_sa_sdst(0)
	s_fmac_f32 s6, s0, s4
	s_cvt_u32_f32 s0, s20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, s9, v137, vcc_lo
	s_wait_dscnt 0x1
	v_max_i32_e32 v138, v41, v42
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v43
	v_mov_b32_e32 v48, v8
	s_wait_alu depctr_sa_sdst(0)
	s_bitset0_b32 s6, 31
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v56, v8
	ds_bpermute_b32 v139, v65, v138
	ds_bpermute_b32 v140, v65, v0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_f32 s6, s4
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v72, v8
	s_add_co_ci_u32 s6, s0, 0
	s_lshl_b32 s4, s3, 8
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v80, v8
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v88, v8
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[4:5]
	s_add_nc_u64 s[20:21], s[12:13], s[4:5]
	s_mul_i32 s4, s7, 0x1800
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v96, v8
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v104, v8
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v138, v139
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v140
	v_add_co_u32 v148, vcc_lo, v135, 48
	s_and_b32 s22, s6, 0xffff
	ds_bpermute_b32 v134, v129, v130
	ds_bpermute_b32 v129, v129, v0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[8:9], s[4:5]
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v112, v8
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v120, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v149, null, 0, v136, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v152, vcc_lo, s6, v131
	v_dual_mov_b32 v113, v1 :: v_dual_mov_b32 v128, v8
	v_add_co_u32 v150, s0, s0, v133
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	s_wait_dscnt 0x1
	v_max_i32_e32 v130, v130, v134
	s_wait_dscnt 0x0
	v_min_i32_e32 v0, v0, v129
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_mov_b32_e32 v121, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, s7, v132, vcc_lo
	v_readfirstlane_b32 s26, v130
	v_mov_b32_e32 v136, v8
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v13, v5
	v_dual_mov_b32 v12, v4 :: v_dual_mov_b32 v11, v3
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v9, v1
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v30, v6 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v28, v4 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v26, v2 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v71, v7 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v79, v7 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v87, v7 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v95, v7 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v103, v7 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v114, v2
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v122, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v151, null, s1, 0, s0
	v_readfirstlane_b32 s27, v0
	v_dual_mov_b32 v135, v7 :: v_dual_mov_b32 v132, v4
	v_dual_mov_b32 v134, v6 :: v_dual_mov_b32 v133, v5
	v_dual_mov_b32 v130, v2 :: v_dual_mov_b32 v131, v3
	v_mov_b32_e32 v129, v1
	s_mul_i32 s24, s18, s22
	s_lshl_b32 s4, s3, 1
	s_add_co_i32 s25, s24, s22
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[10:11], s[4:5]
	s_add_nc_u64 s[12:13], s[12:13], s[4:5]
	s_lshl_b32 s11, s24, 6
	s_branch .LBB3_14
.LBB3_11:                               ; %Flow290
                                        ;   in Loop: Header=BB3_14 Depth=1
	v_mov_b32_e32 v154, v2
.LBB3_12:                               ; %Flow291
                                        ;   in Loop: Header=BB3_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB3_13:                               ;   in Loop: Header=BB3_14 Depth=1
	s_add_co_i32 s24, s24, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s24, s25
	s_cselect_b32 s0, -1, 0
	s_xor_b32 s1, s28, -1
	s_add_co_i32 s11, s11, 64
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_88
.LBB3_14:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_17 Depth 2
                                        ;     Child Loop BB3_23 Depth 2
                                        ;     Child Loop BB3_45 Depth 2
                                        ;       Child Loop BB3_50 Depth 3
	s_lshl_b32 s29, s24, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s29, s26
	s_cselect_b32 s28, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_13
; %bb.15:                               ; %.preheader447.i.preheader
                                        ;   in Loop: Header=BB3_14 Depth=1
	v_mov_b32_e32 v0, v157
	v_mov_b32_e32 v6, v158
	s_mov_b32 s0, 8
	s_branch .LBB3_17
.LBB3_16:                               ;   in Loop: Header=BB3_17 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_and_b32_e32 v7, 0xe0, v6
	v_add_nc_u32_e32 v6, 16, v6
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	v_add_nc_u32_e32 v7, v7, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_or_b32 v7, v0, 15, v7
	v_add_nc_u32_e32 v0, 8, v0
	v_lshl_add_u32 v7, v7, 3, 0
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v7, v[2:3], v[4:5] offset1:16
	s_cbranch_scc1 .LBB3_19
.LBB3_17:                               ; %.preheader447.i
                                        ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v7, s11, v0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	s_mov_b32 s1, exec_lo
	v_cmpx_ge_i32_e64 s26, v7
	s_cbranch_execz .LBB3_16
; %bb.18:                               ;   in Loop: Header=BB3_17 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v7, v[150:151]
	global_load_b128 v[2:5], v[2:3], off
	s_branch .LBB3_16
.LBB3_19:                               ; %.preheader446.i
                                        ;   in Loop: Header=BB3_14 Depth=1
	v_or_b32_e32 v6, s29, v166
	v_dual_mov_b32 v7, v167 :: v_dual_mov_b32 v8, v163
	s_movk_i32 s0, 0xc000
	s_branch .LBB3_23
.LBB3_20:                               ; %Flow292
                                        ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
.LBB3_21:                               ; %Flow293
                                        ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
.LBB3_22:                               ; %.loopexit444.i
                                        ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v0, s0, v168
	v_add_nc_u32_e32 v8, 4, v8
	v_add_nc_u32_e32 v7, 64, v7
	s_addk_co_i32 s0, 0x400
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 0
	ds_store_b64 v0, v[2:3] offset:32768
	s_cbranch_scc1 .LBB3_39
.LBB3_23:                               ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0x70, v8
	v_add_nc_u32_e32 v137, v0, v6
	v_and_or_b32 v0, 0xf0, v7, v161
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v2, 7, v137
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
                                        ; implicit-def: $vgpr2_vgpr3
	s_and_saveexec_b32 s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB3_25
; %bb.24:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, s[20:21]
	v_mov_b16_e32 v5.h, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v139.l, v5.h
	v_add_co_u32 v2, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_u8 v0, v[2:3], off offset:1032
	global_load_u8 v4, v[2:3], off offset:2064
	global_load_d16_u8 v5, v[2:3], off
	global_load_u8 v137, v[2:3], off offset:3096
	global_load_u8 v138, v[2:3], off offset:5160
	global_load_u8 v140, v[2:3], off offset:4128
	global_load_u8 v141, v[2:3], off offset:7224
	global_load_d16_hi_u8 v139, v[2:3], off offset:6192
	s_wait_loadcnt 0x7
	v_lshlrev_b32_e32 v0, 8, v0
	s_wait_loadcnt 0x6
	v_lshlrev_b32_e32 v2, 16, v4
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v3, 24, v137
	s_wait_loadcnt 0x3
	v_lshlrev_b32_e32 v4, 8, v138
	v_or_b32_e32 v0, v0, v5
                                        ; implicit-def: $vgpr137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or3_b32 v0, v0, v2, v3
	s_wait_loadcnt 0x2
	v_or3_b32 v2, 0, v140, v4
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v3, 24, v141
	v_or3_b32 v0, v0, 0, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v3, v2, v139, v3
	v_or3_b32 v2, v0, 0, 0
                                        ; implicit-def: $vgpr0
.LBB3_25:                               ; %Flow294
                                        ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB3_22
; %bb.26:                               ;   in Loop: Header=BB3_23 Depth=2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_ge_i32_e64 s26, v137
	s_cbranch_execz .LBB3_21
; %bb.27:                               ; %.preheader443.i
                                        ;   in Loop: Header=BB3_23 Depth=2
	v_add_co_u32 v4, s4, s20, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s21, 0, s4
	v_mov_b16_e32 v0.h, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, 0x408, v137, v[4:5]
	global_load_d16_u8 v0, v[2:3], off
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
	v_cmpx_gt_i32_e64 s26, v137
	s_cbranch_execz .LBB3_29
; %bb.28:                               ;   in Loop: Header=BB3_23 Depth=2
	v_or_b32_e32 v2, 1, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, v[4:5]
	global_load_u8 v2, v[2:3], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v0, v2, 8, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v2, v0
.LBB3_29:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 2, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_31
; %bb.30:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 16, v2
.LBB3_31:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 3, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_33
; %bb.32:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshl_or_b32 v2, v0, 24, v2
.LBB3_33:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 4, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_35
; %bb.34:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
.LBB3_35:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 5, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_37
; %bb.36:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[138:139], null, 0x408, v0, v[4:5]
	global_load_u8 v0, v[138:139], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 8, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v3, v0, v3
.LBB3_37:                               ;   in Loop: Header=BB3_23 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v0, 6, v137
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ge_i32_e64 s26, v0
	s_cbranch_execz .LBB3_20
; %bb.38:                               ;   in Loop: Header=BB3_23 Depth=2
	v_mad_co_i64_i32 v[4:5], null, 0x408, v0, v[4:5]
	v_mov_b16_e32 v0.l, 0
	global_load_d16_hi_u8 v0, v[4:5], off
	s_wait_loadcnt 0x0
	v_or_b32_e32 v3, v0, v3
	s_branch .LBB3_20
.LBB3_39:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB3_12
; %bb.40:                               ;   in Loop: Header=BB3_14 Depth=1
	v_mov_b32_e32 v0, s16
	s_and_saveexec_b32 s0, s19
	s_cbranch_execz .LBB3_42
; %bb.41:                               ;   in Loop: Header=BB3_14 Depth=1
	global_load_b32 v0, v[152:153], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v0, s16, v0
.LBB3_42:                               ;   in Loop: Header=BB3_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_or_b32 s0, s29, 16
	v_mov_b32_e32 v4, v162
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s26
	s_mov_b32 s31, 0
	s_cselect_b32 s33, -1, 0
	s_branch .LBB3_45
.LBB3_43:                               ; %.loopexit438.loopexit.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_sub_f32_e32 v143, v154, v2
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v154
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	v_div_scale_f32 v172, null, v3, v3, v140
	v_div_scale_f32 v174, null, v3, v3, v139
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	v_div_scale_f32 v177, null, v3, v3, v137
	v_div_scale_f32 v176, null, v3, v3, v8
	v_rcp_f32_e32 v173, v172
	v_lshl_add_u32 v189, s31, 12, v162
	v_exp_f32_e32 v143, v143
	v_rcp_f32_e32 v180, v177
	v_rcp_f32_e32 v179, v176
	v_fma_f32 v175, -v172, v173, 1.0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v143, 0, v143, vcc_lo
	v_dual_fmac_f32 v173, v175, v173 :: v_dual_mul_f32 v144, v156, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v154, null, v3, v3, v144
	v_div_scale_f32 v170, vcc_lo, v144, v3, v144
	v_rcp_f32_e32 v156, v154
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v169, -v154, v156, 1.0
	v_fmac_f32_e32 v156, v169, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v169, v170, v156
	v_fma_f32 v171, -v154, v169, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v169, v171, v156
	v_fma_f32 v154, -v154, v169, v170
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v154, v154, v156, v169
	v_div_fixup_f32 v6, v154, v3, v144
	v_div_scale_f32 v154, null, v3, v3, v141
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v134, v134, v6
	v_dual_mul_f32 v130, v130, v6 :: v_dual_fmac_f32 v5, v155, v143
	v_mul_f32_e32 v128, v128, v6
	v_div_scale_f32 v143, null, v3, v3, v142
	v_rcp_f32_e32 v156, v154
	v_dual_mul_f32 v136, v136, v6 :: v_dual_mul_f32 v135, v135, v6
	v_mul_f32_e32 v126, v126, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v144, v143
	v_dual_mul_f32 v133, v133, v6 :: v_dual_mul_f32 v124, v124, v6
	v_dual_mul_f32 v132, v132, v6 :: v_dual_mul_f32 v131, v131, v6
	v_mul_f32_e32 v122, v122, v6
	v_fma_f32 v170, -v154, v156, 1.0
	v_mul_f32_e32 v25, v25, v6
	v_dual_mul_f32 v129, v129, v6 :: v_dual_mul_f32 v120, v120, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v155, -v143, v144, 1.0
	v_dual_mul_f32 v33, v33, v6 :: v_dual_fmac_f32 v156, v170, v156
	v_div_scale_f32 v170, s0, v141, v3, v141
	v_dual_mul_f32 v127, v127, v6 :: v_dual_mul_f32 v118, v118, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v144, v155, v144
	v_div_scale_f32 v155, vcc_lo, v142, v3, v142
	v_dual_mul_f32 v16, v16, v6 :: v_dual_mul_f32 v125, v125, v6
	v_dual_mul_f32 v116, v116, v6 :: v_dual_mul_f32 v123, v123, v6
	v_dual_mul_f32 v114, v114, v6 :: v_dual_mul_f32 v169, v155, v144
	v_dual_mul_f32 v121, v121, v6 :: v_dual_mul_f32 v112, v112, v6
	v_dual_mul_f32 v119, v119, v6 :: v_dual_mul_f32 v110, v110, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v171, -v143, v169, v155
	v_dual_mul_f32 v117, v117, v6 :: v_dual_mul_f32 v108, v108, v6
	v_dual_mul_f32 v115, v115, v6 :: v_dual_mul_f32 v106, v106, v6
	v_fmac_f32_e32 v169, v171, v144
	v_mul_f32_e32 v171, v170, v156
	v_dual_mul_f32 v113, v113, v6 :: v_dual_mul_f32 v104, v104, v6
	v_dual_mul_f32 v111, v111, v6 :: v_dual_mul_f32 v102, v102, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v143, -v143, v169, v155
	v_fma_f32 v155, -v154, v171, v170
	v_dual_mul_f32 v109, v109, v6 :: v_dual_mul_f32 v100, v100, v6
	v_dual_mul_f32 v107, v107, v6 :: v_dual_mul_f32 v98, v98, v6
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v143, v143, v144, v169
	v_fmac_f32_e32 v171, v155, v156
	v_div_scale_f32 v155, s1, v140, v3, v140
	s_mov_b32 vcc_lo, s0
	v_rcp_f32_e32 v144, v174
	v_div_fixup_f32 v143, v143, v3, v142
	v_fma_f32 v142, -v154, v171, v170
	v_mul_f32_e32 v154, v155, v173
	v_div_scale_f32 v170, null, v3, v3, v7
	v_dual_mul_f32 v105, v105, v6 :: v_dual_mul_f32 v96, v96, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v156, v142, v156, v171
	v_fma_f32 v171, -v172, v154, v155
	s_mov_b32 vcc_lo, s1
	v_rcp_f32_e32 v175, v170
	v_fma_f32 v169, -v174, v144, 1.0
	v_mul_f32_e32 v19, v19, v6
	v_fmac_f32_e32 v154, v171, v173
	v_mov_b16_e64 v142.l, v1.l
	v_div_fixup_f32 v156, v156, v3, v141
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, s0, v139, v3, v139
	v_fma_f32 v155, -v172, v154, v155
	v_fma_f32 v178, -v170, v175, 1.0
	v_mov_b16_e64 v141.l, v142.l
	v_mov_b16_e64 v142.h, 0
	v_dual_mul_f32 v103, v103, v6 :: v_dual_mul_f32 v94, v94, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v155, v173, v154
	v_div_scale_f32 v155, null, v3, v3, v138
	v_fma_f32 v173, -v176, v179, 1.0
	s_mov_b32 vcc_lo, s0
	v_cvt_pk_fp8_f32 v141.l, v143, v156
	v_div_fixup_f32 v140, v154, v3, v140
	v_fma_f32 v154, -v177, v180, 1.0
	v_mul_f32_e32 v15, v15, v6
	v_dual_fmac_f32 v179, v173, v179 :: v_dual_mov_b32 v156, v3
	v_div_scale_f32 v173, s3, v8, v3, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v180, v154, v180
	v_div_scale_f32 v154, s4, v137, v3, v137
	v_mul_f32_e32 v10, v10, v6
	v_mov_b16_e64 v141.h, v142.h
	v_dual_mul_f32 v101, v101, v6 :: v_dual_mul_f32 v92, v92, v6
	v_mul_f32_e32 v182, v154, v180
	v_fmac_f32_e32 v175, v178, v175
	v_mul_f32_e32 v171, v169, v144
	v_rcp_f32_e32 v178, v155
	v_dual_mul_f32 v99, v99, v6 :: v_dual_mul_f32 v90, v90, v6
	v_dual_mul_f32 v97, v97, v6 :: v_dual_mul_f32 v88, v88, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f32 v172, -v174, v171, v169
	v_dual_mul_f32 v95, v95, v6 :: v_dual_mul_f32 v86, v86, v6
	v_dual_mul_f32 v93, v93, v6 :: v_dual_mul_f32 v84, v84, v6
	v_fmac_f32_e32 v171, v172, v144
	v_div_scale_f32 v172, s1, v7, v3, v7
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v181, -v155, v178, 1.0
	v_mul_f32_e32 v17, v17, v6
	v_dual_mul_f32 v91, v91, v6 :: v_dual_mul_f32 v82, v82, v6
	v_fma_f32 v169, -v174, v171, v169
	v_mul_f32_e32 v174, v172, v175
	v_fmac_f32_e32 v178, v181, v178
	v_div_scale_f32 v181, s0, v138, v3, v138
	v_dual_mul_f32 v89, v89, v6 :: v_dual_mul_f32 v80, v80, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v169, v144, v171
	v_mul_f32_e32 v171, v173, v179
	v_fma_f32 v169, -v170, v174, v172
	v_mul_f32_e32 v183, v181, v178
	s_mov_b32 vcc_lo, s1
	v_div_fixup_f32 v139, v144, v3, v139
	v_fma_f32 v144, -v176, v171, v173
	v_fmac_f32_e32 v174, v169, v175
	v_fma_f32 v169, -v177, v182, v154
	v_dual_mul_f32 v87, v87, v6 :: v_dual_mul_f32 v78, v78, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v171, v144, v179
	v_fma_f32 v144, -v155, v183, v181
	v_fma_f32 v170, -v170, v174, v172
	v_fmac_f32_e32 v182, v169, v180
	v_cvt_pk_fp8_f32 v141.h, v140, v139
	v_dual_mul_f32 v85, v85, v6 :: v_dual_mul_f32 v76, v76, v6
	v_fmac_f32_e32 v183, v144, v178
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v169, v170, v175, v174
	v_fma_f32 v170, -v176, v171, v173
	v_fma_f32 v144, -v177, v182, v154
	s_mov_b32 vcc_lo, s3
	v_fma_f32 v155, -v155, v183, v181
	v_div_fixup_f32 v7, v169, v3, v7
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v154, v170, v179, v171
	s_mov_b32 vcc_lo, s4
	v_dual_mul_f32 v83, v83, v6 :: v_dual_mul_f32 v74, v74, v6
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v144, v144, v180, v182
	s_mov_b32 vcc_lo, s0
	v_div_fixup_f32 v8, v154, v3, v8
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v155, v155, v178, v183
	v_dual_mul_f32 v81, v81, v6 :: v_dual_mul_f32 v72, v72, v6
	v_div_fixup_f32 v137, v144, v3, v137
	v_cvt_pk_fp8_f32 v142.l, v7, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_div_fixup_f32 v138, v155, v3, v138
	ds_load_b64 v[7:8], v189 offset:16384
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v79, v79, v6 :: v_dual_mul_f32 v70, v70, v6
	v_cvt_pk_fp8_f32 v142.h, v137, v138
	ds_load_b64 v[137:138], v189 offset:16640
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[139:140], v189 offset:16896
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[143:144], v189 offset:17152
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[154:155], v189 offset:17408
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[169:170], v189 offset:17664
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[171:172], v189 offset:17920
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[173:174], v189 offset:18176
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[175:176], v189 offset:18432
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[177:178], v189 offset:18688
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[179:180], v189 offset:18944
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[181:182], v189 offset:19200
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[183:184], v189 offset:19456
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[185:186], v189 offset:19712
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[187:188], v189 offset:19968
	;;#ASMSTART
	;;#ASMEND
	ds_load_b64 v[189:190], v189 offset:20224
	v_dual_mul_f32 v77, v77, v6 :: v_dual_mul_f32 v68, v68, v6
	v_dual_mul_f32 v75, v75, v6 :: v_dual_mul_f32 v66, v66, v6
	v_dual_mul_f32 v73, v73, v6 :: v_dual_mul_f32 v64, v64, v6
	v_dual_mul_f32 v71, v71, v6 :: v_dual_mul_f32 v62, v62, v6
	v_dual_mul_f32 v69, v69, v6 :: v_dual_mul_f32 v60, v60, v6
	v_dual_mul_f32 v67, v67, v6 :: v_dual_mul_f32 v58, v58, v6
	v_dual_mul_f32 v65, v65, v6 :: v_dual_mul_f32 v56, v56, v6
	v_dual_mul_f32 v63, v63, v6 :: v_dual_mul_f32 v54, v54, v6
	v_dual_mul_f32 v61, v61, v6 :: v_dual_mul_f32 v52, v52, v6
	v_dual_mul_f32 v59, v59, v6 :: v_dual_mul_f32 v50, v50, v6
	v_dual_mul_f32 v57, v57, v6 :: v_dual_mul_f32 v48, v48, v6
	v_dual_mul_f32 v55, v55, v6 :: v_dual_mul_f32 v46, v46, v6
	v_dual_mul_f32 v53, v53, v6 :: v_dual_mul_f32 v44, v44, v6
	v_dual_mul_f32 v51, v51, v6 :: v_dual_mul_f32 v42, v42, v6
	v_dual_mul_f32 v49, v49, v6 :: v_dual_mul_f32 v40, v40, v6
	v_dual_mul_f32 v47, v47, v6 :: v_dual_mul_f32 v38, v38, v6
	v_dual_mul_f32 v45, v45, v6 :: v_dual_mul_f32 v36, v36, v6
	v_dual_mul_f32 v43, v43, v6 :: v_dual_mul_f32 v34, v34, v6
	v_dual_mul_f32 v41, v41, v6 :: v_dual_mul_f32 v32, v32, v6
	v_dual_mul_f32 v39, v39, v6 :: v_dual_mul_f32 v30, v30, v6
	v_dual_mul_f32 v37, v37, v6 :: v_dual_mul_f32 v28, v28, v6
	v_dual_mul_f32 v35, v35, v6 :: v_dual_mul_f32 v26, v26, v6
	v_dual_mul_f32 v31, v31, v6 :: v_dual_mul_f32 v24, v24, v6
	v_dual_mul_f32 v29, v29, v6 :: v_dual_mul_f32 v22, v22, v6
	v_dual_mul_f32 v27, v27, v6 :: v_dual_mul_f32 v20, v20, v6
	v_dual_mul_f32 v23, v23, v6 :: v_dual_mul_f32 v18, v18, v6
	v_dual_mul_f32 v21, v21, v6 :: v_dual_mul_f32 v14, v14, v6
	v_dual_mul_f32 v13, v13, v6 :: v_dual_mul_f32 v12, v12, v6
	v_mul_f32_e32 v11, v11, v6
	v_mul_f32_e32 v9, v9, v6
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[7:8], v[141:142], v[129:136]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[137:138], v[141:142], v[121:128]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[139:140], v[141:142], v[113:120]
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[143:144], v[141:142], v[105:112]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[154:155], v[141:142], v[97:104]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[169:170], v[141:142], v[89:96]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[171:172], v[141:142], v[81:88]
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[173:174], v[141:142], v[73:80]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[175:176], v[141:142], v[65:72]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[177:178], v[141:142], v[57:64]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[179:180], v[141:142], v[49:56]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[181:182], v[141:142], v[41:48]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[183:184], v[141:142], v[33:40]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[185:186], v[141:142], v[25:32]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[187:188], v[141:142], v[17:24]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[189:190], v[141:142], v[9:16]
	v_mov_b32_e32 v155, v5
	;;#ASMSTART
	;;#ASMEND
.LBB3_44:                               ; %.loopexit438.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	v_add_nc_u32_e32 v4, 0x100, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v154, v2
	s_add_co_i32 s31, s31, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s31, 4
	s_cbranch_scc0 .LBB3_11
.LBB3_45:                               ; %NodeBlock
                                        ;   Parent Loop BB3_14 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB3_50 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s31, 1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB3_48
; %bb.46:                               ; %LeafBlock
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_cmp_eq_u32 s31, 1
	s_mov_b32 s0, s33
	s_cbranch_scc1 .LBB3_48
; %bb.47:                               ;   in Loop: Header=BB3_45 Depth=2
	s_cmp_eq_u32 s31, 2
	s_cselect_b32 s0, 32, 48
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, s29
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s26, s0
	s_cselect_b32 s0, -1, 0
.LBB3_48:                               ; %Flow288
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_85
; %bb.49:                               ; %.preheader442.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v2, v148
	v_mov_b32_e32 v3, v149
	s_movk_i32 s0, 0xc000
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
	v_mov_b32_e32 v144, v137
.LBB3_50:                               ; %.preheader.i
                                        ;   Parent Loop BB3_14 Depth=1
                                        ;     Parent Loop BB3_45 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_clause 0x3
	global_load_b64 v[173:174], v[2:3], off offset:-48
	global_load_b64 v[175:176], v[2:3], off offset:-32
	global_load_b64 v[177:178], v[2:3], off offset:-16
	global_load_b64 v[179:180], v[2:3], off
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v169, s0, v4
	v_add_co_u32 v2, vcc_lo, v2, 64
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	ds_load_2addr_stride64_b64 v[5:8], v169 offset0:32 offset1:34
	ds_load_2addr_stride64_b64 v[169:172], v169 offset0:36 offset1:38
	s_addk_co_i32 s0, 0x1000
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	s_wait_loadcnt_dscnt 0x301
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[5:6], v[173:174], v[137:144]
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[7:8], v[175:176], v[137:144]
	s_wait_loadcnt_dscnt 0x100
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[169:170], v[177:178], v[137:144]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[171:172], v[179:180], v[137:144]
	s_cbranch_scc1 .LBB3_50
; %bb.51:                               ;   in Loop: Header=BB3_45 Depth=2
	s_lshl4_add_u32 s0, s31, s29
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v170, 0
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 15
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s27
	s_cselect_b32 s34, -1, 0
	s_cmp_gt_i32 s1, s27
	s_cselect_b32 s1, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s3, s1, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s3
	s_cbranch_execz .LBB3_53
; %bb.52:                               ;   in Loop: Header=BB3_45 Depth=2
	global_load_b32 v170, v[146:147], off
.LBB3_53:                               ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v2, s0, v164
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e32 vcc_lo, s26, v2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_55
; %bb.54:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[5:6], null, 0x408, v2, s[22:23]
	global_load_d16_b16 v3, v[5:6], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v173, v3.l
.LBB3_55:                               ; %_ZL11fa2_scale_nPKhiii.exit.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v5, 1, v2
	v_cmp_gt_i32_e64 s0, s26, v2
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_57
; %bb.56:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[6:7], null, 0x408, v5, s[22:23]
	global_load_d16_b16 v3, v[6:7], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v175, v3.l
.LBB3_57:                               ; %_ZL11fa2_scale_nPKhiii.exit.1.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, 2, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s1, s26, v6
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB3_59
; %bb.58:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[7:8], null, 0x408, v6, s[22:23]
	global_load_d16_b16 v3, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v3.l
.LBB3_59:                               ; %_ZL11fa2_scale_nPKhiii.exit.2.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 3, v2
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v178, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s3, s26, v7
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_61
; %bb.60:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v7, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v178, v3.l
.LBB3_61:                               ; %_ZL11fa2_scale_nPKhiii.exit.3.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v8, 4, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s4, s26, v8
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_63
; %bb.62:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v8, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v176, v3.l
.LBB3_63:                               ; %_ZL11fa2_scale_nPKhiii.exit.4.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v169, 5, v2
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_ge_i32_e64 s5, s26, v169
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_65
; %bb.64:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v169, s[22:23]
	global_load_d16_b16 v3, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v179, v3.l
.LBB3_65:                               ; %_ZL11fa2_scale_nPKhiii.exit.5.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v171, 6, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_i32_e64 s6, s26, v171
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB3_67
; %bb.66:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[180:181], null, 0x408, v171, s[22:23]
	global_load_d16_b16 v3, v[180:181], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v177, v3.l
.LBB3_67:                               ; %_ZL11fa2_scale_nPKhiii.exit.6.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v172, 7, v2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v180, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_ge_i32_e64 s7, s26, v172
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_69
; %bb.68:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[180:181], null, 0x408, v172, s[22:23]
	global_load_d16_b16 v180, v[180:181], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v180, v180.l
.LBB3_69:                               ; %_ZL11fa2_scale_nPKhiii.exit.7.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mul_f32 v137, v0, v137 :: v_dual_mul_f32 v138, v0, v138
	s_wait_loadcnt 0x0
	v_cmp_le_i32_e64 s8, v2, v170
	v_cmp_lt_i32_e64 s9, v2, v170
	v_dual_mul_f32 v139, v0, v139 :: v_dual_mul_f32 v140, v0, v140
	v_cmp_le_i32_e64 s10, v6, v170
	v_dual_mul_f32 v137, v137, v173 :: v_dual_mul_f32 v138, v138, v175
	s_or_b32 s8, s34, s8
	s_or_b32 s9, s34, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	v_dual_mul_f32 v173, v139, v174 :: v_dual_mul_f32 v144, v0, v144
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v137, 0xff800000, v137, s8
	s_or_b32 s10, s34, s10
	v_cmp_le_i32_e64 s8, v7, v170
	s_and_b32 s9, s19, s9
	v_mul_f32_e32 v142, v0, v142
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v139, 0xff800000, v138, s9
	s_and_b32 s9, s19, s10
	v_mul_f32_e32 v140, v140, v178
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v138, 0xff800000, v173, s9
	v_mul_f32_e32 v173, v0, v141
	s_or_b32 s10, s34, s8
	v_cmp_le_i32_e64 s8, v8, v170
	v_cmp_le_i32_e64 s9, v169, v170
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s10, s19, s10
	v_mul_f32_e32 v143, v0, v143
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v141, 0xff800000, v140, s10
	v_mul_f32_e32 v140, v173, v176
	s_or_b32 s8, s34, s8
	v_mul_f32_e32 v173, v142, v179
	s_or_b32 s9, s34, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	v_dual_mul_f32 v143, v143, v177 :: v_dual_mul_f32 v144, v144, v180
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v142, 0xff800000, v140, s8
	s_and_b32 s8, s19, s9
	v_cmp_le_i32_e64 s9, v172, v170
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v140, 0xff800000, v173, s8
	v_cmp_le_i32_e64 s8, v171, v170
	v_max3_num_f32 v170, v137, 0xff800000, v139
	s_or_b32 s9, s34, s9
	s_or_b32 s8, s34, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v170, v170, v138, v141
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s8, s19, s8
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v173, 0xff800000, v143, s8
	s_and_b32 s8, s19, s9
	v_max3_num_f32 v143, v170, v142, v140
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v144, 0xff800000, v144, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v175, v143, v173, v144
	ds_bpermute_b32 v176, v159, v175
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB3_71
; %bb.70:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[2:3], null, 0x408, v2, s[12:13]
	global_load_d16_b16 v2, v[2:3], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v3, v2.l
.LBB3_71:                               ; %_ZL11fa2_scale_nPKhiii.exit412.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v170, 0
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB3_73
; %bb.72:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[177:178], null, 0x408, v5, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v170, v2.l
.LBB3_73:                               ; %_ZL11fa2_scale_nPKhiii.exit412.1.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_75
; %bb.74:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[5:6], null, 0x408, v6, s[12:13]
	global_load_d16_b16 v2, v[5:6], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v143, v2.l
.LBB3_75:                               ; %_ZL11fa2_scale_nPKhiii.exit412.2.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_and_saveexec_b32 s0, s3
	s_cbranch_execz .LBB3_77
; %bb.76:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[6:7], null, 0x408, v7, s[12:13]
	global_load_d16_b16 v2, v[6:7], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v6, v2.l
.LBB3_77:                               ; %_ZL11fa2_scale_nPKhiii.exit412.3.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s4
	s_cbranch_execz .LBB3_79
; %bb.78:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[7:8], null, 0x408, v8, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v5, v2.l
.LBB3_79:                               ; %_ZL11fa2_scale_nPKhiii.exit412.4.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v8, 0
	s_and_saveexec_b32 s0, s5
	s_cbranch_execnz .LBB3_86
; %bb.80:                               ; %_ZL11fa2_scale_nPKhiii.exit412.5.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execnz .LBB3_87
.LBB3_81:                               ; %_ZL11fa2_scale_nPKhiii.exit412.6.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB3_83
.LBB3_82:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[171:172], null, 0x408, v172, s[12:13]
	global_load_d16_b16 v2, v[171:172], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v169, v2.l
.LBB3_83:                               ; %_ZL11fa2_scale_nPKhiii.exit412.7.i
                                        ;   in Loop: Header=BB3_45 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_dscnt 0x0
	v_max3_num_f32 v2, v154, v175, v176
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_sub_f32 v7, v137, v2 :: v_dual_sub_f32 v138, v138, v2
	v_dual_sub_f32 v137, v139, v2 :: v_dual_sub_f32 v140, v140, v2
	v_dual_sub_f32 v139, v141, v2 :: v_dual_sub_f32 v144, v144, v2
	v_dual_mul_f32 v7, 0x3fb8aa3b, v7 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	v_dual_sub_f32 v141, v142, v2 :: v_dual_sub_f32 v142, v173, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v140, 0x3fb8aa3b, v140
	v_exp_f32_e32 v7, v7
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v2
	v_exp_f32_e32 v138, v138
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_mul_f32 v144, 0x3fb8aa3b, v144
	v_exp_f32_e32 v137, v137
	v_exp_f32_e32 v140, v140
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v139, v139
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, v7, 0, vcc_lo
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v171, 0x3fb8aa3b, v142
	v_exp_f32_e32 v144, v144
	v_mul_f32_e32 v142, v7, v3
	v_exp_f32_e32 v172, v141
	v_cndmask_b32_e64 v137, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
	v_add_f32_e32 v3, v7, v137
	v_mul_f32_e32 v141, v137, v170
	v_cndmask_b32_e64 v170, v139, 0, vcc_lo
	v_exp_f32_e32 v137, v171
	v_cndmask_b32_e64 v171, v172, 0, vcc_lo
	v_cndmask_b32_e64 v172, v140, 0, vcc_lo
	v_mul_f32_e32 v140, v138, v143
	v_cndmask_b32_e64 v143, v144, 0, vcc_lo
	v_mul_f32_e32 v139, v170, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v7, v171, v5 :: v_dual_mul_f32 v8, v172, v8
	v_dual_add_f32 v3, v138, v3 :: v_dual_mul_f32 v138, v143, v169
	v_max3_num_f32 v6, v142, 0, v141
	v_cndmask_b32_e64 v5, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v3, v170, v3
	v_max3_num_f32 v6, v6, v140, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v137, v5, v174
	v_add_f32_e32 v3, v171, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v6, v6, v7, v8
	v_add_f32_e32 v3, v172, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v144, v6, v137, v138
	v_add_f32_e32 v3, v5, v3
	ds_bpermute_b32 v169, v159, v144
	v_add_f32_e32 v5, v143, v3
	ds_bpermute_b32 v6, v159, v5
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v3, v169, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v143, v144, v3
	v_mov_b32_e32 v3, v156
	v_cmpx_lt_f32_e32 0, v143
	s_cbranch_execz .LBB3_43
; %bb.84:                               ;   in Loop: Header=BB3_45 Depth=2
	v_div_scale_f32 v3, null, 0x43e00000, 0x43e00000, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v144, v3
	v_fma_f32 v169, -v3, v144, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v144, v169, v144
	v_div_scale_f32 v169, vcc_lo, v143, 0x43e00000, v143
	v_mul_f32_e32 v170, v169, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v171, -v3, v170, v169
	v_fmac_f32_e32 v170, v171, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v170, v169
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v3, v3, v144, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v3, v3, 0x43e00000, v143
	v_max_num_f32_e32 v3, 0x1f800000, v3
	s_branch .LBB3_43
.LBB3_85:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mov_b32_e32 v2, v154
	s_branch .LBB3_44
.LBB3_86:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[7:8], null, 0x408, v169, s[12:13]
	global_load_d16_b16 v2, v[7:8], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e32 v8, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, s6
	s_cbranch_execz .LBB3_81
.LBB3_87:                               ;   in Loop: Header=BB3_45 Depth=2
	v_mad_co_i64_i32 v[177:178], null, 0x408, v171, s[12:13]
	global_load_d16_b16 v2, v[177:178], off offset:1024
	s_wait_loadcnt 0x0
	v_cvt_f32_f16_e64 v174, v2.l
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v169, 0
	s_and_saveexec_b32 s0, s7
	s_cbranch_execnz .LBB3_82
	s_branch .LBB3_83
.LBB3_88:                               ; %._crit_edge.i
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB3_93
; %bb.89:
	v_cmp_eq_u32_e32 vcc_lo, 0, v160
	s_and_b32 s1, vcc_lo, s19
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_91
; %bb.90:
	v_mad_co_u64_u32 v[0:1], null, v145, s17, s[18:19]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v0, 0x102, v0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_store_b64 v[0:1], v[154:155], off
.LBB3_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB3_93
; %bb.92:                               ; %.loopexit.loopexit.i
	v_mad_co_u64_u32 v[0:1], null, v145, s17, s[18:19]
	v_dual_mul_f32 v6, v156, v130 :: v_dual_mul_f32 v9, v9, v156
	v_mul_f32_e32 v8, v122, v156
	v_mul_f32_e32 v98, v98, v156
	v_dual_mul_f32 v90, v90, v156 :: v_dual_mul_f32 v83, v83, v156
	v_dual_mul_f32 v82, v82, v156 :: v_dual_mul_f32 v75, v75, v156
	v_mul_lo_u32 v0, 0x102, v0
	v_dual_mov_b32 v1, 0 :: v_dual_mul_f32 v74, v74, v156
	v_dual_mul_f32 v67, v67, v156 :: v_dual_mul_f32 v66, v66, v156
	v_dual_mul_f32 v59, v59, v156 :: v_dual_mul_f32 v42, v42, v156
	v_mul_f32_e32 v35, v35, v156
	v_add_nc_u32_e32 v2, v0, v164
	v_mul_f32_e32 v5, v156, v129
	v_lshlrev_b64_e32 v[3:4], 2, v[0:1]
	v_mul_f32_e32 v7, v121, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v113, v113, v156 :: v_dual_add_nc_u32 v0, 2, v2
	v_mul_f32_e32 v105, v105, v156
	v_mul_f32_e32 v97, v97, v156
	v_mul_f32_e32 v89, v89, v156
	v_mul_f32_e32 v81, v81, v156
	v_lshlrev_b64_e32 v[137:138], 2, v[0:1]
	v_dual_mul_f32 v73, v73, v156 :: v_dual_add_nc_u32 v0, 18, v2
	v_mul_f32_e32 v65, v65, v156
	v_mul_f32_e32 v57, v57, v156
	v_add_co_u32 v3, vcc_lo, s14, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[139:140], 2, v[0:1]
	v_add_nc_u32_e32 v0, 34, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s15, v4, vcc_lo
	v_add_co_u32 v137, vcc_lo, v3, v137
	v_lshlrev_b64_e32 v[141:142], 2, v[0:1]
	v_add_nc_u32_e32 v0, 50, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v138, null, v4, v138, vcc_lo
	v_add_co_u32 v139, vcc_lo, v3, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[143:144], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x42, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v140, null, v4, v140, vcc_lo
	v_add_co_u32 v141, vcc_lo, v3, v141
	v_lshlrev_b64_e32 v[145:146], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x52, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v142, null, v4, v142, vcc_lo
	v_add_co_u32 v143, vcc_lo, v3, v143
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[147:148], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x62, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v144, null, v4, v144, vcc_lo
	v_add_co_u32 v145, vcc_lo, v3, v145
	v_lshlrev_b64_e32 v[149:150], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x72, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v146, null, v4, v146, vcc_lo
	v_add_co_u32 v147, vcc_lo, v3, v147
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[151:152], 2, v[0:1]
	v_dual_mul_f32 v49, v49, v156 :: v_dual_add_nc_u32 v0, 0x82, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v148, null, v4, v148, vcc_lo
	v_add_co_u32 v149, vcc_lo, v3, v149
	v_lshlrev_b64_e32 v[153:154], 2, v[0:1]
	v_dual_mul_f32 v41, v41, v156 :: v_dual_add_nc_u32 v0, 0x92, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v150, null, v4, v150, vcc_lo
	v_mul_f32_e32 v33, v33, v156
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[157:158], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xa2, v2
	v_add_co_u32 v151, vcc_lo, v3, v151
	v_mul_f32_e32 v25, v25, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v152, null, v4, v152, vcc_lo
	v_lshlrev_b64_e32 v[159:160], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xb2, v2
	v_add_co_u32 v153, vcc_lo, v3, v153
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v154, null, v4, v154, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[161:162], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xc2, v2
	v_add_co_u32 v157, vcc_lo, v3, v157
	s_clause 0x1
	global_store_b64 v[137:138], v[5:6], off
	global_store_b64 v[139:140], v[7:8], off
	v_lshlrev_b64_e32 v[163:164], 2, v[0:1]
	v_dual_mul_f32 v17, v17, v156 :: v_dual_add_nc_u32 v0, 0xd2, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v158, null, v4, v158, vcc_lo
	v_add_co_u32 v159, vcc_lo, v3, v159
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[165:166], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xe2, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v160, null, v4, v160, vcc_lo
	v_add_co_u32 v161, vcc_lo, v3, v161
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xf2, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v162, null, v4, v162, vcc_lo
	v_add_co_u32 v129, vcc_lo, v3, v163
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_add_nc_u32_e32 v0, 4, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, v4, v164, vcc_lo
	v_add_co_u32 v121, vcc_lo, v3, v165
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v122, null, v4, v166, vcc_lo
	v_add_co_u32 v5, vcc_lo, v3, v5
	v_dual_mul_f32 v58, v58, v156 :: v_dual_mul_f32 v51, v51, v156
	v_dual_mul_f32 v50, v50, v156 :: v_dual_mul_f32 v43, v43, v156
	s_clause 0x7
	global_store_b64 v[145:146], v[97:98], off
	global_store_b64 v[147:148], v[89:90], off
	global_store_b64 v[149:150], v[81:82], off
	global_store_b64 v[151:152], v[73:74], off
	global_store_b64 v[153:154], v[65:66], off
	global_store_b64 v[157:158], v[57:58], off
	global_store_b64 v[159:160], v[49:50], off
	global_store_b64 v[161:162], v[41:42], off
	v_dual_mul_f32 v34, v34, v156 :: v_dual_mul_f32 v19, v19, v156
	v_lshlrev_b64_e32 v[41:42], 2, v[0:1]
	v_add_nc_u32_e32 v0, 20, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_add_co_u32 v7, vcc_lo, v3, v7
	v_mul_f32_e32 v26, v26, v156
	v_mul_f32_e32 v18, v18, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v4, v8, vcc_lo
	v_mul_f32_e32 v10, v10, v156
	s_clause 0x3
	global_store_b64 v[129:130], v[33:34], off
	global_store_b64 v[121:122], v[25:26], off
	global_store_b64 v[5:6], v[17:18], off
	global_store_b64 v[7:8], v[9:10], off
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_add_co_u32 v8, vcc_lo, v3, v41
	v_dual_mul_f32 v5, v156, v131 :: v_dual_add_nc_u32 v0, 36, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v4, v42, vcc_lo
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v33, vcc_lo, v3, v6
	v_mul_f32_e32 v6, v156, v132
	v_dual_mul_f32 v114, v114, v156 :: v_dual_mul_f32 v17, v123, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v4, v7, vcc_lo
	v_mul_f32_e32 v18, v124, v156
	global_store_b64 v[8:9], v[5:6], off
	v_mul_f32_e32 v9, v27, v156
	v_mul_f32_e32 v27, v109, v156
	v_lshlrev_b64_e32 v[25:26], 2, v[0:1]
	v_dual_mul_f32 v7, v115, v156 :: v_dual_add_nc_u32 v0, 52, v2
	s_clause 0x1
	global_store_b64 v[141:142], v[113:114], off
	global_store_b64 v[33:34], v[17:18], off
	v_mul_f32_e32 v33, v101, v156
	v_lshlrev_b64_e32 v[41:42], 2, v[0:1]
	v_dual_mul_f32 v53, v53, v156 :: v_dual_add_nc_u32 v0, 0x44, v2
	v_mul_f32_e32 v45, v45, v156
	v_add_co_u32 v25, vcc_lo, v3, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[57:58], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x54, v2
	v_dual_mul_f32 v106, v106, v156 :: v_dual_mul_f32 v49, v107, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v26, vcc_lo
	v_lshlrev_b64_e32 v[65:66], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x64, v2
	v_dual_mul_f32 v8, v116, v156 :: v_dual_mul_f32 v37, v37, v156
	v_add_co_u32 v41, vcc_lo, v3, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[81:82], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x74, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v4, v42, vcc_lo
	v_mul_f32_e32 v50, v108, v156
	s_clause 0x1
	global_store_b64 v[25:26], v[7:8], off
	global_store_b64 v[143:144], v[105:106], off
	v_lshlrev_b64_e32 v[89:90], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x84, v2
	v_add_co_u32 v57, vcc_lo, v3, v57
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, v4, v58, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[97:98], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x94, v2
	v_add_co_u32 v65, vcc_lo, v3, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v4, v66, vcc_lo
	v_lshlrev_b64_e32 v[105:106], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xa4, v2
	v_add_co_u32 v81, vcc_lo, v3, v81
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, v4, v82, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[113:114], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xb4, v2
	v_add_co_u32 v89, vcc_lo, v3, v89
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v90, null, v4, v90, vcc_lo
	v_lshlrev_b64_e32 v[121:122], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xc4, v2
	v_add_co_u32 v97, vcc_lo, v3, v97
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v98, null, v4, v98, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[129:130], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xd4, v2
	v_add_co_u32 v105, vcc_lo, v3, v105
	v_mul_f32_e32 v10, v99, v156
	v_mul_f32_e32 v7, v11, v156
	v_lshlrev_b64_e32 v[137:138], 2, v[0:1]
	v_dual_mul_f32 v11, v100, v156 :: v_dual_add_nc_u32 v0, 0xe4, v2
	v_mul_f32_e32 v73, v91, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v106, null, v4, v106, vcc_lo
	v_add_co_u32 v113, vcc_lo, v3, v113
	v_mul_f32_e32 v74, v92, v156
	v_mul_f32_e32 v84, v84, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v114, null, v4, v114, vcc_lo
	v_add_co_u32 v121, vcc_lo, v3, v121
	v_mul_f32_e32 v76, v76, v156
	v_dual_mul_f32 v68, v68, v156 :: v_dual_mul_f32 v55, v55, v156
	v_dual_mul_f32 v60, v60, v156 :: v_dual_mul_f32 v47, v47, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v122, null, v4, v122, vcc_lo
	v_dual_mul_f32 v52, v52, v156 :: v_dual_mul_f32 v39, v39, v156
	v_dual_mul_f32 v44, v44, v156 :: v_dual_mul_f32 v31, v31, v156
	s_clause 0x7
	global_store_b64 v[57:58], v[10:11], off
	global_store_b64 v[65:66], v[73:74], off
	global_store_b64 v[81:82], v[83:84], off
	global_store_b64 v[89:90], v[75:76], off
	global_store_b64 v[97:98], v[67:68], off
	global_store_b64 v[105:106], v[59:60], off
	global_store_b64 v[113:114], v[51:52], off
	global_store_b64 v[121:122], v[43:44], off
	v_mul_f32_e32 v60, v69, v156
	v_mul_f32_e32 v69, v61, v156
	v_mul_f32_e32 v61, v70, v156
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xf4, v2
	v_add_co_u32 v129, vcc_lo, v3, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, v4, v130, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_dual_mul_f32 v15, v15, v156 :: v_dual_add_nc_u32 v0, 6, v2
	v_add_co_u32 v123, vcc_lo, v3, v137
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v124, null, v4, v138, vcc_lo
	v_add_co_u32 v5, vcc_lo, v3, v5
	v_mul_f32_e32 v36, v36, v156
	v_mul_f32_e32 v8, v12, v156
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 22, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_add_co_u32 v17, vcc_lo, v3, v17
	v_mul_f32_e32 v10, v28, v156
	v_mul_f32_e32 v20, v20, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v4, v18, vcc_lo
	s_clause 0x3
	global_store_b64 v[129:130], v[35:36], off
	global_store_b64 v[123:124], v[9:10], off
	global_store_b64 v[5:6], v[19:20], off
	global_store_b64 v[17:18], v[7:8], off
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_add_nc_u32_e32 v0, 38, v2
	v_add_co_u32 v8, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 54, v2
	v_add_co_u32 v17, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v4, v7, vcc_lo
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x46, v2
	v_add_co_u32 v25, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x56, v2
	v_add_co_u32 v19, vcc_lo, v3, v19
	global_store_b64 v[41:42], v[49:50], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v4, v20, vcc_lo
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x66, v2
	v_add_co_u32 v41, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[43:44], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x76, v2
	v_add_co_u32 v49, vcc_lo, v3, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, v4, v35, vcc_lo
	v_lshlrev_b64_e32 v[51:52], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x86, v2
	v_add_co_u32 v43, vcc_lo, v3, v43
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v44, null, v4, v44, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x96, v2
	v_add_co_u32 v51, vcc_lo, v3, v51
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, v4, v52, vcc_lo
	v_lshlrev_b64_e32 v[65:66], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xa6, v2
	v_add_co_u32 v67, vcc_lo, v3, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, v4, v59, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xb6, v2
	v_add_co_u32 v65, vcc_lo, v3, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v4, v66, vcc_lo
	v_lshlrev_b64_e32 v[73:74], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xc6, v2
	v_add_co_u32 v75, vcc_lo, v3, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, v4, v59, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_dual_mul_f32 v5, v156, v133 :: v_dual_add_nc_u32 v0, 0xd6, v2
	v_mul_f32_e32 v6, v156, v134
	v_dual_mul_f32 v10, v125, v156 :: v_dual_mul_f32 v11, v126, v156
	v_lshlrev_b64_e32 v[81:82], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xe6, v2
	global_store_b64 v[8:9], v[5:6], off
	v_add_co_u32 v73, vcc_lo, v3, v73
	v_mul_f32_e32 v7, v117, v156
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xf6, v2
	v_mul_f32_e32 v8, v118, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v74, null, v4, v74, vcc_lo
	v_add_co_u32 v83, vcc_lo, v3, v58
	v_mul_f32_e32 v28, v110, v156
	s_clause 0x1
	global_store_b64 v[17:18], v[10:11], off
	global_store_b64 v[25:26], v[7:8], off
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v84, null, v4, v59, vcc_lo
	v_add_co_u32 v81, vcc_lo, v3, v81
	v_mul_f32_e32 v34, v102, v156
	v_add_nc_u32_e32 v0, 8, v2
	v_mul_f32_e32 v12, v93, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v82, null, v4, v82, vcc_lo
	v_add_co_u32 v5, vcc_lo, v3, v5
	v_mul_f32_e32 v7, v13, v156
	v_mul_f32_e32 v13, v94, v156
	v_dual_mul_f32 v35, v85, v156 :: v_dual_mul_f32 v36, v86, v156
	v_mul_f32_e32 v57, v77, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v4, v6, vcc_lo
	v_add_co_u32 v25, vcc_lo, v3, v10
	v_mul_f32_e32 v58, v78, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v11, vcc_lo
	v_mul_f32_e32 v70, v62, v156
	v_mul_f32_e32 v54, v54, v156
	v_mul_f32_e32 v46, v46, v156
	s_clause 0x7
	global_store_b64 v[41:42], v[33:34], off
	global_store_b64 v[49:50], v[12:13], off
	global_store_b64 v[43:44], v[35:36], off
	global_store_b64 v[51:52], v[57:58], off
	global_store_b64 v[67:68], v[60:61], off
	global_store_b64 v[65:66], v[69:70], off
	global_store_b64 v[75:76], v[53:54], off
	global_store_b64 v[73:74], v[45:46], off
	v_mul_f32_e32 v38, v38, v156
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_dual_mul_f32 v9, v29, v156 :: v_dual_add_nc_u32 v0, 24, v2
	v_dual_mul_f32 v10, v30, v156 :: v_dual_mul_f32 v17, v21, v156
	v_mul_f32_e32 v18, v22, v156
	v_mul_f32_e32 v8, v14, v156
	s_clause 0x3
	global_store_b64 v[83:84], v[37:38], off
	global_store_b64 v[81:82], v[9:10], off
	global_store_b64 v[5:6], v[17:18], off
	global_store_b64 v[25:26], v[7:8], off
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_add_nc_u32_e32 v0, 40, v2
	v_add_co_u32 v8, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v4, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 56, v2
	v_add_co_u32 v13, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v4, v7, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x48, v2
	global_store_b64 v[19:20], v[27:28], off
	v_add_co_u32 v19, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v4, v12, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x58, v2
	v_add_co_u32 v25, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v4, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x68, v2
	v_add_co_u32 v29, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v4, v12, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x78, v2
	v_add_co_u32 v35, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, v4, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x88, v2
	v_add_co_u32 v41, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v42, null, v4, v12, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0x98, v2
	v_add_co_u32 v45, vcc_lo, v3, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, v4, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[49:50], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xa8, v2
	v_add_co_u32 v51, vcc_lo, v3, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, v4, v12, vcc_lo
	v_lshlrev_b64_e32 v[53:54], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xb8, v2
	v_add_co_u32 v49, vcc_lo, v3, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v50, null, v4, v50, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[57:58], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xc8, v2
	v_add_co_u32 v53, vcc_lo, v3, v53
	v_mul_f32_e32 v5, v156, v135
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v54, null, v4, v54, vcc_lo
	v_lshlrev_b64_e32 v[59:60], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xd8, v2
	v_add_co_u32 v57, vcc_lo, v3, v57
	v_mul_f32_e32 v6, v156, v136
	v_mul_f32_e32 v10, v127, v156
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[61:62], 2, v[0:1]
	v_dual_mul_f32 v11, v128, v156 :: v_dual_add_nc_u32 v0, 0xe8, v2
	v_mul_f32_e32 v17, v119, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v58, null, v4, v58, vcc_lo
	v_lshlrev_b64_e32 v[65:66], 2, v[0:1]
	v_add_nc_u32_e32 v0, 0xf8, v2
	v_add_co_u32 v59, vcc_lo, v3, v59
	v_dual_mul_f32 v18, v120, v156 :: v_dual_mul_f32 v21, v111, v156
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_dual_mul_f32 v22, v112, v156 :: v_dual_mul_f32 v27, v103, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v60, null, v4, v60, vcc_lo
	v_add_co_u32 v61, vcc_lo, v3, v61
	v_dual_mul_f32 v28, v104, v156 :: v_dual_mul_f32 v33, v95, v156
	v_mul_f32_e32 v37, v87, v156
	v_mul_f32_e32 v43, v79, v156
	v_dual_mul_f32 v7, v71, v156 :: v_dual_mul_f32 v34, v96, v156
	v_mul_f32_e32 v38, v88, v156
	v_mul_f32_e32 v44, v80, v156
	s_clause 0x7
	global_store_b64 v[8:9], v[5:6], off
	global_store_b64 v[13:14], v[10:11], off
	global_store_b64 v[19:20], v[17:18], off
	global_store_b64 v[25:26], v[21:22], off
	global_store_b64 v[29:30], v[27:28], off
	global_store_b64 v[35:36], v[33:34], off
	global_store_b64 v[41:42], v[37:38], off
	global_store_b64 v[45:46], v[43:44], off
	v_mul_f32_e32 v8, v72, v156
	v_mul_f32_e32 v12, v63, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v62, null, v4, v62, vcc_lo
	v_add_co_u32 v65, vcc_lo, v3, v65
	v_mul_f32_e32 v13, v64, v156
	v_mul_f32_e32 v56, v56, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, v4, v66, vcc_lo
	v_add_co_u32 v0, vcc_lo, v3, v0
	v_mul_f32_e32 v48, v48, v156
	v_mul_f32_e32 v40, v40, v156
	v_mul_f32_e32 v2, v23, v156
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v4, v1, vcc_lo
	v_mul_f32_e32 v32, v32, v156
	v_mul_f32_e32 v3, v24, v156
	v_mul_f32_e32 v16, v16, v156
	s_clause 0x7
	global_store_b64 v[51:52], v[7:8], off
	global_store_b64 v[49:50], v[12:13], off
	global_store_b64 v[53:54], v[55:56], off
	global_store_b64 v[57:58], v[47:48], off
	global_store_b64 v[59:60], v[39:40], off
	global_store_b64 v[61:62], v[31:32], off
	global_store_b64 v[65:66], v[2:3], off
	global_store_b64 v[0:1], v[15:16], off
.LBB3_93:                               ; %_Z16fa2_stageb_nbodyILb1EEvPKhS1_S1_PfPKiifiiiiii.exit
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end3:
