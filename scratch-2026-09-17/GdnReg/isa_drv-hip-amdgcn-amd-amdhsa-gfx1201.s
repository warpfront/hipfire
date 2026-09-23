	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gated_delta_net_q8_register_scan_gfx1201 ; -- Begin function gated_delta_net_q8_register_scan_gfx1201
	.globl	gated_delta_net_q8_register_scan_gfx1201
	.p2align	8
	.type	gated_delta_net_q8_register_scan_gfx1201,@function
gated_delta_net_q8_register_scan_gfx1201: ; @gated_delta_net_q8_register_scan_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[56:59], s[0:1], 0x40
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 ttmp9, s57
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s58, 0x80
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lt_i32 s56, 1
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_384
; %bb.1:                                ; %.preheader969
	v_lshrrev_b32_e32 v1, 2, v0
	s_load_b512 s[40:55], s[0:1], 0x0
	v_mbcnt_lo_u32_b32 v15, -1, 0
	s_and_b32 s2, ttmp7, 0xffff
	s_mov_b32 s69, 0
	v_and_b32_e32 v1, 0xf0, v1
	s_lshr_b32 s68, ttmp7, 16
	v_xor_b32_e32 v18, 16, v15
	v_and_b32_e32 v10, 15, v0
	s_ashr_i32 s67, s57, 31
	v_lshl_add_u32 v1, s2, 6, v1
	s_mov_b32 s66, s57
	s_mov_b32 s60, ttmp9
	s_mul_u64 s[2:3], s[66:67], s[68:69]
	s_ashr_i32 s61, ttmp9, 31
	v_or_b32_e32 v9, v1, v10
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[64:65], s[2:3], 14
	v_bfe_u32 v22, v0, 5, 1
	s_lshl_b64 s[62:63], s[60:61], 14
	v_bfe_u32 v21, v0, 4, 1
	v_lshlrev_b32_e32 v1, 7, v9
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[50:51], s[64:65]
	v_lshlrev_b32_e32 v3, 6, v22
	s_add_nc_u64 s[4:5], s[4:5], s[62:63]
	v_lshlrev_b32_e32 v78, 3, v21
	scratch_store_b32 off, v1, off offset:52 ; 4-byte Folded Spill
	v_add_co_u32 v1, s4, s4, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s5, 0, s4
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v1, off offset:32
	scratch_store_b32 off, v3, off offset:48
	v_add_co_u32 v1, vcc_lo, v1, v3
	scratch_store_b32 off, v2, off offset:36 ; 4-byte Folded Spill
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_lshrrev_b32_e32 v13, 5, v0
	v_bfi_b32 v19, v15, 0, 32
	v_lshlrev_b32_e32 v14, 5, v0
	v_lshrrev_b32_e32 v156, 6, v0
	v_add_co_u32 v7, vcc_lo, v1, v78
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v2, vcc_lo
	v_lshlrev_b32_e32 v16, 1, v0
	v_lshlrev_b32_e32 v13, 3, v13
	v_cmp_lt_u32_e32 vcc_lo, v18, v19
	s_lshl_b32 s4, ttmp9, 7
	v_and_b32_e32 v81, 63, v0
	v_and_b32_e32 v17, 0x1e0, v14
	v_xor_b32_e32 v88, 0x400, v14
	v_xor_b32_e32 v89, 0x404, v14
	v_xor_b32_e32 v90, 0x408, v14
	v_xor_b32_e32 v91, 0x40c, v14
	v_xor_b32_e32 v92, 0x410, v14
	v_xor_b32_e32 v93, 0x414, v14
	v_xor_b32_e32 v94, 0x418, v14
	v_xor_b32_e32 v95, 0x41c, v14
	v_lshlrev_b32_e32 v14, 2, v156
	v_or_b32_e32 v157, 33, v78
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	v_and_b32_e32 v16, 32, v16
	v_and_or_b32 v164, v0, 7, v13
	v_or_b32_e32 v114, v13, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v15, v18, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[8:9], s[4:5], 2
	v_lshl_or_b32 v115, v81, 8, v14
	v_and_b32_e32 v14, 9, v157
	s_add_nc_u64 s[14:15], s[40:41], s[8:9]
	v_lshlrev_b32_e32 v135, 2, v13
	v_add_co_u32 v13, s16, s14, v16
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v137, 2, v14
	v_add_co_ci_u32_e64 v14, null, s15, 0, s16
	s_clause 0x3
	global_load_b64 v[1:2], v[7:8], off
	global_load_b64 v[3:4], v[7:8], off offset:16
	global_load_b64 v[5:6], v[7:8], off offset:32
	global_load_b64 v[7:8], v[7:8], off offset:48
	scratch_store_b32 off, v22, off offset:44 ; 4-byte Folded Spill
	s_lshl_b64 s[2:3], s[2:3], 9
	scratch_store_b64 off, v[13:14], off offset:16 ; 8-byte Folded Spill
	v_add_co_u32 v13, s14, s14, v17
	v_lshlrev_b32_e32 v11, 2, v9
	v_lshlrev_b32_e32 v20, 8, v22
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[52:53], s[2:3]
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v13, off
	scratch_store_b32 off, v164, off offset:12
	v_add_co_ci_u32_e64 v13, null, s15, 0, s14
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[2:3], s[8:9]
	s_add_nc_u64 s[10:11], s[42:43], s[8:9]
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v21, off offset:40
	scratch_store_b32 off, v13, off offset:4
	v_add_co_u32 v152, s16, s10, v17
	v_add_co_u32 v13, s10, s10, v20
	scratch_store_b32 off, v9, off offset:56 ; 4-byte Folded Spill
	global_load_b32 v9, v11, s[6:7]
	v_and_b32_e32 v12, 16, v0
	v_lshlrev_b32_e32 v96, 11, v21
	v_cmp_gt_u32_e64 s4, 8, v10
	v_lshlrev_b32_e32 v87, 7, v10
	v_or_b32_e32 v108, 34, v78
	v_lshl_or_b32 v85, v10, 8, v12
	v_lshl_or_b32 v116, v10, 2, v96
	v_add_nc_u32_e32 v10, -1, v114
	v_lshl_or_b32 v134, v164, 8, v12
	v_add_nc_u32_e32 v12, 1, v114
	v_and_b32_e32 v15, 10, v108
	scratch_store_b32 off, v13, off offset:24 ; 4-byte Folded Spill
	v_mul_i32_i24_e32 v10, v10, v114
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s11, 0, s10
	v_add_co_u32 v14, s6, s6, v11
	v_or_b32_e32 v113, 35, v78
	v_or_b32_e32 v67, 36, v78
	v_or_b32_e32 v109, 37, v78
	v_or_b32_e32 v154, 38, v78
	v_or_b32_e32 v155, 39, v78
	v_or_b32_e32 v103, 48, v78
	v_or_b32_e32 v166, 49, v78
	v_or_b32_e32 v79, 50, v78
	v_or_b32_e32 v82, 51, v78
	v_or_b32_e32 v84, 52, v78
	v_or_b32_e32 v98, 53, v78
	v_or_b32_e32 v99, 54, v78
	v_or_b32_e32 v102, 55, v78
	v_mul_u32_u24_e32 v12, v12, v114
	v_lshlrev_b32_e32 v138, 2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s7, 0, s6
	scratch_store_b32 off, v13, off offset:28 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v13, 31, v10
	v_cmp_eq_u32_e64 s5, 0, v22
	v_lshlrev_b32_e32 v97, 5, v21
	v_lshl_or_b32 v133, v22, 7, v85
	v_and_b32_e32 v18, 11, v113
	v_and_b32_e32 v19, 12, v67
	v_and_b32_e32 v21, 13, v109
	v_and_b32_e32 v22, 14, v154
	v_and_b32_e32 v23, 15, v155
	v_and_b32_e32 v24, 24, v103
	v_and_b32_e32 v25, 25, v166
	v_and_b32_e32 v26, 26, v79
	v_and_b32_e32 v27, 27, v82
	v_and_b32_e32 v28, 28, v84
	v_and_b32_e32 v29, 29, v98
	v_and_b32_e32 v30, 30, v99
	v_and_b32_e32 v31, 31, v102
	v_lshlrev_b32_e32 v12, 1, v12
	v_add_lshl_u32 v10, v10, v13, 1
	s_add_nc_u64 s[12:13], s[44:45], s[8:9]
	s_add_nc_u64 s[8:9], s[54:55], s[8:9]
	v_add_co_u32 v158, s6, s12, v11
	v_lshlrev_b32_e32 v139, 2, v18
	v_lshlrev_b32_e32 v140, 2, v19
	v_lshlrev_b32_e32 v141, 2, v21
	v_lshlrev_b32_e32 v142, 2, v22
	v_lshlrev_b32_e32 v143, 2, v23
	v_lshlrev_b32_e32 v144, 2, v24
	v_lshlrev_b32_e32 v145, 2, v25
	v_lshlrev_b32_e32 v146, 2, v26
	v_lshlrev_b32_e32 v147, 2, v27
	v_lshlrev_b32_e32 v148, 2, v28
	v_lshlrev_b32_e32 v149, 2, v29
	v_lshlrev_b32_e32 v150, 2, v30
	v_lshlrev_b32_e32 v151, 2, v31
	scratch_store_b64 off, v[14:15], off offset:60 ; 8-byte Folded Spill
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v159, null, s13, 0, s6
	v_add_co_u32 v160, s6, s8, v11
	v_and_b32_e32 v162, -4, v12
	v_and_b32_e32 v163, -4, v10
	s_lshl_b32 s50, s57, 7
	v_and_b32_e32 v72, 31, v0
	v_cmp_gt_u32_e64 s2, 64, v0
	v_cmp_eq_u32_e64 s3, 0, v0
	v_lshlrev_b32_e32 v80, 3, v0
	v_lshrrev_b32_e32 v112, 4, v0
	v_lshlrev_b32_e32 v83, 4, v0
	v_lshlrev_b32_e32 v86, 2, v81
	v_or_b32_e32 v117, 1, v78
	v_or_b32_e32 v118, 2, v78
	v_or_b32_e32 v119, 3, v78
	v_or_b32_e32 v120, 4, v78
	v_or_b32_e32 v121, 5, v78
	v_or_b32_e32 v122, 6, v78
	v_or_b32_e32 v123, 7, v78
	v_or_b32_e32 v124, 16, v78
	v_or_b32_e32 v125, 17, v78
	v_or_b32_e32 v126, 18, v78
	v_or_b32_e32 v127, 19, v78
	v_or_b32_e32 v128, 20, v78
	v_or_b32_e32 v129, 21, v78
	v_or_b32_e32 v130, 22, v78
	v_or_b32_e32 v131, 23, v78
	v_or_b32_e32 v104, 32, v78
	v_lshlrev_b32_e32 v136, 2, v114
	v_add_co_ci_u32_e64 v153, null, s11, 0, s16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v161, null, s9, 0, s6
	s_ashr_i32 s51, s50, 31
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[52:53], s[50:51], 2
	s_wait_loadcnt 0x4
	v_bfe_i32 v10, v1, 0, 8
	v_bfe_i32 v11, v1, 8, 8
	v_bfe_i32 v12, v1, 16, 8
	v_ashrrev_i32_e32 v1, 24, v1
	v_bfe_i32 v13, v2, 0, 8
	v_bfe_i32 v14, v2, 8, 8
	v_bfe_i32 v15, v2, 16, 8
	v_ashrrev_i32_e32 v2, 24, v2
	s_wait_loadcnt 0x3
	v_bfe_i32 v16, v3, 0, 8
	v_bfe_i32 v17, v3, 8, 8
	v_bfe_i32 v18, v3, 16, 8
	v_ashrrev_i32_e32 v3, 24, v3
	v_bfe_i32 v19, v4, 0, 8
	v_bfe_i32 v20, v4, 8, 8
	v_bfe_i32 v21, v4, 16, 8
	v_ashrrev_i32_e32 v4, 24, v4
	s_wait_loadcnt 0x2
	v_bfe_i32 v22, v5, 0, 8
	v_bfe_i32 v23, v5, 8, 8
	v_bfe_i32 v24, v5, 16, 8
	v_ashrrev_i32_e32 v5, 24, v5
	v_bfe_i32 v25, v6, 0, 8
	v_bfe_i32 v26, v6, 8, 8
	v_bfe_i32 v27, v6, 16, 8
	v_ashrrev_i32_e32 v6, 24, v6
	s_wait_loadcnt 0x1
	v_bfe_i32 v28, v7, 0, 8
	v_bfe_i32 v29, v7, 8, 8
	v_bfe_i32 v30, v7, 16, 8
	v_ashrrev_i32_e32 v7, 24, v7
	v_bfe_i32 v31, v8, 0, 8
	v_bfe_i32 v32, v8, 8, 8
	v_bfe_i32 v33, v8, 16, 8
	v_ashrrev_i32_e32 v8, 24, v8
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v15, v15
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v22, v22
	v_cvt_f32_i32_e32 v23, v23
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v8, v8
	s_wait_loadcnt 0x0
	v_fma_mixlo_f16 v57, v9, v10, 0
	v_fma_mixhi_f16 v57, v9, v11, 0
	v_fma_mixlo_f16 v58, v9, v12, 0
	v_fma_mixhi_f16 v58, v9, v1, 0
	v_fma_mixlo_f16 v59, v9, v13, 0
	v_fma_mixhi_f16 v59, v9, v14, 0
	v_fma_mixlo_f16 v60, v9, v15, 0
	v_fma_mixhi_f16 v60, v9, v2, 0
	v_fma_mixlo_f16 v53, v9, v16, 0
	v_fma_mixhi_f16 v53, v9, v17, 0
	v_fma_mixlo_f16 v54, v9, v18, 0
	v_fma_mixhi_f16 v54, v9, v3, 0
	v_fma_mixlo_f16 v55, v9, v19, 0
	v_fma_mixhi_f16 v55, v9, v20, 0
	v_fma_mixlo_f16 v56, v9, v21, 0
	v_fma_mixhi_f16 v56, v9, v4, 0
	v_fma_mixlo_f16 v49, v9, v22, 0
	v_fma_mixhi_f16 v49, v9, v23, 0
	v_fma_mixlo_f16 v50, v9, v24, 0
	v_fma_mixhi_f16 v50, v9, v5, 0
	v_fma_mixlo_f16 v51, v9, v25, 0
	v_fma_mixhi_f16 v51, v9, v26, 0
	v_fma_mixlo_f16 v52, v9, v27, 0
	v_fma_mixhi_f16 v52, v9, v6, 0
	v_fma_mixlo_f16 v61, v9, v28, 0
	v_fma_mixhi_f16 v61, v9, v29, 0
	v_fma_mixlo_f16 v62, v9, v30, 0
	v_fma_mixhi_f16 v62, v9, v7, 0
	v_fma_mixlo_f16 v63, v9, v31, 0
	v_fma_mixhi_f16 v63, v9, v32, 0
	v_fma_mixlo_f16 v64, v9, v33, 0
	v_fma_mixhi_f16 v64, v9, v8, 0
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v64.h, v40
	v_cvt_f16_f32_e32 v64.l, v39
	v_cvt_f16_f32_e32 v63.h, v38
	v_cvt_f16_f32_e32 v63.l, v37
	v_cvt_f16_f32_e32 v62.h, v36
	v_cvt_f16_f32_e32 v62.l, v35
	v_cvt_f16_f32_e32 v61.h, v34
	v_cvt_f16_f32_e32 v61.l, v33
	v_add_nc_u32_e32 v164, 64, v164
	s_and_b32 vcc_lo, exec_lo, s57
	s_mov_b32 s69, s45
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_378
.LBB0_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_9 Depth 2
                                        ;     Child Loop BB0_16 Depth 2
                                        ;     Child Loop BB0_21 Depth 2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;     Child Loop BB0_44 Depth 2
                                        ;     Child Loop BB0_185 Depth 2
                                        ;     Child Loop BB0_188 Depth 2
                                        ;     Child Loop BB0_193 Depth 2
                                        ;       Child Loop BB0_195 Depth 3
                                        ;     Child Loop BB0_216 Depth 2
                                        ;     Child Loop BB0_221 Depth 2
                                        ;     Child Loop BB0_235 Depth 2
                                        ;     Child Loop BB0_249 Depth 2
                                        ;     Child Loop BB0_263 Depth 2
                                        ;     Child Loop BB0_276 Depth 2
                                        ;     Child Loop BB0_280 Depth 2
                                        ;     Child Loop BB0_298 Depth 2
                                        ;     Child Loop BB0_316 Depth 2
                                        ;     Child Loop BB0_334 Depth 2
	s_and_saveexec_b32 s6, s2
	s_cbranch_execz .LBB0_7
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	v_or_b32_e32 v3, s69, v0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s56, v3
	s_cbranch_execz .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[1:2], null, v3, s66, s[60:61]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, v3, s67, v[2:3]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s46, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s47, v2, vcc_lo
	v_add_co_u32 v5, vcc_lo, s48, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s49, v2, vcc_lo
	global_load_b32 v2, v[3:4], off
	global_load_b32 v1, v[5:6], off
.LBB0_6:                                ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_lshlrev_b32_e32 v3, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, 0x80, v3
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b32 v3, v2, v1 offset0:112 offset1:114
.LBB0_7:                                ; %Flow2762
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s6, s3
	s_cbranch_execz .LBB0_10
; %bb.8:                                ; %.preheader966.preheader
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_mov_b32_e32 v1, 0
	s_movk_i32 s7, 0xff00
.LBB0_9:                                ; %.preheader966
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v2, s7
	s_add_co_i32 s7, s7, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s7, 0
	ds_load_b32 v3, v2 offset:29056
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v3
	ds_store_b32 v2, v1 offset:29056
	s_cbranch_scc1 .LBB0_9
.LBB0_10:                               ; %Flow2761
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s6, s2
	s_cbranch_execz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_3 Depth=1
	v_lshlrev_b32_e32 v5, 2, v0
	ds_load_b32 v1, v5 offset:28800
	s_wait_dscnt 0x0
	v_mul_f32_e32 v2, 0x3fb8aa3b, v1
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_rndne_f32_e32 v3, v2
	v_fma_f32 v4, 0x3fb8aa3b, v1, -v2
	v_sub_f32_e32 v2, v2, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v4, 0x32a5705f, v1
	v_cvt_i32_f32_e32 v3, v3
	v_add_f32_e32 v2, v2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v2, v2
	v_ldexp_f32 v2, v2, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x7f800000, v2, vcc_lo
	ds_store_b32 v5, v1 offset:29056
.LBB0_12:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_add_co_i32 s45, s69, 64
	s_sub_co_i32 s6, s56, s69
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s45, s56
	s_cselect_b32 s57, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s7, s57, exec_lo
	s_cselect_b32 s58, s6, 64
	s_and_saveexec_b32 s6, s3
	s_cbranch_execz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s7, s58, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v1, s7
	ds_load_b32 v2, v1 offset:28796
	s_wait_dscnt 0x0
	v_readfirstlane_b32 s7, v2
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v2
	s_mul_f32 s7, s7, 0x3fb8aa3b
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_2)
	s_xor_b32 s8, s7, 0x80000000
	s_wait_alu depctr_sa_sdst(0)
	v_fma_f32 v3, 0x3fb8aa3b, v2, s8
	s_rndne_f32 s8, s7
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s7, s7, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v3, 0x32a5705f, v2
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_add_f32_e32 v3, s7, v3
	s_cvt_i32_f32 s7, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v3, v3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_1)
	v_ldexp_f32 v3, v3, s7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v2
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v2, 0x7f800000, v3 :: v_dual_mov_b32 v3, 0
	ds_store_b32 v3, v2 offset:29568
	ds_load_b32 v1, v1 offset:28796
	s_wait_dscnt 0x0
	ds_store_b32 v3, v1 offset:29572
.LBB0_14:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s6, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_16
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v8.h, v8
	v_cvt_f16_f32_e32 v8.l, v7
	v_cvt_f16_f32_e32 v7.h, v6
	v_cvt_f16_f32_e32 v7.l, v5
	v_cvt_f16_f32_e32 v6.h, v4
	v_cvt_f16_f32_e32 v6.l, v3
	v_cvt_f16_f32_e32 v5.h, v2
	v_cvt_f16_f32_e32 v5.l, v1
	v_lshl_add_u32 v1, s7, 1, v83
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, 4
	ds_store_b128 v1, v[5:8] offset:8320
	s_cbranch_scc1 .LBB0_18
.LBB0_16:                               ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s7, s6, 11
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v8, 0
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v1, s7, v80
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v1, 7, v1
	s_mov_b32 s8, exec_lo
	v_dual_mov_b32 v4, 0 :: v_dual_add_nc_u32 v9, s69, v1
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s56, v9
	s_cbranch_execz .LBB0_15
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=2
	v_mad_co_u64_u32 v[1:2], null, v9, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, v9, s51, v[2:3]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v152, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v153, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[1:4], v[5:6], off
	global_load_b128 v[5:8], v[5:6], off offset:16
	s_branch .LBB0_15
.LBB0_18:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_cmp_le_i32_e64 s6, s58, v114
	s_mov_b32 s7, 0
	v_cmp_gt_i32_e32 vcc_lo, s58, v114
	s_xor_b32 s8, s6, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_21
.LBB0_19:                               ; %.loopexit961.loopexit
                                        ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	ds_store_b32 v1, v2 offset:28
.LBB0_20:                               ; %Flow2759
                                        ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_add_co_i32 s7, s7, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 4
	s_cbranch_scc1 .LBB0_42
.LBB0_21:                               ; %.preheader962
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_22 Depth 3
	v_mov_b32_e32 v1, 0
	s_wait_dscnt 0x9
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v9, s7, 12, v85
	s_mov_b32 s6, 0
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v3, v1
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_mov_b32_e32 v8, v1
.LBB0_22:                               ; %.preheader950
                                        ;   Parent Loop BB0_3 Depth=1
                                        ;     Parent Loop BB0_21 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s9, s6, 5
	s_add_co_i32 s6, s6, 1
	s_wait_dscnt 0x8
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v10, s9, v134
	v_add_nc_u32_e32 v14, s9, v9
	s_cmp_lg_u32 s6, 8
	ds_load_b128 v[10:13], v10 offset:8320
	ds_load_b128 v[14:17], v14 offset:8320
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[10:13], v[14:17], v[1:8]
	s_cbranch_scc1 .LBB0_22
; %bb.23:                               ;   in Loop: Header=BB0_21 Depth=2
	s_and_saveexec_b32 s9, s4
	s_cbranch_execz .LBB0_20
; %bb.24:                               ;   in Loop: Header=BB0_21 Depth=2
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v9, 0
	s_and_saveexec_b32 s6, vcc_lo
; %bb.25:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v9, v136 offset:29312
; %bb.26:                               ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_load_b32 v10, v136 offset:28800
	v_lshl_or_b32 v12, s7, 4, v78
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_u32_e64 s6, v12, v114
	v_lshlrev_b32_e32 v11, 2, v12
	s_and_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v13, v11 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v13, v10, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v14, 0x3fb8aa3b, v13
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v13
	v_fma_f32 v15, 0x3fb8aa3b, v13, -v14
	v_rndne_f32_e32 v16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v15, 0x32a5705f, v13 :: v_dual_sub_f32 v14, v14, v16
	v_add_f32_e32 v14, v14, v15
	v_cvt_i32_f32_e32 v15, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v14, v14
	v_ldexp_f32 v14, v14, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v14, 0, v14, s6
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v13
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v13, 0x7f800000, v14, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v13, v9, v13
	v_mul_f32_e32 v14, v1, v13
.LBB0_28:                               ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v13, 1, v12
	v_lshl_add_u32 v1, v12, 2, v163
	v_mov_b32_e32 v15, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_lt_u32_e64 s6, v13, v114
	v_mov_b32_e32 v13, 0
	ds_store_b32 v1, v14
	s_and_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v14, v11 offset:28804
	s_wait_dscnt 0x0
	v_sub_f32_e32 v14, v10, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, 0x3fb8aa3b, v14
	v_fma_f32 v16, 0x3fb8aa3b, v14, -v15
	v_rndne_f32_e32 v17, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v15, v15, v17 :: v_dual_fmac_f32 v16, 0x32a5705f, v14
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v14
	v_add_f32_e32 v15, v15, v16
	v_cvt_i32_f32_e32 v16, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v15, v15
	v_ldexp_f32 v15, v15, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v15, 0, v15, s6
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v14
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v14, 0x7f800000, v15, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v9, v14
	v_mul_f32_e32 v15, v2, v14
.LBB0_30:                               ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v2, 2, v12
	ds_store_b32 v1, v15 offset:4
	v_cmp_lt_u32_e64 s6, v2, v114
	s_and_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v2, v11 offset:28808
	s_wait_dscnt 0x0
	v_sub_f32_e32 v2, v10, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v13, 0x3fb8aa3b, v2
	v_fma_f32 v14, 0x3fb8aa3b, v2, -v13
	v_rndne_f32_e32 v15, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v13, v13, v15 :: v_dual_fmac_f32 v14, 0x32a5705f, v2
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v2
	v_add_f32_e32 v13, v13, v14
	v_cvt_i32_f32_e32 v14, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v13, v13
	v_ldexp_f32 v13, v13, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v13, 0, v13, s6
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, 0x7f800000, v13, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v9, v2
	v_mul_f32_e32 v13, v3, v2
.LBB0_32:                               ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v2, 3, v12
	ds_store_b32 v1, v13 offset:8
	v_mov_b32_e32 v3, 0
	v_cmp_lt_u32_e64 s6, v2, v114
	v_mov_b32_e32 v2, 0
	s_and_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v3, v11 offset:28812
	s_wait_dscnt 0x0
	v_sub_f32_e32 v3, v10, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v13, 0x3fb8aa3b, v3
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v3
	v_fma_f32 v14, 0x3fb8aa3b, v3, -v13
	v_rndne_f32_e32 v15, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v14, 0x32a5705f, v3
	v_sub_f32_e32 v13, v13, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v13, v13, v14
	v_cvt_i32_f32_e32 v14, v15
	v_exp_f32_e32 v13, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v13, v13, v14
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v13, 0, v13, s6
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v3
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v3, 0x7f800000, v13, s6
	v_mul_f32_e32 v3, v9, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v3, v4, v3
.LBB0_34:                               ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v4, 4, v12
	ds_store_b32 v1, v3 offset:12
	v_cmp_lt_u32_e64 s6, v4, v114
	s_and_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v2, v11 offset:28816
	s_wait_dscnt 0x0
	v_sub_f32_e32 v2, v10, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v13, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v3, v3, v13 :: v_dual_fmac_f32 v4, 0x32a5705f, v2
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v2
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v3, v3
	v_ldexp_f32 v3, v3, v4
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v3, 0, v3, s6
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, 0x7f800000, v3, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v9, v2
	v_mul_f32_e32 v2, v5, v2
.LBB0_36:                               ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v3, 5, v12
	ds_store_b32 v1, v2 offset:16
	v_mov_b32_e32 v4, 0
	v_cmp_lt_u32_e64 s6, v3, v114
	v_mov_b32_e32 v3, 0
	s_and_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v2, v11 offset:28820
	s_wait_dscnt 0x0
	v_sub_f32_e32 v2, v10, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v4, 0x3fb8aa3b, v2
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v2
	v_fma_f32 v5, 0x3fb8aa3b, v2, -v4
	v_rndne_f32_e32 v13, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v5, 0x32a5705f, v2 :: v_dual_sub_f32 v4, v4, v13
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v4, v4
	v_ldexp_f32 v4, v4, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v4, 0, v4, s6
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, 0x7f800000, v4, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v9, v2
	v_mul_f32_e32 v4, v6, v2
.LBB0_38:                               ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v2, 6, v12
	ds_store_b32 v1, v4 offset:20
	v_cmp_lt_u32_e64 s6, v2, v114
	s_and_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v2, v11 offset:28824
	s_wait_dscnt 0x0
	v_sub_f32_e32 v2, v10, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v3, v3, v5 :: v_dual_fmac_f32 v4, 0x32a5705f, v2
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v2
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v3, v3
	v_ldexp_f32 v3, v3, v4
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v3, 0, v3, s6
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, 0x7f800000, v3, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v9, v2
	v_mul_f32_e32 v3, v7, v2
.LBB0_40:                               ;   in Loop: Header=BB0_21 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v2, 7, v12
	ds_store_b32 v1, v3 offset:24
	v_cmp_lt_u32_e64 s6, v2, v114
	v_mov_b32_e32 v2, 0
	s_and_b32 s6, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s6
	s_cbranch_execz .LBB0_19
; %bb.41:                               ;   in Loop: Header=BB0_21 Depth=2
	ds_load_b32 v2, v11 offset:28828
	s_wait_dscnt 0x0
	v_sub_f32_e32 v2, v10, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, 0x3fb8aa3b, v2
	v_fma_f32 v4, 0x3fb8aa3b, v2, -v3
	v_rndne_f32_e32 v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v3, v3, v5 :: v_dual_fmac_f32 v4, 0x32a5705f, v2
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v2
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v3, v3
	v_ldexp_f32 v3, v3, v4
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v3, 0, v3, s6
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, 0x7f800000, v3, s6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v9, v2
	v_mul_f32_e32 v2, v8, v2
	s_branch .LBB0_19
.LBB0_42:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v1, v156
	s_movk_i32 s6, 0xff00
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_44
.LBB0_43:                               ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v3, s6, v115
	v_add_nc_u32_e32 v1, 4, v1
	s_add_co_i32 s6, s6, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, 0
	s_wait_dscnt 0x0
	ds_store_b32 v3, v2 offset:8576
	s_cbranch_scc1 .LBB0_48
.LBB0_44:                               ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mov_b32 s7, exec_lo
                                        ; implicit-def: $vgpr2
	v_cmpx_ge_u32_e64 v81, v1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s7, exec_lo, s7
; %bb.45:                               ;   in Loop: Header=BB0_44 Depth=2
	v_cmp_eq_u32_e32 vcc_lo, v81, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v2, 0, 1.0, vcc_lo
; %bb.46:                               ; %Flow2758
                                        ;   in Loop: Header=BB0_44 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s7, s7
	s_cbranch_execz .LBB0_43
; %bb.47:                               ;   in Loop: Header=BB0_44 Depth=2
	v_add_nc_u32_e32 v2, -1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, v1
	v_lshlrev_b32_e32 v2, 1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v2, -4, v2
	v_add_nc_u32_e32 v2, v86, v2
	ds_load_b32 v2, v2
	s_branch .LBB0_43
.LBB0_48:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v48, s69, v112
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v1, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0
	v_mov_b32_e32 v4, 0
	v_cmp_gt_i32_e64 s41, s56, v48
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s6, s41
	s_cbranch_execz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[1:2], null, v48, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, v48, s51, v[2:3]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v152, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v153, v2, vcc_lo
	s_clause 0x1
	global_load_b128 v[1:4], v[5:6], off
	global_load_b128 v[5:8], v[5:6], off offset:16
.LBB0_50:                               ; %.preheader949
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v8.h, v8
	v_cvt_f16_f32_e32 v8.l, v7
	v_cvt_f16_f32_e32 v7.h, v6
	v_cvt_f16_f32_e32 v7.l, v5
	v_cvt_f16_f32_e32 v6.h, v4
	v_cvt_f16_f32_e32 v6.l, v3
	v_cvt_f16_f32_e32 v5.h, v2
	v_cvt_f16_f32_e32 v5.l, v1
	v_add_nc_u32_e32 v165, v83, v83
	v_or_b32_e32 v74, s69, v78
	ds_store_b128 v83, v[5:8] offset:24704
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_i32_e64 s40, s56, v74
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[10:13], v133 offset:24704
	ds_load_b128 v[15:18], v133 offset:24736
	ds_load_b128 v[19:22], v133 offset:24768
	ds_load_b128 v[23:26], v133 offset:24800
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[2:9], v[10:13], v[57:60], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[2:9], v[15:18], v[53:56], v[2:9]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[2:9], v[19:22], v[49:52], v[2:9]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[2:9], v[23:26], v[61:64], v[2:9]
	ds_store_b128 v165, v[2:5]
	ds_store_b128 v165, v[6:9] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v18, v88
	ds_load_b32 v17, v89
	ds_load_b32 v16, v90
	ds_load_b32 v15, v91
	ds_load_b32 v13, v92
	ds_load_b32 v12, v93
	ds_load_b32 v10, v94
	ds_load_b32 v11, v95
	s_and_saveexec_b32 s6, s40
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[19:20], null, v74, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[20:21], null, v74, s51, v[20:21]
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v19, vcc_lo, v158, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v159, v20, vcc_lo
	global_load_b32 v14, v[19:20], off
.LBB0_52:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v1, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v78
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v1, 0x80, v97
	ds_load_2addr_stride64_b32 v[20:21], v1 offset0:113 offset1:114
	s_wait_dscnt 0x8
	v_add_f32_e32 v1, v2, v18
	s_wait_loadcnt_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v1, -v1, v20, v14
	v_mul_f32_e32 v1, v21, v1
.LBB0_54:                               ; %.critedge
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v73, s69, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s39, s56, v73
	s_and_saveexec_b32 s6, s39
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x7
	v_mad_co_u64_u32 v[18:19], null, v73, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[19:20], null, v73, s51, v[19:20]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc_lo, v158, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v159, v19, vcc_lo
	global_load_b32 v19, v[18:19], off
.LBB0_56:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v2, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v117
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v2, 0x84, v97
	ds_load_2addr_stride64_b32 v[20:21], v2 offset0:113 offset1:114
	s_wait_dscnt 0x7
	v_add_f32_e32 v2, v3, v17
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v2, v20, v19
	v_mul_f32_e32 v2, v21, v2
.LBB0_58:                               ; %.critedge.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v66, s69, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s38, s56, v66
	s_and_saveexec_b32 s6, s38
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x6
	v_mad_co_u64_u32 v[17:18], null, v66, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[18:19], null, v66, s51, v[18:19]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v158, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v159, v18, vcc_lo
	global_load_b32 v14, v[17:18], off
.LBB0_60:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x6
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v3, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v118
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v3, 0x88, v97
	ds_load_2addr_stride64_b32 v[18:19], v3 offset0:113 offset1:114
	s_wait_dscnt 0x6
	v_add_f32_e32 v3, v4, v16
	s_wait_loadcnt_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v3, v18, v14
	v_mul_f32_e32 v3, v19, v3
.LBB0_62:                               ; %.critedge.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v65, s69, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s37, s56, v65
	s_and_saveexec_b32 s6, s37
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x5
	v_mad_co_u64_u32 v[16:17], null, v65, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[17:18], null, v65, s51, v[17:18]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, v158, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v159, v17, vcc_lo
	global_load_b32 v17, v[16:17], off
.LBB0_64:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v119
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v4, 0x8c, v97
	ds_load_2addr_stride64_b32 v[18:19], v4 offset0:113 offset1:114
	s_wait_dscnt 0x5
	v_add_f32_e32 v4, v5, v15
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v4, -v4, v18, v17
	v_mul_f32_e32 v4, v19, v4
.LBB0_66:                               ; %.critedge.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v71, s69, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s36, s56, v71
	s_and_saveexec_b32 s6, s36
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x4
	v_mad_co_u64_u32 v[14:15], null, v71, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[15:16], null, v71, s51, v[15:16]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, vcc_lo, v158, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, v159, v15, vcc_lo
	global_load_b32 v14, v[14:15], off
.LBB0_68:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x4
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v5, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v120
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v5, 0x90, v97
	ds_load_2addr_stride64_b32 v[16:17], v5 offset0:113 offset1:114
	s_wait_dscnt 0x4
	v_add_f32_e32 v5, v6, v13
	s_wait_loadcnt_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v5, v16, v14
	v_mul_f32_e32 v5, v17, v5
.LBB0_70:                               ; %.critedge.4
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v70, s69, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s35, s56, v70
	s_and_saveexec_b32 s6, s35
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x3
	v_mad_co_u64_u32 v[13:14], null, v70, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[14:15], null, v70, s51, v[14:15]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, vcc_lo, v158, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v159, v14, vcc_lo
	global_load_b32 v15, v[13:14], off
.LBB0_72:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x3
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v6, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v121
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v6, 0x94, v97
	ds_load_2addr_stride64_b32 v[16:17], v6 offset0:113 offset1:114
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v7, v12
	s_wait_loadcnt_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v6, v16, v15
	v_mul_f32_e32 v6, v17, v6
.LBB0_74:                               ; %.critedge.5
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v69, s69, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s34, s56, v69
	s_and_saveexec_b32 s6, s34
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x2
	v_mad_co_u64_u32 v[12:13], null, v69, s50, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[13:14], null, v69, s51, v[13:14]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v158, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v159, v13, vcc_lo
	global_load_b32 v13, v[12:13], off
.LBB0_76:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x2
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v7, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v122
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v7, 0x98, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[14:15], v7 offset0:113 offset1:114
	s_wait_dscnt 0x2
	v_add_f32_e32 v7, v8, v10
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v7, -v7, v14, v13
	v_mul_f32_e32 v7, v15, v7
.LBB0_78:                               ; %.critedge.6
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v77, s69, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s33, s56, v77
	s_and_saveexec_b32 s6, s33
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[12:13], null, v77, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[13:14], null, v77, s51, v[13:14]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v158, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v159, v13, vcc_lo
	global_load_b32 v12, v[12:13], off
.LBB0_80:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x1
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v8, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v123
	s_cbranch_execz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v8, 0x9c, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[13:14], v8 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v8, v9, v11
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v8, -v8, v13, v12
	v_mul_f32_e32 v8, v14, v8
.LBB0_82:                               ; %.critedge.7
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_add3_u32 v76, v112, s69, 16
	s_wait_loadcnt_dscnt 0x0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v14, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s31, s56, v76
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v16, 0
	v_mov_b32_e32 v17, 0
	s_and_saveexec_b32 s6, s31
	s_cbranch_execz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[9:10], null, v76, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[10:11], null, v76, s51, v[10:11]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v152, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v153, v10, vcc_lo
	s_clause 0x1
	global_load_b128 v[14:17], v[9:10], off
	global_load_b128 v[10:13], v[9:10], off offset:16
.LBB0_84:                               ; %.preheader949.11199
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v13.h, v13
	v_cvt_f16_f32_e32 v13.l, v12
	v_cvt_f16_f32_e32 v12.h, v11
	v_cvt_f16_f32_e32 v12.l, v10
	v_cvt_f16_f32_e32 v11.h, v17
	v_cvt_f16_f32_e32 v11.l, v16
	v_cvt_f16_f32_e32 v10.h, v15
	v_cvt_f16_f32_e32 v10.l, v14
	v_or_b32_e32 v191, s69, v124
	v_mov_b32_e32 v9, 0
	ds_store_b128 v83, v[10:13] offset:24704
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_i32_e64 s23, s56, v191
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[18:21], v133 offset:24704
	ds_load_b128 v[22:25], v133 offset:24736
	ds_load_b128 v[26:29], v133 offset:24768
	ds_load_b128 v[30:33], v133 offset:24800
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[10:17], v[18:21], v[57:60], 0
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[10:17], v[22:25], v[53:56], v[10:17]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[10:17], v[26:29], v[49:52], v[10:17]
	v_mov_b32_e32 v26, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[10:17], v[30:33], v[61:64], v[10:17]
	ds_store_b128 v165, v[10:13]
	ds_store_b128 v165, v[14:17] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v25, v88
	ds_load_b32 v24, v89
	ds_load_b32 v23, v90
	ds_load_b32 v22, v91
	ds_load_b32 v21, v92
	ds_load_b32 v20, v93
	ds_load_b32 v19, v94
	ds_load_b32 v18, v95
	s_and_saveexec_b32 s6, s23
	s_cbranch_execz .LBB0_86
; %bb.85:                               ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[26:27], null, v191, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[27:28], null, v191, s51, v[27:28]
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, v158, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v159, v27, vcc_lo
	global_load_b32 v26, v[26:27], off
.LBB0_86:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v124
	s_cbranch_execz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v9, 0xc0, v97
	ds_load_2addr_stride64_b32 v[27:28], v9 offset0:113 offset1:114
	s_wait_dscnt 0x8
	v_add_f32_e32 v9, v10, v25
	s_wait_loadcnt_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v9, -v9, v27, v26
	v_mul_f32_e32 v9, v28, v9
.LBB0_88:                               ; %.critedge.11202
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v185, s69, v125
	s_wait_dscnt 0x7
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v25, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s24, s56, v185
	s_and_saveexec_b32 s6, s24
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[25:26], null, v185, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[26:27], null, v185, s51, v[26:27]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v158, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v159, v26, vcc_lo
	global_load_b32 v25, v[25:26], off
.LBB0_90:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v125
	s_cbranch_execz .LBB0_92
; %bb.91:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v10, 0xc4, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[26:27], v10 offset0:113 offset1:114
	s_wait_dscnt 0x7
	v_add_f32_e32 v10, v11, v24
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v10, -v10, v26, v25
	v_mul_f32_e32 v10, v27, v10
.LBB0_92:                               ; %.critedge.1.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v186, s69, v126
	s_wait_dscnt 0x6
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v24, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s25, s56, v186
	s_and_saveexec_b32 s6, s25
	s_cbranch_execz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[24:25], null, v186, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[25:26], null, v186, s51, v[25:26]
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v24, vcc_lo, v158, v24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v25, null, v159, v25, vcc_lo
	global_load_b32 v24, v[24:25], off
.LBB0_94:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v126
	s_cbranch_execz .LBB0_96
; %bb.95:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v11, 0xc8, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[25:26], v11 offset0:113 offset1:114
	s_wait_dscnt 0x6
	v_add_f32_e32 v11, v12, v23
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v11, -v11, v25, v24
	v_mul_f32_e32 v11, v26, v11
.LBB0_96:                               ; %.critedge.2.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v187, s69, v127
	s_wait_dscnt 0x5
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v23, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s26, s56, v187
	s_and_saveexec_b32 s6, s26
	s_cbranch_execz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[23:24], null, v187, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[24:25], null, v187, s51, v[24:25]
	v_lshlrev_b64_e32 v[23:24], 2, v[23:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v23, vcc_lo, v158, v23
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, v159, v24, vcc_lo
	global_load_b32 v23, v[23:24], off
.LBB0_98:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v127
	s_cbranch_execz .LBB0_100
; %bb.99:                               ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v12, 0xcc, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[24:25], v12 offset0:113 offset1:114
	s_wait_dscnt 0x5
	v_add_f32_e32 v12, v13, v22
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v12, -v12, v24, v23
	v_mul_f32_e32 v12, v25, v12
.LBB0_100:                              ; %.critedge.3.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v188, s69, v128
	s_wait_dscnt 0x4
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v22, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s27, s56, v188
	s_and_saveexec_b32 s6, s27
	s_cbranch_execz .LBB0_102
; %bb.101:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[22:23], null, v188, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[23:24], null, v188, s51, v[23:24]
	v_lshlrev_b64_e32 v[22:23], 2, v[22:23]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v22, vcc_lo, v158, v22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v23, null, v159, v23, vcc_lo
	global_load_b32 v22, v[22:23], off
.LBB0_102:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v128
	s_cbranch_execz .LBB0_104
; %bb.103:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v13, 0xd0, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[23:24], v13 offset0:113 offset1:114
	s_wait_dscnt 0x4
	v_add_f32_e32 v13, v14, v21
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v13, v23, v22
	v_mul_f32_e32 v13, v24, v13
.LBB0_104:                              ; %.critedge.4.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v189, s69, v129
	s_wait_dscnt 0x3
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v21, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s28, s56, v189
	s_and_saveexec_b32 s6, s28
	s_cbranch_execz .LBB0_106
; %bb.105:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[21:22], null, v189, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[22:23], null, v189, s51, v[22:23]
	v_lshlrev_b64_e32 v[21:22], 2, v[21:22]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v21, vcc_lo, v158, v21
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, v159, v22, vcc_lo
	global_load_b32 v21, v[21:22], off
.LBB0_106:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v129
	s_cbranch_execz .LBB0_108
; %bb.107:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v14, 0xd4, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[22:23], v14 offset0:113 offset1:114
	s_wait_dscnt 0x3
	v_add_f32_e32 v14, v15, v20
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v14, v22, v21
	v_mul_f32_e32 v14, v23, v14
.LBB0_108:                              ; %.critedge.5.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v190, s69, v130
	s_wait_dscnt 0x2
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v20, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s29, s56, v190
	s_and_saveexec_b32 s6, s29
	s_cbranch_execz .LBB0_110
; %bb.109:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[20:21], null, v190, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[21:22], null, v190, s51, v[21:22]
	v_lshlrev_b64_e32 v[20:21], 2, v[20:21]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, v158, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, v159, v21, vcc_lo
	global_load_b32 v20, v[20:21], off
.LBB0_110:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v130
	s_cbranch_execz .LBB0_112
; %bb.111:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v15, 0xd8, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[21:22], v15 offset0:113 offset1:114
	s_wait_dscnt 0x2
	v_add_f32_e32 v15, v16, v19
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v15, v21, v20
	v_mul_f32_e32 v15, v22, v15
.LBB0_112:                              ; %.critedge.6.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v184, s69, v131
	s_wait_dscnt 0x1
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v19, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s30, s56, v184
	s_and_saveexec_b32 s6, s30
	s_cbranch_execz .LBB0_114
; %bb.113:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[19:20], null, v184, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[20:21], null, v184, s51, v[20:21]
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v19, vcc_lo, v158, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, v159, v20, vcc_lo
	global_load_b32 v19, v[19:20], off
.LBB0_114:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v131
	s_cbranch_execz .LBB0_116
; %bb.115:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v16, 0xdc, v97
	s_wait_loadcnt 0x0
	ds_load_2addr_stride64_b32 v[20:21], v16 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v16, v17, v18
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v16, v20, v19
	v_mul_f32_e32 v16, v21, v16
.LBB0_116:                              ; %.critedge.7.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v22, 0 :: v_dual_add_nc_u32 v183, 16, v76
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v24, 0
	s_wait_dscnt 0x0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v20, 0
	v_cmp_gt_i32_e64 s22, s56, v183
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v19, 0
	s_and_saveexec_b32 s6, s22
	s_cbranch_execz .LBB0_118
; %bb.117:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[17:18], null, v183, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[18:19], null, v183, s51, v[18:19]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v21, vcc_lo, v152, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, v153, v18, vcc_lo
	s_clause 0x1
	global_load_b128 v[17:20], v[21:22], off
	global_load_b128 v[21:24], v[21:22], off offset:16
.LBB0_118:                              ; %.preheader949.21203
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v24.h, v24
	v_cvt_f16_f32_e32 v24.l, v23
	v_cvt_f16_f32_e32 v23.h, v22
	v_cvt_f16_f32_e32 v23.l, v21
	v_cvt_f16_f32_e32 v22.h, v20
	v_cvt_f16_f32_e32 v22.l, v19
	v_cvt_f16_f32_e32 v21.h, v18
	v_cvt_f16_f32_e32 v21.l, v17
	v_or_b32_e32 v174, s69, v104
	ds_store_b128 v83, v[21:24] offset:24704
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_i32_e64 s13, s56, v174
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[26:29], v133 offset:24704
	ds_load_b128 v[32:35], v133 offset:24736
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[18:25], v[26:29], v[57:60], 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[18:25], v[32:35], v[53:56], v[18:25]
	ds_load_b128 v[26:29], v133 offset:24768
	ds_load_b128 v[32:35], v133 offset:24800
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[18:25], v[26:29], v[49:52], v[18:25]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[18:25], v[32:35], v[61:64], v[18:25]
	ds_store_b128 v165, v[18:21]
	ds_store_b128 v165, v[22:25] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v35, v88
	ds_load_b32 v33, v89
	ds_load_b32 v32, v90
	ds_load_b32 v30, v91
	ds_load_b32 v29, v92
	ds_load_b32 v28, v93
	ds_load_b32 v26, v94
	ds_load_b32 v27, v95
	s_and_saveexec_b32 s6, s13
	s_cbranch_execz .LBB0_120
; %bb.119:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[36:37], null, v174, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[37:38], null, v174, s51, v[37:38]
	v_lshlrev_b64_e32 v[36:37], 2, v[36:37]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v36, vcc_lo, v158, v36
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v37, null, v159, v37, vcc_lo
	global_load_b32 v31, v[36:37], off
.LBB0_120:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v17, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v104
	s_cbranch_execz .LBB0_122
; %bb.121:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x7
	v_add_f32_e32 v35, v18, v35
	ds_load_2addr_stride64_b32 v[17:18], v97 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v17, -v35, v17, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, v18, v17
.LBB0_122:                              ; %.critedge.21206
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v175, s69, v157
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s14, s56, v175
	s_and_saveexec_b32 s6, s14
	s_cbranch_execz .LBB0_124
; %bb.123:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x7
	v_mad_co_u64_u32 v[34:35], null, v175, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[35:36], null, v175, s51, v[35:36]
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v34, vcc_lo, v158, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v159, v35, vcc_lo
	global_load_b32 v34, v[34:35], off
.LBB0_124:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v18, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v157
	s_cbranch_execz .LBB0_126
; %bb.125:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v18, 4, v97
	s_wait_dscnt 0x6
	v_add_f32_e32 v33, v19, v33
	ds_load_2addr_stride64_b32 v[18:19], v18 offset0:114 offset1:115
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v33, v18, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v18, v19, v18
.LBB0_126:                              ; %.critedge.1.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v176, s69, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s15, s56, v176
	s_and_saveexec_b32 s6, s15
	s_cbranch_execz .LBB0_128
; %bb.127:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x6
	v_mad_co_u64_u32 v[33:34], null, v176, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[34:35], null, v176, s51, v[34:35]
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v33, vcc_lo, v158, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v159, v34, vcc_lo
	global_load_b32 v31, v[33:34], off
.LBB0_128:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x6
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v19, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v108
	s_cbranch_execz .LBB0_130
; %bb.129:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x5
	v_dual_add_f32 v32, v20, v32 :: v_dual_add_nc_u32 v19, 8, v97
	ds_load_2addr_stride64_b32 v[19:20], v19 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v19, -v32, v19, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v19, v20, v19
.LBB0_130:                              ; %.critedge.2.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v177, s69, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s16, s56, v177
	s_and_saveexec_b32 s6, s16
	s_cbranch_execz .LBB0_132
; %bb.131:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x5
	v_mad_co_u64_u32 v[31:32], null, v177, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[32:33], null, v177, s51, v[32:33]
	v_lshlrev_b64_e32 v[31:32], 2, v[31:32]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v31, vcc_lo, v158, v31
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v32, null, v159, v32, vcc_lo
	global_load_b32 v33, v[31:32], off
.LBB0_132:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v20, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v113
	s_cbranch_execz .LBB0_134
; %bb.133:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v20, 12, v97
	s_wait_dscnt 0x4
	v_add_f32_e32 v30, v21, v30
	ds_load_2addr_stride64_b32 v[20:21], v20 offset0:114 offset1:115
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v30, v20, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v20, v21, v20
.LBB0_134:                              ; %.critedge.3.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v178, s69, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s17, s56, v178
	s_and_saveexec_b32 s6, s17
	s_cbranch_execz .LBB0_136
; %bb.135:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x4
	v_mad_co_u64_u32 v[30:31], null, v178, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[31:32], null, v178, s51, v[31:32]
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v30, vcc_lo, v158, v30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v31, null, v159, v31, vcc_lo
	global_load_b32 v31, v[30:31], off
.LBB0_136:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x4
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v21, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v67
	s_cbranch_execz .LBB0_138
; %bb.137:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v21, 16, v97
	s_wait_dscnt 0x3
	v_add_f32_e32 v29, v22, v29
	ds_load_2addr_stride64_b32 v[21:22], v21 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v21, -v29, v21, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, v22, v21
.LBB0_138:                              ; %.critedge.4.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v179, s69, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s18, s56, v179
	s_and_saveexec_b32 s6, s18
	s_cbranch_execz .LBB0_140
; %bb.139:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x3
	v_mad_co_u64_u32 v[29:30], null, v179, s50, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[30:31], null, v179, s51, v[30:31]
	v_lshlrev_b64_e32 v[29:30], 2, v[29:30]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, v158, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v159, v30, vcc_lo
	global_load_b32 v30, v[29:30], off
.LBB0_140:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x3
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v22, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v109
	s_cbranch_execz .LBB0_142
; %bb.141:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v22, 20, v97
	s_wait_dscnt 0x2
	v_add_f32_e32 v28, v23, v28
	ds_load_2addr_stride64_b32 v[22:23], v22 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v22, -v28, v22, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, v23, v22
.LBB0_142:                              ; %.critedge.5.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v180, s69, v154
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s20, s56, v180
	s_and_saveexec_b32 s6, s20
	s_cbranch_execz .LBB0_144
; %bb.143:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x2
	v_mad_co_u64_u32 v[28:29], null, v180, s50, 0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[29:30], null, v180, s51, v[29:30]
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v158, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v159, v29, vcc_lo
	global_load_b32 v29, v[28:29], off
.LBB0_144:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x2
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v23, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v154
	s_cbranch_execz .LBB0_146
; %bb.145:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x1
	v_dual_add_f32 v26, v24, v26 :: v_dual_add_nc_u32 v23, 24, v97
	ds_load_2addr_stride64_b32 v[23:24], v23 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v23, -v26, v23, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, v24, v23
.LBB0_146:                              ; %.critedge.6.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v181, s69, v155
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s21, s56, v181
	s_and_saveexec_b32 s6, s21
	s_cbranch_execz .LBB0_148
; %bb.147:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[28:29], null, v181, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[29:30], null, v181, s51, v[29:30]
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, v158, v28
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, v159, v29, vcc_lo
	global_load_b32 v28, v[28:29], off
.LBB0_148:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x1
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v24, 0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_i32_e64 s58, v155
	s_cbranch_execz .LBB0_150
; %bb.149:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x0
	v_dual_add_f32 v27, v25, v27 :: v_dual_add_nc_u32 v24, 28, v97
	ds_load_2addr_stride64_b32 v[24:25], v24 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v24, -v27, v24, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, v25, v24
.LBB0_150:                              ; %.critedge.7.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_dscnt 0x0
	v_dual_mov_b32 v27, 0 :: v_dual_add_nc_u32 v182, 32, v76
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s19, s56, v182
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v33, 0
	s_and_saveexec_b32 s6, s19
	s_cbranch_execz .LBB0_152
; %bb.151:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[25:26], null, v182, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[26:27], null, v182, s51, v[26:27]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v152, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v153, v26, vcc_lo
	s_clause 0x1
	global_load_b128 v[30:33], v[25:26], off
	global_load_b128 v[26:29], v[25:26], off offset:16
.LBB0_152:                              ; %.preheader949.31207
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v29.h, v29
	v_cvt_f16_f32_e32 v29.l, v28
	v_cvt_f16_f32_e32 v28.h, v27
	v_cvt_f16_f32_e32 v28.l, v26
	v_cvt_f16_f32_e32 v27.h, v33
	v_cvt_f16_f32_e32 v27.l, v32
	v_cvt_f16_f32_e32 v26.h, v31
	v_cvt_f16_f32_e32 v26.l, v30
	v_or_b32_e32 v45, s69, v103
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v25, 0
	ds_store_b128 v83, v[26:29] offset:24704
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_i32_e32 vcc_lo, s56, v45
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[34:37], v133 offset:24704
	ds_load_b128 v[38:41], v133 offset:24736
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[26:33], v[34:37], v[57:60], 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[26:33], v[38:41], v[53:56], v[26:33]
	ds_load_b128 v[34:37], v133 offset:24768
	ds_load_b128 v[38:41], v133 offset:24800
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[26:33], v[34:37], v[49:52], v[26:33]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[26:33], v[38:41], v[61:64], v[26:33]
	ds_store_b128 v165, v[26:29]
	ds_store_b128 v165, v[30:33] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v41, v88
	ds_load_b32 v40, v89
	ds_load_b32 v39, v90
	ds_load_b32 v38, v91
	ds_load_b32 v37, v92
	ds_load_b32 v36, v93
	ds_load_b32 v35, v94
	ds_load_b32 v34, v95
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB0_154
; %bb.153:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[42:43], null, v45, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[43:44], null, v45, s51, v[43:44]
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, s6, v158, v42
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v43, null, v159, v43, s6
	global_load_b32 v42, v[42:43], off
.LBB0_154:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s7, exec_lo
	v_cmpx_gt_i32_e64 s58, v103
	s_cbranch_execz .LBB0_156
; %bb.155:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v25, 64, v97
	s_wait_dscnt 0x7
	v_add_f32_e32 v41, v26, v41
	ds_load_2addr_stride64_b32 v[25:26], v25 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v25, -v41, v25, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, v26, v25
.LBB0_156:                              ; %.critedge.31210
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_or_b32_e32 v167, s69, v166
	s_wait_dscnt 0x7
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v41, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s6, s56, v167
	s_and_saveexec_b32 s8, s6
	s_cbranch_execz .LBB0_158
; %bb.157:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[41:42], null, v167, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[42:43], null, v167, s51, v[42:43]
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v41, s7, v158, v41
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v42, null, v159, v42, s7
	global_load_b32 v41, v[41:42], off
.LBB0_158:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s8, exec_lo
	v_cmpx_gt_i32_e64 s58, v166
	s_cbranch_execz .LBB0_160
; %bb.159:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v26, 0x44, v97
	s_wait_dscnt 0x6
	v_add_f32_e32 v40, v27, v40
	ds_load_2addr_stride64_b32 v[26:27], v26 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v26, -v40, v26, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, v27, v26
.LBB0_160:                              ; %.critedge.1.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v168, s69, v79
	s_wait_dscnt 0x6
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v40, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s7, s56, v168
	s_and_saveexec_b32 s9, s7
	s_cbranch_execz .LBB0_162
; %bb.161:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[40:41], null, v168, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[41:42], null, v168, s51, v[41:42]
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v40, s8, v158, v40
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v41, null, v159, v41, s8
	global_load_b32 v40, v[40:41], off
.LBB0_162:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s9, exec_lo
	v_cmpx_gt_i32_e64 s58, v79
	s_cbranch_execz .LBB0_164
; %bb.163:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v27, 0x48, v97
	s_wait_dscnt 0x5
	v_add_f32_e32 v39, v28, v39
	ds_load_2addr_stride64_b32 v[27:28], v27 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v27, -v39, v27, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v27, v28, v27
.LBB0_164:                              ; %.critedge.2.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v169, s69, v82
	s_wait_dscnt 0x5
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v39, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s8, s56, v169
	s_and_saveexec_b32 s10, s8
	s_cbranch_execz .LBB0_166
; %bb.165:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[39:40], null, v169, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[40:41], null, v169, s51, v[40:41]
	v_lshlrev_b64_e32 v[39:40], 2, v[39:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v39, s9, v158, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v40, null, v159, v40, s9
	global_load_b32 v39, v[39:40], off
.LBB0_166:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s10, exec_lo
	v_cmpx_gt_i32_e64 s58, v82
	s_cbranch_execz .LBB0_168
; %bb.167:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v28, 0x4c, v97
	s_wait_dscnt 0x4
	v_add_f32_e32 v38, v29, v38
	ds_load_2addr_stride64_b32 v[28:29], v28 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v28, -v38, v28, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v28, v29, v28
.LBB0_168:                              ; %.critedge.3.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v170, s69, v84
	s_wait_dscnt 0x4
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v38, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s9, s56, v170
	s_and_saveexec_b32 s11, s9
	s_cbranch_execz .LBB0_170
; %bb.169:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[38:39], null, v170, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[39:40], null, v170, s51, v[39:40]
	v_lshlrev_b64_e32 v[38:39], 2, v[38:39]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v38, s10, v158, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v39, null, v159, v39, s10
	global_load_b32 v38, v[38:39], off
.LBB0_170:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s11, exec_lo
	v_cmpx_gt_i32_e64 s58, v84
	s_cbranch_execz .LBB0_172
; %bb.171:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v29, 0x50, v97
	s_wait_dscnt 0x3
	v_add_f32_e32 v37, v30, v37
	ds_load_2addr_stride64_b32 v[29:30], v29 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v29, -v37, v29, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v29, v30, v29
.LBB0_172:                              ; %.critedge.4.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v171, s69, v98
	s_wait_dscnt 0x3
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v37, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s10, s56, v171
	s_and_saveexec_b32 s12, s10
	s_cbranch_execz .LBB0_174
; %bb.173:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[37:38], null, v171, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[38:39], null, v171, s51, v[38:39]
	v_lshlrev_b64_e32 v[37:38], 2, v[37:38]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v37, s11, v158, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v38, null, v159, v38, s11
	global_load_b32 v37, v[37:38], off
.LBB0_174:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e64 s58, v98
	s_cbranch_execz .LBB0_176
; %bb.175:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v30, 0x54, v97
	s_wait_dscnt 0x2
	v_add_f32_e32 v36, v31, v36
	ds_load_2addr_stride64_b32 v[30:31], v30 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v30, -v36, v30, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v30, v31, v30
.LBB0_176:                              ; %.critedge.5.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_or_b32_e32 v172, s69, v99
	s_wait_dscnt 0x2
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v36, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s11, s56, v172
	s_and_saveexec_b32 s42, s11
	s_cbranch_execz .LBB0_178
; %bb.177:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[36:37], null, v172, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[37:38], null, v172, s51, v[37:38]
	v_lshlrev_b64_e32 v[36:37], 2, v[36:37]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v36, s12, v158, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v37, null, v159, v37, s12
	global_load_b32 v36, v[36:37], off
.LBB0_178:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s42
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s42, exec_lo
	v_cmpx_gt_i32_e64 s58, v99
	s_cbranch_execz .LBB0_180
; %bb.179:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v31, 0x58, v97
	s_wait_dscnt 0x1
	v_add_f32_e32 v35, v32, v35
	ds_load_2addr_stride64_b32 v[31:32], v31 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v31, -v35, v31, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v31, v32, v31
.LBB0_180:                              ; %.critedge.6.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s42
	v_or_b32_e32 v173, s69, v102
	s_wait_dscnt 0x1
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v35, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s12, s56, v173
	s_and_saveexec_b32 s43, s12
	s_cbranch_execz .LBB0_182
; %bb.181:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[35:36], null, v173, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[36:37], null, v173, s51, v[36:37]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s42, v158, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, v159, v36, s42
	global_load_b32 v35, v[35:36], off
.LBB0_182:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s43
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s43, exec_lo
	v_cmpx_gt_i32_e64 s58, v102
	s_cbranch_execz .LBB0_184
; %bb.183:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_nc_u32_e32 v32, 0x5c, v97
	s_wait_dscnt 0x0
	v_add_f32_e32 v34, v33, v34
	ds_load_2addr_stride64_b32 v[32:33], v32 offset0:114 offset1:115
	s_wait_loadcnt_dscnt 0x0
	v_fma_f32 v32, -v34, v32, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v32, v33, v32
.LBB0_184:                              ; %.critedge.7.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s43
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s43, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_185:                              ; %.preheader958
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e64 s42, s43, v96
	v_or_b32_e32 v34, 0x100, v96
	v_lshlrev_b32_e32 v36, 2, v78
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, 0, v1, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x200, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v2, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x300, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v3, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x400, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v4, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x500, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v5, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x600, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v6, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x700, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v7, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x1000, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v8, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x1100, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v9, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x1200, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v10, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x1300, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v11, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x1400, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v12, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x1500, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v13, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x1600, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v14, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x1700, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v15, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x2000, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v16, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x2100, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v17, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x2200, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v18, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x2300, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v19, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x2400, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v20, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x2500, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v21, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x2600, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v22, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x2700, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v23, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x3000, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v24, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x3100, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v25, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x3200, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v26, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x3300, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v27, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x3400, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v28, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x3500, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v29, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x3600, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v30, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	v_or_b32_e32 v34, 0x3700, v96
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v33, v33, v31, s42
	v_cmp_eq_u32_e64 s42, s43, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v33, v33, v32, s42
	ds_bpermute_b32 v34, v135, v33
	s_wait_dscnt 0x0
	v_dual_add_f32 v33, v33, v34 :: v_dual_lshlrev_b32 v34, 2, v72
	v_add_nc_u32_e32 v34, s43, v34
	s_addk_co_i32 s43, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s43, 0x4000
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, 0x2000, v34
	ds_load_2addr_b32 v[34:35], v34 offset0:32 offset1:64
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v36, v34
	s_wait_dscnt 0x0
	v_fma_f32 v1, -v33, v37, v1
	v_lshlrev_b32_e32 v37, 2, v117
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v2, -v33, v37, v2
	v_lshlrev_b32_e32 v37, 2, v118
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v3, -v33, v37, v3
	v_lshlrev_b32_e32 v37, 2, v119
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v4, -v33, v37, v4
	v_lshlrev_b32_e32 v37, 2, v120
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v5, -v33, v37, v5
	v_lshlrev_b32_e32 v37, 2, v121
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v6, -v33, v37, v6
	v_lshlrev_b32_e32 v37, 2, v122
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v7, -v33, v37, v7
	v_lshlrev_b32_e32 v37, 2, v123
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v8, -v33, v37, v8
	v_lshlrev_b32_e32 v37, 2, v124
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v33, v37, v9
	v_lshlrev_b32_e32 v37, 2, v125
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v33, v37, v10
	v_lshlrev_b32_e32 v37, 2, v126
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v33, v37, v11
	v_lshlrev_b32_e32 v37, 2, v127
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v33, v37, v12
	v_lshlrev_b32_e32 v37, 2, v128
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v33, v37, v13
	v_lshlrev_b32_e32 v37, 2, v129
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v33, v37, v14
	v_lshlrev_b32_e32 v37, 2, v130
	ds_bpermute_b32 v37, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v33, v37, v15
	v_lshlrev_b32_e32 v37, 2, v131
	ds_bpermute_b32 v34, v37, v34
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v33, v34, v16
	ds_bpermute_b32 v34, v36, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v33, v34, v17
	ds_bpermute_b32 v34, v137, v35
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v33, v34, v18
	ds_bpermute_b32 v34, v138, v35
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v33, v34, v19
	ds_bpermute_b32 v34, v139, v35
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v33, v34, v20
	ds_bpermute_b32 v34, v140, v35
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v33, v34, v21
	ds_bpermute_b32 v34, v141, v35
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v33, v34, v22
	ds_bpermute_b32 v34, v142, v35
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v33, v34, v23
	ds_bpermute_b32 v34, v143, v35
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v33, v34, v24
	ds_bpermute_b32 v34, v144, v35
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v33, v34, v25
	ds_bpermute_b32 v34, v145, v35
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v33, v34, v26
	ds_bpermute_b32 v34, v146, v35
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v33, v34, v27
	ds_bpermute_b32 v34, v147, v35
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v33, v34, v28
	ds_bpermute_b32 v34, v148, v35
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v33, v34, v29
	ds_bpermute_b32 v34, v149, v35
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v33, v34, v30
	ds_bpermute_b32 v34, v150, v35
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v33, v34, v31
	ds_bpermute_b32 v34, v151, v35
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v33, v34, v32
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc1 .LBB0_185
; %bb.186:                              ; %.preheader965.preheader
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_mov_b32 s43, 0
	s_branch .LBB0_188
.LBB0_187:                              ;   in Loop: Header=BB0_188 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s54
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v40.h, v40
	v_cvt_f16_f32_e32 v40.l, v39
	v_cvt_f16_f32_e32 v39.h, v38
	v_cvt_f16_f32_e32 v39.l, v37
	v_cvt_f16_f32_e32 v38.h, v36
	v_cvt_f16_f32_e32 v38.l, v35
	v_cvt_f16_f32_e32 v37.h, v34
	v_cvt_f16_f32_e32 v37.l, v33
	v_lshl_add_u32 v33, s44, 1, v83
	s_add_co_i32 s43, s43, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s43, 4
	ds_store_b128 v33, v[37:40] offset:8320
	s_cbranch_scc1 .LBB0_190
.LBB0_188:                              ; %.preheader965
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s44, s43, 11
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v36, 0 :: v_dual_add_nc_u32 v33, s44, v80
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v37, 0
	s_mov_b32 s54, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v33, 7, v33
	v_mov_b32_e32 v40, 0
	v_dual_mov_b32 v38, 0 :: v_dual_add_nc_u32 v41, s69, v33
	v_mov_b32_e32 v33, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s56, v41
	s_cbranch_execz .LBB0_187
; %bb.189:                              ;   in Loop: Header=BB0_188 Depth=2
	v_mad_co_u64_u32 v[33:34], null, v41, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[34:35], null, v41, s51, v[34:35]
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v37, s42, v152, v33
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v38, null, v153, v34, s42
	s_clause 0x1
	global_load_b128 v[33:36], v[37:38], off
	global_load_b128 v[37:40], v[37:38], off offset:16
	s_branch .LBB0_187
.LBB0_190:                              ;   in Loop: Header=BB0_3 Depth=1
	scratch_load_b64 v[33:34], off, off offset:16 ; 8-byte Folded Reload
	scratch_store_b32 off, v45, off offset:8 ; 4-byte Folded Spill
	v_cmp_gt_i32_e64 s43, s58, v114
	s_mov_b32 s68, 0
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[45:46], null, s52, v164, v[33:34]
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[46:47], null, s53, v164, v[46:47]
	v_mov_b32_e32 v47, v85
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v33, off, off offset:12 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, s69, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s42, s56, v33
	s_branch .LBB0_193
.LBB0_191:                              ; %.loopexit956.loopexit
                                        ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s55
	ds_store_b32 v33, v34 offset:28
.LBB0_192:                              ; %Flow2757
                                        ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s54
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v47, 0x1000, v47
	s_add_co_i32 s68, s68, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s68, 4
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB0_214
.LBB0_193:                              ; %.preheader957
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_195 Depth 3
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v75, v47
	s_mov_b64 s[54:55], 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v34, v33 :: v_dual_mov_b32 v35, v33
	v_dual_mov_b32 v36, v33 :: v_dual_mov_b32 v37, v33
	v_dual_mov_b32 v38, v33 :: v_dual_mov_b32 v39, v33
	v_mov_b32_e32 v40, v33
	s_branch .LBB0_195
.LBB0_194:                              ; %.loopexit946
                                        ;   in Loop: Header=BB0_195 Depth=3
	s_or_b32 exec_lo, exec_lo, s70
	ds_load_b128 v[102:105], v75 offset:8320
	v_add_nc_u32_e32 v75, 32, v75
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[54:55], s[54:55], 64
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s54, 0x200
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[102:105], v[33:40]
	s_cbranch_scc1 .LBB0_197
.LBB0_195:                              ;   Parent Loop BB0_3 Depth=1
                                        ;     Parent Loop BB0_193 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0
	s_and_saveexec_b32 s70, s42
	s_cbranch_execz .LBB0_194
; %bb.196:                              ; %.loopexit946.loopexit
                                        ;   in Loop: Header=BB0_195 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v98, s44, v45, s54
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v99, null, s55, v46, s44
	s_clause 0x1
	global_load_b128 v[41:44], v[98:99], off offset:16
	global_load_b128 v[102:105], v[98:99], off
	s_wait_loadcnt 0x1
	v_cvt_f16_f32_e32 v44.h, v44
	v_cvt_f16_f32_e32 v44.l, v43
	v_cvt_f16_f32_e32 v43.h, v42
	v_cvt_f16_f32_e32 v43.l, v41
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v42.h, v105
	v_cvt_f16_f32_e32 v42.l, v104
	v_cvt_f16_f32_e32 v41.h, v103
	v_cvt_f16_f32_e32 v41.l, v102
	s_branch .LBB0_194
.LBB0_197:                              ;   in Loop: Header=BB0_193 Depth=2
	s_and_saveexec_b32 s54, s4
	s_cbranch_execz .LBB0_192
; %bb.198:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v41, v136 offset:28800
	v_lshl_or_b32 v43, s68, 4, v78
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v75, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s44, v43, v114
	v_lshlrev_b32_e32 v42, 2, v43
	s_and_b32 s44, s44, s43
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s55, s44
	s_cbranch_execz .LBB0_200
; %bb.199:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v75, v42 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v75, v41, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v79, 0x3fb8aa3b, v75
	v_cmp_ngt_f32_e64 s44, 0xc2ce8ed0, v75
	v_fma_f32 v82, 0x3fb8aa3b, v75, -v79
	v_rndne_f32_e32 v84, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v82, 0x32a5705f, v75 :: v_dual_sub_f32 v79, v79, v84
	v_add_f32_e32 v79, v79, v82
	v_cvt_i32_f32_e32 v82, v84
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v79, v79
	v_ldexp_f32 v79, v79, v82
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v79, 0, v79, s44
	v_cmp_nlt_f32_e64 s44, 0x42b17218, v75
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v75, 0x7f800000, v79, s44
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v75, v33, v75
.LBB0_200:                              ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s55
	v_cmp_lt_u32_e64 s44, v43, v114
	v_lshl_add_u32 v33, v43, 2, v162
	s_and_b32 s44, s44, s43
	ds_store_b32 v33, v75
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s55, s44
	s_cbranch_execz .LBB0_202
; %bb.201:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v44, v42 offset:28804
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v41, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v75, 0x3fb8aa3b, v44
	v_fma_f32 v79, 0x3fb8aa3b, v44, -v75
	v_rndne_f32_e32 v82, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v75, v75, v82
	v_fmac_f32_e32 v79, 0x32a5705f, v44
	v_cmp_ngt_f32_e64 s44, 0xc2ce8ed0, v44
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v75, v75, v79
	v_cvt_i32_f32_e32 v79, v82
	v_exp_f32_e32 v75, v75
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v75, v75, v79
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v75, 0, v75, s44
	v_cmp_nlt_f32_e64 s44, 0x42b17218, v44
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v44, 0x7f800000, v75, s44
	v_mul_f32_e32 v44, v34, v44
.LBB0_202:                              ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s55
	v_or_b32_e32 v34, 2, v43
	ds_store_b32 v33, v44 offset:4
	v_mov_b32_e32 v44, 0
	v_cmp_le_u32_e64 s44, v34, v114
	v_mov_b32_e32 v34, 0
	s_and_b32 s44, s44, s43
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s55, s44
	s_cbranch_execz .LBB0_204
; %bb.203:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v44, v42 offset:28808
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v41, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v75, 0x3fb8aa3b, v44
	v_fma_f32 v79, 0x3fb8aa3b, v44, -v75
	v_rndne_f32_e32 v82, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v75, v75, v82
	v_fmac_f32_e32 v79, 0x32a5705f, v44
	v_cmp_ngt_f32_e64 s44, 0xc2ce8ed0, v44
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v75, v75, v79
	v_cvt_i32_f32_e32 v79, v82
	v_exp_f32_e32 v75, v75
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v75, v75, v79
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v75, 0, v75, s44
	v_cmp_nlt_f32_e64 s44, 0x42b17218, v44
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v44, 0x7f800000, v75, s44
	v_mul_f32_e32 v44, v35, v44
.LBB0_204:                              ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s55
	v_or_b32_e32 v35, 3, v43
	ds_store_b32 v33, v44 offset:8
	v_cmp_le_u32_e64 s44, v35, v114
	s_and_b32 s44, s44, s43
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s55, s44
	s_cbranch_execz .LBB0_206
; %bb.205:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v34, v42 offset:28812
	s_wait_dscnt 0x0
	v_sub_f32_e32 v34, v41, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, 0x3fb8aa3b, v34
	v_fma_f32 v44, 0x3fb8aa3b, v34, -v35
	v_rndne_f32_e32 v75, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v35, v35, v75 :: v_dual_fmac_f32 v44, 0x32a5705f, v34
	v_cmp_ngt_f32_e64 s44, 0xc2ce8ed0, v34
	v_add_f32_e32 v35, v35, v44
	v_cvt_i32_f32_e32 v44, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v35, v35
	v_ldexp_f32 v35, v35, v44
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v35, 0, v35, s44
	v_cmp_nlt_f32_e64 s44, 0x42b17218, v34
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v34, 0x7f800000, v35, s44
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v34, v36, v34
.LBB0_206:                              ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s55
	ds_store_b32 v33, v34 offset:12
	v_or_b32_e32 v34, 4, v43
	v_mov_b32_e32 v35, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s44, v34, v114
	v_mov_b32_e32 v34, 0
	s_and_b32 s44, s44, s43
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s55, s44
	s_cbranch_execz .LBB0_208
; %bb.207:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v35, v42 offset:28816
	s_wait_dscnt 0x0
	v_sub_f32_e32 v35, v41, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, 0x3fb8aa3b, v35
	v_fma_f32 v44, 0x3fb8aa3b, v35, -v36
	v_rndne_f32_e32 v75, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, 0x32a5705f, v35
	v_sub_f32_e32 v36, v36, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v36, v36, v44
	v_cvt_i32_f32_e32 v44, v75
	v_cmp_ngt_f32_e64 s44, 0xc2ce8ed0, v35
	v_exp_f32_e32 v36, v36
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v36, v36, v44
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0, v36, s44
	v_cmp_nlt_f32_e64 s44, 0x42b17218, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v35, 0x7f800000, v36, s44
	v_mul_f32_e32 v35, v37, v35
.LBB0_208:                              ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s55
	ds_store_b32 v33, v35 offset:16
	v_or_b32_e32 v35, 5, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s44, v35, v114
	s_and_b32 s44, s44, s43
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s55, s44
	s_cbranch_execz .LBB0_210
; %bb.209:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v34, v42 offset:28820
	s_wait_dscnt 0x0
	v_sub_f32_e32 v34, v41, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, 0x3fb8aa3b, v34
	v_fma_f32 v36, 0x3fb8aa3b, v34, -v35
	v_rndne_f32_e32 v37, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v35, v35, v37 :: v_dual_fmac_f32 v36, 0x32a5705f, v34
	v_cmp_ngt_f32_e64 s44, 0xc2ce8ed0, v34
	v_add_f32_e32 v35, v35, v36
	v_cvt_i32_f32_e32 v36, v37
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v35, v35
	v_ldexp_f32 v35, v35, v36
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v35, 0, v35, s44
	v_cmp_nlt_f32_e64 s44, 0x42b17218, v34
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v34, 0x7f800000, v35, s44
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v34, v38, v34
.LBB0_210:                              ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s55
	ds_store_b32 v33, v34 offset:20
	v_or_b32_e32 v34, 6, v43
	v_mov_b32_e32 v35, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_le_u32_e64 s44, v34, v114
	v_mov_b32_e32 v34, 0
	s_and_b32 s44, s44, s43
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s55, s44
	s_cbranch_execz .LBB0_212
; %bb.211:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v35, v42 offset:28824
	s_wait_dscnt 0x0
	v_sub_f32_e32 v35, v41, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, 0x3fb8aa3b, v35
	v_fma_f32 v37, 0x3fb8aa3b, v35, -v36
	v_rndne_f32_e32 v38, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v36, v36, v38 :: v_dual_fmac_f32 v37, 0x32a5705f, v35
	v_cmp_ngt_f32_e64 s44, 0xc2ce8ed0, v35
	v_add_f32_e32 v36, v36, v37
	v_cvt_i32_f32_e32 v37, v38
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v36, v36
	v_ldexp_f32 v36, v36, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v36, 0, v36, s44
	v_cmp_nlt_f32_e64 s44, 0x42b17218, v35
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v35, 0x7f800000, v36, s44
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v35, v39, v35
.LBB0_212:                              ;   in Loop: Header=BB0_193 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s55
	ds_store_b32 v33, v35 offset:24
	v_or_b32_e32 v35, 7, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e64 s44, v35, v114
	s_and_b32 s44, s44, s43
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s55, s44
	s_cbranch_execz .LBB0_191
; %bb.213:                              ;   in Loop: Header=BB0_193 Depth=2
	ds_load_b32 v34, v42 offset:28828
	s_wait_dscnt 0x0
	v_sub_f32_e32 v34, v41, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, 0x3fb8aa3b, v34
	v_fma_f32 v36, 0x3fb8aa3b, v34, -v35
	v_rndne_f32_e32 v37, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v35, v35, v37 :: v_dual_fmac_f32 v36, 0x32a5705f, v34
	v_cmp_ngt_f32_e64 s44, 0xc2ce8ed0, v34
	v_add_f32_e32 v35, v35, v36
	v_cvt_i32_f32_e32 v36, v37
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v35, v35
	v_ldexp_f32 v35, v35, v36
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v35, 0, v35, s44
	v_cmp_nlt_f32_e64 s44, 0x42b17218, v34
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v34, 0x7f800000, v35, s44
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v34, v40, v34
	s_branch .LBB0_191
.LBB0_214:                              ; %.preheader964.preheader
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_mov_b32_e32 v33, v156
	s_movk_i32 s43, 0xff00
	s_branch .LBB0_216
.LBB0_215:                              ;   in Loop: Header=BB0_216 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s44
	v_add_nc_u32_e32 v35, s43, v115
	v_add_nc_u32_e32 v33, 4, v33
	s_add_co_i32 s43, s43, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s43, 0
	s_wait_dscnt 0x0
	ds_store_b32 v35, v34 offset:8576
	s_cbranch_scc1 .LBB0_218
.LBB0_216:                              ; %.preheader964
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v34, 0
	s_mov_b32 s44, exec_lo
	v_cmpx_le_u32_e64 v81, v33
	s_cbranch_execz .LBB0_215
; %bb.217:                              ;   in Loop: Header=BB0_216 Depth=2
	v_mul_lo_u32 v34, v33, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v34, v34, v33, 1
	v_and_b32_e32 v34, -4, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, v86, v34
	ds_load_b32 v34, v34
	s_branch .LBB0_215
.LBB0_218:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v111, v109 :: v_dual_mov_b32 v68, v67
	v_dual_mov_b32 v67, v108 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v38, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v39, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s42, s41
	s_cbranch_execz .LBB0_220
; %bb.219:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[34:35], null, v48, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[35:36], null, v48, s51, v[35:36]
	scratch_load_b32 v36, off, off          ; 4-byte Folded Reload
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	s_wait_loadcnt 0x0
	v_add_co_u32 v38, s41, v36, v34
	scratch_load_b32 v34, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v39, null, v34, v35, s41
	s_clause 0x1
	global_load_b128 v[34:37], v[38:39], off
	global_load_b128 v[38:41], v[38:39], off offset:16
.LBB0_220:                              ; %.preheader945
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s42
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v41.h, v41
	v_cvt_f16_f32_e32 v41.l, v40
	v_cvt_f16_f32_e32 v40.h, v39
	v_cvt_f16_f32_e32 v40.l, v38
	v_cvt_f16_f32_e32 v39.h, v37
	v_cvt_f16_f32_e32 v39.l, v36
	v_cvt_f16_f32_e32 v38.h, v35
	v_cvt_f16_f32_e32 v38.l, v34
	v_mov_b32_e32 v101, v116
	s_mov_b32 s41, 0
	ds_store_b128 v83, v[38:41] offset:24704
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v38, v33 :: v_dual_mov_b32 v39, v33
	v_mov_b32_e32 v40, v33
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[34:37], v133 offset:24704
	ds_load_b128 v[102:105], v133 offset:24736
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[41:48], v[34:37], v[57:60], 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[41:48], v[102:105], v[53:56], v[41:48]
	ds_load_b128 v[34:37], v133 offset:24768
	ds_load_b128 v[102:105], v133 offset:24800
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[41:48], v[34:37], v[49:52], v[41:48]
	v_dual_mov_b32 v34, v33 :: v_dual_mov_b32 v35, v33
	v_dual_mov_b32 v36, v33 :: v_dual_mov_b32 v37, v33
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[102:105], v[61:64], v[41:48]
	ds_store_b128 v165, v[41:44]
	ds_store_b128 v165, v[45:48] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v100, v88
	ds_load_b32 v99, v89
	ds_load_b32 v98, v90
	ds_load_b32 v132, v91
	ds_load_b32 v84, v92
	ds_load_b32 v82, v93
	ds_load_b32 v79, v94
	ds_load_b32 v75, v95
.LBB0_221:                              ; %.preheader944
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_nc_u32_e32 v106, 0x80, v116
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 m0, s41, 3
	v_add_nc_u32_e32 v116, 0x1000, v116
	v_movrels_b32_e32 v107, v2
	v_movrels_b32_e32 v108, v4
	ds_load_2addr_stride64_b32 v[102:103], v106 offset0:32 offset1:33
	v_movrels_b32_e32 v109, v6
	v_movrels_b32_e32 v110, v8
	s_add_co_i32 s41, s41, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s41, 4
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v102.l, v102
	v_cvt_f16_f32_e32 v102.h, v103
	ds_load_2addr_stride64_b32 v[103:104], v106 offset0:34 offset1:35
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v103.l, v103
	v_cvt_f16_f32_e32 v103.h, v104
	ds_load_2addr_stride64_b32 v[104:105], v106 offset0:36 offset1:37
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v104.l, v104
	v_cvt_f16_f32_e32 v104.h, v105
	ds_load_2addr_stride64_b32 v[105:106], v106 offset0:38 offset1:39
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v105.l, v105
	v_cvt_f16_f32_e32 v105.h, v106
	v_movrels_b32_e32 v106, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v106.l, v106
	v_cvt_f16_f32_e32 v106.h, v107
	v_movrels_b32_e32 v107, v3
	v_cvt_f16_f32_e32 v107.l, v107
	v_cvt_f16_f32_e32 v107.h, v108
	v_movrels_b32_e32 v108, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v108.l, v108
	v_cvt_f16_f32_e32 v108.h, v109
	v_movrels_b32_e32 v109, v7
	v_cvt_f16_f32_e32 v109.l, v109
	v_cvt_f16_f32_e32 v109.h, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[102:105], v[106:109], v[33:40]
	s_cbranch_scc1 .LBB0_221
; %bb.222:                              ;   in Loop: Header=BB0_3 Depth=1
	s_mov_b32 s41, exec_lo
	v_mov_b32_e32 v116, v101
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s42, s41, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 exec_lo, s42
	s_cbranch_execz .LBB0_232
; %bb.223:                              ; %.preheader952
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_and_saveexec_b32 s42, s40
	s_cbranch_execnz .LBB0_350
; %bb.224:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s42
	s_and_saveexec_b32 s40, s39
	s_cbranch_execnz .LBB0_351
.LBB0_225:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s40
	s_and_saveexec_b32 s39, s38
	s_cbranch_execnz .LBB0_352
.LBB0_226:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_and_saveexec_b32 s38, s37
	s_cbranch_execnz .LBB0_353
.LBB0_227:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s38
	s_and_saveexec_b32 s37, s36
	s_cbranch_execnz .LBB0_354
.LBB0_228:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s37
	s_and_saveexec_b32 s36, s35
	s_cbranch_execnz .LBB0_355
.LBB0_229:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s36
	s_and_saveexec_b32 s35, s34
	s_cbranch_execnz .LBB0_356
.LBB0_230:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s35
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s33
	s_cbranch_execz .LBB0_232
.LBB0_231:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29084
	v_add_f32_e32 v33, v48, v75
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v34
	v_mad_co_u64_u32 v[33:34], null, v77, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v77, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s33, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s33
	global_store_b32 v[33:34], v40, off
.LBB0_232:                              ; %Flow2756
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s41
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v38, 0
	v_mov_b32_e32 v39, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s33, s31
	s_cbranch_execz .LBB0_234
; %bb.233:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[34:35], null, v76, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[35:36], null, v76, s51, v[35:36]
	scratch_load_b32 v36, off, off          ; 4-byte Folded Reload
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	s_wait_loadcnt 0x0
	v_add_co_u32 v38, s31, v36, v34
	scratch_load_b32 v34, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v39, null, v34, v35, s31
	s_clause 0x1
	global_load_b128 v[34:37], v[38:39], off
	global_load_b128 v[38:41], v[38:39], off offset:16
.LBB0_234:                              ; %.preheader945.11234
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s33
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v41.h, v41
	v_cvt_f16_f32_e32 v41.l, v40
	v_cvt_f16_f32_e32 v40.h, v39
	v_cvt_f16_f32_e32 v40.l, v38
	v_cvt_f16_f32_e32 v39.h, v37
	v_cvt_f16_f32_e32 v39.l, v36
	v_cvt_f16_f32_e32 v38.h, v35
	v_cvt_f16_f32_e32 v38.l, v34
	s_mov_b32 s31, 0
	ds_store_b128 v83, v[38:41] offset:24704
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v38, v33 :: v_dual_mov_b32 v39, v33
	v_mov_b32_e32 v40, v33
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[34:37], v133 offset:24704
	ds_load_b128 v[73:76], v133 offset:24736
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[41:48], v[34:37], v[57:60], 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[53:56], v[41:48]
	ds_load_b128 v[34:37], v133 offset:24768
	ds_load_b128 v[73:76], v133 offset:24800
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[41:48], v[34:37], v[49:52], v[41:48]
	v_dual_mov_b32 v34, v33 :: v_dual_mov_b32 v35, v33
	v_dual_mov_b32 v36, v33 :: v_dual_mov_b32 v37, v33
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[61:64], v[41:48]
	ds_store_b128 v165, v[41:44]
	ds_store_b128 v165, v[45:48] offset:16
	v_mov_b32_e32 v76, v116
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v75, v88
	ds_load_b32 v74, v89
	ds_load_b32 v73, v90
	ds_load_b32 v71, v91
	ds_load_b32 v70, v92
	ds_load_b32 v69, v93
	ds_load_b32 v66, v94
	ds_load_b32 v65, v95
.LBB0_235:                              ; %.preheader944.1
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_nc_u32_e32 v77, 0xc0, v76
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 m0, s31, 3
	v_add_nc_u32_e32 v76, 0x1000, v76
	v_movrels_b32_e32 v79, v2
	s_add_co_i32 s31, s31, 1
	ds_load_2addr_stride64_b32 v[98:99], v77 offset0:32 offset1:33
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s31, 4
	v_cvt_f16_f32_e32 v106.h, v79
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v102.l, v98
	v_cvt_f16_f32_e32 v102.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v77 offset0:34 offset1:35
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v103.l, v98
	v_cvt_f16_f32_e32 v103.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v77 offset0:36 offset1:37
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v104.l, v98
	v_cvt_f16_f32_e32 v104.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v77 offset0:38 offset1:39
	v_movrels_b32_e32 v77, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v106.l, v77
	v_movrels_b32_e32 v77, v3
	v_cvt_f16_f32_e32 v107.l, v77
	v_movrels_b32_e32 v77, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v107.h, v77
	v_movrels_b32_e32 v77, v5
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v105.l, v98
	v_cvt_f16_f32_e32 v105.h, v99
	v_cvt_f16_f32_e32 v108.l, v77
	v_movrels_b32_e32 v77, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v108.h, v77
	v_movrels_b32_e32 v77, v7
	v_cvt_f16_f32_e32 v109.l, v77
	v_movrels_b32_e32 v77, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v109.h, v77
	v_wmma_f32_16x16x16_f16 v[33:40], v[102:105], v[106:109], v[33:40]
	s_cbranch_scc1 .LBB0_235
; %bb.236:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_saveexec_b32 s31, s5
	s_cbranch_execz .LBB0_246
; %bb.237:                              ; %.preheader952.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_and_saveexec_b32 s33, s23
	s_cbranch_execnz .LBB0_357
; %bb.238:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s33
	s_and_saveexec_b32 s33, s24
	s_cbranch_execnz .LBB0_358
.LBB0_239:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s33
	s_and_saveexec_b32 s24, s25
	s_cbranch_execnz .LBB0_359
.LBB0_240:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s24
	s_and_saveexec_b32 s24, s26
	s_cbranch_execnz .LBB0_360
.LBB0_241:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s24
	s_and_saveexec_b32 s24, s27
	s_cbranch_execnz .LBB0_361
.LBB0_242:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s24
	s_and_saveexec_b32 s24, s28
	s_cbranch_execnz .LBB0_362
.LBB0_243:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s24
	s_and_saveexec_b32 s24, s29
	s_cbranch_execnz .LBB0_363
.LBB0_244:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s24
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s30
	s_cbranch_execz .LBB0_246
.LBB0_245:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29148
	v_add_f32_e32 v33, v48, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v34
	v_mad_co_u64_u32 v[33:34], null, v184, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v184, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s23, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s23
	global_store_b32 v[33:34], v40, off
.LBB0_246:                              ; %Flow2754
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s31
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v38, 0
	v_mov_b32_e32 v39, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s23, s22
	s_cbranch_execz .LBB0_248
; %bb.247:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[34:35], null, v183, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[35:36], null, v183, s51, v[35:36]
	scratch_load_b32 v36, off, off          ; 4-byte Folded Reload
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	s_wait_loadcnt 0x0
	v_add_co_u32 v38, s22, v36, v34
	scratch_load_b32 v34, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v39, null, v34, v35, s22
	s_clause 0x1
	global_load_b128 v[34:37], v[38:39], off
	global_load_b128 v[38:41], v[38:39], off offset:16
.LBB0_248:                              ; %.preheader945.21237
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s23
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v41.h, v41
	v_cvt_f16_f32_e32 v41.l, v40
	v_cvt_f16_f32_e32 v40.h, v39
	v_cvt_f16_f32_e32 v40.l, v38
	v_cvt_f16_f32_e32 v39.h, v37
	v_cvt_f16_f32_e32 v39.l, v36
	v_cvt_f16_f32_e32 v38.h, v35
	v_cvt_f16_f32_e32 v38.l, v34
	s_mov_b32 s22, 0
	ds_store_b128 v83, v[38:41] offset:24704
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v38, v33 :: v_dual_mov_b32 v39, v33
	v_mov_b32_e32 v40, v33
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[34:37], v133 offset:24704
	ds_load_b128 v[73:76], v133 offset:24736
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[41:48], v[34:37], v[57:60], 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[53:56], v[41:48]
	ds_load_b128 v[34:37], v133 offset:24768
	ds_load_b128 v[73:76], v133 offset:24800
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[41:48], v[34:37], v[49:52], v[41:48]
	v_dual_mov_b32 v34, v33 :: v_dual_mov_b32 v35, v33
	v_dual_mov_b32 v36, v33 :: v_dual_mov_b32 v37, v33
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[61:64], v[41:48]
	ds_store_b128 v165, v[41:44]
	ds_store_b128 v165, v[45:48] offset:16
	v_mov_b32_e32 v76, v116
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v75, v88
	ds_load_b32 v74, v89
	ds_load_b32 v73, v90
	ds_load_b32 v71, v91
	ds_load_b32 v70, v92
	ds_load_b32 v69, v93
	ds_load_b32 v66, v94
	ds_load_b32 v65, v95
.LBB0_249:                              ; %.preheader944.2
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_load_2addr_stride64_b32 v[98:99], v76 offset0:33 offset1:34
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 m0, s22, 3
	s_add_co_i32 s22, s22, 1
	v_movrels_b32_e32 v77, v1
	v_movrels_b32_e32 v79, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s22, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v106.l, v77
	v_movrels_b32_e32 v77, v3
	v_cvt_f16_f32_e32 v106.h, v79
	v_cvt_f16_f32_e32 v107.l, v77
	v_movrels_b32_e32 v77, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v107.h, v77
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v102.l, v98
	v_cvt_f16_f32_e32 v102.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v76 offset0:35 offset1:36
	v_movrels_b32_e32 v77, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v108.l, v77
	v_movrels_b32_e32 v77, v6
	v_cvt_f16_f32_e32 v108.h, v77
	v_movrels_b32_e32 v77, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v109.l, v77
	v_movrels_b32_e32 v77, v8
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v103.l, v98
	v_cvt_f16_f32_e32 v103.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v76 offset0:37 offset1:38
	v_cvt_f16_f32_e32 v109.h, v77
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v104.l, v98
	v_cvt_f16_f32_e32 v104.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v76 offset0:39 offset1:40
	v_add_nc_u32_e32 v76, 0x1000, v76
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v105.l, v98
	v_cvt_f16_f32_e32 v105.h, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[102:105], v[106:109], v[33:40]
	s_cbranch_scc1 .LBB0_249
; %bb.250:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_saveexec_b32 s22, s5
	s_cbranch_execz .LBB0_260
; %bb.251:                              ; %.preheader952.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_and_saveexec_b32 s23, s13
	s_cbranch_execnz .LBB0_364
; %bb.252:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s23
	s_and_saveexec_b32 s23, s14
	s_cbranch_execnz .LBB0_365
.LBB0_253:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s23
	s_and_saveexec_b32 s14, s15
	s_cbranch_execnz .LBB0_366
.LBB0_254:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s16
	s_cbranch_execnz .LBB0_367
.LBB0_255:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s17
	s_cbranch_execnz .LBB0_368
.LBB0_256:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s18
	s_cbranch_execnz .LBB0_369
.LBB0_257:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s20
	s_cbranch_execnz .LBB0_370
.LBB0_258:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s21
	s_cbranch_execz .LBB0_260
.LBB0_259:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29212
	v_add_f32_e32 v33, v48, v65
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v40, v33, v34
	v_mad_co_u64_u32 v[33:34], null, v181, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v181, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s13, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s13
	global_store_b32 v[33:34], v40, off
.LBB0_260:                              ; %Flow2752
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v38, 0
	v_mov_b32_e32 v39, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s14, s19
	s_cbranch_execz .LBB0_262
; %bb.261:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[34:35], null, v182, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[35:36], null, v182, s51, v[35:36]
	scratch_load_b32 v36, off, off          ; 4-byte Folded Reload
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	s_wait_loadcnt 0x0
	v_add_co_u32 v38, s13, v36, v34
	scratch_load_b32 v34, off, off offset:4 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v39, null, v34, v35, s13
	s_clause 0x1
	global_load_b128 v[34:37], v[38:39], off
	global_load_b128 v[38:41], v[38:39], off offset:16
.LBB0_262:                              ; %.preheader945.31240
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v41.h, v41
	v_cvt_f16_f32_e32 v41.l, v40
	v_cvt_f16_f32_e32 v40.h, v39
	v_cvt_f16_f32_e32 v40.l, v38
	v_cvt_f16_f32_e32 v39.h, v37
	v_cvt_f16_f32_e32 v39.l, v36
	v_cvt_f16_f32_e32 v38.h, v35
	v_cvt_f16_f32_e32 v38.l, v34
	s_mov_b32 s13, 0
	ds_store_b128 v83, v[38:41] offset:24704
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v38, v33 :: v_dual_mov_b32 v39, v33
	v_mov_b32_e32 v40, v33
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[34:37], v133 offset:24704
	ds_load_b128 v[73:76], v133 offset:24736
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[41:48], v[34:37], v[57:60], 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[53:56], v[41:48]
	ds_load_b128 v[34:37], v133 offset:24768
	ds_load_b128 v[73:76], v133 offset:24800
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[41:48], v[34:37], v[49:52], v[41:48]
	v_dual_mov_b32 v34, v33 :: v_dual_mov_b32 v35, v33
	v_dual_mov_b32 v36, v33 :: v_dual_mov_b32 v37, v33
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[61:64], v[41:48]
	ds_store_b128 v165, v[41:44]
	ds_store_b128 v165, v[45:48] offset:16
	v_mov_b32_e32 v76, v116
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v75, v88
	ds_load_b32 v74, v89
	ds_load_b32 v73, v90
	ds_load_b32 v71, v91
	ds_load_b32 v70, v92
	ds_load_b32 v69, v93
	ds_load_b32 v66, v94
	ds_load_b32 v65, v95
.LBB0_263:                              ; %.preheader944.3
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_nc_u32_e32 v77, 64, v76
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 m0, s13, 3
	v_add_nc_u32_e32 v76, 0x1000, v76
	v_movrels_b32_e32 v79, v2
	s_add_co_i32 s13, s13, 1
	ds_load_2addr_stride64_b32 v[98:99], v77 offset0:33 offset1:34
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s13, 4
	v_cvt_f16_f32_e32 v106.h, v79
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v102.l, v98
	v_cvt_f16_f32_e32 v102.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v77 offset0:35 offset1:36
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v103.l, v98
	v_cvt_f16_f32_e32 v103.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v77 offset0:37 offset1:38
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v104.l, v98
	v_cvt_f16_f32_e32 v104.h, v99
	ds_load_2addr_stride64_b32 v[98:99], v77 offset0:39 offset1:40
	v_movrels_b32_e32 v77, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v106.l, v77
	v_movrels_b32_e32 v77, v3
	v_cvt_f16_f32_e32 v107.l, v77
	v_movrels_b32_e32 v77, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v107.h, v77
	v_movrels_b32_e32 v77, v5
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v105.l, v98
	v_cvt_f16_f32_e32 v105.h, v99
	v_cvt_f16_f32_e32 v108.l, v77
	v_movrels_b32_e32 v77, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v108.h, v77
	v_movrels_b32_e32 v77, v7
	v_cvt_f16_f32_e32 v109.l, v77
	v_movrels_b32_e32 v77, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v109.h, v77
	v_wmma_f32_16x16x16_f16 v[33:40], v[102:105], v[106:109], v[33:40]
	s_cbranch_scc1 .LBB0_263
; %bb.264:                              ;   in Loop: Header=BB0_3 Depth=1
	s_mov_b32 s13, exec_lo
	v_dual_mov_b32 v108, v67 :: v_dual_mov_b32 v67, v68
	v_mov_b32_e32 v109, v111
	v_or_b32_e32 v103, 48, v78
	v_or_b32_e32 v79, 50, v78
	v_or_b32_e32 v82, 51, v78
	v_or_b32_e32 v84, 52, v78
	v_or_b32_e32 v98, 53, v78
	v_or_b32_e32 v99, 54, v78
	v_or_b32_e32 v102, 55, v78
	v_or_b32_e32 v104, 32, v78
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s14, s13, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 exec_lo, s14
	s_cbranch_execz .LBB0_274
; %bb.265:                              ; %.preheader952.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_and_saveexec_b32 s14, vcc_lo
	s_cbranch_execnz .LBB0_371
; %bb.266:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s6
	s_cbranch_execnz .LBB0_372
.LBB0_267:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s6, s7
	s_cbranch_execnz .LBB0_373
.LBB0_268:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s8
	s_cbranch_execnz .LBB0_374
.LBB0_269:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s9
	s_cbranch_execnz .LBB0_375
.LBB0_270:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s10
	s_cbranch_execnz .LBB0_376
.LBB0_271:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s11
	s_cbranch_execnz .LBB0_377
.LBB0_272:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, s12
	s_cbranch_execz .LBB0_274
.LBB0_273:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[33:34], null, v173, s50, 0
	ds_load_b32 v36, v97 offset:29276
	v_mad_co_u64_u32 v[34:35], null, v173, s51, v[34:35]
	v_add_f32_e32 v35, v48, v65
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	s_wait_dscnt 0x0
	v_fmac_f32_e32 v40, v35, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v33, vcc_lo, v160, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v34, null, v161, v34, vcc_lo
	global_store_b32 v[33:34], v40, off
.LBB0_274:                              ; %Flow2750
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v36, s69, v81
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s7, 0
	v_mad_co_u64_u32 v[33:34], null, v36, s50, 0
	v_cmp_gt_i32_e32 vcc_lo, s56, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[34:35], null, v36, s51, v[34:35]
	v_lshlrev_b64_e32 v[34:35], 2, v[33:34]
	scratch_load_b32 v33, off, off offset:24 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	v_add_co_u32 v34, s6, v33, v34
	scratch_load_b32 v33, off, off offset:28 ; 4-byte Folded Reload
	s_barrier_wait -1
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v35, null, v33, v35, s6
	s_branch .LBB0_276
.LBB0_275:                              ;   in Loop: Header=BB0_276 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_lshlrev_b32_e32 v36, 1, v36
	s_add_co_i32 s7, s7, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 16
	ds_store_b16 v36, v33 offset:8320
	s_cbranch_scc1 .LBB0_278
.LBB0_276:                              ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v36, s7, 8, v0
	v_mov_b16_e32 v33.l, 0
	s_and_saveexec_b32 s8, vcc_lo
	s_cbranch_execz .LBB0_275
; %bb.277:                              ;   in Loop: Header=BB0_276 Depth=2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v33, 4, v36
	v_and_b32_e32 v33, 0x7fffffc, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v37, s6, v34, v33
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v38, null, 0, v35, s6
	global_load_b32 v33, v[37:38], off
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v33.l, v33
	s_branch .LBB0_275
.LBB0_278:                              ; %.preheader951
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v33, 0
	s_mov_b32 s6, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b64 v[45:46], v33 offset:29568
	s_wait_dscnt 0x0
	v_fma_mix_f32 v33, v57, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v34, v57, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v35, v58, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v36, v58, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v37, v59, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v38, v59, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v39, v60, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v40, v60, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_branch .LBB0_280
.LBB0_279:                              ;   in Loop: Header=BB0_280 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_lshl_b32 m0, s6, 3
	s_add_co_i32 s6, s6, 1
	v_movrels_b32_e32 v57, v1
	v_movrels_b32_e32 v58, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_mixlo_f16 v57, v57, v48, 0
	v_fma_mixhi_f16 v57, v58, v47, 0
	v_movrels_b32_e32 v47, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v58, v47, v65, 0
	v_movrels_b32_e32 v47, v4
	v_fma_mixhi_f16 v58, v47, v60, 0
	v_movrels_b32_e32 v47, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v59, v47, v69, 0
	v_movrels_b32_e32 v47, v6
	v_fma_mixhi_f16 v59, v47, v66, 0
	v_movrels_b32_e32 v47, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v60, v47, v71, 0
	v_movrels_b32_e32 v47, v8
	v_fma_mixhi_f16 v60, v47, v70, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[57:60], v[33:40]
	s_cbranch_scc1 .LBB0_296
.LBB0_280:                              ; %.preheader943
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v58, s6, 4, v78
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u32 v59, v58, 1, v87
	v_lshlrev_b32_e32 v57, 2, v58
	ds_load_u16_d16 v41, v59 offset:8320
	v_cmpx_gt_i32_e64 s58, v58
	s_cbranch_execz .LBB0_282
; %bb.281:                              ;   in Loop: Header=BB0_280 Depth=2
	ds_load_b32 v42, v57 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v42, v46, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_fma_f32 v44, 0x3fb8aa3b, v42, -v43
	v_rndne_f32_e32 v48, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v43, v43, v48
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v42
	v_fmac_f32_e32 v44, 0x32a5705f, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v43, v43, v44
	v_cvt_i32_f32_e32 v44, v48
	v_exp_f32_e32 v43, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v43, v43, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v43, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v48, 0x7f800000, v43, vcc_lo
.LBB0_282:                              ;   in Loop: Header=BB0_280 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v41, v59 offset:8322
	v_or_b32_e32 v42, 1, v58
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v42
	s_cbranch_execz .LBB0_284
; %bb.283:                              ;   in Loop: Header=BB0_280 Depth=2
	ds_load_b32 v42, v57 offset:28804
	s_wait_dscnt 0x0
	v_sub_f32_e32 v42, v46, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_fma_f32 v44, 0x3fb8aa3b, v42, -v43
	v_rndne_f32_e32 v47, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v43, v43, v47
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v42
	v_fmac_f32_e32 v44, 0x32a5705f, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v43, v43, v44
	v_cvt_i32_f32_e32 v44, v47
	v_exp_f32_e32 v43, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v43, v43, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v43, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v47, 0x7f800000, v43, vcc_lo
.LBB0_284:                              ;   in Loop: Header=BB0_280 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v42, v59 offset:8324
	v_or_b32_e32 v43, 2, v58
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v65, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s58, v43
	s_cbranch_execz .LBB0_286
; %bb.285:                              ;   in Loop: Header=BB0_280 Depth=2
	ds_load_b32 v43, v57 offset:28808
	s_wait_dscnt 0x0
	v_sub_f32_e32 v43, v46, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_fma_f32 v65, 0x3fb8aa3b, v43, -v44
	v_rndne_f32_e32 v66, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v66
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v43
	v_fmac_f32_e32 v65, 0x32a5705f, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v44, v44, v65
	v_cvt_i32_f32_e32 v65, v66
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v44, v44, v65
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0, v44, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v43
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v65, 0x7f800000, v44, vcc_lo
.LBB0_286:                              ;   in Loop: Header=BB0_280 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v42, v59 offset:8326
	v_or_b32_e32 v43, 3, v58
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v43
	s_cbranch_execz .LBB0_288
; %bb.287:                              ;   in Loop: Header=BB0_280 Depth=2
	ds_load_b32 v43, v57 offset:28812
	s_wait_dscnt 0x0
	v_sub_f32_e32 v43, v46, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_fma_f32 v60, 0x3fb8aa3b, v43, -v44
	v_rndne_f32_e32 v66, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v44, v44, v66
	v_fmac_f32_e32 v60, 0x32a5705f, v43
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v44, v44, v60
	v_cvt_i32_f32_e32 v60, v66
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v44, v44, v60
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0, v44, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v43
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v60, 0x7f800000, v44, vcc_lo
.LBB0_288:                              ;   in Loop: Header=BB0_280 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v43, v59 offset:8328
	v_or_b32_e32 v44, 4, v58
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v69, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s58, v44
	s_cbranch_execz .LBB0_290
; %bb.289:                              ;   in Loop: Header=BB0_280 Depth=2
	ds_load_b32 v44, v57 offset:28816
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v69, 0x3fb8aa3b, v44
	v_fma_f32 v70, 0x3fb8aa3b, v44, -v69
	v_rndne_f32_e32 v71, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v69, v69, v71
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fmac_f32_e32 v70, 0x32a5705f, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v69, v69, v70
	v_cvt_i32_f32_e32 v70, v71
	v_exp_f32_e32 v69, v69
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v69, v69, v70
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v69, 0, v69, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v69, 0x7f800000, v69, vcc_lo
.LBB0_290:                              ;   in Loop: Header=BB0_280 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v43, v59 offset:8330
	v_or_b32_e32 v44, 5, v58
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v44
	s_cbranch_execz .LBB0_292
; %bb.291:                              ;   in Loop: Header=BB0_280 Depth=2
	ds_load_b32 v44, v57 offset:28820
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v66, 0x3fb8aa3b, v44
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fma_f32 v70, 0x3fb8aa3b, v44, -v66
	v_rndne_f32_e32 v71, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v70, 0x32a5705f, v44
	v_sub_f32_e32 v66, v66, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v66, v66, v70
	v_cvt_i32_f32_e32 v70, v71
	v_exp_f32_e32 v66, v66
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v66, v66, v70
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v66, 0, v66, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v66, 0x7f800000, v66, vcc_lo
.LBB0_292:                              ;   in Loop: Header=BB0_280 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v44, v59 offset:8332
	v_or_b32_e32 v70, 6, v58
	v_mov_b32_e32 v71, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s58, v70
	v_mov_b32_e32 v70, 0
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB0_294
; %bb.293:                              ;   in Loop: Header=BB0_280 Depth=2
	ds_load_b32 v71, v57 offset:28824
	s_wait_dscnt 0x0
	v_sub_f32_e32 v71, v46, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v73, 0x3fb8aa3b, v71
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v71
	v_fma_f32 v74, 0x3fb8aa3b, v71, -v73
	v_rndne_f32_e32 v75, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v74, 0x32a5705f, v71
	v_sub_f32_e32 v73, v73, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v73, v73, v74
	v_cvt_i32_f32_e32 v74, v75
	v_exp_f32_e32 v73, v73
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v73, v73, v74
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v73, 0, v73, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v71, 0x7f800000, v73, vcc_lo
.LBB0_294:                              ;   in Loop: Header=BB0_280 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v44, v59 offset:8334
	v_or_b32_e32 v58, 7, v58
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v58
	s_cbranch_execz .LBB0_279
; %bb.295:                              ;   in Loop: Header=BB0_280 Depth=2
	ds_load_b32 v57, v57 offset:28828
	s_wait_dscnt 0x0
	v_sub_f32_e32 v57, v46, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v58, 0x3fb8aa3b, v57
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v57
	v_fma_f32 v59, 0x3fb8aa3b, v57, -v58
	v_rndne_f32_e32 v70, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v59, 0x32a5705f, v57 :: v_dual_sub_f32 v58, v58, v70
	v_add_f32_e32 v58, v58, v59
	v_cvt_i32_f32_e32 v59, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v58, v58
	v_ldexp_f32 v58, v58, v59
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v58, 0, v58, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v57
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v70, 0x7f800000, v58, vcc_lo
	s_branch .LBB0_279
.LBB0_296:                              ; %.preheader951.1
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v60.h, v40
	v_cvt_f16_f32_e32 v60.l, v39
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v59.h, v38
	v_cvt_f16_f32_e32 v59.l, v37
	v_cvt_f16_f32_e32 v58.h, v36
	v_cvt_f16_f32_e32 v58.l, v35
	v_cvt_f16_f32_e32 v57.h, v34
	v_cvt_f16_f32_e32 v57.l, v33
	v_fma_mix_f32 v33, v53, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v34, v53, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v35, v54, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v36, v54, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v37, v55, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v38, v55, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v39, v56, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v40, v56, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_mov_b32 s6, 0
	s_branch .LBB0_298
.LBB0_297:                              ;   in Loop: Header=BB0_298 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_lshl_b32 m0, s6, 3
	s_add_co_i32 s6, s6, 1
	v_movrels_b32_e32 v53, v1
	v_movrels_b32_e32 v54, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_mixlo_f16 v53, v53, v48, 0
	v_fma_mixhi_f16 v53, v54, v47, 0
	v_movrels_b32_e32 v47, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v54, v47, v65, 0
	v_movrels_b32_e32 v47, v4
	v_fma_mixhi_f16 v54, v47, v56, 0
	v_movrels_b32_e32 v47, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v55, v47, v69, 0
	v_movrels_b32_e32 v47, v6
	v_fma_mixhi_f16 v55, v47, v66, 0
	v_movrels_b32_e32 v47, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v56, v47, v71, 0
	v_movrels_b32_e32 v47, v8
	v_fma_mixhi_f16 v56, v47, v70, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[53:56], v[33:40]
	s_cbranch_scc0 .LBB0_314
.LBB0_298:                              ; %.preheader943.1
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v54, s6, 4, v78
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u32 v55, v54, 1, v87
	v_lshlrev_b32_e32 v53, 2, v54
	ds_load_u16_d16 v41, v55 offset:10368
	v_cmpx_gt_i32_e64 s58, v54
	s_cbranch_execz .LBB0_300
; %bb.299:                              ;   in Loop: Header=BB0_298 Depth=2
	ds_load_b32 v42, v53 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v42, v46, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_fma_f32 v44, 0x3fb8aa3b, v42, -v43
	v_rndne_f32_e32 v48, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v43, v43, v48
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v42
	v_fmac_f32_e32 v44, 0x32a5705f, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v43, v43, v44
	v_cvt_i32_f32_e32 v44, v48
	v_exp_f32_e32 v43, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v43, v43, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v43, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v48, 0x7f800000, v43, vcc_lo
.LBB0_300:                              ;   in Loop: Header=BB0_298 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v41, v55 offset:10370
	v_or_b32_e32 v42, 1, v54
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v42
	s_cbranch_execz .LBB0_302
; %bb.301:                              ;   in Loop: Header=BB0_298 Depth=2
	ds_load_b32 v42, v53 offset:28804
	s_wait_dscnt 0x0
	v_sub_f32_e32 v42, v46, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_fma_f32 v44, 0x3fb8aa3b, v42, -v43
	v_rndne_f32_e32 v47, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v43, v43, v47
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v42
	v_fmac_f32_e32 v44, 0x32a5705f, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v43, v43, v44
	v_cvt_i32_f32_e32 v44, v47
	v_exp_f32_e32 v43, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v43, v43, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v43, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v47, 0x7f800000, v43, vcc_lo
.LBB0_302:                              ;   in Loop: Header=BB0_298 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v42, v55 offset:10372
	v_or_b32_e32 v43, 2, v54
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v65, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s58, v43
	s_cbranch_execz .LBB0_304
; %bb.303:                              ;   in Loop: Header=BB0_298 Depth=2
	ds_load_b32 v43, v53 offset:28808
	s_wait_dscnt 0x0
	v_sub_f32_e32 v43, v46, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_fma_f32 v65, 0x3fb8aa3b, v43, -v44
	v_rndne_f32_e32 v66, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v66
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v43
	v_fmac_f32_e32 v65, 0x32a5705f, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v44, v44, v65
	v_cvt_i32_f32_e32 v65, v66
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v44, v44, v65
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0, v44, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v43
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v65, 0x7f800000, v44, vcc_lo
.LBB0_304:                              ;   in Loop: Header=BB0_298 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v42, v55 offset:10374
	v_or_b32_e32 v43, 3, v54
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v43
	s_cbranch_execz .LBB0_306
; %bb.305:                              ;   in Loop: Header=BB0_298 Depth=2
	ds_load_b32 v43, v53 offset:28812
	s_wait_dscnt 0x0
	v_sub_f32_e32 v43, v46, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_fma_f32 v56, 0x3fb8aa3b, v43, -v44
	v_rndne_f32_e32 v66, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v44, v44, v66
	v_fmac_f32_e32 v56, 0x32a5705f, v43
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v44, v44, v56
	v_cvt_i32_f32_e32 v56, v66
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v44, v44, v56
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0, v44, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v43
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v56, 0x7f800000, v44, vcc_lo
.LBB0_306:                              ;   in Loop: Header=BB0_298 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v43, v55 offset:10376
	v_or_b32_e32 v44, 4, v54
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v69, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s58, v44
	s_cbranch_execz .LBB0_308
; %bb.307:                              ;   in Loop: Header=BB0_298 Depth=2
	ds_load_b32 v44, v53 offset:28816
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v69, 0x3fb8aa3b, v44
	v_fma_f32 v70, 0x3fb8aa3b, v44, -v69
	v_rndne_f32_e32 v71, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v69, v69, v71
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fmac_f32_e32 v70, 0x32a5705f, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v69, v69, v70
	v_cvt_i32_f32_e32 v70, v71
	v_exp_f32_e32 v69, v69
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v69, v69, v70
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v69, 0, v69, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v69, 0x7f800000, v69, vcc_lo
.LBB0_308:                              ;   in Loop: Header=BB0_298 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v43, v55 offset:10378
	v_or_b32_e32 v44, 5, v54
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v44
	s_cbranch_execz .LBB0_310
; %bb.309:                              ;   in Loop: Header=BB0_298 Depth=2
	ds_load_b32 v44, v53 offset:28820
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v66, 0x3fb8aa3b, v44
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fma_f32 v70, 0x3fb8aa3b, v44, -v66
	v_rndne_f32_e32 v71, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v70, 0x32a5705f, v44
	v_sub_f32_e32 v66, v66, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v66, v66, v70
	v_cvt_i32_f32_e32 v70, v71
	v_exp_f32_e32 v66, v66
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v66, v66, v70
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v66, 0, v66, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v66, 0x7f800000, v66, vcc_lo
.LBB0_310:                              ;   in Loop: Header=BB0_298 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v44, v55 offset:10380
	v_or_b32_e32 v70, 6, v54
	v_mov_b32_e32 v71, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s58, v70
	v_mov_b32_e32 v70, 0
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB0_312
; %bb.311:                              ;   in Loop: Header=BB0_298 Depth=2
	ds_load_b32 v71, v53 offset:28824
	s_wait_dscnt 0x0
	v_sub_f32_e32 v71, v46, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v73, 0x3fb8aa3b, v71
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v71
	v_fma_f32 v74, 0x3fb8aa3b, v71, -v73
	v_rndne_f32_e32 v75, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v74, 0x32a5705f, v71
	v_sub_f32_e32 v73, v73, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v73, v73, v74
	v_cvt_i32_f32_e32 v74, v75
	v_exp_f32_e32 v73, v73
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v73, v73, v74
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v73, 0, v73, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v71, 0x7f800000, v73, vcc_lo
.LBB0_312:                              ;   in Loop: Header=BB0_298 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v44, v55 offset:10382
	v_or_b32_e32 v54, 7, v54
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v54
	s_cbranch_execz .LBB0_297
; %bb.313:                              ;   in Loop: Header=BB0_298 Depth=2
	ds_load_b32 v53, v53 offset:28828
	s_wait_dscnt 0x0
	v_sub_f32_e32 v53, v46, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v54, 0x3fb8aa3b, v53
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v53
	v_fma_f32 v55, 0x3fb8aa3b, v53, -v54
	v_rndne_f32_e32 v70, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v55, 0x32a5705f, v53 :: v_dual_sub_f32 v54, v54, v70
	v_add_f32_e32 v54, v54, v55
	v_cvt_i32_f32_e32 v55, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v54, v54
	v_ldexp_f32 v54, v54, v55
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v54, 0, v54, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v53
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v70, 0x7f800000, v54, vcc_lo
	s_branch .LBB0_297
.LBB0_314:                              ; %.preheader951.2
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v56.h, v40
	v_cvt_f16_f32_e32 v56.l, v39
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v55.h, v38
	v_cvt_f16_f32_e32 v55.l, v37
	v_cvt_f16_f32_e32 v54.h, v36
	v_cvt_f16_f32_e32 v54.l, v35
	v_cvt_f16_f32_e32 v53.h, v34
	v_cvt_f16_f32_e32 v53.l, v33
	v_fma_mix_f32 v33, v49, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v34, v49, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v35, v50, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v36, v50, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v37, v51, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v38, v51, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v39, v52, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v40, v52, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_mov_b32 s6, 0
	s_branch .LBB0_316
.LBB0_315:                              ;   in Loop: Header=BB0_316 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_lshl_b32 m0, s6, 3
	s_add_co_i32 s6, s6, 1
	v_movrels_b32_e32 v49, v1
	v_movrels_b32_e32 v50, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_mixlo_f16 v48, v49, v48, 0
	v_fma_mixhi_f16 v48, v50, v47, 0
	v_movrels_b32_e32 v47, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v49, v47, v65, 0
	v_movrels_b32_e32 v47, v4
	v_fma_mixhi_f16 v49, v47, v52, 0
	v_movrels_b32_e32 v47, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v50, v47, v69, 0
	v_movrels_b32_e32 v47, v6
	v_fma_mixhi_f16 v50, v47, v66, 0
	v_movrels_b32_e32 v47, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixlo_f16 v51, v47, v71, 0
	v_movrels_b32_e32 v47, v8
	v_fma_mixhi_f16 v51, v47, v70, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[48:51], v[33:40]
	s_cbranch_scc0 .LBB0_332
.LBB0_316:                              ; %.preheader943.2
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v50, s6, 4, v78
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u32 v51, v50, 1, v87
	v_lshlrev_b32_e32 v49, 2, v50
	ds_load_u16_d16 v41, v51 offset:12416
	v_cmpx_gt_i32_e64 s58, v50
	s_cbranch_execz .LBB0_318
; %bb.317:                              ;   in Loop: Header=BB0_316 Depth=2
	ds_load_b32 v42, v49 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v42, v46, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_fma_f32 v44, 0x3fb8aa3b, v42, -v43
	v_rndne_f32_e32 v48, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v43, v43, v48
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v42
	v_fmac_f32_e32 v44, 0x32a5705f, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v43, v43, v44
	v_cvt_i32_f32_e32 v44, v48
	v_exp_f32_e32 v43, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v43, v43, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v43, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v48, 0x7f800000, v43, vcc_lo
.LBB0_318:                              ;   in Loop: Header=BB0_316 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v41, v51 offset:12418
	v_or_b32_e32 v42, 1, v50
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v42
	s_cbranch_execz .LBB0_320
; %bb.319:                              ;   in Loop: Header=BB0_316 Depth=2
	ds_load_b32 v42, v49 offset:28804
	s_wait_dscnt 0x0
	v_sub_f32_e32 v42, v46, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_fma_f32 v44, 0x3fb8aa3b, v42, -v43
	v_rndne_f32_e32 v47, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v43, v43, v47
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v42
	v_fmac_f32_e32 v44, 0x32a5705f, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v43, v43, v44
	v_cvt_i32_f32_e32 v44, v47
	v_exp_f32_e32 v43, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v43, v43, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v43, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v47, 0x7f800000, v43, vcc_lo
.LBB0_320:                              ;   in Loop: Header=BB0_316 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v42, v51 offset:12420
	v_or_b32_e32 v43, 2, v50
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v65, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s58, v43
	s_cbranch_execz .LBB0_322
; %bb.321:                              ;   in Loop: Header=BB0_316 Depth=2
	ds_load_b32 v43, v49 offset:28808
	s_wait_dscnt 0x0
	v_sub_f32_e32 v43, v46, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_fma_f32 v65, 0x3fb8aa3b, v43, -v44
	v_rndne_f32_e32 v66, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v66
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v43
	v_fmac_f32_e32 v65, 0x32a5705f, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v44, v44, v65
	v_cvt_i32_f32_e32 v65, v66
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v44, v44, v65
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0, v44, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v43
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v65, 0x7f800000, v44, vcc_lo
.LBB0_322:                              ;   in Loop: Header=BB0_316 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v42, v51 offset:12422
	v_or_b32_e32 v43, 3, v50
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v43
	s_cbranch_execz .LBB0_324
; %bb.323:                              ;   in Loop: Header=BB0_316 Depth=2
	ds_load_b32 v43, v49 offset:28812
	s_wait_dscnt 0x0
	v_sub_f32_e32 v43, v46, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_fma_f32 v52, 0x3fb8aa3b, v43, -v44
	v_rndne_f32_e32 v66, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v44, v44, v66
	v_fmac_f32_e32 v52, 0x32a5705f, v43
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v44, v44, v52
	v_cvt_i32_f32_e32 v52, v66
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v44, v44, v52
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0, v44, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v43
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v52, 0x7f800000, v44, vcc_lo
.LBB0_324:                              ;   in Loop: Header=BB0_316 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v43, v51 offset:12424
	v_or_b32_e32 v44, 4, v50
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v69, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s58, v44
	s_cbranch_execz .LBB0_326
; %bb.325:                              ;   in Loop: Header=BB0_316 Depth=2
	ds_load_b32 v44, v49 offset:28816
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v69, 0x3fb8aa3b, v44
	v_fma_f32 v70, 0x3fb8aa3b, v44, -v69
	v_rndne_f32_e32 v71, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v69, v69, v71
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fmac_f32_e32 v70, 0x32a5705f, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v69, v69, v70
	v_cvt_i32_f32_e32 v70, v71
	v_exp_f32_e32 v69, v69
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v69, v69, v70
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v69, 0, v69, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v69, 0x7f800000, v69, vcc_lo
.LBB0_326:                              ;   in Loop: Header=BB0_316 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v43, v51 offset:12426
	v_or_b32_e32 v44, 5, v50
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v44
	s_cbranch_execz .LBB0_328
; %bb.327:                              ;   in Loop: Header=BB0_316 Depth=2
	ds_load_b32 v44, v49 offset:28820
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v66, 0x3fb8aa3b, v44
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fma_f32 v70, 0x3fb8aa3b, v44, -v66
	v_rndne_f32_e32 v71, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v70, 0x32a5705f, v44
	v_sub_f32_e32 v66, v66, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v66, v66, v70
	v_cvt_i32_f32_e32 v70, v71
	v_exp_f32_e32 v66, v66
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v66, v66, v70
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v66, 0, v66, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v66, 0x7f800000, v66, vcc_lo
.LBB0_328:                              ;   in Loop: Header=BB0_316 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v44, v51 offset:12428
	v_or_b32_e32 v70, 6, v50
	v_mov_b32_e32 v71, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s58, v70
	v_mov_b32_e32 v70, 0
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB0_330
; %bb.329:                              ;   in Loop: Header=BB0_316 Depth=2
	ds_load_b32 v71, v49 offset:28824
	s_wait_dscnt 0x0
	v_sub_f32_e32 v71, v46, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v73, 0x3fb8aa3b, v71
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v71
	v_fma_f32 v74, 0x3fb8aa3b, v71, -v73
	v_rndne_f32_e32 v75, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v74, 0x32a5705f, v71
	v_sub_f32_e32 v73, v73, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v73, v73, v74
	v_cvt_i32_f32_e32 v74, v75
	v_exp_f32_e32 v73, v73
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v73, v73, v74
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v73, 0, v73, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v71, 0x7f800000, v73, vcc_lo
.LBB0_330:                              ;   in Loop: Header=BB0_316 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v44, v51 offset:12430
	v_or_b32_e32 v50, 7, v50
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v50
	s_cbranch_execz .LBB0_315
; %bb.331:                              ;   in Loop: Header=BB0_316 Depth=2
	ds_load_b32 v49, v49 offset:28828
	s_wait_dscnt 0x0
	v_sub_f32_e32 v49, v46, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v50, 0x3fb8aa3b, v49
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v49
	v_fma_f32 v51, 0x3fb8aa3b, v49, -v50
	v_rndne_f32_e32 v70, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v51, 0x32a5705f, v49 :: v_dual_sub_f32 v50, v50, v70
	v_add_f32_e32 v50, v50, v51
	v_cvt_i32_f32_e32 v51, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v50, v50
	v_ldexp_f32 v50, v50, v51
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v50, 0, v50, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v49
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v70, 0x7f800000, v50, vcc_lo
	s_branch .LBB0_315
.LBB0_332:                              ; %.preheader951.3
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v52.h, v40
	v_cvt_f16_f32_e32 v52.l, v39
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v51.h, v38
	v_cvt_f16_f32_e32 v51.l, v37
	v_cvt_f16_f32_e32 v50.h, v36
	v_cvt_f16_f32_e32 v50.l, v35
	v_cvt_f16_f32_e32 v49.h, v34
	v_cvt_f16_f32_e32 v49.l, v33
	v_fma_mix_f32 v33, v61, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v34, v61, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v35, v62, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v36, v62, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v37, v63, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v38, v63, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v39, v64, v45, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v40, v64, v45, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_mov_b32 s6, 0
	s_branch .LBB0_334
.LBB0_333:                              ;   in Loop: Header=BB0_334 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_lshl_b32 m0, s6, 3
	s_add_co_i32 s6, s6, 1
	v_movrels_b32_e32 v62, v2
	v_movrels_b32_e32 v48, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_mixhi_f16 v61, v62, v45, 0
	v_movrels_b32_e32 v45, v3
	v_fma_mixlo_f16 v61, v48, v47, 0
	v_fma_mixlo_f16 v62, v45, v64, 0
	v_movrels_b32_e32 v45, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixhi_f16 v62, v45, v63, 0
	v_movrels_b32_e32 v45, v5
	v_fma_mixlo_f16 v63, v45, v66, 0
	v_movrels_b32_e32 v45, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixhi_f16 v63, v45, v65, 0
	v_movrels_b32_e32 v45, v7
	v_fma_mixlo_f16 v64, v45, v70, 0
	v_movrels_b32_e32 v45, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mixhi_f16 v64, v45, v69, 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[61:64], v[33:40]
	s_cbranch_scc0 .LBB0_2
.LBB0_334:                              ; %.preheader943.3
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v61, s6, 4, v78
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshl_add_u32 v62, v61, 1, v87
	v_lshlrev_b32_e32 v48, 2, v61
	ds_load_u16_d16 v41, v62 offset:14464
	v_cmpx_gt_i32_e64 s58, v61
	s_cbranch_execz .LBB0_336
; %bb.335:                              ;   in Loop: Header=BB0_334 Depth=2
	ds_load_b32 v42, v48 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v42, v46, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_fma_f32 v44, 0x3fb8aa3b, v42, -v43
	v_rndne_f32_e32 v47, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v43, v43, v47
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v42
	v_fmac_f32_e32 v44, 0x32a5705f, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v43, v43, v44
	v_cvt_i32_f32_e32 v44, v47
	v_exp_f32_e32 v43, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v43, v43, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v43, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v47, 0x7f800000, v43, vcc_lo
.LBB0_336:                              ;   in Loop: Header=BB0_334 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v41, v62 offset:14466
	v_or_b32_e32 v42, 1, v61
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v42
	s_cbranch_execz .LBB0_338
; %bb.337:                              ;   in Loop: Header=BB0_334 Depth=2
	ds_load_b32 v42, v48 offset:28804
	s_wait_dscnt 0x0
	v_sub_f32_e32 v42, v46, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v43, 0x3fb8aa3b, v42
	v_fma_f32 v44, 0x3fb8aa3b, v42, -v43
	v_rndne_f32_e32 v45, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v43, v43, v45
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v42
	v_fmac_f32_e32 v44, 0x32a5705f, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v43, v43, v44
	v_cvt_i32_f32_e32 v44, v45
	v_exp_f32_e32 v43, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v43, v43, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v43, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v45, 0x7f800000, v43, vcc_lo
.LBB0_338:                              ;   in Loop: Header=BB0_334 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v42, v62 offset:14468
	v_or_b32_e32 v43, 2, v61
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v64, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s58, v43
	s_cbranch_execz .LBB0_340
; %bb.339:                              ;   in Loop: Header=BB0_334 Depth=2
	ds_load_b32 v43, v48 offset:28808
	s_wait_dscnt 0x0
	v_sub_f32_e32 v43, v46, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_fma_f32 v64, 0x3fb8aa3b, v43, -v44
	v_rndne_f32_e32 v65, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v44, v44, v65
	v_fmac_f32_e32 v64, 0x32a5705f, v43
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v44, v44, v64
	v_cvt_i32_f32_e32 v64, v65
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v44, v44, v64
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0, v44, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v43
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v64, 0x7f800000, v44, vcc_lo
.LBB0_340:                              ;   in Loop: Header=BB0_334 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v42, v62 offset:14470
	v_or_b32_e32 v43, 3, v61
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v43
	s_cbranch_execz .LBB0_342
; %bb.341:                              ;   in Loop: Header=BB0_334 Depth=2
	ds_load_b32 v43, v48 offset:28812
	s_wait_dscnt 0x0
	v_sub_f32_e32 v43, v46, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v43
	v_fma_f32 v63, 0x3fb8aa3b, v43, -v44
	v_rndne_f32_e32 v65, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v44, v44, v65
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v43
	v_fmac_f32_e32 v63, 0x32a5705f, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v44, v44, v63
	v_cvt_i32_f32_e32 v63, v65
	v_exp_f32_e32 v44, v44
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v44, v44, v63
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, 0, v44, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v43
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v63, 0x7f800000, v44, vcc_lo
.LBB0_342:                              ;   in Loop: Header=BB0_334 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v43, v62 offset:14472
	v_or_b32_e32 v44, 4, v61
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s58, v44
	s_cbranch_execz .LBB0_344
; %bb.343:                              ;   in Loop: Header=BB0_334 Depth=2
	ds_load_b32 v44, v48 offset:28816
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v66, 0x3fb8aa3b, v44
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fma_f32 v69, 0x3fb8aa3b, v44, -v66
	v_rndne_f32_e32 v70, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v69, 0x32a5705f, v44 :: v_dual_sub_f32 v66, v66, v70
	v_add_f32_e32 v66, v66, v69
	v_cvt_i32_f32_e32 v69, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v66, v66
	v_ldexp_f32 v66, v66, v69
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v66, 0, v66, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v66, 0x7f800000, v66, vcc_lo
.LBB0_344:                              ;   in Loop: Header=BB0_334 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v43, v62 offset:14474
	v_or_b32_e32 v44, 5, v61
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v44
	s_cbranch_execz .LBB0_346
; %bb.345:                              ;   in Loop: Header=BB0_334 Depth=2
	ds_load_b32 v44, v48 offset:28820
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v65, 0x3fb8aa3b, v44
	v_fma_f32 v69, 0x3fb8aa3b, v44, -v65
	v_rndne_f32_e32 v70, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v65, v65, v70
	v_fmac_f32_e32 v69, 0x32a5705f, v44
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v65, v65, v69
	v_cvt_i32_f32_e32 v69, v70
	v_exp_f32_e32 v65, v65
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v65, v65, v69
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v65, 0, v65, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v65, 0x7f800000, v65, vcc_lo
.LBB0_346:                              ;   in Loop: Header=BB0_334 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16 v44, v62 offset:14476
	v_or_b32_e32 v69, 6, v61
	v_mov_b32_e32 v70, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s58, v69
	v_mov_b32_e32 v69, 0
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB0_348
; %bb.347:                              ;   in Loop: Header=BB0_334 Depth=2
	ds_load_b32 v70, v48 offset:28824
	s_wait_dscnt 0x0
	v_sub_f32_e32 v70, v46, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v71, 0x3fb8aa3b, v70
	v_fma_f32 v73, 0x3fb8aa3b, v70, -v71
	v_rndne_f32_e32 v74, v71
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v73, 0x32a5705f, v70
	v_sub_f32_e32 v71, v71, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v71, v71, v73
	v_cvt_i32_f32_e32 v73, v74
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v70
	v_exp_f32_e32 v71, v71
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v71, v71, v73
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v71, 0, v71, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v70
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v70, 0x7f800000, v71, vcc_lo
.LBB0_348:                              ;   in Loop: Header=BB0_334 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	ds_load_u16_d16_hi v44, v62 offset:14478
	v_or_b32_e32 v61, 7, v61
	s_mov_b32 s7, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s58, v61
	s_cbranch_execz .LBB0_333
; %bb.349:                              ;   in Loop: Header=BB0_334 Depth=2
	ds_load_b32 v48, v48 offset:28828
	s_wait_dscnt 0x0
	v_sub_f32_e32 v48, v46, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v61, 0x3fb8aa3b, v48
	v_fma_f32 v62, 0x3fb8aa3b, v48, -v61
	v_rndne_f32_e32 v69, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v61, v61, v69
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v48
	v_fmac_f32_e32 v62, 0x32a5705f, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v61, v61, v62
	v_cvt_i32_f32_e32 v62, v69
	v_exp_f32_e32 v61, v61
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v61, v61, v62
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v61, 0, v61, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v48
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v69, 0x7f800000, v61, vcc_lo
	s_branch .LBB0_333
.LBB0_350:                              ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[102:103], null, v74, s50, 0
	v_add_f32_e32 v41, v41, v100
	ds_load_b32 v100, v97 offset:29056
	v_mad_co_u64_u32 v[103:104], null, v74, s51, v[103:104]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[102:103], 2, v[102:103]
	s_wait_dscnt 0x0
	v_fma_f32 v33, v41, v100, v33
	v_add_co_u32 v102, s40, v160, v102
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v103, null, v161, v103, s40
	global_store_b32 v[102:103], v33, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s42
	s_and_saveexec_b32 s40, s39
	s_cbranch_execz .LBB0_225
.LBB0_351:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v41, v97 offset:29060
	v_mad_co_u64_u32 v[102:103], null, v73, s50, 0
	v_add_f32_e32 v33, v42, v99
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[103:104], null, v73, s51, v[103:104]
	s_wait_dscnt 0x0
	v_fma_f32 v41, v33, v41, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[102:103]
	v_add_co_u32 v33, s39, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s39
	global_store_b32 v[33:34], v41, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s40
	s_and_saveexec_b32 s39, s38
	s_cbranch_execz .LBB0_226
.LBB0_352:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29064
	v_add_f32_e32 v33, v43, v98
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v33, v34, v35
	v_mad_co_u64_u32 v[33:34], null, v66, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v66, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s38, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s38
	global_store_b32 v[33:34], v41, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s39
	s_and_saveexec_b32 s38, s37
	s_cbranch_execz .LBB0_227
.LBB0_353:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29068
	v_add_f32_e32 v33, v44, v132
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v36
	v_mad_co_u64_u32 v[33:34], null, v65, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v65, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s37, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s37
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s38
	s_and_saveexec_b32 s37, s36
	s_cbranch_execz .LBB0_228
.LBB0_354:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29072
	v_add_f32_e32 v33, v45, v84
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v37
	v_mad_co_u64_u32 v[33:34], null, v71, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v71, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s36, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s36
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s37
	s_and_saveexec_b32 s36, s35
	s_cbranch_execz .LBB0_229
.LBB0_355:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29076
	v_add_f32_e32 v33, v46, v82
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v38
	v_mad_co_u64_u32 v[33:34], null, v70, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v70, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s35, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s35
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s36
	s_and_saveexec_b32 s35, s34
	s_cbranch_execz .LBB0_230
.LBB0_356:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29080
	v_add_f32_e32 v33, v47, v79
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v39
	v_mad_co_u64_u32 v[33:34], null, v69, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v69, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s34, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s34
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s35
	s_and_b32 exec_lo, exec_lo, s33
	s_cbranch_execnz .LBB0_231
	s_branch .LBB0_232
.LBB0_357:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_f32_e32 v41, v41, v75
	ds_load_b32 v75, v97 offset:29120
	s_wait_dscnt 0x0
	v_fma_f32 v33, v41, v75, v33
	v_mad_co_u64_u32 v[75:76], null, v191, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[76:77], null, v191, s51, v[76:77]
	v_lshlrev_b64_e32 v[75:76], 2, v[75:76]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v75, s23, v160, v75
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v76, null, v161, v76, s23
	global_store_b32 v[75:76], v33, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s33
	s_and_saveexec_b32 s33, s24
	s_cbranch_execz .LBB0_239
.LBB0_358:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v41, v97 offset:29124
	v_add_f32_e32 v33, v42, v74
	v_mad_co_u64_u32 v[74:75], null, v185, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[75:76], null, v185, s51, v[75:76]
	s_wait_dscnt 0x0
	v_fma_f32 v41, v33, v41, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[74:75]
	v_add_co_u32 v33, s23, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s23
	global_store_b32 v[33:34], v41, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s33
	s_and_saveexec_b32 s24, s25
	s_cbranch_execz .LBB0_240
.LBB0_359:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29128
	v_add_f32_e32 v33, v43, v73
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v33, v34, v35
	v_mad_co_u64_u32 v[33:34], null, v186, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v186, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s23, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s23
	global_store_b32 v[33:34], v41, off
	s_or_b32 exec_lo, exec_lo, s24
	s_and_saveexec_b32 s24, s26
	s_cbranch_execz .LBB0_241
.LBB0_360:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29132
	v_add_f32_e32 v33, v44, v71
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v36
	v_mad_co_u64_u32 v[33:34], null, v187, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v187, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s23, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s23
	global_store_b32 v[33:34], v36, off
	s_or_b32 exec_lo, exec_lo, s24
	s_and_saveexec_b32 s24, s27
	s_cbranch_execz .LBB0_242
.LBB0_361:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29136
	v_add_f32_e32 v33, v45, v70
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v37
	v_mad_co_u64_u32 v[33:34], null, v188, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v188, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s23, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s23
	global_store_b32 v[33:34], v36, off
	s_or_b32 exec_lo, exec_lo, s24
	s_and_saveexec_b32 s24, s28
	s_cbranch_execz .LBB0_243
.LBB0_362:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29140
	v_add_f32_e32 v33, v46, v69
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v38
	v_mad_co_u64_u32 v[33:34], null, v189, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v189, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s23, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s23
	global_store_b32 v[33:34], v36, off
	s_or_b32 exec_lo, exec_lo, s24
	s_and_saveexec_b32 s24, s29
	s_cbranch_execz .LBB0_244
.LBB0_363:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29144
	v_add_f32_e32 v33, v47, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v39
	v_mad_co_u64_u32 v[33:34], null, v190, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v190, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s23, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s23
	global_store_b32 v[33:34], v36, off
	s_or_b32 exec_lo, exec_lo, s24
	s_and_b32 exec_lo, exec_lo, s30
	s_cbranch_execnz .LBB0_245
	s_branch .LBB0_246
.LBB0_364:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_f32_e32 v41, v41, v75
	ds_load_b32 v75, v97 offset:29184
	s_wait_dscnt 0x0
	v_fma_f32 v33, v41, v75, v33
	v_mad_co_u64_u32 v[75:76], null, v174, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[76:77], null, v174, s51, v[76:77]
	v_lshlrev_b64_e32 v[75:76], 2, v[75:76]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v75, s13, v160, v75
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v76, null, v161, v76, s13
	global_store_b32 v[75:76], v33, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s23
	s_and_saveexec_b32 s23, s14
	s_cbranch_execz .LBB0_253
.LBB0_365:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v41, v97 offset:29188
	v_add_f32_e32 v33, v42, v74
	v_mad_co_u64_u32 v[74:75], null, v175, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[75:76], null, v175, s51, v[75:76]
	s_wait_dscnt 0x0
	v_fma_f32 v41, v33, v41, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[74:75]
	v_add_co_u32 v33, s13, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s13
	global_store_b32 v[33:34], v41, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s23
	s_and_saveexec_b32 s14, s15
	s_cbranch_execz .LBB0_254
.LBB0_366:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29192
	v_add_f32_e32 v33, v43, v73
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v33, v34, v35
	v_mad_co_u64_u32 v[33:34], null, v176, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v176, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s13, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s13
	global_store_b32 v[33:34], v41, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s16
	s_cbranch_execz .LBB0_255
.LBB0_367:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29196
	v_add_f32_e32 v33, v44, v71
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v36
	v_mad_co_u64_u32 v[33:34], null, v177, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v177, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s13, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s13
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s17
	s_cbranch_execz .LBB0_256
.LBB0_368:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29200
	v_add_f32_e32 v33, v45, v70
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v37
	v_mad_co_u64_u32 v[33:34], null, v178, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v178, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s13, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s13
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s18
	s_cbranch_execz .LBB0_257
.LBB0_369:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29204
	v_add_f32_e32 v33, v46, v69
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v38
	v_mad_co_u64_u32 v[33:34], null, v179, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v179, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s13, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s13
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s20
	s_cbranch_execz .LBB0_258
.LBB0_370:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29208
	v_add_f32_e32 v33, v47, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v39
	v_mad_co_u64_u32 v[33:34], null, v180, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v180, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, s13, v160, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v34, null, v161, v34, s13
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_b32 exec_lo, exec_lo, s21
	s_cbranch_execnz .LBB0_259
	s_branch .LBB0_260
.LBB0_371:                              ;   in Loop: Header=BB0_3 Depth=1
	v_add_f32_e32 v41, v41, v75
	ds_load_b32 v75, v97 offset:29248
	s_wait_dscnt 0x0
	v_fma_f32 v33, v41, v75, v33
	scratch_load_b32 v41, off, off offset:8 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	v_mad_co_u64_u32 v[75:76], null, v41, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[76:77], null, v41, s51, v[76:77]
	v_lshlrev_b64_e32 v[75:76], 2, v[75:76]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v75, vcc_lo, v160, v75
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, v161, v76, vcc_lo
	global_store_b32 v[75:76], v33, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s14, s6
	s_cbranch_execz .LBB0_267
.LBB0_372:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v41, v97 offset:29252
	v_add_f32_e32 v33, v42, v74
	v_mad_co_u64_u32 v[74:75], null, v167, s50, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[75:76], null, v167, s51, v[75:76]
	s_wait_dscnt 0x0
	v_fma_f32 v41, v33, v41, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[74:75]
	v_add_co_u32 v33, vcc_lo, v160, v33
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, vcc_lo
	global_store_b32 v[33:34], v41, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_and_saveexec_b32 s6, s7
	s_cbranch_execz .LBB0_268
.LBB0_373:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29256
	v_add_f32_e32 v33, v43, v73
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v41, v33, v34, v35
	v_mad_co_u64_u32 v[33:34], null, v168, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v168, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, vcc_lo, v160, v33
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, vcc_lo
	global_store_b32 v[33:34], v41, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s8
	s_cbranch_execz .LBB0_269
.LBB0_374:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29260
	v_add_f32_e32 v33, v44, v71
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v36
	v_mad_co_u64_u32 v[33:34], null, v169, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v169, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, vcc_lo, v160, v33
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, vcc_lo
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s9
	s_cbranch_execz .LBB0_270
.LBB0_375:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29264
	v_add_f32_e32 v33, v45, v70
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v37
	v_mad_co_u64_u32 v[33:34], null, v170, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v170, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, vcc_lo, v160, v33
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, vcc_lo
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s10
	s_cbranch_execz .LBB0_271
.LBB0_376:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29268
	v_add_f32_e32 v33, v46, v69
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v38
	v_mad_co_u64_u32 v[33:34], null, v171, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v171, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, vcc_lo, v160, v33
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v161, v34, vcc_lo
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_saveexec_b32 s6, s11
	s_cbranch_execz .LBB0_272
.LBB0_377:                              ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v34, v97 offset:29272
	v_add_f32_e32 v33, v47, v66
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v36, v33, v34, v39
	v_mad_co_u64_u32 v[33:34], null, v172, s50, 0
	v_mad_co_u64_u32 v[34:35], null, v172, s51, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]
	v_add_co_u32 v33, vcc_lo, v160, v33
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v34, null, v161, v34, vcc_lo
	global_store_b32 v[33:34], v36, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 exec_lo, exec_lo, s12
	s_cbranch_execnz .LBB0_273
	s_branch .LBB0_274
.LBB0_378:                              ; %._crit_edge
	scratch_load_b32 v1, off, off offset:52 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_load_b64 s[0:1], s[0:1], 0x50
	v_cvt_f32_f16_e32 v39, v52.h
	v_cvt_f32_f16_e32 v38, v52.l
	v_cvt_f32_f16_e32 v68, v57.h
	v_cvt_f32_f16_e32 v73, v58.h
	v_cvt_f32_f16_e32 v67, v59.h
	v_cvt_f32_f16_e32 v70, v60.h
	v_cvt_f32_f16_e32 v71, v57.l
	v_cvt_f32_f16_e32 v75, v58.l
	v_cvt_f32_f16_e32 v69, v59.l
	v_cvt_f32_f16_e32 v74, v60.l
	v_cvt_f32_f16_e32 v59, v53.h
	v_cvt_f32_f16_e32 v65, v54.h
	v_cvt_f32_f16_e32 v57, v55.h
	v_cvt_f32_f16_e32 v58, v56.h
	v_cvt_f32_f16_e32 v60, v53.l
	v_cvt_f32_f16_e32 v66, v54.l
	v_cvt_f32_f16_e32 v53, v55.l
	v_cvt_f32_f16_e32 v54, v56.l
	v_cvt_f32_f16_e32 v40, v49.h
	s_wait_kmcnt 0x0
	s_cmp_lg_u64 s[0:1], 0
	v_cvt_f32_f16_e32 v42, v50.h
	s_cselect_b32 s2, -1, 0
	s_lshl_b64 s[4:5], s[64:65], 1
	v_cvt_f32_f16_e32 v36, v51.h
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	s_lshl_b64 s[4:5], s[62:63], 1
	v_cvt_f32_f16_e32 v41, v49.l
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	v_cvt_f32_f16_e32 v43, v50.l
	v_cvt_f32_f16_e32 v37, v51.l
	v_cvt_f32_f16_e32 v29, v61.h
	v_cvt_f32_f16_e32 v31, v62.h
	v_cvt_f32_f16_e32 v3, v63.h
	v_cvt_f32_f16_e32 v27, v64.h
	v_cvt_f32_f16_e32 v30, v61.l
	v_cvt_f32_f16_e32 v32, v62.l
	v_cvt_f32_f16_e32 v2, v63.l
	v_cvt_f32_f16_e32 v28, v64.l
	s_and_b32 vcc_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 1, v1
	scratch_load_b32 v1, off, off offset:48 th:TH_LOAD_LU ; 4-byte Folded Reload
	v_add_co_u32 v52, s0, s0, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v55, null, s1, 0, s0
	s_wait_loadcnt 0x0
	v_or_b32_e32 v1, v78, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v56, 1, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_380
; %bb.379:                              ; %.preheader942
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v16, vcc_lo, v52, v56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, 0, v55, vcc_lo
	s_clause 0x3
	global_load_b128 v[4:7], v[16:17], off
	global_load_b128 v[8:11], v[16:17], off offset:32
	global_load_b128 v[12:15], v[16:17], off offset:64
	global_load_b128 v[16:19], v[16:17], off offset:96
	s_wait_loadcnt 0x3
	v_fma_mix_f32 v71, v4, 1.0, v71 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v68, v4, 1.0, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v75, v5, 1.0, v75 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v73, v5, 1.0, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v69, v6, 1.0, v69 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v67, v6, 1.0, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v74, v7, 1.0, v74 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v70, v7, 1.0, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_wait_loadcnt 0x2
	v_fma_mix_f32 v60, v8, 1.0, v60 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v59, v8, 1.0, v59 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v66, v9, 1.0, v66 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v65, v9, 1.0, v65 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v53, v10, 1.0, v53 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v57, v10, 1.0, v57 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v54, v11, 1.0, v54 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v58, v11, 1.0, v58 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_wait_loadcnt 0x1
	v_fma_mix_f32 v41, v12, 1.0, v41 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v40, v12, 1.0, v40 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v43, v13, 1.0, v43 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v42, v13, 1.0, v42 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v37, v14, 1.0, v37 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v36, v14, 1.0, v36 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v38, v15, 1.0, v38 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v39, v15, 1.0, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v30, v16, 1.0, v30 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v29, v16, 1.0, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v32, v17, 1.0, v32 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v31, v17, 1.0, v31 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v2, v18, 1.0, v2 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v3, v18, 1.0, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v28, v19, 1.0, v28 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v27, v19, 1.0, v27 op_sel:[1,0,0] op_sel_hi:[1,1,0]
.LBB0_380:                              ; %.loopexit
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v89, off, off offset:32
	scratch_load_b32 v90, off, off offset:36
	v_max3_num_f32 v4, |v71|, 0, |v68|
	v_lshlrev_b32_e32 v5, 2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v4, v4, |v75|, |v73|
	v_max3_num_f32 v4, v4, |v69|, |v67|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v4, v4, |v74|, |v70|
	v_max3_num_f32 v4, v4, |v60|, |v59|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v4, v4, |v66|, |v65|
	v_max3_num_f32 v4, v4, |v53|, |v57|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v4, v4, |v54|, |v58|
	v_max3_num_f32 v4, v4, |v41|, |v40|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v4, v4, |v43|, |v42|
	v_max3_num_f32 v4, v4, |v37|, |v36|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v4, v4, |v38|, |v39|
	v_max3_num_f32 v4, v4, |v30|, |v29|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v4, v4, |v32|, |v31|
	v_max3_num_f32 v4, v4, |v2|, |v3|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_max3_num_f32 v4, v4, |v28|, |v27|
	ds_store_b32 v5, v4
	v_and_b32_e32 v4, 0x3cf, v0
	v_lshl_or_b32 v0, v0, 2, 0xc0
	v_lshlrev_b32_e32 v6, 2, v4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_b32 v[4:5], v6 offset1:16
	ds_load_b32 v6, v6 offset:128
	ds_load_b32 v0, v0
	s_wait_dscnt 0x2
	v_dual_max_num_f32 v5, v5, v5 :: v_dual_max_num_f32 v4, v4, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v4, v4, v5
	s_wait_dscnt 0x0
	v_max3_num_f32 v4, v4, v6, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_scale_f32 v0, null, v4, v4, 0x42fe0000
	v_div_scale_f32 v7, vcc_lo, 0x42fe0000, v4, 0x42fe0000
	v_cmp_lt_f32_e64 s0, 0, v4
	v_rcp_f32_e32 v5, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v0, v5, 1.0
	v_fmac_f32_e32 v5, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, v7, v5
	v_fma_f32 v8, -v0, v6, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v8, v5
	v_fma_f32 v0, -v0, v6, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v0, v0, v5, v6
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v5, off, off offset:40 th:TH_LOAD_LU
	scratch_load_b32 v6, off, off offset:44 th:TH_LOAD_LU
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_mov_b32 s2, 0
	v_div_fixup_f32 v0, v0, v4, 0x42fe0000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v0, 0, v0, s0
	v_dual_mul_f32 v49, v0, v71 :: v_dual_mul_f32 v50, v0, v68
	v_mul_f32_e32 v45, v0, v69
	v_dual_mul_f32 v51, v0, v75 :: v_dual_mul_f32 v44, v0, v73
	v_dual_mul_f32 v46, v0, v67 :: v_dual_mul_f32 v47, v0, v74
	v_dual_mul_f32 v48, v0, v70 :: v_dual_mul_f32 v33, v0, v60
	v_dual_mul_f32 v34, v0, v59 :: v_dual_mul_f32 v35, v0, v66
	v_dual_mul_f32 v22, v0, v65 :: v_dual_mul_f32 v25, v0, v54
	v_dual_mul_f32 v23, v0, v53 :: v_dual_mul_f32 v26, v0, v58
	v_dual_mul_f32 v24, v0, v57 :: v_dual_mul_f32 v21, v0, v43
	v_dual_mul_f32 v19, v0, v41 :: v_dual_mul_f32 v20, v0, v40
	v_dual_mul_f32 v14, v0, v42 :: v_dual_mul_f32 v15, v0, v37
	v_dual_mul_f32 v16, v0, v36 :: v_dual_mul_f32 v17, v0, v38
	v_dual_mul_f32 v18, v0, v39 :: v_dual_mul_f32 v11, v0, v30
	v_dual_mul_f32 v12, v0, v29 :: v_dual_mul_f32 v13, v0, v32
	v_dual_mul_f32 v8, v0, v3 :: v_dual_mul_f32 v9, v0, v28
	v_dual_mul_f32 v10, v0, v27 :: v_dual_mul_f32 v7, v0, v2
	s_wait_loadcnt 0x0
	v_or_b32_e32 v5, v6, v5
	v_mul_f32_e32 v6, v0, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e64 s1, 0, v5
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_385
; %bb.381:                              ; %.preheader939
	v_div_scale_f32 v0, null, 0x42fe0000, 0x42fe0000, v4
	v_div_scale_f32 v62, vcc_lo, v4, 0x42fe0000, v4
	v_add_co_u32 v76, s2, v52, v56
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v77, null, 0, v55, s2
	v_rcp_f32_e32 v61, v0
	v_rndne_f32_e32 v55, v51
	v_rndne_f32_e32 v56, v44
	v_rndne_f32_e32 v80, v45
	v_rndne_f32_e32 v81, v46
	v_rndne_f32_e32 v82, v47
	s_mov_b32 s3, 0xc3000000
	v_rndne_f32_e32 v83, v48
	s_wait_alu depctr_sa_sdst(0)
	v_med3_num_f32 v86, v55, s3, 0x42fe0000
	v_fma_f32 v64, -v0, v61, 1.0
	v_med3_num_f32 v87, v56, s3, 0x42fe0000
	v_med3_num_f32 v80, v80, s3, 0x42fe0000
	v_med3_num_f32 v81, v81, s3, 0x42fe0000
	v_med3_num_f32 v82, v82, s3, 0x42fe0000
	v_fmac_f32_e32 v61, v64, v61
	v_rndne_f32_e32 v52, v50
	v_med3_num_f32 v83, v83, s3, 0x42fe0000
	v_rndne_f32_e32 v63, v49
	v_cvt_i32_f32_e32 v56, v87
	v_mul_f32_e32 v64, v62, v61
	v_cvt_i32_f32_e32 v55, v82
	v_med3_num_f32 v85, v52, s3, 0x42fe0000
	v_cvt_i32_f32_e32 v52, v83
	v_med3_num_f32 v63, v63, s3, 0x42fe0000
	v_fma_f32 v84, -v0, v64, v62
	v_and_b16 v52.h, 0xff, v55.l
	v_lshlrev_b16 v56.l, 8, v56.l
	v_lshlrev_b16 v52.l, 8, v52.l
	v_cvt_i32_f32_e32 v88, v63
	v_fmac_f32_e32 v64, v84, v61
	v_cvt_i32_f32_e32 v84, v85
	v_add_co_u32 v78, s2, v89, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, 0, v90, s2
	v_fma_f32 v0, -v0, v64, v62
	v_cvt_i32_f32_e32 v62, v86
	s_and_b32 s2, s1, exec_lo
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_div_fmas_f32 v0, v0, v61, v64
	v_cvt_i32_f32_e32 v61, v81
	v_cvt_i32_f32_e32 v64, v80
	v_and_b16 v56.h, 0xff, v62.l
	v_div_fixup_f32 v0, v0, 0x42fe0000, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b16 v55.l, 8, v61.l
	v_and_b16 v55.h, 0xff, v64.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v0, 1.0, v0, s0
	v_fma_mixhi_f16 v61, -v0, v85, v68
	v_fma_mixlo_f16 v61, -v0, v63, v71
	v_fma_mixhi_f16 v63, -v0, v81, v67
	v_fma_mixlo_f16 v63, -v0, v80, v69
	v_or_b16 v68.l, v55.h, v55.l
	v_or_b16 v67.h, v56.h, v56.l
	v_rndne_f32_e32 v55, v22
	v_rndne_f32_e32 v56, v35
	v_rndne_f32_e32 v69, v34
	v_fma_mixhi_f16 v62, -v0, v87, v73
	v_or_b16 v68.h, v52.h, v52.l
	v_lshlrev_b16 v52.l, 8, v84.l
	v_and_b16 v52.h, 0xff, v88.l
	v_med3_num_f32 v71, v55, s3, 0x42fe0000
	v_med3_num_f32 v73, v56, s3, 0x42fe0000
	v_med3_num_f32 v69, v69, s3, 0x42fe0000
	v_fma_mixhi_f16 v64, -v0, v83, v70
	v_rndne_f32_e32 v70, v33
	v_or_b16 v67.l, v52.h, v52.l
	v_fma_mixhi_f16 v56, -v0, v71, v65
	v_fma_mixlo_f16 v56, -v0, v73, v66
	v_fma_mixhi_f16 v55, -v0, v69, v59
	v_cvt_i32_f32_e32 v59, v71
	v_rndne_f32_e32 v52, v26
	v_rndne_f32_e32 v65, v25
	v_rndne_f32_e32 v66, v24
	v_rndne_f32_e32 v71, v23
	v_med3_num_f32 v70, v70, s3, 0x42fe0000
	v_med3_num_f32 v52, v52, s3, 0x42fe0000
	v_med3_num_f32 v65, v65, s3, 0x42fe0000
	v_med3_num_f32 v66, v66, s3, 0x42fe0000
	v_med3_num_f32 v71, v71, s3, 0x42fe0000
	v_fma_mixlo_f16 v55, -v0, v70, v60
	v_cvt_i32_f32_e32 v60, v73
	v_fma_mixhi_f16 v58, -v0, v52, v58
	v_fma_mixlo_f16 v58, -v0, v65, v54
	v_fma_mixhi_f16 v57, -v0, v66, v57
	v_cvt_i32_f32_e32 v54, v65
	v_cvt_i32_f32_e32 v65, v66
	v_cvt_i32_f32_e32 v66, v71
	v_cvt_i32_f32_e32 v52, v52
	v_fma_mixlo_f16 v62, -v0, v86, v75
	v_fma_mixlo_f16 v64, -v0, v82, v74
	v_fma_mixlo_f16 v57, -v0, v71, v53
	v_and_b16 v52.h, 0xff, v54.l
	v_lshlrev_b16 v53.l, 8, v65.l
	v_and_b16 v53.h, 0xff, v66.l
	v_lshlrev_b16 v54.l, 8, v59.l
	v_and_b16 v54.h, 0xff, v60.l
	v_cvt_i32_f32_e32 v69, v69
	v_cvt_i32_f32_e32 v70, v70
	v_lshlrev_b16 v52.l, 8, v52.l
	s_clause 0x1
	global_store_b128 v[76:77], v[61:64], off
	global_store_b128 v[76:77], v[55:58], off offset:32
	v_or_b16 v57.l, v53.h, v53.l
	v_or_b16 v56.h, v54.h, v54.l
	v_rndne_f32_e32 v53, v14
	v_rndne_f32_e32 v54, v21
	v_rndne_f32_e32 v55, v20
	v_rndne_f32_e32 v58, v19
	v_or_b16 v57.h, v52.h, v52.l
	v_lshlrev_b16 v52.l, 8, v69.l
	v_and_b16 v52.h, 0xff, v70.l
	v_med3_num_f32 v59, v53, s3, 0x42fe0000
	v_med3_num_f32 v54, v54, s3, 0x42fe0000
	v_med3_num_f32 v55, v55, s3, 0x42fe0000
	v_med3_num_f32 v58, v58, s3, 0x42fe0000
	v_or_b16 v56.l, v52.h, v52.l
	v_fma_mixhi_f16 v53, -v0, v59, v42
	v_fma_mixlo_f16 v53, -v0, v54, v43
	v_fma_mixhi_f16 v52, -v0, v55, v40
	v_fma_mixlo_f16 v52, -v0, v58, v41
	v_cvt_i32_f32_e32 v40, v59
	v_cvt_i32_f32_e32 v41, v54
	v_rndne_f32_e32 v42, v18
	v_cvt_i32_f32_e32 v43, v55
	v_rndne_f32_e32 v54, v17
	v_rndne_f32_e32 v55, v16
	v_rndne_f32_e32 v59, v15
	v_med3_num_f32 v42, v42, s3, 0x42fe0000
	v_cvt_i32_f32_e32 v58, v58
	v_med3_num_f32 v54, v54, s3, 0x42fe0000
	v_med3_num_f32 v60, v55, s3, 0x42fe0000
	v_med3_num_f32 v59, v59, s3, 0x42fe0000
	v_fma_mixhi_f16 v55, -v0, v42, v39
	v_cvt_i32_f32_e32 v39, v42
	v_cvt_i32_f32_e32 v42, v54
	v_cvt_i32_f32_e32 v61, v60
	v_cvt_i32_f32_e32 v62, v59
	v_fma_mixlo_f16 v55, -v0, v54, v38
	v_lshlrev_b16 v38.l, 8, v39.l
	v_and_b16 v38.h, 0xff, v42.l
	v_lshlrev_b16 v39.l, 8, v61.l
	v_and_b16 v39.h, 0xff, v62.l
	v_and_b16 v40.h, 0xff, v41.l
	v_fma_mixlo_f16 v54, -v0, v59, v37
	v_or_b16 v41.h, v38.h, v38.l
	v_rndne_f32_e32 v37, v6
	v_or_b16 v41.l, v39.h, v39.l
	v_rndne_f32_e32 v38, v13
	v_rndne_f32_e32 v39, v12
	v_rndne_f32_e32 v42, v11
	v_lshlrev_b16 v40.l, 8, v40.l
	v_fma_mixhi_f16 v54, -v0, v60, v36
	v_lshlrev_b16 v36.l, 8, v43.l
	v_and_b16 v36.h, 0xff, v58.l
	v_med3_num_f32 v43, v37, s3, 0x42fe0000
	v_med3_num_f32 v38, v38, s3, 0x42fe0000
	v_med3_num_f32 v39, v39, s3, 0x42fe0000
	v_med3_num_f32 v42, v42, s3, 0x42fe0000
	v_or_b16 v40.h, v40.h, v40.l
	v_or_b16 v40.l, v36.h, v36.l
	v_fma_mixhi_f16 v37, -v0, v43, v31
	v_fma_mixlo_f16 v37, -v0, v38, v32
	v_fma_mixhi_f16 v36, -v0, v39, v29
	v_fma_mixlo_f16 v36, -v0, v42, v30
	v_cvt_i32_f32_e32 v29, v43
	v_cvt_i32_f32_e32 v30, v38
	v_rndne_f32_e32 v31, v10
	v_rndne_f32_e32 v32, v9
	v_rndne_f32_e32 v38, v8
	v_cvt_i32_f32_e32 v43, v39
	v_rndne_f32_e32 v39, v7
	v_med3_num_f32 v31, v31, s3, 0x42fe0000
	v_med3_num_f32 v32, v32, s3, 0x42fe0000
	v_med3_num_f32 v58, v38, s3, 0x42fe0000
	v_cvt_i32_f32_e32 v42, v42
	v_med3_num_f32 v59, v39, s3, 0x42fe0000
	v_fma_mixhi_f16 v39, -v0, v31, v27
	v_fma_mixlo_f16 v39, -v0, v32, v28
	v_fma_mixhi_f16 v38, -v0, v58, v3
	v_cvt_i32_f32_e32 v3, v31
	v_cvt_i32_f32_e32 v27, v32
	v_cvt_i32_f32_e32 v28, v58
	v_cvt_i32_f32_e32 v31, v59
	v_fma_mixlo_f16 v38, -v0, v59, v2
	v_lshlrev_b16 v2.l, 8, v3.l
	v_and_b16 v2.h, 0xff, v27.l
	v_lshlrev_b16 v3.l, 8, v28.l
	v_and_b16 v3.h, 0xff, v31.l
	v_lshlrev_b16 v27.l, 8, v29.l
	v_and_b16 v27.h, 0xff, v30.l
	v_lshlrev_b16 v28.l, 8, v43.l
	v_and_b16 v28.h, 0xff, v42.l
	v_or_b16 v30.h, v2.h, v2.l
	v_or_b16 v30.l, v3.h, v3.l
	v_or_b16 v29.h, v27.h, v27.l
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b16 v29.l, v28.h, v28.l
	s_clause 0x1
	global_store_b128 v[76:77], v[52:55], off offset:64
	global_store_b128 v[76:77], v[36:39], off offset:96
	s_clause 0x3
	global_store_b64 v[78:79], v[67:68], off
	global_store_b64 v[78:79], v[56:57], off offset:16
	global_store_b64 v[78:79], v[40:41], off offset:32
	global_store_b64 v[78:79], v[29:30], off offset:48
	s_cbranch_execz .LBB0_386
; %bb.382:                              ; %Flow2746
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_384
.LBB0_383:                              ; %.sink.split
	scratch_load_b64 v[1:2], off, off offset:60 th:TH_LOAD_LU ; 8-byte Folded Reload
	s_wait_loadcnt 0x0
	global_store_b32 v[1:2], v0, off
.LBB0_384:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.LBB0_385:
                                        ; implicit-def: $vgpr0
.LBB0_386:                              ; %.preheader
	scratch_load_b32 v0, off, off offset:56 th:TH_LOAD_LU ; 4-byte Folded Reload
	s_mul_i32 s1, ttmp9, 0x19660d
	s_mov_b32 s3, 0x19660d
	s_wait_alu depctr_sa_sdst(0)
	v_mad_u32_u24 v2, 0x343fd, v72, s1
	s_mul_i32 s1, s59, 0x269ec3
	s_mov_b32 s4, 0x17385ca9
	s_mov_b32 s5, 0xaf490a95
	s_mov_b32 s7, 0xea890021
	s_mov_b32 s6, 0x6e587165
	s_wait_loadcnt 0x0
	v_mul_lo_u32 v0, 0x3c6ef35f, v0
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v27, v2, v0, s1
	s_mov_b32 s1, 0xc3000000
	v_add_co_u32 v0, vcc_lo, v89, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v90, vcc_lo
	v_mad_co_u64_u32 v[2:3], null, v27, s3, 0x3c6ef35f
	v_mad_co_u64_u32 v[28:29], null, v27, s4, 0x47502932
	v_mad_co_u64_u32 v[29:30], null, v27, s5, 0xffffffffd1ccf6e9
	s_mov_b32 s5, 0xbf69fab9
	s_mov_b32 s4, 0xaa9d885d
	s_mov_b32 s3, 0x979e791
	v_lshrrev_b32_e32 v2, 8, v2
	v_lshrrev_b32_e32 v3, 8, v28
	v_lshrrev_b32_e32 v28, 8, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_u32_e32 v30, v2
	v_cvt_f32_u32_e32 v31, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_u32_e32 v32, v28
	v_mad_co_u64_u32 v[28:29], null, v27, s7, 0xffffffffa3d95fa8
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[2:3], null, v27, s4, 0x6252e503
	v_dual_fmac_f32 v49, 0x33800000, v30 :: v_dual_fmac_f32 v50, 0x33800000, v31
	v_mad_co_u64_u32 v[29:30], null, v27, s5, 0xffffffff9f2ec686
	v_mad_co_u64_u32 v[30:31], null, v27, s6, 0x57fe6c2d
	s_mov_b32 s4, 0x74275d35
	v_floor_f32_e32 v3, v49
	v_lshrrev_b32_e32 v28, 8, v28
	v_lshrrev_b32_e32 v2, 8, v2
	v_floor_f32_e32 v36, v50
	s_mov_b32 s6, 0x77520441
	v_lshrrev_b32_e32 v29, 8, v29
	v_med3_num_f32 v38, v3, s1, 0x42fe0000
	v_lshrrev_b32_e32 v3, 8, v30
	v_cvt_f32_u32_e32 v28, v28
	v_cvt_f32_u32_e32 v2, v2
	v_cvt_f32_u32_e32 v29, v29
	v_fmac_f32_e32 v51, 0x33800000, v32
	v_mad_co_u64_u32 v[31:32], null, v27, s3, 0xffffffffaaf95334
	v_cvt_f32_u32_e32 v3, v3
	v_fmac_f32_e32 v48, 0x33800000, v28
	v_dual_fmac_f32 v46, 0x33800000, v29 :: v_dual_fmac_f32 v45, 0x33800000, v2
	s_mov_b32 s3, 0x823b27ad
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v47, 0x33800000, v3
	v_floor_f32_e32 v2, v48
	v_lshrrev_b32_e32 v30, 8, v31
	v_floor_f32_e32 v28, v46
	v_floor_f32_e32 v29, v45
	v_floor_f32_e32 v3, v47
	v_med3_num_f32 v2, v2, s1, 0x42fe0000
	v_cvt_f32_u32_e32 v30, v30
	v_med3_num_f32 v28, v28, s1, 0x42fe0000
	v_med3_num_f32 v29, v29, s1, 0x42fe0000
	v_med3_num_f32 v3, v3, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v2, v2
	v_fmac_f32_e32 v44, 0x33800000, v30
	v_cvt_i32_f32_e32 v28, v28
	v_cvt_i32_f32_e32 v29, v29
	v_cvt_i32_f32_e32 v3, v3
	v_floor_f32_e32 v37, v51
	v_floor_f32_e32 v30, v44
	v_med3_num_f32 v36, v36, s1, 0x42fe0000
	v_lshlrev_b16 v2.l, 8, v2.l
	v_and_b16 v2.h, 0xff, v3.l
	v_lshlrev_b16 v3.l, 8, v28.l
	v_med3_num_f32 v30, v30, s1, 0x42fe0000
	v_and_b16 v28.l, 0xff, v29.l
	v_med3_num_f32 v37, v37, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v36, v36
	v_or_b16 v3.h, v2.h, v2.l
	v_cvt_i32_f32_e32 v39, v30
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, v27, s3, 0xffffffff81fdbee7
	s_mov_b32 s3, 0xeb4f1c9
	v_or_b16 v3.l, v28.l, v3.l
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[30:31], null, v27, s3, 0xffffffff94f0af1a
	v_mad_co_u64_u32 v[31:32], null, v27, s4, 0xffffffffcbf633b1
	v_cvt_i32_f32_e32 v32, v37
	v_cvt_i32_f32_e32 v37, v38
	v_lshlrev_b16 v2.l, 8, v39.l
	v_lshrrev_b32_e32 v28, 8, v29
	s_mov_b32 s7, 0x3a739e05
	v_and_b16 v2.h, 0xff, v32.l
	v_lshrrev_b32_e32 v29, 8, v30
	v_lshrrev_b32_e32 v30, 8, v31
	v_cvt_f32_u32_e32 v31, v28
	v_lshlrev_b16 v28.l, 8, v36.l
	v_and_b16 v28.h, 0xff, v37.l
	v_cvt_f32_u32_e32 v29, v29
	v_or_b16 v2.h, v2.h, v2.l
	v_cvt_f32_u32_e32 v30, v30
	s_mov_b32 s5, 0xf3aa51d9
	v_or_b16 v2.l, v28.h, v28.l
	v_fmac_f32_e32 v34, 0x33800000, v29
	v_mad_co_u64_u32 v[28:29], null, v27, s6, 0xffffffff83c6b450
	v_fmac_f32_e32 v33, 0x33800000, v31
	s_mov_b32 s4, 0xfa1393fd
	s_mov_b32 s3, 0xaf4fd9b1
	v_floor_f32_e32 v34, v34
	s_mov_b32 s6, 0x125b8c61
	v_floor_f32_e32 v36, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshrrev_b32_e32 v28, 8, v28
	v_med3_num_f32 v34, v34, s1, 0x42fe0000
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_f32_u32_e32 v28, v28
	v_fmac_f32_e32 v35, 0x33800000, v30
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, v27, s7, 0x1ba5175
	v_mad_co_u64_u32 v[30:31], null, v27, s5, 0xffffffffe296f6ee
	v_mad_co_u64_u32 v[31:32], null, v27, s4, 0xffffffff9d23e50b
	v_mad_co_u64_u32 v[32:33], null, v27, s3, 0xffffffffbcd1195c
	v_fmac_f32_e32 v26, 0x33800000, v28
	s_mov_b32 s3, 0x27351d4d
	s_mov_b32 s4, 0x4c7003d5
	v_floor_f32_e32 v35, v35
	v_lshrrev_b32_e32 v29, 8, v29
	v_lshrrev_b32_e32 v30, 8, v30
	v_lshrrev_b32_e32 v31, 8, v31
	v_lshrrev_b32_e32 v32, 8, v32
	v_floor_f32_e32 v26, v26
	v_cvt_f32_u32_e32 v29, v29
	v_cvt_f32_u32_e32 v30, v30
	v_cvt_f32_u32_e32 v31, v31
	v_cvt_f32_u32_e32 v32, v32
	v_med3_num_f32 v26, v26, s1, 0x42fe0000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v25, 0x33800000, v29 :: v_dual_fmac_f32 v24, 0x33800000, v30
	v_dual_fmac_f32 v23, 0x33800000, v31 :: v_dual_fmac_f32 v22, 0x33800000, v32
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_i32_f32_e32 v26, v26
	v_floor_f32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_floor_f32_e32 v24, v24
	v_floor_f32_e32 v23, v23
	v_floor_f32_e32 v22, v22
	v_med3_num_f32 v35, v35, s1, 0x42fe0000
	v_med3_num_f32 v25, v25, s1, 0x42fe0000
	v_med3_num_f32 v24, v24, s1, 0x42fe0000
	v_med3_num_f32 v23, v23, s1, 0x42fe0000
	v_med3_num_f32 v22, v22, s1, 0x42fe0000
	v_med3_num_f32 v33, v36, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v25, v25
	v_cvt_i32_f32_e32 v24, v24
	v_cvt_i32_f32_e32 v28, v23
	v_cvt_i32_f32_e32 v30, v22
	v_lshlrev_b16 v22.l, 8, v26.l
	v_and_b16 v22.h, 0xff, v25.l
	v_lshlrev_b16 v23.l, 8, v24.l
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[24:25], null, v27, s3, 0xffffffffb52dfb6f
	s_mov_b32 s3, 0x3e42ae9
	v_and_b16 v23.h, 0xff, v28.l
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[25:26], null, v27, s3, 0x4fc9f202
	v_mad_co_u64_u32 v[28:29], null, v27, s4, 0x624f0979
	v_cvt_i32_f32_e32 v29, v35
	v_cvt_i32_f32_e32 v31, v34
	v_or_b16 v26.l, v23.h, v23.l
	v_lshrrev_b32_e32 v23, 8, v24
	v_cvt_i32_f32_e32 v32, v33
	v_or_b16 v26.h, v22.h, v22.l
	v_lshrrev_b32_e32 v24, 8, v25
	v_lshrrev_b32_e32 v25, 8, v28
	v_cvt_f32_u32_e32 v28, v23
	v_lshlrev_b16 v22.l, 8, v30.l
	v_and_b16 v22.h, 0xff, v29.l
	v_cvt_f32_u32_e32 v24, v24
	v_cvt_f32_u32_e32 v25, v25
	v_fmac_f32_e32 v19, 0x33800000, v28
	v_lshlrev_b16 v23.l, 8, v31.l
	v_and_b16 v23.h, 0xff, v32.l
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v20, 0x33800000, v24 :: v_dual_fmac_f32 v21, 0x33800000, v25
	s_mov_b32 s7, 0xab945ea5
	v_floor_f32_e32 v28, v19
	s_mov_b32 s5, 0x90158cf9
	v_floor_f32_e32 v29, v20
	v_mad_co_u64_u32 v[19:20], null, v27, s6, 0x3f469df8
	v_floor_f32_e32 v30, v21
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[20:21], null, v27, s7, 0x667adfbd
	s_mov_b32 s4, 0xb0eb139d
	v_or_b16 v25.h, v22.h, v22.l
	v_mad_co_u64_u32 v[21:22], null, v27, s5, 0xffffffff8aad3456
	v_or_b16 v25.l, v23.h, v23.l
	s_mov_b32 s3, 0xe3040fd1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[22:23], null, v27, s4, 0xffffffff865ce613
	v_mad_co_u64_u32 v[23:24], null, v27, s3, 0xffffffffa509a484
	v_lshrrev_b32_e32 v19, 8, v19
	v_lshrrev_b32_e32 v20, 8, v20
	s_mov_b32 s3, 0x1e0dc6ed
	v_lshrrev_b32_e32 v21, 8, v21
	s_mov_b32 s4, 0x996d7e75
	v_cvt_f32_u32_e32 v19, v19
	v_lshrrev_b32_e32 v22, 8, v22
	v_lshrrev_b32_e32 v23, 8, v23
	v_cvt_f32_u32_e32 v20, v20
	v_cvt_f32_u32_e32 v21, v21
	v_fmac_f32_e32 v18, 0x33800000, v19
	v_cvt_f32_u32_e32 v22, v22
	v_cvt_f32_u32_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v17, 0x33800000, v20 :: v_dual_fmac_f32 v16, 0x33800000, v21
	v_floor_f32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v15, 0x33800000, v22 :: v_dual_fmac_f32 v14, 0x33800000, v23
	v_floor_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_floor_f32_e32 v16, v16
	v_med3_num_f32 v18, v18, s1, 0x42fe0000
	s_delay_alu instid0(VALU_DEP_4)
	v_floor_f32_e32 v15, v15
	v_floor_f32_e32 v14, v14
	v_med3_num_f32 v17, v17, s1, 0x42fe0000
	v_med3_num_f32 v16, v16, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v18, v18
	v_med3_num_f32 v15, v15, s1, 0x42fe0000
	v_med3_num_f32 v14, v14, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v17, v17
	v_cvt_i32_f32_e32 v19, v16
	v_med3_num_f32 v24, v28, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v20, v15
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[15:16], null, v27, s3, 0x32dc8f7
	s_mov_b32 s3, 0x711a8809
	v_cvt_i32_f32_e32 v21, v14
	v_and_b16 v14.h, 0xff, v17.l
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[16:17], null, v27, s3, 0x43f391ea
	v_lshlrev_b16 v14.l, 8, v18.l
	v_mad_co_u64_u32 v[17:18], null, v27, s4, 0xfffffffffbca9841
	v_med3_num_f32 v28, v29, s1, 0x42fe0000
	v_med3_num_f32 v29, v30, s1, 0x42fe0000
	s_mov_b32 s6, 0x27b61881
	v_or_b16 v18.h, v14.h, v14.l
	v_lshlrev_b16 v14.l, 8, v19.l
	v_lshrrev_b32_e32 v19, 8, v15
	v_lshrrev_b32_e32 v16, 8, v16
	v_cvt_i32_f32_e32 v22, v29
	v_cvt_i32_f32_e32 v23, v28
	v_and_b16 v14.h, 0xff, v20.l
	v_cvt_f32_u32_e32 v19, v19
	v_cvt_f32_u32_e32 v16, v16
	v_and_b16 v15.h, 0xff, v22.l
	v_lshrrev_b32_e32 v17, 8, v17
	v_or_b16 v18.l, v14.h, v14.l
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v11, 0x33800000, v19 :: v_dual_fmac_f32 v12, 0x33800000, v16
	v_lshlrev_b16 v14.l, 8, v23.l
	v_cvt_f32_u32_e32 v20, v17
	v_cvt_i32_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_floor_f32_e32 v22, v11
	v_floor_f32_e32 v23, v12
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[11:12], null, v27, s6, 0x5abbca0
	v_lshlrev_b16 v15.l, 8, v21.l
	s_mov_b32 s4, 0xcf52873d
	s_mov_b32 s3, 0xc45f09f1
	s_mov_b32 s5, 0x9e082c19
	s_mov_b32 s7, 0x966d3345
	v_or_b16 v17.h, v15.h, v15.l
	v_and_b16 v14.h, 0xff, v24.l
	v_lshrrev_b32_e32 v11, 8, v11
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[15:16], null, v27, s5, 0x22331ebe
	v_med3_num_f32 v16, v23, s1, 0x42fe0000
	v_or_b16 v17.l, v14.h, v14.l
	v_cvt_f32_u32_e32 v11, v11
	v_fmac_f32_e32 v13, 0x33800000, v20
	v_mad_co_u64_u32 v[19:20], null, v27, s4, 0x73fe081b
	v_mad_co_u64_u32 v[20:21], null, v27, s3, 0xffffffff9cbb94ac
	v_lshrrev_b32_e32 v15, 8, v15
	v_fmac_f32_e32 v10, 0x33800000, v11
	v_floor_f32_e32 v24, v13
	v_mad_co_u64_u32 v[12:13], null, v27, s7, 0x57d53705
	v_med3_num_f32 v13, v22, s1, 0x42fe0000
	v_cvt_f32_u32_e32 v15, v15
	v_lshrrev_b32_e32 v19, 8, v19
	v_lshrrev_b32_e32 v20, 8, v20
	v_floor_f32_e32 v10, v10
	v_med3_num_f32 v21, v24, s1, 0x42fe0000
	v_fmac_f32_e32 v8, 0x33800000, v15
	v_lshrrev_b32_e32 v12, 8, v12
	v_cvt_f32_u32_e32 v19, v19
	v_cvt_f32_u32_e32 v20, v20
	v_med3_num_f32 v10, v10, s1, 0x42fe0000
	v_floor_f32_e32 v8, v8
	v_cvt_f32_u32_e32 v12, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v7, 0x33800000, v19 :: v_dual_fmac_f32 v6, 0x33800000, v20
	v_cvt_i32_f32_e32 v10, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v8, v8, s1, 0x42fe0000
	v_fmac_f32_e32 v9, 0x33800000, v12
	s_delay_alu instid0(VALU_DEP_4)
	v_floor_f32_e32 v7, v7
	v_floor_f32_e32 v6, v6
	v_cvt_i32_f32_e32 v15, v21
	v_cvt_i32_f32_e32 v8, v8
	v_floor_f32_e32 v9, v9
	v_med3_num_f32 v7, v7, s1, 0x42fe0000
	v_med3_num_f32 v6, v6, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v16, v16
	v_cvt_i32_f32_e32 v13, v13
	v_med3_num_f32 v9, v9, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v11, v7
	v_cvt_i32_f32_e32 v12, v6
	v_lshlrev_b16 v6.l, 8, v10.l
	v_lshlrev_b16 v7.l, 8, v8.l
	v_cvt_i32_f32_e32 v9, v9
	v_and_b16 v7.h, 0xff, v11.l
	v_lshlrev_b16 v8.l, 8, v12.l
	v_and_b16 v8.h, 0xff, v15.l
	v_and_b16 v9.h, 0xff, v13.l
	v_and_b16 v6.h, 0xff, v9.l
	v_lshlrev_b16 v9.l, 8, v16.l
	v_or_b16 v11.l, v7.h, v7.l
	v_or_b16 v10.h, v8.h, v8.l
	s_mov_b32 s1, exec_lo
	v_or_b16 v11.h, v6.h, v6.l
	v_or_b16 v10.l, v9.h, v9.l
	s_clause 0x3
	global_store_b64 v[0:1], v[2:3], off
	global_store_b64 v[0:1], v[25:26], off offset:16
	global_store_b64 v[0:1], v[17:18], off offset:32
	global_store_b64 v[0:1], v[10:11], off offset:48
                                        ; implicit-def: $vgpr0
	v_cmpx_eq_u32_e32 0, v5
	s_cbranch_execz .LBB0_388
; %bb.387:
	v_div_scale_f32 v0, null, 0x42fe0000, 0x42fe0000, v4
	s_or_b32 s2, s2, exec_lo
	v_rcp_f32_e32 v1, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v0, v1, 1.0
	v_fmac_f32_e32 v1, v2, v1
	v_div_scale_f32 v2, vcc_lo, v4, 0x42fe0000, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, v2, v1
	v_fma_f32 v5, -v0, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v5, v1
	v_fma_f32 v0, -v0, v3, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v0, v0, v1, v3
	v_div_fixup_f32 v0, v0, 0x42fe0000, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v0, 1.0, v0, s0
.LBB0_388:                              ; %Flow2747
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s0, s2
	s_cbranch_execnz .LBB0_383
	s_branch .LBB0_384
.Lfunc_end0:
	.size	gated_delta_net_q8_register_scan_gfx1201, .Lfunc_end0-gated_delta_net_q8_register_scan_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gated_delta_net_q8_register_scan_gfx1201
		.amdhsa_group_segment_fixed_size 29824
		.amdhsa_private_segment_fixed_size 72
		.amdhsa_kernarg_size 88
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
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 192
		.amdhsa_next_free_sgpr 71
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gated_delta_net_q8_register_scan_gfx1201)<<4)&4080)>>4
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
	.set .Lgated_delta_net_q8_register_scan_gfx1201.num_vgpr, 192
	.set .Lgated_delta_net_q8_register_scan_gfx1201.num_agpr, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.numbered_sgpr, 71
	.set .Lgated_delta_net_q8_register_scan_gfx1201.num_named_barrier, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.private_seg_size, 72
	.set .Lgated_delta_net_q8_register_scan_gfx1201.uses_vcc, 1
	.set .Lgated_delta_net_q8_register_scan_gfx1201.uses_flat_scratch, 1
	.set .Lgated_delta_net_q8_register_scan_gfx1201.has_dyn_sized_stack, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.has_recursion, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 35440
; TotalNumSgprs: 73
; NumVgprs: 192
; ScratchSize: 72
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 29824 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 73
; NumVGPRsForWavesPerEU: 192
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
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
	.type	__hip_cuid_49ff18740787d09b,@object ; @__hip_cuid_49ff18740787d09b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_49ff18740787d09b
__hip_cuid_49ff18740787d09b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_49ff18740787d09b, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_49ff18740787d09b
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
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
    .gfx1250_revision: B0
    .group_segment_fixed_size: 29824
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           gated_delta_net_q8_register_scan_gfx1201
    .private_segment_fixed_size: 72
    .sgpr_count:     73
    .sgpr_spill_count: 0
    .symbol:         gated_delta_net_q8_register_scan_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     192
    .vgpr_spill_count: 17
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
