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
	s_load_b128 s[28:31], s[0:1], 0x40
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 ttmp9, s29
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s30, 0x80
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_cmp_lt_i32 s28, 1
	s_cselect_b32 s3, -1, 0
	s_or_b32 s2, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_265
; %bb.1:                                ; %.preheader1049
	v_lshrrev_b32_e32 v1, 2, v0
	s_load_b512 s[12:27], s[0:1], 0x0
	s_and_b32 s2, ttmp7, 0xffff
	v_and_b32_e32 v101, 15, v0
	s_mov_b32 s37, 0
	v_and_b32_e32 v42, 0xf0, v1
	v_bfe_u32 v100, v0, 5, 1
	s_lshl_b32 s34, ttmp9, 14
	s_mov_b32 s35, s37
	v_bfe_u32 v99, v0, 4, 1
	v_lshl_add_u32 v1, s2, 6, v42
	s_lshr_b32 s2, ttmp7, 16
	v_lshlrev_b32_e32 v103, 6, v100
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s2, s29, s2
	v_lshlrev_b32_e32 v104, 3, v99
	v_or_b32_e32 v97, v1, v101
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s2, 14
	s_lshl_b32 s4, ttmp9, 7
	v_mbcnt_lo_u32_b32 v43, -1, 0
	v_lshrrev_b32_e32 v106, 6, v0
	v_lshlrev_b32_e32 v102, 7, v97
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[2:3], s[22:23], s[36:37]
	s_mov_b32 s5, s37
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[2:3], s[34:35]
	v_xor_b32_e32 v45, 16, v43
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v95, s2, s2, v102
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v96, null, s3, 0, s2
	v_bfi_b32 v46, v43, 0, 32
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, v95, v103
	v_add_co_ci_u32_e64 v2, null, 0, v96, vcc_lo
	s_lshl_b64 s[6:7], s[4:5], 2
	v_add_co_u32 v1, vcc_lo, v1, v104
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_cmp_eq_u32_e64 s2, 0, v0
	v_lshrrev_b32_e32 v108, 4, v0
	s_mov_b32 s30, ttmp9
	s_clause 0x3
	global_load_b64 v[33:34], v[1:2], off
	global_load_b64 v[35:36], v[1:2], off offset:16
	global_load_b64 v[37:38], v[1:2], off offset:32
	global_load_b64 v[39:40], v[1:2], off offset:48
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v93, s4, v97
	v_and_b32_e32 v105, 63, v0
	v_lshlrev_b32_e32 v47, 2, v106
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v11, v2 :: v_dual_and_b32 v98, 31, v0
	v_dual_mov_b32 v6, v2 :: v_dual_add_nc_u32 v1, s36, v93
	v_dual_mov_b32 v8, v2 :: v_dual_mov_b32 v7, v2
	v_mov_b32_e32 v14, v2
	v_or_b32_e32 v55, 7, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[3:4], 2, v[1:2]
	v_dual_mov_b32 v10, v2 :: v_dual_and_b32 v1, 16, v0
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v20, v2
	v_or_b32_e32 v51, 3, v104
	v_add_co_u32 v91, vcc_lo, s24, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v92, null, s25, v4, vcc_lo
	v_lshlrev_b32_e32 v3, 5, v0
	v_lshl_or_b32 v110, v101, 8, v1
	v_cmp_lt_u32_e32 vcc_lo, v45, v46
	global_load_b32 v41, v[91:92], off
	v_lshl_or_b32 v123, v105, 8, v47
	v_dual_mov_b32 v13, v2 :: v_dual_and_b32 v44, 0x1e0, v3
	v_xor_b32_e32 v111, 0x400, v3
	v_xor_b32_e32 v112, 0x404, v3
	v_xor_b32_e32 v113, 0x408, v3
	v_xor_b32_e32 v114, 0x40c, v3
	v_xor_b32_e32 v115, 0x410, v3
	v_xor_b32_e32 v116, 0x414, v3
	v_xor_b32_e32 v117, 0x418, v3
	v_xor_b32_e32 v118, 0x41c, v3
	v_mov_b32_e32 v3, v2
	v_or_b32_e32 v121, v42, v101
	v_or_b32_e32 v122, v104, v42
	v_or_b32_e32 v42, 21, v104
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v43, v43, v45 :: v_dual_lshlrev_b32 v152, 2, v51
	v_lshl_or_b32 v148, v121, 8, v1
	v_add_nc_u32_e32 v1, -1, v122
	v_or_b32_e32 v45, 20, v104
	v_or_b32_e32 v46, 22, v104
	v_or_b32_e32 v47, 23, v104
	v_lshlrev_b32_e32 v156, 2, v55
	v_mul_i32_i24_e32 v1, v1, v122
	v_or_b32_e32 v165, 7, v122
	v_or_b32_e32 v166, 1, v122
	v_or_b32_e32 v167, 2, v122
	v_or_b32_e32 v168, 3, v122
	v_or_b32_e32 v169, 4, v122
	v_ashrrev_i32_e32 v180, 1, v1
	v_dual_mov_b32 v1, v2 :: v_dual_lshlrev_b32 v162, 2, v42
	v_add_nc_u32_e32 v42, 8, v122
	v_dual_mov_b32 v9, v2 :: v_dual_lshlrev_b32 v4, 1, v0
	v_lshlrev_b32_e32 v149, 2, v43
	v_lshlrev_b32_e32 v161, 2, v45
	v_lshlrev_b32_e32 v163, 2, v46
	v_lshlrev_b32_e32 v164, 2, v47
	v_mul_u32_u24_e32 v42, v42, v165
	v_mul_u32_u24_e32 v43, v122, v166
	v_mul_u32_u24_e32 v45, v166, v167
	v_mul_u32_u24_e32 v46, v167, v168
	v_mul_u32_u24_e32 v47, v168, v169
	v_dual_mov_b32 v15, v2 :: v_dual_and_b32 v48, 32, v4
	v_mov_b32_e32 v4, v2
	v_or_b32_e32 v94, s4, v103
	s_add_nc_u64 s[4:5], s[14:15], s[6:7]
	s_add_nc_u64 s[6:7], s[12:13], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v174, s3, s4, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v175, null, s5, 0, s3
	v_add_co_u32 v176, s3, s6, v44
	v_lshrrev_b32_e32 v181, 1, v43
	v_lshrrev_b32_e32 v182, 1, v45
	v_lshrrev_b32_e32 v183, 1, v46
	v_lshrrev_b32_e32 v184, 1, v47
	v_lshrrev_b32_e32 v188, 1, v42
	v_dual_mov_b32 v27, v2 :: v_dual_mov_b32 v32, v2
	v_or_b32_e32 v49, 1, v104
	v_or_b32_e32 v50, 2, v104
	v_or_b32_e32 v57, 17, v104
	v_or_b32_e32 v59, 19, v104
	v_or_b32_e32 v170, 5, v122
	v_or_b32_e32 v171, 6, v122
	v_dual_mov_b32 v16, v2 :: v_dual_lshlrev_b32 v107, 3, v0
	v_dual_mov_b32 v18, v2 :: v_dual_lshlrev_b32 v109, 4, v0
	v_mov_b32_e32 v5, v2
	v_dual_mov_b32 v17, v2 :: v_dual_mov_b32 v22, v2
	v_dual_mov_b32 v19, v2 :: v_dual_mov_b32 v24, v2
	v_dual_mov_b32 v21, v2 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v23, v2 :: v_dual_mov_b32 v28, v2
	v_dual_mov_b32 v25, v2 :: v_dual_mov_b32 v30, v2
	v_dual_mov_b32 v29, v2 :: v_dual_lshlrev_b32 v120, 2, v105
	v_dual_mov_b32 v31, v2 :: v_dual_lshlrev_b32 v150, 2, v49
	v_lshlrev_b32_e32 v119, 11, v99
	v_or_b32_e32 v52, 4, v104
	v_or_b32_e32 v53, 5, v104
	v_or_b32_e32 v54, 6, v104
	v_or_b32_e32 v56, 16, v104
	v_or_b32_e32 v58, 18, v104
	v_mul_u32_u24_e32 v49, v169, v170
	v_mul_u32_u24_e32 v51, v171, v165
	v_dual_mov_b32 v173, v121 :: v_dual_lshlrev_b32 v154, 2, v53
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v177, null, s7, 0, s3
	v_add_co_u32 v178, s3, s6, v48
	v_lshrrev_b32_e32 v185, 1, v49
	v_lshrrev_b32_e32 v187, 1, v51
	v_lshl_or_b32 v124, v100, 7, v110
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v179, null, s7, 0, s3
	s_lshl_b32 s24, s29, 7
	s_add_co_i32 s25, s28, -1
	s_mov_b32 s40, s37
	s_wait_loadcnt 0x4
	v_bfe_i32 v42, v33, 0, 8
	v_bfe_i32 v43, v33, 8, 8
	v_bfe_i32 v44, v33, 16, 8
	v_ashrrev_i32_e32 v33, 24, v33
	v_bfe_i32 v45, v34, 0, 8
	v_bfe_i32 v46, v34, 8, 8
	v_bfe_i32 v47, v34, 16, 8
	v_ashrrev_i32_e32 v34, 24, v34
	v_cvt_f32_i32_e32 v33, v33
	s_wait_loadcnt 0x3
	v_bfe_i32 v48, v35, 0, 8
	v_bfe_i32 v49, v35, 8, 8
	v_bfe_i32 v51, v36, 0, 8
	v_cvt_f32_i32_e32 v34, v34
	v_bfe_i32 v53, v36, 16, 8
	s_wait_loadcnt 0x2
	v_bfe_i32 v55, v37, 8, 8
	s_wait_loadcnt 0x1
	v_bfe_i32 v60, v39, 0, 8
	v_bfe_i32 v61, v39, 8, 8
	v_bfe_i32 v62, v39, 16, 8
	v_ashrrev_i32_e32 v39, 24, v39
	v_bfe_i32 v63, v40, 0, 8
	v_bfe_i32 v64, v40, 8, 8
	v_bfe_i32 v65, v40, 16, 8
	v_ashrrev_i32_e32 v40, 24, v40
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	v_cvt_f32_i32_e32 v47, v47
	s_wait_loadcnt 0x0
	v_fma_mixhi_f16 v76, v41, v33, 0
	v_fma_mixhi_f16 v78, v41, v34, 0
	v_mov_b32_e32 v34, v32
	v_dual_mov_b32 v33, v31 :: v_dual_lshlrev_b32 v158, 2, v57
	v_mov_b32_e32 v32, v30
	v_dual_mov_b32 v31, v29 :: v_dual_lshlrev_b32 v160, 2, v59
	v_dual_mov_b32 v30, v28 :: v_dual_lshlrev_b32 v151, 2, v50
	v_mul_u32_u24_e32 v50, v170, v171
	v_bfe_i32 v57, v38, 0, 8
	v_bfe_i32 v59, v38, 16, 8
	v_mov_b32_e32 v29, v27
	v_dual_mov_b32 v28, v26 :: v_dual_lshlrev_b32 v153, 2, v52
	v_lshrrev_b32_e32 v186, 1, v50
	v_bfe_i32 v50, v35, 16, 8
	v_ashrrev_i32_e32 v35, 24, v35
	v_bfe_i32 v52, v36, 8, 8
	v_ashrrev_i32_e32 v36, 24, v36
	v_mov_b32_e32 v27, v25
	v_dual_mov_b32 v26, v24 :: v_dual_lshlrev_b32 v155, 2, v54
	v_bfe_i32 v54, v37, 0, 8
	v_mov_b32_e32 v25, v23
	v_dual_mov_b32 v24, v22 :: v_dual_lshlrev_b32 v157, 2, v56
	v_dual_mov_b32 v23, v21 :: v_dual_lshlrev_b32 v172, 2, v122
	v_dual_mov_b32 v22, v20 :: v_dual_lshlrev_b32 v159, 2, v58
	v_bfe_i32 v56, v37, 16, 8
	v_ashrrev_i32_e32 v37, 24, v37
	v_bfe_i32 v58, v38, 8, 8
	v_ashrrev_i32_e32 v38, 24, v38
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	v_cvt_f32_i32_e32 v48, v48
	v_cvt_f32_i32_e32 v38, v38
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v50, v50
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v54, v54
	v_cvt_f32_i32_e32 v55, v55
	v_cvt_f32_i32_e32 v56, v56
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fma_mixhi_f16 v80, v41, v35, 0
	v_fma_mixhi_f16 v82, v41, v36, 0
	v_fma_mixhi_f16 v84, v41, v37, 0
	v_fma_mixhi_f16 v86, v41, v38, 0
	v_mov_b32_e32 v21, v19
	v_mov_b32_e32 v20, v18
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v18, v16
	v_mov_b32_e32 v17, v15
	v_mov_b32_e32 v16, v14
	v_mov_b32_e32 v15, v13
	v_mov_b32_e32 v14, v12
	v_mov_b32_e32 v13, v11
	v_mov_b32_e32 v12, v10
	v_mov_b32_e32 v11, v9
	v_mov_b32_e32 v10, v8
	v_mov_b32_e32 v9, v7
	v_mov_b32_e32 v8, v6
	v_mov_b32_e32 v7, v5
	v_mov_b32_e32 v6, v4
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v3, v1
	v_cvt_f32_i32_e32 v1, v62
	v_cvt_f32_i32_e32 v35, v39
	v_cvt_f32_i32_e32 v36, v63
	v_cvt_f32_i32_e32 v37, v64
	v_cvt_f32_i32_e32 v38, v65
	v_cvt_f32_i32_e32 v39, v40
	v_lshl_or_b32 v125, v101, 2, v119
	v_or_b32_e32 v126, 0x2600, v119
	v_or_b32_e32 v127, 0x2500, v119
	v_or_b32_e32 v128, 0x2400, v119
	v_or_b32_e32 v129, 0x2300, v119
	v_or_b32_e32 v130, 0x2200, v119
	v_or_b32_e32 v131, 0x2100, v119
	v_or_b32_e32 v132, 0x2000, v119
	v_or_b32_e32 v133, 0x1700, v119
	v_or_b32_e32 v134, 0x1600, v119
	v_or_b32_e32 v135, 0x1500, v119
	v_or_b32_e32 v136, 0x1400, v119
	v_or_b32_e32 v137, 0x1300, v119
	v_or_b32_e32 v138, 0x1200, v119
	v_or_b32_e32 v139, 0x1100, v119
	v_or_b32_e32 v140, 0x1000, v119
	v_or_b32_e32 v141, 0x700, v119
	v_or_b32_e32 v142, 0x600, v119
	v_or_b32_e32 v143, 0x500, v119
	v_or_b32_e32 v144, 0x400, v119
	v_or_b32_e32 v145, 0x300, v119
	v_or_b32_e32 v146, 0x200, v119
	v_or_b32_e32 v147, 0x100, v119
	v_fma_mixlo_f16 v75, v41, v42, 0
	v_fma_mixhi_f16 v75, v41, v43, 0
	v_fma_mixlo_f16 v76, v41, v44, 0
	v_fma_mixlo_f16 v77, v41, v45, 0
	v_fma_mixhi_f16 v77, v41, v46, 0
	v_fma_mixlo_f16 v78, v41, v47, 0
	v_fma_mixlo_f16 v79, v41, v48, 0
	v_fma_mixhi_f16 v79, v41, v49, 0
	v_fma_mixlo_f16 v80, v41, v50, 0
	v_fma_mixlo_f16 v81, v41, v51, 0
	v_fma_mixhi_f16 v81, v41, v52, 0
	v_fma_mixlo_f16 v82, v41, v53, 0
	v_fma_mixlo_f16 v83, v41, v54, 0
	v_fma_mixhi_f16 v83, v41, v55, 0
	v_fma_mixlo_f16 v84, v41, v56, 0
	v_fma_mixlo_f16 v85, v41, v57, 0
	v_fma_mixhi_f16 v85, v41, v58, 0
	v_fma_mixlo_f16 v86, v41, v59, 0
	v_fma_mixlo_f16 v87, v41, v60, 0
	v_fma_mixhi_f16 v87, v41, v61, 0
	v_fma_mixlo_f16 v88, v41, v1, 0
	v_fma_mixhi_f16 v88, v41, v35, 0
	v_fma_mixlo_f16 v89, v41, v36, 0
	v_fma_mixhi_f16 v89, v41, v37, 0
	v_fma_mixlo_f16 v90, v41, v38, 0
	v_fma_mixhi_f16 v90, v41, v39, 0
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v173, 64, v173
	s_and_b32 vcc_lo, exec_lo, s38
	s_mov_b32 s40, s33
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_193
.LBB0_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_5 Depth 2
                                        ;     Child Loop BB0_9 Depth 2
                                        ;     Child Loop BB0_12 Depth 2
                                        ;       Child Loop BB0_13 Depth 3
                                        ;     Child Loop BB0_48 Depth 2
                                        ;     Child Loop BB0_52 Depth 2
                                        ;     Child Loop BB0_81 Depth 2
                                        ;     Child Loop BB0_83 Depth 2
                                        ;     Child Loop BB0_86 Depth 2
                                        ;       Child Loop BB0_88 Depth 3
                                        ;     Child Loop BB0_122 Depth 2
                                        ;     Child Loop BB0_126 Depth 2
                                        ;     Child Loop BB0_143 Depth 2
                                        ;     Child Loop BB0_147 Depth 2
                                        ;       Child Loop BB0_156 Depth 3
	v_or_b32_e32 v1, s40, v105
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_i32_e32 v189, s25, v1
	v_cmp_gt_i32_e64 s3, s28, v1
	v_add_nc_u32_e32 v1, 0x80, v120
	v_mad_co_u64_u32 v[35:36], null, v189, s29, s[30:31]
	v_mov_b32_e32 v36, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v37, vcc_lo, s18, v35
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v38, null, s19, v36, vcc_lo
	v_add_co_u32 v35, vcc_lo, s20, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, s21, v36, vcc_lo
	global_load_b32 v37, v[37:38], off
	global_load_b32 v35, v[35:36], off
	s_wait_loadcnt 0x1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0, v37, s3
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v35, 0, v35, s3
	ds_store_2addr_stride64_b32 v1, v36, v35 offset0:112 offset1:114
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_6
; %bb.4:                                ; %.preheader1047.preheader
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_mov_b32_e32 v1, 0
	s_movk_i32 s5, 0xff00
.LBB0_5:                                ; %.preheader1047
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v35, s5
	s_add_co_i32 s5, s5, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s5, 0
	ds_load_b32 v36, v35 offset:29056
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v36
	ds_store_b32 v35, v1 offset:29056
	s_cbranch_scc1 .LBB0_5
.LBB0_6:                                ; %Flow2371
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s33, s40, 64
	s_sub_co_i32 s4, s28, s40
	s_cmp_ge_i32 s33, s28
	v_min_i32_e32 v191, s25, v173
	s_cselect_b32 s38, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s38, exec_lo
	s_cselect_b32 s39, s4, 64
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v1, v120 offset:28800
	s_wait_dscnt 0x0
	v_mul_f32_e32 v35, 0x3fb8aa3b, v1
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_rndne_f32_e32 v36, v35
	v_fma_f32 v37, 0x3fb8aa3b, v1, -v35
	v_sub_f32_e32 v35, v35, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v37, 0x32a5705f, v1
	v_cvt_i32_f32_e32 v36, v36
	v_add_f32_e32 v35, v35, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v35, v35
	v_ldexp_f32 v35, v35, v36
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v35, 0, v35, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x7f800000, v35, vcc_lo
	ds_store_b32 v120, v1 offset:29056
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s5, s39, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v1, s5
	ds_load_b32 v35, v1 offset:28796
	s_wait_dscnt 0x0
	v_readfirstlane_b32 s5, v35
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v35
	s_mul_f32 s5, s5, 0x3fb8aa3b
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_2)
	s_xor_b32 s6, s5, 0x80000000
	s_wait_alu depctr_sa_sdst(0)
	v_fma_f32 v36, 0x3fb8aa3b, v35, s6
	s_rndne_f32 s6, s5
	s_wait_alu depctr_sa_sdst(0)
	s_sub_f32 s5, s5, s6
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v36, 0x32a5705f, v35
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_add_f32_e32 v36, s5, v36
	s_cvt_i32_f32 s5, s6
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v36, v36
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_1)
	v_ldexp_f32 v36, v36, s5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v36, 0, v36, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v35
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v35, 0x7f800000, v36, vcc_lo
	ds_store_b32 v2, v35 offset:29568
	ds_load_b32 v1, v1 offset:28796
	s_wait_dscnt 0x0
	ds_store_b32 v2, v1 offset:29572
.LBB0_8:                                ;   in Loop: Header=BB0_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s4, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_9:                                ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v1, s4, 11, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v1, 7, v1
	v_add_nc_u32_e32 v43, s40, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v1, s25, v43
	v_mul_lo_u32 v1, v1, s24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[35:36], 2, v[1:2]
	v_lshl_add_u32 v1, s4, 12, v109
	s_add_co_i32 s4, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 4
	v_add_co_u32 v39, vcc_lo, v174, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, v175, v36, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v43
	s_clause 0x1
	global_load_b128 v[35:38], v[39:40], off
	global_load_b128 v[39:42], v[39:40], off offset:16
	s_wait_loadcnt 0x1
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v43, 0, v35 :: v_dual_cndmask_b32 v44, 0, v37
	s_wait_loadcnt 0x0
	v_dual_cndmask_b32 v35, 0, v36 :: v_dual_cndmask_b32 v36, 0, v42
	v_dual_cndmask_b32 v37, 0, v41 :: v_dual_cndmask_b32 v40, 0, v40
	v_cndmask_b32_e32 v39, 0, v39, vcc_lo
	v_cndmask_b32_e32 v41, 0, v38, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v38.h, v36
	v_cvt_f16_f32_e32 v38.l, v37
	v_cvt_f16_f32_e32 v37.h, v40
	v_cvt_f16_f32_e32 v37.l, v39
	v_cvt_f16_f32_e32 v36.h, v41
	v_cvt_f16_f32_e32 v36.l, v44
	v_cvt_f16_f32_e32 v35.h, v35
	v_cvt_f16_f32_e32 v35.l, v43
	ds_store_b128 v1, v[35:38] offset:8320
	s_cbranch_scc0 .LBB0_9
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_i32_e64 s4, s39, v122
	v_cmp_gt_i32_e64 s5, s39, v166
	v_cmp_gt_i32_e64 s6, s39, v167
	v_cmp_gt_i32_e64 s7, s39, v168
	v_cmp_gt_i32_e64 s8, s39, v169
	v_cmp_gt_i32_e64 s9, s39, v170
	v_cmp_gt_i32_e64 s10, s39, v171
	v_cmp_gt_i32_e64 s11, s39, v165
	s_mov_b32 s13, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_12
.LBB0_11:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v1, v187, v1
	s_add_co_i32 s13, s13, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s13, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, 0x81f, v1, vcc_lo
	v_lshlrev_b32_e32 v1, 2, v1
	ds_store_b32 v1, v37
	s_cbranch_scc1 .LBB0_46
.LBB0_12:                               ; %.preheader1043
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_13 Depth 3
	s_wait_dscnt 0x1
	v_mov_b32_e32 v35, 0
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v1, s13, 12, v110
	s_mov_b32 s12, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v36, v35 :: v_dual_mov_b32 v37, v35
	v_dual_mov_b32 v38, v35 :: v_dual_mov_b32 v39, v35
	v_dual_mov_b32 v40, v35 :: v_dual_mov_b32 v41, v35
	v_mov_b32_e32 v42, v35
.LBB0_13:                               ; %.preheader1034
                                        ;   Parent Loop BB0_3 Depth=1
                                        ;     Parent Loop BB0_12 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s22, s12, 5
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v43, s22, v148
	v_add_nc_u32_e32 v47, s22, v1
	s_cmp_eq_u32 s12, 8
	ds_load_b128 v[43:46], v43 offset:8320
	ds_load_b128 v[47:50], v47 offset:8320
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[35:42], v[43:46], v[47:50], v[35:42]
	s_cbranch_scc0 .LBB0_13
; %bb.14:                               ; %.preheader1042
                                        ;   in Loop: Header=BB0_12 Depth=2
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	s_and_saveexec_b32 s12, s4
; %bb.15:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v44, v172 offset:29312
; %bb.16:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_lshl_or_b32 v1, s13, 4, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_u32_e32 vcc_lo, v1, v122
	v_lshlrev_b32_e32 v43, 2, v1
	s_and_b32 vcc_lo, vcc_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v45, v172 offset:28800
	ds_load_b32 v46, v43 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v45, v45, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v46, 0x3fb8aa3b, v45
	v_fma_f32 v47, 0x3fb8aa3b, v45, -v46
	v_rndne_f32_e32 v48, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v46, v46, v48 :: v_dual_fmac_f32 v47, 0x32a5705f, v45
	v_cmp_ngt_f32_e64 s12, 0xc2ce8ed0, v45
	v_add_f32_e32 v46, v46, v47
	v_cvt_i32_f32_e32 v47, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v46, v46
	v_ldexp_f32 v46, v46, v47
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v46, 0, v46, s12
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v45
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v45, 0x7f800000, v46, s12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v44, v44, v45
	v_mul_f32_e32 v45, v35, v44
.LBB0_18:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x0
	v_dual_mov_b32 v44, 0 :: v_dual_add_nc_u32 v35, v180, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v35, 0x81f, v35, vcc_lo
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v46, 2, v35
	ds_store_b32 v46, v45
	s_and_saveexec_b32 s12, s5
; %bb.19:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v35, v172 offset:29316
; %bb.20:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_cmp_le_u32_e32 vcc_lo, v1, v122
	s_and_b32 vcc_lo, vcc_lo, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v44, v172 offset:28804
	ds_load_b32 v45, v43 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v44, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v45, 0x3fb8aa3b, v44
	v_fma_f32 v46, 0x3fb8aa3b, v44, -v45
	v_rndne_f32_e32 v47, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v45, v45, v47 :: v_dual_fmac_f32 v46, 0x32a5705f, v44
	v_cmp_ngt_f32_e64 s12, 0xc2ce8ed0, v44
	v_add_f32_e32 v45, v45, v46
	v_cvt_i32_f32_e32 v46, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v45, v45
	v_ldexp_f32 v45, v45, v46
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, 0, v45, s12
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v44
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v44, 0x7f800000, v45, s12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, v35, v44
	v_mul_f32_e32 v44, v36, v35
.LBB0_22:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x0
	v_dual_mov_b32 v36, 0 :: v_dual_add_nc_u32 v35, v181, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v35, 0x81f, v35, vcc_lo
	v_lshlrev_b32_e32 v45, 2, v35
	v_mov_b32_e32 v35, 0
	ds_store_b32 v45, v44
	s_and_saveexec_b32 s12, s6
; %bb.23:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v35, v172 offset:29320
; %bb.24:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_cmp_lt_u32_e32 vcc_lo, v1, v167
	s_and_b32 vcc_lo, vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v36, v172 offset:28808
	ds_load_b32 v44, v43 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v36, v36, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v44, 0x3fb8aa3b, v36
	v_cmp_ngt_f32_e64 s12, 0xc2ce8ed0, v36
	v_fma_f32 v45, 0x3fb8aa3b, v36, -v44
	v_rndne_f32_e32 v46, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v45, 0x32a5705f, v36 :: v_dual_sub_f32 v44, v44, v46
	v_add_f32_e32 v44, v44, v45
	v_cvt_i32_f32_e32 v45, v46
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v44, v44
	v_ldexp_f32 v44, v44, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v44, 0, v44, s12
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v36
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0x7f800000, v44, s12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, v35, v36
	v_mul_f32_e32 v36, v37, v35
.LBB0_26:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v35, v182, v1
	v_mov_b32_e32 v37, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v35, 0x81f, v35, vcc_lo
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v44, 2, v35
	ds_store_b32 v44, v36
	s_and_saveexec_b32 s12, s7
; %bb.27:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v35, v172 offset:29324
; %bb.28:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_cmp_lt_u32_e32 vcc_lo, v1, v168
	s_and_b32 vcc_lo, vcc_lo, s7
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v36, v172 offset:28812
	ds_load_b32 v37, v43 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v36, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, 0x3fb8aa3b, v36
	v_fma_f32 v44, 0x3fb8aa3b, v36, -v37
	v_rndne_f32_e32 v45, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v44, 0x32a5705f, v36 :: v_dual_sub_f32 v37, v37, v45
	v_add_f32_e32 v37, v37, v44
	v_cvt_i32_f32_e32 v44, v45
	v_cmp_ngt_f32_e64 s12, 0xc2ce8ed0, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v37, v37
	v_ldexp_f32 v37, v37, v44
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v37, 0, v37, s12
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v36
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0x7f800000, v37, s12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, v35, v36
	v_mul_f32_e32 v37, v38, v35
.LBB0_30:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x0
	v_dual_mov_b32 v36, 0 :: v_dual_add_nc_u32 v35, v183, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v35, 0x81f, v35, vcc_lo
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v38, 2, v35
	ds_store_b32 v38, v37
	s_and_saveexec_b32 s12, s8
; %bb.31:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v35, v172 offset:29328
; %bb.32:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_cmp_lt_u32_e32 vcc_lo, v1, v169
	s_and_b32 vcc_lo, vcc_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v36, v172 offset:28816
	ds_load_b32 v37, v43 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v36, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, 0x3fb8aa3b, v36
	v_fma_f32 v38, 0x3fb8aa3b, v36, -v37
	v_rndne_f32_e32 v44, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v37, v37, v44
	v_fmac_f32_e32 v38, 0x32a5705f, v36
	v_cmp_ngt_f32_e64 s12, 0xc2ce8ed0, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v37, v37, v38
	v_cvt_i32_f32_e32 v38, v44
	v_exp_f32_e32 v37, v37
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v37, v37, v38
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v37, 0, v37, s12
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v36
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v36, 0x7f800000, v37, s12
	v_mul_f32_e32 v35, v35, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v36, v39, v35
.LBB0_34:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v35, v184, v1
	v_mov_b32_e32 v37, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v35, 0x81f, v35, vcc_lo
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v38, 2, v35
	ds_store_b32 v38, v36
	s_and_saveexec_b32 s12, s9
; %bb.35:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v35, v172 offset:29332
; %bb.36:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_cmp_lt_u32_e32 vcc_lo, v1, v170
	s_and_b32 vcc_lo, vcc_lo, s9
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v36, v172 offset:28820
	ds_load_b32 v37, v43 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v36, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, 0x3fb8aa3b, v36
	v_fma_f32 v38, 0x3fb8aa3b, v36, -v37
	v_rndne_f32_e32 v39, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v37, v37, v39 :: v_dual_fmac_f32 v38, 0x32a5705f, v36
	v_cmp_ngt_f32_e64 s12, 0xc2ce8ed0, v36
	v_add_f32_e32 v37, v37, v38
	v_cvt_i32_f32_e32 v38, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v37, v37
	v_ldexp_f32 v37, v37, v38
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v37, 0, v37, s12
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v36
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0x7f800000, v37, s12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, v35, v36
	v_mul_f32_e32 v37, v40, v35
.LBB0_38:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x0
	v_dual_mov_b32 v36, 0 :: v_dual_add_nc_u32 v35, v185, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v35, 0x81f, v35, vcc_lo
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v38, 2, v35
	ds_store_b32 v38, v37
	s_and_saveexec_b32 s12, s10
; %bb.39:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v35, v172 offset:29336
; %bb.40:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_cmp_lt_u32_e32 vcc_lo, v1, v171
	s_and_b32 vcc_lo, vcc_lo, s10
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v36, v172 offset:28824
	ds_load_b32 v37, v43 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v36, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, 0x3fb8aa3b, v36
	v_fma_f32 v38, 0x3fb8aa3b, v36, -v37
	v_rndne_f32_e32 v39, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v37, v37, v39 :: v_dual_fmac_f32 v38, 0x32a5705f, v36
	v_cmp_ngt_f32_e64 s12, 0xc2ce8ed0, v36
	v_add_f32_e32 v37, v37, v38
	v_cvt_i32_f32_e32 v38, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v37, v37
	v_ldexp_f32 v37, v37, v38
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v37, 0, v37, s12
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v36
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0x7f800000, v37, s12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, v35, v36
	v_mul_f32_e32 v36, v41, v35
.LBB0_42:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v35, v186, v1
	v_mov_b32_e32 v37, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v35, 0x81f, v35, vcc_lo
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v38, 2, v35
	ds_store_b32 v38, v36
	s_and_saveexec_b32 s12, s11
; %bb.43:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v35, v172 offset:29340
; %bb.44:                               ;   in Loop: Header=BB0_12 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_cmp_lt_u32_e32 vcc_lo, v1, v165
	s_and_b32 vcc_lo, vcc_lo, s11
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_11
; %bb.45:                               ;   in Loop: Header=BB0_12 Depth=2
	ds_load_b32 v36, v172 offset:28828
	ds_load_b32 v37, v43 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v36, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, 0x3fb8aa3b, v36
	v_fma_f32 v38, 0x3fb8aa3b, v36, -v37
	v_rndne_f32_e32 v39, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v37, v37, v39 :: v_dual_fmac_f32 v38, 0x32a5705f, v36
	v_cmp_ngt_f32_e64 s12, 0xc2ce8ed0, v36
	v_add_f32_e32 v37, v37, v38
	v_cvt_i32_f32_e32 v38, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v37, v37
	v_ldexp_f32 v37, v37, v38
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v37, 0, v37, s12
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v36
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0x7f800000, v37, s12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v35, v35, v36
	v_mul_f32_e32 v37, v42, v35
	s_branch .LBB0_11
.LBB0_46:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v1, v106
	s_movk_i32 s12, 0xff00
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_48
.LBB0_47:                               ;   in Loop: Header=BB0_48 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_add_nc_u32_e32 v36, s12, v123
	v_add_nc_u32_e32 v1, 4, v1
	s_add_co_i32 s12, s12, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, 0
	s_wait_dscnt 0x0
	ds_store_b32 v36, v35 offset:8576
	s_cbranch_scc1 .LBB0_50
.LBB0_48:                               ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v35, 0
	s_mov_b32 s13, exec_lo
	v_cmpx_lt_u32_e64 v105, v1
	s_cbranch_execz .LBB0_47
; %bb.49:                               ;   in Loop: Header=BB0_48 Depth=2
	v_add_nc_u32_e32 v35, -1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v35, v35, v1
	v_lshlrev_b32_e32 v35, 1, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v35, -4, v35
	v_add_nc_u32_e32 v35, v120, v35
	ds_load_b32 v35, v35
	s_branch .LBB0_47
.LBB0_50:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s12, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_52
.LBB0_51:                               ;   in Loop: Header=BB0_52 Depth=2
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v3, v43 :: v_dual_mov_b32 v4, v44
	v_dual_mov_b32 v5, v45 :: v_dual_mov_b32 v6, v46
	v_dual_mov_b32 v7, v47 :: v_dual_mov_b32 v8, v48
	v_dual_mov_b32 v9, v49 :: v_dual_mov_b32 v10, v50
	v_dual_mov_b32 v11, v51 :: v_dual_mov_b32 v12, v52
	v_dual_mov_b32 v13, v53 :: v_dual_mov_b32 v14, v54
	v_dual_mov_b32 v15, v55 :: v_dual_mov_b32 v16, v56
	v_dual_mov_b32 v17, v57 :: v_dual_mov_b32 v18, v58
	v_dual_mov_b32 v19, v59 :: v_dual_mov_b32 v20, v60
	v_dual_mov_b32 v21, v61 :: v_dual_mov_b32 v22, v62
	v_dual_mov_b32 v23, v63 :: v_dual_mov_b32 v24, v64
	v_dual_mov_b32 v25, v65 :: v_dual_mov_b32 v26, v66
	v_dual_mov_b32 v27, v67 :: v_dual_mov_b32 v28, v68
	v_dual_mov_b32 v29, v69 :: v_dual_mov_b32 v30, v70
	v_dual_mov_b32 v31, v71 :: v_dual_mov_b32 v32, v72
	v_dual_mov_b32 v33, v73 :: v_dual_mov_b32 v34, v74
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, 4
	s_cbranch_scc1 .LBB0_80
.LBB0_52:                               ; %.preheader1033
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s13, s12, 4
	v_add_nc_u32_e32 v190, v109, v109
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s22, s13, s40
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v43, s22, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v1, s25, v43
	v_mul_lo_u32 v1, v1, s24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[1:2]
	v_add_co_u32 v39, vcc_lo, v174, v35
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v40, null, v175, v36, vcc_lo
	s_clause 0x1
	global_load_b128 v[35:38], v[39:40], off offset:16
	global_load_b128 v[39:42], v[39:40], off
	v_cmp_gt_i32_e32 vcc_lo, s28, v43
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v38, 0, v38 :: v_dual_cndmask_b32 v1, 0, v39
	v_dual_cndmask_b32 v39, 0, v40 :: v_dual_cndmask_b32 v40, 0, v41
	v_dual_cndmask_b32 v37, 0, v37 :: v_dual_cndmask_b32 v36, 0, v36
	v_cndmask_b32_e32 v35, 0, v35, vcc_lo
	v_cndmask_b32_e32 v41, 0, v42, vcc_lo
	v_cvt_f16_f32_e32 v38.h, v38
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f16_f32_e32 v38.l, v37
	v_cvt_f16_f32_e32 v37.h, v36
	v_cvt_f16_f32_e32 v37.l, v35
	v_cvt_f16_f32_e32 v36.h, v41
	v_cvt_f16_f32_e32 v36.l, v40
	v_cvt_f16_f32_e32 v35.h, v39
	v_cvt_f16_f32_e32 v35.l, v1
	ds_store_b128 v109, v[35:38] offset:24704
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v35, 0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[45:48], v124 offset:24704
	ds_load_b128 v[49:52], v124 offset:24736
	ds_load_b128 v[53:56], v124 offset:24768
	ds_load_b128 v[57:60], v124 offset:24800
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[37:44], v[45:48], v[75:78], 0
	v_or_b32_e32 v45, s13, v104
	s_mov_b32 s13, exec_lo
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[37:44], v[49:52], v[79:82], v[37:44]
	v_lshlrev_b32_e32 v1, 2, v45
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[37:44], v[53:56], v[83:86], v[37:44]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[37:44], v[57:60], v[87:90], v[37:44]
	ds_store_b128 v190, v[37:40]
	ds_store_b128 v190, v[41:44] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmpx_gt_i32_e64 s39, v45
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_52 Depth=2
	v_or_b32_e32 v35, s40, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v46, s25, v35
	v_mad_co_u64_u32 v[46:47], null, v46, s24, v[93:94]
	v_mov_b32_e32 v47, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[46:47], 2, v[46:47]
	v_add_co_u32 v46, vcc_lo, s16, v46
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v47, null, s17, v47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v35
	global_load_b32 v48, v[46:47], off
	v_add_nc_u32_e32 v46, 0x80, v1
	ds_load_b32 v49, v111
	ds_load_2addr_stride64_b32 v[46:47], v46 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v37, v37, v49
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v35, 0, v48, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v35, -v46, v37, v35
	v_mul_f32_e32 v35, v47, v35
.LBB0_54:                               ; %.critedge
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v37, 1, v45
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s39, v37
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_52 Depth=2
	v_or_b32_e32 v46, s40, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v36, s25, v46
	v_mad_co_u64_u32 v[36:37], null, v36, s24, v[93:94]
	v_mov_b32_e32 v37, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[36:37], 2, v[36:37]
	v_add_co_u32 v36, vcc_lo, s16, v36
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v37, null, s17, v37, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v46
	global_load_b32 v47, v[36:37], off
	v_add_nc_u32_e32 v36, 0x84, v1
	ds_load_b32 v48, v112
	ds_load_2addr_stride64_b32 v[36:37], v36 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v38, v38, v48
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v47, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v36, -v36, v38, v46
	v_mul_f32_e32 v36, v37, v36
.LBB0_56:                               ; %.critedge.1
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v46, 2, v45
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s39, v46
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_52 Depth=2
	v_or_b32_e32 v37, s40, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v46, s25, v37
	v_mad_co_u64_u32 v[46:47], null, v46, s24, v[93:94]
	v_mov_b32_e32 v47, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[46:47], 2, v[46:47]
	v_add_co_u32 v46, vcc_lo, s16, v46
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v47, null, s17, v47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v37
	global_load_b32 v48, v[46:47], off
	v_add_nc_u32_e32 v46, 0x88, v1
	ds_load_b32 v49, v113
	ds_load_2addr_stride64_b32 v[46:47], v46 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v39, v39, v49
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, 0, v48, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v46, v39, v37
	v_mul_f32_e32 v37, v47, v37
.LBB0_58:                               ; %.critedge.2
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v39, 3, v45
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s39, v39
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_52 Depth=2
	v_or_b32_e32 v46, s40, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v38, s25, v46
	v_mad_co_u64_u32 v[38:39], null, v38, s24, v[93:94]
	v_mov_b32_e32 v39, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[38:39], 2, v[38:39]
	v_add_co_u32 v38, vcc_lo, s16, v38
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v39, null, s17, v39, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v46
	global_load_b32 v47, v[38:39], off
	v_add_nc_u32_e32 v38, 0x8c, v1
	ds_load_b32 v48, v114
	ds_load_2addr_stride64_b32 v[38:39], v38 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v40, v40, v48
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v47, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v38, -v38, v40, v46
	v_mul_f32_e32 v38, v39, v38
.LBB0_60:                               ; %.critedge.3
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v46, 4, v45
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s39, v46
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_52 Depth=2
	v_or_b32_e32 v39, s40, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v46, s25, v39
	v_mad_co_u64_u32 v[46:47], null, v46, s24, v[93:94]
	v_mov_b32_e32 v47, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[46:47], 2, v[46:47]
	v_add_co_u32 v46, vcc_lo, s16, v46
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v47, null, s17, v47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v39
	global_load_b32 v48, v[46:47], off
	v_add_nc_u32_e32 v46, 0x90, v1
	ds_load_b32 v49, v115
	ds_load_2addr_stride64_b32 v[46:47], v46 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v41, v41, v49
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v39, 0, v48, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v39, -v46, v41, v39
	v_mul_f32_e32 v39, v47, v39
.LBB0_62:                               ; %.critedge.4
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v41, 5, v45
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s39, v41
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_52 Depth=2
	v_or_b32_e32 v46, s40, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v40, s25, v46
	v_mad_co_u64_u32 v[40:41], null, v40, s24, v[93:94]
	v_mov_b32_e32 v41, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]
	v_add_co_u32 v40, vcc_lo, s16, v40
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v41, null, s17, v41, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v46
	global_load_b32 v47, v[40:41], off
	v_add_nc_u32_e32 v40, 0x94, v1
	ds_load_b32 v48, v116
	ds_load_2addr_stride64_b32 v[40:41], v40 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v42, v42, v48
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v47, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v40, v42, v46
	v_mul_f32_e32 v40, v41, v40
.LBB0_64:                               ; %.critedge.5
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v46, 6, v45
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s39, v46
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_52 Depth=2
	v_or_b32_e32 v41, s40, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v46, s25, v41
	v_mad_co_u64_u32 v[46:47], null, v46, s24, v[93:94]
	v_mov_b32_e32 v47, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[46:47], 2, v[46:47]
	v_add_co_u32 v46, vcc_lo, s16, v46
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v47, null, s17, v47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v41
	global_load_b32 v48, v[46:47], off
	v_add_nc_u32_e32 v46, 0x98, v1
	ds_load_b32 v49, v117
	ds_load_2addr_stride64_b32 v[46:47], v46 offset0:113 offset1:114
	s_wait_dscnt 0x1
	v_add_f32_e32 v43, v43, v49
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v41, 0, v48, vcc_lo
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v41, -v46, v43, v41
	v_mul_f32_e32 v41, v47, v41
.LBB0_66:                               ; %.critedge.6
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	v_or_b32_e32 v43, 7, v45
	s_mov_b32 s13, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s39, v43
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_52 Depth=2
	v_or_b32_e32 v45, s40, v43
	v_add_nc_u32_e32 v1, 0x9c, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v42, s25, v45
	v_mad_co_u64_u32 v[42:43], null, v42, s24, v[93:94]
	v_mov_b32_e32 v43, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v42, vcc_lo, s16, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, s17, v43, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v45
	global_load_b32 v46, v[42:43], off
	ds_load_b32 v47, v118
	ds_load_2addr_stride64_b32 v[42:43], v1 offset0:113 offset1:114
	s_wait_loadcnt_dscnt 0x1
	s_wait_alu depctr_va_vcc(0)
	v_dual_add_f32 v44, v44, v47 :: v_dual_cndmask_b32 v1, 0, v46
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v1, -v42, v44, v1
	v_mul_f32_e32 v42, v43, v1
.LBB0_68:                               ; %.critedge.7
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s13
	s_cmp_lt_i32 s12, 1
	s_mov_b32 s13, -1
                                        ; implicit-def: $vgpr43_vgpr44_vgpr45_vgpr46_vgpr47_vgpr48_vgpr49_vgpr50_vgpr51_vgpr52_vgpr53_vgpr54_vgpr55_vgpr56_vgpr57_vgpr58_vgpr59_vgpr60_vgpr61_vgpr62_vgpr63_vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71_vgpr72_vgpr73_vgpr74
	s_cbranch_scc1 .LBB0_78
; %bb.69:                               ; %NodeBlock
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_cmp_lt_i32 s12, 2
                                        ; implicit-def: $vgpr43_vgpr44_vgpr45_vgpr46_vgpr47_vgpr48_vgpr49_vgpr50_vgpr51_vgpr52_vgpr53_vgpr54_vgpr55_vgpr56_vgpr57_vgpr58_vgpr59_vgpr60_vgpr61_vgpr62_vgpr63_vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71_vgpr72_vgpr73_vgpr74
	s_cbranch_scc1 .LBB0_75
; %bb.70:                               ; %LeafBlock
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_cmp_lg_u32 s12, 2
                                        ; implicit-def: $vgpr43_vgpr44_vgpr45_vgpr46_vgpr47_vgpr48_vgpr49_vgpr50_vgpr51_vgpr52_vgpr53_vgpr54_vgpr55_vgpr56_vgpr57_vgpr58_vgpr59_vgpr60_vgpr61_vgpr62_vgpr63_vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71_vgpr72_vgpr73_vgpr74
	s_cbranch_scc0 .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_52 Depth=2
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v48, v8
	v_dual_mov_b32 v49, v9 :: v_dual_mov_b32 v50, v10
	v_dual_mov_b32 v51, v11 :: v_dual_mov_b32 v52, v12
	v_dual_mov_b32 v53, v13 :: v_dual_mov_b32 v54, v14
	v_dual_mov_b32 v55, v15 :: v_dual_mov_b32 v56, v16
	v_dual_mov_b32 v57, v17 :: v_dual_mov_b32 v58, v18
	v_dual_mov_b32 v59, v19 :: v_dual_mov_b32 v60, v20
	v_dual_mov_b32 v61, v21 :: v_dual_mov_b32 v62, v22
	v_dual_mov_b32 v63, v23 :: v_dual_mov_b32 v64, v24
	v_dual_mov_b32 v65, v25 :: v_dual_mov_b32 v66, v26
	v_dual_mov_b32 v67, v35 :: v_dual_mov_b32 v68, v36
	v_dual_mov_b32 v69, v37 :: v_dual_mov_b32 v70, v38
	v_dual_mov_b32 v71, v39 :: v_dual_mov_b32 v72, v40
	v_dual_mov_b32 v73, v41 :: v_dual_mov_b32 v74, v42
	s_mov_b32 s13, 0
.LBB0_72:                               ; %Flow2365
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_52 Depth=2
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v48, v8
	v_dual_mov_b32 v49, v9 :: v_dual_mov_b32 v50, v10
	v_dual_mov_b32 v51, v11 :: v_dual_mov_b32 v52, v12
	v_dual_mov_b32 v53, v13 :: v_dual_mov_b32 v54, v14
	v_dual_mov_b32 v55, v15 :: v_dual_mov_b32 v56, v16
	v_dual_mov_b32 v57, v17 :: v_dual_mov_b32 v58, v18
	v_dual_mov_b32 v59, v35 :: v_dual_mov_b32 v60, v36
	v_dual_mov_b32 v61, v37 :: v_dual_mov_b32 v62, v38
	v_dual_mov_b32 v63, v39 :: v_dual_mov_b32 v64, v40
	v_dual_mov_b32 v65, v41 :: v_dual_mov_b32 v66, v42
	v_dual_mov_b32 v67, v27 :: v_dual_mov_b32 v68, v28
	v_dual_mov_b32 v69, v29 :: v_dual_mov_b32 v70, v30
	v_dual_mov_b32 v71, v31 :: v_dual_mov_b32 v72, v32
	v_dual_mov_b32 v73, v33 :: v_dual_mov_b32 v74, v34
.LBB0_74:                               ; %Flow2366
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_mov_b32 s13, 0
.LBB0_75:                               ; %Flow2367
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_77
; %bb.76:                               ;   in Loop: Header=BB0_52 Depth=2
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v48, v8
	v_dual_mov_b32 v49, v9 :: v_dual_mov_b32 v50, v10
	v_dual_mov_b32 v51, v35 :: v_dual_mov_b32 v52, v36
	v_dual_mov_b32 v53, v37 :: v_dual_mov_b32 v54, v38
	v_dual_mov_b32 v55, v39 :: v_dual_mov_b32 v56, v40
	v_dual_mov_b32 v57, v41 :: v_dual_mov_b32 v58, v42
	v_dual_mov_b32 v59, v19 :: v_dual_mov_b32 v60, v20
	v_dual_mov_b32 v61, v21 :: v_dual_mov_b32 v62, v22
	v_dual_mov_b32 v63, v23 :: v_dual_mov_b32 v64, v24
	v_dual_mov_b32 v65, v25 :: v_dual_mov_b32 v66, v26
	v_dual_mov_b32 v67, v27 :: v_dual_mov_b32 v68, v28
	v_dual_mov_b32 v69, v29 :: v_dual_mov_b32 v70, v30
	v_dual_mov_b32 v71, v31 :: v_dual_mov_b32 v72, v32
	v_dual_mov_b32 v73, v33 :: v_dual_mov_b32 v74, v34
.LBB0_77:                               ; %Flow2368
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_cbranch_execnz .LBB0_51
	s_branch .LBB0_79
.LBB0_78:                               ; %Flow2369
                                        ;   in Loop: Header=BB0_52 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_51
.LBB0_79:                               ;   in Loop: Header=BB0_52 Depth=2
	v_dual_mov_b32 v65, v33 :: v_dual_mov_b32 v66, v34
	v_dual_mov_b32 v43, v11 :: v_dual_mov_b32 v44, v12
	v_dual_mov_b32 v45, v13 :: v_dual_mov_b32 v46, v14
	v_dual_mov_b32 v47, v15 :: v_dual_mov_b32 v48, v16
	v_dual_mov_b32 v49, v17 :: v_dual_mov_b32 v50, v18
	v_dual_mov_b32 v51, v19 :: v_dual_mov_b32 v52, v20
	v_dual_mov_b32 v53, v21 :: v_dual_mov_b32 v54, v22
	v_dual_mov_b32 v55, v23 :: v_dual_mov_b32 v56, v24
	v_dual_mov_b32 v57, v25 :: v_dual_mov_b32 v58, v26
	v_dual_mov_b32 v59, v27 :: v_dual_mov_b32 v60, v28
	v_dual_mov_b32 v61, v29 :: v_dual_mov_b32 v62, v30
	v_dual_mov_b32 v63, v31 :: v_dual_mov_b32 v64, v32
	v_mov_b32_e32 v74, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v73, v65 :: v_dual_mov_b32 v72, v64
	v_dual_mov_b32 v71, v63 :: v_dual_mov_b32 v70, v62
	v_dual_mov_b32 v69, v61 :: v_dual_mov_b32 v68, v60
	v_dual_mov_b32 v67, v59 :: v_dual_mov_b32 v66, v58
	v_mov_b32_e32 v65, v57
	v_mov_b32_e32 v64, v56
	v_mov_b32_e32 v63, v55
	v_mov_b32_e32 v62, v54
	v_mov_b32_e32 v61, v53
	v_mov_b32_e32 v60, v52
	v_mov_b32_e32 v59, v51
	v_mov_b32_e32 v58, v50
	v_mov_b32_e32 v57, v49
	v_mov_b32_e32 v56, v48
	v_mov_b32_e32 v55, v47
	v_mov_b32_e32 v54, v46
	v_mov_b32_e32 v53, v45
	v_mov_b32_e32 v52, v44
	v_mov_b32_e32 v51, v43
	v_mov_b32_e32 v50, v42
	v_mov_b32_e32 v49, v41
	v_mov_b32_e32 v48, v40
	v_mov_b32_e32 v47, v39
	v_mov_b32_e32 v46, v38
	v_mov_b32_e32 v45, v37
	v_mov_b32_e32 v44, v36
	v_mov_b32_e32 v43, v35
	s_branch .LBB0_51
.LBB0_80:                               ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v3, v43 :: v_dual_mov_b32 v4, v44
	v_dual_mov_b32 v5, v45 :: v_dual_mov_b32 v6, v46
	v_dual_mov_b32 v7, v47 :: v_dual_mov_b32 v8, v48
	v_dual_mov_b32 v9, v49 :: v_dual_mov_b32 v10, v50
	v_dual_mov_b32 v11, v51 :: v_dual_mov_b32 v12, v52
	v_dual_mov_b32 v13, v53 :: v_dual_mov_b32 v14, v54
	v_dual_mov_b32 v15, v55 :: v_dual_mov_b32 v16, v56
	v_dual_mov_b32 v17, v57 :: v_dual_mov_b32 v18, v58
	v_dual_mov_b32 v19, v59 :: v_dual_mov_b32 v20, v60
	v_dual_mov_b32 v21, v61 :: v_dual_mov_b32 v22, v62
	v_dual_mov_b32 v23, v63 :: v_dual_mov_b32 v24, v64
	v_dual_mov_b32 v25, v65 :: v_dual_mov_b32 v26, v66
	v_dual_mov_b32 v27, v67 :: v_dual_mov_b32 v28, v68
	v_dual_mov_b32 v29, v69 :: v_dual_mov_b32 v30, v70
	v_dual_mov_b32 v31, v71 :: v_dual_mov_b32 v32, v72
	v_dual_mov_b32 v33, v73 :: v_dual_mov_b32 v34, v74
	s_mov_b32 s12, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB0_81:                               ; %.preheader1040
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e32 vcc_lo, s12, v119
	v_or_b32_e32 v35, 0x2700, v119
	v_lshlrev_b32_e32 v37, 2, v104
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v3, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v147
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v4, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v146
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v5, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v145
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v6, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v144
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v143
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v8, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v142
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v9, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v141
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v10, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v140
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v11, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v139
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v12, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v138
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v13, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v137
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v14, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v136
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v15, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v135
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v16, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v134
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v17, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v133
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v18, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v132
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v19, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v131
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v20, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v130
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v21, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v129
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v22, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v128
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v23, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v127
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v24, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v126
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v25, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	v_or_b32_e32 v35, 0x3000, v119
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v26, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	v_or_b32_e32 v35, 0x3100, v119
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v27, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	v_or_b32_e32 v35, 0x3200, v119
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v28, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	v_or_b32_e32 v35, 0x3300, v119
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v29, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	v_or_b32_e32 v35, 0x3400, v119
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v30, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	v_or_b32_e32 v35, 0x3500, v119
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v31, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	v_or_b32_e32 v35, 0x3600, v119
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v32, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	v_or_b32_e32 v35, 0x3700, v119
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v33, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s12, v35
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, v1, v34, vcc_lo
	ds_bpermute_b32 v35, v149, v1
	s_wait_dscnt 0x0
	v_add_f32_e32 v1, v1, v35
	v_lshlrev_b32_e32 v35, 2, v98
	v_add_nc_u32_e32 v35, s12, v35
	s_addk_co_i32 s12, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s12, 0x4000
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v35, 0x2000, v35
	ds_load_2addr_b32 v[35:36], v35 offset0:32 offset1:64
	s_wait_dscnt 0x0
	ds_bpermute_b32 v38, v37, v35
	s_wait_dscnt 0x0
	v_fma_f32 v3, -v1, v38, v3
	ds_bpermute_b32 v38, v150, v35
	s_wait_dscnt 0x0
	v_fma_f32 v4, -v1, v38, v4
	ds_bpermute_b32 v38, v151, v35
	s_wait_dscnt 0x0
	v_fma_f32 v5, -v1, v38, v5
	ds_bpermute_b32 v38, v152, v35
	s_wait_dscnt 0x0
	v_fma_f32 v6, -v1, v38, v6
	ds_bpermute_b32 v38, v153, v35
	s_wait_dscnt 0x0
	v_fma_f32 v7, -v1, v38, v7
	ds_bpermute_b32 v38, v154, v35
	s_wait_dscnt 0x0
	v_fma_f32 v8, -v1, v38, v8
	ds_bpermute_b32 v38, v155, v35
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v1, v38, v9
	ds_bpermute_b32 v38, v156, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v1, v38, v10
	ds_bpermute_b32 v38, v157, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v1, v38, v11
	ds_bpermute_b32 v38, v158, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v1, v38, v12
	ds_bpermute_b32 v38, v159, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v1, v38, v13
	ds_bpermute_b32 v38, v160, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v1, v38, v14
	ds_bpermute_b32 v38, v161, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v1, v38, v15
	ds_bpermute_b32 v38, v162, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v1, v38, v16
	ds_bpermute_b32 v38, v163, v35
	ds_bpermute_b32 v35, v164, v35
	s_wait_dscnt 0x1
	v_fma_f32 v17, -v1, v38, v17
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v1, v35, v18
	ds_bpermute_b32 v35, v37, v36
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v1, v35, v19
	ds_bpermute_b32 v35, v150, v36
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v1, v35, v20
	ds_bpermute_b32 v35, v151, v36
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v1, v35, v21
	ds_bpermute_b32 v35, v152, v36
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v1, v35, v22
	ds_bpermute_b32 v35, v153, v36
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v1, v35, v23
	ds_bpermute_b32 v35, v154, v36
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v1, v35, v24
	ds_bpermute_b32 v35, v155, v36
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v1, v35, v25
	ds_bpermute_b32 v35, v156, v36
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v1, v35, v26
	ds_bpermute_b32 v35, v157, v36
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v1, v35, v27
	ds_bpermute_b32 v35, v158, v36
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v1, v35, v28
	ds_bpermute_b32 v35, v159, v36
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v1, v35, v29
	ds_bpermute_b32 v35, v160, v36
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v1, v35, v30
	ds_bpermute_b32 v35, v161, v36
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v1, v35, v31
	ds_bpermute_b32 v35, v162, v36
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v1, v35, v32
	ds_bpermute_b32 v35, v163, v36
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v1, v35, v33
	ds_bpermute_b32 v35, v164, v36
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v1, v35, v34
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc1 .LBB0_81
; %bb.82:                               ; %.preheader1046.preheader
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_mov_b32 s12, 0
.LBB0_83:                               ; %.preheader1046
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v1, s12, 11, v107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v1, 7, v1
	v_add_nc_u32_e32 v43, s40, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_i32_e32 v1, s25, v43
	v_mul_lo_u32 v1, v1, s24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[35:36], 2, v[1:2]
	v_lshl_add_u32 v1, s12, 12, v109
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, 4
	v_add_co_u32 v39, vcc_lo, v174, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, v175, v36, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s28, v43
	s_clause 0x1
	global_load_b128 v[35:38], v[39:40], off
	global_load_b128 v[39:42], v[39:40], off offset:16
	s_wait_loadcnt 0x1
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v43, 0, v35 :: v_dual_cndmask_b32 v44, 0, v37
	s_wait_loadcnt 0x0
	v_dual_cndmask_b32 v35, 0, v36 :: v_dual_cndmask_b32 v36, 0, v42
	v_dual_cndmask_b32 v37, 0, v41 :: v_dual_cndmask_b32 v40, 0, v40
	v_cndmask_b32_e32 v39, 0, v39, vcc_lo
	v_cndmask_b32_e32 v41, 0, v38, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v38.h, v36
	v_cvt_f16_f32_e32 v38.l, v37
	v_cvt_f16_f32_e32 v37.h, v40
	v_cvt_f16_f32_e32 v37.l, v39
	v_cvt_f16_f32_e32 v36.h, v41
	v_cvt_f16_f32_e32 v36.l, v44
	v_cvt_f16_f32_e32 v35.h, v35
	v_cvt_f16_f32_e32 v35.l, v43
	ds_store_b128 v1, v[35:38] offset:8320
	s_cbranch_scc0 .LBB0_83
; %bb.84:                               ;   in Loop: Header=BB0_3 Depth=1
	v_mul_lo_u32 v1, s24, v191
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v50, v110 :: v_dual_add_nc_u32 v37, s40, v121
	s_mov_b32 s41, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, s28, v37
	v_lshlrev_b64_e32 v[35:36], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s12, v178, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v49, null, v179, v36, s12
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_86
.LBB0_85:                               ;   in Loop: Header=BB0_86 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v36, v188, v43
	v_add_nc_u32_e32 v50, 0x1000, v50
	s_add_co_i32 s41, s41, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s41, 4
	v_cndmask_b32_e64 v36, 0x820, v36, s12
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v36, 2, v36
	ds_store_b32 v36, v35
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc0 .LBB0_120
.LBB0_86:                               ; %.preheader1039
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_88 Depth 3
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v51, v50
	s_mov_b64 s[22:23], 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v36, v35 :: v_dual_mov_b32 v37, v35
	v_dual_mov_b32 v38, v35 :: v_dual_mov_b32 v39, v35
	v_dual_mov_b32 v40, v35 :: v_dual_mov_b32 v41, v35
	v_mov_b32_e32 v42, v35
	s_branch .LBB0_88
.LBB0_87:                               ; %.preheader1030
                                        ;   in Loop: Header=BB0_88 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	ds_load_b128 v[52:55], v51 offset:8320
	v_add_nc_u32_e32 v51, 32, v51
	s_add_nc_u64 s[22:23], s[22:23], 64
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s22, 0x200
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[35:42], v[43:46], v[52:55], v[35:42]
	s_cbranch_scc1 .LBB0_104
.LBB0_88:                               ;   Parent Loop BB0_3 Depth=1
                                        ;     Parent Loop BB0_86 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b16_e32 v43.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v47, s12, v1, s22
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v48, null, s23, v49, s12
	v_mov_b16_e32 v43.l, v43.h
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_88 Depth=3
	global_load_b32 v44, v[47:48], off
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v43.l, v44
.LBB0_90:                               ;   in Loop: Header=BB0_88 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_92
; %bb.91:                               ;   in Loop: Header=BB0_88 Depth=3
	global_load_b32 v44, v[47:48], off offset:4
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v43.h, v44
.LBB0_92:                               ;   in Loop: Header=BB0_88 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_mov_b16_e32 v44.h, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b16_e32 v44.l, v44.h
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_88 Depth=3
	global_load_b32 v45, v[47:48], off offset:8
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v44.l, v45
.LBB0_94:                               ;   in Loop: Header=BB0_88 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_96
; %bb.95:                               ;   in Loop: Header=BB0_88 Depth=3
	global_load_b32 v45, v[47:48], off offset:12
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v44.h, v45
.LBB0_96:                               ;   in Loop: Header=BB0_88 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_mov_b16_e32 v45.h, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b16_e32 v45.l, v45.h
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_88 Depth=3
	global_load_b32 v46, v[47:48], off offset:16
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v45.l, v46
.LBB0_98:                               ;   in Loop: Header=BB0_88 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_100
; %bb.99:                               ;   in Loop: Header=BB0_88 Depth=3
	global_load_b32 v46, v[47:48], off offset:20
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v45.h, v46
.LBB0_100:                              ;   in Loop: Header=BB0_88 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	v_mov_b16_e32 v46.h, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b16_e32 v46.l, v46.h
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_102
; %bb.101:                              ;   in Loop: Header=BB0_88 Depth=3
	global_load_b32 v52, v[47:48], off offset:24
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v46.l, v52
.LBB0_102:                              ;   in Loop: Header=BB0_88 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s12
	s_and_saveexec_b32 s12, vcc_lo
	s_cbranch_execz .LBB0_87
; %bb.103:                              ;   in Loop: Header=BB0_88 Depth=3
	global_load_b32 v47, v[47:48], off offset:28
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v46.h, v47
	s_branch .LBB0_87
.LBB0_104:                              ; %.preheader1038
                                        ;   in Loop: Header=BB0_86 Depth=2
	v_lshl_or_b32 v43, s41, 4, v101
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v44, 2, v43
	v_cmp_le_u32_e64 s12, v43, v122
	s_and_b32 s12, s12, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, s12
	s_cbranch_execz .LBB0_106
; %bb.105:                              ;   in Loop: Header=BB0_86 Depth=2
	ds_load_b32 v46, v172 offset:28800
	ds_load_b32 v47, v44 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v46, v46, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v47, 0x3fb8aa3b, v46
	v_fma_f32 v48, 0x3fb8aa3b, v46, -v47
	v_rndne_f32_e32 v51, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v48, 0x32a5705f, v46 :: v_dual_sub_f32 v47, v47, v51
	v_add_f32_e32 v47, v47, v48
	v_cvt_i32_f32_e32 v48, v51
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v47, v47
	v_ldexp_f32 v47, v47, v48
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v47, 0, v47, s13
	v_cmp_nlt_f32_e64 s13, 0x42b17218, v46
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v46, 0x7f800000, v47, s13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v46, v35, v46
.LBB0_106:                              ;   in Loop: Header=BB0_86 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v35, v181, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v35, 0x820, v35, s12
	v_cmp_le_u32_e64 s12, v43, v166
	v_lshlrev_b32_e32 v35, 2, v35
	s_and_b32 s12, s12, s5
	ds_store_b32 v35, v46
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, s12
	s_cbranch_execz .LBB0_108
; %bb.107:                              ;   in Loop: Header=BB0_86 Depth=2
	ds_load_b32 v35, v172 offset:28804
	ds_load_b32 v45, v44 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v35, v35, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v45, 0x3fb8aa3b, v35
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v35
	v_fma_f32 v46, 0x3fb8aa3b, v35, -v45
	v_rndne_f32_e32 v47, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v46, 0x32a5705f, v35
	v_sub_f32_e32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v45, v45, v46
	v_cvt_i32_f32_e32 v46, v47
	v_exp_f32_e32 v45, v45
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v45, v45, v46
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v45, 0, v45, s13
	v_cmp_nlt_f32_e64 s13, 0x42b17218, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v35, 0x7f800000, v45, s13
	v_mul_f32_e32 v45, v36, v35
.LBB0_108:                              ;   in Loop: Header=BB0_86 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mov_b32 v36, 0 :: v_dual_add_nc_u32 v35, v182, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v35, 0x820, v35, s12
	v_cmp_le_u32_e64 s12, v43, v167
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v46, 2, v35
	s_and_b32 s12, s12, s6
	ds_store_b32 v46, v45
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, s12
	s_cbranch_execz .LBB0_110
; %bb.109:                              ;   in Loop: Header=BB0_86 Depth=2
	ds_load_b32 v36, v172 offset:28808
	ds_load_b32 v45, v44 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v36, v36, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v45, 0x3fb8aa3b, v36
	v_fma_f32 v46, 0x3fb8aa3b, v36, -v45
	v_rndne_f32_e32 v47, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v45, v45, v47 :: v_dual_fmac_f32 v46, 0x32a5705f, v36
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v36
	v_add_f32_e32 v45, v45, v46
	v_cvt_i32_f32_e32 v46, v47
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v45, v45
	v_ldexp_f32 v45, v45, v46
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v45, 0, v45, s13
	v_cmp_nlt_f32_e64 s13, 0x42b17218, v36
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0x7f800000, v45, s13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v36, v37, v36
.LBB0_110:                              ;   in Loop: Header=BB0_86 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v37, v183, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v37, 0x820, v37, s12
	v_cmp_le_u32_e64 s12, v43, v168
	v_lshlrev_b32_e32 v37, 2, v37
	s_and_b32 s12, s12, s7
	ds_store_b32 v37, v36
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, s12
	s_cbranch_execz .LBB0_112
; %bb.111:                              ;   in Loop: Header=BB0_86 Depth=2
	ds_load_b32 v35, v172 offset:28812
	ds_load_b32 v36, v44 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v35, v35, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, 0x3fb8aa3b, v35
	v_fma_f32 v37, 0x3fb8aa3b, v35, -v36
	v_rndne_f32_e32 v45, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v36, v36, v45 :: v_dual_fmac_f32 v37, 0x32a5705f, v35
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v35
	v_add_f32_e32 v36, v36, v37
	v_cvt_i32_f32_e32 v37, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v36, v36
	v_ldexp_f32 v36, v36, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v36, 0, v36, s13
	v_cmp_nlt_f32_e64 s13, 0x42b17218, v35
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v35, 0x7f800000, v36, s13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v35, v38, v35
.LBB0_112:                              ;   in Loop: Header=BB0_86 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_dual_mov_b32 v37, 0 :: v_dual_add_nc_u32 v36, v184, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v36, 0x820, v36, s12
	v_cmp_le_u32_e64 s12, v43, v169
	v_lshlrev_b32_e32 v38, 2, v36
	v_mov_b32_e32 v36, 0
	s_and_b32 s12, s12, s8
	ds_store_b32 v38, v35
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, s12
	s_cbranch_execz .LBB0_114
; %bb.113:                              ;   in Loop: Header=BB0_86 Depth=2
	ds_load_b32 v35, v172 offset:28816
	ds_load_b32 v37, v44 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v35, v35, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v37, 0x3fb8aa3b, v35
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v35
	v_fma_f32 v38, 0x3fb8aa3b, v35, -v37
	v_rndne_f32_e32 v45, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v38, 0x32a5705f, v35 :: v_dual_sub_f32 v37, v37, v45
	v_add_f32_e32 v37, v37, v38
	v_cvt_i32_f32_e32 v38, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v37, v37
	v_ldexp_f32 v37, v37, v38
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v37, 0, v37, s13
	v_cmp_nlt_f32_e64 s13, 0x42b17218, v35
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v35, 0x7f800000, v37, s13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v37, v39, v35
.LBB0_114:                              ;   in Loop: Header=BB0_86 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v35, v185, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v35, 0x820, v35, s12
	v_cmp_le_u32_e64 s12, v43, v170
	v_lshlrev_b32_e32 v35, 2, v35
	s_and_b32 s12, s12, s9
	ds_store_b32 v35, v37
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, s12
	s_cbranch_execz .LBB0_116
; %bb.115:                              ;   in Loop: Header=BB0_86 Depth=2
	ds_load_b32 v35, v172 offset:28820
	ds_load_b32 v36, v44 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v35, v35, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, 0x3fb8aa3b, v35
	v_fma_f32 v37, 0x3fb8aa3b, v35, -v36
	v_rndne_f32_e32 v38, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v36, v36, v38 :: v_dual_fmac_f32 v37, 0x32a5705f, v35
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v35
	v_add_f32_e32 v36, v36, v37
	v_cvt_i32_f32_e32 v37, v38
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v36, v36
	v_ldexp_f32 v36, v36, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v36, 0, v36, s13
	v_cmp_nlt_f32_e64 s13, 0x42b17218, v35
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v35, 0x7f800000, v36, s13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v36, v40, v35
.LBB0_116:                              ;   in Loop: Header=BB0_86 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v35, v186, v43
	v_mov_b32_e32 v37, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v35, 0x820, v35, s12
	v_cmp_le_u32_e64 s12, v43, v171
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v38, 2, v35
	s_and_b32 s12, s12, s10
	ds_store_b32 v38, v36
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, s12
	s_cbranch_execz .LBB0_118
; %bb.117:                              ;   in Loop: Header=BB0_86 Depth=2
	ds_load_b32 v36, v172 offset:28824
	ds_load_b32 v37, v44 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v36, v36, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v37, 0x3fb8aa3b, v36
	v_fma_f32 v38, 0x3fb8aa3b, v36, -v37
	v_rndne_f32_e32 v39, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v37, v37, v39 :: v_dual_fmac_f32 v38, 0x32a5705f, v36
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v36
	v_add_f32_e32 v37, v37, v38
	v_cvt_i32_f32_e32 v38, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v37, v37
	v_ldexp_f32 v37, v37, v38
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v37, 0, v37, s13
	v_cmp_nlt_f32_e64 s13, 0x42b17218, v36
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v36, 0x7f800000, v37, s13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v37, v41, v36
.LBB0_118:                              ;   in Loop: Header=BB0_86 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v36, v187, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v36, 0x820, v36, s12
	v_cmp_le_u32_e64 s12, v43, v165
	v_lshlrev_b32_e32 v36, 2, v36
	s_and_b32 s12, s12, s11
	ds_store_b32 v36, v37
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s22, s12
	s_cbranch_execz .LBB0_85
; %bb.119:                              ;   in Loop: Header=BB0_86 Depth=2
	ds_load_b32 v35, v172 offset:28828
	ds_load_b32 v36, v44 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v35, v35, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v36, 0x3fb8aa3b, v35
	v_fma_f32 v37, 0x3fb8aa3b, v35, -v36
	v_rndne_f32_e32 v38, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v36, v36, v38 :: v_dual_fmac_f32 v37, 0x32a5705f, v35
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v35
	v_add_f32_e32 v36, v36, v37
	v_cvt_i32_f32_e32 v37, v38
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v36, v36
	v_ldexp_f32 v36, v36, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v36, 0, v36, s13
	v_cmp_nlt_f32_e64 s13, 0x42b17218, v35
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v35, 0x7f800000, v36, s13
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v35, v42, v35
	s_branch .LBB0_85
.LBB0_120:                              ; %.preheader1045.preheader
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_mov_b32_e32 v1, v106
	s_movk_i32 s4, 0xff00
	s_branch .LBB0_122
.LBB0_121:                              ;   in Loop: Header=BB0_122 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v36, s4, v123
	v_add_nc_u32_e32 v1, 4, v1
	s_add_co_i32 s4, s4, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 0
	s_wait_dscnt 0x0
	ds_store_b32 v36, v35 offset:8576
	s_cbranch_scc1 .LBB0_124
.LBB0_122:                              ; %.preheader1045
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v35, 0
	s_mov_b32 s5, exec_lo
	v_cmpx_le_u32_e64 v105, v1
	s_cbranch_execz .LBB0_121
; %bb.123:                              ;   in Loop: Header=BB0_122 Depth=2
	v_mul_lo_u32 v35, v1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v35, v35, v1, 1
	v_and_b32_e32 v35, -4, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v35, v120, v35
	ds_load_b32 v35, v35
	s_branch .LBB0_121
.LBB0_124:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cvt_f16_f32_e32 v43.l, v3
	v_cvt_f16_f32_e32 v43.h, v4
	v_cvt_f16_f32_e32 v44.l, v5
	v_cvt_f16_f32_e32 v44.h, v6
	v_cvt_f16_f32_e32 v45.l, v7
	v_cvt_f16_f32_e32 v45.h, v8
	v_cvt_f16_f32_e32 v46.l, v9
	v_cvt_f16_f32_e32 v46.h, v10
	v_cvt_f16_f32_e32 v47.l, v11
	v_cvt_f16_f32_e32 v47.h, v12
	v_cvt_f16_f32_e32 v48.l, v13
	v_cvt_f16_f32_e32 v48.h, v14
	v_cvt_f16_f32_e32 v49.l, v15
	v_cvt_f16_f32_e32 v49.h, v16
	v_cvt_f16_f32_e32 v50.l, v17
	v_cvt_f16_f32_e32 v50.h, v18
	v_cvt_f16_f32_e32 v51.l, v19
	v_cvt_f16_f32_e32 v51.h, v20
	v_cvt_f16_f32_e32 v52.l, v21
	v_cvt_f16_f32_e32 v52.h, v22
	v_cvt_f16_f32_e32 v53.l, v23
	v_cvt_f16_f32_e32 v53.h, v24
	v_cvt_f16_f32_e32 v54.l, v25
	v_cvt_f16_f32_e32 v54.h, v26
	v_cvt_f16_f32_e32 v55.l, v27
	v_cvt_f16_f32_e32 v55.h, v28
	v_cvt_f16_f32_e32 v56.l, v29
	v_cvt_f16_f32_e32 v56.h, v30
	v_cvt_f16_f32_e32 v57.l, v31
	v_cvt_f16_f32_e32 v57.h, v32
	v_cvt_f16_f32_e32 v58.l, v33
	v_cvt_f16_f32_e32 v58.h, v34
	s_or_b32 s5, s40, 1
	s_or_b32 s6, s40, 2
	s_or_b32 s7, s40, 3
	s_or_b32 s8, s40, 4
	s_or_b32 s9, s40, 5
	s_or_b32 s10, s40, 6
	s_or_b32 s11, s40, 7
	s_mov_b32 s12, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_branch .LBB0_126
.LBB0_125:                              ;   in Loop: Header=BB0_126 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s12, 4
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB0_142
.LBB0_126:                              ; %.preheader
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s13, s12, 4
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s13, s40
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s4, v108
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s28, v1
	v_min_i32_e32 v1, s25, v1
	v_mul_lo_u32 v1, v1, s24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[1:2]
	v_add_co_u32 v39, s4, v176, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v40, null, v177, v36, s4
	s_mov_b32 s4, exec_lo
	s_clause 0x1
	global_load_b128 v[35:38], v[39:40], off offset:16
	global_load_b128 v[39:42], v[39:40], off
	s_wait_loadcnt 0x1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v35, 0, v35, vcc_lo
	s_wait_loadcnt 0x0
	v_dual_cndmask_b32 v1, 0, v39 :: v_dual_cndmask_b32 v36, 0, v36
	v_dual_cndmask_b32 v39, 0, v40 :: v_dual_cndmask_b32 v38, 0, v38
	v_dual_cndmask_b32 v40, 0, v41 :: v_dual_cndmask_b32 v41, 0, v42
	v_cndmask_b32_e32 v37, 0, v37, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v38.h, v38
	v_cvt_f16_f32_e32 v38.l, v37
	v_cvt_f16_f32_e32 v37.h, v36
	v_cvt_f16_f32_e32 v37.l, v35
	v_cvt_f16_f32_e32 v36.h, v41
	v_cvt_f16_f32_e32 v36.l, v40
	v_cvt_f16_f32_e32 v35.h, v39
	v_cvt_f16_f32_e32 v35.l, v1
	v_lshl_add_u32 v1, s12, 6, v125
	ds_store_b128 v109, v[35:38] offset:24704
	v_add_nc_u32_e32 v1, 0x80, v1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b128 v[59:62], v124 offset:24704
	ds_load_b128 v[63:66], v124 offset:24736
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[35:42], v[59:62], v[75:78], 0
	ds_load_b128 v[59:62], v124 offset:24768
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[35:42], v[63:66], v[79:82], v[35:42]
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[35:42], v[59:62], v[83:86], v[35:42]
	ds_load_b128 v[59:62], v124 offset:24800
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[35:42], v[59:62], v[87:90], v[35:42]
	ds_store_b128 v190, v[35:38]
	ds_store_b128 v190, v[39:42] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[35:36], v1 offset0:32 offset1:33
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v59.l, v35
	v_cvt_f16_f32_e32 v59.h, v36
	ds_load_2addr_stride64_b32 v[35:36], v1 offset0:34 offset1:35
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v60.l, v35
	v_cvt_f16_f32_e32 v60.h, v36
	ds_load_2addr_stride64_b32 v[35:36], v1 offset0:36 offset1:37
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v61.l, v35
	v_cvt_f16_f32_e32 v61.h, v36
	ds_load_2addr_stride64_b32 v[35:36], v1 offset0:38 offset1:39
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v62.l, v35
	v_cvt_f16_f32_e32 v62.h, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[35:42], v[59:62], v[43:46], 0
	ds_load_2addr_stride64_b32 v[59:60], v1 offset0:48 offset1:49
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v59.l, v59
	v_cvt_f16_f32_e32 v59.h, v60
	ds_load_2addr_stride64_b32 v[60:61], v1 offset0:50 offset1:51
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v60.l, v60
	v_cvt_f16_f32_e32 v60.h, v61
	ds_load_2addr_stride64_b32 v[61:62], v1 offset0:52 offset1:53
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v61.l, v61
	v_cvt_f16_f32_e32 v61.h, v62
	ds_load_2addr_stride64_b32 v[62:63], v1 offset0:54 offset1:55
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v62.l, v62
	v_cvt_f16_f32_e32 v62.h, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[35:42], v[59:62], v[47:50], v[35:42]
	ds_load_2addr_stride64_b32 v[59:60], v1 offset0:64 offset1:65
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v59.l, v59
	v_cvt_f16_f32_e32 v59.h, v60
	ds_load_2addr_stride64_b32 v[60:61], v1 offset0:66 offset1:67
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v60.l, v60
	v_cvt_f16_f32_e32 v60.h, v61
	ds_load_2addr_stride64_b32 v[61:62], v1 offset0:68 offset1:69
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v61.l, v61
	v_cvt_f16_f32_e32 v61.h, v62
	ds_load_2addr_stride64_b32 v[62:63], v1 offset0:70 offset1:71
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v62.l, v62
	v_cvt_f16_f32_e32 v62.h, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[35:42], v[59:62], v[51:54], v[35:42]
	ds_load_2addr_stride64_b32 v[59:60], v1 offset0:80 offset1:81
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v59.l, v59
	v_cvt_f16_f32_e32 v59.h, v60
	ds_load_2addr_stride64_b32 v[60:61], v1 offset0:82 offset1:83
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v60.l, v60
	v_cvt_f16_f32_e32 v60.h, v61
	ds_load_2addr_stride64_b32 v[61:62], v1 offset0:84 offset1:85
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v61.l, v61
	v_cvt_f16_f32_e32 v61.h, v62
	ds_load_2addr_stride64_b32 v[62:63], v1 offset0:86 offset1:87
	v_or_b32_e32 v1, s13, v104
	s_wait_dscnt 0x0
	v_cvt_f16_f32_e32 v62.l, v62
	v_cvt_f16_f32_e32 v62.h, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[35:42], v[59:62], v[55:58], v[35:42]
	v_or_b32_e32 v60, s40, v1
	v_lshlrev_b32_e32 v59, 2, v1
	v_cmpx_gt_i32_e64 s28, v60
	s_cbranch_execz .LBB0_128
; %bb.127:                              ;   in Loop: Header=BB0_126 Depth=2
	ds_load_b32 v62, v190
	ds_load_b32 v63, v111
	v_mad_co_u64_u32 v[60:61], null, v60, s24, v[93:94]
	v_mov_b32_e32 v61, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	v_add_co_u32 v60, vcc_lo, s26, v60
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, s27, v61, vcc_lo
	s_wait_dscnt 0x0
	v_add_f32_e32 v62, v62, v63
	ds_load_b32 v63, v59 offset:29056
	s_wait_dscnt 0x0
	v_fma_f32 v35, v63, v62, v35
	global_store_b32 v[60:61], v35, off
.LBB0_128:                              ;   in Loop: Header=BB0_126 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v35, s5, v1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s28, v35
	s_cbranch_execz .LBB0_130
; %bb.129:                              ;   in Loop: Header=BB0_126 Depth=2
	ds_load_b32 v62, v190 offset:4
	ds_load_b32 v63, v59 offset:29060
	ds_load_b32 v64, v112
	v_mad_co_u64_u32 v[60:61], null, v35, s24, v[93:94]
	v_mov_b32_e32 v61, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]
	s_wait_dscnt 0x0
	v_add_f32_e32 v35, v62, v64
	v_fma_f32 v62, v63, v35, v36
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v35, vcc_lo, s26, v60
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, s27, v61, vcc_lo
	global_store_b32 v[35:36], v62, off
.LBB0_130:                              ;   in Loop: Header=BB0_126 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v35, s6, v1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s28, v35
	s_cbranch_execz .LBB0_132
; %bb.131:                              ;   in Loop: Header=BB0_126 Depth=2
	ds_load_b32 v60, v190 offset:8
	ds_load_b32 v61, v59 offset:29064
	ds_load_b32 v62, v113
	v_mad_co_u64_u32 v[35:36], null, v35, s24, v[93:94]
	v_mov_b32_e32 v36, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, vcc_lo, s26, v35
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v36, null, s27, v36, vcc_lo
	s_wait_dscnt 0x0
	v_add_f32_e32 v60, v60, v62
	v_fma_f32 v37, v61, v60, v37
	global_store_b32 v[35:36], v37, off
.LBB0_132:                              ;   in Loop: Header=BB0_126 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v35, s7, v1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s28, v35
	s_cbranch_execz .LBB0_134
; %bb.133:                              ;   in Loop: Header=BB0_126 Depth=2
	ds_load_b32 v37, v190 offset:12
	ds_load_b32 v60, v59 offset:29068
	ds_load_b32 v61, v114
	v_mad_co_u64_u32 v[35:36], null, v35, s24, v[93:94]
	s_wait_dscnt 0x0
	v_dual_mov_b32 v36, v2 :: v_dual_add_f32 v37, v37, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_fma_f32 v37, v60, v37, v38
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v35, vcc_lo, s26, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, s27, v36, vcc_lo
	global_store_b32 v[35:36], v37, off
.LBB0_134:                              ;   in Loop: Header=BB0_126 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v35, s8, v1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s28, v35
	s_cbranch_execz .LBB0_136
; %bb.135:                              ;   in Loop: Header=BB0_126 Depth=2
	ds_load_b32 v37, v190 offset:16
	ds_load_b32 v38, v59 offset:29072
	ds_load_b32 v60, v115
	v_mad_co_u64_u32 v[35:36], null, v35, s24, v[93:94]
	s_wait_dscnt 0x0
	v_dual_mov_b32 v36, v2 :: v_dual_add_f32 v37, v37, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_fma_f32 v37, v38, v37, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v35, vcc_lo, s26, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, s27, v36, vcc_lo
	global_store_b32 v[35:36], v37, off
.LBB0_136:                              ;   in Loop: Header=BB0_126 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v35, s9, v1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s28, v35
	s_cbranch_execz .LBB0_138
; %bb.137:                              ;   in Loop: Header=BB0_126 Depth=2
	ds_load_b32 v37, v190 offset:20
	ds_load_b32 v38, v59 offset:29076
	ds_load_b32 v39, v116
	v_mad_co_u64_u32 v[35:36], null, v35, s24, v[93:94]
	s_wait_dscnt 0x0
	v_dual_mov_b32 v36, v2 :: v_dual_add_f32 v37, v37, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_fma_f32 v37, v38, v37, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v35, vcc_lo, s26, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, s27, v36, vcc_lo
	global_store_b32 v[35:36], v37, off
.LBB0_138:                              ;   in Loop: Header=BB0_126 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v35, s10, v1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s28, v35
	s_cbranch_execz .LBB0_140
; %bb.139:                              ;   in Loop: Header=BB0_126 Depth=2
	ds_load_b32 v37, v190 offset:24
	ds_load_b32 v38, v59 offset:29080
	ds_load_b32 v39, v117
	v_mad_co_u64_u32 v[35:36], null, v35, s24, v[93:94]
	s_wait_dscnt 0x0
	v_dual_mov_b32 v36, v2 :: v_dual_add_f32 v37, v37, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_fma_f32 v37, v38, v37, v41
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v35, vcc_lo, s26, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, s27, v36, vcc_lo
	global_store_b32 v[35:36], v37, off
.LBB0_140:                              ;   in Loop: Header=BB0_126 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v1, s11, v1
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s28, v1
	s_cbranch_execz .LBB0_125
; %bb.141:                              ;   in Loop: Header=BB0_126 Depth=2
	ds_load_b32 v37, v190 offset:28
	ds_load_b32 v38, v59 offset:29084
	ds_load_b32 v39, v118
	v_mad_co_u64_u32 v[35:36], null, v1, s24, v[93:94]
	s_wait_dscnt 0x0
	v_dual_mov_b32 v36, v2 :: v_dual_add_f32 v1, v37, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_fmac_f32_e32 v42, v38, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v35, vcc_lo, s26, v35
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v36, null, s27, v36, vcc_lo
	global_store_b32 v[35:36], v42, off
	s_branch .LBB0_125
.LBB0_142:                              ; %.preheader1044
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_mad_co_u64_u32 v[35:36], null, v189, s24, v[94:95]
	s_mov_b32 s4, 0
.LBB0_143:                              ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v38, s4, 8, v0
	s_add_co_i32 s4, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s4, 16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v1, 6, v38
	v_add_nc_u32_e32 v1, v35, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[36:37], 2, v[1:2]
	v_add_co_u32 v36, vcc_lo, s14, v36
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v37, null, s15, v37, vcc_lo
	global_load_b32 v1, v[36:37], off
	v_lshlrev_b32_e32 v36, 1, v38
	s_wait_loadcnt 0x0
	v_cvt_f16_f32_e32 v1.l, v1
	v_cndmask_b16 v1.l, 0, v1.l, s3
	ds_store_b16 v36, v1 offset:8320
	s_cbranch_scc0 .LBB0_143
; %bb.144:                              ;   in Loop: Header=BB0_3 Depth=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v47, v75 :: v_dual_mov_b32 v48, v76
	v_dual_mov_b32 v49, v77 :: v_dual_mov_b32 v50, v78
	v_dual_mov_b32 v51, v79 :: v_dual_mov_b32 v52, v80
	v_dual_mov_b32 v53, v81 :: v_dual_mov_b32 v54, v82
	v_dual_mov_b32 v55, v83 :: v_dual_mov_b32 v56, v84
	v_dual_mov_b32 v57, v85 :: v_dual_mov_b32 v58, v86
	v_dual_mov_b32 v59, v87 :: v_dual_mov_b32 v60, v88
	v_dual_mov_b32 v61, v89 :: v_dual_mov_b32 v62, v90
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b64 v[71:72], v2 offset:29568
	s_mov_b32 s3, 0
	s_branch .LBB0_147
.LBB0_145:                              ;   in Loop: Header=BB0_147 Depth=2
	v_dual_mov_b32 v45, v53 :: v_dual_mov_b32 v46, v54
	v_dual_mov_b32 v53, v61 :: v_dual_mov_b32 v54, v62
	v_dual_mov_b32 v43, v51 :: v_dual_mov_b32 v44, v52
	v_dual_mov_b32 v47, v55 :: v_dual_mov_b32 v48, v56
	v_dual_mov_b32 v49, v57 :: v_dual_mov_b32 v50, v58
	v_dual_mov_b32 v51, v59 :: v_dual_mov_b32 v52, v60
	v_dual_mov_b32 v90, v54 :: v_dual_mov_b32 v89, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v86, v50 :: v_dual_mov_b32 v85, v49
	v_dual_mov_b32 v88, v52 :: v_dual_mov_b32 v87, v51
	v_dual_mov_b32 v84, v48 :: v_dual_mov_b32 v83, v47
	v_dual_mov_b32 v82, v46 :: v_dual_mov_b32 v81, v45
	v_dual_mov_b32 v80, v44 :: v_dual_mov_b32 v79, v43
	v_dual_mov_b32 v78, v42 :: v_dual_mov_b32 v77, v41
	v_dual_mov_b32 v76, v40 :: v_dual_mov_b32 v75, v39
.LBB0_146:                              ;   in Loop: Header=BB0_147 Depth=2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v47, v75 :: v_dual_mov_b32 v48, v76
	v_dual_mov_b32 v49, v77 :: v_dual_mov_b32 v50, v78
	v_dual_mov_b32 v51, v79 :: v_dual_mov_b32 v52, v80
	v_dual_mov_b32 v53, v81 :: v_dual_mov_b32 v54, v82
	v_dual_mov_b32 v55, v83 :: v_dual_mov_b32 v56, v84
	v_dual_mov_b32 v57, v85 :: v_dual_mov_b32 v58, v86
	v_dual_mov_b32 v59, v87 :: v_dual_mov_b32 v60, v88
	v_dual_mov_b32 v61, v89 :: v_dual_mov_b32 v62, v90
	s_add_co_i32 s3, s3, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s3, 4
	s_cbranch_scc1 .LBB0_2
.LBB0_147:                              ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_156 Depth 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s3, 0
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_191
; %bb.148:                              ;   in Loop: Header=BB0_147 Depth=2
	s_cmp_lg_u32 s3, 1
	s_mov_b32 s5, -1
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr35
                                        ; implicit-def: $vgpr36
                                        ; implicit-def: $vgpr37
	s_cbranch_scc0 .LBB0_150
; %bb.149:                              ;   in Loop: Header=BB0_147 Depth=2
	s_cmp_eq_u32 s3, 2
	s_mov_b32 s5, 0
	s_cselect_b32 vcc_lo, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_dual_cndmask_b32 v37, v59, v55 :: v_dual_cndmask_b32 v36, v60, v56
	v_cndmask_b32_e32 v35, v61, v57, vcc_lo
	v_cndmask_b32_e32 v1, v62, v58, vcc_lo
.LBB0_150:                              ; %Flow2362
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_152
; %bb.151:                              ;   in Loop: Header=BB0_147 Depth=2
	v_dual_mov_b32 v1, v54 :: v_dual_mov_b32 v36, v52
	v_mov_b32_e32 v35, v53
	v_mov_b32_e32 v37, v51
.LBB0_152:                              ; %Flow2363
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_cbranch_execnz .LBB0_154
.LBB0_153:                              ;   in Loop: Header=BB0_147 Depth=2
	v_dual_mov_b32 v37, v47 :: v_dual_mov_b32 v36, v48
	v_mov_b32_e32 v35, v49
	v_mov_b32_e32 v1, v50
.LBB0_154:                              ; %.preheader1035
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_mix_f32 v65, v36, v71, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v66, v36, v71, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_lshlrev_b32_e32 v36, 7, v101
	v_fma_mix_f32 v63, v37, v71, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v64, v37, v71, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v35, v71, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v35, v71, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v1, v71, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v70, v1, v71, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_lshl_or_b32 v1, s3, 11, v36
	s_mov_b32 s5, 0
	s_branch .LBB0_156
.LBB0_155:                              ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_fma_mixlo_f16 v35, v35, v74, 0
	v_fma_mixhi_f16 v35, v36, v73, 0
	v_fma_mixlo_f16 v36, v37, v79, 0
	v_fma_mixhi_f16 v36, v38, v78, 0
	v_fma_mixlo_f16 v37, v39, v81, 0
	v_fma_mixhi_f16 v37, v40, v80, 0
	v_fma_mixlo_f16 v38, v41, v83, 0
	v_fma_mixhi_f16 v38, v42, v82, 0
	s_add_co_i32 s5, s5, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s5, 4
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[63:70], v[43:46], v[35:38], v[63:70]
	s_cbranch_scc1 .LBB0_181
.LBB0_156:                              ; %NodeBlock2343
                                        ;   Parent Loop BB0_3 Depth=1
                                        ;     Parent Loop BB0_147 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_dual_mov_b32 v42, v10 :: v_dual_mov_b32 v41, v9
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v35, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s5, 1
	s_cbranch_scc1 .LBB0_165
; %bb.157:                              ; %NodeBlock2341
                                        ;   in Loop: Header=BB0_156 Depth=3
	s_cmp_lt_i32 s5, 2
	s_mov_b32 s6, -1
                                        ; implicit-def: $vgpr35_vgpr36_vgpr37_vgpr38_vgpr39_vgpr40_vgpr41_vgpr42
	s_cbranch_scc1 .LBB0_163
; %bb.158:                              ; %LeafBlock2339
                                        ;   in Loop: Header=BB0_156 Depth=3
	s_cmp_lg_u32 s5, 2
	s_cbranch_scc0 .LBB0_160
; %bb.159:                              ;   in Loop: Header=BB0_156 Depth=3
	s_mov_b32 s6, 0
.LBB0_160:                              ; %Flow2357
                                        ;   in Loop: Header=BB0_156 Depth=3
	v_dual_mov_b32 v42, v34 :: v_dual_mov_b32 v41, v33
	v_dual_mov_b32 v40, v32 :: v_dual_mov_b32 v39, v31
	v_dual_mov_b32 v38, v30 :: v_dual_mov_b32 v37, v29
	v_dual_mov_b32 v36, v28 :: v_dual_mov_b32 v35, v27
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_162
; %bb.161:                              ; %switch.edge1216
                                        ;   in Loop: Header=BB0_156 Depth=3
	v_dual_mov_b32 v42, v26 :: v_dual_mov_b32 v41, v25
	v_dual_mov_b32 v40, v24 :: v_dual_mov_b32 v39, v23
	v_dual_mov_b32 v38, v22 :: v_dual_mov_b32 v37, v21
	v_dual_mov_b32 v36, v20 :: v_dual_mov_b32 v35, v19
.LBB0_162:                              ; %Flow2358
                                        ;   in Loop: Header=BB0_156 Depth=3
	s_mov_b32 s6, 0
.LBB0_163:                              ; %Flow2359
                                        ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_165
; %bb.164:                              ;   in Loop: Header=BB0_156 Depth=3
	v_dual_mov_b32 v42, v18 :: v_dual_mov_b32 v41, v17
	v_dual_mov_b32 v40, v16 :: v_dual_mov_b32 v39, v15
	v_dual_mov_b32 v38, v14 :: v_dual_mov_b32 v37, v13
	v_dual_mov_b32 v36, v12 :: v_dual_mov_b32 v35, v11
.LBB0_165:                              ;   in Loop: Header=BB0_156 Depth=3
	v_lshl_or_b32 v76, s5, 4, v104
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v74, 0
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_add_u32 v77, v76, 1, v1
	v_lshlrev_b32_e32 v75, 2, v76
	ds_load_u16_d16 v43, v77 offset:8320
	v_cmpx_gt_i32_e64 s39, v76
	s_cbranch_execz .LBB0_167
; %bb.166:                              ;   in Loop: Header=BB0_156 Depth=3
	ds_load_b32 v44, v75 offset:28800
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v72, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v45, 0x3fb8aa3b, v44
	v_fma_f32 v46, 0x3fb8aa3b, v44, -v45
	v_rndne_f32_e32 v74, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v45, v45, v74
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fmac_f32_e32 v46, 0x32a5705f, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v45, v45, v46
	v_cvt_i32_f32_e32 v46, v74
	v_exp_f32_e32 v45, v45
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v45, v45, v46
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v45, 0, v45, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v74, 0x7f800000, v45, vcc_lo
.LBB0_167:                              ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_load_u16_d16_hi v43, v77 offset:8322
	v_or_b32_e32 v44, 1, v76
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s39, v44
	s_cbranch_execz .LBB0_169
; %bb.168:                              ;   in Loop: Header=BB0_156 Depth=3
	ds_load_b32 v44, v75 offset:28804
	s_wait_dscnt 0x0
	v_sub_f32_e32 v44, v72, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v45, 0x3fb8aa3b, v44
	v_fma_f32 v46, 0x3fb8aa3b, v44, -v45
	v_rndne_f32_e32 v73, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v45, v45, v73
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v44
	v_fmac_f32_e32 v46, 0x32a5705f, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v45, v45, v46
	v_cvt_i32_f32_e32 v46, v73
	v_exp_f32_e32 v45, v45
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v45, v45, v46
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v45, 0, v45, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v44
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v73, 0x7f800000, v45, vcc_lo
.LBB0_169:                              ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_load_u16_d16 v44, v77 offset:8324
	v_or_b32_e32 v45, 2, v76
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v79, 0
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s39, v45
	s_cbranch_execz .LBB0_171
; %bb.170:                              ;   in Loop: Header=BB0_156 Depth=3
	ds_load_b32 v45, v75 offset:28808
	s_wait_dscnt 0x0
	v_sub_f32_e32 v45, v72, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v46, 0x3fb8aa3b, v45
	v_fma_f32 v79, 0x3fb8aa3b, v45, -v46
	v_rndne_f32_e32 v80, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v46, v46, v80
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v45
	v_fmac_f32_e32 v79, 0x32a5705f, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v46, v46, v79
	v_cvt_i32_f32_e32 v79, v80
	v_exp_f32_e32 v46, v46
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v46, v46, v79
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v46, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v45
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v79, 0x7f800000, v46, vcc_lo
.LBB0_171:                              ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_load_u16_d16_hi v44, v77 offset:8326
	v_or_b32_e32 v45, 3, v76
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s39, v45
	s_cbranch_execz .LBB0_173
; %bb.172:                              ;   in Loop: Header=BB0_156 Depth=3
	ds_load_b32 v45, v75 offset:28812
	s_wait_dscnt 0x0
	v_sub_f32_e32 v45, v72, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v46, 0x3fb8aa3b, v45
	v_fma_f32 v78, 0x3fb8aa3b, v45, -v46
	v_rndne_f32_e32 v80, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v46, v46, v80
	v_fmac_f32_e32 v78, 0x32a5705f, v45
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v46, v46, v78
	v_cvt_i32_f32_e32 v78, v80
	v_exp_f32_e32 v46, v46
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v46, v46, v78
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, 0, v46, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v45
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v78, 0x7f800000, v46, vcc_lo
.LBB0_173:                              ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_load_u16_d16 v45, v77 offset:8328
	v_or_b32_e32 v46, 4, v76
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v81, 0
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s39, v46
	s_cbranch_execz .LBB0_175
; %bb.174:                              ;   in Loop: Header=BB0_156 Depth=3
	ds_load_b32 v46, v75 offset:28816
	s_wait_dscnt 0x0
	v_sub_f32_e32 v46, v72, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v81, 0x3fb8aa3b, v46
	v_fma_f32 v82, 0x3fb8aa3b, v46, -v81
	v_rndne_f32_e32 v83, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v81, v81, v83
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v46
	v_fmac_f32_e32 v82, 0x32a5705f, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v81, v81, v82
	v_cvt_i32_f32_e32 v82, v83
	v_exp_f32_e32 v81, v81
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v81, v81, v82
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v81, 0, v81, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v46
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v81, 0x7f800000, v81, vcc_lo
.LBB0_175:                              ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_load_u16_d16_hi v45, v77 offset:8330
	v_or_b32_e32 v46, 5, v76
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s39, v46
	s_cbranch_execz .LBB0_177
; %bb.176:                              ;   in Loop: Header=BB0_156 Depth=3
	ds_load_b32 v46, v75 offset:28820
	s_wait_dscnt 0x0
	v_sub_f32_e32 v46, v72, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v80, 0x3fb8aa3b, v46
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v46
	v_fma_f32 v82, 0x3fb8aa3b, v46, -v80
	v_rndne_f32_e32 v83, v80
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v82, 0x32a5705f, v46
	v_sub_f32_e32 v80, v80, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v80, v80, v82
	v_cvt_i32_f32_e32 v82, v83
	v_exp_f32_e32 v80, v80
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v80, v80, v82
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v80, 0, v80, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v46
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v80, 0x7f800000, v80, vcc_lo
.LBB0_177:                              ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_load_u16_d16 v46, v77 offset:8332
	v_or_b32_e32 v82, 6, v76
	v_mov_b32_e32 v83, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s39, v82
	v_mov_b32_e32 v82, 0
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_179
; %bb.178:                              ;   in Loop: Header=BB0_156 Depth=3
	ds_load_b32 v83, v75 offset:28824
	s_wait_dscnt 0x0
	v_sub_f32_e32 v83, v72, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v84, 0x3fb8aa3b, v83
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v83
	v_fma_f32 v85, 0x3fb8aa3b, v83, -v84
	v_rndne_f32_e32 v86, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v85, 0x32a5705f, v83 :: v_dual_sub_f32 v84, v84, v86
	v_add_f32_e32 v84, v84, v85
	v_cvt_i32_f32_e32 v85, v86
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v84, v84
	v_ldexp_f32 v84, v84, v85
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v84, 0, v84, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v83, 0x7f800000, v84, vcc_lo
.LBB0_179:                              ;   in Loop: Header=BB0_156 Depth=3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	ds_load_u16_d16_hi v46, v77 offset:8334
	v_or_b32_e32 v76, 7, v76
	s_mov_b32 s6, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s39, v76
	s_cbranch_execz .LBB0_155
; %bb.180:                              ;   in Loop: Header=BB0_156 Depth=3
	ds_load_b32 v75, v75 offset:28828
	s_wait_dscnt 0x0
	v_sub_f32_e32 v75, v72, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v76, 0x3fb8aa3b, v75
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v75
	v_fma_f32 v77, 0x3fb8aa3b, v75, -v76
	v_rndne_f32_e32 v82, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v77, 0x32a5705f, v75 :: v_dual_sub_f32 v76, v76, v82
	v_add_f32_e32 v76, v76, v77
	v_cvt_i32_f32_e32 v77, v82
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v76, v76
	v_ldexp_f32 v76, v76, v77
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v76, 0, v76, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v75
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v82, 0x7f800000, v76, vcc_lo
	s_branch .LBB0_155
.LBB0_181:                              ;   in Loop: Header=BB0_147 Depth=2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v39.h, v64
	v_cvt_f16_f32_e32 v39.l, v63
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v40.h, v66
	v_cvt_f16_f32_e32 v40.l, v65
	v_cvt_f16_f32_e32 v41.h, v68
	v_cvt_f16_f32_e32 v41.l, v67
	v_cvt_f16_f32_e32 v42.h, v70
	v_cvt_f16_f32_e32 v42.l, v69
	s_and_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB0_192
; %bb.182:                              ; %NodeBlock2347
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_cmp_lt_i32 s3, 2
	s_mov_b32 s4, -1
                                        ; implicit-def: $vgpr75_vgpr76_vgpr77_vgpr78_vgpr79_vgpr80_vgpr81_vgpr82_vgpr83_vgpr84_vgpr85_vgpr86_vgpr87_vgpr88_vgpr89_vgpr90
	s_cbranch_scc1 .LBB0_188
; %bb.183:                              ; %LeafBlock2345
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_cmp_lg_u32 s3, 2
                                        ; implicit-def: $vgpr75_vgpr76_vgpr77_vgpr78_vgpr79_vgpr80_vgpr81_vgpr82_vgpr83_vgpr84_vgpr85_vgpr86_vgpr87_vgpr88_vgpr89_vgpr90
	s_cbranch_scc0 .LBB0_185
; %bb.184:                              ;   in Loop: Header=BB0_147 Depth=2
	v_dual_mov_b32 v75, v47 :: v_dual_mov_b32 v76, v48
	v_dual_mov_b32 v77, v49 :: v_dual_mov_b32 v78, v50
	v_dual_mov_b32 v79, v51 :: v_dual_mov_b32 v80, v52
	v_dual_mov_b32 v81, v53 :: v_dual_mov_b32 v82, v54
	v_dual_mov_b32 v83, v55 :: v_dual_mov_b32 v84, v56
	v_dual_mov_b32 v85, v57 :: v_dual_mov_b32 v86, v58
	v_dual_mov_b32 v87, v39 :: v_dual_mov_b32 v88, v40
	v_dual_mov_b32 v89, v41 :: v_dual_mov_b32 v90, v42
	s_mov_b32 s4, 0
.LBB0_185:                              ; %Flow2352
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_187
; %bb.186:                              ;   in Loop: Header=BB0_147 Depth=2
	v_dual_mov_b32 v75, v47 :: v_dual_mov_b32 v76, v48
	v_dual_mov_b32 v77, v49 :: v_dual_mov_b32 v78, v50
	v_dual_mov_b32 v79, v51 :: v_dual_mov_b32 v80, v52
	v_dual_mov_b32 v81, v53 :: v_dual_mov_b32 v82, v54
	v_dual_mov_b32 v83, v39 :: v_dual_mov_b32 v84, v40
	v_dual_mov_b32 v85, v41 :: v_dual_mov_b32 v86, v42
	v_dual_mov_b32 v87, v59 :: v_dual_mov_b32 v88, v60
	v_dual_mov_b32 v89, v61 :: v_dual_mov_b32 v90, v62
.LBB0_187:                              ; %Flow2353
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_mov_b32 s4, 0
.LBB0_188:                              ; %Flow2354
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_190
; %bb.189:                              ;   in Loop: Header=BB0_147 Depth=2
	v_dual_mov_b32 v37, v49 :: v_dual_mov_b32 v38, v50
	v_dual_mov_b32 v49, v61 :: v_dual_mov_b32 v50, v62
	v_dual_mov_b32 v35, v47 :: v_dual_mov_b32 v36, v48
	v_dual_mov_b32 v43, v55 :: v_dual_mov_b32 v44, v56
	v_dual_mov_b32 v45, v57 :: v_dual_mov_b32 v46, v58
	v_dual_mov_b32 v47, v59 :: v_dual_mov_b32 v48, v60
	v_dual_mov_b32 v90, v50 :: v_dual_mov_b32 v89, v49
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v86, v46 :: v_dual_mov_b32 v85, v45
	v_dual_mov_b32 v88, v48 :: v_dual_mov_b32 v87, v47
	v_dual_mov_b32 v84, v44 :: v_dual_mov_b32 v83, v43
	v_dual_mov_b32 v82, v42 :: v_dual_mov_b32 v81, v41
	v_dual_mov_b32 v80, v40 :: v_dual_mov_b32 v79, v39
	v_dual_mov_b32 v78, v38 :: v_dual_mov_b32 v77, v37
	v_dual_mov_b32 v76, v36 :: v_dual_mov_b32 v75, v35
.LBB0_190:                              ; %Flow2355
                                        ;   in Loop: Header=BB0_147 Depth=2
	s_branch .LBB0_146
.LBB0_191:                              ;   in Loop: Header=BB0_147 Depth=2
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr35
                                        ; implicit-def: $vgpr36
                                        ; implicit-def: $vgpr37
	s_branch .LBB0_153
.LBB0_192:                              ;   in Loop: Header=BB0_147 Depth=2
                                        ; implicit-def: $vgpr75_vgpr76_vgpr77_vgpr78_vgpr79_vgpr80_vgpr81_vgpr82_vgpr83_vgpr84_vgpr85_vgpr86_vgpr87_vgpr88_vgpr89_vgpr90
	s_cbranch_execz .LBB0_146
	s_branch .LBB0_145
.LBB0_193:                              ; %._crit_edge
	s_load_b64 s[2:3], s[0:1], 0x50
	v_cvt_f32_f16_e32 v24, v75.l
	v_lshlrev_b32_e32 v4, 1, v102
	v_or_b32_e32 v34, v104, v103
	v_cvt_f32_f16_e32 v17, v78.h
	v_cvt_f32_f16_e32 v3, v77.h
	v_mov_b32_e32 v35, v24
	v_cvt_f32_f16_e32 v2, v76.h
	v_cvt_f32_f16_e32 v22, v75.h
	v_cvt_f32_f16_e32 v20, v78.l
	v_cvt_f32_f16_e32 v19, v77.l
	v_cvt_f32_f16_e32 v23, v76.l
	v_lshlrev_b32_e32 v1, 1, v34
	s_wait_kmcnt 0x0
	s_cmp_lg_u64 s[2:3], 0
	s_cselect_b32 s0, -1, 0
	s_lshl_b64 s[4:5], s[36:37], 1
	s_lshl_b64 s[6:7], s[34:35], 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[2:3], s[4:5]
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[2:3], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v38, s1, s2, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v39, null, s3, 0, s1
	s_cbranch_vccz .LBB0_195
; %bb.194:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v35, v4, 1.0, v24 op_sel_hi:[1,1,0]
.LBB0_195:
	v_cndmask_b32_e64 v37, 0, 1, s0
	v_mov_b32_e32 v36, v22
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_197
; %bb.196:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v36, v4, 1.0, v22 op_sel_hi:[1,1,0]
.LBB0_197:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v40, v23
	s_cbranch_vccnz .LBB0_199
; %bb.198:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v40, v4, 1.0, v23 op_sel_hi:[1,1,0]
.LBB0_199:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v41, v2
	s_cbranch_vccnz .LBB0_201
; %bb.200:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v41, v4, 1.0, v2 op_sel_hi:[1,1,0]
.LBB0_201:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v42, v19
	s_cbranch_vccnz .LBB0_203
; %bb.202:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:8
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v42, v4, 1.0, v19 op_sel_hi:[1,1,0]
.LBB0_203:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v43, v3
	s_cbranch_vccnz .LBB0_205
; %bb.204:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:10
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v43, v4, 1.0, v3 op_sel_hi:[1,1,0]
.LBB0_205:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v44, v20
	s_cbranch_vccnz .LBB0_207
; %bb.206:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:12
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v44, v4, 1.0, v20 op_sel_hi:[1,1,0]
.LBB0_207:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v45, v17
	s_cbranch_vccnz .LBB0_209
; %bb.208:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:14
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v45, v4, 1.0, v17 op_sel_hi:[1,1,0]
.LBB0_209:
	v_cvt_f32_f16_e32 v33, v79.l
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_cvt_f32_f16_e32 v21, v80.h
	v_cvt_f32_f16_e32 v28, v79.h
	v_cvt_f32_f16_e32 v14, v82.h
	v_cvt_f32_f16_e32 v13, v81.h
	v_cvt_f32_f16_e32 v18, v81.l
	v_cvt_f32_f16_e32 v31, v80.l
	v_cvt_f32_f16_e32 v15, v82.l
	v_mov_b32_e32 v46, v33
	s_cbranch_vccnz .LBB0_211
; %bb.210:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:32
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v46, v4, 1.0, v33 op_sel_hi:[1,1,0]
.LBB0_211:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v47, v28
	s_cbranch_vccnz .LBB0_213
; %bb.212:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:34
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v47, v4, 1.0, v28 op_sel_hi:[1,1,0]
.LBB0_213:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v48, v31
	s_cbranch_vccnz .LBB0_215
; %bb.214:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:36
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v48, v4, 1.0, v31 op_sel_hi:[1,1,0]
.LBB0_215:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v49, v21
	s_cbranch_vccnz .LBB0_217
; %bb.216:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:38
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v49, v4, 1.0, v21 op_sel_hi:[1,1,0]
.LBB0_217:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v50, v18
	s_cbranch_vccnz .LBB0_219
; %bb.218:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:40
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v50, v4, 1.0, v18 op_sel_hi:[1,1,0]
.LBB0_219:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v51, v13
	s_cbranch_vccnz .LBB0_221
; %bb.220:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:42
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v51, v4, 1.0, v13 op_sel_hi:[1,1,0]
.LBB0_221:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v52, v15
	s_cbranch_vccnz .LBB0_223
; %bb.222:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:44
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v52, v4, 1.0, v15 op_sel_hi:[1,1,0]
.LBB0_223:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v53, v14
	s_cbranch_vccnz .LBB0_225
; %bb.224:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:46
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v53, v4, 1.0, v14 op_sel_hi:[1,1,0]
.LBB0_225:
	v_cvt_f32_f16_e32 v32, v83.l
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_cvt_f32_f16_e32 v9, v84.h
	v_cvt_f32_f16_e32 v29, v83.h
	v_cvt_f32_f16_e32 v10, v86.h
	v_cvt_f32_f16_e32 v6, v85.h
	v_cvt_f32_f16_e32 v12, v85.l
	v_cvt_f32_f16_e32 v30, v84.l
	v_cvt_f32_f16_e32 v16, v86.l
	v_mov_b32_e32 v54, v32
	s_cbranch_vccnz .LBB0_227
; %bb.226:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:64
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v54, v4, 1.0, v32 op_sel_hi:[1,1,0]
.LBB0_227:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v55, v29
	s_cbranch_vccnz .LBB0_229
; %bb.228:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:66
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v55, v4, 1.0, v29 op_sel_hi:[1,1,0]
.LBB0_229:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v56, v30
	s_cbranch_vccnz .LBB0_231
; %bb.230:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:68
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v56, v4, 1.0, v30 op_sel_hi:[1,1,0]
.LBB0_231:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v57, v9
	s_cbranch_vccnz .LBB0_233
; %bb.232:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:70
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v57, v4, 1.0, v9 op_sel_hi:[1,1,0]
.LBB0_233:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v58, v12
	s_cbranch_vccnz .LBB0_235
; %bb.234:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:72
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v58, v4, 1.0, v12 op_sel_hi:[1,1,0]
.LBB0_235:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v59, v6
	s_cbranch_vccnz .LBB0_237
; %bb.236:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:74
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v59, v4, 1.0, v6 op_sel_hi:[1,1,0]
.LBB0_237:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v60, v16
	s_cbranch_vccnz .LBB0_239
; %bb.238:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:76
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v60, v4, 1.0, v16 op_sel_hi:[1,1,0]
.LBB0_239:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v61, v10
	s_cbranch_vccnz .LBB0_241
; %bb.240:
	v_add_co_u32 v4, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v39, vcc_lo
	global_load_d16_b16 v4, v[4:5], off offset:78
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v61, v4, 1.0, v10 op_sel_hi:[1,1,0]
.LBB0_241:
	v_cvt_f32_f16_e32 v26, v87.l
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_cvt_f32_f16_e32 v4, v88.h
	v_cvt_f32_f16_e32 v25, v87.h
	v_cvt_f32_f16_e32 v7, v90.h
	v_cvt_f32_f16_e32 v5, v89.h
	v_cvt_f32_f16_e32 v8, v89.l
	v_cvt_f32_f16_e32 v27, v88.l
	v_cvt_f32_f16_e32 v11, v90.l
	v_mov_b32_e32 v62, v26
	s_cbranch_vccnz .LBB0_243
; %bb.242:
	v_add_co_u32 v62, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, 0, v39, vcc_lo
	global_load_d16_b16 v62, v[62:63], off offset:96
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v62, v62, 1.0, v26 op_sel_hi:[1,1,0]
.LBB0_243:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v63, v25
	s_cbranch_vccnz .LBB0_245
; %bb.244:
	v_add_co_u32 v63, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v64, null, 0, v39, vcc_lo
	global_load_d16_b16 v63, v[63:64], off offset:98
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v63, v63, 1.0, v25 op_sel_hi:[1,1,0]
.LBB0_245:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v64, v27
	s_cbranch_vccnz .LBB0_247
; %bb.246:
	v_add_co_u32 v64, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, 0, v39, vcc_lo
	global_load_d16_b16 v64, v[64:65], off offset:100
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v64, v64, 1.0, v27 op_sel_hi:[1,1,0]
.LBB0_247:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v65, v4
	s_cbranch_vccnz .LBB0_249
; %bb.248:
	v_add_co_u32 v65, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, 0, v39, vcc_lo
	global_load_d16_b16 v65, v[65:66], off offset:102
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v65, v65, 1.0, v4 op_sel_hi:[1,1,0]
.LBB0_249:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v66, v8
	s_cbranch_vccnz .LBB0_251
; %bb.250:
	v_add_co_u32 v66, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, 0, v39, vcc_lo
	global_load_d16_b16 v66, v[66:67], off offset:104
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v66, v66, 1.0, v8 op_sel_hi:[1,1,0]
.LBB0_251:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v67, v5
	s_cbranch_vccnz .LBB0_253
; %bb.252:
	v_add_co_u32 v67, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, 0, v39, vcc_lo
	global_load_d16_b16 v67, v[67:68], off offset:106
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v67, v67, 1.0, v5 op_sel_hi:[1,1,0]
.LBB0_253:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v68, v11
	s_cbranch_vccnz .LBB0_255
; %bb.254:
	v_add_co_u32 v68, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v69, null, 0, v39, vcc_lo
	global_load_d16_b16 v68, v[68:69], off offset:108
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v68, v68, 1.0, v11 op_sel_hi:[1,1,0]
.LBB0_255:
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_mov_b32_e32 v69, v7
	s_cbranch_vccnz .LBB0_257
; %bb.256:
	v_add_co_u32 v69, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, 0, v39, vcc_lo
	global_load_d16_b16 v69, v[69:70], off offset:110
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v69, v69, 1.0, v7 op_sel_hi:[1,1,0]
.LBB0_257:
	v_max3_num_f32 v35, |v35|, 0, |v36|
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_max3_num_f32 v35, v35, |v40|, |v41|
	v_lshlrev_b32_e32 v40, 2, v0
	v_and_b32_e32 v0, 0x3cf, v0
	v_max3_num_f32 v35, v35, |v42|, |v43|
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v0, 2, v0
	v_max3_num_f32 v35, v35, |v44|, |v45|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v35, v35, |v46|, |v47|
	v_max3_num_f32 v35, v35, |v48|, |v49|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v35, v35, |v50|, |v51|
	v_max3_num_f32 v35, v35, |v52|, |v53|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v35, v35, |v54|, |v55|
	v_max3_num_f32 v35, v35, |v56|, |v57|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v35, v35, |v58|, |v59|
	v_max3_num_f32 v35, v35, |v60|, |v61|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v35, v35, |v62|, |v63|
	v_max3_num_f32 v35, v35, |v64|, |v65|
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v35, v35, |v66|, |v67|
	v_max3_num_f32 v35, v35, |v68|, |v69|
	ds_store_b32 v40, v35
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_b32 v[35:36], v0 offset1:16
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v35, v35, v35
	v_or_b32_e32 v40, 0xc0, v40
	ds_load_b32 v0, v0 offset:128
	ds_load_b32 v40, v40
	v_max_num_f32_e32 v36, v36, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v35, v35, v36
	s_wait_dscnt 0x0
	v_max3_num_f32 v35, v35, v0, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_scale_f32 v0, null, v35, v35, 0x42fe0000
	v_div_scale_f32 v41, vcc_lo, 0x42fe0000, v35, 0x42fe0000
	v_cmp_lt_f32_e64 s0, 0, v35
	v_rcp_f32_e32 v36, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v40, -v0, v36, 1.0
	v_fmac_f32_e32 v36, v40, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v40, v41, v36
	v_fma_f32 v42, -v0, v40, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v40, v42, v36
	v_fma_f32 v0, -v0, v40, v41
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v0, v0, v36, v40
	v_or_b32_e32 v36, v100, v99
	v_cmp_ne_u32_e32 vcc_lo, 1, v37
	v_div_fixup_f32 v0, v0, v35, 0x42fe0000
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e64 s1, 0, v36
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v37, 0, v0, s0
	s_cbranch_vccnz .LBB0_262
; %bb.258:
	v_add_co_u32 v0, vcc_lo, v38, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v39, vcc_lo
	v_div_scale_f32 v38, null, 0x42fe0000, 0x42fe0000, v35
	v_div_scale_f32 v57, vcc_lo, v35, 0x42fe0000, v35
	s_mov_b32 s4, 0xc3000000
	s_clause 0x3
	global_load_b128 v[39:42], v[0:1], off
	global_load_b128 v[43:46], v[0:1], off offset:32
	global_load_b128 v[47:50], v[0:1], off offset:64
	global_load_b128 v[51:54], v[0:1], off offset:96
	v_rcp_f32_e32 v55, v38
	s_and_b32 s2, s1, exec_lo
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v56, -v38, v55, 1.0
	v_fmac_f32_e32 v55, v56, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v56, v57, v55
	v_fma_f32 v58, -v38, v56, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v56, v58, v55
	v_fma_f32 v38, -v38, v56, v57
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v38, v38, v55, v56
	v_add_co_u32 v63, vcc_lo, v95, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v64, null, 0, v96, vcc_lo
	v_div_fixup_f32 v38, v38, 0x42fe0000, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v38, 1.0, v38, s0
	s_wait_loadcnt 0x3
	v_fma_mix_f32 v55, v39, 1.0, v24 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v56, v39, 1.0, v22 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v57, v40, 1.0, v23 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v58, v40, 1.0, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v59, v41, 1.0, v19 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v60, v41, 1.0, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v61, v42, 1.0, v20 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v62, v42, 1.0, v17 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_wait_loadcnt 0x2
	v_fma_mix_f32 v65, v43, 1.0, v33 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v66, v43, 1.0, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v67, v44, 1.0, v31 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v68, v44, 1.0, v21 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v69, v45, 1.0, v18 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v70, v45, 1.0, v13 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v71, v46, 1.0, v15 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v72, v46, 1.0, v14 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_wait_loadcnt 0x1
	v_fma_mix_f32 v73, v47, 1.0, v32 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v74, v47, 1.0, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v75, v48, 1.0, v30 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v76, v48, 1.0, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v77, v49, 1.0, v12 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v78, v49, 1.0, v6 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v79, v50, 1.0, v16 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v80, v50, 1.0, v10 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v81, v51, 1.0, v26 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v82, v51, 1.0, v25 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v83, v52, 1.0, v27 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v84, v52, 1.0, v4 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v85, v53, 1.0, v8 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v86, v53, 1.0, v5 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v87, v54, 1.0, v11 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v88, v54, 1.0, v7 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_dual_mul_f32 v39, v37, v55 :: v_dual_mul_f32 v40, v37, v56
	v_dual_mul_f32 v41, v37, v57 :: v_dual_mul_f32 v42, v37, v58
	v_dual_mul_f32 v43, v37, v59 :: v_dual_mul_f32 v44, v37, v60
	v_dual_mul_f32 v45, v37, v61 :: v_dual_mul_f32 v46, v37, v62
	v_dual_mul_f32 v47, v37, v65 :: v_dual_mul_f32 v48, v37, v66
	v_dual_mul_f32 v49, v37, v67 :: v_dual_mul_f32 v50, v37, v68
	v_dual_mul_f32 v51, v37, v69 :: v_dual_mul_f32 v52, v37, v70
	v_dual_mul_f32 v53, v37, v71 :: v_dual_mul_f32 v54, v37, v72
	v_dual_mul_f32 v89, v37, v73 :: v_dual_mul_f32 v90, v37, v74
	v_dual_mul_f32 v93, v37, v75 :: v_dual_mul_f32 v94, v37, v76
	v_dual_mul_f32 v99, v37, v77 :: v_dual_mul_f32 v100, v37, v78
	v_dual_mul_f32 v101, v37, v79 :: v_dual_mul_f32 v102, v37, v80
	v_dual_mul_f32 v103, v37, v81 :: v_dual_mul_f32 v104, v37, v82
	v_dual_mul_f32 v105, v37, v83 :: v_dual_mul_f32 v106, v37, v84
	v_dual_mul_f32 v107, v37, v85 :: v_dual_mul_f32 v108, v37, v86
	v_dual_mul_f32 v109, v37, v87 :: v_dual_mul_f32 v110, v37, v88
	v_rndne_f32_e32 v39, v39
	v_rndne_f32_e32 v40, v40
	v_rndne_f32_e32 v41, v41
	v_rndne_f32_e32 v42, v42
	v_rndne_f32_e32 v43, v43
	v_rndne_f32_e32 v44, v44
	v_rndne_f32_e32 v45, v45
	v_rndne_f32_e32 v46, v46
	v_rndne_f32_e32 v47, v47
	v_rndne_f32_e32 v48, v48
	v_rndne_f32_e32 v49, v49
	v_rndne_f32_e32 v50, v50
	v_rndne_f32_e32 v51, v51
	v_rndne_f32_e32 v52, v52
	v_rndne_f32_e32 v53, v53
	v_rndne_f32_e32 v54, v54
	v_rndne_f32_e32 v89, v89
	v_rndne_f32_e32 v90, v90
	v_rndne_f32_e32 v93, v93
	v_rndne_f32_e32 v94, v94
	v_rndne_f32_e32 v99, v99
	v_rndne_f32_e32 v100, v100
	v_rndne_f32_e32 v101, v101
	v_rndne_f32_e32 v102, v102
	v_rndne_f32_e32 v103, v103
	v_rndne_f32_e32 v104, v104
	v_rndne_f32_e32 v105, v105
	v_rndne_f32_e32 v106, v106
	v_rndne_f32_e32 v107, v107
	v_rndne_f32_e32 v108, v108
	v_rndne_f32_e32 v109, v109
	v_rndne_f32_e32 v110, v110
	s_wait_alu depctr_sa_sdst(0)
	v_med3_num_f32 v111, v39, s4, 0x42fe0000
	v_med3_num_f32 v112, v40, s4, 0x42fe0000
	v_med3_num_f32 v41, v41, s4, 0x42fe0000
	v_med3_num_f32 v42, v42, s4, 0x42fe0000
	v_med3_num_f32 v43, v43, s4, 0x42fe0000
	v_med3_num_f32 v44, v44, s4, 0x42fe0000
	v_med3_num_f32 v45, v45, s4, 0x42fe0000
	v_med3_num_f32 v46, v46, s4, 0x42fe0000
	v_med3_num_f32 v47, v47, s4, 0x42fe0000
	v_med3_num_f32 v48, v48, s4, 0x42fe0000
	v_med3_num_f32 v49, v49, s4, 0x42fe0000
	v_med3_num_f32 v50, v50, s4, 0x42fe0000
	v_med3_num_f32 v51, v51, s4, 0x42fe0000
	v_med3_num_f32 v52, v52, s4, 0x42fe0000
	v_med3_num_f32 v53, v53, s4, 0x42fe0000
	v_med3_num_f32 v54, v54, s4, 0x42fe0000
	v_med3_num_f32 v89, v89, s4, 0x42fe0000
	v_med3_num_f32 v90, v90, s4, 0x42fe0000
	v_med3_num_f32 v93, v93, s4, 0x42fe0000
	v_med3_num_f32 v94, v94, s4, 0x42fe0000
	v_med3_num_f32 v99, v99, s4, 0x42fe0000
	v_med3_num_f32 v100, v100, s4, 0x42fe0000
	v_med3_num_f32 v101, v101, s4, 0x42fe0000
	v_med3_num_f32 v102, v102, s4, 0x42fe0000
	v_med3_num_f32 v103, v103, s4, 0x42fe0000
	v_med3_num_f32 v104, v104, s4, 0x42fe0000
	v_med3_num_f32 v105, v105, s4, 0x42fe0000
	v_med3_num_f32 v106, v106, s4, 0x42fe0000
	v_med3_num_f32 v107, v107, s4, 0x42fe0000
	v_med3_num_f32 v108, v108, s4, 0x42fe0000
	v_med3_num_f32 v109, v109, s4, 0x42fe0000
	v_med3_num_f32 v110, v110, s4, 0x42fe0000
	v_fma_mixlo_f16 v39, -v38, v111, v55
	v_fma_mixhi_f16 v39, -v38, v112, v56
	v_fma_mixlo_f16 v40, -v38, v41, v57
	v_fma_mixhi_f16 v40, -v38, v42, v58
	v_cvt_i32_f32_e32 v57, v42
	v_cvt_i32_f32_e32 v58, v41
	v_cvt_i32_f32_e32 v112, v112
	v_cvt_i32_f32_e32 v111, v111
	v_fma_mixlo_f16 v41, -v38, v43, v59
	v_fma_mixhi_f16 v41, -v38, v44, v60
	v_cvt_i32_f32_e32 v55, v46
	v_cvt_i32_f32_e32 v56, v45
	v_cvt_i32_f32_e32 v59, v44
	v_cvt_i32_f32_e32 v60, v43
	v_fma_mixlo_f16 v42, -v38, v45, v61
	v_fma_mixhi_f16 v42, -v38, v46, v62
	v_fma_mixlo_f16 v43, -v38, v47, v65
	v_fma_mixhi_f16 v43, -v38, v48, v66
	v_fma_mixlo_f16 v44, -v38, v49, v67
	v_fma_mixhi_f16 v44, -v38, v50, v68
	v_cvt_i32_f32_e32 v61, v50
	v_cvt_i32_f32_e32 v62, v49
	v_cvt_i32_f32_e32 v65, v48
	v_cvt_i32_f32_e32 v66, v47
	v_fma_mixlo_f16 v45, -v38, v51, v69
	v_fma_mixhi_f16 v45, -v38, v52, v70
	v_cvt_i32_f32_e32 v67, v54
	v_cvt_i32_f32_e32 v68, v53
	v_cvt_i32_f32_e32 v69, v52
	v_cvt_i32_f32_e32 v70, v51
	v_fma_mixlo_f16 v46, -v38, v53, v71
	v_fma_mixhi_f16 v46, -v38, v54, v72
	v_fma_mixlo_f16 v47, -v38, v89, v73
	v_fma_mixhi_f16 v47, -v38, v90, v74
	v_fma_mixlo_f16 v48, -v38, v93, v75
	v_fma_mixhi_f16 v48, -v38, v94, v76
	v_cvt_i32_f32_e32 v71, v94
	v_cvt_i32_f32_e32 v72, v93
	v_cvt_i32_f32_e32 v73, v90
	v_cvt_i32_f32_e32 v74, v89
	v_fma_mixlo_f16 v49, -v38, v99, v77
	v_fma_mixhi_f16 v49, -v38, v100, v78
	v_cvt_i32_f32_e32 v75, v102
	v_cvt_i32_f32_e32 v76, v101
	v_cvt_i32_f32_e32 v77, v100
	v_cvt_i32_f32_e32 v78, v99
	v_fma_mixlo_f16 v50, -v38, v101, v79
	v_fma_mixhi_f16 v50, -v38, v102, v80
	v_fma_mixlo_f16 v51, -v38, v103, v81
	v_fma_mixhi_f16 v51, -v38, v104, v82
	v_fma_mixlo_f16 v52, -v38, v105, v83
	v_fma_mixhi_f16 v52, -v38, v106, v84
	v_cvt_i32_f32_e32 v79, v106
	v_cvt_i32_f32_e32 v80, v105
	v_cvt_i32_f32_e32 v81, v104
	v_cvt_i32_f32_e32 v82, v103
	v_fma_mixlo_f16 v53, -v38, v107, v85
	v_fma_mixhi_f16 v53, -v38, v108, v86
	v_cvt_i32_f32_e32 v83, v110
	v_cvt_i32_f32_e32 v84, v109
	v_cvt_i32_f32_e32 v85, v108
	v_cvt_i32_f32_e32 v86, v107
	v_lshlrev_b16 v55.l, 8, v55.l
	v_and_b16 v55.h, 0xff, v56.l
	v_lshlrev_b16 v56.l, 8, v59.l
	v_and_b16 v56.h, 0xff, v60.l
	v_lshlrev_b16 v57.l, 8, v57.l
	v_and_b16 v57.h, 0xff, v58.l
	v_lshlrev_b16 v58.l, 8, v112.l
	v_and_b16 v58.h, 0xff, v111.l
	s_clause 0x1
	global_store_b128 v[0:1], v[39:42], off
	global_store_b128 v[0:1], v[43:46], off offset:32
	v_lshlrev_b16 v39.l, 8, v67.l
	v_and_b16 v39.h, 0xff, v68.l
	v_lshlrev_b16 v40.l, 8, v69.l
	v_and_b16 v40.h, 0xff, v70.l
	v_lshlrev_b16 v41.l, 8, v61.l
	v_and_b16 v41.h, 0xff, v62.l
	v_lshlrev_b16 v42.l, 8, v65.l
	v_and_b16 v42.h, 0xff, v66.l
	v_lshlrev_b16 v43.l, 8, v75.l
	v_and_b16 v43.h, 0xff, v76.l
	v_lshlrev_b16 v44.l, 8, v77.l
	v_and_b16 v44.h, 0xff, v78.l
	v_lshlrev_b16 v45.l, 8, v71.l
	v_and_b16 v45.h, 0xff, v72.l
	v_lshlrev_b16 v46.l, 8, v73.l
	v_and_b16 v46.h, 0xff, v74.l
	v_lshlrev_b16 v59.l, 8, v83.l
	v_and_b16 v59.h, 0xff, v84.l
	v_lshlrev_b16 v60.l, 8, v85.l
	v_and_b16 v60.h, 0xff, v86.l
	v_lshlrev_b16 v61.l, 8, v79.l
	v_and_b16 v61.h, 0xff, v80.l
	v_lshlrev_b16 v62.l, 8, v81.l
	v_and_b16 v62.h, 0xff, v82.l
	v_fma_mixlo_f16 v54, -v38, v109, v87
	v_fma_mixhi_f16 v54, -v38, v110, v88
	v_or_b16 v66.h, v55.h, v55.l
	v_or_b16 v66.l, v56.h, v56.l
	v_or_b16 v65.h, v57.h, v57.l
	v_or_b16 v65.l, v58.h, v58.l
	v_or_b16 v56.h, v39.h, v39.l
	v_or_b16 v56.l, v40.h, v40.l
	v_or_b16 v55.h, v41.h, v41.l
	v_or_b16 v55.l, v42.h, v42.l
	v_or_b16 v40.h, v43.h, v43.l
	v_or_b16 v40.l, v44.h, v44.l
	v_or_b16 v39.h, v45.h, v45.l
	v_or_b16 v39.l, v46.h, v46.l
	v_or_b16 v42.h, v59.h, v59.l
	v_or_b16 v42.l, v60.h, v60.l
	v_or_b16 v41.h, v61.h, v61.l
	v_or_b16 v41.l, v62.h, v62.l
	s_clause 0x1
	global_store_b128 v[0:1], v[47:50], off offset:64
	global_store_b128 v[0:1], v[51:54], off offset:96
	s_clause 0x3
	global_store_b64 v[63:64], v[65:66], off
	global_store_b64 v[63:64], v[55:56], off offset:16
	global_store_b64 v[63:64], v[39:40], off offset:32
	global_store_b64 v[63:64], v[41:42], off offset:48
	s_cbranch_execnz .LBB0_263
.LBB0_259:
	v_mul_lo_u32 v0, 0x3c6ef35f, v97
	s_mul_i32 s1, ttmp9, 0x19660d
	s_mov_b32 s3, 0x19660d
	s_wait_alu depctr_sa_sdst(0)
	v_mad_u32_u24 v1, 0x343fd, v98, s1
	s_mul_i32 s1, s31, 0x269ec3
	s_mov_b32 s4, 0x17385ca9
	s_mov_b32 s5, 0xaf490a95
	s_mov_b32 s6, 0x6e587165
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v38, v1, v0, s1
	v_add_co_u32 v0, vcc_lo, v95, v34
	s_mov_b32 s7, 0xea890021
	s_mov_b32 s1, 0xc3000000
	v_mad_co_u64_u32 v[39:40], null, v38, s3, 0x3c6ef35f
	v_mad_co_u64_u32 v[40:41], null, v38, s4, 0x47502932
	v_mad_co_u64_u32 v[41:42], null, v38, s5, 0xffffffffd1ccf6e9
	s_mov_b32 s3, 0x979e791
	s_mov_b32 s5, 0xbf69fab9
	s_mov_b32 s4, 0xaa9d885d
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v96, vcc_lo
	v_lshrrev_b32_e32 v34, 8, v39
	v_lshrrev_b32_e32 v39, 8, v40
	v_lshrrev_b32_e32 v40, 8, v41
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_u32_e32 v34, v34
	v_cvt_f32_u32_e32 v42, v39
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_u32_e32 v43, v40
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[39:40], null, v38, s3, 0xffffffffaaf95334
	v_mad_co_u64_u32 v[40:41], null, v38, s5, 0xffffffff9f2ec686
	v_mul_f32_e32 v34, 0x33800000, v34
	s_mov_b32 s3, 0x823b27ad
	v_dual_mul_f32 v45, 0x33800000, v43 :: v_dual_mul_f32 v44, 0x33800000, v42
	v_mad_co_u64_u32 v[41:42], null, v38, s6, 0x57fe6c2d
	v_mad_co_u64_u32 v[42:43], null, v38, s7, 0xffffffffa3d95fa8
	v_lshrrev_b32_e32 v39, 8, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v45, v37, v23 :: v_dual_fmac_f32 v44, v37, v22
	v_mad_co_u64_u32 v[22:23], null, v38, s4, 0x6252e503
	v_lshrrev_b32_e32 v23, 8, v40
	v_cvt_f32_u32_e32 v39, v39
	v_lshrrev_b32_e32 v41, 8, v41
	v_lshrrev_b32_e32 v42, 8, v42
	v_floor_f32_e32 v40, v45
	v_cvt_f32_u32_e32 v23, v23
	s_mov_b32 s4, 0xf3aa51d9
	v_lshrrev_b32_e32 v22, 8, v22
	v_cvt_f32_u32_e32 v42, v42
	v_cvt_f32_u32_e32 v41, v41
	v_dual_fmac_f32 v34, v37, v24 :: v_dual_mul_f32 v39, 0x33800000, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_u32_e32 v22, v22
	v_dual_mul_f32 v42, 0x33800000, v42 :: v_dual_mul_f32 v41, 0x33800000, v41
	v_mul_f32_e32 v23, 0x33800000, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_floor_f32_e32 v24, v34
	v_mul_f32_e32 v22, 0x33800000, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v42, v37, v17 :: v_dual_fmac_f32 v41, v37, v20
	v_fmac_f32_e32 v39, v37, v2
	v_fmac_f32_e32 v23, v37, v3
	v_fmac_f32_e32 v22, v37, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_floor_f32_e32 v2, v42
	v_floor_f32_e32 v3, v41
	v_floor_f32_e32 v20, v39
	v_floor_f32_e32 v17, v23
	v_floor_f32_e32 v19, v22
	v_med3_num_f32 v2, v2, s1, 0x42fe0000
	v_med3_num_f32 v3, v3, s1, 0x42fe0000
	v_med3_num_f32 v20, v20, s1, 0x42fe0000
	v_med3_num_f32 v17, v17, s1, 0x42fe0000
	v_med3_num_f32 v19, v19, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v2, v2
	v_cvt_i32_f32_e32 v3, v3
	v_cvt_i32_f32_e32 v41, v20
	v_cvt_i32_f32_e32 v17, v17
	v_cvt_i32_f32_e32 v39, v19
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[19:20], null, v38, s3, 0xffffffff81fdbee7
	s_mov_b32 s3, 0xeb4f1c9
	v_and_b16 v2.h, 0xff, v3.l
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[22:23], null, v38, s3, 0xffffffff94f0af1a
	s_mov_b32 s3, 0x74275d35
	v_lshlrev_b16 v3.l, 8, v17.l
	v_and_b16 v17.l, 0xff, v39.l
	v_floor_f32_e32 v34, v44
	v_lshrrev_b32_e32 v23, 8, v19
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[19:20], null, v38, s3, 0xffffffffcbf633b1
	v_med3_num_f32 v24, v24, s1, 0x42fe0000
	v_lshrrev_b32_e32 v22, 8, v22
	v_or_b16 v3.l, v17.l, v3.l
	v_cvt_f32_u32_e32 v23, v23
	v_med3_num_f32 v34, v34, s1, 0x42fe0000
	v_med3_num_f32 v40, v40, s1, 0x42fe0000
	v_cvt_f32_u32_e32 v22, v22
	v_lshrrev_b32_e32 v17, 8, v19
	v_mul_f32_e32 v19, 0x33800000, v23
	v_cvt_i32_f32_e32 v20, v24
	v_cvt_i32_f32_e32 v40, v40
	v_mul_f32_e32 v22, 0x33800000, v22
	v_cvt_f32_u32_e32 v23, v17
	v_fmac_f32_e32 v19, v37, v33
	v_cvt_i32_f32_e32 v34, v34
	v_lshlrev_b16 v2.l, 8, v2.l
	s_mov_b32 s3, 0xaf4fd9b1
	v_mul_f32_e32 v23, 0x33800000, v23
	v_and_b16 v17.h, 0xff, v20.l
	v_fmac_f32_e32 v22, v37, v28
	v_floor_f32_e32 v24, v19
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[19:20], null, v38, s3, 0xffffffffbcd1195c
	v_or_b16 v3.h, v2.h, v2.l
	v_lshlrev_b16 v2.l, 8, v41.l
	v_and_b16 v2.h, 0xff, v40.l
	v_lshlrev_b16 v17.l, 8, v34.l
	v_fmac_f32_e32 v23, v37, v31
	v_med3_num_f32 v20, v24, s1, 0x42fe0000
	v_floor_f32_e32 v22, v22
	v_or_b16 v2.h, v2.h, v2.l
	v_or_b16 v2.l, v17.h, v17.l
	v_floor_f32_e32 v17, v23
	v_lshrrev_b32_e32 v23, 8, v19
	s_mov_b32 s3, 0xfa1393fd
	v_cvt_i32_f32_e32 v28, v20
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[19:20], null, v38, s3, 0xffffffff9d23e50b
	s_mov_b32 s3, 0x77520441
	v_med3_num_f32 v31, v22, s1, 0x42fe0000
	v_cvt_f32_u32_e32 v20, v23
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[22:23], null, v38, s3, 0xffffffff83c6b450
	v_med3_num_f32 v17, v17, s1, 0x42fe0000
	v_mad_co_u64_u32 v[23:24], null, v38, s4, 0xffffffffe296f6ee
	v_cvt_i32_f32_e32 v24, v31
	s_mov_b32 s3, 0x3a739e05
	v_lshrrev_b32_e32 v19, 8, v19
	v_cvt_i32_f32_e32 v31, v17
	v_mul_f32_e32 v17, 0x33800000, v20
	v_lshrrev_b32_e32 v20, 8, v22
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[33:34], null, v38, s3, 0x1ba5175
	v_cvt_f32_u32_e32 v19, v19
	v_lshrrev_b32_e32 v22, 8, v23
	s_mov_b32 s3, 0x27351d4d
	v_cvt_f32_u32_e32 v20, v20
	s_mov_b32 s4, 0xab945ea5
	v_mul_f32_e32 v19, 0x33800000, v19
	s_mov_b32 s6, 0x27b61881
	v_lshrrev_b32_e32 v23, 8, v33
	v_mul_f32_e32 v20, 0x33800000, v20
	s_mov_b32 s7, 0x966d3345
	v_fmac_f32_e32 v19, v37, v18
	s_mov_b32 s5, 0x9e082c19
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v20, v37, v14 :: v_dual_fmac_f32 v17, v37, v21
	v_cvt_f32_u32_e32 v21, v22
	v_cvt_f32_u32_e32 v22, v23
	v_floor_f32_e32 v14, v19
	v_floor_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v18, 0x33800000, v21 :: v_dual_mul_f32 v21, 0x33800000, v22
	v_med3_num_f32 v17, v17, s1, 0x42fe0000
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v18, v37, v13
	v_floor_f32_e32 v13, v20
	v_fmac_f32_e32 v21, v37, v15
	v_med3_num_f32 v15, v14, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v19, v17
	v_floor_f32_e32 v17, v18
	v_med3_num_f32 v20, v13, s1, 0x42fe0000
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[13:14], null, v38, s3, 0xffffffffb52dfb6f
	v_floor_f32_e32 v18, v21
	s_mov_b32 s3, 0x3e42ae9
	v_cvt_i32_f32_e32 v21, v15
	v_med3_num_f32 v17, v17, s1, 0x42fe0000
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[14:15], null, v38, s3, 0x4fc9f202
	v_med3_num_f32 v18, v18, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v20, v20
	v_lshrrev_b32_e32 v22, 8, v13
	s_mov_b32 s3, 0x4c7003d5
	v_cvt_i32_f32_e32 v15, v17
	v_cvt_i32_f32_e32 v23, v18
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[17:18], null, v38, s3, 0x624f0979
	v_cvt_f32_u32_e32 v18, v22
	v_lshlrev_b16 v13.l, 8, v20.l
	v_lshrrev_b32_e32 v20, 8, v14
	v_lshlrev_b16 v14.l, 8, v15.l
	v_and_b16 v14.h, 0xff, v21.l
	v_mul_f32_e32 v15, 0x33800000, v18
	v_and_b16 v13.h, 0xff, v23.l
	v_cvt_f32_u32_e32 v20, v20
	v_lshrrev_b32_e32 v17, 8, v17
	v_or_b16 v18.l, v14.h, v14.l
	v_fmac_f32_e32 v15, v37, v32
	v_or_b16 v18.h, v13.h, v13.l
	v_mul_f32_e32 v20, 0x33800000, v20
	v_cvt_f32_u32_e32 v14, v17
	v_lshlrev_b16 v13.l, 8, v19.l
	v_floor_f32_e32 v15, v15
	v_and_b16 v13.h, 0xff, v31.l
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v20, v37, v29 :: v_dual_mul_f32 v19, 0x33800000, v14
	v_lshlrev_b16 v14.l, 8, v24.l
	v_and_b16 v14.h, 0xff, v28.l
	v_med3_num_f32 v15, v15, s1, 0x42fe0000
	v_floor_f32_e32 v20, v20
	v_fmac_f32_e32 v19, v37, v30
	s_mov_b32 s3, 0xe3040fd1
	v_or_b16 v17.h, v13.h, v13.l
	v_or_b16 v17.l, v14.h, v14.l
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[13:14], null, v38, s3, 0xffffffffa509a484
	s_mov_b32 s3, 0x125b8c61
	v_cvt_i32_f32_e32 v22, v15
	v_mad_co_u64_u32 v[14:15], null, v38, s4, 0x667adfbd
	v_med3_num_f32 v23, v20, s1, 0x42fe0000
	v_floor_f32_e32 v24, v19
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[19:20], null, v38, s3, 0x3f469df8
	s_mov_b32 s4, 0xb0eb139d
	s_mov_b32 s3, 0x90158cf9
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[20:21], null, v38, s4, 0xffffffff865ce613
	v_cvt_i32_f32_e32 v15, v23
	v_med3_num_f32 v21, v24, s1, 0x42fe0000
	v_lshrrev_b32_e32 v23, 8, v13
	v_lshrrev_b32_e32 v24, 8, v14
	v_mad_co_u64_u32 v[13:14], null, v38, s3, 0xffffffff8aad3456
	v_lshrrev_b32_e32 v14, 8, v19
	s_mov_b32 s3, 0x1e0dc6ed
	v_lshrrev_b32_e32 v19, 8, v20
	v_cvt_i32_f32_e32 v20, v21
	v_cvt_f32_u32_e32 v21, v24
	v_cvt_f32_u32_e32 v14, v14
	v_cvt_f32_u32_e32 v23, v23
	v_lshrrev_b32_e32 v13, 8, v13
	v_cvt_f32_u32_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v21, 0x33800000, v21 :: v_dual_mul_f32 v14, 0x33800000, v14
	v_mul_f32_e32 v23, 0x33800000, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_u32_e32 v13, v13
	v_mul_f32_e32 v19, 0x33800000, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v21, v37, v16 :: v_dual_fmac_f32 v14, v37, v10
	v_fmac_f32_e32 v23, v37, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v9, 0x33800000, v13
	v_fmac_f32_e32 v19, v37, v12
	s_delay_alu instid0(VALU_DEP_4)
	v_floor_f32_e32 v10, v21
	v_floor_f32_e32 v13, v14
	v_floor_f32_e32 v12, v23
	v_fmac_f32_e32 v9, v37, v6
	v_floor_f32_e32 v14, v19
	v_med3_num_f32 v6, v10, s1, 0x42fe0000
	v_med3_num_f32 v10, v13, s1, 0x42fe0000
	s_mov_b32 s4, 0x996d7e75
	v_floor_f32_e32 v21, v9
	v_med3_num_f32 v16, v12, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v23, v6
	v_cvt_i32_f32_e32 v6, v10
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[9:10], null, v38, s3, 0x32dc8f7
	s_mov_b32 s3, 0x711a8809
	v_med3_num_f32 v19, v14, s1, 0x42fe0000
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[12:13], null, v38, s3, 0x43f391ea
	v_mad_co_u64_u32 v[13:14], null, v38, s4, 0xfffffffffbca9841
	v_lshlrev_b16 v6.l, 8, v6.l
	v_and_b16 v6.h, 0xff, v23.l
	v_cvt_i32_f32_e32 v14, v19
	v_lshrrev_b32_e32 v9, 8, v9
	v_cvt_i32_f32_e32 v10, v16
	v_med3_num_f32 v16, v21, s1, 0x42fe0000
	v_lshrrev_b32_e32 v12, 8, v12
	v_lshrrev_b32_e32 v13, 8, v13
	v_cvt_f32_u32_e32 v9, v9
	v_or_b16 v10.h, v6.h, v6.l
	v_and_b16 v6.h, 0xff, v14.l
	v_cvt_f32_u32_e32 v12, v12
	v_cvt_f32_u32_e32 v13, v13
	v_mul_f32_e32 v14, 0x33800000, v9
	v_cvt_i32_f32_e32 v16, v16
	v_lshlrev_b16 v9.l, 8, v10.l
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v12, 0x33800000, v12 :: v_dual_mul_f32 v13, 0x33800000, v13
	v_fmac_f32_e32 v14, v37, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b16 v6.l, 8, v16.l
	s_mov_b32 s3, 0xc45f09f1
	v_dual_fmac_f32 v12, v37, v25 :: v_dual_fmac_f32 v13, v37, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_floor_f32_e32 v14, v14
	v_or_b16 v10.l, v6.h, v6.l
	v_and_b16 v6.h, 0xff, v22.l
	s_delay_alu instid0(VALU_DEP_4)
	v_floor_f32_e32 v12, v12
	v_floor_f32_e32 v13, v13
	v_med3_num_f32 v21, v14, s1, 0x42fe0000
	s_mov_b32 s4, 0xcf52873d
	v_lshlrev_b16 v6.l, 8, v15.l
	v_med3_num_f32 v22, v12, s1, 0x42fe0000
	v_med3_num_f32 v23, v13, s1, 0x42fe0000
	v_mad_co_u64_u32 v[12:13], null, v38, s6, 0x5abbca0
	v_mad_co_u64_u32 v[13:14], null, v38, s7, 0x57d53705
	v_mad_co_u64_u32 v[14:15], null, v38, s5, 0x22331ebe
	v_and_b16 v9.h, 0xff, v20.l
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[15:16], null, v38, s4, 0x73fe081b
	v_mad_co_u64_u32 v[19:20], null, v38, s3, 0xffffffff9cbb94ac
	v_cvt_i32_f32_e32 v16, v21
	v_cvt_i32_f32_e32 v20, v22
	v_lshrrev_b32_e32 v12, 8, v12
	v_lshrrev_b32_e32 v13, 8, v13
	v_lshrrev_b32_e32 v14, 8, v14
	v_cvt_i32_f32_e32 v21, v23
	v_lshrrev_b32_e32 v15, 8, v15
	v_lshrrev_b32_e32 v19, 8, v19
	v_cvt_f32_u32_e32 v12, v12
	v_cvt_f32_u32_e32 v13, v13
	v_cvt_f32_u32_e32 v14, v14
	v_cvt_f32_u32_e32 v15, v15
	v_cvt_f32_u32_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v12, 0x33800000, v12 :: v_dual_mul_f32 v13, 0x33800000, v13
	v_dual_mul_f32 v14, 0x33800000, v14 :: v_dual_mul_f32 v15, 0x33800000, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v19, 0x33800000, v19
	v_fmac_f32_e32 v12, v37, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v13, v37, v11 :: v_dual_fmac_f32 v14, v37, v5
	v_fmac_f32_e32 v15, v37, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v19, v37, v4
	v_floor_f32_e32 v4, v12
	s_delay_alu instid0(VALU_DEP_4)
	v_floor_f32_e32 v5, v13
	v_floor_f32_e32 v7, v14
	v_floor_f32_e32 v8, v15
	v_floor_f32_e32 v11, v19
	v_med3_num_f32 v4, v4, s1, 0x42fe0000
	v_med3_num_f32 v5, v5, s1, 0x42fe0000
	v_med3_num_f32 v7, v7, s1, 0x42fe0000
	v_med3_num_f32 v8, v8, s1, 0x42fe0000
	v_med3_num_f32 v11, v11, s1, 0x42fe0000
	v_cvt_i32_f32_e32 v4, v4
	v_cvt_i32_f32_e32 v5, v5
	v_cvt_i32_f32_e32 v7, v7
	v_cvt_i32_f32_e32 v8, v8
	v_cvt_i32_f32_e32 v11, v11
	v_lshlrev_b16 v4.l, 8, v4.l
	v_and_b16 v4.h, 0xff, v5.l
	v_lshlrev_b16 v5.l, 8, v7.l
	v_and_b16 v5.h, 0xff, v8.l
	v_lshlrev_b16 v7.l, 8, v11.l
	v_and_b16 v7.h, 0xff, v21.l
	v_lshlrev_b16 v8.l, 8, v20.l
	v_and_b16 v8.h, 0xff, v16.l
	v_or_b16 v9.h, v9.h, v9.l
	v_or_b16 v9.l, v6.h, v6.l
	v_or_b16 v6.h, v4.h, v4.l
	v_or_b16 v6.l, v5.h, v5.l
	v_or_b16 v5.h, v7.h, v7.l
	v_or_b16 v5.l, v8.h, v8.l
	s_mov_b32 s1, exec_lo
	s_clause 0x3
	global_store_b64 v[0:1], v[2:3], off
	global_store_b64 v[0:1], v[17:18], off offset:16
	global_store_b64 v[0:1], v[9:10], off offset:32
	global_store_b64 v[0:1], v[5:6], off offset:48
                                        ; implicit-def: $vgpr38
	v_cmpx_eq_u32_e32 0, v36
	s_cbranch_execz .LBB0_261
; %bb.260:
	v_div_scale_f32 v0, null, 0x42fe0000, 0x42fe0000, v35
	s_or_b32 s2, s2, exec_lo
	v_rcp_f32_e32 v1, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v2, -v0, v1, 1.0
	v_fmac_f32_e32 v1, v2, v1
	v_div_scale_f32 v2, vcc_lo, v35, 0x42fe0000, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v3, v2, v1
	v_fma_f32 v4, -v0, v3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v4, v1
	v_fma_f32 v0, -v0, v3, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v0, v0, v1, v3
	v_div_fixup_f32 v0, v0, 0x42fe0000, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v38, 1.0, v0, s0
.LBB0_261:                              ; %Flow2350
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s0, s2
	s_cbranch_execnz .LBB0_264
	s_branch .LBB0_265
.LBB0_262:
                                        ; implicit-def: $vgpr38
	s_branch .LBB0_259
.LBB0_263:                              ; %Flow2349
	s_and_saveexec_b32 s0, s2
	s_cbranch_execz .LBB0_265
.LBB0_264:                              ; %.sink.split
	global_store_b32 v[91:92], v38, off
.LBB0_265:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	gated_delta_net_q8_register_scan_gfx1201, .Lfunc_end0-gated_delta_net_q8_register_scan_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gated_delta_net_q8_register_scan_gfx1201
		.amdhsa_group_segment_fixed_size 29824
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 88
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
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 192
		.amdhsa_next_free_sgpr 42
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
	.set .Lgated_delta_net_q8_register_scan_gfx1201.numbered_sgpr, 42
	.set .Lgated_delta_net_q8_register_scan_gfx1201.num_named_barrier, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.private_seg_size, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.uses_vcc, 1
	.set .Lgated_delta_net_q8_register_scan_gfx1201.uses_flat_scratch, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.has_dyn_sized_stack, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.has_recursion, 0
	.set .Lgated_delta_net_q8_register_scan_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 24064
; TotalNumSgprs: 44
; NumVgprs: 192
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 29824 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 192
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
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
	.type	__hip_cuid_67ef605b9f2aff2f,@object ; @__hip_cuid_67ef605b9f2aff2f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_67ef605b9f2aff2f
__hip_cuid_67ef605b9f2aff2f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_67ef605b9f2aff2f, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_67ef605b9f2aff2f
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
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         gated_delta_net_q8_register_scan_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     192
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
