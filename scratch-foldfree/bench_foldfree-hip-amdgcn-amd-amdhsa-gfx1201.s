	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201 ; -- Begin function gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.globl	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.p2align	8
	.type	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201,@function
gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201: ; @gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_lshlrev_b32_e32 v130, 2, v0
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x80, v0
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_getpc_b64 s[4:5]
	s_sext_i32_i16 s5, s5
	s_add_co_u32 s4, s4, ff_fold_lut@rel32@lo+8
	s_add_co_ci_u32 s5, s5, ff_fold_lut@rel32@hi+16
	v_add_nc_u32_e32 v2, 0, v130
	global_load_b32 v1, v130, s[4:5]
	s_wait_loadcnt 0x0
	ds_store_b32 v2, v1 offset:9216
.LBB0_2:                                ; %.preheader198
	s_or_b32 exec_lo, exec_lo, s2
	s_clause 0x1
	s_load_b96 s[8:10], s[0:1], 0x30
	s_load_b128 s[4:7], s[0:1], 0x0
	v_and_b32_e32 v1, 0xc0, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_lshrrev_b32_e32 v138, 1, v0
	s_mov_b32 s3, 0
	v_lshl_or_b32 v136, ttmp7, 8, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 4, v136
	s_barrier_wait -1
	s_wait_kmcnt 0x0
	s_ashr_i32 s24, s9, 4
	s_cmp_gt_i32 s9, 0
	v_mul_lo_u32 v1, v1, s24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[3:4], null, s24, 3, v[1:2]
	v_add_nc_u32_e32 v4, s24, v1
	v_lshl_add_u32 v5, s24, 1, v1
	v_lshlrev_b32_e32 v2, 1, v0
	v_readfirstlane_b32 s2, v1
	v_readfirstlane_b32 s16, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_readfirstlane_b32 s18, v5
	v_readfirstlane_b32 s20, v3
	s_cbranch_scc1 .LBB0_4
; %bb.3:                                ; %.preheader198..preheader192_crit_edge
	v_lshrrev_b32_e32 v1, 1, v0
	s_ashr_i32 s13, s8, 31
	s_mov_b32 s12, s8
	s_branch .LBB0_5
.LBB0_4:
	s_mov_b32 s3, -1
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $sgpr12_sgpr13
.LBB0_5:                                ; %Flow1334
	v_mov_b32_e32 v129, 0
	v_and_b32_e32 v137, 64, v2
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_lshl_b32 s11, ttmp9, 7
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v128, v129 :: v_dual_mov_b32 v127, v129
	v_dual_mov_b32 v126, v129 :: v_dual_mov_b32 v125, v129
	v_dual_mov_b32 v124, v129 :: v_dual_mov_b32 v123, v129
	v_dual_mov_b32 v122, v129 :: v_dual_mov_b32 v121, v129
	v_dual_mov_b32 v120, v129 :: v_dual_mov_b32 v119, v129
	v_dual_mov_b32 v118, v129 :: v_dual_mov_b32 v117, v129
	v_dual_mov_b32 v116, v129 :: v_dual_mov_b32 v115, v129
	v_dual_mov_b32 v114, v129 :: v_dual_mov_b32 v113, v129
	v_dual_mov_b32 v112, v129 :: v_dual_mov_b32 v111, v129
	v_dual_mov_b32 v110, v129 :: v_dual_mov_b32 v109, v129
	v_dual_mov_b32 v108, v129 :: v_dual_mov_b32 v107, v129
	v_dual_mov_b32 v106, v129 :: v_dual_mov_b32 v105, v129
	v_dual_mov_b32 v104, v129 :: v_dual_mov_b32 v103, v129
	v_dual_mov_b32 v102, v129 :: v_dual_mov_b32 v101, v129
	v_dual_mov_b32 v100, v129 :: v_dual_mov_b32 v99, v129
	v_dual_mov_b32 v98, v129 :: v_dual_mov_b32 v97, v129
	v_dual_mov_b32 v96, v129 :: v_dual_mov_b32 v95, v129
	v_dual_mov_b32 v94, v129 :: v_dual_mov_b32 v93, v129
	v_dual_mov_b32 v92, v129 :: v_dual_mov_b32 v91, v129
	v_dual_mov_b32 v90, v129 :: v_dual_mov_b32 v89, v129
	v_dual_mov_b32 v88, v129 :: v_dual_mov_b32 v87, v129
	v_dual_mov_b32 v86, v129 :: v_dual_mov_b32 v85, v129
	v_dual_mov_b32 v84, v129 :: v_dual_mov_b32 v83, v129
	v_dual_mov_b32 v82, v129 :: v_dual_mov_b32 v81, v129
	v_dual_mov_b32 v80, v129 :: v_dual_mov_b32 v79, v129
	v_dual_mov_b32 v78, v129 :: v_dual_mov_b32 v77, v129
	v_dual_mov_b32 v76, v129 :: v_dual_mov_b32 v75, v129
	v_dual_mov_b32 v74, v129 :: v_dual_mov_b32 v73, v129
	v_dual_mov_b32 v72, v129 :: v_dual_mov_b32 v71, v129
	v_dual_mov_b32 v70, v129 :: v_dual_mov_b32 v69, v129
	v_dual_mov_b32 v68, v129 :: v_dual_mov_b32 v67, v129
	v_dual_mov_b32 v66, v129 :: v_dual_mov_b32 v65, v129
	v_dual_mov_b32 v64, v129 :: v_dual_mov_b32 v63, v129
	v_dual_mov_b32 v62, v129 :: v_dual_mov_b32 v61, v129
	v_dual_mov_b32 v60, v129 :: v_dual_mov_b32 v59, v129
	v_dual_mov_b32 v58, v129 :: v_dual_mov_b32 v57, v129
	v_dual_mov_b32 v56, v129 :: v_dual_mov_b32 v55, v129
	v_dual_mov_b32 v54, v129 :: v_dual_mov_b32 v53, v129
	v_dual_mov_b32 v52, v129 :: v_dual_mov_b32 v51, v129
	v_dual_mov_b32 v50, v129 :: v_dual_mov_b32 v49, v129
	v_dual_mov_b32 v48, v129 :: v_dual_mov_b32 v47, v129
	v_dual_mov_b32 v46, v129 :: v_dual_mov_b32 v45, v129
	v_dual_mov_b32 v44, v129 :: v_dual_mov_b32 v43, v129
	v_dual_mov_b32 v42, v129 :: v_dual_mov_b32 v41, v129
	v_dual_mov_b32 v40, v129 :: v_dual_mov_b32 v39, v129
	v_dual_mov_b32 v38, v129 :: v_dual_mov_b32 v37, v129
	v_dual_mov_b32 v36, v129 :: v_dual_mov_b32 v35, v129
	v_dual_mov_b32 v34, v129 :: v_dual_mov_b32 v33, v129
	v_dual_mov_b32 v32, v129 :: v_dual_mov_b32 v31, v129
	v_dual_mov_b32 v30, v129 :: v_dual_mov_b32 v29, v129
	v_dual_mov_b32 v28, v129 :: v_dual_mov_b32 v27, v129
	v_dual_mov_b32 v26, v129 :: v_dual_mov_b32 v25, v129
	v_dual_mov_b32 v24, v129 :: v_dual_mov_b32 v23, v129
	v_dual_mov_b32 v22, v129 :: v_dual_mov_b32 v21, v129
	v_dual_mov_b32 v20, v129 :: v_dual_mov_b32 v19, v129
	v_dual_mov_b32 v18, v129 :: v_dual_mov_b32 v17, v129
	v_dual_mov_b32 v16, v129 :: v_dual_mov_b32 v15, v129
	v_dual_mov_b32 v14, v129 :: v_dual_mov_b32 v13, v129
	v_dual_mov_b32 v12, v129 :: v_dual_mov_b32 v11, v129
	v_dual_mov_b32 v10, v129 :: v_dual_mov_b32 v9, v129
	v_dual_mov_b32 v8, v129 :: v_dual_mov_b32 v7, v129
	v_dual_mov_b32 v6, v129 :: v_dual_mov_b32 v5, v129
	v_dual_mov_b32 v4, v129 :: v_dual_mov_b32 v3, v129
	v_mov_b32_e32 v2, v129
	s_cbranch_vccnz .LBB0_9
; %bb.6:                                ; %.lr.ph
	s_load_b64 s[26:27], s[0:1], 0x10
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v9, 3, v0
	v_and_b32_e32 v10, 12, v130
	s_lshl_b32 s2, s2, 8
	s_mov_b32 s3, 0
	s_ashr_i32 s13, s8, 31
	s_mov_b32 s12, s8
	v_and_or_b32 v1, 0x70, v138, v10
	v_lshrrev_b32_e32 v11, 3, v0
	v_lshlrev_b32_e32 v12, 4, v0
	s_ashr_i32 s22, s8, 4
	v_dual_mov_b32 v14, v2 :: v_dual_mov_b32 v15, v2
	v_add_nc_u32_e32 v6, s11, v1
	v_bfe_u32 v1, v0, 3, 2
	v_dual_mov_b32 v16, v2 :: v_dual_mov_b32 v17, v2
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v19, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_ashrrev_i32_e32 v7, 4, v6
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[14:15], s[26:27], s[2:3]
	s_lshl_b32 s2, s16, 8
	v_dual_mov_b32 v20, v2 :: v_dual_mov_b32 v21, v2
	v_mad_co_u64_u32 v[3:4], null, s24, v7, v[1:2]
	s_add_nc_u64 s[16:17], s[26:27], s[2:3]
	s_lshl_b32 s2, s18, 8
	v_ashrrev_i32_e32 v8, 31, v7
	s_add_nc_u64 s[18:19], s[26:27], s[2:3]
	s_lshl_b32 s2, s20, 8
	v_and_b32_e32 v1, 0xf8, v9
	s_add_nc_u64 s[20:21], s[26:27], s[2:3]
	s_mov_b32 s2, s9
	v_mad_co_u64_u32 v[4:5], null, s24, v8, v[4:5]
	s_mul_u64 s[24:25], s[2:3], s[12:13]
	v_lshlrev_b64_e32 v[8:9], 6, v[7:8]
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b64 s[24:25], s[24:25], 1
	v_ashrrev_i32_e32 v7, 31, v6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[24:25], s[4:5], s[24:25]
	v_dual_mov_b32 v22, v2 :: v_dual_mov_b32 v23, v2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v8, vcc_lo, s24, v8
	v_lshlrev_b64_e32 v[5:6], 2, v[6:7]
	v_add_co_ci_u32_e64 v9, null, s25, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v130, vcc_lo, v8, v10
	v_lshlrev_b64_e32 v[3:4], 7, v[3:4]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v131, null, 0, v9, vcc_lo
	v_add_co_u32 v132, vcc_lo, s6, v5
	v_lshlrev_b32_e32 v5, 5, v0
	v_mad_u32_u24 v7, 0x120, v11, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v133, null, s7, v6, vcc_lo
	v_lshrrev_b32_e32 v6, 2, v137
	v_and_b32_e32 v5, 0xe0, v5
	v_dual_mov_b32 v9, v2 :: v_dual_mov_b32 v10, v2
	v_mov_b32_e32 v13, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_u32_u24_e32 v6, 0x120, v6
	v_add_nc_u32_e32 v139, v7, v5
	v_mov_b32_e32 v7, v2
	v_and_or_b32 v3, 0x70, v12, v3
	v_mov_b32_e32 v5, v2
	v_dual_mov_b32 v11, v2 :: v_dual_mov_b32 v12, v2
	v_dual_mov_b32 v24, v2 :: v_dual_mov_b32 v25, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v134, vcc_lo, s4, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, s5, v4, vcc_lo
	v_mov_b32_e32 v4, v2
	v_dual_mov_b32 v3, v2 :: v_dual_add_nc_u32 v8, 0, v1
	v_dual_mov_b32 v26, v2 :: v_dual_mov_b32 v27, v2
	v_dual_mov_b32 v28, v2 :: v_dual_mov_b32 v29, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v140, v8, v6
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v8, v2
	v_dual_mov_b32 v30, v2 :: v_dual_mov_b32 v31, v2
	v_dual_mov_b32 v32, v2 :: v_dual_mov_b32 v33, v2
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v35, v2
	v_dual_mov_b32 v36, v2 :: v_dual_mov_b32 v37, v2
	v_dual_mov_b32 v38, v2 :: v_dual_mov_b32 v39, v2
	v_dual_mov_b32 v40, v2 :: v_dual_mov_b32 v41, v2
	v_dual_mov_b32 v42, v2 :: v_dual_mov_b32 v43, v2
	v_dual_mov_b32 v44, v2 :: v_dual_mov_b32 v45, v2
	v_dual_mov_b32 v46, v2 :: v_dual_mov_b32 v47, v2
	v_dual_mov_b32 v48, v2 :: v_dual_mov_b32 v49, v2
	v_dual_mov_b32 v50, v2 :: v_dual_mov_b32 v51, v2
	v_dual_mov_b32 v52, v2 :: v_dual_mov_b32 v53, v2
	v_dual_mov_b32 v54, v2 :: v_dual_mov_b32 v55, v2
	v_dual_mov_b32 v56, v2 :: v_dual_mov_b32 v57, v2
	v_dual_mov_b32 v58, v2 :: v_dual_mov_b32 v59, v2
	v_dual_mov_b32 v60, v2 :: v_dual_mov_b32 v61, v2
	v_dual_mov_b32 v62, v2 :: v_dual_mov_b32 v63, v2
	v_dual_mov_b32 v64, v2 :: v_dual_mov_b32 v65, v2
	v_dual_mov_b32 v66, v2 :: v_dual_mov_b32 v67, v2
	v_dual_mov_b32 v68, v2 :: v_dual_mov_b32 v69, v2
	v_dual_mov_b32 v70, v2 :: v_dual_mov_b32 v71, v2
	v_dual_mov_b32 v72, v2 :: v_dual_mov_b32 v73, v2
	v_dual_mov_b32 v74, v2 :: v_dual_mov_b32 v75, v2
	v_dual_mov_b32 v76, v2 :: v_dual_mov_b32 v77, v2
	v_dual_mov_b32 v78, v2 :: v_dual_mov_b32 v79, v2
	v_dual_mov_b32 v80, v2 :: v_dual_mov_b32 v81, v2
	v_dual_mov_b32 v82, v2 :: v_dual_mov_b32 v83, v2
	v_dual_mov_b32 v84, v2 :: v_dual_mov_b32 v85, v2
	v_dual_mov_b32 v86, v2 :: v_dual_mov_b32 v87, v2
	v_dual_mov_b32 v88, v2 :: v_dual_mov_b32 v89, v2
	v_dual_mov_b32 v90, v2 :: v_dual_mov_b32 v91, v2
	v_dual_mov_b32 v92, v2 :: v_dual_mov_b32 v93, v2
	v_dual_mov_b32 v94, v2 :: v_dual_mov_b32 v95, v2
	v_dual_mov_b32 v96, v2 :: v_dual_mov_b32 v97, v2
	v_dual_mov_b32 v98, v2 :: v_dual_mov_b32 v99, v2
	v_dual_mov_b32 v100, v2 :: v_dual_mov_b32 v101, v2
	v_dual_mov_b32 v102, v2 :: v_dual_mov_b32 v103, v2
	v_dual_mov_b32 v104, v2 :: v_dual_mov_b32 v105, v2
	v_dual_mov_b32 v106, v2 :: v_dual_mov_b32 v107, v2
	v_dual_mov_b32 v108, v2 :: v_dual_mov_b32 v109, v2
	v_dual_mov_b32 v110, v2 :: v_dual_mov_b32 v111, v2
	v_dual_mov_b32 v112, v2 :: v_dual_mov_b32 v113, v2
	v_dual_mov_b32 v114, v2 :: v_dual_mov_b32 v115, v2
	v_dual_mov_b32 v116, v2 :: v_dual_mov_b32 v117, v2
	v_dual_mov_b32 v118, v2 :: v_dual_mov_b32 v119, v2
	v_dual_mov_b32 v120, v2 :: v_dual_mov_b32 v121, v2
	v_dual_mov_b32 v122, v2 :: v_dual_mov_b32 v123, v2
	v_dual_mov_b32 v124, v2 :: v_dual_mov_b32 v125, v2
	v_dual_mov_b32 v126, v2 :: v_dual_mov_b32 v127, v2
	v_dual_mov_b32 v128, v2 :: v_dual_mov_b32 v129, v2
	v_add_nc_u32_e32 v141, 0x800, v140
	v_add_nc_u32_e32 v142, 0xc00, v140
	s_ashr_i32 s23, s22, 31
	s_movk_i32 s2, 0x7f
	s_lshl_b64 s[4:5], s[22:23], 6
	s_mov_b32 s22, 0x4040404
.LBB0_7:                                ; %.preheader196
                                        ; =>This Inner Loop Header: Depth=1
	s_lshr_b32 s23, s3, 7
	s_clause 0x7
	global_load_b64 v[151:152], v1, s[14:15]
	global_load_b64 v[153:154], v1, s[14:15] offset:256
	global_load_b64 v[155:156], v1, s[14:15] offset:512
	global_load_b64 v[157:158], v1, s[14:15] offset:768
	global_load_b64 v[159:160], v1, s[16:17]
	global_load_b64 v[161:162], v1, s[16:17] offset:256
	global_load_b64 v[163:164], v1, s[16:17] offset:512
	global_load_b64 v[165:166], v1, s[16:17] offset:768
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[147:148], null, s4, s23, v[130:131]
	s_clause 0x7
	global_load_b64 v[167:168], v1, s[18:19]
	global_load_b64 v[169:170], v1, s[18:19] offset:256
	global_load_b64 v[171:172], v1, s[18:19] offset:512
	global_load_b64 v[173:174], v1, s[18:19] offset:768
	global_load_b64 v[175:176], v1, s[20:21]
	global_load_b64 v[177:178], v1, s[20:21] offset:256
	global_load_b64 v[179:180], v1, s[20:21] offset:512
	global_load_b64 v[181:182], v1, s[20:21] offset:768
	;;#ASMSTART
	;;#ASMEND
	v_mad_co_u64_u32 v[148:149], null, s5, s23, v[148:149]
	global_load_b128 v[143:146], v[132:133], off
	global_load_b32 v187, v[147:148], off
	global_load_b128 v[147:150], v[134:135], off
	s_wait_loadcnt 0x2
	v_bfe_u32 v143, v143, 23, 8
	s_wait_loadcnt 0x1
	v_bfe_u32 v183, v187, 2, 5
	v_and_b32_e32 v184, 3, v187
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v188, 4, v147
	v_bfe_u32 v144, v144, 23, 8
	v_bfe_u32 v145, v145, 23, 8
	v_sub_nc_u32_e32 v143, v143, v183
	v_cmp_eq_u32_e32 vcc_lo, 0, v184
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_med3_i32 v143, 0x70, v143, s2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v183, 16, 0, vcc_lo
	v_add_nc_u32_e32 v143, v143, v183
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v143, v143, 4, 0
	v_add_nc_u32_e32 v183, 0x1d00, v143
	v_add_nc_u32_e32 v143, 0x1d08, v143
	ds_load_2addr_b32 v[183:184], v183 offset1:1
	ds_load_2addr_b32 v[185:186], v143 offset1:1
	v_and_b32_e32 v143, 0xf0f0f0f, v147
	v_and_b32_e32 v147, 0xf0f0f0f, v188
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v188, v147, v143, 0x5010400
	v_perm_b32 v143, v147, v143, 0x7030602
	v_and_b32_e32 v147, 0x7070707, v188
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_and_b32_e32 v189, 0x7070707, v143
	v_lshrrev_b32_e32 v143, 1, v143
	s_wait_dscnt 0x1
	v_perm_b32 v190, v184, v183, v147
	s_delay_alu instid0(VALU_DEP_3)
	v_perm_b32 v183, v184, v183, v189
	s_wait_dscnt 0x0
	v_perm_b32 v147, v186, v185, v147
	v_perm_b32 v184, v186, v185, v189
	v_bfe_u32 v185, v187, 10, 5
	v_lshrrev_b32_e32 v186, 1, v188
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_nc_u32_e32 v144, v144, v185
	v_and_b32_e32 v185, 0x300, v187
	v_med3_i32 v144, 0x70, v144, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0, v185
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v185, 16, 0, vcc_lo
	v_add_nc_u32_e32 v144, v144, v185
	v_and_or_b32 v185, v186, s22, 0x3020100
	v_and_or_b32 v186, v143, s22, 0x3020100
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v188, v144, 4, 0
	v_perm_b32 v143, v147, v190, v185
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_perm_b32 v144, v184, v183, v186
	v_lshrrev_b32_e32 v185, 4, v148
	v_add_nc_u32_e32 v147, 0x1d00, v188
	v_add_nc_u32_e32 v183, 0x1d08, v188
	ds_store_b64 v139, v[143:144]
	ds_load_2addr_b32 v[143:144], v147 offset1:1
	ds_load_2addr_b32 v[183:184], v183 offset1:1
	v_and_b32_e32 v147, 0xf0f0f0f, v148
	v_and_b32_e32 v148, 0xf0f0f0f, v185
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v185, v148, v147, 0x5010400
	v_perm_b32 v147, v148, v147, 0x7030602
	v_and_b32_e32 v148, 0x7070707, v185
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_and_b32_e32 v186, 0x7070707, v147
	v_lshrrev_b32_e32 v147, 1, v147
	s_wait_dscnt 0x1
	v_perm_b32 v188, v144, v143, v148
	s_delay_alu instid0(VALU_DEP_3)
	v_perm_b32 v144, v144, v143, v186
	s_wait_dscnt 0x0
	v_perm_b32 v143, v184, v183, v148
	v_perm_b32 v148, v184, v183, v186
	v_bfe_u32 v183, v187, 18, 5
	v_lshrrev_b32_e32 v184, 1, v185
	v_and_or_b32 v147, v147, s22, 0x3020100
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_sub_nc_u32_e32 v145, v145, v183
	v_and_b32_e32 v183, 0x30000, v187
	v_perm_b32 v144, v148, v144, v147
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_i32 v145, 0x70, v145, s2
	v_cmp_eq_u32_e32 vcc_lo, 0, v183
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v183, 16, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v145, v145, v183
	v_and_or_b32 v183, v184, s22, 0x3020100
	v_lshrrev_b32_e32 v184, 4, v149
	v_and_b32_e32 v149, 0xf0f0f0f, v149
	v_lshl_add_u32 v145, v145, 4, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_perm_b32 v143, v143, v188, v183
	v_and_b32_e32 v183, 0xf0f0f0f, v184
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v147, 0x1d00, v145
	v_add_nc_u32_e32 v145, 0x1d08, v145
	ds_store_b64 v139, v[143:144] offset:8
	ds_load_2addr_b32 v[143:144], v147 offset1:1
	ds_load_2addr_b32 v[147:148], v145 offset1:1
	v_bfe_u32 v145, v146, 23, 8
	v_bfe_u32 v146, v187, 26, 5
	v_perm_b32 v184, v183, v149, 0x5010400
	v_perm_b32 v149, v183, v149, 0x7030602
	v_and_b32_e32 v183, 0x3000000, v187
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_nc_u32_e32 v145, v145, v146
	v_lshrrev_b32_e32 v146, 1, v184
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v185, 1, v149
	v_cmp_eq_u32_e32 vcc_lo, 0, v183
	v_and_b32_e32 v184, 0x7070707, v184
	v_med3_i32 v145, 0x70, v145, s2
	v_and_b32_e32 v149, 0x7070707, v149
	v_and_or_b32 v146, v146, s22, 0x3020100
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v183, 16, 0, vcc_lo
	v_and_or_b32 v185, v185, s22, 0x3020100
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v145, v145, v183
	s_wait_dscnt 0x1
	v_perm_b32 v183, v144, v143, v184
	s_wait_dscnt 0x0
	v_perm_b32 v184, v148, v147, v184
	v_perm_b32 v144, v144, v143, v149
	v_perm_b32 v147, v148, v147, v149
	v_lshl_add_u32 v145, v145, 4, 0
	v_and_b32_e32 v148, 0xf0f0f0f, v150
	v_perm_b32 v143, v184, v183, v146
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_perm_b32 v144, v147, v144, v185
	v_add_nc_u32_e32 v146, 0x1d00, v145
	v_add_nc_u32_e32 v145, 0x1d08, v145
	v_lshrrev_b32_e32 v147, 4, v150
	ds_store_b64 v139, v[143:144] offset:16
	ds_load_2addr_b32 v[143:144], v146 offset1:1
	ds_load_2addr_b32 v[145:146], v145 offset1:1
	v_and_b32_e32 v147, 0xf0f0f0f, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_perm_b32 v149, v147, v148, 0x5010400
	v_perm_b32 v147, v147, v148, 0x7030602
	v_lshrrev_b32_e32 v148, 1, v149
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v150, 1, v147
	v_and_b32_e32 v149, 0x7070707, v149
	v_and_b32_e32 v147, 0x7070707, v147
	v_and_or_b32 v148, v148, s22, 0x3020100
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v150, v150, s22, 0x3020100
	s_wait_dscnt 0x1
	v_perm_b32 v183, v144, v143, v149
	s_wait_dscnt 0x0
	v_perm_b32 v149, v146, v145, v149
	v_perm_b32 v144, v144, v143, v147
	v_perm_b32 v145, v146, v145, v147
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_perm_b32 v143, v149, v183, v148
	v_perm_b32 v144, v145, v144, v150
	ds_store_b64 v139, v[143:144] offset:24
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_2addr_b64 v[143:146], v140 offset1:144
	ds_load_2addr_b64 v[147:150], v141 offset0:32 offset1:176
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[122:129], v[143:144], v[151:152], v[122:129]
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[145:146], v[151:152], v[114:121]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[147:148], v[151:152], v[106:113]
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[149:150], v[151:152], v[98:105]
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[143:144], v[159:160], v[90:97]
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[145:146], v[159:160], v[82:89]
	v_wmma_f32_16x16x16_fp8_fp8 v[74:81], v[147:148], v[159:160], v[74:81]
	v_wmma_f32_16x16x16_fp8_fp8 v[66:73], v[149:150], v[159:160], v[66:73]
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[143:144], v[167:168], v[58:65]
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[145:146], v[167:168], v[50:57]
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[147:148], v[167:168], v[42:49]
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[149:150], v[167:168], v[34:41]
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[143:144], v[175:176], v[26:33]
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[145:146], v[175:176], v[18:25]
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[147:148], v[175:176], v[10:17]
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[149:150], v[175:176], v[2:9]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[143:146], v140 offset0:36 offset1:180
	ds_load_2addr_b64 v[147:150], v141 offset0:68 offset1:212
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[122:129], v[143:144], v[153:154], v[122:129]
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[145:146], v[153:154], v[114:121]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[147:148], v[153:154], v[106:113]
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[149:150], v[153:154], v[98:105]
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[143:144], v[161:162], v[90:97]
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[145:146], v[161:162], v[82:89]
	v_wmma_f32_16x16x16_fp8_fp8 v[74:81], v[147:148], v[161:162], v[74:81]
	v_wmma_f32_16x16x16_fp8_fp8 v[66:73], v[149:150], v[161:162], v[66:73]
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[143:144], v[169:170], v[58:65]
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[145:146], v[169:170], v[50:57]
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[147:148], v[169:170], v[42:49]
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[149:150], v[169:170], v[34:41]
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[143:144], v[177:178], v[26:33]
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[145:146], v[177:178], v[18:25]
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[147:148], v[177:178], v[10:17]
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[149:150], v[177:178], v[2:9]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[143:146], v140 offset0:72 offset1:216
	ds_load_2addr_b64 v[147:150], v141 offset0:104 offset1:248
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[122:129], v[143:144], v[155:156], v[122:129]
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[145:146], v[155:156], v[114:121]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[147:148], v[155:156], v[106:113]
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[149:150], v[155:156], v[98:105]
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[143:144], v[163:164], v[90:97]
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[145:146], v[163:164], v[82:89]
	v_wmma_f32_16x16x16_fp8_fp8 v[74:81], v[147:148], v[163:164], v[74:81]
	v_wmma_f32_16x16x16_fp8_fp8 v[66:73], v[149:150], v[163:164], v[66:73]
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[143:144], v[171:172], v[58:65]
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[145:146], v[171:172], v[50:57]
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[147:148], v[171:172], v[42:49]
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[149:150], v[171:172], v[34:41]
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[143:144], v[179:180], v[26:33]
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[145:146], v[179:180], v[18:25]
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[147:148], v[179:180], v[10:17]
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[149:150], v[179:180], v[2:9]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[143:146], v140 offset0:108 offset1:252
	ds_load_2addr_b64 v[147:150], v142 offset0:12 offset1:156
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[122:129], v[143:144], v[157:158], v[122:129]
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[145:146], v[157:158], v[114:121]
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[143:144], v[165:166], v[90:97]
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[145:146], v[165:166], v[82:89]
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[143:144], v[173:174], v[58:65]
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[145:146], v[173:174], v[50:57]
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[143:144], v[181:182], v[26:33]
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[145:146], v[181:182], v[18:25]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[147:148], v[157:158], v[106:113]
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[149:150], v[157:158], v[98:105]
	v_wmma_f32_16x16x16_fp8_fp8 v[74:81], v[147:148], v[165:166], v[74:81]
	v_wmma_f32_16x16x16_fp8_fp8 v[66:73], v[149:150], v[165:166], v[66:73]
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[147:148], v[173:174], v[42:49]
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[149:150], v[173:174], v[34:41]
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[147:148], v[181:182], v[10:17]
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[149:150], v[181:182], v[2:9]
	; sched_barrier mask(0x00000000)
	s_barrier_signal -1
	v_add_co_u32 v134, vcc_lo, 0x200, v134
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v135, null, 0, v135, vcc_lo
	v_add_nc_u32_e32 v1, 0x400, v1
	s_add_co_i32 s3, s3, 64
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s3, s9
	s_barrier_wait -1
	s_cbranch_scc0 .LBB0_7
; %bb.8:                                ; %Flow1333
	v_mov_b32_e32 v1, v138
.LBB0_9:                                ; %Flow1335
	s_load_b128 s[0:3], s[0:1], 0x20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 8, v1
	v_and_or_b32 v130, v0, 15, v136
	s_mov_b32 s4, exec_lo
	v_or3_b32 v0, s11, v1, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_ashrrev_i32_e32 v131, 31, v130
	v_cmpx_gt_i32_e64 s10, v130
	s_cbranch_execz .LBB0_74
; %bb.10:                               ; %.preheader
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[132:133], 2, v[130:131]
	v_mul_lo_u32 v1, s13, v130
	v_mul_lo_u32 v135, s12, v131
	s_mov_b32 s5, exec_lo
	s_wait_kmcnt 0x0
	v_add_co_u32 v132, vcc_lo, s0, v132
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v133, null, s1, v133, vcc_lo
	global_load_b32 v132, v[132:133], off
	v_mad_co_u64_u32 v[133:134], null, s12, v130, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v134, v134, v135, v1
	v_lshlrev_b64_e32 v[133:134], 2, v[133:134]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v133, vcc_lo, s2, v133
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v134, null, s3, v134, vcc_lo
	v_cmpx_gt_i32_e64 s8, v0
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[135:136], 2, v[0:1]
	v_add_co_u32 v137, vcc_lo, s6, v135
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v138, null, s7, v136, vcc_lo
	v_add_co_u32 v135, vcc_lo, v133, v135
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, v134, v136, vcc_lo
	global_load_b32 v1, v[137:138], off
	global_load_b32 v137, v[135:136], off
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v137, v1, v122
	global_store_b32 v[135:136], v137, off
.LBB0_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 1, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[135:136], 2, v[0:1]
	v_add_co_u32 v137, vcc_lo, s6, v135
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v138, null, s7, v136, vcc_lo
	v_add_co_u32 v135, vcc_lo, v133, v135
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v136, null, v134, v136, vcc_lo
	global_load_b32 v1, v[137:138], off offset:4
	global_load_b32 v122, v[135:136], off offset:4
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v122, v1, v123
	global_store_b32 v[135:136], v122, off offset:4
.LBB0_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 2, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_16
; %bb.15:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[122:123], 2, v[0:1]
	v_add_co_u32 v135, vcc_lo, s6, v122
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v136, null, s7, v123, vcc_lo
	v_add_co_u32 v122, vcc_lo, v133, v122
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v123, null, v134, v123, vcc_lo
	global_load_b32 v1, v[135:136], off offset:8
	global_load_b32 v135, v[122:123], off offset:8
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v135, v1, v124
	global_store_b32 v[122:123], v135, off offset:8
.LBB0_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 3, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_18
; %bb.17:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[122:123], 2, v[0:1]
	v_add_co_u32 v135, vcc_lo, s6, v122
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v136, null, s7, v123, vcc_lo
	v_add_co_u32 v122, vcc_lo, v133, v122
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v123, null, v134, v123, vcc_lo
	global_load_b32 v1, v[135:136], off offset:12
	global_load_b32 v124, v[122:123], off offset:12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v124, v1, v125
	global_store_b32 v[122:123], v124, off offset:12
.LBB0_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 4, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[122:123], 2, v[0:1]
	v_add_co_u32 v124, vcc_lo, s6, v122
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v125, null, s7, v123, vcc_lo
	v_add_co_u32 v122, vcc_lo, v133, v122
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v123, null, v134, v123, vcc_lo
	global_load_b32 v1, v[124:125], off offset:16
	global_load_b32 v124, v[122:123], off offset:16
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v124, v1, v126
	global_store_b32 v[122:123], v124, off offset:16
.LBB0_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 5, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[122:123], 2, v[0:1]
	v_add_co_u32 v124, vcc_lo, s6, v122
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v125, null, s7, v123, vcc_lo
	v_add_co_u32 v122, vcc_lo, v133, v122
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v123, null, v134, v123, vcc_lo
	global_load_b32 v1, v[124:125], off offset:20
	global_load_b32 v124, v[122:123], off offset:20
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v124, v1, v127
	global_store_b32 v[122:123], v124, off offset:20
.LBB0_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 6, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[122:123], 2, v[0:1]
	v_add_co_u32 v124, vcc_lo, s6, v122
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v125, null, s7, v123, vcc_lo
	v_add_co_u32 v122, vcc_lo, v133, v122
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v123, null, v134, v123, vcc_lo
	global_load_b32 v1, v[124:125], off offset:24
	global_load_b32 v124, v[122:123], off offset:24
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v124, v1, v128
	global_store_b32 v[122:123], v124, off offset:24
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 7, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[122:123], 2, v[0:1]
	v_add_co_u32 v124, vcc_lo, s6, v122
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v125, null, s7, v123, vcc_lo
	v_add_co_u32 v122, vcc_lo, v133, v122
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v123, null, v134, v123, vcc_lo
	global_load_b32 v1, v[124:125], off offset:28
	global_load_b32 v124, v[122:123], off offset:28
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v124, v1, v129
	global_store_b32 v[122:123], v124, off offset:28
.LBB0_26:                               ; %.preheader.1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 16, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[122:123], 2, v[0:1]
	v_add_co_u32 v124, vcc_lo, s6, v122
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v125, null, s7, v123, vcc_lo
	v_add_co_u32 v122, vcc_lo, v133, v122
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v123, null, v134, v123, vcc_lo
	global_load_b32 v1, v[124:125], off offset:64
	global_load_b32 v124, v[122:123], off offset:64
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v124, v1, v114
	global_store_b32 v[122:123], v124, off offset:64
.LBB0_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 17, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[122:123], 2, v[0:1]
	v_add_co_u32 v124, vcc_lo, s6, v122
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v125, null, s7, v123, vcc_lo
	v_add_co_u32 v122, vcc_lo, v133, v122
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v123, null, v134, v123, vcc_lo
	global_load_b32 v1, v[124:125], off offset:68
	global_load_b32 v114, v[122:123], off offset:68
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v114, v1, v115
	global_store_b32 v[122:123], v114, off offset:68
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 18, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[114:115], 2, v[0:1]
	v_add_co_u32 v122, vcc_lo, s6, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v123, null, s7, v115, vcc_lo
	v_add_co_u32 v114, vcc_lo, v133, v114
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v115, null, v134, v115, vcc_lo
	global_load_b32 v1, v[122:123], off offset:72
	global_load_b32 v122, v[114:115], off offset:72
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v122, v1, v116
	global_store_b32 v[114:115], v122, off offset:72
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 19, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[114:115], 2, v[0:1]
	v_add_co_u32 v122, vcc_lo, s6, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v123, null, s7, v115, vcc_lo
	v_add_co_u32 v114, vcc_lo, v133, v114
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v115, null, v134, v115, vcc_lo
	global_load_b32 v1, v[122:123], off offset:76
	global_load_b32 v116, v[114:115], off offset:76
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v116, v1, v117
	global_store_b32 v[114:115], v116, off offset:76
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 20, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[114:115], 2, v[0:1]
	v_add_co_u32 v116, vcc_lo, s6, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v117, null, s7, v115, vcc_lo
	v_add_co_u32 v114, vcc_lo, v133, v114
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v115, null, v134, v115, vcc_lo
	global_load_b32 v1, v[116:117], off offset:80
	global_load_b32 v116, v[114:115], off offset:80
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v116, v1, v118
	global_store_b32 v[114:115], v116, off offset:80
.LBB0_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 21, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[114:115], 2, v[0:1]
	v_add_co_u32 v116, vcc_lo, s6, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v117, null, s7, v115, vcc_lo
	v_add_co_u32 v114, vcc_lo, v133, v114
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v115, null, v134, v115, vcc_lo
	global_load_b32 v1, v[116:117], off offset:84
	global_load_b32 v116, v[114:115], off offset:84
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v116, v1, v119
	global_store_b32 v[114:115], v116, off offset:84
.LBB0_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 22, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[114:115], 2, v[0:1]
	v_add_co_u32 v116, vcc_lo, s6, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v117, null, s7, v115, vcc_lo
	v_add_co_u32 v114, vcc_lo, v133, v114
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v115, null, v134, v115, vcc_lo
	global_load_b32 v1, v[116:117], off offset:88
	global_load_b32 v116, v[114:115], off offset:88
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v116, v1, v120
	global_store_b32 v[114:115], v116, off offset:88
.LBB0_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 23, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[114:115], 2, v[0:1]
	v_add_co_u32 v116, vcc_lo, s6, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v117, null, s7, v115, vcc_lo
	v_add_co_u32 v114, vcc_lo, v133, v114
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v115, null, v134, v115, vcc_lo
	global_load_b32 v1, v[116:117], off offset:92
	global_load_b32 v116, v[114:115], off offset:92
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v116, v1, v121
	global_store_b32 v[114:115], v116, off offset:92
.LBB0_42:                               ; %.preheader.2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 32, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[114:115], 2, v[0:1]
	v_add_co_u32 v116, vcc_lo, s6, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v117, null, s7, v115, vcc_lo
	v_add_co_u32 v114, vcc_lo, v133, v114
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v115, null, v134, v115, vcc_lo
	global_load_b32 v1, v[116:117], off offset:128
	global_load_b32 v116, v[114:115], off offset:128
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v116, v1, v106
	global_store_b32 v[114:115], v116, off offset:128
.LBB0_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 33, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[114:115], 2, v[0:1]
	v_add_co_u32 v116, vcc_lo, s6, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v117, null, s7, v115, vcc_lo
	v_add_co_u32 v114, vcc_lo, v133, v114
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v115, null, v134, v115, vcc_lo
	global_load_b32 v1, v[116:117], off offset:132
	global_load_b32 v106, v[114:115], off offset:132
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v106, v1, v107
	global_store_b32 v[114:115], v106, off offset:132
.LBB0_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 34, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[106:107], 2, v[0:1]
	v_add_co_u32 v114, vcc_lo, s6, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v115, null, s7, v107, vcc_lo
	v_add_co_u32 v106, vcc_lo, v133, v106
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v107, null, v134, v107, vcc_lo
	global_load_b32 v1, v[114:115], off offset:136
	global_load_b32 v114, v[106:107], off offset:136
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v114, v1, v108
	global_store_b32 v[106:107], v114, off offset:136
.LBB0_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 35, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[106:107], 2, v[0:1]
	v_add_co_u32 v114, vcc_lo, s6, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v115, null, s7, v107, vcc_lo
	v_add_co_u32 v106, vcc_lo, v133, v106
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v107, null, v134, v107, vcc_lo
	global_load_b32 v1, v[114:115], off offset:140
	global_load_b32 v108, v[106:107], off offset:140
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v108, v1, v109
	global_store_b32 v[106:107], v108, off offset:140
.LBB0_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 36, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[106:107], 2, v[0:1]
	v_add_co_u32 v108, vcc_lo, s6, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v109, null, s7, v107, vcc_lo
	v_add_co_u32 v106, vcc_lo, v133, v106
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v107, null, v134, v107, vcc_lo
	global_load_b32 v1, v[108:109], off offset:144
	global_load_b32 v108, v[106:107], off offset:144
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v108, v1, v110
	global_store_b32 v[106:107], v108, off offset:144
.LBB0_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 37, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[106:107], 2, v[0:1]
	v_add_co_u32 v108, vcc_lo, s6, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v109, null, s7, v107, vcc_lo
	v_add_co_u32 v106, vcc_lo, v133, v106
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v107, null, v134, v107, vcc_lo
	global_load_b32 v1, v[108:109], off offset:148
	global_load_b32 v108, v[106:107], off offset:148
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v108, v1, v111
	global_store_b32 v[106:107], v108, off offset:148
.LBB0_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 38, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[106:107], 2, v[0:1]
	v_add_co_u32 v108, vcc_lo, s6, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v109, null, s7, v107, vcc_lo
	v_add_co_u32 v106, vcc_lo, v133, v106
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v107, null, v134, v107, vcc_lo
	global_load_b32 v1, v[108:109], off offset:152
	global_load_b32 v108, v[106:107], off offset:152
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v108, v1, v112
	global_store_b32 v[106:107], v108, off offset:152
.LBB0_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 39, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[106:107], 2, v[0:1]
	v_add_co_u32 v108, vcc_lo, s6, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v109, null, s7, v107, vcc_lo
	v_add_co_u32 v106, vcc_lo, v133, v106
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v107, null, v134, v107, vcc_lo
	global_load_b32 v1, v[108:109], off offset:156
	global_load_b32 v108, v[106:107], off offset:156
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v108, v1, v113
	global_store_b32 v[106:107], v108, off offset:156
.LBB0_58:                               ; %.preheader.3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 48, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[106:107], 2, v[0:1]
	v_add_co_u32 v108, vcc_lo, s6, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v109, null, s7, v107, vcc_lo
	v_add_co_u32 v106, vcc_lo, v133, v106
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v107, null, v134, v107, vcc_lo
	global_load_b32 v1, v[108:109], off offset:192
	global_load_b32 v108, v[106:107], off offset:192
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v108, v1, v98
	global_store_b32 v[106:107], v108, off offset:192
.LBB0_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 49, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[106:107], 2, v[0:1]
	v_add_co_u32 v108, vcc_lo, s6, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v109, null, s7, v107, vcc_lo
	v_add_co_u32 v106, vcc_lo, v133, v106
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v107, null, v134, v107, vcc_lo
	global_load_b32 v1, v[108:109], off offset:196
	global_load_b32 v98, v[106:107], off offset:196
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v98, v1, v99
	global_store_b32 v[106:107], v98, off offset:196
.LBB0_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 50, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[98:99], 2, v[0:1]
	v_add_co_u32 v106, vcc_lo, s6, v98
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v107, null, s7, v99, vcc_lo
	v_add_co_u32 v98, vcc_lo, v133, v98
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, v134, v99, vcc_lo
	global_load_b32 v1, v[106:107], off offset:200
	global_load_b32 v106, v[98:99], off offset:200
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v106, v1, v100
	global_store_b32 v[98:99], v106, off offset:200
.LBB0_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 51, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[98:99], 2, v[0:1]
	v_add_co_u32 v106, vcc_lo, s6, v98
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v107, null, s7, v99, vcc_lo
	v_add_co_u32 v98, vcc_lo, v133, v98
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, v134, v99, vcc_lo
	global_load_b32 v1, v[106:107], off offset:204
	global_load_b32 v100, v[98:99], off offset:204
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v100, v1, v101
	global_store_b32 v[98:99], v100, off offset:204
.LBB0_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 52, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[98:99], 2, v[0:1]
	v_add_co_u32 v100, vcc_lo, s6, v98
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v101, null, s7, v99, vcc_lo
	v_add_co_u32 v98, vcc_lo, v133, v98
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, v134, v99, vcc_lo
	global_load_b32 v1, v[100:101], off offset:208
	global_load_b32 v100, v[98:99], off offset:208
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v100, v1, v102
	global_store_b32 v[98:99], v100, off offset:208
.LBB0_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 53, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[98:99], 2, v[0:1]
	v_add_co_u32 v100, vcc_lo, s6, v98
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v101, null, s7, v99, vcc_lo
	v_add_co_u32 v98, vcc_lo, v133, v98
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, v134, v99, vcc_lo
	global_load_b32 v1, v[100:101], off offset:212
	global_load_b32 v100, v[98:99], off offset:212
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v100, v1, v103
	global_store_b32 v[98:99], v100, off offset:212
.LBB0_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 54, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[98:99], 2, v[0:1]
	v_add_co_u32 v100, vcc_lo, s6, v98
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v101, null, s7, v99, vcc_lo
	v_add_co_u32 v98, vcc_lo, v133, v98
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, v134, v99, vcc_lo
	global_load_b32 v1, v[100:101], off offset:216
	global_load_b32 v100, v[98:99], off offset:216
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v100, v1, v104
	global_store_b32 v[98:99], v100, off offset:216
.LBB0_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 55, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s8, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[98:99], 2, v[0:1]
	v_add_co_u32 v100, vcc_lo, s6, v98
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v101, null, s7, v99, vcc_lo
	v_add_co_u32 v98, vcc_lo, v133, v98
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, v134, v99, vcc_lo
	global_load_b32 v1, v[100:101], off offset:220
	global_load_b32 v100, v[98:99], off offset:220
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v132, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v100, v1, v105
	global_store_b32 v[98:99], v100, off offset:220
.LBB0_74:                               ; %Flow1332
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v1, 16, v130
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s10, v1
	s_cbranch_execz .LBB0_139
; %bb.75:                               ; %.preheader.1233
	v_lshlrev_b64_e32 v[98:99], 2, v[130:131]
	v_mul_lo_u32 v101, s13, v1
	s_mov_b32 s5, exec_lo
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_u32 v98, vcc_lo, s0, v98
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v99, null, s1, v99, vcc_lo
	global_load_b32 v98, v[98:99], off offset:64
	v_ashrrev_i32_e32 v99, 31, v1
	v_mul_lo_u32 v102, s12, v99
	v_mad_co_u64_u32 v[99:100], null, s12, v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v100, v100, v102, v101
	v_lshlrev_b64_e32 v[99:100], 2, v[99:100]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v99, vcc_lo, s2, v99
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v100, null, s3, v100, vcc_lo
	v_cmpx_gt_i32_e64 s8, v0
	s_cbranch_execz .LBB0_77
; %bb.76:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[101:102], 2, v[0:1]
	v_add_co_u32 v103, vcc_lo, s6, v101
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v104, null, s7, v102, vcc_lo
	v_add_co_u32 v101, vcc_lo, v99, v101
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v102, null, v100, v102, vcc_lo
	global_load_b32 v1, v[103:104], off
	global_load_b32 v103, v[101:102], off
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v103, v1, v90
	global_store_b32 v[101:102], v103, off
.LBB0_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 1, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_79
; %bb.78:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[101:102], 2, v[0:1]
	v_add_co_u32 v103, vcc_lo, s6, v101
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v104, null, s7, v102, vcc_lo
	v_add_co_u32 v101, vcc_lo, v99, v101
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v102, null, v100, v102, vcc_lo
	global_load_b32 v1, v[103:104], off offset:4
	global_load_b32 v90, v[101:102], off offset:4
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v90, v1, v91
	global_store_b32 v[101:102], v90, off offset:4
.LBB0_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 2, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_81
; %bb.80:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[90:91], 2, v[0:1]
	v_add_co_u32 v101, vcc_lo, s6, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v102, null, s7, v91, vcc_lo
	v_add_co_u32 v90, vcc_lo, v99, v90
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v100, v91, vcc_lo
	global_load_b32 v1, v[101:102], off offset:8
	global_load_b32 v101, v[90:91], off offset:8
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v101, v1, v92
	global_store_b32 v[90:91], v101, off offset:8
.LBB0_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 3, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_83
; %bb.82:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[90:91], 2, v[0:1]
	v_add_co_u32 v101, vcc_lo, s6, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v102, null, s7, v91, vcc_lo
	v_add_co_u32 v90, vcc_lo, v99, v90
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v100, v91, vcc_lo
	global_load_b32 v1, v[101:102], off offset:12
	global_load_b32 v92, v[90:91], off offset:12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v92, v1, v93
	global_store_b32 v[90:91], v92, off offset:12
.LBB0_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 4, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_85
; %bb.84:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[90:91], 2, v[0:1]
	v_add_co_u32 v92, vcc_lo, s6, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v93, null, s7, v91, vcc_lo
	v_add_co_u32 v90, vcc_lo, v99, v90
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v100, v91, vcc_lo
	global_load_b32 v1, v[92:93], off offset:16
	global_load_b32 v92, v[90:91], off offset:16
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v92, v1, v94
	global_store_b32 v[90:91], v92, off offset:16
.LBB0_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 5, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_87
; %bb.86:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[90:91], 2, v[0:1]
	v_add_co_u32 v92, vcc_lo, s6, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v93, null, s7, v91, vcc_lo
	v_add_co_u32 v90, vcc_lo, v99, v90
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v100, v91, vcc_lo
	global_load_b32 v1, v[92:93], off offset:20
	global_load_b32 v92, v[90:91], off offset:20
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v92, v1, v95
	global_store_b32 v[90:91], v92, off offset:20
.LBB0_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 6, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_89
; %bb.88:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[90:91], 2, v[0:1]
	v_add_co_u32 v92, vcc_lo, s6, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v93, null, s7, v91, vcc_lo
	v_add_co_u32 v90, vcc_lo, v99, v90
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v100, v91, vcc_lo
	global_load_b32 v1, v[92:93], off offset:24
	global_load_b32 v92, v[90:91], off offset:24
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v92, v1, v96
	global_store_b32 v[90:91], v92, off offset:24
.LBB0_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 7, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_91
; %bb.90:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[90:91], 2, v[0:1]
	v_add_co_u32 v92, vcc_lo, s6, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v93, null, s7, v91, vcc_lo
	v_add_co_u32 v90, vcc_lo, v99, v90
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v100, v91, vcc_lo
	global_load_b32 v1, v[92:93], off offset:28
	global_load_b32 v92, v[90:91], off offset:28
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v92, v1, v97
	global_store_b32 v[90:91], v92, off offset:28
.LBB0_91:                               ; %.preheader.1.1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 16, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_93
; %bb.92:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[90:91], 2, v[0:1]
	v_add_co_u32 v92, vcc_lo, s6, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v93, null, s7, v91, vcc_lo
	v_add_co_u32 v90, vcc_lo, v99, v90
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v100, v91, vcc_lo
	global_load_b32 v1, v[92:93], off offset:64
	global_load_b32 v92, v[90:91], off offset:64
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v92, v1, v82
	global_store_b32 v[90:91], v92, off offset:64
.LBB0_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 17, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_95
; %bb.94:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[90:91], 2, v[0:1]
	v_add_co_u32 v92, vcc_lo, s6, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v93, null, s7, v91, vcc_lo
	v_add_co_u32 v90, vcc_lo, v99, v90
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v91, null, v100, v91, vcc_lo
	global_load_b32 v1, v[92:93], off offset:68
	global_load_b32 v82, v[90:91], off offset:68
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v82, v1, v83
	global_store_b32 v[90:91], v82, off offset:68
.LBB0_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 18, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_97
; %bb.96:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[82:83], 2, v[0:1]
	v_add_co_u32 v90, vcc_lo, s6, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v91, null, s7, v83, vcc_lo
	v_add_co_u32 v82, vcc_lo, v99, v82
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, v100, v83, vcc_lo
	global_load_b32 v1, v[90:91], off offset:72
	global_load_b32 v90, v[82:83], off offset:72
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v90, v1, v84
	global_store_b32 v[82:83], v90, off offset:72
.LBB0_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 19, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_99
; %bb.98:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[82:83], 2, v[0:1]
	v_add_co_u32 v90, vcc_lo, s6, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v91, null, s7, v83, vcc_lo
	v_add_co_u32 v82, vcc_lo, v99, v82
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, v100, v83, vcc_lo
	global_load_b32 v1, v[90:91], off offset:76
	global_load_b32 v84, v[82:83], off offset:76
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v84, v1, v85
	global_store_b32 v[82:83], v84, off offset:76
.LBB0_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 20, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_101
; %bb.100:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[82:83], 2, v[0:1]
	v_add_co_u32 v84, vcc_lo, s6, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v85, null, s7, v83, vcc_lo
	v_add_co_u32 v82, vcc_lo, v99, v82
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, v100, v83, vcc_lo
	global_load_b32 v1, v[84:85], off offset:80
	global_load_b32 v84, v[82:83], off offset:80
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v84, v1, v86
	global_store_b32 v[82:83], v84, off offset:80
.LBB0_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 21, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_103
; %bb.102:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[82:83], 2, v[0:1]
	v_add_co_u32 v84, vcc_lo, s6, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v85, null, s7, v83, vcc_lo
	v_add_co_u32 v82, vcc_lo, v99, v82
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, v100, v83, vcc_lo
	global_load_b32 v1, v[84:85], off offset:84
	global_load_b32 v84, v[82:83], off offset:84
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v84, v1, v87
	global_store_b32 v[82:83], v84, off offset:84
.LBB0_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 22, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_105
; %bb.104:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[82:83], 2, v[0:1]
	v_add_co_u32 v84, vcc_lo, s6, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v85, null, s7, v83, vcc_lo
	v_add_co_u32 v82, vcc_lo, v99, v82
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, v100, v83, vcc_lo
	global_load_b32 v1, v[84:85], off offset:88
	global_load_b32 v84, v[82:83], off offset:88
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v84, v1, v88
	global_store_b32 v[82:83], v84, off offset:88
.LBB0_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 23, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_107
; %bb.106:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[82:83], 2, v[0:1]
	v_add_co_u32 v84, vcc_lo, s6, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v85, null, s7, v83, vcc_lo
	v_add_co_u32 v82, vcc_lo, v99, v82
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, v100, v83, vcc_lo
	global_load_b32 v1, v[84:85], off offset:92
	global_load_b32 v84, v[82:83], off offset:92
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v84, v1, v89
	global_store_b32 v[82:83], v84, off offset:92
.LBB0_107:                              ; %.preheader.2.1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 32, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_109
; %bb.108:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[82:83], 2, v[0:1]
	v_add_co_u32 v84, vcc_lo, s6, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v85, null, s7, v83, vcc_lo
	v_add_co_u32 v82, vcc_lo, v99, v82
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, v100, v83, vcc_lo
	global_load_b32 v1, v[84:85], off offset:128
	global_load_b32 v84, v[82:83], off offset:128
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v84, v1, v74
	global_store_b32 v[82:83], v84, off offset:128
.LBB0_109:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 33, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_111
; %bb.110:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[82:83], 2, v[0:1]
	v_add_co_u32 v84, vcc_lo, s6, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v85, null, s7, v83, vcc_lo
	v_add_co_u32 v82, vcc_lo, v99, v82
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, v100, v83, vcc_lo
	global_load_b32 v1, v[84:85], off offset:132
	global_load_b32 v74, v[82:83], off offset:132
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v74, v1, v75
	global_store_b32 v[82:83], v74, off offset:132
.LBB0_111:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 34, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_113
; %bb.112:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[74:75], 2, v[0:1]
	v_add_co_u32 v82, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v83, null, s7, v75, vcc_lo
	v_add_co_u32 v74, vcc_lo, v99, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v100, v75, vcc_lo
	global_load_b32 v1, v[82:83], off offset:136
	global_load_b32 v82, v[74:75], off offset:136
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v82, v1, v76
	global_store_b32 v[74:75], v82, off offset:136
.LBB0_113:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 35, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_115
; %bb.114:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[74:75], 2, v[0:1]
	v_add_co_u32 v82, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v83, null, s7, v75, vcc_lo
	v_add_co_u32 v74, vcc_lo, v99, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v100, v75, vcc_lo
	global_load_b32 v1, v[82:83], off offset:140
	global_load_b32 v76, v[74:75], off offset:140
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v76, v1, v77
	global_store_b32 v[74:75], v76, off offset:140
.LBB0_115:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 36, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_117
; %bb.116:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[74:75], 2, v[0:1]
	v_add_co_u32 v76, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, s7, v75, vcc_lo
	v_add_co_u32 v74, vcc_lo, v99, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v100, v75, vcc_lo
	global_load_b32 v1, v[76:77], off offset:144
	global_load_b32 v76, v[74:75], off offset:144
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v76, v1, v78
	global_store_b32 v[74:75], v76, off offset:144
.LBB0_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 37, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_119
; %bb.118:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[74:75], 2, v[0:1]
	v_add_co_u32 v76, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, s7, v75, vcc_lo
	v_add_co_u32 v74, vcc_lo, v99, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v100, v75, vcc_lo
	global_load_b32 v1, v[76:77], off offset:148
	global_load_b32 v76, v[74:75], off offset:148
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v76, v1, v79
	global_store_b32 v[74:75], v76, off offset:148
.LBB0_119:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 38, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_121
; %bb.120:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[74:75], 2, v[0:1]
	v_add_co_u32 v76, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, s7, v75, vcc_lo
	v_add_co_u32 v74, vcc_lo, v99, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v100, v75, vcc_lo
	global_load_b32 v1, v[76:77], off offset:152
	global_load_b32 v76, v[74:75], off offset:152
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v76, v1, v80
	global_store_b32 v[74:75], v76, off offset:152
.LBB0_121:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 39, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_123
; %bb.122:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[74:75], 2, v[0:1]
	v_add_co_u32 v76, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, s7, v75, vcc_lo
	v_add_co_u32 v74, vcc_lo, v99, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v100, v75, vcc_lo
	global_load_b32 v1, v[76:77], off offset:156
	global_load_b32 v76, v[74:75], off offset:156
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v76, v1, v81
	global_store_b32 v[74:75], v76, off offset:156
.LBB0_123:                              ; %.preheader.3.1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 48, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_125
; %bb.124:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[74:75], 2, v[0:1]
	v_add_co_u32 v76, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, s7, v75, vcc_lo
	v_add_co_u32 v74, vcc_lo, v99, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v100, v75, vcc_lo
	global_load_b32 v1, v[76:77], off offset:192
	global_load_b32 v76, v[74:75], off offset:192
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v76, v1, v66
	global_store_b32 v[74:75], v76, off offset:192
.LBB0_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 49, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_127
; %bb.126:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[74:75], 2, v[0:1]
	v_add_co_u32 v76, vcc_lo, s6, v74
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v77, null, s7, v75, vcc_lo
	v_add_co_u32 v74, vcc_lo, v99, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v100, v75, vcc_lo
	global_load_b32 v1, v[76:77], off offset:196
	global_load_b32 v66, v[74:75], off offset:196
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v66, v1, v67
	global_store_b32 v[74:75], v66, off offset:196
.LBB0_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 50, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_129
; %bb.128:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[66:67], 2, v[0:1]
	v_add_co_u32 v74, vcc_lo, s6, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v75, null, s7, v67, vcc_lo
	v_add_co_u32 v66, vcc_lo, v99, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, v100, v67, vcc_lo
	global_load_b32 v1, v[74:75], off offset:200
	global_load_b32 v74, v[66:67], off offset:200
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v74, v1, v68
	global_store_b32 v[66:67], v74, off offset:200
.LBB0_129:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 51, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_131
; %bb.130:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[66:67], 2, v[0:1]
	v_add_co_u32 v74, vcc_lo, s6, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v75, null, s7, v67, vcc_lo
	v_add_co_u32 v66, vcc_lo, v99, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, v100, v67, vcc_lo
	global_load_b32 v1, v[74:75], off offset:204
	global_load_b32 v68, v[66:67], off offset:204
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v68, v1, v69
	global_store_b32 v[66:67], v68, off offset:204
.LBB0_131:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 52, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_133
; %bb.132:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[66:67], 2, v[0:1]
	v_add_co_u32 v68, vcc_lo, s6, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v69, null, s7, v67, vcc_lo
	v_add_co_u32 v66, vcc_lo, v99, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, v100, v67, vcc_lo
	global_load_b32 v1, v[68:69], off offset:208
	global_load_b32 v68, v[66:67], off offset:208
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v68, v1, v70
	global_store_b32 v[66:67], v68, off offset:208
.LBB0_133:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 53, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_135
; %bb.134:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[66:67], 2, v[0:1]
	v_add_co_u32 v68, vcc_lo, s6, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v69, null, s7, v67, vcc_lo
	v_add_co_u32 v66, vcc_lo, v99, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, v100, v67, vcc_lo
	global_load_b32 v1, v[68:69], off offset:212
	global_load_b32 v68, v[66:67], off offset:212
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v68, v1, v71
	global_store_b32 v[66:67], v68, off offset:212
.LBB0_135:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 54, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_137
; %bb.136:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[66:67], 2, v[0:1]
	v_add_co_u32 v68, vcc_lo, s6, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v69, null, s7, v67, vcc_lo
	v_add_co_u32 v66, vcc_lo, v99, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, v100, v67, vcc_lo
	global_load_b32 v1, v[68:69], off offset:216
	global_load_b32 v68, v[66:67], off offset:216
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v68, v1, v72
	global_store_b32 v[66:67], v68, off offset:216
.LBB0_137:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 55, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s8, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_139
; %bb.138:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[66:67], 2, v[0:1]
	v_add_co_u32 v68, vcc_lo, s6, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v69, null, s7, v67, vcc_lo
	v_add_co_u32 v66, vcc_lo, v99, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, v100, v67, vcc_lo
	global_load_b32 v1, v[68:69], off offset:220
	global_load_b32 v68, v[66:67], off offset:220
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v98, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v68, v1, v73
	global_store_b32 v[66:67], v68, off offset:220
.LBB0_139:                              ; %Flow1330
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v1, 32, v130
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s10, v1
	s_cbranch_execz .LBB0_204
; %bb.140:                              ; %.preheader.2234
	v_lshlrev_b64_e32 v[66:67], 2, v[130:131]
	v_mul_lo_u32 v69, s13, v1
	s_mov_b32 s5, exec_lo
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, vcc_lo, s0, v66
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v67, null, s1, v67, vcc_lo
	global_load_b32 v66, v[66:67], off offset:128
	v_ashrrev_i32_e32 v67, 31, v1
	v_mul_lo_u32 v70, s12, v67
	v_mad_co_u64_u32 v[67:68], null, s12, v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v68, v68, v70, v69
	v_lshlrev_b64_e32 v[67:68], 2, v[67:68]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v67, vcc_lo, s2, v67
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, s3, v68, vcc_lo
	v_cmpx_gt_i32_e64 s8, v0
	s_cbranch_execz .LBB0_142
; %bb.141:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[69:70], 2, v[0:1]
	v_add_co_u32 v71, vcc_lo, s6, v69
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v72, null, s7, v70, vcc_lo
	v_add_co_u32 v69, vcc_lo, v67, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, v68, v70, vcc_lo
	global_load_b32 v1, v[71:72], off
	global_load_b32 v71, v[69:70], off
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v71, v1, v58
	global_store_b32 v[69:70], v71, off
.LBB0_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 1, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_144
; %bb.143:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[69:70], 2, v[0:1]
	v_add_co_u32 v71, vcc_lo, s6, v69
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v72, null, s7, v70, vcc_lo
	v_add_co_u32 v69, vcc_lo, v67, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, v68, v70, vcc_lo
	global_load_b32 v1, v[71:72], off offset:4
	global_load_b32 v58, v[69:70], off offset:4
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v58, v1, v59
	global_store_b32 v[69:70], v58, off offset:4
.LBB0_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 2, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_146
; %bb.145:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_co_u32 v69, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v70, null, s7, v59, vcc_lo
	v_add_co_u32 v58, vcc_lo, v67, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v68, v59, vcc_lo
	global_load_b32 v1, v[69:70], off offset:8
	global_load_b32 v69, v[58:59], off offset:8
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v69, v1, v60
	global_store_b32 v[58:59], v69, off offset:8
.LBB0_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 3, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_148
; %bb.147:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_co_u32 v69, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v70, null, s7, v59, vcc_lo
	v_add_co_u32 v58, vcc_lo, v67, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v68, v59, vcc_lo
	global_load_b32 v1, v[69:70], off offset:12
	global_load_b32 v60, v[58:59], off offset:12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v60, v1, v61
	global_store_b32 v[58:59], v60, off offset:12
.LBB0_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 4, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_150
; %bb.149:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_co_u32 v60, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, s7, v59, vcc_lo
	v_add_co_u32 v58, vcc_lo, v67, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v68, v59, vcc_lo
	global_load_b32 v1, v[60:61], off offset:16
	global_load_b32 v60, v[58:59], off offset:16
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v60, v1, v62
	global_store_b32 v[58:59], v60, off offset:16
.LBB0_150:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 5, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_152
; %bb.151:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_co_u32 v60, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, s7, v59, vcc_lo
	v_add_co_u32 v58, vcc_lo, v67, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v68, v59, vcc_lo
	global_load_b32 v1, v[60:61], off offset:20
	global_load_b32 v60, v[58:59], off offset:20
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v60, v1, v63
	global_store_b32 v[58:59], v60, off offset:20
.LBB0_152:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 6, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_154
; %bb.153:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_co_u32 v60, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, s7, v59, vcc_lo
	v_add_co_u32 v58, vcc_lo, v67, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v68, v59, vcc_lo
	global_load_b32 v1, v[60:61], off offset:24
	global_load_b32 v60, v[58:59], off offset:24
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v60, v1, v64
	global_store_b32 v[58:59], v60, off offset:24
.LBB0_154:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 7, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_156
; %bb.155:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_co_u32 v60, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, s7, v59, vcc_lo
	v_add_co_u32 v58, vcc_lo, v67, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v68, v59, vcc_lo
	global_load_b32 v1, v[60:61], off offset:28
	global_load_b32 v60, v[58:59], off offset:28
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v60, v1, v65
	global_store_b32 v[58:59], v60, off offset:28
.LBB0_156:                              ; %.preheader.1.2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 16, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_158
; %bb.157:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_co_u32 v60, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, s7, v59, vcc_lo
	v_add_co_u32 v58, vcc_lo, v67, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v68, v59, vcc_lo
	global_load_b32 v1, v[60:61], off offset:64
	global_load_b32 v60, v[58:59], off offset:64
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v60, v1, v50
	global_store_b32 v[58:59], v60, off offset:64
.LBB0_158:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 17, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_160
; %bb.159:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[0:1]
	v_add_co_u32 v60, vcc_lo, s6, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v61, null, s7, v59, vcc_lo
	v_add_co_u32 v58, vcc_lo, v67, v58
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, v68, v59, vcc_lo
	global_load_b32 v1, v[60:61], off offset:68
	global_load_b32 v50, v[58:59], off offset:68
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v50, v1, v51
	global_store_b32 v[58:59], v50, off offset:68
.LBB0_160:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 18, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_162
; %bb.161:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[0:1]
	v_add_co_u32 v58, vcc_lo, s6, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, s7, v51, vcc_lo
	v_add_co_u32 v50, vcc_lo, v67, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v68, v51, vcc_lo
	global_load_b32 v1, v[58:59], off offset:72
	global_load_b32 v58, v[50:51], off offset:72
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v58, v1, v52
	global_store_b32 v[50:51], v58, off offset:72
.LBB0_162:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 19, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_164
; %bb.163:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[0:1]
	v_add_co_u32 v58, vcc_lo, s6, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, s7, v51, vcc_lo
	v_add_co_u32 v50, vcc_lo, v67, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v68, v51, vcc_lo
	global_load_b32 v1, v[58:59], off offset:76
	global_load_b32 v52, v[50:51], off offset:76
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v52, v1, v53
	global_store_b32 v[50:51], v52, off offset:76
.LBB0_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 20, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_166
; %bb.165:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[0:1]
	v_add_co_u32 v52, vcc_lo, s6, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v53, null, s7, v51, vcc_lo
	v_add_co_u32 v50, vcc_lo, v67, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v68, v51, vcc_lo
	global_load_b32 v1, v[52:53], off offset:80
	global_load_b32 v52, v[50:51], off offset:80
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v52, v1, v54
	global_store_b32 v[50:51], v52, off offset:80
.LBB0_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 21, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_168
; %bb.167:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[0:1]
	v_add_co_u32 v52, vcc_lo, s6, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v53, null, s7, v51, vcc_lo
	v_add_co_u32 v50, vcc_lo, v67, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v68, v51, vcc_lo
	global_load_b32 v1, v[52:53], off offset:84
	global_load_b32 v52, v[50:51], off offset:84
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v52, v1, v55
	global_store_b32 v[50:51], v52, off offset:84
.LBB0_168:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 22, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_170
; %bb.169:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[0:1]
	v_add_co_u32 v52, vcc_lo, s6, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v53, null, s7, v51, vcc_lo
	v_add_co_u32 v50, vcc_lo, v67, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v68, v51, vcc_lo
	global_load_b32 v1, v[52:53], off offset:88
	global_load_b32 v52, v[50:51], off offset:88
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v52, v1, v56
	global_store_b32 v[50:51], v52, off offset:88
.LBB0_170:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 23, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_172
; %bb.171:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[0:1]
	v_add_co_u32 v52, vcc_lo, s6, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v53, null, s7, v51, vcc_lo
	v_add_co_u32 v50, vcc_lo, v67, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v68, v51, vcc_lo
	global_load_b32 v1, v[52:53], off offset:92
	global_load_b32 v52, v[50:51], off offset:92
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v52, v1, v57
	global_store_b32 v[50:51], v52, off offset:92
.LBB0_172:                              ; %.preheader.2.2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 32, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_174
; %bb.173:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[0:1]
	v_add_co_u32 v52, vcc_lo, s6, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v53, null, s7, v51, vcc_lo
	v_add_co_u32 v50, vcc_lo, v67, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v68, v51, vcc_lo
	global_load_b32 v1, v[52:53], off offset:128
	global_load_b32 v52, v[50:51], off offset:128
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v52, v1, v42
	global_store_b32 v[50:51], v52, off offset:128
.LBB0_174:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 33, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_176
; %bb.175:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[0:1]
	v_add_co_u32 v52, vcc_lo, s6, v50
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v53, null, s7, v51, vcc_lo
	v_add_co_u32 v50, vcc_lo, v67, v50
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v51, null, v68, v51, vcc_lo
	global_load_b32 v1, v[52:53], off offset:132
	global_load_b32 v42, v[50:51], off offset:132
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v1, v43
	global_store_b32 v[50:51], v42, off offset:132
.LBB0_176:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 34, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_178
; %bb.177:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]
	v_add_co_u32 v50, vcc_lo, s6, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v51, null, s7, v43, vcc_lo
	v_add_co_u32 v42, vcc_lo, v67, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v68, v43, vcc_lo
	global_load_b32 v1, v[50:51], off offset:136
	global_load_b32 v50, v[42:43], off offset:136
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v50, v1, v44
	global_store_b32 v[42:43], v50, off offset:136
.LBB0_178:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 35, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_180
; %bb.179:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]
	v_add_co_u32 v50, vcc_lo, s6, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v51, null, s7, v43, vcc_lo
	v_add_co_u32 v42, vcc_lo, v67, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v68, v43, vcc_lo
	global_load_b32 v1, v[50:51], off offset:140
	global_load_b32 v44, v[42:43], off offset:140
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v1, v45
	global_store_b32 v[42:43], v44, off offset:140
.LBB0_180:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 36, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_182
; %bb.181:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]
	v_add_co_u32 v44, vcc_lo, s6, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, s7, v43, vcc_lo
	v_add_co_u32 v42, vcc_lo, v67, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v68, v43, vcc_lo
	global_load_b32 v1, v[44:45], off offset:144
	global_load_b32 v44, v[42:43], off offset:144
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v1, v46
	global_store_b32 v[42:43], v44, off offset:144
.LBB0_182:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 37, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_184
; %bb.183:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]
	v_add_co_u32 v44, vcc_lo, s6, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, s7, v43, vcc_lo
	v_add_co_u32 v42, vcc_lo, v67, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v68, v43, vcc_lo
	global_load_b32 v1, v[44:45], off offset:148
	global_load_b32 v44, v[42:43], off offset:148
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v1, v47
	global_store_b32 v[42:43], v44, off offset:148
.LBB0_184:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 38, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_186
; %bb.185:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]
	v_add_co_u32 v44, vcc_lo, s6, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, s7, v43, vcc_lo
	v_add_co_u32 v42, vcc_lo, v67, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v68, v43, vcc_lo
	global_load_b32 v1, v[44:45], off offset:152
	global_load_b32 v44, v[42:43], off offset:152
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v1, v48
	global_store_b32 v[42:43], v44, off offset:152
.LBB0_186:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 39, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_188
; %bb.187:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]
	v_add_co_u32 v44, vcc_lo, s6, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, s7, v43, vcc_lo
	v_add_co_u32 v42, vcc_lo, v67, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v68, v43, vcc_lo
	global_load_b32 v1, v[44:45], off offset:156
	global_load_b32 v44, v[42:43], off offset:156
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v1, v49
	global_store_b32 v[42:43], v44, off offset:156
.LBB0_188:                              ; %.preheader.3.2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 48, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_190
; %bb.189:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]
	v_add_co_u32 v44, vcc_lo, s6, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, s7, v43, vcc_lo
	v_add_co_u32 v42, vcc_lo, v67, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v68, v43, vcc_lo
	global_load_b32 v1, v[44:45], off offset:192
	global_load_b32 v44, v[42:43], off offset:192
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v44, v1, v34
	global_store_b32 v[42:43], v44, off offset:192
.LBB0_190:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 49, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_192
; %bb.191:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]
	v_add_co_u32 v44, vcc_lo, s6, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v45, null, s7, v43, vcc_lo
	v_add_co_u32 v42, vcc_lo, v67, v42
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v43, null, v68, v43, vcc_lo
	global_load_b32 v1, v[44:45], off offset:196
	global_load_b32 v34, v[42:43], off offset:196
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v34, v1, v35
	global_store_b32 v[42:43], v34, off offset:196
.LBB0_192:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 50, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_194
; %bb.193:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]
	v_add_co_u32 v42, vcc_lo, s6, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, s7, v35, vcc_lo
	v_add_co_u32 v34, vcc_lo, v67, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v68, v35, vcc_lo
	global_load_b32 v1, v[42:43], off offset:200
	global_load_b32 v42, v[34:35], off offset:200
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v42, v1, v36
	global_store_b32 v[34:35], v42, off offset:200
.LBB0_194:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 51, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_196
; %bb.195:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]
	v_add_co_u32 v42, vcc_lo, s6, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, s7, v35, vcc_lo
	v_add_co_u32 v34, vcc_lo, v67, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v68, v35, vcc_lo
	global_load_b32 v1, v[42:43], off offset:204
	global_load_b32 v36, v[34:35], off offset:204
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v36, v1, v37
	global_store_b32 v[34:35], v36, off offset:204
.LBB0_196:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 52, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_198
; %bb.197:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]
	v_add_co_u32 v36, vcc_lo, s6, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v37, null, s7, v35, vcc_lo
	v_add_co_u32 v34, vcc_lo, v67, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v68, v35, vcc_lo
	global_load_b32 v1, v[36:37], off offset:208
	global_load_b32 v36, v[34:35], off offset:208
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v36, v1, v38
	global_store_b32 v[34:35], v36, off offset:208
.LBB0_198:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 53, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_200
; %bb.199:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]
	v_add_co_u32 v36, vcc_lo, s6, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v37, null, s7, v35, vcc_lo
	v_add_co_u32 v34, vcc_lo, v67, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v68, v35, vcc_lo
	global_load_b32 v1, v[36:37], off offset:212
	global_load_b32 v36, v[34:35], off offset:212
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v36, v1, v39
	global_store_b32 v[34:35], v36, off offset:212
.LBB0_200:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 54, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v1
	s_cbranch_execz .LBB0_202
; %bb.201:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]
	v_add_co_u32 v36, vcc_lo, s6, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v37, null, s7, v35, vcc_lo
	v_add_co_u32 v34, vcc_lo, v67, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v68, v35, vcc_lo
	global_load_b32 v1, v[36:37], off offset:216
	global_load_b32 v36, v[34:35], off offset:216
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v36, v1, v40
	global_store_b32 v[34:35], v36, off offset:216
.LBB0_202:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v1, 55, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s8, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_204
; %bb.203:
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]
	v_add_co_u32 v36, vcc_lo, s6, v34
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v37, null, s7, v35, vcc_lo
	v_add_co_u32 v34, vcc_lo, v67, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, v68, v35, vcc_lo
	global_load_b32 v1, v[36:37], off offset:220
	global_load_b32 v36, v[34:35], off offset:220
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v66, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v36, v1, v41
	global_store_b32 v[34:35], v36, off offset:220
.LBB0_204:                              ; %Flow1328
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v1, 48, v130
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s10, v1
	s_cbranch_execz .LBB0_269
; %bb.205:                              ; %.preheader.3235
	v_lshlrev_b64_e32 v[34:35], 2, v[130:131]
	v_mul_lo_u32 v37, s13, v1
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v34, vcc_lo, s0, v34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v35, null, s1, v35, vcc_lo
	s_mov_b32 s0, exec_lo
	global_load_b32 v34, v[34:35], off offset:192
	v_ashrrev_i32_e32 v35, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v38, s12, v35
	v_mad_co_u64_u32 v[35:36], null, s12, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	v_add3_u32 v36, v36, v38, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, vcc_lo, s2, v35
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s3, v36, vcc_lo
	v_cmpx_gt_i32_e64 s8, v0
	s_cbranch_execz .LBB0_207
; %bb.206:
	v_lshlrev_b64_e32 v[37:38], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v39, vcc_lo, s6, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, s7, v38, vcc_lo
	v_add_co_u32 v37, vcc_lo, v35, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v38, null, v36, v38, vcc_lo
	global_load_b32 v39, v[39:40], off
	global_load_b32 v40, v[37:38], off
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v39, v34, v39
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v40, v39, v26
	global_store_b32 v[37:38], v40, off
.LBB0_207:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 1, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v26
	s_cbranch_execz .LBB0_209
; %bb.208:
	v_lshlrev_b64_e32 v[37:38], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v39, vcc_lo, s6, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v40, null, s7, v38, vcc_lo
	v_add_co_u32 v37, vcc_lo, v35, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v38, null, v36, v38, vcc_lo
	global_load_b32 v26, v[39:40], off offset:4
	global_load_b32 v39, v[37:38], off offset:4
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v26, v34, v26
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v39, v26, v27
	global_store_b32 v[37:38], v39, off offset:4
.LBB0_209:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 2, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v26
	s_cbranch_execz .LBB0_211
; %bb.210:
	v_lshlrev_b64_e32 v[26:27], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v37, vcc_lo, s6, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v38, null, s7, v27, vcc_lo
	v_add_co_u32 v26, vcc_lo, v35, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v36, v27, vcc_lo
	global_load_b32 v37, v[37:38], off offset:8
	global_load_b32 v38, v[26:27], off offset:8
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v37, v34, v37
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v38, v37, v28
	global_store_b32 v[26:27], v38, off offset:8
.LBB0_211:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 3, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v26
	s_cbranch_execz .LBB0_213
; %bb.212:
	v_lshlrev_b64_e32 v[26:27], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v37, vcc_lo, s6, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v38, null, s7, v27, vcc_lo
	v_add_co_u32 v26, vcc_lo, v35, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v36, v27, vcc_lo
	global_load_b32 v28, v[37:38], off offset:12
	global_load_b32 v37, v[26:27], off offset:12
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v28, v34, v28
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v37, v28, v29
	global_store_b32 v[26:27], v37, off offset:12
.LBB0_213:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 4, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v26
	s_cbranch_execz .LBB0_215
; %bb.214:
	v_lshlrev_b64_e32 v[26:27], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, s6, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v27, vcc_lo
	v_add_co_u32 v26, vcc_lo, v35, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v36, v27, vcc_lo
	global_load_b32 v28, v[28:29], off offset:16
	global_load_b32 v29, v[26:27], off offset:16
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v28, v34, v28
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v29, v28, v30
	global_store_b32 v[26:27], v29, off offset:16
.LBB0_215:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 5, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v26
	s_cbranch_execz .LBB0_217
; %bb.216:
	v_lshlrev_b64_e32 v[26:27], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, s6, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v27, vcc_lo
	v_add_co_u32 v26, vcc_lo, v35, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v36, v27, vcc_lo
	global_load_b32 v28, v[28:29], off offset:20
	global_load_b32 v29, v[26:27], off offset:20
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v28, v34, v28
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v29, v28, v31
	global_store_b32 v[26:27], v29, off offset:20
.LBB0_217:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 6, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v26
	s_cbranch_execz .LBB0_219
; %bb.218:
	v_lshlrev_b64_e32 v[26:27], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, s6, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v27, vcc_lo
	v_add_co_u32 v26, vcc_lo, v35, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v36, v27, vcc_lo
	global_load_b32 v28, v[28:29], off offset:24
	global_load_b32 v29, v[26:27], off offset:24
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v28, v34, v28
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v29, v28, v32
	global_store_b32 v[26:27], v29, off offset:24
.LBB0_219:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 7, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v26
	s_cbranch_execz .LBB0_221
; %bb.220:
	v_lshlrev_b64_e32 v[26:27], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, s6, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v27, vcc_lo
	v_add_co_u32 v26, vcc_lo, v35, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v36, v27, vcc_lo
	global_load_b32 v28, v[28:29], off offset:28
	global_load_b32 v29, v[26:27], off offset:28
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v28, v34, v28
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v29, v28, v33
	global_store_b32 v[26:27], v29, off offset:28
.LBB0_221:                              ; %.preheader.1.3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v26, 16, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v26
	s_cbranch_execz .LBB0_223
; %bb.222:
	v_lshlrev_b64_e32 v[26:27], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, s6, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v27, vcc_lo
	v_add_co_u32 v26, vcc_lo, v35, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v36, v27, vcc_lo
	global_load_b32 v28, v[28:29], off offset:64
	global_load_b32 v29, v[26:27], off offset:64
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v28, v34, v28
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v29, v28, v18
	global_store_b32 v[26:27], v29, off offset:64
.LBB0_223:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 17, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v18
	s_cbranch_execz .LBB0_225
; %bb.224:
	v_lshlrev_b64_e32 v[26:27], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v28, vcc_lo, s6, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v29, null, s7, v27, vcc_lo
	v_add_co_u32 v26, vcc_lo, v35, v26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, v36, v27, vcc_lo
	global_load_b32 v18, v[28:29], off offset:68
	global_load_b32 v28, v[26:27], off offset:68
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v18, v34, v18
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v18, v19
	global_store_b32 v[26:27], v28, off offset:68
.LBB0_225:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 18, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v18
	s_cbranch_execz .LBB0_227
; %bb.226:
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s7, v19, vcc_lo
	v_add_co_u32 v18, vcc_lo, v35, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v36, v19, vcc_lo
	global_load_b32 v26, v[26:27], off offset:72
	global_load_b32 v27, v[18:19], off offset:72
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v26, v34, v26
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, v26, v20
	global_store_b32 v[18:19], v27, off offset:72
.LBB0_227:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 19, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v18
	s_cbranch_execz .LBB0_229
; %bb.228:
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v27, null, s7, v19, vcc_lo
	v_add_co_u32 v18, vcc_lo, v35, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v36, v19, vcc_lo
	global_load_b32 v20, v[26:27], off offset:76
	global_load_b32 v26, v[18:19], off offset:76
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v20, v34, v20
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v26, v20, v21
	global_store_b32 v[18:19], v26, off offset:76
.LBB0_229:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 20, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v18
	s_cbranch_execz .LBB0_231
; %bb.230:
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, s6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s7, v19, vcc_lo
	v_add_co_u32 v18, vcc_lo, v35, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v36, v19, vcc_lo
	global_load_b32 v20, v[20:21], off offset:80
	global_load_b32 v21, v[18:19], off offset:80
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v20, v34, v20
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v21, v20, v22
	global_store_b32 v[18:19], v21, off offset:80
.LBB0_231:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 21, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v18
	s_cbranch_execz .LBB0_233
; %bb.232:
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, s6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s7, v19, vcc_lo
	v_add_co_u32 v18, vcc_lo, v35, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v36, v19, vcc_lo
	global_load_b32 v20, v[20:21], off offset:84
	global_load_b32 v21, v[18:19], off offset:84
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v20, v34, v20
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v21, v20, v23
	global_store_b32 v[18:19], v21, off offset:84
.LBB0_233:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 22, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v18
	s_cbranch_execz .LBB0_235
; %bb.234:
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, s6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s7, v19, vcc_lo
	v_add_co_u32 v18, vcc_lo, v35, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v36, v19, vcc_lo
	global_load_b32 v20, v[20:21], off offset:88
	global_load_b32 v21, v[18:19], off offset:88
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v20, v34, v20
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v21, v20, v24
	global_store_b32 v[18:19], v21, off offset:88
.LBB0_235:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 23, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v18
	s_cbranch_execz .LBB0_237
; %bb.236:
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, s6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s7, v19, vcc_lo
	v_add_co_u32 v18, vcc_lo, v35, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v36, v19, vcc_lo
	global_load_b32 v20, v[20:21], off offset:92
	global_load_b32 v21, v[18:19], off offset:92
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v20, v34, v20
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v21, v20, v25
	global_store_b32 v[18:19], v21, off offset:92
.LBB0_237:                              ; %.preheader.2.3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 32, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v18
	s_cbranch_execz .LBB0_239
; %bb.238:
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, s6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s7, v19, vcc_lo
	v_add_co_u32 v18, vcc_lo, v35, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v36, v19, vcc_lo
	global_load_b32 v20, v[20:21], off offset:128
	global_load_b32 v21, v[18:19], off offset:128
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v20, v34, v20
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v21, v20, v10
	global_store_b32 v[18:19], v21, off offset:128
.LBB0_239:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 33, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v10
	s_cbranch_execz .LBB0_241
; %bb.240:
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, s6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, s7, v19, vcc_lo
	v_add_co_u32 v18, vcc_lo, v35, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v36, v19, vcc_lo
	global_load_b32 v10, v[20:21], off offset:132
	global_load_b32 v20, v[18:19], off offset:132
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v10, v34, v10
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v20, v10, v11
	global_store_b32 v[18:19], v20, off offset:132
.LBB0_241:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 34, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v10
	s_cbranch_execz .LBB0_243
; %bb.242:
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc_lo, s6, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s7, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v35, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v36, v11, vcc_lo
	global_load_b32 v18, v[18:19], off offset:136
	global_load_b32 v19, v[10:11], off offset:136
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v18, v34, v18
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v19, v18, v12
	global_store_b32 v[10:11], v19, off offset:136
.LBB0_243:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 35, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v10
	s_cbranch_execz .LBB0_245
; %bb.244:
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc_lo, s6, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s7, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v35, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v36, v11, vcc_lo
	global_load_b32 v12, v[18:19], off offset:140
	global_load_b32 v18, v[10:11], off offset:140
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, v34, v12
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v18, v12, v13
	global_store_b32 v[10:11], v18, off offset:140
.LBB0_245:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 36, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v10
	s_cbranch_execz .LBB0_247
; %bb.246:
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, s6, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s7, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v35, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v36, v11, vcc_lo
	global_load_b32 v12, v[12:13], off offset:144
	global_load_b32 v13, v[10:11], off offset:144
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, v34, v12
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v12, v14
	global_store_b32 v[10:11], v13, off offset:144
.LBB0_247:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 37, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v10
	s_cbranch_execz .LBB0_249
; %bb.248:
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, s6, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s7, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v35, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v36, v11, vcc_lo
	global_load_b32 v12, v[12:13], off offset:148
	global_load_b32 v13, v[10:11], off offset:148
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, v34, v12
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v12, v15
	global_store_b32 v[10:11], v13, off offset:148
.LBB0_249:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 38, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v10
	s_cbranch_execz .LBB0_251
; %bb.250:
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, s6, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s7, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v35, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v36, v11, vcc_lo
	global_load_b32 v12, v[12:13], off offset:152
	global_load_b32 v13, v[10:11], off offset:152
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, v34, v12
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v12, v16
	global_store_b32 v[10:11], v13, off offset:152
.LBB0_251:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 39, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v10
	s_cbranch_execz .LBB0_253
; %bb.252:
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, s6, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s7, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v35, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v36, v11, vcc_lo
	global_load_b32 v12, v[12:13], off offset:156
	global_load_b32 v13, v[10:11], off offset:156
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, v34, v12
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v12, v17
	global_store_b32 v[10:11], v13, off offset:156
.LBB0_253:                              ; %.preheader.3.3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 48, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v10
	s_cbranch_execz .LBB0_255
; %bb.254:
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, s6, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s7, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v35, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v36, v11, vcc_lo
	global_load_b32 v12, v[12:13], off offset:192
	global_load_b32 v13, v[10:11], off offset:192
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v12, v34, v12
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, v12, v2
	global_store_b32 v[10:11], v13, off offset:192
.LBB0_255:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 49, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v2
	s_cbranch_execz .LBB0_257
; %bb.256:
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, s6, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s7, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v35, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v36, v11, vcc_lo
	global_load_b32 v2, v[12:13], off offset:196
	global_load_b32 v12, v[10:11], off offset:196
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v2, v34, v2
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, v2, v3
	global_store_b32 v[10:11], v12, off offset:196
.LBB0_257:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 50, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v2
	s_cbranch_execz .LBB0_259
; %bb.258:
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s7, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, v35, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, v36, v3, vcc_lo
	global_load_b32 v10, v[10:11], off offset:200
	global_load_b32 v11, v[2:3], off offset:200
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v10, v34, v10
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, v10, v4
	global_store_b32 v[2:3], v11, off offset:200
.LBB0_259:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 51, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v2
	s_cbranch_execz .LBB0_261
; %bb.260:
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s7, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, v35, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, v36, v3, vcc_lo
	global_load_b32 v4, v[10:11], off offset:204
	global_load_b32 v10, v[2:3], off offset:204
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v4, v34, v4
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v10, v4, v5
	global_store_b32 v[2:3], v10, off offset:204
.LBB0_261:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 52, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v2
	s_cbranch_execz .LBB0_263
; %bb.262:
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s7, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, v35, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, v36, v3, vcc_lo
	global_load_b32 v4, v[4:5], off offset:208
	global_load_b32 v5, v[2:3], off offset:208
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v4, v34, v4
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v5, v4, v6
	global_store_b32 v[2:3], v5, off offset:208
.LBB0_263:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 53, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v2
	s_cbranch_execz .LBB0_265
; %bb.264:
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s7, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, v35, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, v36, v3, vcc_lo
	global_load_b32 v4, v[4:5], off offset:212
	global_load_b32 v5, v[2:3], off offset:212
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v4, v34, v4
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v5, v4, v7
	global_store_b32 v[2:3], v5, off offset:212
.LBB0_265:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 54, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s8, v2
	s_cbranch_execz .LBB0_267
; %bb.266:
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s6, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s7, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, v35, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, v36, v3, vcc_lo
	global_load_b32 v4, v[4:5], off offset:216
	global_load_b32 v5, v[2:3], off offset:216
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v4, v34, v4
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v5, v4, v8
	global_store_b32 v[2:3], v5, off offset:216
.LBB0_267:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 55, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s8, v2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_269
; %bb.268:
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s7, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, v35, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v36, v1, vcc_lo
	global_load_b32 v2, v[2:3], off offset:220
	global_load_b32 v3, v[0:1], off offset:220
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v2, v34, v2
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v3, v2, v9
	global_store_b32 v[0:1], v3, off offset:220
.LBB0_269:                              ; %.loopexit.3
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201, .Lfunc_end0-gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 60
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
		.amdhsa_next_free_vgpr 191
		.amdhsa_next_free_sgpr 28
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.num_vgpr, 191
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.numbered_sgpr, 28
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 22428
; TotalNumSgprs: 30
; NumVgprs: 191
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 191
; Occupancy: 8
; WaveLimiterHint : 1
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
	.protected	ff_fold_lut             ; @ff_fold_lut
	.type	ff_fold_lut,@object
	.section	.rodata,"a",@progbits
	.globl	ff_fold_lut
	.p2align	4, 0x0
ff_fold_lut:
	.long	3402419920                      ; 0xcaccced0
	.long	3099641032                      ; 0xb8c0c4c8
	.long	1145059328                      ; 0x44403800
	.long	1313622600                      ; 0x4e4c4a48
	.long	3267675848                      ; 0xc2c4c6c8
	.long	2964896960                      ; 0xb0b8bcc0
	.long	1010315264                      ; 0x3c383000
	.long	1178878528                      ; 0x46444240
	.long	3132931776                      ; 0xbabcbec0
	.long	2830152888                      ; 0xa8b0b4b8
	.long	875571200                       ; 0x34302800
	.long	1044134456                      ; 0x3e3c3a38
	.long	2998187704                      ; 0xb2b4b6b8
	.long	2695408816                      ; 0xa0a8acb0
	.long	740827136                       ; 0x2c282000
	.long	909390384                       ; 0x36343230
	.long	2863443632                      ; 0xaaacaeb0
	.long	2560664744                      ; 0x98a0a4a8
	.long	606083072                       ; 0x24201800
	.long	774646312                       ; 0x2e2c2a28
	.long	2728699560                      ; 0xa2a4a6a8
	.long	2425920672                      ; 0x90989ca0
	.long	471339008                       ; 0x1c181000
	.long	639902240                       ; 0x26242220
	.long	2593955488                      ; 0x9a9c9ea0
	.long	2291176600                      ; 0x88909498
	.long	336594944                       ; 0x14100800
	.long	505158168                       ; 0x1e1c1a18
	.long	2459211416                      ; 0x92949698
	.long	2223541392                      ; 0x84888c90
	.long	201851904                       ; 0xc080400
	.long	370414096                       ; 0x16141210
	.long	2324467344                      ; 0x8a8c8e90
	.long	2189723272                      ; 0x82848688
	.long	100925952                       ; 0x6040200
	.long	235670024                       ; 0xe0c0a08
	.long	2240186248                      ; 0x85868788
	.long	2172814212                      ; 0x81828384
	.long	50462976                        ; 0x3020100
	.long	117835012                       ; 0x7060504
	.long	2189657220                      ; 0x82838484
	.long	8487554                         ; 0x818282
	.long	33619968                        ; 0x2010000
	.long	67305986                        ; 0x4030202
	.long	2172813954                      ; 0x81828282
	.long	33153                           ; 0x8181
	.long	16777216                        ; 0x1000000
	.long	33685761                        ; 0x2020101
	.long	2172748161                      ; 0x81818181
	.long	0                               ; 0x0
	.long	0                               ; 0x0
	.long	16843008                        ; 0x1010100
	.zero	16
	.zero	16
	.zero	16
	.long	3486634708                      ; 0xcfd1d2d4
	.long	3167013324                      ; 0xbcc4c9cc
	.long	1229208576                      ; 0x49443c00
	.long	1381060428                      ; 0x52514f4c
	.long	3351890636                      ; 0xc7c9cacc
	.long	3032269252                      ; 0xb4bcc1c4
	.long	1094464512                      ; 0x413c3400
	.long	1246316356                      ; 0x4a494744
	.long	3217146564                      ; 0xbfc1c2c4
	.long	2897525180                      ; 0xacb4b9bc
	.long	959720448                       ; 0x39342c00
	.long	1111572284                      ; 0x42413f3c
	.long	3082402492                      ; 0xb7b9babc
	.long	2762781108                      ; 0xa4acb1b4
	.long	824976384                       ; 0x312c2400
	.long	976828212                       ; 0x3a393734
	.long	2947658420                      ; 0xafb1b2b4
	.long	2628037036                      ; 0x9ca4a9ac
	.long	690232320                       ; 0x29241c00
	.long	842084140                       ; 0x32312f2c
	.long	2812914348                      ; 0xa7a9aaac
	.long	2493292964                      ; 0x949ca1a4
	.long	555488256                       ; 0x211c1400
	.long	707340068                       ; 0x2a292724
	.long	2678170276                      ; 0x9fa1a2a4
	.long	2358548892                      ; 0x8c94999c
	.long	420744192                       ; 0x19140c00
	.long	572595996                       ; 0x22211f1c
	.long	2543426204                      ; 0x97999a9c
	.long	2257359252                      ; 0x868c9194
	.long	286000640                       ; 0x110c0600
	.long	437851924                       ; 0x1a191714
	.long	2408682132                      ; 0x8f919294
	.long	2206632332                      ; 0x8386898c
	.long	151388928                       ; 0x9060300
	.long	303107852                       ; 0x12110f0c
	.long	2290715276                      ; 0x88898a8c
	.long	2189657222                      ; 0x82838486
	.long	67305984                        ; 0x4030200
	.long	168364038                       ; 0xa090806
	.long	2223277446                      ; 0x84848586
	.long	2172813955                      ; 0x81828283
	.long	33685760                        ; 0x2020100
	.long	84149251                        ; 0x5040403
	.long	2189591171                      ; 0x82828283
	.long	8487298                         ; 0x818182
	.long	16842752                        ; 0x1010000
	.long	33686018                        ; 0x2020202
	.long	2172748162                      ; 0x81818182
	.long	33153                           ; 0x8181
	.long	16777216                        ; 0x1000000
	.long	16843009                        ; 0x1010101
	.long	8487297                         ; 0x818181
	.long	0                               ; 0x0
	.long	0                               ; 0x0
	.long	16842752                        ; 0x1010000
	.zero	16
	.zero	16
	.size	ff_fold_lut, 512

	.type	__hip_cuid_4dbe48b018d034c5,@object ; @__hip_cuid_4dbe48b018d034c5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4dbe48b018d034c5
__hip_cuid_4dbe48b018d034c5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4dbe48b018d034c5, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym ff_fold_lut
	.addrsig_sym __hip_cuid_4dbe48b018d034c5
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
      - .actual_access:  read_only
        .address_space:  global
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
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 60
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     191
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
