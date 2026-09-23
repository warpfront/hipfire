	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.section	.text._Z12floor_kernelILi0EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi0EEvPKjPfi,comdat
	.protected	_Z12floor_kernelILi0EEvPKjPfi ; -- Begin function _Z12floor_kernelILi0EEvPKjPfi
	.globl	_Z12floor_kernelILi0EEvPKjPfi
	.p2align	8
	.type	_Z12floor_kernelILi0EEvPKjPfi,@function
_Z12floor_kernelILi0EEvPKjPfi:          ; @_Z12floor_kernelILi0EEvPKjPfi
	.cfi_startproc
; %bb.0:                                ; %.preheader183
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_lshl_add_u32 v140, v0, 2, 0
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s2, s[0:1], 0x10
	s_mov_b32 s8, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s9, s8
	ds_store_b32 v140, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s12, s8
	s_mov_b32 s13, s8
	s_mov_b32 s14, s8
	s_mov_b32 s15, s8
	v_lshlrev_b32_e32 v1, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0xf8, v1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_load_b64 v[149:150], v1, s[4:5]
	v_dual_mov_b32 v1, s8 :: v_dual_mov_b32 v6, s13
	v_dual_mov_b32 v3, s10 :: v_dual_mov_b32 v8, s15
	v_mov_b32_e32 v2, s9
	v_dual_mov_b32 v4, s11 :: v_dual_mov_b32 v5, s12
	v_mov_b32_e32 v7, s14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v128, v8
	v_mov_b32_e32 v120, v8
	v_mov_b32_e32 v112, v8
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v103, v7
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v95, v7
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v87, v7
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v122, v2
	v_mov_b32_e32 v121, v1
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v114, v2
	v_mov_b32_e32 v113, v1
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v10, v2
	v_mov_b32_e32 v9, v1
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
	s_cbranch_scc1 .LBB0_5
; %bb.1:                                ; %.preheader181.lr.ph
	v_lshrrev_b32_e32 v129, 1, v0
	v_mbcnt_lo_u32_b32 v171, -1, 0
	v_dual_mov_b32 v173, 1.0 :: v_dual_mov_b32 v174, 0xff800000
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v151, v149 :: v_dual_mov_b32 v154, v150
	v_and_b32_e32 v129, 8, v129
	v_xor_b32_e32 v172, 16, v171
	v_dual_mov_b32 v153, v149 :: v_dual_mov_b32 v152, v150
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v131, 3, v129
	v_cvt_f32_ubyte0_e32 v131, v131
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v156, 0x3a83126f, v131
	v_or_b32_e32 v130, 4, v129
	v_or_b32_e32 v132, 2, v129
	v_or_b32_e32 v133, 1, v129
	v_or_b32_e32 v135, 7, v129
	v_or_b32_e32 v136, 6, v129
	v_cvt_f32_ubyte0_e32 v130, v130
	v_add_nc_u32_e32 v134, 8, v129
	v_or_b32_e32 v129, 5, v129
	v_cvt_f32_ubyte0_e32 v133, v133
	v_cvt_f32_ubyte0_e32 v136, v136
	v_cvt_f32_ubyte0_e32 v132, v132
	v_cvt_f32_ubyte0_e32 v135, v135
	v_cvt_f32_ubyte0_e32 v129, v129
	v_mul_f32_e32 v155, 0x3a83126f, v130
	v_cvt_f32_ubyte0_e32 v134, v134
	v_mul_f32_e32 v161, 0x3a83126f, v136
	v_dual_mul_f32 v157, 0x3a83126f, v132 :: v_dual_mul_f32 v158, 0x3a83126f, v133
	v_mul_f32_e32 v162, 0x3a83126f, v129
	v_dual_mul_f32 v167, 0x3c23d70a, v129 :: v_dual_mul_f32 v168, 0x3c23d70a, v136
	v_dual_mov_b32 v129, 0 :: v_dual_mul_f32 v170, 0x3c23d70a, v134
	v_dual_mul_f32 v159, 0x3a83126f, v134 :: v_dual_mul_f32 v160, 0x3a83126f, v135
	v_dual_mul_f32 v163, 0x3c23d70a, v133 :: v_dual_mul_f32 v164, 0x3c23d70a, v132
	v_dual_mul_f32 v165, 0x3c23d70a, v131 :: v_dual_mul_f32 v166, 0x3c23d70a, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v169, 0x3c23d70a, v135 :: v_dual_mov_b32 v130, v129
	v_dual_mov_b32 v131, v129 :: v_dual_mov_b32 v132, v129
	v_dual_mov_b32 v133, v129 :: v_dual_mov_b32 v134, v129
	v_dual_mov_b32 v135, v129 :: v_dual_mov_b32 v136, v129
	v_mov_b32_e32 v175, v129
	s_branch .LBB0_3
.LBB0_2:                                ; %.preheader179.preheader
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_dscnt 0x0
	v_add_f32_e32 v138, v138, v176
	v_sub_f32_e32 v177, v174, v137
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v174
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mul_f32_e32 v177, 0x3fb8aa3b, v177
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_exp_f32_e32 v177, v177
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
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v174, 0, v177, vcc_lo
	;;#ASMSTART
	;;#ASMEND
	s_add_co_i32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_cmp_eq_u32 s2, 0
	v_dual_mul_f32 v173, v173, v174 :: v_dual_fmac_f32 v138, v175, v174
	v_div_scale_f32 v174, null, v139, v139, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v175, v174
	v_fma_f32 v176, -v174, v175, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v175, v176, v175
	v_div_scale_f32 v176, vcc_lo, v141, v139, v141
	v_mul_f32_e32 v177, v176, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v178, -v174, v177, v176
	v_fmac_f32_e32 v177, v178, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v174, -v174, v177, v176
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v174, v174, v175, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v175, v174, v139, v141
	v_div_scale_f32 v174, null, v139, v139, v142
	v_rcp_f32_e32 v176, v174
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v174, v176, 1.0
	v_fmac_f32_e32 v176, v177, v176
	v_div_scale_f32 v177, vcc_lo, v142, v139, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v178, v177, v176
	v_fma_f32 v179, -v174, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v178, v179, v176
	v_fma_f32 v174, -v174, v178, v177
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v174, v174, v176, v178
	v_div_fixup_f32 v176, v174, v139, v142
	v_div_scale_f32 v174, null, v139, v139, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v177, v174
	v_fma_f32 v178, -v174, v177, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v177, v178, v177
	v_div_scale_f32 v178, vcc_lo, v143, v139, v143
	v_mul_f32_e32 v179, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v174, v179, v178
	v_fmac_f32_e32 v179, v180, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v174, -v174, v179, v178
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v174, v174, v177, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v177, v174, v139, v143
	v_div_scale_f32 v174, null, v139, v139, v144
	v_rcp_f32_e32 v178, v174
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v174, v178, 1.0
	v_fmac_f32_e32 v178, v179, v178
	v_div_scale_f32 v179, vcc_lo, v144, v139, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v180, v179, v178
	v_fma_f32 v181, -v174, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v180, v181, v178
	v_fma_f32 v174, -v174, v180, v179
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v174, v174, v178, v180
	v_div_fixup_f32 v178, v174, v139, v144
	v_div_scale_f32 v174, null, v139, v139, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v179, v174
	v_fma_f32 v180, -v174, v179, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v145, v139, v145
	v_mul_f32_e32 v181, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v182, -v174, v181, v180
	v_fmac_f32_e32 v181, v182, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v174, -v174, v181, v180
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v174, v174, v179, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v179, v174, v139, v145
	v_div_scale_f32 v174, null, v139, v139, v146
	v_rcp_f32_e32 v180, v174
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v181, -v174, v180, 1.0
	v_fmac_f32_e32 v180, v181, v180
	v_div_scale_f32 v181, vcc_lo, v146, v139, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v182, v181, v180
	v_fma_f32 v183, -v174, v182, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v182, v183, v180
	v_fma_f32 v174, -v174, v182, v181
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v174, v174, v180, v182
	v_div_fixup_f32 v180, v174, v139, v146
	v_div_scale_f32 v174, null, v139, v139, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v181, v174
	v_fma_f32 v182, -v174, v181, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v181, v182, v181
	v_div_scale_f32 v182, vcc_lo, v147, v139, v147
	v_mul_f32_e32 v183, v182, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v184, -v174, v183, v182
	v_fmac_f32_e32 v183, v184, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v174, -v174, v183, v182
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v174, v174, v181, v183
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v181, v174, v139, v147
	v_div_scale_f32 v174, null, v139, v139, v148
	v_rcp_f32_e32 v182, v174
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v183, -v174, v182, 1.0
	v_fmac_f32_e32 v182, v183, v182
	v_div_scale_f32 v183, vcc_lo, v148, v139, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v184, v183, v182
	v_fma_f32 v185, -v174, v184, v183
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v184, v185, v182
	v_fma_f32 v174, -v174, v184, v183
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v174, v174, v182, v184
	v_div_fixup_f32 v182, v174, v139, v148
	v_div_scale_f32 v174, null, v139, v139, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v183, v174
	v_fma_f32 v184, -v174, v183, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v183, v184, v183
	v_div_scale_f32 v184, vcc_lo, v173, v139, v173
	v_mul_f32_e32 v185, v184, v183
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v186, -v174, v185, v184
	v_fmac_f32_e32 v185, v186, v183
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v174, -v174, v185, v184
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v174, v174, v183, v185
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v183, v174, v139, v173
	v_mov_b16_e64 v174.l, v129.l
	v_mov_b16_e64 v174.h, 0
	v_mul_f32_e32 v128, v183, v128
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v173.l, v174.l
	v_mov_b16_e64 v173.h, v174.h
	v_cvt_pk_fp8_f32 v174.l, v179, v180
	v_cvt_pk_fp8_f32 v174.h, v181, v182
	v_dual_mul_f32 v127, v183, v127 :: v_dual_mul_f32 v124, v183, v124
	v_cvt_pk_fp8_f32 v173.l, v175, v176
	v_cvt_pk_fp8_f32 v173.h, v177, v178
	v_dual_mul_f32 v126, v183, v126 :: v_dual_mul_f32 v125, v183, v125
	v_dual_mul_f32 v122, v183, v122 :: v_dual_mul_f32 v123, v183, v123
	v_dual_mul_f32 v120, v183, v120 :: v_dual_mul_f32 v121, v183, v121
	v_dual_mul_f32 v118, v183, v118 :: v_dual_mul_f32 v119, v183, v119
	v_dual_mul_f32 v116, v183, v116 :: v_dual_mul_f32 v117, v183, v117
	v_dual_mul_f32 v114, v183, v114 :: v_dual_mul_f32 v115, v183, v115
	v_dual_mul_f32 v112, v183, v112 :: v_dual_mul_f32 v113, v183, v113
	v_mul_f32_e32 v110, v183, v110
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v111, v183, v111 :: v_dual_mul_f32 v108, v183, v108
	v_dual_mul_f32 v109, v183, v109 :: v_dual_mul_f32 v106, v183, v106
	v_dual_mul_f32 v107, v183, v107 :: v_dual_mul_f32 v104, v183, v104
	v_dual_mul_f32 v105, v183, v105 :: v_dual_mul_f32 v102, v183, v102
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[149:150], v[173:174], v[121:128]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v103, v183, v103 :: v_dual_mul_f32 v100, v183, v100
	v_dual_mul_f32 v101, v183, v101 :: v_dual_mul_f32 v98, v183, v98
	v_dual_mul_f32 v99, v183, v99 :: v_dual_mul_f32 v96, v183, v96
	v_dual_mul_f32 v97, v183, v97 :: v_dual_mul_f32 v94, v183, v94
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[149:150], v[173:174], v[113:120]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v95, v183, v95 :: v_dual_mul_f32 v92, v183, v92
	v_dual_mul_f32 v93, v183, v93 :: v_dual_mul_f32 v90, v183, v90
	v_dual_mul_f32 v91, v183, v91 :: v_dual_mul_f32 v88, v183, v88
	v_dual_mul_f32 v89, v183, v89 :: v_dual_mul_f32 v86, v183, v86
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[149:150], v[173:174], v[105:112]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v87, v183, v87 :: v_dual_mul_f32 v84, v183, v84
	v_dual_mul_f32 v85, v183, v85 :: v_dual_mul_f32 v82, v183, v82
	v_dual_mul_f32 v83, v183, v83 :: v_dual_mul_f32 v80, v183, v80
	v_dual_mul_f32 v81, v183, v81 :: v_dual_mul_f32 v78, v183, v78
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[149:150], v[173:174], v[97:104]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v79, v183, v79 :: v_dual_mul_f32 v76, v183, v76
	v_dual_mul_f32 v77, v183, v77 :: v_dual_mul_f32 v74, v183, v74
	v_dual_mul_f32 v75, v183, v75 :: v_dual_mul_f32 v72, v183, v72
	v_dual_mul_f32 v73, v183, v73 :: v_dual_mul_f32 v70, v183, v70
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[149:150], v[173:174], v[89:96]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v71, v183, v71 :: v_dual_mul_f32 v68, v183, v68
	v_dual_mul_f32 v69, v183, v69 :: v_dual_mul_f32 v66, v183, v66
	v_dual_mul_f32 v67, v183, v67 :: v_dual_mul_f32 v64, v183, v64
	v_dual_mul_f32 v65, v183, v65 :: v_dual_mul_f32 v62, v183, v62
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[149:150], v[173:174], v[81:88]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v63, v183, v63 :: v_dual_mul_f32 v60, v183, v60
	v_dual_mul_f32 v61, v183, v61 :: v_dual_mul_f32 v58, v183, v58
	v_dual_mul_f32 v59, v183, v59 :: v_dual_mul_f32 v56, v183, v56
	v_dual_mul_f32 v57, v183, v57 :: v_dual_mul_f32 v54, v183, v54
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[149:150], v[173:174], v[73:80]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v55, v183, v55 :: v_dual_mul_f32 v52, v183, v52
	v_dual_mul_f32 v53, v183, v53 :: v_dual_mul_f32 v50, v183, v50
	v_dual_mul_f32 v51, v183, v51 :: v_dual_mul_f32 v48, v183, v48
	v_dual_mul_f32 v49, v183, v49 :: v_dual_mul_f32 v46, v183, v46
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[149:150], v[173:174], v[65:72]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v47, v183, v47 :: v_dual_mul_f32 v44, v183, v44
	v_dual_mul_f32 v45, v183, v45 :: v_dual_mul_f32 v42, v183, v42
	v_dual_mul_f32 v43, v183, v43 :: v_dual_mul_f32 v40, v183, v40
	v_dual_mul_f32 v41, v183, v41 :: v_dual_mul_f32 v38, v183, v38
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[149:150], v[173:174], v[57:64]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v39, v183, v39 :: v_dual_mul_f32 v36, v183, v36
	v_dual_mul_f32 v37, v183, v37 :: v_dual_mul_f32 v34, v183, v34
	v_dual_mul_f32 v35, v183, v35 :: v_dual_mul_f32 v32, v183, v32
	v_dual_mul_f32 v33, v183, v33 :: v_dual_mul_f32 v30, v183, v30
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[149:150], v[173:174], v[49:56]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v31, v183, v31 :: v_dual_mul_f32 v28, v183, v28
	v_dual_mul_f32 v29, v183, v29 :: v_dual_mul_f32 v26, v183, v26
	v_dual_mul_f32 v27, v183, v27 :: v_dual_mul_f32 v24, v183, v24
	v_dual_mul_f32 v25, v183, v25 :: v_dual_mul_f32 v22, v183, v22
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[149:150], v[173:174], v[41:48]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v23, v183, v23 :: v_dual_mul_f32 v20, v183, v20
	v_dual_mul_f32 v21, v183, v21 :: v_dual_mul_f32 v18, v183, v18
	v_dual_mul_f32 v19, v183, v19 :: v_dual_mul_f32 v16, v183, v16
	v_dual_mul_f32 v17, v183, v17 :: v_dual_mul_f32 v14, v183, v14
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[149:150], v[173:174], v[33:40]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v15, v183, v15 :: v_dual_mul_f32 v12, v183, v12
	v_dual_mul_f32 v13, v183, v13 :: v_dual_mul_f32 v10, v183, v10
	v_dual_mul_f32 v11, v183, v11 :: v_dual_mul_f32 v8, v183, v8
	v_dual_mul_f32 v9, v183, v9 :: v_dual_mul_f32 v6, v183, v6
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[149:150], v[173:174], v[25:32]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v7, v183, v7 :: v_dual_mul_f32 v4, v183, v4
	v_dual_mul_f32 v5, v183, v5 :: v_dual_mul_f32 v2, v183, v2
	v_mul_f32_e32 v3, v183, v3
	v_mul_f32_e32 v1, v183, v1
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[149:150], v[173:174], v[17:24]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[149:150], v[173:174], v[9:16]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[149:150], v[173:174], v[1:8]
	v_dual_mov_b32 v174, v137 :: v_dual_mov_b32 v175, v138
	v_mov_b32_e32 v173, v139
	s_cbranch_scc1 .LBB0_6
.LBB0_3:                                ; %.preheader181
                                        ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v148, v136 :: v_dual_mov_b32 v147, v135
	v_dual_mov_b32 v146, v134 :: v_dual_mov_b32 v145, v133
	v_dual_mov_b32 v144, v132 :: v_dual_mov_b32 v143, v131
	v_dual_mov_b32 v142, v130 :: v_dual_mov_b32 v141, v129
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	v_cmp_gt_u32_e32 vcc_lo, 32, v172
	s_mov_b32 s3, exec_lo
	v_dual_mul_f32 v138, 0x3e000000, v142 :: v_dual_mul_f32 v139, 0x3e000000, v141
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v141, 0x3e000000, v143 :: v_dual_mul_f32 v142, 0x3e000000, v144
	v_mul_f32_e32 v144, 0x3e000000, v145
	v_mul_f32_e32 v137, v138, v157
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v143, v139, v158
	v_dual_mul_f32 v145, 0x3e000000, v146 :: v_dual_mul_f32 v146, v142, v155
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v176, v141, v156 :: v_dual_mul_f32 v177, v144, v162
	v_max3_num_f32 v137, v143, 0xff800000, v137
	v_mul_f32_e32 v143, 0x3e000000, v147
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v147, 0x3e000000, v148 :: v_dual_mul_f32 v148, v145, v161
	v_max3_num_f32 v137, v137, v176, v146
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v146, v171, v172, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v176, v147, v159
	v_mul_f32_e32 v178, v143, v160
	v_max3_num_f32 v137, v137, v177, v148
	v_lshlrev_b32_e32 v177, 2, v146
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max3_num_f32 v137, v137, v178, v176
	ds_bpermute_b32 v146, v177, v137
	s_wait_dscnt 0x0
	v_max3_num_f32 v137, v174, v137, v146
	v_fma_f32 v138, v138, v157, -v137
	v_fma_f32 v141, v141, v156, -v137
	v_fma_f32 v142, v142, v155, -v137
	v_fma_f32 v139, v139, v158, -v137
	v_fma_f32 v145, v145, v161, -v137
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v138, 0x3fb8aa3b, v138 :: v_dual_mul_f32 v141, 0x3fb8aa3b, v141
	v_mul_f32_e32 v142, 0x3fb8aa3b, v142
	v_fma_f32 v144, v144, v162, -v137
	v_fma_f32 v143, v143, v160, -v137
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v138, v138
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v137
	v_mul_f32_e32 v139, 0x3fb8aa3b, v139
	v_mul_f32_e32 v145, 0x3fb8aa3b, v145
	v_exp_f32_e32 v142, v142
	v_dual_mul_f32 v144, 0x3fb8aa3b, v144 :: v_dual_mul_f32 v143, 0x3fb8aa3b, v143
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v139, v139
	v_exp_f32_e32 v141, v141
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_exp_f32_e32 v144, v144
	v_exp_f32_e32 v143, v143
	v_exp_f32_e32 v145, v145
	v_cndmask_b32_e64 v176, v142, 0, vcc_lo
	v_mul_f32_e32 v142, v164, v138
	v_fma_f32 v146, v147, v159, -v137
	v_cndmask_b32_e64 v139, v139, 0, vcc_lo
	v_cndmask_b32_e64 v147, v141, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v178, v144, 0, vcc_lo
	v_dual_mul_f32 v146, 0x3fb8aa3b, v146 :: v_dual_mul_f32 v141, v163, v139
	v_cndmask_b32_e64 v180, v143, 0, vcc_lo
	v_add_f32_e32 v138, v139, v138
	v_cndmask_b32_e64 v179, v145, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v146, v146
	v_dual_mul_f32 v143, v165, v147 :: v_dual_mul_f32 v144, v166, v176
	v_max3_num_f32 v148, v141, 0, v142
	v_dual_add_f32 v138, v138, v147 :: v_dual_mul_f32 v145, v167, v178
	v_mul_f32_e32 v147, v169, v180
	v_max3_num_f32 v181, v148, v143, v144
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v139, v146, 0, vcc_lo
	v_mul_f32_e32 v146, v168, v179
	v_add_f32_e32 v138, v138, v176
	v_mul_f32_e32 v148, v170, v139
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v181, v181, v145, v146
	v_add_f32_e32 v138, v138, v178
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v181, v181, v147, v148
	v_add_f32_e32 v138, v138, v179
	ds_bpermute_b32 v178, v177, v181
	v_add_f32_e32 v138, v138, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v138, v138, v139
	ds_bpermute_b32 v176, v177, v138
	s_wait_dscnt 0x1
	v_max_num_f32_e32 v139, v178, v178
	v_max_num_f32_e32 v177, v181, v139
	v_mov_b32_e32 v139, v173
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_lt_f32_e32 0, v177
	s_cbranch_execz .LBB0_2
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	v_div_scale_f32 v139, null, 0x43e00000, 0x43e00000, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v178, v139
	v_fma_f32 v179, -v139, v178, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v178, v179, v178
	v_div_scale_f32 v179, vcc_lo, v177, 0x43e00000, v177
	v_mul_f32_e32 v180, v179, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v181, -v139, v180, v179
	v_fmac_f32_e32 v180, v181, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v139, -v139, v180, v179
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v139, v139, v178, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v139, v139, 0x43e00000, v177
	v_max_num_f32_e32 v139, 0x1f800000, v139
	s_branch .LBB0_2
.LBB0_5:
	v_dual_mov_b32 v139, 1.0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v137, 0xff800000 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v138, 0
	v_mov_b32_e32 v141, 0
.LBB0_6:                                ; %Flow
	s_load_b32 s0, s[0:1], 0x24
	v_dual_mul_f32 v25, v139, v25 :: v_dual_mul_f32 v26, v139, v26
	v_dual_mul_f32 v27, v139, v27 :: v_dual_mul_f32 v28, v139, v28
	v_dual_mul_f32 v29, v139, v29 :: v_dual_mul_f32 v30, v139, v30
	v_dual_mul_f32 v31, v139, v31 :: v_dual_mul_f32 v32, v139, v32
	v_dual_mul_f32 v121, v139, v121 :: v_dual_mul_f32 v122, v139, v122
	v_dual_mul_f32 v123, v139, v123 :: v_dual_mul_f32 v124, v139, v124
	v_dual_mul_f32 v97, v139, v97 :: v_dual_mul_f32 v98, v139, v98
	v_dual_mul_f32 v99, v139, v99 :: v_dual_mul_f32 v100, v139, v100
	v_dual_mul_f32 v73, v139, v73 :: v_dual_mul_f32 v74, v139, v74
	v_dual_mul_f32 v75, v139, v75 :: v_dual_mul_f32 v76, v139, v76
	s_wait_kmcnt 0x0
	s_and_b32 s0, 0xffff, s0
	v_dual_mul_f32 v49, v139, v49 :: v_dual_mul_f32 v50, v139, v50
	v_mad_co_u64_u32 v[129:130], null, ttmp9, s0, v[0:1]
	v_dual_mul_f32 v51, v139, v51 :: v_dual_mul_f32 v52, v139, v52
	v_dual_mul_f32 v17, v139, v17 :: v_dual_mul_f32 v18, v139, v18
	v_dual_mul_f32 v19, v139, v19 :: v_dual_mul_f32 v20, v139, v20
	v_dual_mul_f32 v9, v139, v9 :: v_dual_mul_f32 v10, v139, v10
	v_mul_lo_u32 v129, 0x8c, v129
	v_dual_mul_f32 v11, v139, v11 :: v_dual_mul_f32 v12, v139, v12
	v_dual_mul_f32 v0, v139, v1 :: v_dual_mul_f32 v1, v139, v2
	v_dual_mul_f32 v2, v139, v3 :: v_dual_mul_f32 v3, v139, v4
	v_dual_mul_f32 v125, v139, v125 :: v_dual_mul_f32 v126, v139, v126
	v_ashrrev_i32_e32 v130, 31, v129
	v_dual_mul_f32 v127, v139, v127 :: v_dual_mul_f32 v128, v139, v128
	v_dual_mul_f32 v101, v139, v101 :: v_dual_mul_f32 v102, v139, v102
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[129:130], 2, v[129:130]
	v_dual_mul_f32 v103, v139, v103 :: v_dual_mul_f32 v104, v139, v104
	v_dual_mul_f32 v77, v139, v77 :: v_dual_mul_f32 v78, v139, v78
	v_dual_mul_f32 v79, v139, v79 :: v_dual_mul_f32 v80, v139, v80
	v_add_co_u32 v129, vcc_lo, s6, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, s7, v130, vcc_lo
	s_clause 0x1
	global_store_b128 v[129:130], v[25:28], off offset:384
	global_store_b128 v[129:130], v[29:32], off offset:400
	ds_load_b32 v25, v140
	v_dual_mul_f32 v53, v139, v53 :: v_dual_mul_f32 v54, v139, v54
	v_dual_mul_f32 v55, v139, v55 :: v_dual_mul_f32 v56, v139, v56
	v_dual_mul_f32 v21, v139, v21 :: v_dual_mul_f32 v22, v139, v22
	v_dual_mul_f32 v23, v139, v23 :: v_dual_mul_f32 v24, v139, v24
	v_dual_mul_f32 v13, v139, v13 :: v_dual_mul_f32 v14, v139, v14
	v_dual_mul_f32 v15, v139, v15 :: v_dual_mul_f32 v16, v139, v16
	v_dual_mul_f32 v4, v139, v5 :: v_dual_mul_f32 v5, v139, v6
	v_dual_mul_f32 v6, v139, v7 :: v_dual_mul_f32 v7, v139, v8
	v_dual_mul_f32 v113, v139, v113 :: v_dual_mul_f32 v114, v139, v114
	v_dual_mul_f32 v115, v139, v115 :: v_dual_mul_f32 v116, v139, v116
	v_dual_mul_f32 v89, v139, v89 :: v_dual_mul_f32 v90, v139, v90
	v_dual_mul_f32 v91, v139, v91 :: v_dual_mul_f32 v92, v139, v92
	v_dual_mul_f32 v65, v139, v65 :: v_dual_mul_f32 v66, v139, v66
	v_dual_mul_f32 v67, v139, v67 :: v_dual_mul_f32 v68, v139, v68
	v_dual_mul_f32 v41, v139, v41 :: v_dual_mul_f32 v42, v139, v42
	v_dual_mul_f32 v43, v139, v43 :: v_dual_mul_f32 v44, v139, v44
	v_dual_mul_f32 v117, v139, v117 :: v_dual_mul_f32 v118, v139, v118
	v_dual_mul_f32 v119, v139, v119 :: v_dual_mul_f32 v120, v139, v120
	v_dual_mul_f32 v93, v139, v93 :: v_dual_mul_f32 v94, v139, v94
	v_dual_mul_f32 v95, v139, v95 :: v_dual_mul_f32 v96, v139, v96
	v_dual_mul_f32 v69, v139, v69 :: v_dual_mul_f32 v70, v139, v70
	v_dual_mul_f32 v71, v139, v71 :: v_dual_mul_f32 v72, v139, v72
	v_dual_mul_f32 v45, v139, v45 :: v_dual_mul_f32 v46, v139, v46
	v_dual_mul_f32 v47, v139, v47 :: v_dual_mul_f32 v48, v139, v48
	v_dual_mul_f32 v105, v139, v105 :: v_dual_mul_f32 v106, v139, v106
	v_dual_mul_f32 v107, v139, v107 :: v_dual_mul_f32 v108, v139, v108
	v_dual_mul_f32 v81, v139, v81 :: v_dual_mul_f32 v82, v139, v82
	v_dual_mul_f32 v83, v139, v83 :: v_dual_mul_f32 v84, v139, v84
	v_dual_mul_f32 v57, v139, v57 :: v_dual_mul_f32 v58, v139, v58
	v_dual_mul_f32 v59, v139, v59 :: v_dual_mul_f32 v60, v139, v60
	v_dual_mul_f32 v33, v139, v33 :: v_dual_mul_f32 v34, v139, v34
	v_dual_mul_f32 v35, v139, v35 :: v_dual_mul_f32 v36, v139, v36
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v140, v25
	v_dual_mul_f32 v109, v139, v109 :: v_dual_mul_f32 v110, v139, v110
	v_dual_mul_f32 v111, v139, v111 :: v_dual_mul_f32 v112, v139, v112
	s_clause 0x5
	global_store_b128 v[129:130], v[121:124], off
	global_store_b128 v[129:130], v[125:128], off offset:16
	global_store_b128 v[129:130], v[113:116], off offset:32
	global_store_b128 v[129:130], v[117:120], off offset:48
	global_store_b128 v[129:130], v[105:108], off offset:64
	global_store_b128 v[129:130], v[109:112], off offset:80
	v_dual_mul_f32 v85, v139, v85 :: v_dual_mul_f32 v86, v139, v86
	v_dual_mul_f32 v87, v139, v87 :: v_dual_mul_f32 v88, v139, v88
	s_clause 0x5
	global_store_b128 v[129:130], v[97:100], off offset:96
	global_store_b128 v[129:130], v[101:104], off offset:112
	global_store_b128 v[129:130], v[89:92], off offset:128
	global_store_b128 v[129:130], v[93:96], off offset:144
	global_store_b128 v[129:130], v[81:84], off offset:160
	global_store_b128 v[129:130], v[85:88], off offset:176
	v_dual_mul_f32 v61, v139, v61 :: v_dual_mul_f32 v62, v139, v62
	v_dual_mul_f32 v63, v139, v63 :: v_dual_mul_f32 v64, v139, v64
	s_clause 0x5
	global_store_b128 v[129:130], v[73:76], off offset:192
	global_store_b128 v[129:130], v[77:80], off offset:208
	global_store_b128 v[129:130], v[65:68], off offset:224
	global_store_b128 v[129:130], v[69:72], off offset:240
	global_store_b128 v[129:130], v[57:60], off offset:256
	global_store_b128 v[129:130], v[61:64], off offset:272
	v_dual_mul_f32 v37, v139, v37 :: v_dual_mul_f32 v38, v139, v38
	v_dual_mul_f32 v39, v139, v39 :: v_dual_mul_f32 v40, v139, v40
	s_clause 0xe
	global_store_b128 v[129:130], v[49:52], off offset:288
	global_store_b128 v[129:130], v[53:56], off offset:304
	global_store_b128 v[129:130], v[41:44], off offset:320
	global_store_b128 v[129:130], v[45:48], off offset:336
	global_store_b128 v[129:130], v[33:36], off offset:352
	global_store_b128 v[129:130], v[37:40], off offset:368
	global_store_b128 v[129:130], v[17:20], off offset:416
	global_store_b128 v[129:130], v[21:24], off offset:432
	global_store_b128 v[129:130], v[9:12], off offset:448
	global_store_b128 v[129:130], v[13:16], off offset:464
	global_store_b128 v[129:130], v[0:3], off offset:480
	global_store_b128 v[129:130], v[4:7], off offset:496
	global_store_b128 v[129:130], v[141:144], off offset:512
	global_store_b128 v[129:130], v[145:148], off offset:528
	global_store_b128 v[129:130], v[137:140], off offset:544
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	_Z12floor_kernelILi0EEvPKjPfi, .Lfunc_end0-_Z12floor_kernelILi0EEvPKjPfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12floor_kernelILi0EEvPKjPfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 187
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z12floor_kernelILi0EEvPKjPfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z12floor_kernelILi0EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi0EEvPKjPfi,comdat
                                        ; -- End function
	.set .L_Z12floor_kernelILi0EEvPKjPfi.num_vgpr, 187
	.set .L_Z12floor_kernelILi0EEvPKjPfi.num_agpr, 0
	.set .L_Z12floor_kernelILi0EEvPKjPfi.numbered_sgpr, 16
	.set .L_Z12floor_kernelILi0EEvPKjPfi.num_named_barrier, 0
	.set .L_Z12floor_kernelILi0EEvPKjPfi.private_seg_size, 0
	.set .L_Z12floor_kernelILi0EEvPKjPfi.uses_vcc, 1
	.set .L_Z12floor_kernelILi0EEvPKjPfi.uses_flat_scratch, 0
	.set .L_Z12floor_kernelILi0EEvPKjPfi.has_dyn_sized_stack, 0
	.set .L_Z12floor_kernelILi0EEvPKjPfi.has_recursion, 0
	.set .L_Z12floor_kernelILi0EEvPKjPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4568
; TotalNumSgprs: 18
; NumVgprs: 187
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 187
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z12floor_kernelILi1EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi1EEvPKjPfi,comdat
	.protected	_Z12floor_kernelILi1EEvPKjPfi ; -- Begin function _Z12floor_kernelILi1EEvPKjPfi
	.globl	_Z12floor_kernelILi1EEvPKjPfi
	.p2align	8
	.type	_Z12floor_kernelILi1EEvPKjPfi,@function
_Z12floor_kernelILi1EEvPKjPfi:          ; @_Z12floor_kernelILi1EEvPKjPfi
	.cfi_startproc
; %bb.0:                                ; %.preheader181
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_lshl_add_u32 v140, v0, 2, 0
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s2, s[0:1], 0x10
	s_mov_b32 s8, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s9, s8
	ds_store_b32 v140, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s12, s8
	s_mov_b32 s13, s8
	s_mov_b32 s14, s8
	s_mov_b32 s15, s8
	v_lshlrev_b32_e32 v1, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0xf8, v1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_load_b64 v[149:150], v1, s[4:5]
	v_dual_mov_b32 v1, s8 :: v_dual_mov_b32 v6, s13
	v_dual_mov_b32 v3, s10 :: v_dual_mov_b32 v8, s15
	v_mov_b32_e32 v2, s9
	v_dual_mov_b32 v4, s11 :: v_dual_mov_b32 v5, s12
	v_mov_b32_e32 v7, s14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v128, v8
	v_mov_b32_e32 v120, v8
	v_mov_b32_e32 v112, v8
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v103, v7
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v95, v7
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v87, v7
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v122, v2
	v_mov_b32_e32 v121, v1
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v114, v2
	v_mov_b32_e32 v113, v1
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v10, v2
	v_mov_b32_e32 v9, v1
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
	s_cbranch_scc1 .LBB1_5
; %bb.1:                                ; %.preheader179.lr.ph
	v_lshrrev_b32_e32 v129, 1, v0
	v_dual_mov_b32 v171, 1.0 :: v_dual_mov_b32 v172, 0xff800000
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v151, v149 :: v_dual_mov_b32 v154, v150
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_and_b32_e32 v129, 8, v129
	v_mov_b32_e32 v153, v149
	s_mov_b32 s3, 0x76543210
	v_mov_b32_e32 v152, v150
	v_or_b32_e32 v131, 3, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v131, v131
	v_mul_f32_e32 v156, 0x3a83126f, v131
	v_or_b32_e32 v130, 4, v129
	v_or_b32_e32 v132, 2, v129
	v_or_b32_e32 v133, 1, v129
	v_or_b32_e32 v135, 7, v129
	v_or_b32_e32 v136, 6, v129
	v_cvt_f32_ubyte0_e32 v130, v130
	v_add_nc_u32_e32 v134, 8, v129
	v_or_b32_e32 v129, 5, v129
	v_cvt_f32_ubyte0_e32 v133, v133
	v_cvt_f32_ubyte0_e32 v136, v136
	v_cvt_f32_ubyte0_e32 v132, v132
	v_cvt_f32_ubyte0_e32 v135, v135
	v_cvt_f32_ubyte0_e32 v129, v129
	v_mul_f32_e32 v155, 0x3a83126f, v130
	v_cvt_f32_ubyte0_e32 v134, v134
	v_mul_f32_e32 v161, 0x3a83126f, v136
	v_dual_mul_f32 v157, 0x3a83126f, v132 :: v_dual_mul_f32 v158, 0x3a83126f, v133
	v_mul_f32_e32 v162, 0x3a83126f, v129
	v_dual_mul_f32 v167, 0x3c23d70a, v129 :: v_dual_mul_f32 v168, 0x3c23d70a, v136
	v_dual_mov_b32 v129, 0 :: v_dual_mul_f32 v170, 0x3c23d70a, v134
	v_dual_mul_f32 v159, 0x3a83126f, v134 :: v_dual_mul_f32 v160, 0x3a83126f, v135
	v_dual_mul_f32 v163, 0x3c23d70a, v133 :: v_dual_mul_f32 v164, 0x3c23d70a, v132
	v_dual_mul_f32 v165, 0x3c23d70a, v131 :: v_dual_mul_f32 v166, 0x3c23d70a, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v169, 0x3c23d70a, v135 :: v_dual_mov_b32 v130, v129
	v_dual_mov_b32 v131, v129 :: v_dual_mov_b32 v132, v129
	v_dual_mov_b32 v133, v129 :: v_dual_mov_b32 v134, v129
	v_dual_mov_b32 v135, v129 :: v_dual_mov_b32 v136, v129
	v_mov_b32_e32 v173, v129
	s_branch .LBB1_3
.LBB1_2:                                ; %.preheader177.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_sub_f32 v175, v172, v137 :: v_dual_add_f32 v138, v138, v174
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v172
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v175, 0x3fb8aa3b, v175
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_exp_f32_e32 v175, v175
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
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v172, 0, v175, vcc_lo
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_add_co_i32 s2, s2, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, 0
	v_dual_fmac_f32 v138, v173, v172 :: v_dual_mul_f32 v171, v171, v172
	v_div_scale_f32 v172, null, v139, v139, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v173, v172
	v_fma_f32 v174, -v172, v173, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v173, v174, v173
	v_div_scale_f32 v174, vcc_lo, v141, v139, v141
	v_mul_f32_e32 v175, v174, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v176, -v172, v175, v174
	v_fmac_f32_e32 v175, v176, v173
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v172, -v172, v175, v174
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v172, v172, v173, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v173, v172, v139, v141
	v_div_scale_f32 v172, null, v139, v139, v142
	v_rcp_f32_e32 v174, v172
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v175, -v172, v174, 1.0
	v_fmac_f32_e32 v174, v175, v174
	v_div_scale_f32 v175, vcc_lo, v142, v139, v142
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v176, v175, v174
	v_fma_f32 v177, -v172, v176, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v176, v177, v174
	v_fma_f32 v172, -v172, v176, v175
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v172, v172, v174, v176
	v_div_fixup_f32 v174, v172, v139, v142
	v_div_scale_f32 v172, null, v139, v139, v143
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v175, v172
	v_fma_f32 v176, -v172, v175, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v175, v176, v175
	v_div_scale_f32 v176, vcc_lo, v143, v139, v143
	v_mul_f32_e32 v177, v176, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v178, -v172, v177, v176
	v_fmac_f32_e32 v177, v178, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v172, -v172, v177, v176
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v172, v172, v175, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v175, v172, v139, v143
	v_div_scale_f32 v172, null, v139, v139, v144
	v_rcp_f32_e32 v176, v172
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v177, -v172, v176, 1.0
	v_fmac_f32_e32 v176, v177, v176
	v_div_scale_f32 v177, vcc_lo, v144, v139, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v178, v177, v176
	v_fma_f32 v179, -v172, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v178, v179, v176
	v_fma_f32 v172, -v172, v178, v177
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v172, v172, v176, v178
	v_div_fixup_f32 v176, v172, v139, v144
	v_div_scale_f32 v172, null, v139, v139, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v177, v172
	v_fma_f32 v178, -v172, v177, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v177, v178, v177
	v_div_scale_f32 v178, vcc_lo, v145, v139, v145
	v_mul_f32_e32 v179, v178, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v180, -v172, v179, v178
	v_fmac_f32_e32 v179, v180, v177
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v172, -v172, v179, v178
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v172, v172, v177, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v177, v172, v139, v145
	v_div_scale_f32 v172, null, v139, v139, v146
	v_rcp_f32_e32 v178, v172
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v172, v178, 1.0
	v_fmac_f32_e32 v178, v179, v178
	v_div_scale_f32 v179, vcc_lo, v146, v139, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v180, v179, v178
	v_fma_f32 v181, -v172, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v180, v181, v178
	v_fma_f32 v172, -v172, v180, v179
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v172, v172, v178, v180
	v_div_fixup_f32 v178, v172, v139, v146
	v_div_scale_f32 v172, null, v139, v139, v147
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v179, v172
	v_fma_f32 v180, -v172, v179, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v179, v180, v179
	v_div_scale_f32 v180, vcc_lo, v147, v139, v147
	v_mul_f32_e32 v181, v180, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v182, -v172, v181, v180
	v_fmac_f32_e32 v181, v182, v179
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v172, -v172, v181, v180
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v172, v172, v179, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v179, v172, v139, v147
	v_div_scale_f32 v172, null, v139, v139, v148
	v_rcp_f32_e32 v180, v172
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v181, -v172, v180, 1.0
	v_fmac_f32_e32 v180, v181, v180
	v_div_scale_f32 v181, vcc_lo, v148, v139, v148
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v182, v181, v180
	v_fma_f32 v183, -v172, v182, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v182, v183, v180
	v_fma_f32 v172, -v172, v182, v181
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v172, v172, v180, v182
	v_div_fixup_f32 v180, v172, v139, v148
	v_div_scale_f32 v172, null, v139, v139, v171
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v181, v172
	v_fma_f32 v182, -v172, v181, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v181, v182, v181
	v_div_scale_f32 v182, vcc_lo, v171, v139, v171
	v_mul_f32_e32 v183, v182, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v184, -v172, v183, v182
	v_fmac_f32_e32 v183, v184, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v172, -v172, v183, v182
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v172, v172, v181, v183
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v181, v172, v139, v171
	v_mov_b16_e64 v172.l, v129.l
	v_mov_b16_e64 v172.h, 0
	v_mul_f32_e32 v128, v181, v128
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mov_b16_e64 v171.l, v172.l
	v_mov_b16_e64 v171.h, v172.h
	v_cvt_pk_fp8_f32 v172.l, v177, v178
	v_cvt_pk_fp8_f32 v172.h, v179, v180
	v_dual_mul_f32 v127, v181, v127 :: v_dual_mul_f32 v124, v181, v124
	v_cvt_pk_fp8_f32 v171.l, v173, v174
	v_cvt_pk_fp8_f32 v171.h, v175, v176
	v_dual_mul_f32 v126, v181, v126 :: v_dual_mul_f32 v125, v181, v125
	v_dual_mul_f32 v122, v181, v122 :: v_dual_mul_f32 v123, v181, v123
	v_dual_mul_f32 v120, v181, v120 :: v_dual_mul_f32 v121, v181, v121
	v_dual_mul_f32 v118, v181, v118 :: v_dual_mul_f32 v119, v181, v119
	v_dual_mul_f32 v116, v181, v116 :: v_dual_mul_f32 v117, v181, v117
	v_dual_mul_f32 v114, v181, v114 :: v_dual_mul_f32 v115, v181, v115
	v_dual_mul_f32 v112, v181, v112 :: v_dual_mul_f32 v113, v181, v113
	v_mul_f32_e32 v110, v181, v110
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v111, v181, v111 :: v_dual_mul_f32 v108, v181, v108
	v_dual_mul_f32 v109, v181, v109 :: v_dual_mul_f32 v106, v181, v106
	v_dual_mul_f32 v107, v181, v107 :: v_dual_mul_f32 v104, v181, v104
	v_dual_mul_f32 v105, v181, v105 :: v_dual_mul_f32 v102, v181, v102
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[149:150], v[171:172], v[121:128]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v103, v181, v103 :: v_dual_mul_f32 v100, v181, v100
	v_dual_mul_f32 v101, v181, v101 :: v_dual_mul_f32 v98, v181, v98
	v_dual_mul_f32 v99, v181, v99 :: v_dual_mul_f32 v96, v181, v96
	v_dual_mul_f32 v97, v181, v97 :: v_dual_mul_f32 v94, v181, v94
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[149:150], v[171:172], v[113:120]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v95, v181, v95 :: v_dual_mul_f32 v92, v181, v92
	v_dual_mul_f32 v93, v181, v93 :: v_dual_mul_f32 v90, v181, v90
	v_dual_mul_f32 v91, v181, v91 :: v_dual_mul_f32 v88, v181, v88
	v_dual_mul_f32 v89, v181, v89 :: v_dual_mul_f32 v86, v181, v86
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[149:150], v[171:172], v[105:112]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v87, v181, v87 :: v_dual_mul_f32 v84, v181, v84
	v_dual_mul_f32 v85, v181, v85 :: v_dual_mul_f32 v82, v181, v82
	v_dual_mul_f32 v83, v181, v83 :: v_dual_mul_f32 v80, v181, v80
	v_dual_mul_f32 v81, v181, v81 :: v_dual_mul_f32 v78, v181, v78
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[149:150], v[171:172], v[97:104]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v79, v181, v79 :: v_dual_mul_f32 v76, v181, v76
	v_dual_mul_f32 v77, v181, v77 :: v_dual_mul_f32 v74, v181, v74
	v_dual_mul_f32 v75, v181, v75 :: v_dual_mul_f32 v72, v181, v72
	v_dual_mul_f32 v73, v181, v73 :: v_dual_mul_f32 v70, v181, v70
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[149:150], v[171:172], v[89:96]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v71, v181, v71 :: v_dual_mul_f32 v68, v181, v68
	v_dual_mul_f32 v69, v181, v69 :: v_dual_mul_f32 v66, v181, v66
	v_dual_mul_f32 v67, v181, v67 :: v_dual_mul_f32 v64, v181, v64
	v_dual_mul_f32 v65, v181, v65 :: v_dual_mul_f32 v62, v181, v62
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[149:150], v[171:172], v[81:88]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v63, v181, v63 :: v_dual_mul_f32 v60, v181, v60
	v_dual_mul_f32 v61, v181, v61 :: v_dual_mul_f32 v58, v181, v58
	v_dual_mul_f32 v59, v181, v59 :: v_dual_mul_f32 v56, v181, v56
	v_dual_mul_f32 v57, v181, v57 :: v_dual_mul_f32 v54, v181, v54
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[149:150], v[171:172], v[73:80]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v55, v181, v55 :: v_dual_mul_f32 v52, v181, v52
	v_dual_mul_f32 v53, v181, v53 :: v_dual_mul_f32 v50, v181, v50
	v_dual_mul_f32 v51, v181, v51 :: v_dual_mul_f32 v48, v181, v48
	v_dual_mul_f32 v49, v181, v49 :: v_dual_mul_f32 v46, v181, v46
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[149:150], v[171:172], v[65:72]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v47, v181, v47 :: v_dual_mul_f32 v44, v181, v44
	v_dual_mul_f32 v45, v181, v45 :: v_dual_mul_f32 v42, v181, v42
	v_dual_mul_f32 v43, v181, v43 :: v_dual_mul_f32 v40, v181, v40
	v_dual_mul_f32 v41, v181, v41 :: v_dual_mul_f32 v38, v181, v38
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[149:150], v[171:172], v[57:64]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v39, v181, v39 :: v_dual_mul_f32 v36, v181, v36
	v_dual_mul_f32 v37, v181, v37 :: v_dual_mul_f32 v34, v181, v34
	v_dual_mul_f32 v35, v181, v35 :: v_dual_mul_f32 v32, v181, v32
	v_dual_mul_f32 v33, v181, v33 :: v_dual_mul_f32 v30, v181, v30
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[149:150], v[171:172], v[49:56]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v31, v181, v31 :: v_dual_mul_f32 v28, v181, v28
	v_dual_mul_f32 v29, v181, v29 :: v_dual_mul_f32 v26, v181, v26
	v_dual_mul_f32 v27, v181, v27 :: v_dual_mul_f32 v24, v181, v24
	v_dual_mul_f32 v25, v181, v25 :: v_dual_mul_f32 v22, v181, v22
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[149:150], v[171:172], v[41:48]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v23, v181, v23 :: v_dual_mul_f32 v20, v181, v20
	v_dual_mul_f32 v21, v181, v21 :: v_dual_mul_f32 v18, v181, v18
	v_dual_mul_f32 v19, v181, v19 :: v_dual_mul_f32 v16, v181, v16
	v_dual_mul_f32 v17, v181, v17 :: v_dual_mul_f32 v14, v181, v14
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[149:150], v[171:172], v[33:40]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v15, v181, v15 :: v_dual_mul_f32 v12, v181, v12
	v_dual_mul_f32 v13, v181, v13 :: v_dual_mul_f32 v10, v181, v10
	v_dual_mul_f32 v11, v181, v11 :: v_dual_mul_f32 v8, v181, v8
	v_dual_mul_f32 v9, v181, v9 :: v_dual_mul_f32 v6, v181, v6
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[149:150], v[171:172], v[25:32]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v7, v181, v7 :: v_dual_mul_f32 v4, v181, v4
	v_dual_mul_f32 v5, v181, v5 :: v_dual_mul_f32 v2, v181, v2
	v_mul_f32_e32 v3, v181, v3
	v_mul_f32_e32 v1, v181, v1
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[149:150], v[171:172], v[17:24]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[149:150], v[171:172], v[9:16]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[149:150], v[171:172], v[1:8]
	v_dual_mov_b32 v172, v137 :: v_dual_mov_b32 v173, v138
	v_mov_b32_e32 v171, v139
	s_cbranch_scc1 .LBB1_6
.LBB1_3:                                ; %.preheader179
                                        ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v148, v136 :: v_dual_mov_b32 v147, v135
	v_dual_mov_b32 v146, v134 :: v_dual_mov_b32 v145, v133
	v_dual_mov_b32 v144, v132 :: v_dual_mov_b32 v143, v131
	v_dual_mov_b32 v142, v130 :: v_dual_mov_b32 v141, v129
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	s_mov_b32 s4, exec_lo
	v_dual_mul_f32 v138, 0x3e000000, v142 :: v_dual_mul_f32 v139, 0x3e000000, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v141, 0x3e000000, v143 :: v_dual_mul_f32 v142, 0x3e000000, v144
	v_mul_f32_e32 v144, 0x3e000000, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v137, v138, v157
	v_mul_f32_e32 v143, v139, v158
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v145, 0x3e000000, v146 :: v_dual_mul_f32 v146, v142, v155
	v_mul_f32_e32 v174, v141, v156
	v_max3_num_f32 v137, v143, 0xff800000, v137
	v_mul_f32_e32 v143, 0x3e000000, v147
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v147, 0x3e000000, v148 :: v_dual_mul_f32 v148, v145, v161
	v_mul_f32_e32 v175, v144, v162
	v_max3_num_f32 v137, v137, v174, v146
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v174, v143, v160
	v_mul_f32_e32 v146, v147, v159
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v137, v137, v175, v148
	v_max3_num_f32 v137, v137, v174, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_e32 v146, v137
	v_permlanex16_b32 v146, v146, s3, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v137, v172, v137, v146
	v_fma_f32 v139, v139, v158, -v137
	v_fma_f32 v138, v138, v157, -v137
	v_fma_f32 v141, v141, v156, -v137
	v_fma_f32 v142, v142, v155, -v137
	v_fma_f32 v144, v144, v162, -v137
	v_fma_f32 v143, v143, v160, -v137
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_fma_f32 v145, v145, v161, -v137
	v_mul_f32_e32 v142, 0x3fb8aa3b, v142
	v_dual_mul_f32 v144, 0x3fb8aa3b, v144 :: v_dual_mul_f32 v143, 0x3fb8aa3b, v143
	v_exp_f32_e32 v139, v139
	v_exp_f32_e32 v138, v138
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v137
	v_fma_f32 v146, v147, v159, -v137
	v_mul_f32_e32 v145, 0x3fb8aa3b, v145
	v_exp_f32_e32 v141, v141
	v_exp_f32_e32 v142, v142
	v_exp_f32_e32 v143, v143
	v_exp_f32_e32 v144, v144
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v139, v139, 0, vcc_lo
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v146, 0x3fb8aa3b, v146
	v_exp_f32_e32 v145, v145
	v_cndmask_b32_e64 v147, v141, 0, vcc_lo
	v_cndmask_b32_e64 v174, v142, 0, vcc_lo
	v_dual_mul_f32 v141, v163, v139 :: v_dual_mul_f32 v142, v164, v138
	v_cndmask_b32_e64 v177, v143, 0, vcc_lo
	v_add_f32_e32 v138, v139, v138
	v_exp_f32_e32 v146, v146
	v_cndmask_b32_e64 v175, v144, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_2)
	v_cndmask_b32_e64 v176, v145, 0, vcc_lo
	v_dual_mul_f32 v143, v165, v147 :: v_dual_mul_f32 v144, v166, v174
	v_max3_num_f32 v139, v141, 0, v142
	v_add_f32_e32 v138, v138, v147
	v_mul_f32_e32 v145, v167, v175
	v_mul_f32_e32 v147, v169, v177
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v178, v146, 0, vcc_lo
	v_mul_f32_e32 v146, v168, v176
	v_max3_num_f32 v139, v139, v143, v144
	v_add_f32_e32 v138, v138, v174
	v_mul_f32_e32 v148, v170, v178
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v139, v139, v145, v146
	v_add_f32_e32 v138, v138, v175
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v139, v139, v147, v148
	v_add_f32_e32 v138, v138, v176
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v174, v139
	v_add_f32_e32 v138, v138, v177
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_permlanex16_b32 v174, v174, s3, 0xfedcba98
	v_add_f32_e32 v138, v138, v178
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v175, v174, v174
	v_dual_mov_b32 v174, v138 :: v_dual_max_num_f32 v175, v139, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v174, v174, s3, 0xfedcba98
	v_mov_b32_e32 v139, v171
	v_cmpx_lt_f32_e32 0, v175
	s_cbranch_execz .LBB1_2
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_div_scale_f32 v139, null, 0x43e00000, 0x43e00000, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v176, v139
	v_fma_f32 v177, -v139, v176, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v176, v177, v176
	v_div_scale_f32 v177, vcc_lo, v175, 0x43e00000, v175
	v_mul_f32_e32 v178, v177, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v139, v178, v177
	v_fmac_f32_e32 v178, v179, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v139, -v139, v178, v177
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v139, v139, v176, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v139, v139, 0x43e00000, v175
	v_max_num_f32_e32 v139, 0x1f800000, v139
	s_branch .LBB1_2
.LBB1_5:
	v_dual_mov_b32 v139, 1.0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v137, 0xff800000 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v138, 0
	v_mov_b32_e32 v141, 0
.LBB1_6:                                ; %Flow
	s_load_b32 s0, s[0:1], 0x24
	v_dual_mul_f32 v25, v139, v25 :: v_dual_mul_f32 v26, v139, v26
	v_dual_mul_f32 v27, v139, v27 :: v_dual_mul_f32 v28, v139, v28
	v_dual_mul_f32 v29, v139, v29 :: v_dual_mul_f32 v30, v139, v30
	v_dual_mul_f32 v31, v139, v31 :: v_dual_mul_f32 v32, v139, v32
	v_dual_mul_f32 v121, v139, v121 :: v_dual_mul_f32 v122, v139, v122
	v_dual_mul_f32 v123, v139, v123 :: v_dual_mul_f32 v124, v139, v124
	v_dual_mul_f32 v97, v139, v97 :: v_dual_mul_f32 v98, v139, v98
	v_dual_mul_f32 v99, v139, v99 :: v_dual_mul_f32 v100, v139, v100
	v_dual_mul_f32 v73, v139, v73 :: v_dual_mul_f32 v74, v139, v74
	v_dual_mul_f32 v75, v139, v75 :: v_dual_mul_f32 v76, v139, v76
	s_wait_kmcnt 0x0
	s_and_b32 s0, 0xffff, s0
	v_dual_mul_f32 v49, v139, v49 :: v_dual_mul_f32 v50, v139, v50
	v_mad_co_u64_u32 v[129:130], null, ttmp9, s0, v[0:1]
	v_dual_mul_f32 v51, v139, v51 :: v_dual_mul_f32 v52, v139, v52
	v_dual_mul_f32 v17, v139, v17 :: v_dual_mul_f32 v18, v139, v18
	v_dual_mul_f32 v19, v139, v19 :: v_dual_mul_f32 v20, v139, v20
	v_dual_mul_f32 v9, v139, v9 :: v_dual_mul_f32 v10, v139, v10
	v_mul_lo_u32 v129, 0x8c, v129
	v_dual_mul_f32 v11, v139, v11 :: v_dual_mul_f32 v12, v139, v12
	v_dual_mul_f32 v0, v139, v1 :: v_dual_mul_f32 v1, v139, v2
	v_dual_mul_f32 v2, v139, v3 :: v_dual_mul_f32 v3, v139, v4
	v_dual_mul_f32 v125, v139, v125 :: v_dual_mul_f32 v126, v139, v126
	v_ashrrev_i32_e32 v130, 31, v129
	v_dual_mul_f32 v127, v139, v127 :: v_dual_mul_f32 v128, v139, v128
	v_dual_mul_f32 v101, v139, v101 :: v_dual_mul_f32 v102, v139, v102
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[129:130], 2, v[129:130]
	v_dual_mul_f32 v103, v139, v103 :: v_dual_mul_f32 v104, v139, v104
	v_dual_mul_f32 v77, v139, v77 :: v_dual_mul_f32 v78, v139, v78
	v_dual_mul_f32 v79, v139, v79 :: v_dual_mul_f32 v80, v139, v80
	v_add_co_u32 v129, vcc_lo, s6, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, s7, v130, vcc_lo
	s_clause 0x1
	global_store_b128 v[129:130], v[25:28], off offset:384
	global_store_b128 v[129:130], v[29:32], off offset:400
	ds_load_b32 v25, v140
	v_dual_mul_f32 v53, v139, v53 :: v_dual_mul_f32 v54, v139, v54
	v_dual_mul_f32 v55, v139, v55 :: v_dual_mul_f32 v56, v139, v56
	v_dual_mul_f32 v21, v139, v21 :: v_dual_mul_f32 v22, v139, v22
	v_dual_mul_f32 v23, v139, v23 :: v_dual_mul_f32 v24, v139, v24
	v_dual_mul_f32 v13, v139, v13 :: v_dual_mul_f32 v14, v139, v14
	v_dual_mul_f32 v15, v139, v15 :: v_dual_mul_f32 v16, v139, v16
	v_dual_mul_f32 v4, v139, v5 :: v_dual_mul_f32 v5, v139, v6
	v_dual_mul_f32 v6, v139, v7 :: v_dual_mul_f32 v7, v139, v8
	v_dual_mul_f32 v113, v139, v113 :: v_dual_mul_f32 v114, v139, v114
	v_dual_mul_f32 v115, v139, v115 :: v_dual_mul_f32 v116, v139, v116
	v_dual_mul_f32 v89, v139, v89 :: v_dual_mul_f32 v90, v139, v90
	v_dual_mul_f32 v91, v139, v91 :: v_dual_mul_f32 v92, v139, v92
	v_dual_mul_f32 v65, v139, v65 :: v_dual_mul_f32 v66, v139, v66
	v_dual_mul_f32 v67, v139, v67 :: v_dual_mul_f32 v68, v139, v68
	v_dual_mul_f32 v41, v139, v41 :: v_dual_mul_f32 v42, v139, v42
	v_dual_mul_f32 v43, v139, v43 :: v_dual_mul_f32 v44, v139, v44
	v_dual_mul_f32 v117, v139, v117 :: v_dual_mul_f32 v118, v139, v118
	v_dual_mul_f32 v119, v139, v119 :: v_dual_mul_f32 v120, v139, v120
	v_dual_mul_f32 v93, v139, v93 :: v_dual_mul_f32 v94, v139, v94
	v_dual_mul_f32 v95, v139, v95 :: v_dual_mul_f32 v96, v139, v96
	v_dual_mul_f32 v69, v139, v69 :: v_dual_mul_f32 v70, v139, v70
	v_dual_mul_f32 v71, v139, v71 :: v_dual_mul_f32 v72, v139, v72
	v_dual_mul_f32 v45, v139, v45 :: v_dual_mul_f32 v46, v139, v46
	v_dual_mul_f32 v47, v139, v47 :: v_dual_mul_f32 v48, v139, v48
	v_dual_mul_f32 v105, v139, v105 :: v_dual_mul_f32 v106, v139, v106
	v_dual_mul_f32 v107, v139, v107 :: v_dual_mul_f32 v108, v139, v108
	v_dual_mul_f32 v81, v139, v81 :: v_dual_mul_f32 v82, v139, v82
	v_dual_mul_f32 v83, v139, v83 :: v_dual_mul_f32 v84, v139, v84
	v_dual_mul_f32 v57, v139, v57 :: v_dual_mul_f32 v58, v139, v58
	v_dual_mul_f32 v59, v139, v59 :: v_dual_mul_f32 v60, v139, v60
	v_dual_mul_f32 v33, v139, v33 :: v_dual_mul_f32 v34, v139, v34
	v_dual_mul_f32 v35, v139, v35 :: v_dual_mul_f32 v36, v139, v36
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v140, v25
	v_dual_mul_f32 v109, v139, v109 :: v_dual_mul_f32 v110, v139, v110
	v_dual_mul_f32 v111, v139, v111 :: v_dual_mul_f32 v112, v139, v112
	s_clause 0x5
	global_store_b128 v[129:130], v[121:124], off
	global_store_b128 v[129:130], v[125:128], off offset:16
	global_store_b128 v[129:130], v[113:116], off offset:32
	global_store_b128 v[129:130], v[117:120], off offset:48
	global_store_b128 v[129:130], v[105:108], off offset:64
	global_store_b128 v[129:130], v[109:112], off offset:80
	v_dual_mul_f32 v85, v139, v85 :: v_dual_mul_f32 v86, v139, v86
	v_dual_mul_f32 v87, v139, v87 :: v_dual_mul_f32 v88, v139, v88
	s_clause 0x5
	global_store_b128 v[129:130], v[97:100], off offset:96
	global_store_b128 v[129:130], v[101:104], off offset:112
	global_store_b128 v[129:130], v[89:92], off offset:128
	global_store_b128 v[129:130], v[93:96], off offset:144
	global_store_b128 v[129:130], v[81:84], off offset:160
	global_store_b128 v[129:130], v[85:88], off offset:176
	v_dual_mul_f32 v61, v139, v61 :: v_dual_mul_f32 v62, v139, v62
	v_dual_mul_f32 v63, v139, v63 :: v_dual_mul_f32 v64, v139, v64
	s_clause 0x5
	global_store_b128 v[129:130], v[73:76], off offset:192
	global_store_b128 v[129:130], v[77:80], off offset:208
	global_store_b128 v[129:130], v[65:68], off offset:224
	global_store_b128 v[129:130], v[69:72], off offset:240
	global_store_b128 v[129:130], v[57:60], off offset:256
	global_store_b128 v[129:130], v[61:64], off offset:272
	v_dual_mul_f32 v37, v139, v37 :: v_dual_mul_f32 v38, v139, v38
	v_dual_mul_f32 v39, v139, v39 :: v_dual_mul_f32 v40, v139, v40
	s_clause 0xe
	global_store_b128 v[129:130], v[49:52], off offset:288
	global_store_b128 v[129:130], v[53:56], off offset:304
	global_store_b128 v[129:130], v[41:44], off offset:320
	global_store_b128 v[129:130], v[45:48], off offset:336
	global_store_b128 v[129:130], v[33:36], off offset:352
	global_store_b128 v[129:130], v[37:40], off offset:368
	global_store_b128 v[129:130], v[17:20], off offset:416
	global_store_b128 v[129:130], v[21:24], off offset:432
	global_store_b128 v[129:130], v[9:12], off offset:448
	global_store_b128 v[129:130], v[13:16], off offset:464
	global_store_b128 v[129:130], v[0:3], off offset:480
	global_store_b128 v[129:130], v[4:7], off offset:496
	global_store_b128 v[129:130], v[141:144], off offset:512
	global_store_b128 v[129:130], v[145:148], off offset:528
	global_store_b128 v[129:130], v[137:140], off offset:544
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end1:
	.size	_Z12floor_kernelILi1EEvPKjPfi, .Lfunc_end1-_Z12floor_kernelILi1EEvPKjPfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12floor_kernelILi1EEvPKjPfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 185
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_Z12floor_kernelILi1EEvPKjPfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z12floor_kernelILi1EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi1EEvPKjPfi,comdat
                                        ; -- End function
	.set .L_Z12floor_kernelILi1EEvPKjPfi.num_vgpr, 185
	.set .L_Z12floor_kernelILi1EEvPKjPfi.num_agpr, 0
	.set .L_Z12floor_kernelILi1EEvPKjPfi.numbered_sgpr, 16
	.set .L_Z12floor_kernelILi1EEvPKjPfi.num_named_barrier, 0
	.set .L_Z12floor_kernelILi1EEvPKjPfi.private_seg_size, 0
	.set .L_Z12floor_kernelILi1EEvPKjPfi.uses_vcc, 1
	.set .L_Z12floor_kernelILi1EEvPKjPfi.uses_flat_scratch, 0
	.set .L_Z12floor_kernelILi1EEvPKjPfi.has_dyn_sized_stack, 0
	.set .L_Z12floor_kernelILi1EEvPKjPfi.has_recursion, 0
	.set .L_Z12floor_kernelILi1EEvPKjPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4564
; TotalNumSgprs: 18
; NumVgprs: 185
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 185
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z12floor_kernelILi2EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi2EEvPKjPfi,comdat
	.protected	_Z12floor_kernelILi2EEvPKjPfi ; -- Begin function _Z12floor_kernelILi2EEvPKjPfi
	.globl	_Z12floor_kernelILi2EEvPKjPfi
	.p2align	8
	.type	_Z12floor_kernelILi2EEvPKjPfi,@function
_Z12floor_kernelILi2EEvPKjPfi:          ; @_Z12floor_kernelILi2EEvPKjPfi
	.cfi_startproc
; %bb.0:                                ; %.preheader182
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_lshl_add_u32 v140, v0, 2, 0
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s2, s[0:1], 0x10
	s_mov_b32 s8, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s9, s8
	ds_store_b32 v140, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s12, s8
	s_mov_b32 s13, s8
	s_mov_b32 s14, s8
	s_mov_b32 s15, s8
	v_lshlrev_b32_e32 v1, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0xf8, v1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_load_b64 v[149:150], v1, s[4:5]
	v_dual_mov_b32 v1, s8 :: v_dual_mov_b32 v6, s13
	v_dual_mov_b32 v3, s10 :: v_dual_mov_b32 v8, s15
	v_mov_b32_e32 v2, s9
	v_dual_mov_b32 v4, s11 :: v_dual_mov_b32 v5, s12
	v_mov_b32_e32 v7, s14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v128, v8
	v_mov_b32_e32 v120, v8
	v_mov_b32_e32 v112, v8
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v103, v7
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v95, v7
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v87, v7
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	v_dual_mov_b32 v127, v7 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v122, v2
	v_mov_b32_e32 v121, v1
	v_dual_mov_b32 v119, v7 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v114, v2
	v_mov_b32_e32 v113, v1
	v_dual_mov_b32 v111, v7 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v10, v2
	v_mov_b32_e32 v9, v1
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
	s_cbranch_scc1 .LBB2_5
; %bb.1:                                ; %.preheader180.lr.ph
	v_lshrrev_b32_e32 v129, 1, v0
	v_dual_mov_b32 v171, 1.0 :: v_dual_mov_b32 v172, 0xff800000
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v151, v149 :: v_dual_mov_b32 v154, v150
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_and_b32_e32 v129, 8, v129
	v_mov_b32_e32 v153, v149
	s_mov_b32 s3, 0x76543210
	v_mov_b32_e32 v152, v150
	v_or_b32_e32 v131, 3, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v131, v131
	v_mul_f32_e32 v156, 0x3a83126f, v131
	v_or_b32_e32 v130, 4, v129
	v_or_b32_e32 v132, 2, v129
	v_or_b32_e32 v133, 1, v129
	v_or_b32_e32 v135, 7, v129
	v_or_b32_e32 v136, 6, v129
	v_cvt_f32_ubyte0_e32 v130, v130
	v_add_nc_u32_e32 v134, 8, v129
	v_or_b32_e32 v129, 5, v129
	v_cvt_f32_ubyte0_e32 v133, v133
	v_cvt_f32_ubyte0_e32 v136, v136
	v_cvt_f32_ubyte0_e32 v132, v132
	v_cvt_f32_ubyte0_e32 v135, v135
	v_cvt_f32_ubyte0_e32 v129, v129
	v_mul_f32_e32 v155, 0x3a83126f, v130
	v_cvt_f32_ubyte0_e32 v134, v134
	v_mul_f32_e32 v161, 0x3a83126f, v136
	v_dual_mul_f32 v157, 0x3a83126f, v132 :: v_dual_mul_f32 v158, 0x3a83126f, v133
	v_mul_f32_e32 v162, 0x3a83126f, v129
	v_dual_mul_f32 v167, 0x3c23d70a, v129 :: v_dual_mul_f32 v168, 0x3c23d70a, v136
	v_dual_mov_b32 v129, 0 :: v_dual_mul_f32 v170, 0x3c23d70a, v134
	v_dual_mul_f32 v159, 0x3a83126f, v134 :: v_dual_mul_f32 v160, 0x3a83126f, v135
	v_dual_mul_f32 v163, 0x3c23d70a, v133 :: v_dual_mul_f32 v164, 0x3c23d70a, v132
	v_dual_mul_f32 v165, 0x3c23d70a, v131 :: v_dual_mul_f32 v166, 0x3c23d70a, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v169, 0x3c23d70a, v135 :: v_dual_mov_b32 v130, v129
	v_dual_mov_b32 v131, v129 :: v_dual_mov_b32 v132, v129
	v_dual_mov_b32 v133, v129 :: v_dual_mov_b32 v134, v129
	v_dual_mov_b32 v135, v129 :: v_dual_mov_b32 v136, v129
	v_mov_b32_e32 v173, v129
	s_branch .LBB2_3
.LBB2_2:                                ; %.preheader178.preheader
                                        ;   in Loop: Header=BB2_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_sub_f32 v175, v172, v137 :: v_dual_add_f32 v138, v138, v174
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v172
	v_rcp_f32_e32 v176, v139
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v175, 0x3fb8aa3b, v175
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_exp_f32_e32 v177, v175
	v_mov_b16_e64 v175.l, v129.l
	v_mov_b16_e64 v175.h, 0
	v_dual_mul_f32 v179, v176, v143 :: v_dual_mul_f32 v180, v176, v144
	v_dual_mul_f32 v181, v176, v145 :: v_dual_mul_f32 v184, v176, v148
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mov_b16_e64 v174.l, v175.l
	v_mov_b16_e64 v174.h, v175.h
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v172, 0, v177, vcc_lo
	v_dual_mul_f32 v177, v176, v141 :: v_dual_mul_f32 v178, v176, v142
	v_mul_f32_e32 v183, v176, v147
	v_cvt_pk_fp8_f32 v174.h, v179, v180
	v_dual_mul_f32 v171, v171, v172 :: v_dual_mul_f32 v182, v176, v146
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_cvt_pk_fp8_f32 v174.l, v177, v178
	v_fmac_f32_e32 v138, v173, v172
	v_cvt_pk_fp8_f32 v175.h, v183, v184
	v_dual_mul_f32 v171, v171, v176 :: v_dual_mov_b32 v172, v137
	v_cvt_pk_fp8_f32 v175.l, v181, v182
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v128, v171, v128 :: v_dual_mul_f32 v127, v171, v127
	v_dual_mul_f32 v126, v171, v126 :: v_dual_mul_f32 v125, v171, v125
	v_dual_mul_f32 v124, v171, v124 :: v_dual_mul_f32 v123, v171, v123
	v_dual_mul_f32 v122, v171, v122 :: v_dual_mul_f32 v121, v171, v121
	v_dual_mul_f32 v120, v171, v120 :: v_dual_mul_f32 v119, v171, v119
	v_dual_mul_f32 v118, v171, v118 :: v_dual_mul_f32 v117, v171, v117
	v_dual_mul_f32 v116, v171, v116 :: v_dual_mul_f32 v115, v171, v115
	v_dual_mul_f32 v114, v171, v114 :: v_dual_mul_f32 v113, v171, v113
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
	v_dual_mul_f32 v112, v171, v112 :: v_dual_mul_f32 v111, v171, v111
	v_dual_mul_f32 v110, v171, v110 :: v_dual_mul_f32 v109, v171, v109
	v_dual_mul_f32 v108, v171, v108 :: v_dual_mul_f32 v107, v171, v107
	v_dual_mul_f32 v106, v171, v106 :: v_dual_mul_f32 v105, v171, v105
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[149:150], v[174:175], v[121:128]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v104, v171, v104 :: v_dual_mul_f32 v103, v171, v103
	v_dual_mul_f32 v102, v171, v102 :: v_dual_mul_f32 v101, v171, v101
	v_dual_mul_f32 v100, v171, v100 :: v_dual_mul_f32 v99, v171, v99
	v_dual_mul_f32 v98, v171, v98 :: v_dual_mul_f32 v97, v171, v97
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[149:150], v[174:175], v[113:120]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v96, v171, v96 :: v_dual_mul_f32 v95, v171, v95
	v_dual_mul_f32 v94, v171, v94 :: v_dual_mul_f32 v93, v171, v93
	v_dual_mul_f32 v92, v171, v92 :: v_dual_mul_f32 v91, v171, v91
	v_dual_mul_f32 v90, v171, v90 :: v_dual_mul_f32 v89, v171, v89
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[149:150], v[174:175], v[105:112]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v88, v171, v88 :: v_dual_mul_f32 v87, v171, v87
	v_dual_mul_f32 v86, v171, v86 :: v_dual_mul_f32 v85, v171, v85
	v_dual_mul_f32 v84, v171, v84 :: v_dual_mul_f32 v83, v171, v83
	v_dual_mul_f32 v82, v171, v82 :: v_dual_mul_f32 v81, v171, v81
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[149:150], v[174:175], v[97:104]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v80, v171, v80 :: v_dual_mul_f32 v79, v171, v79
	v_dual_mul_f32 v78, v171, v78 :: v_dual_mul_f32 v77, v171, v77
	v_dual_mul_f32 v76, v171, v76 :: v_dual_mul_f32 v75, v171, v75
	v_dual_mul_f32 v74, v171, v74 :: v_dual_mul_f32 v73, v171, v73
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[149:150], v[174:175], v[89:96]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v72, v171, v72 :: v_dual_mul_f32 v71, v171, v71
	v_dual_mul_f32 v70, v171, v70 :: v_dual_mul_f32 v69, v171, v69
	v_dual_mul_f32 v68, v171, v68 :: v_dual_mul_f32 v67, v171, v67
	v_dual_mul_f32 v66, v171, v66 :: v_dual_mul_f32 v65, v171, v65
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[149:150], v[174:175], v[81:88]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v64, v171, v64 :: v_dual_mul_f32 v63, v171, v63
	v_dual_mul_f32 v62, v171, v62 :: v_dual_mul_f32 v61, v171, v61
	v_dual_mul_f32 v60, v171, v60 :: v_dual_mul_f32 v59, v171, v59
	v_dual_mul_f32 v58, v171, v58 :: v_dual_mul_f32 v57, v171, v57
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[149:150], v[174:175], v[73:80]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v56, v171, v56 :: v_dual_mul_f32 v55, v171, v55
	v_dual_mul_f32 v54, v171, v54 :: v_dual_mul_f32 v53, v171, v53
	v_dual_mul_f32 v52, v171, v52 :: v_dual_mul_f32 v51, v171, v51
	v_dual_mul_f32 v50, v171, v50 :: v_dual_mul_f32 v49, v171, v49
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[149:150], v[174:175], v[65:72]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v48, v171, v48 :: v_dual_mul_f32 v47, v171, v47
	v_dual_mul_f32 v46, v171, v46 :: v_dual_mul_f32 v45, v171, v45
	v_dual_mul_f32 v44, v171, v44 :: v_dual_mul_f32 v43, v171, v43
	v_dual_mul_f32 v42, v171, v42 :: v_dual_mul_f32 v41, v171, v41
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[149:150], v[174:175], v[57:64]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v40, v171, v40 :: v_dual_mul_f32 v39, v171, v39
	v_dual_mul_f32 v38, v171, v38 :: v_dual_mul_f32 v37, v171, v37
	v_dual_mul_f32 v36, v171, v36 :: v_dual_mul_f32 v35, v171, v35
	v_dual_mul_f32 v34, v171, v34 :: v_dual_mul_f32 v33, v171, v33
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[149:150], v[174:175], v[49:56]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v32, v171, v32 :: v_dual_mul_f32 v31, v171, v31
	v_dual_mul_f32 v30, v171, v30 :: v_dual_mul_f32 v29, v171, v29
	v_dual_mul_f32 v28, v171, v28 :: v_dual_mul_f32 v27, v171, v27
	v_dual_mul_f32 v26, v171, v26 :: v_dual_mul_f32 v25, v171, v25
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[149:150], v[174:175], v[41:48]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v24, v171, v24 :: v_dual_mul_f32 v23, v171, v23
	v_dual_mul_f32 v22, v171, v22 :: v_dual_mul_f32 v21, v171, v21
	v_dual_mul_f32 v20, v171, v20 :: v_dual_mul_f32 v19, v171, v19
	v_dual_mul_f32 v18, v171, v18 :: v_dual_mul_f32 v17, v171, v17
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[149:150], v[174:175], v[33:40]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v16, v171, v16 :: v_dual_mul_f32 v15, v171, v15
	v_dual_mul_f32 v14, v171, v14 :: v_dual_mul_f32 v13, v171, v13
	v_dual_mul_f32 v12, v171, v12 :: v_dual_mul_f32 v11, v171, v11
	v_dual_mul_f32 v10, v171, v10 :: v_dual_mul_f32 v9, v171, v9
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[149:150], v[174:175], v[25:32]
	;;#ASMSTART
	;;#ASMEND
	v_dual_mul_f32 v8, v171, v8 :: v_dual_mul_f32 v7, v171, v7
	v_dual_mul_f32 v6, v171, v6 :: v_dual_mul_f32 v5, v171, v5
	v_dual_mul_f32 v4, v171, v4 :: v_dual_mul_f32 v3, v171, v3
	v_dual_mul_f32 v2, v171, v2 :: v_dual_mul_f32 v1, v171, v1
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[149:150], v[174:175], v[17:24]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[149:150], v[174:175], v[9:16]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[149:150], v[174:175], v[1:8]
	v_mov_b32_e32 v173, v138
	v_mov_b32_e32 v171, v139
	s_add_co_i32 s2, s2, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .LBB2_6
.LBB2_3:                                ; %.preheader180
                                        ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v148, v136 :: v_dual_mov_b32 v147, v135
	v_dual_mov_b32 v146, v134 :: v_dual_mov_b32 v145, v133
	v_dual_mov_b32 v144, v132 :: v_dual_mov_b32 v143, v131
	v_dual_mov_b32 v142, v130 :: v_dual_mov_b32 v141, v129
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[141:148], v[151:152], v[153:154], v[141:148]
	s_mov_b32 s4, exec_lo
	v_dual_mul_f32 v138, 0x3e000000, v142 :: v_dual_mul_f32 v139, 0x3e000000, v141
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v141, 0x3e000000, v143 :: v_dual_mul_f32 v142, 0x3e000000, v144
	v_mul_f32_e32 v144, 0x3e000000, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v137, v138, v157
	v_mul_f32_e32 v143, v139, v158
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v145, 0x3e000000, v146 :: v_dual_mul_f32 v146, v142, v155
	v_mul_f32_e32 v174, v141, v156
	v_max3_num_f32 v137, v143, 0xff800000, v137
	v_mul_f32_e32 v143, 0x3e000000, v147
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v147, 0x3e000000, v148 :: v_dual_mul_f32 v148, v145, v161
	v_mul_f32_e32 v175, v144, v162
	v_max3_num_f32 v137, v137, v174, v146
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v174, v143, v160
	v_mul_f32_e32 v146, v147, v159
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v137, v137, v175, v148
	v_max3_num_f32 v137, v137, v174, v146
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_e32 v146, v137
	v_permlanex16_b32 v146, v146, s3, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v137, v172, v137, v146
	v_fma_f32 v139, v139, v158, -v137
	v_fma_f32 v138, v138, v157, -v137
	v_fma_f32 v141, v141, v156, -v137
	v_fma_f32 v142, v142, v155, -v137
	v_fma_f32 v144, v144, v162, -v137
	v_fma_f32 v143, v143, v160, -v137
	v_dual_mul_f32 v139, 0x3fb8aa3b, v139 :: v_dual_mul_f32 v138, 0x3fb8aa3b, v138
	v_mul_f32_e32 v141, 0x3fb8aa3b, v141
	v_fma_f32 v145, v145, v161, -v137
	v_mul_f32_e32 v142, 0x3fb8aa3b, v142
	v_dual_mul_f32 v144, 0x3fb8aa3b, v144 :: v_dual_mul_f32 v143, 0x3fb8aa3b, v143
	v_exp_f32_e32 v139, v139
	v_exp_f32_e32 v138, v138
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v137
	v_fma_f32 v146, v147, v159, -v137
	v_mul_f32_e32 v145, 0x3fb8aa3b, v145
	v_exp_f32_e32 v141, v141
	v_exp_f32_e32 v142, v142
	v_exp_f32_e32 v143, v143
	v_exp_f32_e32 v144, v144
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v139, v139, 0, vcc_lo
	v_cndmask_b32_e64 v138, v138, 0, vcc_lo
	v_mul_f32_e32 v146, 0x3fb8aa3b, v146
	v_exp_f32_e32 v145, v145
	v_cndmask_b32_e64 v147, v141, 0, vcc_lo
	v_cndmask_b32_e64 v174, v142, 0, vcc_lo
	v_dual_mul_f32 v141, v163, v139 :: v_dual_mul_f32 v142, v164, v138
	v_cndmask_b32_e64 v177, v143, 0, vcc_lo
	v_add_f32_e32 v138, v139, v138
	v_exp_f32_e32 v146, v146
	v_cndmask_b32_e64 v175, v144, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_2)
	v_cndmask_b32_e64 v176, v145, 0, vcc_lo
	v_dual_mul_f32 v143, v165, v147 :: v_dual_mul_f32 v144, v166, v174
	v_max3_num_f32 v139, v141, 0, v142
	v_add_f32_e32 v138, v138, v147
	v_mul_f32_e32 v145, v167, v175
	v_mul_f32_e32 v147, v169, v177
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v178, v146, 0, vcc_lo
	v_mul_f32_e32 v146, v168, v176
	v_max3_num_f32 v139, v139, v143, v144
	v_add_f32_e32 v138, v138, v174
	v_mul_f32_e32 v148, v170, v178
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v139, v139, v145, v146
	v_add_f32_e32 v138, v138, v175
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v139, v139, v147, v148
	v_add_f32_e32 v138, v138, v176
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v174, v139
	v_add_f32_e32 v138, v138, v177
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_permlanex16_b32 v174, v174, s3, 0xfedcba98
	v_add_f32_e32 v138, v138, v178
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v175, v174, v174
	v_dual_mov_b32 v174, v138 :: v_dual_max_num_f32 v175, v139, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v174, v174, s3, 0xfedcba98
	v_mov_b32_e32 v139, v171
	v_cmpx_lt_f32_e32 0, v175
	s_cbranch_execz .LBB2_2
; %bb.4:                                ;   in Loop: Header=BB2_3 Depth=1
	v_div_scale_f32 v139, null, 0x43e00000, 0x43e00000, v175
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v176, v139
	v_fma_f32 v177, -v139, v176, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v176, v177, v176
	v_div_scale_f32 v177, vcc_lo, v175, 0x43e00000, v175
	v_mul_f32_e32 v178, v177, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v179, -v139, v178, v177
	v_fmac_f32_e32 v178, v179, v176
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v139, -v139, v178, v177
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v139, v139, v176, v178
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v139, v139, 0x43e00000, v175
	v_max_num_f32_e32 v139, 0x1f800000, v139
	s_branch .LBB2_2
.LBB2_5:
	v_dual_mov_b32 v139, 1.0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v137, 0xff800000 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v138, 0
	v_mov_b32_e32 v141, 0
.LBB2_6:                                ; %Flow
	s_load_b32 s0, s[0:1], 0x24
	v_dual_mul_f32 v25, v139, v25 :: v_dual_mul_f32 v26, v139, v26
	v_dual_mul_f32 v27, v139, v27 :: v_dual_mul_f32 v28, v139, v28
	v_dual_mul_f32 v29, v139, v29 :: v_dual_mul_f32 v30, v139, v30
	v_dual_mul_f32 v31, v139, v31 :: v_dual_mul_f32 v32, v139, v32
	v_dual_mul_f32 v121, v139, v121 :: v_dual_mul_f32 v122, v139, v122
	v_dual_mul_f32 v123, v139, v123 :: v_dual_mul_f32 v124, v139, v124
	v_dual_mul_f32 v97, v139, v97 :: v_dual_mul_f32 v98, v139, v98
	v_dual_mul_f32 v99, v139, v99 :: v_dual_mul_f32 v100, v139, v100
	v_dual_mul_f32 v73, v139, v73 :: v_dual_mul_f32 v74, v139, v74
	v_dual_mul_f32 v75, v139, v75 :: v_dual_mul_f32 v76, v139, v76
	s_wait_kmcnt 0x0
	s_and_b32 s0, 0xffff, s0
	v_dual_mul_f32 v49, v139, v49 :: v_dual_mul_f32 v50, v139, v50
	v_mad_co_u64_u32 v[129:130], null, ttmp9, s0, v[0:1]
	v_dual_mul_f32 v51, v139, v51 :: v_dual_mul_f32 v52, v139, v52
	v_dual_mul_f32 v17, v139, v17 :: v_dual_mul_f32 v18, v139, v18
	v_dual_mul_f32 v19, v139, v19 :: v_dual_mul_f32 v20, v139, v20
	v_dual_mul_f32 v9, v139, v9 :: v_dual_mul_f32 v10, v139, v10
	v_mul_lo_u32 v129, 0x8c, v129
	v_dual_mul_f32 v11, v139, v11 :: v_dual_mul_f32 v12, v139, v12
	v_dual_mul_f32 v0, v139, v1 :: v_dual_mul_f32 v1, v139, v2
	v_dual_mul_f32 v2, v139, v3 :: v_dual_mul_f32 v3, v139, v4
	v_dual_mul_f32 v125, v139, v125 :: v_dual_mul_f32 v126, v139, v126
	v_ashrrev_i32_e32 v130, 31, v129
	v_dual_mul_f32 v127, v139, v127 :: v_dual_mul_f32 v128, v139, v128
	v_dual_mul_f32 v101, v139, v101 :: v_dual_mul_f32 v102, v139, v102
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[129:130], 2, v[129:130]
	v_dual_mul_f32 v103, v139, v103 :: v_dual_mul_f32 v104, v139, v104
	v_dual_mul_f32 v77, v139, v77 :: v_dual_mul_f32 v78, v139, v78
	v_dual_mul_f32 v79, v139, v79 :: v_dual_mul_f32 v80, v139, v80
	v_add_co_u32 v129, vcc_lo, s6, v129
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v130, null, s7, v130, vcc_lo
	s_clause 0x1
	global_store_b128 v[129:130], v[25:28], off offset:384
	global_store_b128 v[129:130], v[29:32], off offset:400
	ds_load_b32 v25, v140
	v_dual_mul_f32 v53, v139, v53 :: v_dual_mul_f32 v54, v139, v54
	v_dual_mul_f32 v55, v139, v55 :: v_dual_mul_f32 v56, v139, v56
	v_dual_mul_f32 v21, v139, v21 :: v_dual_mul_f32 v22, v139, v22
	v_dual_mul_f32 v23, v139, v23 :: v_dual_mul_f32 v24, v139, v24
	v_dual_mul_f32 v13, v139, v13 :: v_dual_mul_f32 v14, v139, v14
	v_dual_mul_f32 v15, v139, v15 :: v_dual_mul_f32 v16, v139, v16
	v_dual_mul_f32 v4, v139, v5 :: v_dual_mul_f32 v5, v139, v6
	v_dual_mul_f32 v6, v139, v7 :: v_dual_mul_f32 v7, v139, v8
	v_dual_mul_f32 v113, v139, v113 :: v_dual_mul_f32 v114, v139, v114
	v_dual_mul_f32 v115, v139, v115 :: v_dual_mul_f32 v116, v139, v116
	v_dual_mul_f32 v89, v139, v89 :: v_dual_mul_f32 v90, v139, v90
	v_dual_mul_f32 v91, v139, v91 :: v_dual_mul_f32 v92, v139, v92
	v_dual_mul_f32 v65, v139, v65 :: v_dual_mul_f32 v66, v139, v66
	v_dual_mul_f32 v67, v139, v67 :: v_dual_mul_f32 v68, v139, v68
	v_dual_mul_f32 v41, v139, v41 :: v_dual_mul_f32 v42, v139, v42
	v_dual_mul_f32 v43, v139, v43 :: v_dual_mul_f32 v44, v139, v44
	v_dual_mul_f32 v117, v139, v117 :: v_dual_mul_f32 v118, v139, v118
	v_dual_mul_f32 v119, v139, v119 :: v_dual_mul_f32 v120, v139, v120
	v_dual_mul_f32 v93, v139, v93 :: v_dual_mul_f32 v94, v139, v94
	v_dual_mul_f32 v95, v139, v95 :: v_dual_mul_f32 v96, v139, v96
	v_dual_mul_f32 v69, v139, v69 :: v_dual_mul_f32 v70, v139, v70
	v_dual_mul_f32 v71, v139, v71 :: v_dual_mul_f32 v72, v139, v72
	v_dual_mul_f32 v45, v139, v45 :: v_dual_mul_f32 v46, v139, v46
	v_dual_mul_f32 v47, v139, v47 :: v_dual_mul_f32 v48, v139, v48
	v_dual_mul_f32 v105, v139, v105 :: v_dual_mul_f32 v106, v139, v106
	v_dual_mul_f32 v107, v139, v107 :: v_dual_mul_f32 v108, v139, v108
	v_dual_mul_f32 v81, v139, v81 :: v_dual_mul_f32 v82, v139, v82
	v_dual_mul_f32 v83, v139, v83 :: v_dual_mul_f32 v84, v139, v84
	v_dual_mul_f32 v57, v139, v57 :: v_dual_mul_f32 v58, v139, v58
	v_dual_mul_f32 v59, v139, v59 :: v_dual_mul_f32 v60, v139, v60
	v_dual_mul_f32 v33, v139, v33 :: v_dual_mul_f32 v34, v139, v34
	v_dual_mul_f32 v35, v139, v35 :: v_dual_mul_f32 v36, v139, v36
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v140, v25
	v_dual_mul_f32 v109, v139, v109 :: v_dual_mul_f32 v110, v139, v110
	v_dual_mul_f32 v111, v139, v111 :: v_dual_mul_f32 v112, v139, v112
	s_clause 0x5
	global_store_b128 v[129:130], v[121:124], off
	global_store_b128 v[129:130], v[125:128], off offset:16
	global_store_b128 v[129:130], v[113:116], off offset:32
	global_store_b128 v[129:130], v[117:120], off offset:48
	global_store_b128 v[129:130], v[105:108], off offset:64
	global_store_b128 v[129:130], v[109:112], off offset:80
	v_dual_mul_f32 v85, v139, v85 :: v_dual_mul_f32 v86, v139, v86
	v_dual_mul_f32 v87, v139, v87 :: v_dual_mul_f32 v88, v139, v88
	s_clause 0x5
	global_store_b128 v[129:130], v[97:100], off offset:96
	global_store_b128 v[129:130], v[101:104], off offset:112
	global_store_b128 v[129:130], v[89:92], off offset:128
	global_store_b128 v[129:130], v[93:96], off offset:144
	global_store_b128 v[129:130], v[81:84], off offset:160
	global_store_b128 v[129:130], v[85:88], off offset:176
	v_dual_mul_f32 v61, v139, v61 :: v_dual_mul_f32 v62, v139, v62
	v_dual_mul_f32 v63, v139, v63 :: v_dual_mul_f32 v64, v139, v64
	s_clause 0x5
	global_store_b128 v[129:130], v[73:76], off offset:192
	global_store_b128 v[129:130], v[77:80], off offset:208
	global_store_b128 v[129:130], v[65:68], off offset:224
	global_store_b128 v[129:130], v[69:72], off offset:240
	global_store_b128 v[129:130], v[57:60], off offset:256
	global_store_b128 v[129:130], v[61:64], off offset:272
	v_dual_mul_f32 v37, v139, v37 :: v_dual_mul_f32 v38, v139, v38
	v_dual_mul_f32 v39, v139, v39 :: v_dual_mul_f32 v40, v139, v40
	s_clause 0xe
	global_store_b128 v[129:130], v[49:52], off offset:288
	global_store_b128 v[129:130], v[53:56], off offset:304
	global_store_b128 v[129:130], v[41:44], off offset:320
	global_store_b128 v[129:130], v[45:48], off offset:336
	global_store_b128 v[129:130], v[33:36], off offset:352
	global_store_b128 v[129:130], v[37:40], off offset:368
	global_store_b128 v[129:130], v[17:20], off offset:416
	global_store_b128 v[129:130], v[21:24], off offset:432
	global_store_b128 v[129:130], v[9:12], off offset:448
	global_store_b128 v[129:130], v[13:16], off offset:464
	global_store_b128 v[129:130], v[0:3], off offset:480
	global_store_b128 v[129:130], v[4:7], off offset:496
	global_store_b128 v[129:130], v[141:144], off offset:512
	global_store_b128 v[129:130], v[145:148], off offset:528
	global_store_b128 v[129:130], v[137:140], off offset:544
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end2:
	.size	_Z12floor_kernelILi2EEvPKjPfi, .Lfunc_end2-_Z12floor_kernelILi2EEvPKjPfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12floor_kernelILi2EEvPKjPfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 185
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_Z12floor_kernelILi2EEvPKjPfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z12floor_kernelILi2EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi2EEvPKjPfi,comdat
                                        ; -- End function
	.set .L_Z12floor_kernelILi2EEvPKjPfi.num_vgpr, 185
	.set .L_Z12floor_kernelILi2EEvPKjPfi.num_agpr, 0
	.set .L_Z12floor_kernelILi2EEvPKjPfi.numbered_sgpr, 16
	.set .L_Z12floor_kernelILi2EEvPKjPfi.num_named_barrier, 0
	.set .L_Z12floor_kernelILi2EEvPKjPfi.private_seg_size, 0
	.set .L_Z12floor_kernelILi2EEvPKjPfi.uses_vcc, 1
	.set .L_Z12floor_kernelILi2EEvPKjPfi.uses_flat_scratch, 0
	.set .L_Z12floor_kernelILi2EEvPKjPfi.has_dyn_sized_stack, 0
	.set .L_Z12floor_kernelILi2EEvPKjPfi.has_recursion, 0
	.set .L_Z12floor_kernelILi2EEvPKjPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3768
; TotalNumSgprs: 18
; NumVgprs: 185
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 185
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z12floor_kernelILi3EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi3EEvPKjPfi,comdat
	.protected	_Z12floor_kernelILi3EEvPKjPfi ; -- Begin function _Z12floor_kernelILi3EEvPKjPfi
	.globl	_Z12floor_kernelILi3EEvPKjPfi
	.p2align	8
	.type	_Z12floor_kernelILi3EEvPKjPfi,@function
_Z12floor_kernelILi3EEvPKjPfi:          ; @_Z12floor_kernelILi3EEvPKjPfi
	.cfi_startproc
; %bb.0:                                ; %.preheader87
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_lshl_add_u32 v151, v0, 2, 0
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s2, s[0:1], 0x10
	s_mov_b32 s8, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s9, s8
	ds_store_b32 v151, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s12, s8
	s_mov_b32 s13, s8
	s_mov_b32 s14, s8
	s_mov_b32 s15, s8
	v_lshlrev_b32_e32 v1, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0xf8, v1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_load_b64 v[145:146], v1, s[4:5]
	v_dual_mov_b32 v1, s8 :: v_dual_mov_b32 v6, s13
	v_dual_mov_b32 v3, s10 :: v_dual_mov_b32 v8, s15
	v_mov_b32_e32 v2, s9
	v_dual_mov_b32 v4, s11 :: v_dual_mov_b32 v5, s12
	v_mov_b32_e32 v7, s14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mov_b32_e32 v16, v8
	v_mov_b32_e32 v24, v8
	v_mov_b32_e32 v32, v8
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v87, v7
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v95, v7
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v103, v7
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v111, v7
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v119, v7
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v127, v7
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v10, v2
	v_mov_b32_e32 v9, v1
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v18, v2
	v_mov_b32_e32 v17, v1
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v58, v2
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v70, v6
	v_dual_mov_b32 v69, v5 :: v_dual_mov_b32 v68, v4
	v_dual_mov_b32 v67, v3 :: v_dual_mov_b32 v66, v2
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v78, v6
	v_dual_mov_b32 v77, v5 :: v_dual_mov_b32 v76, v4
	v_dual_mov_b32 v75, v3 :: v_dual_mov_b32 v74, v2
	v_dual_mov_b32 v73, v1 :: v_dual_mov_b32 v86, v6
	v_dual_mov_b32 v85, v5 :: v_dual_mov_b32 v84, v4
	v_dual_mov_b32 v83, v3 :: v_dual_mov_b32 v82, v2
	v_dual_mov_b32 v81, v1 :: v_dual_mov_b32 v94, v6
	v_dual_mov_b32 v93, v5 :: v_dual_mov_b32 v92, v4
	v_dual_mov_b32 v91, v3 :: v_dual_mov_b32 v90, v2
	v_dual_mov_b32 v89, v1 :: v_dual_mov_b32 v102, v6
	v_dual_mov_b32 v101, v5 :: v_dual_mov_b32 v100, v4
	v_dual_mov_b32 v99, v3 :: v_dual_mov_b32 v98, v2
	v_dual_mov_b32 v97, v1 :: v_dual_mov_b32 v110, v6
	v_dual_mov_b32 v109, v5 :: v_dual_mov_b32 v108, v4
	v_dual_mov_b32 v107, v3 :: v_dual_mov_b32 v106, v2
	v_dual_mov_b32 v105, v1 :: v_dual_mov_b32 v118, v6
	v_dual_mov_b32 v117, v5 :: v_dual_mov_b32 v116, v4
	v_dual_mov_b32 v115, v3 :: v_dual_mov_b32 v114, v2
	v_dual_mov_b32 v113, v1 :: v_dual_mov_b32 v126, v6
	v_dual_mov_b32 v125, v5 :: v_dual_mov_b32 v124, v4
	v_dual_mov_b32 v123, v3 :: v_dual_mov_b32 v122, v2
	v_mov_b32_e32 v121, v1
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
	s_cbranch_scc1 .LBB3_3
; %bb.1:                                ; %.preheader85.preheader
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v148, v146
	v_dual_mov_b32 v149, v145 :: v_dual_mov_b32 v150, v146
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v147, v145 :: v_dual_mov_b32 v144, v137
	v_dual_mov_b32 v138, v137 :: v_dual_mov_b32 v139, v137
	v_dual_mov_b32 v140, v137 :: v_dual_mov_b32 v141, v137
	v_dual_mov_b32 v142, v137 :: v_dual_mov_b32 v143, v137
.LBB3_2:                                ; %.preheader85
                                        ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v129, v137 :: v_dual_mov_b32 v130, v138
	v_dual_mov_b32 v131, v139 :: v_dual_mov_b32 v132, v140
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v133, v141 :: v_dual_mov_b32 v134, v142
	v_dual_mov_b32 v135, v143 :: v_dual_mov_b32 v136, v144
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[145:146], v[149:150], v[9:16]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[145:146], v[149:150], v[17:24]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[145:146], v[149:150], v[25:32]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[145:146], v[149:150], v[33:40]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[145:146], v[149:150], v[41:48]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[145:146], v[149:150], v[49:56]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[145:146], v[149:150], v[57:64]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[145:146], v[149:150], v[65:72]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[145:146], v[149:150], v[73:80]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[145:146], v[149:150], v[81:88]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[145:146], v[149:150], v[89:96]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[145:146], v[149:150], v[97:104]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[145:146], v[149:150], v[105:112]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[145:146], v[149:150], v[113:120]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[145:146], v[149:150], v[121:128]
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[147:148], v[149:150], v[129:136]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[145:146], v[149:150], v[1:8]
	s_add_co_i32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB3_2
	s_branch .LBB3_4
.LBB3_3:
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v129, 0
.LBB3_4:                                ; %Flow
	s_load_b32 s0, s[0:1], 0x24
	s_wait_kmcnt 0x0
	s_and_b32 s0, 0xffff, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[137:138], null, ttmp9, s0, v[0:1]
	ds_load_b32 v0, v151
	v_mul_lo_u32 v137, 0x8c, v137
	v_ashrrev_i32_e32 v138, 31, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[137:138], 2, v[137:138]
	v_add_co_u32 v137, vcc_lo, s6, v137
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v138, null, s7, v138, vcc_lo
	s_clause 0x1d
	global_store_b128 v[137:138], v[9:12], off
	global_store_b128 v[137:138], v[13:16], off offset:16
	global_store_b128 v[137:138], v[17:20], off offset:32
	global_store_b128 v[137:138], v[21:24], off offset:48
	global_store_b128 v[137:138], v[25:28], off offset:64
	global_store_b128 v[137:138], v[29:32], off offset:80
	global_store_b128 v[137:138], v[33:36], off offset:96
	global_store_b128 v[137:138], v[37:40], off offset:112
	global_store_b128 v[137:138], v[41:44], off offset:128
	global_store_b128 v[137:138], v[45:48], off offset:144
	global_store_b128 v[137:138], v[49:52], off offset:160
	global_store_b128 v[137:138], v[53:56], off offset:176
	global_store_b128 v[137:138], v[57:60], off offset:192
	global_store_b128 v[137:138], v[61:64], off offset:208
	global_store_b128 v[137:138], v[65:68], off offset:224
	global_store_b128 v[137:138], v[69:72], off offset:240
	global_store_b128 v[137:138], v[73:76], off offset:256
	global_store_b128 v[137:138], v[77:80], off offset:272
	global_store_b128 v[137:138], v[81:84], off offset:288
	global_store_b128 v[137:138], v[85:88], off offset:304
	global_store_b128 v[137:138], v[89:92], off offset:320
	global_store_b128 v[137:138], v[93:96], off offset:336
	global_store_b128 v[137:138], v[97:100], off offset:352
	global_store_b128 v[137:138], v[101:104], off offset:368
	global_store_b128 v[137:138], v[105:108], off offset:384
	global_store_b128 v[137:138], v[109:112], off offset:400
	global_store_b128 v[137:138], v[113:116], off offset:416
	global_store_b128 v[137:138], v[117:120], off offset:432
	global_store_b128 v[137:138], v[121:124], off offset:448
	global_store_b128 v[137:138], v[125:128], off offset:464
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v12, v0
	v_dual_mov_b32 v11, 1.0 :: v_dual_mov_b32 v10, 0
	v_mov_b32_e32 v9, 0xff800000
	s_clause 0x4
	global_store_b128 v[137:138], v[1:4], off offset:480
	global_store_b128 v[137:138], v[5:8], off offset:496
	global_store_b128 v[137:138], v[129:132], off offset:512
	global_store_b128 v[137:138], v[133:136], off offset:528
	global_store_b128 v[137:138], v[9:12], off offset:544
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end3:
	.size	_Z12floor_kernelILi3EEvPKjPfi, .Lfunc_end3-_Z12floor_kernelILi3EEvPKjPfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12floor_kernelILi3EEvPKjPfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 152
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_Z12floor_kernelILi3EEvPKjPfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z12floor_kernelILi3EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi3EEvPKjPfi,comdat
                                        ; -- End function
	.set .L_Z12floor_kernelILi3EEvPKjPfi.num_vgpr, 152
	.set .L_Z12floor_kernelILi3EEvPKjPfi.num_agpr, 0
	.set .L_Z12floor_kernelILi3EEvPKjPfi.numbered_sgpr, 16
	.set .L_Z12floor_kernelILi3EEvPKjPfi.num_named_barrier, 0
	.set .L_Z12floor_kernelILi3EEvPKjPfi.private_seg_size, 0
	.set .L_Z12floor_kernelILi3EEvPKjPfi.uses_vcc, 1
	.set .L_Z12floor_kernelILi3EEvPKjPfi.uses_flat_scratch, 0
	.set .L_Z12floor_kernelILi3EEvPKjPfi.has_dyn_sized_stack, 0
	.set .L_Z12floor_kernelILi3EEvPKjPfi.has_recursion, 0
	.set .L_Z12floor_kernelILi3EEvPKjPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1636
; TotalNumSgprs: 18
; NumVgprs: 152
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 18
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 152
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z12floor_kernelILi4EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi4EEvPKjPfi,comdat
	.protected	_Z12floor_kernelILi4EEvPKjPfi ; -- Begin function _Z12floor_kernelILi4EEvPKjPfi
	.globl	_Z12floor_kernelILi4EEvPKjPfi
	.p2align	8
	.type	_Z12floor_kernelILi4EEvPKjPfi,@function
_Z12floor_kernelILi4EEvPKjPfi:          ; @_Z12floor_kernelILi4EEvPKjPfi
	.cfi_startproc
; %bb.0:                                ; %.preheader165
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x10
	v_lshl_add_u32 v132, v0, 2, 0
	s_mov_b32 s4, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s10, s4
	s_mov_b32 s11, s4
	ds_store_b32 v132, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s5, s4
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s8, s4
	s_mov_b32 s9, s4
	v_dual_mov_b32 v1, s4 :: v_dual_mov_b32 v2, s5
	v_dual_mov_b32 v7, s10 :: v_dual_mov_b32 v8, s11
	v_dual_mov_b32 v3, s6 :: v_dual_mov_b32 v4, s7
	v_dual_mov_b32 v5, s8 :: v_dual_mov_b32 v6, s9
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v127, v7
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v119, v7
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v111, v7
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v103, v7
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v95, v7
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v87, v7
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v79, v7
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v71, v7
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v63, v7
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v55, v7
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v7
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v7
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v7
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v7
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	v_dual_mov_b32 v126, v6 :: v_dual_mov_b32 v125, v5
	v_dual_mov_b32 v124, v4 :: v_dual_mov_b32 v123, v3
	v_dual_mov_b32 v122, v2 :: v_dual_mov_b32 v121, v1
	v_dual_mov_b32 v118, v6 :: v_dual_mov_b32 v117, v5
	v_dual_mov_b32 v116, v4 :: v_dual_mov_b32 v115, v3
	v_dual_mov_b32 v114, v2 :: v_dual_mov_b32 v113, v1
	v_dual_mov_b32 v110, v6 :: v_dual_mov_b32 v109, v5
	v_dual_mov_b32 v108, v4 :: v_dual_mov_b32 v107, v3
	v_dual_mov_b32 v106, v2 :: v_dual_mov_b32 v105, v1
	v_dual_mov_b32 v102, v6 :: v_dual_mov_b32 v101, v5
	v_dual_mov_b32 v100, v4 :: v_dual_mov_b32 v99, v3
	v_dual_mov_b32 v98, v2 :: v_dual_mov_b32 v97, v1
	v_dual_mov_b32 v94, v6 :: v_dual_mov_b32 v93, v5
	v_dual_mov_b32 v92, v4 :: v_dual_mov_b32 v91, v3
	v_dual_mov_b32 v90, v2 :: v_dual_mov_b32 v89, v1
	v_dual_mov_b32 v86, v6 :: v_dual_mov_b32 v85, v5
	v_dual_mov_b32 v84, v4 :: v_dual_mov_b32 v83, v3
	v_dual_mov_b32 v82, v2 :: v_dual_mov_b32 v81, v1
	v_dual_mov_b32 v78, v6 :: v_dual_mov_b32 v77, v5
	v_dual_mov_b32 v76, v4 :: v_dual_mov_b32 v75, v3
	v_dual_mov_b32 v74, v2 :: v_dual_mov_b32 v73, v1
	v_dual_mov_b32 v70, v6 :: v_dual_mov_b32 v69, v5
	v_dual_mov_b32 v68, v4 :: v_dual_mov_b32 v67, v3
	v_dual_mov_b32 v66, v2 :: v_dual_mov_b32 v65, v1
	v_dual_mov_b32 v62, v6 :: v_dual_mov_b32 v61, v5
	v_dual_mov_b32 v60, v4 :: v_dual_mov_b32 v59, v3
	v_dual_mov_b32 v58, v2 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v54, v6 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v52, v4 :: v_dual_mov_b32 v51, v3
	v_dual_mov_b32 v50, v2 :: v_dual_mov_b32 v49, v1
	v_dual_mov_b32 v46, v6 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v44, v4 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v42, v2 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v37, v5
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v35, v3
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v30, v6 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v28, v4 :: v_dual_mov_b32 v27, v3
	v_dual_mov_b32 v26, v2 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v22, v6 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v20, v4 :: v_dual_mov_b32 v19, v3
	v_dual_mov_b32 v18, v2 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v13, v5
	v_dual_mov_b32 v12, v4 :: v_dual_mov_b32 v11, v3
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v9, v1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
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
	s_cbranch_scc1 .LBB4_5
; %bb.1:                                ; %.preheader163.lr.ph
	v_lshrrev_b32_e32 v130, 1, v0
	s_mov_b32 s3, 0x3c23d70a
	v_dual_mov_b32 v167, 0xff800000 :: v_dual_mov_b32 v166, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v165, 0 :: v_dual_and_b32 v130, 8, v130
	v_or_b32_e32 v133, 3, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v133, v133
	v_dual_mul_f32 v150, 0x3a83126f, v133 :: v_dual_and_b32 v129, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v129, v129
	v_mul_f32_e32 v141, 0x3c23d70a, v129
	v_or_b32_e32 v134, 1, v130
	v_or_b32_e32 v135, 2, v130
	v_fmaak_f32 v142, s3, v129, 0x3cf5c28f
	v_fmaak_f32 v143, s3, v129, 0x3db851eb
	v_fmaak_f32 v144, s3, v129, 0x3d75c28f
	v_fmaak_f32 v145, s3, v129, 0x3e570a3d
	v_fmaak_f32 v146, s3, v129, 0x3e3851eb
	v_or_b32_e32 v131, 4, v130
	v_fmaak_f32 v147, s3, v129, 0x3e199999
	v_fmaak_f32 v148, s3, v129, 0x3df5c28f
	v_cvt_f32_ubyte0_e32 v129, v134
	v_cvt_f32_ubyte0_e32 v134, v135
	v_add_nc_u32_e32 v135, 8, v130
	v_or_b32_e32 v136, 7, v130
	v_or_b32_e32 v137, 6, v130
	v_or_b32_e32 v130, 5, v130
	v_cvt_f32_ubyte0_e32 v131, v131
	v_cvt_f32_ubyte0_e32 v135, v135
	v_cvt_f32_ubyte0_e32 v136, v136
	v_cvt_f32_ubyte0_e32 v137, v137
	v_cvt_f32_ubyte0_e32 v130, v130
	v_dual_mul_f32 v149, 0x3a83126f, v131 :: v_dual_mul_f32 v152, 0x3a83126f, v129
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v151, 0x3a83126f, v134 :: v_dual_mul_f32 v154, 0x3a83126f, v136
	v_dual_mul_f32 v153, 0x3a83126f, v135 :: v_dual_mul_f32 v156, 0x3a83126f, v130
	v_mul_f32_e32 v155, 0x3a83126f, v137
	v_dual_mul_f32 v157, 0x3c23d70a, v129 :: v_dual_mul_f32 v158, 0x3c23d70a, v134
	v_dual_mul_f32 v159, 0x3c23d70a, v133 :: v_dual_mul_f32 v160, 0x3c23d70a, v131
	v_dual_mul_f32 v161, 0x3c23d70a, v130 :: v_dual_mul_f32 v162, 0x3c23d70a, v137
	v_dual_mul_f32 v163, 0x3c23d70a, v136 :: v_dual_mul_f32 v164, 0x3c23d70a, v135
	s_mov_b32 s3, 0x76543210
	s_branch .LBB4_3
.LBB4_2:                                ; %.preheader161
                                        ;   in Loop: Header=BB4_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_sub_f32_e32 v169, v167, v129
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v167
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mul_f32_e32 v169, 0x3fb8aa3b, v169
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_exp_f32_e32 v169, v169
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
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v167, 0, v169 :: v_dual_add_f32 v130, v130, v168
	;;#ASMSTART
	;;#ASMEND
	s_add_co_i32 s4, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, s4
	v_mul_f32_e32 v166, v166, v167
	v_fmac_f32_e32 v130, v165, v167
	v_mov_b32_e32 v167, v129
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_div_scale_f32 v169, null, v131, v131, v166
	v_div_scale_f32 v172, vcc_lo, v166, v131, v166
	v_mov_b32_e32 v165, v130
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v170, v169
	v_fma_f32 v171, -v169, v170, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v170, v171, v170
	v_mul_f32_e32 v171, v172, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v169, v171, v172
	v_fmac_f32_e32 v171, v173, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v169, -v169, v171, v172
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v168, v169, v170, v171
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v166, v168, v131, v166
	v_dual_mul_f32 v128, v166, v128 :: v_dual_mul_f32 v127, v166, v127
	v_dual_mul_f32 v126, v166, v126 :: v_dual_mul_f32 v125, v166, v125
	v_dual_mul_f32 v124, v166, v124 :: v_dual_mul_f32 v123, v166, v123
	v_dual_mul_f32 v122, v166, v122 :: v_dual_mul_f32 v121, v166, v121
	v_dual_mul_f32 v120, v166, v120 :: v_dual_mul_f32 v119, v166, v119
	v_dual_mul_f32 v118, v166, v118 :: v_dual_mul_f32 v117, v166, v117
	v_dual_mul_f32 v116, v166, v116 :: v_dual_mul_f32 v115, v166, v115
	v_dual_mul_f32 v114, v166, v114 :: v_dual_mul_f32 v113, v166, v113
	v_dual_mul_f32 v112, v166, v112 :: v_dual_mul_f32 v111, v166, v111
	v_dual_mul_f32 v110, v166, v110 :: v_dual_mul_f32 v109, v166, v109
	v_dual_mul_f32 v108, v166, v108 :: v_dual_mul_f32 v107, v166, v107
	v_dual_mul_f32 v106, v166, v106 :: v_dual_mul_f32 v105, v166, v105
	v_dual_mul_f32 v104, v166, v104 :: v_dual_mul_f32 v103, v166, v103
	v_dual_mul_f32 v102, v166, v102 :: v_dual_mul_f32 v101, v166, v101
	v_dual_mul_f32 v100, v166, v100 :: v_dual_mul_f32 v99, v166, v99
	v_dual_mul_f32 v98, v166, v98 :: v_dual_mul_f32 v97, v166, v97
	v_dual_mul_f32 v96, v166, v96 :: v_dual_mul_f32 v95, v166, v95
	v_dual_mul_f32 v94, v166, v94 :: v_dual_mul_f32 v93, v166, v93
	v_dual_mul_f32 v92, v166, v92 :: v_dual_mul_f32 v91, v166, v91
	v_dual_mul_f32 v90, v166, v90 :: v_dual_mul_f32 v89, v166, v89
	v_dual_mul_f32 v88, v166, v88 :: v_dual_mul_f32 v87, v166, v87
	v_dual_mul_f32 v86, v166, v86 :: v_dual_mul_f32 v85, v166, v85
	v_dual_mul_f32 v84, v166, v84 :: v_dual_mul_f32 v83, v166, v83
	v_dual_mul_f32 v82, v166, v82 :: v_dual_mul_f32 v81, v166, v81
	v_dual_mul_f32 v80, v166, v80 :: v_dual_mul_f32 v79, v166, v79
	v_dual_mul_f32 v78, v166, v78 :: v_dual_mul_f32 v77, v166, v77
	v_dual_mul_f32 v76, v166, v76 :: v_dual_mul_f32 v75, v166, v75
	v_dual_mul_f32 v74, v166, v74 :: v_dual_mul_f32 v73, v166, v73
	v_dual_mul_f32 v72, v166, v72 :: v_dual_mul_f32 v71, v166, v71
	v_dual_mul_f32 v70, v166, v70 :: v_dual_mul_f32 v69, v166, v69
	v_dual_mul_f32 v68, v166, v68 :: v_dual_mul_f32 v67, v166, v67
	v_dual_mul_f32 v66, v166, v66 :: v_dual_mul_f32 v65, v166, v65
	v_dual_mul_f32 v64, v166, v64 :: v_dual_mul_f32 v63, v166, v63
	v_dual_mul_f32 v62, v166, v62 :: v_dual_mul_f32 v61, v166, v61
	v_dual_mul_f32 v60, v166, v60 :: v_dual_mul_f32 v59, v166, v59
	v_dual_mul_f32 v58, v166, v58 :: v_dual_mul_f32 v57, v166, v57
	v_dual_mul_f32 v56, v166, v56 :: v_dual_mul_f32 v55, v166, v55
	v_dual_mul_f32 v54, v166, v54 :: v_dual_mul_f32 v53, v166, v53
	v_dual_mul_f32 v52, v166, v52 :: v_dual_mul_f32 v51, v166, v51
	v_dual_mul_f32 v50, v166, v50 :: v_dual_mul_f32 v49, v166, v49
	v_dual_mul_f32 v48, v166, v48 :: v_dual_mul_f32 v47, v166, v47
	v_dual_mul_f32 v46, v166, v46 :: v_dual_mul_f32 v45, v166, v45
	v_dual_mul_f32 v44, v166, v44 :: v_dual_mul_f32 v43, v166, v43
	v_dual_mul_f32 v42, v166, v42 :: v_dual_mul_f32 v41, v166, v41
	v_dual_mul_f32 v40, v166, v40 :: v_dual_mul_f32 v39, v166, v39
	v_dual_mul_f32 v38, v166, v38 :: v_dual_mul_f32 v37, v166, v37
	v_dual_mul_f32 v36, v166, v36 :: v_dual_mul_f32 v35, v166, v35
	v_dual_mul_f32 v34, v166, v34 :: v_dual_mul_f32 v33, v166, v33
	v_dual_mul_f32 v32, v166, v32 :: v_dual_mul_f32 v31, v166, v31
	v_dual_mul_f32 v30, v166, v30 :: v_dual_mul_f32 v29, v166, v29
	v_dual_mul_f32 v28, v166, v28 :: v_dual_mul_f32 v27, v166, v27
	v_dual_mul_f32 v26, v166, v26 :: v_dual_mul_f32 v25, v166, v25
	v_dual_mul_f32 v24, v166, v24 :: v_dual_mul_f32 v23, v166, v23
	v_dual_mul_f32 v22, v166, v22 :: v_dual_mul_f32 v21, v166, v21
	v_dual_mul_f32 v20, v166, v20 :: v_dual_mul_f32 v19, v166, v19
	v_dual_mul_f32 v18, v166, v18 :: v_dual_mul_f32 v17, v166, v17
	v_dual_mul_f32 v16, v166, v16 :: v_dual_mul_f32 v15, v166, v15
	v_dual_mul_f32 v14, v166, v14 :: v_dual_mul_f32 v13, v166, v13
	v_dual_mul_f32 v12, v166, v12 :: v_dual_mul_f32 v11, v166, v11
	v_dual_mul_f32 v10, v166, v10 :: v_dual_mul_f32 v9, v166, v9
	v_dual_mul_f32 v8, v166, v8 :: v_dual_mul_f32 v7, v166, v7
	v_dual_mul_f32 v6, v166, v6 :: v_dual_mul_f32 v5, v166, v5
	v_dual_mul_f32 v4, v166, v4 :: v_dual_mul_f32 v3, v166, v3
	v_dual_mul_f32 v2, v166, v2 :: v_dual_mul_f32 v1, v166, v1
	v_mov_b32_e32 v166, v131
	s_cbranch_scc1 .LBB4_6
.LBB4_3:                                ; %.preheader163
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s5, s4
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmamk_f32 v131, s5, 0x38d1b717, v142 :: v_dual_fmamk_f32 v134, s5, 0x38d1b717, v145
	v_dual_fmamk_f32 v129, s5, 0x38d1b717, v143 :: v_dual_fmamk_f32 v130, s5, 0x38d1b717, v144
	v_dual_fmamk_f32 v135, s5, 0x38d1b717, v146 :: v_dual_fmamk_f32 v136, s5, 0x38d1b717, v147
	v_dual_mul_f32 v131, 0x3e000000, v131 :: v_dual_mul_f32 v134, 0x3e000000, v134
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v130, 0x3e000000, v130
	v_mul_f32_e32 v138, 0x3e000000, v129
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v136, 0x3e000000, v136 :: v_dual_mul_f32 v129, v131, v151
	v_fmamk_f32 v133, s5, 0x38d1b717, v141
	v_dual_fmamk_f32 v137, s5, 0x38d1b717, v148 :: v_dual_mul_f32 v168, v130, v150
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v140, v138, v149 :: v_dual_mul_f32 v135, 0x3e000000, v135
	v_mul_f32_e32 v133, 0x3e000000, v133
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v139, v133, v152
	v_max3_num_f32 v129, v139, 0xff800000, v129
	v_mul_f32_e32 v139, v136, v155
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v168, v140
	v_mul_f32_e32 v140, v134, v153
	v_dual_mul_f32 v137, 0x3e000000, v137 :: v_dual_mul_f32 v168, v135, v154
	v_mul_f32_e32 v169, v137, v156
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v169, v139
	v_max3_num_f32 v129, v129, v168, v140
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mov_b32_e32 v139, v129
	v_permlanex16_b32 v139, v139, s3, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v129, v167, v129, v139
	v_fma_f32 v138, v138, v149, -v129
	v_fma_f32 v136, v136, v155, -v129
	v_fma_f32 v131, v131, v151, -v129
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v129
	v_fma_f32 v133, v133, v152, -v129
	v_mul_f32_e32 v138, 0x3fb8aa3b, v138
	v_mul_f32_e32 v136, 0x3fb8aa3b, v136
	v_fma_f32 v130, v130, v150, -v129
	v_fma_f32 v134, v134, v153, -v129
	v_mul_f32_e32 v133, 0x3fb8aa3b, v133
	v_exp_f32_e32 v138, v138
	v_exp_f32_e32 v136, v136
	v_fma_f32 v137, v137, v156, -v129
	v_fma_f32 v135, v135, v154, -v129
	v_mul_f32_e32 v130, 0x3fb8aa3b, v130
	v_mul_f32_e32 v134, 0x3fb8aa3b, v134
	v_exp_f32_e32 v133, v133
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(TRANS32_DEP_3)
	v_cndmask_b32_e64 v168, v138, 0, vcc_lo
	v_mul_f32_e32 v131, 0x3fb8aa3b, v131
	v_cndmask_b32_e64 v170, v136, 0, vcc_lo
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	v_exp_f32_e32 v130, v130
	v_mul_f32_e32 v136, v160, v168
	v_exp_f32_e32 v131, v131
	v_cndmask_b32_e64 v140, v133, 0, vcc_lo
	v_exp_f32_e32 v139, v134
	v_mul_f32_e32 v138, v162, v170
	v_exp_f32_e32 v135, v135
	v_cndmask_b32_e64 v130, v130, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v131, v131, 0, vcc_lo
	v_mul_f32_e32 v137, 0x3fb8aa3b, v137
	v_cndmask_b32_e64 v172, v139, 0, vcc_lo
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v171, v135, 0, vcc_lo
	v_mul_f32_e32 v134, v158, v131
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v137, v137
	v_add_f32_e32 v131, v140, v131
	v_mul_f32_e32 v133, v157, v140
	v_mul_f32_e32 v135, v159, v130
	v_add_f32_e32 v130, v131, v130
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_max3_num_f32 v140, v133, 0, v134
	v_cndmask_b32_e64 v169, v137, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v130, v130, v168
	v_max3_num_f32 v131, v140, v135, v136
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v140, v164, v172 :: v_dual_mul_f32 v137, v161, v169
	v_dual_add_f32 v130, v130, v169 :: v_dual_mul_f32 v139, v163, v171
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v131, v131, v137, v138
	v_add_f32_e32 v130, v130, v170
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v131, v131, v139, v140
	v_mov_b32_e32 v168, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_permlanex16_b32 v168, v168, s3, 0xfedcba98
	v_dual_add_f32 v130, v130, v171 :: v_dual_max_num_f32 v169, v168, v168
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v130, v130, v172 :: v_dual_max_num_f32 v169, v131, v169
	v_dual_mov_b32 v168, v130 :: v_dual_mov_b32 v131, v166
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_permlanex16_b32 v168, v168, s3, 0xfedcba98
	v_cmpx_lt_f32_e32 0, v169
	s_cbranch_execz .LBB4_2
; %bb.4:                                ;   in Loop: Header=BB4_3 Depth=1
	v_div_scale_f32 v131, null, 0x43e00000, 0x43e00000, v169
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v170, v131
	v_fma_f32 v171, -v131, v170, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v170, v171, v170
	v_div_scale_f32 v171, vcc_lo, v169, 0x43e00000, v169
	v_mul_f32_e32 v172, v171, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v173, -v131, v172, v171
	v_fmac_f32_e32 v172, v173, v170
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v131, -v131, v172, v171
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v131, v131, v170, v172
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v131, v131, 0x43e00000, v169
	v_max_num_f32_e32 v131, 0x1f800000, v131
	s_branch .LBB4_2
.LBB4_5:
	v_dual_mov_b32 v131, 1.0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v129, 0xff800000 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v130, 0
	v_mov_b32_e32 v133, 0
.LBB4_6:                                ; %Flow
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x24
	s_load_b64 s[0:1], s[0:1], 0x8
	v_dual_mul_f32 v25, v131, v25 :: v_dual_mul_f32 v26, v131, v26
	v_dual_mul_f32 v27, v131, v27 :: v_dual_mul_f32 v28, v131, v28
	v_dual_mul_f32 v29, v131, v29 :: v_dual_mul_f32 v30, v131, v30
	v_dual_mul_f32 v31, v131, v31 :: v_dual_mul_f32 v32, v131, v32
	v_dual_mul_f32 v121, v131, v121 :: v_dual_mul_f32 v122, v131, v122
	v_dual_mul_f32 v123, v131, v123 :: v_dual_mul_f32 v124, v131, v124
	v_dual_mul_f32 v97, v131, v97 :: v_dual_mul_f32 v98, v131, v98
	v_dual_mul_f32 v99, v131, v99 :: v_dual_mul_f32 v100, v131, v100
	v_dual_mul_f32 v73, v131, v73 :: v_dual_mul_f32 v74, v131, v74
	s_wait_kmcnt 0x0
	s_and_b32 s2, 0xffff, s2
	v_dual_mul_f32 v75, v131, v75 :: v_dual_mul_f32 v76, v131, v76
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[141:142], null, ttmp9, s2, v[0:1]
	v_dual_mul_f32 v49, v131, v49 :: v_dual_mul_f32 v50, v131, v50
	v_dual_mul_f32 v51, v131, v51 :: v_dual_mul_f32 v52, v131, v52
	v_dual_mul_f32 v17, v131, v17 :: v_dual_mul_f32 v18, v131, v18
	v_dual_mul_f32 v19, v131, v19 :: v_dual_mul_f32 v20, v131, v20
	v_mul_lo_u32 v141, 0x8c, v141
	v_dual_mul_f32 v9, v131, v9 :: v_dual_mul_f32 v10, v131, v10
	v_dual_mul_f32 v11, v131, v11 :: v_dual_mul_f32 v12, v131, v12
	v_dual_mul_f32 v0, v131, v1 :: v_dual_mul_f32 v1, v131, v2
	v_dual_mul_f32 v2, v131, v3 :: v_dual_mul_f32 v3, v131, v4
	v_ashrrev_i32_e32 v142, 31, v141
	v_dual_mul_f32 v125, v131, v125 :: v_dual_mul_f32 v126, v131, v126
	v_dual_mul_f32 v127, v131, v127 :: v_dual_mul_f32 v128, v131, v128
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[141:142], 2, v[141:142]
	v_dual_mul_f32 v101, v131, v101 :: v_dual_mul_f32 v102, v131, v102
	v_dual_mul_f32 v103, v131, v103 :: v_dual_mul_f32 v104, v131, v104
	v_dual_mul_f32 v77, v131, v77 :: v_dual_mul_f32 v78, v131, v78
	v_add_co_u32 v141, vcc_lo, s0, v141
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v142, null, s1, v142, vcc_lo
	s_clause 0x1
	global_store_b128 v[141:142], v[25:28], off offset:384
	global_store_b128 v[141:142], v[29:32], off offset:400
	ds_load_b32 v25, v132
	v_dual_mul_f32 v79, v131, v79 :: v_dual_mul_f32 v80, v131, v80
	v_dual_mul_f32 v53, v131, v53 :: v_dual_mul_f32 v54, v131, v54
	v_dual_mul_f32 v55, v131, v55 :: v_dual_mul_f32 v56, v131, v56
	v_dual_mul_f32 v21, v131, v21 :: v_dual_mul_f32 v22, v131, v22
	v_dual_mul_f32 v23, v131, v23 :: v_dual_mul_f32 v24, v131, v24
	v_dual_mul_f32 v13, v131, v13 :: v_dual_mul_f32 v14, v131, v14
	v_dual_mul_f32 v15, v131, v15 :: v_dual_mul_f32 v16, v131, v16
	v_dual_mul_f32 v4, v131, v5 :: v_dual_mul_f32 v5, v131, v6
	v_dual_mul_f32 v6, v131, v7 :: v_dual_mul_f32 v7, v131, v8
	v_dual_mul_f32 v113, v131, v113 :: v_dual_mul_f32 v114, v131, v114
	v_dual_mul_f32 v115, v131, v115 :: v_dual_mul_f32 v116, v131, v116
	v_dual_mul_f32 v89, v131, v89 :: v_dual_mul_f32 v90, v131, v90
	v_dual_mul_f32 v91, v131, v91 :: v_dual_mul_f32 v92, v131, v92
	v_dual_mul_f32 v65, v131, v65 :: v_dual_mul_f32 v66, v131, v66
	v_dual_mul_f32 v67, v131, v67 :: v_dual_mul_f32 v68, v131, v68
	v_dual_mul_f32 v41, v131, v41 :: v_dual_mul_f32 v42, v131, v42
	v_dual_mul_f32 v43, v131, v43 :: v_dual_mul_f32 v44, v131, v44
	v_dual_mul_f32 v117, v131, v117 :: v_dual_mul_f32 v118, v131, v118
	v_dual_mul_f32 v119, v131, v119 :: v_dual_mul_f32 v120, v131, v120
	v_dual_mul_f32 v93, v131, v93 :: v_dual_mul_f32 v94, v131, v94
	v_dual_mul_f32 v95, v131, v95 :: v_dual_mul_f32 v96, v131, v96
	v_dual_mul_f32 v69, v131, v69 :: v_dual_mul_f32 v70, v131, v70
	v_dual_mul_f32 v71, v131, v71 :: v_dual_mul_f32 v72, v131, v72
	v_dual_mul_f32 v45, v131, v45 :: v_dual_mul_f32 v46, v131, v46
	v_dual_mul_f32 v47, v131, v47 :: v_dual_mul_f32 v48, v131, v48
	v_dual_mul_f32 v105, v131, v105 :: v_dual_mul_f32 v106, v131, v106
	v_dual_mul_f32 v107, v131, v107 :: v_dual_mul_f32 v108, v131, v108
	v_dual_mul_f32 v81, v131, v81 :: v_dual_mul_f32 v82, v131, v82
	v_dual_mul_f32 v83, v131, v83 :: v_dual_mul_f32 v84, v131, v84
	v_dual_mul_f32 v57, v131, v57 :: v_dual_mul_f32 v58, v131, v58
	v_dual_mul_f32 v59, v131, v59 :: v_dual_mul_f32 v60, v131, v60
	v_dual_mul_f32 v33, v131, v33 :: v_dual_mul_f32 v34, v131, v34
	v_dual_mul_f32 v35, v131, v35 :: v_dual_mul_f32 v36, v131, v36
	s_wait_dscnt 0x0
	v_cvt_f32_u32_e32 v132, v25
	v_dual_mul_f32 v109, v131, v109 :: v_dual_mul_f32 v110, v131, v110
	v_dual_mul_f32 v111, v131, v111 :: v_dual_mul_f32 v112, v131, v112
	s_clause 0x5
	global_store_b128 v[141:142], v[121:124], off
	global_store_b128 v[141:142], v[125:128], off offset:16
	global_store_b128 v[141:142], v[113:116], off offset:32
	global_store_b128 v[141:142], v[117:120], off offset:48
	global_store_b128 v[141:142], v[105:108], off offset:64
	global_store_b128 v[141:142], v[109:112], off offset:80
	v_dual_mul_f32 v85, v131, v85 :: v_dual_mul_f32 v86, v131, v86
	v_dual_mul_f32 v87, v131, v87 :: v_dual_mul_f32 v88, v131, v88
	s_clause 0x5
	global_store_b128 v[141:142], v[97:100], off offset:96
	global_store_b128 v[141:142], v[101:104], off offset:112
	global_store_b128 v[141:142], v[89:92], off offset:128
	global_store_b128 v[141:142], v[93:96], off offset:144
	global_store_b128 v[141:142], v[81:84], off offset:160
	global_store_b128 v[141:142], v[85:88], off offset:176
	v_dual_mul_f32 v61, v131, v61 :: v_dual_mul_f32 v62, v131, v62
	v_dual_mul_f32 v63, v131, v63 :: v_dual_mul_f32 v64, v131, v64
	s_clause 0x5
	global_store_b128 v[141:142], v[73:76], off offset:192
	global_store_b128 v[141:142], v[77:80], off offset:208
	global_store_b128 v[141:142], v[65:68], off offset:224
	global_store_b128 v[141:142], v[69:72], off offset:240
	global_store_b128 v[141:142], v[57:60], off offset:256
	global_store_b128 v[141:142], v[61:64], off offset:272
	v_dual_mul_f32 v37, v131, v37 :: v_dual_mul_f32 v38, v131, v38
	v_dual_mul_f32 v39, v131, v39 :: v_dual_mul_f32 v40, v131, v40
	s_clause 0xe
	global_store_b128 v[141:142], v[49:52], off offset:288
	global_store_b128 v[141:142], v[53:56], off offset:304
	global_store_b128 v[141:142], v[41:44], off offset:320
	global_store_b128 v[141:142], v[45:48], off offset:336
	global_store_b128 v[141:142], v[33:36], off offset:352
	global_store_b128 v[141:142], v[37:40], off offset:368
	global_store_b128 v[141:142], v[17:20], off offset:416
	global_store_b128 v[141:142], v[21:24], off offset:432
	global_store_b128 v[141:142], v[9:12], off offset:448
	global_store_b128 v[141:142], v[13:16], off offset:464
	global_store_b128 v[141:142], v[0:3], off offset:480
	global_store_b128 v[141:142], v[4:7], off offset:496
	global_store_b128 v[141:142], v[133:136], off offset:512
	global_store_b128 v[141:142], v[137:140], off offset:528
	global_store_b128 v[141:142], v[129:132], off offset:544
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end4:
	.size	_Z12floor_kernelILi4EEvPKjPfi, .Lfunc_end4-_Z12floor_kernelILi4EEvPKjPfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12floor_kernelILi4EEvPKjPfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 174
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_Z12floor_kernelILi4EEvPKjPfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z12floor_kernelILi4EEvPKjPfi,"axG",@progbits,_Z12floor_kernelILi4EEvPKjPfi,comdat
                                        ; -- End function
	.set .L_Z12floor_kernelILi4EEvPKjPfi.num_vgpr, 174
	.set .L_Z12floor_kernelILi4EEvPKjPfi.num_agpr, 0
	.set .L_Z12floor_kernelILi4EEvPKjPfi.numbered_sgpr, 12
	.set .L_Z12floor_kernelILi4EEvPKjPfi.num_named_barrier, 0
	.set .L_Z12floor_kernelILi4EEvPKjPfi.private_seg_size, 0
	.set .L_Z12floor_kernelILi4EEvPKjPfi.uses_vcc, 1
	.set .L_Z12floor_kernelILi4EEvPKjPfi.uses_flat_scratch, 0
	.set .L_Z12floor_kernelILi4EEvPKjPfi.has_dyn_sized_stack, 0
	.set .L_Z12floor_kernelILi4EEvPKjPfi.has_recursion, 0
	.set .L_Z12floor_kernelILi4EEvPKjPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3516
; TotalNumSgprs: 14
; NumVgprs: 174
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 174
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_2edb253f577969c9,@object ; @__hip_cuid_2edb253f577969c9
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_2edb253f577969c9
__hip_cuid_2edb253f577969c9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_2edb253f577969c9, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_2edb253f577969c9
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
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         144
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z12floor_kernelILi0EEvPKjPfi
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z12floor_kernelILi0EEvPKjPfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     187
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         144
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z12floor_kernelILi1EEvPKjPfi
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z12floor_kernelILi1EEvPKjPfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     185
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         144
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z12floor_kernelILi2EEvPKjPfi
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z12floor_kernelILi2EEvPKjPfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     185
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         144
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z12floor_kernelILi3EEvPKjPfi
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z12floor_kernelILi3EEvPKjPfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     152
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         144
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z12floor_kernelILi4EEvPKjPfi
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z12floor_kernelILi4EEvPKjPfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     174
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
