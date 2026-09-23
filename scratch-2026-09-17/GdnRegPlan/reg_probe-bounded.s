	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.section	.text._Z5probeILi8ELb0ELb0EEvPfPKfi,"axG",@progbits,_Z5probeILi8ELb0ELb0EEvPfPKfi,comdat
	.protected	_Z5probeILi8ELb0ELb0EEvPfPKfi ; -- Begin function _Z5probeILi8ELb0ELb0EEvPfPKfi
	.globl	_Z5probeILi8ELb0ELb0EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi8ELb0ELb0EEvPfPKfi,@function
_Z5probeILi8ELb0ELb0EEvPfPKfi:          ; @_Z5probeILi8ELb0ELb0EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s0, s[0:1], 0x10
	s_mov_b32 s1, 0x3c23d70a
	v_lshlrev_b32_e32 v1, 2, v0
	s_wait_kmcnt 0x0
	global_load_b32 v2, v1, s[6:7]
	s_cmp_lt_i32 s0, 1
	s_wait_loadcnt 0x0
	v_fma_f32 v49, 0x3c23d70a, v2, 0
	v_fmaak_f32 v50, s1, v2, 0x38d1b717
	v_fmaak_f32 v51, s1, v2, 0x3951b717
	v_fmaak_f32 v52, s1, v2, 0x399d4951
	v_fmaak_f32 v53, s1, v2, 0x39d1b717
	v_fmaak_f32 v54, s1, v2, 0x3a03126e
	v_fmaak_f32 v55, s1, v2, 0x3a1d4951
	v_fmaak_f32 v56, s1, v2, 0x3a378034
	v_fmaak_f32 v64, s1, v2, 0x3a51b717
	v_fmaak_f32 v48, s1, v2, 0x3a6bedfa
	v_fmaak_f32 v40, s1, v2, 0x3a83126e
	v_fmaak_f32 v32, s1, v2, 0x3a902de0
	v_fmaak_f32 v24, s1, v2, 0x3a9d4951
	v_fmaak_f32 v16, s1, v2, 0x3aaa64c3
	v_fmaak_f32 v8, s1, v2, 0x3ab78034
	s_cbranch_scc1 .LBB0_3
; %bb.1:
	v_dual_mov_b32 v129, 0x3b03126f :: v_dual_add_nc_u32 v2, 1, v0
	v_dual_mov_b32 v58, v51 :: v_dual_add_nc_u32 v3, 2, v0
	v_dual_mov_b32 v57, v50 :: v_dual_add_nc_u32 v4, 3, v0
	v_dual_mov_b32 v60, v53 :: v_dual_add_nc_u32 v5, 4, v0
	v_dual_mov_b32 v46, v56 :: v_dual_add_nc_u32 v7, 6, v0
	v_dual_mov_b32 v61, v54 :: v_dual_add_nc_u32 v6, 5, v0
	v_dual_mov_b32 v34, v53 :: v_dual_add_nc_u32 v9, 7, v0
	v_dual_mov_b32 v59, v52 :: v_dual_and_b32 v2, 0xff, v2
	v_dual_mov_b32 v62, v55 :: v_dual_and_b32 v3, 0xff, v3
	v_dual_mov_b32 v33, v52 :: v_dual_add_nc_u32 v10, 9, v0
	v_dual_mov_b32 v63, v56 :: v_dual_and_b32 v4, 0xff, v4
	v_dual_mov_b32 v42, v52 :: v_dual_and_b32 v5, 0xff, v5
	v_dual_mov_b32 v38, v64 :: v_dual_and_b32 v7, 0xff, v7
	v_dual_mov_b32 v43, v53 :: v_dual_and_b32 v6, 0xff, v6
	v_dual_mov_b32 v26, v54 :: v_dual_and_b32 v9, 0xff, v9
	v_dual_mov_b32 v41, v51 :: v_dual_lshlrev_b32 v2, 2, v2
	v_dual_mov_b32 v35, v54 :: v_dual_add_nc_u32 v12, 11, v0
	v_dual_mov_b32 v44, v54 :: v_dual_lshlrev_b32 v3, 2, v3
	v_dual_mov_b32 v37, v56 :: v_dual_and_b32 v10, 0xff, v10
	v_dual_mov_b32 v45, v55 :: v_dual_lshlrev_b32 v4, 2, v4
	v_dual_mov_b32 v28, v56 :: v_dual_add_nc_u32 v11, 10, v0
	v_dual_mov_b32 v36, v55 :: v_dual_lshlrev_b32 v5, 2, v5
	v_dual_mov_b32 v30, v48 :: v_dual_lshlrev_b32 v7, 2, v7
	v_dual_mov_b32 v47, v64 :: v_dual_lshlrev_b32 v6, 2, v6
	v_dual_mov_b32 v18, v55 :: v_dual_lshlrev_b32 v9, 2, v9
	s_clause 0x7
	global_load_b32 v65, v1, s[6:7]
	global_load_b32 v66, v2, s[6:7]
	global_load_b32 v67, v3, s[6:7]
	global_load_b32 v68, v4, s[6:7]
	global_load_b32 v69, v5, s[6:7]
	global_load_b32 v70, v6, s[6:7]
	global_load_b32 v7, v7, s[6:7]
	global_load_b32 v71, v9, s[6:7]
	v_dual_mov_b32 v27, v55 :: v_dual_add_nc_u32 v4, 12, v0
	v_dual_mov_b32 v39, v48 :: v_dual_and_b32 v12, 0xff, v12
	v_add_nc_u32_e32 v5, 13, v0
	v_dual_mov_b32 v22, v40 :: v_dual_lshlrev_b32 v1, 2, v10
	v_dual_mov_b32 v29, v64 :: v_dual_add_nc_u32 v6, 14, v0
	v_dual_mov_b32 v14, v32 :: v_dual_add_nc_u32 v9, 15, v0
	v_dual_mov_b32 v31, v40 :: v_dual_add_nc_u32 v10, 16, v0
	v_dual_mov_b32 v20, v64 :: v_dual_and_b32 v11, 0xff, v11
	v_dual_mov_b32 v17, v54 :: v_dual_and_b32 v4, 0xff, v4
	v_dual_mov_b32 v12, v48 :: v_dual_lshlrev_b32 v3, 2, v12
	v_and_b32_e32 v5, 0xff, v5
	v_dual_mov_b32 v19, v56 :: v_dual_and_b32 v6, 0xff, v6
	v_and_b32_e32 v9, 0xff, v9
	v_dual_mov_b32 v21, v48 :: v_dual_and_b32 v10, 0xff, v10
	v_dual_mov_b32 v25, v53 :: v_dual_lshlrev_b32 v2, 2, v11
	v_dual_mov_b32 v23, v32 :: v_dual_lshlrev_b32 v4, 2, v4
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v6, 2, v6
	v_lshlrev_b32_e32 v9, 2, v9
	v_dual_mov_b32 v11, v64 :: v_dual_lshlrev_b32 v10, 2, v10
	s_clause 0x7
	global_load_b32 v72, v1, s[6:7]
	global_load_b32 v78, v2, s[6:7]
	global_load_b32 v79, v3, s[6:7]
	global_load_b32 v80, v4, s[6:7]
	global_load_b32 v81, v5, s[6:7]
	global_load_b32 v82, v6, s[6:7]
	global_load_b32 v83, v9, s[6:7]
	global_load_b32 v84, v10, s[6:7]
	s_mov_b32 s1, 0x3a83126f
	v_dual_mov_b32 v9, v55 :: v_dual_mov_b32 v10, v56
	v_mov_b32_e32 v13, v40
	v_mov_b32_e32 v15, v24
	v_dual_mov_b32 v1, v56 :: v_dual_mov_b32 v2, v64
	v_dual_mov_b32 v3, v48 :: v_dual_mov_b32 v4, v40
	v_dual_mov_b32 v5, v32 :: v_dual_mov_b32 v6, v24
	s_wait_loadcnt 0xf
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v73, v65, s1, 0
	s_wait_loadcnt 0xe
	v_fma_mixhi_f16 v73, v66, s1, 0
	s_wait_loadcnt 0xd
	v_fma_mixlo_f16 v74, v67, s1, 0
	s_wait_loadcnt 0xc
	v_fma_mixhi_f16 v74, v68, s1, 0
	s_wait_loadcnt 0xb
	v_fma_mixlo_f16 v75, v69, s1, 0
	s_wait_loadcnt 0xa
	v_fma_mixhi_f16 v75, v70, s1, 0
	s_wait_loadcnt 0x9
	v_fma_mixlo_f16 v76, v7, s1, 0
	s_wait_loadcnt 0x8
	v_fma_mixhi_f16 v76, v71, s1, 0
	v_mov_b32_e32 v7, v16
	s_wait_loadcnt 0x7
	v_fma_mixlo_f16 v77, v72, s1, 0
	s_wait_loadcnt 0x6
	v_fma_mixhi_f16 v77, v78, s1, 0
	s_wait_loadcnt 0x5
	v_fma_mixlo_f16 v78, v79, s1, 0
	s_wait_loadcnt 0x4
	v_fma_mixhi_f16 v78, v80, s1, 0
	s_wait_loadcnt 0x3
	v_fma_mixlo_f16 v79, v81, s1, 0
	s_wait_loadcnt 0x2
	v_fma_mixhi_f16 v79, v82, s1, 0
	s_wait_loadcnt 0x1
	v_fma_mixlo_f16 v80, v83, s1, 0
	s_wait_loadcnt 0x0
	v_fma_mixhi_f16 v80, v84, s1, 0
	s_mov_b32 s1, 0x3727c5ac
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], 0
	v_cvt_f16_f32_e32 v84.h, v56
	v_cvt_f16_f32_e32 v84.l, v55
	v_cvt_f16_f32_e32 v83.h, v54
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v83.l, v53
	v_cvt_f16_f32_e32 v82.h, v52
	v_cvt_f16_f32_e32 v82.l, v51
	v_cvt_f16_f32_e32 v81.h, v50
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v81.l, v49
	v_cvt_f16_f32_e32 v92.h, v64
	v_cvt_f16_f32_e32 v92.l, v63
	v_cvt_f16_f32_e32 v91.h, v62
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v91.l, v61
	v_cvt_f16_f32_e32 v90.h, v60
	v_cvt_f16_f32_e32 v90.l, v59
	v_cvt_f16_f32_e32 v89.h, v58
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v89.l, v57
	v_cvt_f16_f32_e32 v88.h, v48
	v_cvt_f16_f32_e32 v88.l, v47
	v_cvt_f16_f32_e32 v87.h, v46
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v87.l, v45
	v_cvt_f16_f32_e32 v86.h, v44
	v_cvt_f16_f32_e32 v86.l, v43
	v_cvt_f16_f32_e32 v85.h, v42
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v85.l, v41
	v_cvt_f16_f32_e32 v96.h, v40
	v_cvt_f16_f32_e32 v96.l, v39
	v_cvt_f16_f32_e32 v95.h, v38
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v95.l, v37
	v_cvt_f16_f32_e32 v94.h, v36
	v_cvt_f16_f32_e32 v94.l, v35
	v_cvt_f16_f32_e32 v93.h, v34
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v93.l, v33
	v_cvt_f16_f32_e32 v100.h, v32
	v_cvt_f16_f32_e32 v100.l, v31
	v_cvt_f16_f32_e32 v99.h, v30
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v99.l, v29
	v_cvt_f16_f32_e32 v98.h, v28
	v_cvt_f16_f32_e32 v98.l, v27
	v_cvt_f16_f32_e32 v97.h, v26
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v97.l, v25
	v_cvt_f16_f32_e32 v104.h, v24
	v_cvt_f16_f32_e32 v104.l, v23
	v_cvt_f16_f32_e32 v103.h, v22
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v103.l, v21
	v_cvt_f16_f32_e32 v102.h, v20
	v_cvt_f16_f32_e32 v102.l, v19
	v_cvt_f16_f32_e32 v101.h, v18
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v101.l, v17
	v_cvt_f16_f32_e32 v108.h, v16
	v_cvt_f16_f32_e32 v108.l, v15
	v_cvt_f16_f32_e32 v107.h, v14
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v107.l, v13
	v_cvt_f16_f32_e32 v106.h, v12
	v_cvt_f16_f32_e32 v106.l, v11
	v_cvt_f16_f32_e32 v105.h, v10
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v105.l, v9
	v_cvt_f16_f32_e32 v112.h, v8
	v_cvt_f16_f32_e32 v112.l, v7
	v_cvt_f16_f32_e32 v111.h, v6
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v111.l, v5
	v_cvt_f16_f32_e32 v110.h, v4
	v_cvt_f16_f32_e32 v110.l, v3
	v_cvt_f16_f32_e32 v109.h, v2
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_cvt_f16_f32_e32 v109.l, v1
	v_dual_mul_f32 v56, 0x3f7d70a4, v56 :: v_dual_mul_f32 v55, 0x3f7d70a4, v55
	v_dual_mul_f32 v54, 0x3f7d70a4, v54 :: v_dual_mul_f32 v53, 0x3f7d70a4, v53
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v50, 0x3f7d70a4, v50 :: v_dual_mul_f32 v49, 0x3f7d70a4, v49
	v_dual_mul_f32 v64, 0x3f7d70a4, v64 :: v_dual_mul_f32 v63, 0x3f7d70a4, v63
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v62, 0x3f7d70a4, v62 :: v_dual_mul_f32 v61, 0x3f7d70a4, v61
	v_dual_mul_f32 v60, 0x3f7d70a4, v60 :: v_dual_mul_f32 v57, 0x3f7d70a4, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v58, 0x3f7d70a4, v58 :: v_dual_mul_f32 v43, 0x3f7d70a4, v43
	v_dual_mul_f32 v46, 0x3f7d70a4, v46 :: v_dual_mul_f32 v39, 0x3f7d70a4, v39
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v42, 0x3f7d70a4, v42 :: v_dual_mul_f32 v37, 0x3f7d70a4, v37
	v_dual_mul_f32 v38, 0x3f7d70a4, v38 :: v_dual_mul_f32 v25, 0x3f7d70a4, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v36, 0x3f7d70a4, v36 :: v_dual_mul_f32 v23, 0x3f7d70a4, v23
	v_dual_mul_f32 v32, 0x3f7d70a4, v32 :: v_dual_mul_f32 v21, 0x3f7d70a4, v21
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v28, 0x3f7d70a4, v28 :: v_dual_mul_f32 v19, 0x3f7d70a4, v19
	v_dual_mul_f32 v26, 0x3f7d70a4, v26 :: v_dual_mul_f32 v17, 0x3f7d70a4, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v22, 0x3f7d70a4, v22 :: v_dual_mul_f32 v15, 0x3f7d70a4, v15
	v_dual_mul_f32 v18, 0x3f7d70a4, v18 :: v_dual_mul_f32 v13, 0x3f7d70a4, v13
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v16, 0x3f7d70a4, v16 :: v_dual_mul_f32 v11, 0x3f7d70a4, v11
	v_dual_mul_f32 v12, 0x3f7d70a4, v12 :: v_dual_mul_f32 v9, 0x3f7d70a4, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v8, 0x3f7d70a4, v8 :: v_dual_mul_f32 v7, 0x3f7d70a4, v7
	v_dual_mul_f32 v6, 0x3f7d70a4, v6 :: v_dual_mul_f32 v5, 0x3f7d70a4, v5
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_dual_mul_f32 v3, 0x3f7d70a4, v3 :: v_dual_mul_f32 v2, 0x3f7d70a4, v2
	v_mul_f32_e32 v1, 0x3f7d70a4, v1
	s_add_co_i32 s0, s0, -1
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	v_mul_f32_e32 v51, 0x3f7d70a4, v51
	v_mul_f32_e32 v29, 0x3f7d70a4, v29
	v_mul_f32_e32 v33, 0x3f7d70a4, v33
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_mul_f32_e32 v31, 0x3f7d70a4, v31
	v_mul_f32_e32 v59, 0x3f7d70a4, v59
	v_mul_f32_e32 v47, 0x3f7d70a4, v47
	v_mul_f32_e32 v45, 0x3f7d70a4, v45
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_mul_f32_e32 v41, 0x3f7d70a4, v41
	v_mul_f32_e32 v35, 0x3f7d70a4, v35
	v_dual_mul_f32 v27, 0x3f7d70a4, v27 :: v_dual_mul_f32 v4, 0x3f7d70a4, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[77:80], v[65:72]
	v_mul_f32_e32 v52, 0x3f7d70a4, v52
	v_mul_f32_e32 v48, 0x3f7d70a4, v48
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v124, 0x3727c5ac, v72 :: v_dual_add_f32 v123, 0x3727c5ac, v71
	v_dual_add_f32 v122, 0x3727c5ac, v70 :: v_dual_add_f32 v121, 0x3727c5ac, v69
	v_dual_add_f32 v120, 0x3727c5ac, v68 :: v_dual_add_f32 v119, 0x3727c5ac, v67
	v_dual_add_f32 v118, 0x3727c5ac, v66 :: v_dual_add_f32 v117, 0x3727c5ac, v65
	v_dual_add_f32 v137, 0x37a7c5ac, v72 :: v_dual_add_f32 v136, 0x37a7c5ac, v71
	v_dual_add_f32 v135, 0x37a7c5ac, v70 :: v_dual_add_f32 v134, 0x37a7c5ac, v69
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[117:124], v[73:76], v[81:84], v[117:124]
	v_dual_add_f32 v133, 0x37a7c5ac, v68 :: v_dual_add_f32 v132, 0x37a7c5ac, v67
	v_dual_add_f32 v131, 0x37a7c5ac, v66 :: v_dual_add_f32 v130, 0x37a7c5ac, v65
	v_wmma_f32_16x16x16_f16 v[117:124], v[73:76], v[89:92], v[117:124]
	v_dual_add_f32 v145, 0x37fba882, v72 :: v_dual_add_f32 v144, 0x37fba882, v71
	v_dual_add_f32 v143, 0x37fba882, v70 :: v_dual_add_f32 v142, 0x37fba882, v69
	v_dual_add_f32 v141, 0x37fba882, v68 :: v_dual_add_f32 v140, 0x37fba882, v67
	v_dual_add_f32 v139, 0x37fba882, v66 :: v_dual_add_f32 v138, 0x37fba882, v65
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[81:84], v[130:137]
	v_wmma_f32_16x16x16_f16 v[162:169], v[77:80], v[81:84], v[65:72]
	v_wmma_f32_16x16x16_f16 v[117:124], v[73:76], v[85:88], v[117:124]
	v_dual_add_f32 v153, 0x3827c5ac, v72 :: v_dual_add_f32 v152, 0x3827c5ac, v71
	v_dual_add_f32 v151, 0x3827c5ac, v70 :: v_dual_add_f32 v150, 0x3827c5ac, v69
	v_dual_add_f32 v149, 0x3827c5ac, v68 :: v_dual_add_f32 v148, 0x3827c5ac, v67
	v_dual_add_f32 v147, 0x3827c5ac, v66 :: v_dual_add_f32 v146, 0x3827c5ac, v65
	v_wmma_f32_16x16x16_f16 v[138:145], v[73:76], v[81:84], v[138:145]
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[89:92], v[130:137]
	v_wmma_f32_16x16x16_f16 v[162:169], v[77:80], v[89:92], v[162:169]
	v_wmma_f32_16x16x16_f16 v[117:124], v[73:76], v[93:96], v[117:124]
	v_wmma_f32_16x16x16_f16 v[146:153], v[73:76], v[81:84], v[146:153]
	v_wmma_f32_16x16x16_f16 v[138:145], v[73:76], v[89:92], v[138:145]
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[85:88], v[130:137]
	v_wmma_f32_16x16x16_f16 v[162:169], v[77:80], v[85:88], v[162:169]
	v_wmma_f32_16x16x16_f16 v[117:124], v[73:76], v[97:100], v[117:124]
	v_wmma_f32_16x16x16_f16 v[146:153], v[73:76], v[89:92], v[146:153]
	v_wmma_f32_16x16x16_f16 v[138:145], v[73:76], v[85:88], v[138:145]
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[93:96], v[130:137]
	v_wmma_f32_16x16x16_f16 v[162:169], v[77:80], v[93:96], v[162:169]
	v_wmma_f32_16x16x16_f16 v[117:124], v[73:76], v[101:104], v[117:124]
	v_wmma_f32_16x16x16_f16 v[146:153], v[73:76], v[85:88], v[146:153]
	v_wmma_f32_16x16x16_f16 v[138:145], v[73:76], v[93:96], v[138:145]
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[97:100], v[130:137]
	v_wmma_f32_16x16x16_f16 v[162:169], v[77:80], v[97:100], v[162:169]
	v_wmma_f32_16x16x16_f16 v[117:124], v[73:76], v[105:108], v[117:124]
	v_wmma_f32_16x16x16_f16 v[146:153], v[73:76], v[93:96], v[146:153]
	v_wmma_f32_16x16x16_f16 v[138:145], v[73:76], v[97:100], v[138:145]
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[101:104], v[130:137]
	v_wmma_f32_16x16x16_f16 v[162:169], v[77:80], v[101:104], v[162:169]
	v_wmma_f32_16x16x16_f16 v[117:124], v[73:76], v[109:112], v[117:124]
	v_wmma_f32_16x16x16_f16 v[146:153], v[73:76], v[97:100], v[146:153]
	v_wmma_f32_16x16x16_f16 v[138:145], v[73:76], v[101:104], v[138:145]
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[105:108], v[130:137]
	v_wmma_f32_16x16x16_f16 v[162:169], v[77:80], v[105:108], v[162:169]
	v_fma_mixhi_f16 v116, v124, s1, 0
	v_fma_mixlo_f16 v116, v123, s1, 0
	v_fma_mixhi_f16 v115, v122, s1, 0
	v_fma_mixlo_f16 v115, v121, s1, 0
	v_fma_mixhi_f16 v114, v120, s1, 0
	v_fma_mixlo_f16 v114, v119, s1, 0
	v_fma_mixhi_f16 v113, v118, s1, 0
	v_fma_mixlo_f16 v113, v117, s1, 0
	v_wmma_f32_16x16x16_f16 v[146:153], v[73:76], v[101:104], v[146:153]
	v_wmma_f32_16x16x16_f16 v[138:145], v[73:76], v[105:108], v[138:145]
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[109:112], v[130:137]
	v_wmma_f32_16x16x16_f16 v[162:169], v[77:80], v[109:112], v[162:169]
	v_dual_add_f32 v161, v72, v72 :: v_dual_add_f32 v160, v71, v71
	v_wmma_f32_16x16x16_f16 v[146:153], v[73:76], v[105:108], v[146:153]
	v_wmma_f32_16x16x16_f16 v[138:145], v[73:76], v[109:112], v[138:145]
	v_fma_mixhi_f16 v124, v137, s1, 0
	v_fma_mixlo_f16 v124, v136, s1, 0
	v_fma_mixhi_f16 v123, v135, s1, 0
	v_fma_mixlo_f16 v123, v134, s1, 0
	v_fma_mixhi_f16 v122, v133, s1, 0
	v_fma_mixlo_f16 v122, v132, s1, 0
	v_fma_mixhi_f16 v121, v131, s1, 0
	v_fma_mixlo_f16 v121, v130, s1, 0
	v_wmma_f32_16x16x16_f16 v[162:169], v[73:76], v[113:116], v[162:169]
	v_wmma_f32_16x16x16_f16 v[146:153], v[73:76], v[109:112], v[146:153]
	v_fma_mixhi_f16 v128, v145, s1, 0
	v_fma_mixlo_f16 v128, v144, s1, 0
	v_fma_mixhi_f16 v127, v143, s1, 0
	v_fma_mixlo_f16 v127, v142, s1, 0
	v_fma_mixhi_f16 v126, v141, s1, 0
	v_fma_mixlo_f16 v126, v140, s1, 0
	v_fma_mixhi_f16 v125, v139, s1, 0
	v_fma_mixlo_f16 v125, v138, s1, 0
	v_wmma_f32_16x16x16_f16 v[162:169], v[73:76], v[121:124], v[162:169]
	v_fma_mixhi_f16 v120, v153, s1, 0
	v_fma_mixlo_f16 v120, v152, s1, 0
	v_fma_mixhi_f16 v119, v151, s1, 0
	v_fma_mixlo_f16 v119, v150, s1, 0
	v_fma_mixhi_f16 v118, v149, s1, 0
	v_fma_mixlo_f16 v118, v148, s1, 0
	v_fma_mixhi_f16 v117, v147, s1, 0
	v_fma_mixlo_f16 v117, v146, s1, 0
	v_wmma_f32_16x16x16_f16 v[162:169], v[73:76], v[125:128], v[162:169]
	v_dual_add_f32 v159, v70, v70 :: v_dual_add_f32 v158, v69, v69
	v_dual_add_f32 v157, v68, v68 :: v_dual_add_f32 v156, v67, v67
	v_dual_add_f32 v155, v66, v66 :: v_dual_add_f32 v154, v65, v65
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[162:169], v[73:76], v[117:120], v[162:169]
	v_dual_mul_f32 v137, 0x40400000, v72 :: v_dual_mul_f32 v136, 0x40400000, v71
	v_mul_f32_e32 v135, 0x40400000, v70
	v_wmma_f32_16x16x16_f16 v[154:161], v[77:80], v[81:84], v[154:161]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_f32_e32 v130, 0, v162
	v_dual_mul_f32 v134, 0x40400000, v69 :: v_dual_mul_f32 v131, 0x40400000, v66
	v_mul_f32_e32 v133, 0x40400000, v68
	v_wmma_f32_16x16x16_f16 v[154:161], v[77:80], v[89:92], v[154:161]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_f32_e32 v130, v130, v163
	v_dual_mul_f32 v132, 0x40400000, v67 :: v_dual_mul_f32 v69, 4.0, v69
	v_mul_f32_e32 v71, 4.0, v71
	v_wmma_f32_16x16x16_f16 v[154:161], v[77:80], v[85:88], v[154:161]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_f32_e32 v138, v130, v164
	v_dual_mul_f32 v130, 0x40400000, v65 :: v_dual_mul_f32 v65, 4.0, v65
	v_mul_f32_e32 v72, 4.0, v72
	v_wmma_f32_16x16x16_f16 v[154:161], v[77:80], v[93:96], v[154:161]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v138, v138, v165
	v_wmma_f32_16x16x16_f16 v[130:137], v[77:80], v[81:84], v[130:137]
	v_dual_mul_f32 v68, 4.0, v68 :: v_dual_mul_f32 v67, 4.0, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[154:161], v[77:80], v[97:100], v[154:161]
	v_add_f32_e32 v138, v138, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[130:137], v[77:80], v[89:92], v[130:137]
	v_mul_f32_e32 v66, 4.0, v66
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[113:116], v[49:56]
	v_wmma_f32_16x16x16_f16 v[154:161], v[77:80], v[101:104], v[154:161]
	v_add_f32_e32 v138, v138, v167
	v_wmma_f32_16x16x16_f16 v[130:137], v[77:80], v[85:88], v[130:137]
	v_wmma_f32_16x16x16_f16 v[57:64], v[73:76], v[113:116], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[121:124], v[49:56]
	v_wmma_f32_16x16x16_f16 v[154:161], v[77:80], v[105:108], v[154:161]
	v_add_f32_e32 v138, v138, v168
	v_wmma_f32_16x16x16_f16 v[130:137], v[77:80], v[93:96], v[130:137]
	v_wmma_f32_16x16x16_f16 v[57:64], v[73:76], v[121:124], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[125:128], v[49:56]
	v_wmma_f32_16x16x16_f16 v[154:161], v[77:80], v[109:112], v[154:161]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[113:116], v[1:8]
	v_wmma_f32_16x16x16_f16 v[130:137], v[77:80], v[97:100], v[130:137]
	v_wmma_f32_16x16x16_f16 v[57:64], v[73:76], v[125:128], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[117:120], v[49:56]
	v_wmma_f32_16x16x16_f16 v[154:161], v[73:76], v[113:116], v[154:161]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[121:124], v[1:8]
	v_wmma_f32_16x16x16_f16 v[130:137], v[77:80], v[101:104], v[130:137]
	v_wmma_f32_16x16x16_f16 v[57:64], v[73:76], v[117:120], v[57:64]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[154:161], v[73:76], v[121:124], v[154:161]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[125:128], v[1:8]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[130:137], v[77:80], v[105:108], v[130:137]
	v_wmma_f32_16x16x16_f16 v[154:161], v[73:76], v[125:128], v[154:161]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[117:120], v[1:8]
	v_wmma_f32_16x16x16_f16 v[130:137], v[77:80], v[109:112], v[130:137]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[154:161], v[73:76], v[117:120], v[154:161]
	v_add_f32_e32 v138, v138, v169
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[113:116], v[130:137]
	v_mul_f32_e32 v70, 4.0, v70
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v138, v138, v154
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[121:124], v[130:137]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[77:80], v[81:84], v[65:72]
	v_add_f32_e32 v138, v138, v155
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[125:128], v[130:137]
	v_wmma_f32_16x16x16_f16 v[65:72], v[77:80], v[89:92], v[65:72]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[130:137], v[73:76], v[117:120], v[130:137]
	v_add_f32_e32 v138, v138, v156
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[65:72], v[77:80], v[85:88], v[65:72]
	v_add_f32_e32 v138, v138, v157
	v_mul_f32_e32 v44, 0x3f7d70a4, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[77:80], v[93:96], v[65:72]
	v_add_f32_e32 v81, v138, v158
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[113:116], v[41:48]
	v_wmma_f32_16x16x16_f16 v[65:72], v[77:80], v[97:100], v[65:72]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v81, v81, v159
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[121:124], v[41:48]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[77:80], v[101:104], v[65:72]
	v_add_f32_e32 v81, v81, v160
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[125:128], v[41:48]
	v_wmma_f32_16x16x16_f16 v[65:72], v[77:80], v[105:108], v[65:72]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v81, v81, v161
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[117:120], v[41:48]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[65:72], v[77:80], v[109:112], v[65:72]
	v_dual_add_f32 v81, v81, v130 :: v_dual_mul_f32 v40, 0x3f7d70a4, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[113:116], v[65:72]
	v_dual_add_f32 v81, v81, v131 :: v_dual_mul_f32 v34, 0x3f7d70a4, v34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[121:124], v[65:72]
	v_dual_add_f32 v81, v81, v132 :: v_dual_mul_f32 v30, 0x3f7d70a4, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[125:128], v[65:72]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[113:116], v[33:40]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v81, v81, v133 :: v_dual_mul_f32 v24, 0x3f7d70a4, v24
	v_wmma_f32_16x16x16_f16 v[65:72], v[73:76], v[117:120], v[65:72]
	v_wmma_f32_16x16x16_f16 v[25:32], v[73:76], v[113:116], v[25:32]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[121:124], v[33:40]
	v_dual_add_f32 v81, v81, v134 :: v_dual_mul_f32 v20, 0x3f7d70a4, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[25:32], v[73:76], v[121:124], v[25:32]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[125:128], v[33:40]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v81, v81, v135 :: v_dual_mul_f32 v14, 0x3f7d70a4, v14
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[113:116], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[73:76], v[125:128], v[25:32]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[117:120], v[33:40]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v81, v81, v136 :: v_dual_mul_f32 v10, 0x3f7d70a4, v10
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[121:124], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[25:32], v[73:76], v[117:120], v[25:32]
	v_add_f32_e32 v81, v81, v137
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[113:116], v[9:16]
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[125:128], v[17:24]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v65, v81, v65
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[121:124], v[9:16]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[117:120], v[17:24]
	v_add_f32_e32 v65, v65, v66
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[125:128], v[9:16]
	v_add_f32_e32 v65, v65, v67
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[117:120], v[9:16]
	v_add_f32_e32 v65, v65, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v69
	v_add_f32_e32 v65, v65, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v65, v65, v71
	v_add_f32_e32 v65, v65, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v65
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v65, 0x3a83126f, v129, vcc_lo
	v_cvt_f16_f32_e32 v77.l, v65
	s_cbranch_scc1 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v7, v16 :: v_dual_mov_b32 v6, v24
	v_dual_mov_b32 v5, v32 :: v_dual_mov_b32 v4, v40
	v_dual_mov_b32 v3, v48 :: v_dual_mov_b32 v2, v64
	v_dual_mov_b32 v1, v56 :: v_dual_mov_b32 v14, v32
	v_dual_mov_b32 v15, v24 :: v_dual_mov_b32 v12, v48
	v_dual_mov_b32 v13, v40 :: v_dual_mov_b32 v10, v56
	v_dual_mov_b32 v11, v64 :: v_dual_mov_b32 v22, v40
	v_dual_mov_b32 v9, v55 :: v_dual_mov_b32 v20, v64
	v_dual_mov_b32 v23, v32 :: v_dual_mov_b32 v18, v55
	v_dual_mov_b32 v21, v48 :: v_dual_mov_b32 v30, v48
	v_dual_mov_b32 v19, v56 :: v_dual_mov_b32 v28, v56
	v_dual_mov_b32 v17, v54 :: v_dual_mov_b32 v26, v54
	v_dual_mov_b32 v31, v40 :: v_dual_mov_b32 v38, v64
	v_dual_mov_b32 v29, v64 :: v_dual_mov_b32 v36, v55
	v_dual_mov_b32 v27, v55 :: v_dual_mov_b32 v34, v53
	v_dual_mov_b32 v25, v53 :: v_dual_mov_b32 v46, v56
	v_dual_mov_b32 v39, v48 :: v_dual_mov_b32 v44, v54
	v_dual_mov_b32 v37, v56 :: v_dual_mov_b32 v42, v52
	v_dual_mov_b32 v35, v54 :: v_dual_mov_b32 v62, v55
	v_dual_mov_b32 v33, v52 :: v_dual_mov_b32 v60, v53
	v_dual_mov_b32 v47, v64 :: v_dual_mov_b32 v58, v51
	v_mov_b32_e32 v45, v55
	v_mov_b32_e32 v43, v53
	v_mov_b32_e32 v41, v51
	v_mov_b32_e32 v63, v56
	v_mov_b32_e32 v61, v54
	v_mov_b32_e32 v59, v52
	v_mov_b32_e32 v57, v50
.LBB0_4:
	v_add_f32_e32 v49, 0, v49
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v50
	v_add_f32_e32 v49, v49, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v52
	v_add_f32_e32 v49, v49, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v54
	v_add_f32_e32 v49, v49, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v56
	v_add_f32_e32 v49, v49, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v58
	v_add_f32_e32 v49, v49, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v60
	v_add_f32_e32 v49, v49, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v62
	v_add_f32_e32 v49, v49, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v64
	v_add_f32_e32 v41, v49, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v42
	v_add_f32_e32 v41, v41, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v44
	v_add_f32_e32 v41, v41, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v46
	v_add_f32_e32 v41, v41, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v48
	v_add_f32_e32 v33, v41, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v34
	v_add_f32_e32 v33, v33, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v36
	v_add_f32_e32 v33, v33, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v38
	v_add_f32_e32 v33, v33, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v40
	v_add_f32_e32 v25, v33, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v26
	v_add_f32_e32 v25, v25, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v28
	v_add_f32_e32 v25, v25, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v30
	v_add_f32_e32 v25, v25, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v32
	v_add_f32_e32 v17, v25, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v1, v6 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_f32_e32 v2, v2, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	_Z5probeILi8ELb0ELb0EEvPfPKfi, .Lfunc_end0-_Z5probeILi8ELb0ELb0EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi8ELb0ELb0EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 170
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z5probeILi8ELb0ELb0EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi8ELb0ELb0EEvPfPKfi,"axG",@progbits,_Z5probeILi8ELb0ELb0EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.num_vgpr, 170
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.numbered_sgpr, 8
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi8ELb0ELb0EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4464
; TotalNumSgprs: 10
; NumVgprs: 170
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 170
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi8ELb1ELb0EEvPfPKfi,"axG",@progbits,_Z5probeILi8ELb1ELb0EEvPfPKfi,comdat
	.protected	_Z5probeILi8ELb1ELb0EEvPfPKfi ; -- Begin function _Z5probeILi8ELb1ELb0EEvPfPKfi
	.globl	_Z5probeILi8ELb1ELb0EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi8ELb1ELb0EEvPfPKfi,@function
_Z5probeILi8ELb1ELb0EEvPfPKfi:          ; @_Z5probeILi8ELb1ELb0EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	s_load_b32 s0, s[0:1], 0x10
	s_mov_b32 s1, 0x3c23d70a
	s_mov_b32 s2, 0x38d1b717
	s_mov_b32 s3, 0x3951b717
	s_mov_b32 s8, 0x399d4951
	s_mov_b32 s9, 0x39d1b717
	s_mov_b32 s10, 0x3a03126e
	s_mov_b32 s11, 0x3a1d4951
	s_mov_b32 s12, 0x3a378034
	s_mov_b32 s13, 0x3a51b717
	s_mov_b32 s14, 0x3a6bedfa
	s_mov_b32 s15, 0x3a83126e
	s_mov_b32 s16, 0x3a902de0
	s_mov_b32 s17, 0x3a9d4951
	s_mov_b32 s18, 0x3aaa64c3
	s_mov_b32 s19, 0x3ab78034
	s_wait_kmcnt 0x0
	global_load_b32 v2, v1, s[6:7]
	s_cmp_lt_i32 s0, 1
	s_wait_loadcnt 0x0
	v_fma_mixhi_f16 v37, v2, s1, s2
	v_fma_mixhi_f16 v25, v2, s1, s3
	v_fma_mixlo_f16 v26, v2, s1, s8
	v_fma_mixhi_f16 v26, v2, s1, s9
	v_fma_mixlo_f16 v27, v2, s1, s10
	v_fma_mixhi_f16 v27, v2, s1, s11
	v_fma_mixlo_f16 v28, v2, s1, s12
	v_fma_mixhi_f16 v28, v2, s1, s13
	v_fma_mixlo_f16 v20, v2, s1, s14
	v_fma_mixhi_f16 v20, v2, s1, s15
	v_fma_mixlo_f16 v16, v2, s1, s16
	v_fma_mixhi_f16 v16, v2, s1, s17
	v_fma_mixhi_f16 v12, v2, s1, s18
	v_fma_mixlo_f16 v37, v2, s1, 0
	v_fma_mixhi_f16 v32, v2, s1, s19
	v_mov_b16_e32 v25.l, v37.h
	v_mov_b16_e32 v38.l, v25.h
	v_mov_b16_e32 v38.h, v26.l
	v_mov_b16_e32 v39.l, v26.h
	v_mov_b16_e32 v39.h, v27.l
	v_mov_b16_e32 v40.l, v27.h
	v_mov_b16_e32 v40.h, v28.l
	v_mov_b16_e32 v36.l, v28.h
	v_mov_b16_e32 v36.h, v20.l
	v_mov_b16_e32 v24.l, v20.h
	v_mov_b16_e32 v24.h, v16.l
	v_mov_b16_e32 v12.l, v16.h
	v_mov_b16_e32 v32.l, v12.h
	s_cbranch_scc1 .LBB1_3
; %bb.1:
	v_add_nc_u32_e32 v9, 9, v0
	v_dual_mov_b32 v29, v28 :: v_dual_add_nc_u32 v10, 10, v0
	v_add_nc_u32_e32 v11, 11, v0
	v_add_nc_u32_e32 v13, 12, v0
	v_add_nc_u32_e32 v14, 13, v0
	v_and_b32_e32 v9, 0xff, v9
	v_add_nc_u32_e32 v15, 14, v0
	v_add_nc_u32_e32 v17, 15, v0
	v_add_nc_u32_e32 v18, 16, v0
	v_and_b32_e32 v10, 0xff, v10
	v_and_b32_e32 v11, 0xff, v11
	v_and_b32_e32 v13, 0xff, v13
	v_and_b32_e32 v14, 0xff, v14
	v_lshlrev_b32_e32 v9, 2, v9
	v_and_b32_e32 v15, 0xff, v15
	v_and_b32_e32 v17, 0xff, v17
	v_and_b32_e32 v18, 0xff, v18
	v_lshlrev_b32_e32 v10, 2, v10
	v_lshlrev_b32_e32 v11, 2, v11
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v15, 2, v15
	v_lshlrev_b32_e32 v17, 2, v17
	v_lshlrev_b32_e32 v18, 2, v18
	s_clause 0x7
	global_load_b32 v31, v9, s[6:7]
	global_load_b32 v46, v10, s[6:7]
	global_load_b32 v47, v11, s[6:7]
	global_load_b32 v48, v13, s[6:7]
	global_load_b32 v49, v14, s[6:7]
	global_load_b32 v50, v15, s[6:7]
	global_load_b32 v51, v17, s[6:7]
	global_load_b32 v52, v18, s[6:7]
	v_dual_mov_b32 v65, 0x3b03126f :: v_dual_add_nc_u32 v2, 1, v0
	v_dual_mov_b32 v34, v39 :: v_dual_add_nc_u32 v3, 2, v0
	v_dual_mov_b32 v33, v38 :: v_dual_add_nc_u32 v4, 3, v0
	v_dual_mov_b32 v18, v27 :: v_dual_add_nc_u32 v5, 4, v0
	v_dual_mov_b32 v17, v26 :: v_dual_add_nc_u32 v6, 5, v0
	v_dual_mov_b32 v30, v20 :: v_dual_add_nc_u32 v7, 6, v0
	v_dual_mov_b32 v13, v27 :: v_dual_add_nc_u32 v8, 7, v0
	v_dual_mov_b32 v35, v40 :: v_dual_and_b32 v2, 0xff, v2
	v_dual_mov_b32 v22, v40 :: v_dual_and_b32 v3, 0xff, v3
	v_dual_mov_b32 v19, v28 :: v_dual_and_b32 v4, 0xff, v4
	v_dual_mov_b32 v14, v28 :: v_dual_and_b32 v5, 0xff, v5
	v_dual_mov_b32 v23, v36 :: v_dual_and_b32 v6, 0xff, v6
	v_and_b32_e32 v7, 0xff, v7
	v_dual_mov_b32 v11, v24 :: v_dual_and_b32 v8, 0xff, v8
	v_dual_mov_b32 v21, v39 :: v_dual_lshlrev_b32 v2, 2, v2
	v_dual_mov_b32 v10, v36 :: v_dual_lshlrev_b32 v3, 2, v3
	v_dual_mov_b32 v15, v20 :: v_dual_lshlrev_b32 v4, 2, v4
	s_mov_b32 s1, 0x3a83126f
	v_lshlrev_b32_e32 v5, 2, v5
	v_dual_mov_b32 v9, v40 :: v_dual_lshlrev_b32 v6, 2, v6
	v_lshlrev_b32_e32 v7, 2, v7
	v_lshlrev_b32_e32 v8, 2, v8
	s_mov_b32 s2, 0x3f7d70a4
	s_wait_loadcnt 0x7
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v45, v31, s1, 0
	v_mov_b32_e32 v31, v16
	s_clause 0x7
	global_load_b32 v1, v1, s[6:7]
	global_load_b32 v2, v2, s[6:7]
	global_load_b32 v3, v3, s[6:7]
	global_load_b32 v4, v4, s[6:7]
	global_load_b32 v5, v5, s[6:7]
	global_load_b32 v6, v6, s[6:7]
	global_load_b32 v7, v7, s[6:7]
	global_load_b32 v8, v8, s[6:7]
	s_wait_loadcnt 0xe
	v_fma_mixhi_f16 v45, v46, s1, 0
	s_wait_loadcnt 0xd
	v_fma_mixlo_f16 v46, v47, s1, 0
	s_wait_loadcnt 0xc
	v_fma_mixhi_f16 v46, v48, s1, 0
	s_wait_loadcnt 0xb
	v_fma_mixlo_f16 v47, v49, s1, 0
	s_wait_loadcnt 0xa
	v_fma_mixhi_f16 v47, v50, s1, 0
	s_wait_loadcnt 0x9
	v_fma_mixlo_f16 v48, v51, s1, 0
	s_wait_loadcnt 0x8
	v_fma_mixhi_f16 v48, v52, s1, 0
	s_wait_loadcnt 0x7
	v_fma_mixlo_f16 v41, v1, s1, 0
	s_wait_loadcnt 0x6
	v_fma_mixhi_f16 v41, v2, s1, 0
	s_wait_loadcnt 0x5
	v_fma_mixlo_f16 v42, v3, s1, 0
	s_wait_loadcnt 0x4
	v_fma_mixhi_f16 v42, v4, s1, 0
	s_wait_loadcnt 0x3
	v_fma_mixlo_f16 v43, v5, s1, 0
	s_wait_loadcnt 0x2
	v_fma_mixhi_f16 v43, v6, s1, 0
	s_wait_loadcnt 0x1
	v_fma_mixlo_f16 v44, v7, s1, 0
	s_wait_loadcnt 0x0
	v_fma_mixhi_f16 v44, v8, s1, 0
	s_mov_b32 s1, 0x3727c5ac
.LBB1_2:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], 0
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v60, 0x3727c5ac, v8 :: v_dual_add_f32 v59, 0x3727c5ac, v7
	v_dual_add_f32 v58, 0x3727c5ac, v6 :: v_dual_add_f32 v57, 0x3727c5ac, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v56, 0x3727c5ac, v4 :: v_dual_add_f32 v55, 0x3727c5ac, v3
	v_dual_add_f32 v54, 0x3727c5ac, v2 :: v_dual_add_f32 v53, 0x3727c5ac, v1
	v_dual_add_f32 v73, 0x37a7c5ac, v8 :: v_dual_add_f32 v72, 0x37a7c5ac, v7
	v_dual_add_f32 v71, 0x37a7c5ac, v6 :: v_dual_add_f32 v70, 0x37a7c5ac, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[53:60], v[41:44], v[37:40], v[53:60]
	v_dual_add_f32 v69, 0x37a7c5ac, v4 :: v_dual_add_f32 v68, 0x37a7c5ac, v3
	v_dual_add_f32 v67, 0x37a7c5ac, v2 :: v_dual_add_f32 v66, 0x37a7c5ac, v1
	v_wmma_f32_16x16x16_f16 v[53:60], v[41:44], v[25:28], v[53:60]
	v_dual_add_f32 v81, 0x37fba882, v8 :: v_dual_add_f32 v80, 0x37fba882, v7
	v_dual_add_f32 v79, 0x37fba882, v6 :: v_dual_add_f32 v78, 0x37fba882, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[37:40], v[66:73]
	v_dual_add_f32 v77, 0x37fba882, v4 :: v_dual_add_f32 v76, 0x37fba882, v3
	v_dual_add_f32 v75, 0x37fba882, v2 :: v_dual_add_f32 v74, 0x37fba882, v1
	v_wmma_f32_16x16x16_f16 v[90:97], v[45:48], v[37:40], v[1:8]
	v_wmma_f32_16x16x16_f16 v[53:60], v[41:44], v[33:36], v[53:60]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[25:28], v[66:73]
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[74:81], v[41:44], v[37:40], v[74:81]
	v_dual_add_f32 v89, 0x3827c5ac, v8 :: v_dual_add_f32 v88, 0x3827c5ac, v7
	v_dual_add_f32 v87, 0x3827c5ac, v6 :: v_dual_add_f32 v86, 0x3827c5ac, v5
	v_dual_add_f32 v85, 0x3827c5ac, v4 :: v_dual_add_f32 v84, 0x3827c5ac, v3
	v_dual_add_f32 v83, 0x3827c5ac, v2 :: v_dual_add_f32 v82, 0x3827c5ac, v1
	v_wmma_f32_16x16x16_f16 v[90:97], v[45:48], v[25:28], v[90:97]
	v_wmma_f32_16x16x16_f16 v[53:60], v[41:44], v[17:20], v[53:60]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[33:36], v[66:73]
	v_wmma_f32_16x16x16_f16 v[74:81], v[41:44], v[25:28], v[74:81]
	v_wmma_f32_16x16x16_f16 v[82:89], v[41:44], v[37:40], v[82:89]
	v_wmma_f32_16x16x16_f16 v[90:97], v[45:48], v[33:36], v[90:97]
	v_wmma_f32_16x16x16_f16 v[53:60], v[41:44], v[21:24], v[53:60]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[17:20], v[66:73]
	v_wmma_f32_16x16x16_f16 v[74:81], v[41:44], v[33:36], v[74:81]
	v_wmma_f32_16x16x16_f16 v[82:89], v[41:44], v[25:28], v[82:89]
	v_wmma_f32_16x16x16_f16 v[90:97], v[45:48], v[17:20], v[90:97]
	v_wmma_f32_16x16x16_f16 v[53:60], v[41:44], v[13:16], v[53:60]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[21:24], v[66:73]
	v_wmma_f32_16x16x16_f16 v[74:81], v[41:44], v[17:20], v[74:81]
	v_wmma_f32_16x16x16_f16 v[82:89], v[41:44], v[33:36], v[82:89]
	v_wmma_f32_16x16x16_f16 v[90:97], v[45:48], v[21:24], v[90:97]
	v_wmma_f32_16x16x16_f16 v[53:60], v[41:44], v[9:12], v[53:60]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[13:16], v[66:73]
	v_wmma_f32_16x16x16_f16 v[74:81], v[41:44], v[21:24], v[74:81]
	v_wmma_f32_16x16x16_f16 v[82:89], v[41:44], v[17:20], v[82:89]
	v_wmma_f32_16x16x16_f16 v[90:97], v[45:48], v[13:16], v[90:97]
	v_wmma_f32_16x16x16_f16 v[53:60], v[41:44], v[29:32], v[53:60]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[9:12], v[66:73]
	v_wmma_f32_16x16x16_f16 v[74:81], v[41:44], v[13:16], v[74:81]
	v_wmma_f32_16x16x16_f16 v[82:89], v[41:44], v[21:24], v[82:89]
	v_wmma_f32_16x16x16_f16 v[90:97], v[45:48], v[9:12], v[90:97]
	v_fma_mixhi_f16 v52, v60, s1, 0
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[29:32], v[66:73]
	v_wmma_f32_16x16x16_f16 v[74:81], v[41:44], v[9:12], v[74:81]
	v_wmma_f32_16x16x16_f16 v[82:89], v[41:44], v[13:16], v[82:89]
	v_wmma_f32_16x16x16_f16 v[90:97], v[45:48], v[29:32], v[90:97]
	v_fma_mixlo_f16 v52, v59, s1, 0
	v_fma_mixhi_f16 v51, v58, s1, 0
	v_fma_mixlo_f16 v51, v57, s1, 0
	v_fma_mixhi_f16 v50, v56, s1, 0
	v_fma_mixlo_f16 v50, v55, s1, 0
	v_fma_mixhi_f16 v49, v54, s1, 0
	v_fma_mixlo_f16 v49, v53, s1, 0
	v_wmma_f32_16x16x16_f16 v[74:81], v[41:44], v[29:32], v[74:81]
	v_wmma_f32_16x16x16_f16 v[82:89], v[41:44], v[9:12], v[82:89]
	v_fma_mixhi_f16 v64, v73, s1, 0
	v_fma_mixlo_f16 v64, v72, s1, 0
	v_wmma_f32_16x16x16_f16 v[90:97], v[41:44], v[49:52], v[90:97]
	v_fma_mixhi_f16 v63, v71, s1, 0
	v_fma_mixlo_f16 v63, v70, s1, 0
	v_fma_mixhi_f16 v62, v69, s1, 0
	v_fma_mixlo_f16 v62, v68, s1, 0
	v_fma_mixhi_f16 v61, v67, s1, 0
	v_fma_mixlo_f16 v61, v66, s1, 0
	v_wmma_f32_16x16x16_f16 v[82:89], v[41:44], v[29:32], v[82:89]
	v_fma_mixhi_f16 v60, v81, s1, 0
	v_fma_mixlo_f16 v60, v80, s1, 0
	v_fma_mixhi_f16 v59, v79, s1, 0
	v_wmma_f32_16x16x16_f16 v[90:97], v[41:44], v[61:64], v[90:97]
	v_fma_mixlo_f16 v59, v78, s1, 0
	v_fma_mixhi_f16 v58, v77, s1, 0
	v_fma_mixlo_f16 v58, v76, s1, 0
	v_fma_mixhi_f16 v57, v75, s1, 0
	v_fma_mixlo_f16 v57, v74, s1, 0
	v_fma_mixhi_f16 v56, v89, s1, 0
	v_fma_mixlo_f16 v56, v88, s1, 0
	v_fma_mixhi_f16 v55, v87, s1, 0
	v_fma_mixlo_f16 v55, v86, s1, 0
	v_wmma_f32_16x16x16_f16 v[90:97], v[41:44], v[57:60], v[90:97]
	v_fma_mixhi_f16 v54, v85, s1, 0
	v_fma_mixlo_f16 v54, v84, s1, 0
	v_fma_mixhi_f16 v53, v83, s1, 0
	v_fma_mixlo_f16 v53, v82, s1, 0
	v_add_f32_e32 v73, v8, v8
	v_dual_add_f32 v69, v4, v4 :: v_dual_add_f32 v72, v7, v7
	v_add_f32_e32 v71, v6, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[90:97], v[41:44], v[53:56], v[90:97]
	v_dual_add_f32 v70, v5, v5 :: v_dual_add_f32 v67, v2, v2
	v_add_f32_e32 v68, v3, v3
	v_add_f32_e32 v66, 0, v90
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v91
	v_add_f32_e32 v66, v66, v92
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v93
	v_add_f32_e32 v66, v66, v94
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v66, v66, v95
	v_add_f32_e32 v66, v66, v96
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v74, v66, v97
	v_add_f32_e32 v66, v1, v1
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[37:40], v[66:73]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[25:28], v[66:73]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[33:36], v[66:73]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[17:20], v[66:73]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[21:24], v[66:73]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[13:16], v[66:73]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[9:12], v[66:73]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[29:32], v[66:73]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[49:52], v[66:73]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[61:64], v[66:73]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[57:60], v[66:73]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[53:56], v[66:73]
	v_add_f32_e32 v66, v74, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v66, v66, v67 :: v_dual_mul_f32 v67, 0x40400000, v2
	v_mul_f32_e32 v2, 4.0, v2
	v_add_f32_e32 v66, v66, v68
	v_dual_mul_f32 v68, 0x40400000, v3 :: v_dual_mul_f32 v3, 4.0, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v66, v66, v69 :: v_dual_mul_f32 v69, 0x40400000, v4
	v_mul_f32_e32 v4, 4.0, v4
	v_add_f32_e32 v66, v66, v70
	v_dual_mul_f32 v70, 0x40400000, v5 :: v_dual_mul_f32 v5, 4.0, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v66, v66, v71 :: v_dual_mul_f32 v71, 0x40400000, v6
	v_mul_f32_e32 v6, 4.0, v6
	v_add_f32_e32 v66, v66, v72
	v_dual_mul_f32 v72, 0x40400000, v7 :: v_dual_mul_f32 v7, 4.0, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_add_f32 v74, v66, v73 :: v_dual_mul_f32 v73, 0x40400000, v8
	v_mul_f32_e32 v66, 0x40400000, v1
	v_dual_mul_f32 v8, 4.0, v8 :: v_dual_mul_f32 v1, 4.0, v1
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[37:40], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[37:40], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[25:28], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[25:28], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[33:36], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[33:36], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[17:20], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[17:20], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[21:24], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[21:24], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[13:16], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[13:16], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[9:12], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[9:12], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[45:48], v[29:32], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[49:52], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[49:52], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[61:64], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[61:64], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[57:60], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[57:60], v[1:8]
	v_wmma_f32_16x16x16_f16 v[66:73], v[41:44], v[53:56], v[66:73]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[53:56], v[1:8]
	v_add_f32_e32 v66, v74, v66
	v_fma_mix_f32 v74, v40, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v66, v66, v67
	v_fma_mix_f32 v67, v37, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v66, v66, v68
	v_fma_mix_f32 v68, v37, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v66, v66, v69
	v_fma_mix_f32 v69, v38, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v66, v66, v70
	v_fma_mix_f32 v70, v38, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v66, v66, v71
	v_fma_mix_f32 v71, v39, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v66, v66, v72
	v_fma_mix_f32 v72, v39, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v66, v66, v73
	v_fma_mix_f32 v73, v40, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v66, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[49:52], v[67:74]
	v_add_f32_e32 v1, v1, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[61:64], v[67:74]
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[57:60], v[67:74]
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[53:56], v[67:74]
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v37.h, v68
	v_cvt_f16_f32_e32 v37.l, v67
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f16_f32_e32 v38.h, v70
	v_cvt_f16_f32_e32 v38.l, v69
	v_cvt_f16_f32_e32 v39.h, v72
	v_cvt_f16_f32_e32 v39.l, v71
	v_cvt_f16_f32_e32 v40.h, v74
	v_cvt_f16_f32_e32 v40.l, v73
	v_fma_mix_f32 v74, v28, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v73, v28, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v27, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v71, v27, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v70, v26, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v26, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v25, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v25, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[49:52], v[67:74]
	v_add_f32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[61:64], v[67:74]
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[57:60], v[67:74]
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[53:56], v[67:74]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x3a83126f, v65, vcc_lo
	v_cvt_f16_f32_e32 v25.h, v68
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v25.l, v67
	v_cvt_f16_f32_e32 v26.h, v70
	v_cvt_f16_f32_e32 v26.l, v69
	v_cvt_f16_f32_e32 v27.h, v72
	v_cvt_f16_f32_e32 v27.l, v71
	v_cvt_f16_f32_e32 v28.h, v74
	v_cvt_f16_f32_e32 v28.l, v73
	v_fma_mix_f32 v74, v36, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v73, v36, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v35, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v71, v35, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v70, v34, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v34, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v33, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v33, s2, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f16_f32_e32 v45.l, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[49:52], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[61:64], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[57:60], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[53:56], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v33.h, v68
	v_cvt_f16_f32_e32 v33.l, v67
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v34.h, v70
	v_cvt_f16_f32_e32 v34.l, v69
	v_cvt_f16_f32_e32 v35.h, v72
	v_cvt_f16_f32_e32 v35.l, v71
	v_cvt_f16_f32_e32 v36.h, v74
	v_cvt_f16_f32_e32 v36.l, v73
	v_fma_mix_f32 v74, v20, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v73, v20, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v19, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v71, v19, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v70, v18, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v18, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v17, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v17, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[49:52], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[61:64], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[57:60], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[53:56], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v18.h, v70
	v_cvt_f16_f32_e32 v18.l, v69
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v19.h, v72
	v_cvt_f16_f32_e32 v19.l, v71
	v_cvt_f16_f32_e32 v20.h, v74
	v_cvt_f16_f32_e32 v20.l, v73
	v_cvt_f16_f32_e32 v17.h, v68
	v_cvt_f16_f32_e32 v17.l, v67
	v_fma_mix_f32 v74, v24, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v73, v24, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v23, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v71, v23, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v70, v22, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v22, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v21, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v21, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[49:52], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[61:64], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[57:60], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[53:56], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v21.h, v68
	v_cvt_f16_f32_e32 v21.l, v67
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v22.h, v70
	v_cvt_f16_f32_e32 v22.l, v69
	v_cvt_f16_f32_e32 v23.h, v72
	v_cvt_f16_f32_e32 v23.l, v71
	v_cvt_f16_f32_e32 v24.h, v74
	v_cvt_f16_f32_e32 v24.l, v73
	v_fma_mix_f32 v74, v16, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v73, v16, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v15, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v71, v15, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v70, v14, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v14, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v13, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v13, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[49:52], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[61:64], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[57:60], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[53:56], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v13.h, v68
	v_cvt_f16_f32_e32 v13.l, v67
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v14.h, v70
	v_cvt_f16_f32_e32 v14.l, v69
	v_cvt_f16_f32_e32 v15.h, v72
	v_cvt_f16_f32_e32 v15.l, v71
	v_cvt_f16_f32_e32 v16.h, v74
	v_cvt_f16_f32_e32 v16.l, v73
	v_fma_mix_f32 v74, v12, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v73, v12, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v11, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v71, v11, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v70, v10, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v10, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v9, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v9, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[49:52], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[61:64], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[57:60], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[53:56], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v9.h, v68
	v_cvt_f16_f32_e32 v9.l, v67
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v10.h, v70
	v_cvt_f16_f32_e32 v10.l, v69
	v_cvt_f16_f32_e32 v11.h, v72
	v_cvt_f16_f32_e32 v11.l, v71
	v_cvt_f16_f32_e32 v12.h, v74
	v_cvt_f16_f32_e32 v12.l, v73
	v_fma_mix_f32 v74, v32, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v73, v32, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v31, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v71, v31, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v70, v30, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v30, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v29, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v29, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[49:52], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[61:64], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[57:60], v[67:74]
	v_wmma_f32_16x16x16_f16 v[67:74], v[41:44], v[53:56], v[67:74]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v30.h, v70
	v_cvt_f16_f32_e32 v30.l, v69
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v31.h, v72
	v_cvt_f16_f32_e32 v31.l, v71
	v_cvt_f16_f32_e32 v32.h, v74
	v_cvt_f16_f32_e32 v32.l, v73
	v_cvt_f16_f32_e32 v29.h, v68
	v_cvt_f16_f32_e32 v29.l, v67
	s_cbranch_scc1 .LBB1_2
	s_branch .LBB1_4
.LBB1_3:
	v_dual_mov_b32 v31, v16 :: v_dual_mov_b32 v30, v20
	v_dual_mov_b32 v29, v28 :: v_dual_mov_b32 v10, v36
	v_dual_mov_b32 v11, v24 :: v_dual_mov_b32 v14, v28
	v_dual_mov_b32 v9, v40 :: v_dual_mov_b32 v22, v40
	v_dual_mov_b32 v15, v20 :: v_dual_mov_b32 v18, v27
	v_dual_mov_b32 v13, v27 :: v_dual_mov_b32 v34, v39
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v21, v39
	v_mov_b32_e32 v19, v28
	v_mov_b32_e32 v17, v26
	v_mov_b32_e32 v35, v40
	v_mov_b32_e32 v33, v38
.LBB1_4:
	v_fma_mix_f32 v1, v37, 1.0, 0 op_sel_hi:[1,1,0]
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v37, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v38, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v38, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v39, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v39, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v40, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v40, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v25, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v25, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v26, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v26, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v27, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v27, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v28, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v28, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v33, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v33, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v34, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v34, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v35, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v35, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v36, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v36, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v17, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v17, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v18, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v18, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v19, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v19, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v20, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v20, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v21, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v21, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v22, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v22, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v23, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v23, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v24, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v24, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v13, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v13, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v14, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v14, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v15, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v15, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v16, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v16, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v9, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v9, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v10, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v10, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v11, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v11, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v12, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v12, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v29, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v29, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v30, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v30, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v31, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v2, v31, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mov_b32_e32 v1, 0
	v_fma_mix_f32 v2, v32, 1.0, v2 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fma_mix_f32 v2, v32, 1.0, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end1:
	.size	_Z5probeILi8ELb1ELb0EEvPfPKfi, .Lfunc_end1-_Z5probeILi8ELb1ELb0EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi8ELb1ELb0EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 98
		.amdhsa_next_free_sgpr 20
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_Z5probeILi8ELb1ELb0EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi8ELb1ELb0EEvPfPKfi,"axG",@progbits,_Z5probeILi8ELb1ELb0EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.num_vgpr, 98
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.numbered_sgpr, 20
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi8ELb1ELb0EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4872
; TotalNumSgprs: 22
; NumVgprs: 98
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 98
; Occupancy: 12
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi4ELb0ELb0EEvPfPKfi,"axG",@progbits,_Z5probeILi4ELb0ELb0EEvPfPKfi,comdat
	.protected	_Z5probeILi4ELb0ELb0EEvPfPKfi ; -- Begin function _Z5probeILi4ELb0ELb0EEvPfPKfi
	.globl	_Z5probeILi4ELb0ELb0EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi4ELb0ELb0EEvPfPKfi,@function
_Z5probeILi4ELb0ELb0EEvPfPKfi:          ; @_Z5probeILi4ELb0ELb0EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s0, s[0:1], 0x10
	s_mov_b32 s1, 0x3c23d70a
	v_lshlrev_b32_e32 v1, 2, v0
	s_wait_kmcnt 0x0
	global_load_b32 v2, v1, s[6:7]
	s_cmp_lt_i32 s0, 1
	s_wait_loadcnt 0x0
	v_fma_f32 v17, 0x3c23d70a, v2, 0
	v_fmaak_f32 v18, s1, v2, 0x38d1b717
	v_fmaak_f32 v19, s1, v2, 0x3951b717
	v_fmaak_f32 v20, s1, v2, 0x399d4951
	v_fmaak_f32 v21, s1, v2, 0x39d1b717
	v_fmaak_f32 v22, s1, v2, 0x3a03126e
	v_fmaak_f32 v23, s1, v2, 0x3a1d4951
	v_fmaak_f32 v24, s1, v2, 0x3a378034
	v_fmaak_f32 v32, s1, v2, 0x3a51b717
	v_fmaak_f32 v16, s1, v2, 0x3a6bedfa
	v_fmaak_f32 v8, s1, v2, 0x3a83126e
	s_cbranch_scc1 .LBB2_3
; %bb.1:
	v_dual_mov_b32 v69, 0x3b03126f :: v_dual_add_nc_u32 v2, 1, v0
	v_dual_mov_b32 v26, v19 :: v_dual_add_nc_u32 v3, 2, v0
	v_dual_mov_b32 v25, v18 :: v_dual_add_nc_u32 v4, 3, v0
	v_dual_mov_b32 v28, v21 :: v_dual_add_nc_u32 v5, 4, v0
	v_dual_mov_b32 v14, v24 :: v_dual_add_nc_u32 v7, 6, v0
	v_dual_mov_b32 v29, v22 :: v_dual_add_nc_u32 v6, 5, v0
	v_add_nc_u32_e32 v9, 7, v0
	v_dual_mov_b32 v27, v20 :: v_dual_and_b32 v2, 0xff, v2
	v_dual_mov_b32 v30, v23 :: v_dual_and_b32 v3, 0xff, v3
	v_dual_mov_b32 v31, v24 :: v_dual_and_b32 v4, 0xff, v4
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v7, 0xff, v7
	v_and_b32_e32 v6, 0xff, v6
	v_and_b32_e32 v9, 0xff, v9
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v3, 2, v3
	v_dual_mov_b32 v13, v23 :: v_dual_lshlrev_b32 v4, 2, v4
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v7, 2, v7
	v_dual_mov_b32 v15, v32 :: v_dual_lshlrev_b32 v6, 2, v6
	v_lshlrev_b32_e32 v9, 2, v9
	s_clause 0x7
	global_load_b32 v33, v1, s[6:7]
	global_load_b32 v34, v2, s[6:7]
	global_load_b32 v35, v3, s[6:7]
	global_load_b32 v36, v4, s[6:7]
	global_load_b32 v37, v5, s[6:7]
	global_load_b32 v38, v6, s[6:7]
	global_load_b32 v7, v7, s[6:7]
	global_load_b32 v39, v9, s[6:7]
	v_add_nc_u32_e32 v10, 9, v0
	v_add_nc_u32_e32 v11, 10, v0
	v_add_nc_u32_e32 v12, 11, v0
	v_add_nc_u32_e32 v4, 12, v0
	v_add_nc_u32_e32 v5, 13, v0
	v_and_b32_e32 v10, 0xff, v10
	v_add_nc_u32_e32 v6, 14, v0
	v_add_nc_u32_e32 v9, 15, v0
	v_and_b32_e32 v11, 0xff, v11
	v_and_b32_e32 v12, 0xff, v12
	v_lshlrev_b32_e32 v1, 2, v10
	v_add_nc_u32_e32 v10, 16, v0
	v_and_b32_e32 v4, 0xff, v4
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v6, 0xff, v6
	v_and_b32_e32 v9, 0xff, v9
	v_and_b32_e32 v10, 0xff, v10
	s_mov_b32 s1, 0x3a83126f
	v_lshlrev_b32_e32 v2, 2, v11
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v6, 2, v6
	v_lshlrev_b32_e32 v9, 2, v9
	v_dual_mov_b32 v11, v21 :: v_dual_lshlrev_b32 v10, 2, v10
	s_wait_loadcnt 0x7
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v41, v33, s1, 0
	s_wait_loadcnt 0x6
	v_fma_mixhi_f16 v41, v34, s1, 0
	s_wait_loadcnt 0x5
	v_fma_mixlo_f16 v42, v35, s1, 0
	s_wait_loadcnt 0x4
	v_fma_mixhi_f16 v42, v36, s1, 0
	s_wait_loadcnt 0x3
	v_fma_mixlo_f16 v43, v37, s1, 0
	s_wait_loadcnt 0x2
	v_fma_mixhi_f16 v43, v38, s1, 0
	s_wait_loadcnt 0x1
	v_fma_mixlo_f16 v44, v7, s1, 0
	v_mov_b32_e32 v7, v16
	v_lshlrev_b32_e32 v3, 2, v12
	s_clause 0x7
	global_load_b32 v40, v1, s[6:7]
	global_load_b32 v46, v2, s[6:7]
	global_load_b32 v47, v3, s[6:7]
	global_load_b32 v48, v4, s[6:7]
	global_load_b32 v49, v5, s[6:7]
	global_load_b32 v50, v6, s[6:7]
	global_load_b32 v51, v9, s[6:7]
	global_load_b32 v52, v10, s[6:7]
	v_dual_mov_b32 v9, v19 :: v_dual_mov_b32 v10, v20
	v_dual_mov_b32 v12, v22 :: v_dual_mov_b32 v1, v20
	v_dual_mov_b32 v2, v21 :: v_dual_mov_b32 v3, v22
	v_dual_mov_b32 v4, v23 :: v_dual_mov_b32 v5, v24
	v_mov_b32_e32 v6, v32
	s_wait_loadcnt 0x8
	v_fma_mixhi_f16 v44, v39, s1, 0
	s_wait_loadcnt 0x7
	v_fma_mixlo_f16 v45, v40, s1, 0
	s_wait_loadcnt 0x6
	v_fma_mixhi_f16 v45, v46, s1, 0
	s_wait_loadcnt 0x5
	v_fma_mixlo_f16 v46, v47, s1, 0
	s_wait_loadcnt 0x4
	v_fma_mixhi_f16 v46, v48, s1, 0
	s_wait_loadcnt 0x3
	v_fma_mixlo_f16 v47, v49, s1, 0
	s_wait_loadcnt 0x2
	v_fma_mixhi_f16 v47, v50, s1, 0
	s_wait_loadcnt 0x1
	v_fma_mixlo_f16 v48, v51, s1, 0
	s_wait_loadcnt 0x0
	v_fma_mixhi_f16 v48, v52, s1, 0
	s_mov_b32 s1, 0x3727c5ac
.LBB2_2:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], 0
	v_cvt_f16_f32_e32 v52.h, v24
	v_cvt_f16_f32_e32 v52.l, v23
	v_cvt_f16_f32_e32 v51.h, v22
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_cvt_f16_f32_e32 v51.l, v21
	v_cvt_f16_f32_e32 v50.h, v20
	v_cvt_f16_f32_e32 v50.l, v19
	v_cvt_f16_f32_e32 v49.h, v18
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_cvt_f16_f32_e32 v49.l, v17
	v_cvt_f16_f32_e32 v56.h, v32
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v55.h, v30
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v54.h, v28
	v_cvt_f16_f32_e32 v54.l, v27
	v_cvt_f16_f32_e32 v53.h, v26
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_cvt_f16_f32_e32 v53.l, v25
	v_cvt_f16_f32_e32 v60.h, v16
	v_cvt_f16_f32_e32 v60.l, v15
	v_cvt_f16_f32_e32 v59.h, v14
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_cvt_f16_f32_e32 v59.l, v13
	v_cvt_f16_f32_e32 v58.h, v12
	v_cvt_f16_f32_e32 v58.l, v11
	v_cvt_f16_f32_e32 v57.h, v10
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_cvt_f16_f32_e32 v57.l, v9
	v_cvt_f16_f32_e32 v64.h, v8
	v_cvt_f16_f32_e32 v64.l, v7
	v_cvt_f16_f32_e32 v63.h, v6
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_cvt_f16_f32_e32 v63.l, v5
	v_cvt_f16_f32_e32 v62.h, v4
	v_cvt_f16_f32_e32 v62.l, v3
	v_cvt_f16_f32_e32 v61.h, v2
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_cvt_f16_f32_e32 v61.l, v1
	v_dual_mul_f32 v24, 0x3f7d70a4, v24 :: v_dual_mul_f32 v25, 0x3f7d70a4, v25
	v_dual_mul_f32 v22, 0x3f7d70a4, v22 :: v_dual_mul_f32 v15, 0x3f7d70a4, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_dual_mul_f32 v18, 0x3f7d70a4, v18 :: v_dual_mul_f32 v13, 0x3f7d70a4, v13
	v_dual_mul_f32 v30, 0x3f7d70a4, v30 :: v_dual_mul_f32 v11, 0x3f7d70a4, v11
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_dual_mul_f32 v28, 0x3f7d70a4, v28 :: v_dual_mul_f32 v9, 0x3f7d70a4, v9
	v_dual_mul_f32 v16, 0x3f7d70a4, v16 :: v_dual_mul_f32 v7, 0x3f7d70a4, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_dual_mul_f32 v14, 0x3f7d70a4, v14 :: v_dual_mul_f32 v5, 0x3f7d70a4, v5
	v_dual_mul_f32 v10, 0x3f7d70a4, v10 :: v_dual_mul_f32 v3, 0x3f7d70a4, v3
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_dual_mul_f32 v8, 0x3f7d70a4, v8 :: v_dual_mul_f32 v1, 0x3f7d70a4, v1
	s_add_co_i32 s0, s0, -1
	v_mul_f32_e32 v19, 0x3f7d70a4, v19
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	v_mul_f32_e32 v31, 0x3f7d70a4, v31
	v_mul_f32_e32 v21, 0x3f7d70a4, v21
	v_mul_f32_e32 v23, 0x3f7d70a4, v23
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_mul_f32_e32 v17, 0x3f7d70a4, v17
	v_mul_f32_e32 v29, 0x3f7d70a4, v29
	v_mul_f32_e32 v27, 0x3f7d70a4, v27
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[45:48], v[33:40]
	v_dual_add_f32 v77, 0x3727c5ac, v40 :: v_dual_add_f32 v76, 0x3727c5ac, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v75, 0x3727c5ac, v38 :: v_dual_add_f32 v74, 0x3727c5ac, v37
	v_dual_add_f32 v73, 0x3727c5ac, v36 :: v_dual_add_f32 v72, 0x3727c5ac, v35
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_add_f32 v71, 0x3727c5ac, v34 :: v_dual_add_f32 v70, 0x3727c5ac, v33
	v_dual_add_f32 v85, 0x37a7c5ac, v40 :: v_dual_add_f32 v84, 0x37a7c5ac, v39
	v_dual_add_f32 v83, 0x37a7c5ac, v38 :: v_dual_add_f32 v82, 0x37a7c5ac, v37
	v_wmma_f32_16x16x16_f16 v[70:77], v[41:44], v[49:52], v[70:77]
	v_dual_add_f32 v81, 0x37a7c5ac, v36 :: v_dual_add_f32 v80, 0x37a7c5ac, v35
	v_dual_add_f32 v79, 0x37a7c5ac, v34 :: v_dual_add_f32 v78, 0x37a7c5ac, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[70:77], v[41:44], v[53:56], v[70:77]
	v_dual_add_f32 v93, 0x37fba882, v40 :: v_dual_add_f32 v92, 0x37fba882, v39
	v_dual_add_f32 v91, 0x37fba882, v38 :: v_dual_add_f32 v90, 0x37fba882, v37
	v_wmma_f32_16x16x16_f16 v[70:77], v[41:44], v[57:60], v[70:77]
	v_dual_add_f32 v89, 0x37fba882, v36 :: v_dual_add_f32 v88, 0x37fba882, v35
	v_dual_add_f32 v87, 0x37fba882, v34 :: v_dual_add_f32 v86, 0x37fba882, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[70:77], v[41:44], v[61:64], v[70:77]
	v_wmma_f32_16x16x16_f16 v[78:85], v[41:44], v[49:52], v[78:85]
	v_dual_add_f32 v101, 0x3827c5ac, v40 :: v_dual_add_f32 v100, 0x3827c5ac, v39
	v_dual_add_f32 v99, 0x3827c5ac, v38 :: v_dual_add_f32 v98, 0x3827c5ac, v37
	v_fma_mixhi_f16 v68, v77, s1, 0
	v_fma_mixlo_f16 v68, v76, s1, 0
	v_fma_mixhi_f16 v67, v75, s1, 0
	v_fma_mixlo_f16 v67, v74, s1, 0
	v_fma_mixhi_f16 v66, v73, s1, 0
	v_fma_mixlo_f16 v66, v72, s1, 0
	v_fma_mixhi_f16 v65, v71, s1, 0
	v_fma_mixlo_f16 v65, v70, s1, 0
	v_wmma_f32_16x16x16_f16 v[70:77], v[45:48], v[49:52], v[33:40]
	v_dual_add_f32 v97, 0x3827c5ac, v36 :: v_dual_add_f32 v96, 0x3827c5ac, v35
	v_dual_add_f32 v95, 0x3827c5ac, v34 :: v_dual_add_f32 v94, 0x3827c5ac, v33
	v_wmma_f32_16x16x16_f16 v[86:93], v[41:44], v[49:52], v[86:93]
	v_wmma_f32_16x16x16_f16 v[78:85], v[41:44], v[53:56], v[78:85]
	v_wmma_f32_16x16x16_f16 v[70:77], v[45:48], v[53:56], v[70:77]
	v_dual_add_f32 v105, v40, v40 :: v_dual_add_f32 v104, v39, v39
	v_wmma_f32_16x16x16_f16 v[94:101], v[41:44], v[49:52], v[94:101]
	v_wmma_f32_16x16x16_f16 v[86:93], v[41:44], v[53:56], v[86:93]
	v_wmma_f32_16x16x16_f16 v[78:85], v[41:44], v[57:60], v[78:85]
	v_wmma_f32_16x16x16_f16 v[70:77], v[45:48], v[57:60], v[70:77]
	v_dual_add_f32 v103, v38, v38 :: v_dual_add_f32 v102, v37, v37
	v_wmma_f32_16x16x16_f16 v[94:101], v[41:44], v[53:56], v[94:101]
	v_wmma_f32_16x16x16_f16 v[86:93], v[41:44], v[57:60], v[86:93]
	v_wmma_f32_16x16x16_f16 v[78:85], v[41:44], v[61:64], v[78:85]
	v_wmma_f32_16x16x16_f16 v[70:77], v[45:48], v[61:64], v[70:77]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[94:101], v[41:44], v[57:60], v[94:101]
	v_wmma_f32_16x16x16_f16 v[86:93], v[41:44], v[61:64], v[86:93]
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_mixhi_f16 v109, v85, s1, 0
	v_fma_mixlo_f16 v109, v84, s1, 0
	v_fma_mixhi_f16 v108, v83, s1, 0
	v_fma_mixlo_f16 v108, v82, s1, 0
	v_fma_mixhi_f16 v107, v81, s1, 0
	v_fma_mixlo_f16 v107, v80, s1, 0
	v_fma_mixhi_f16 v106, v79, s1, 0
	v_fma_mixlo_f16 v106, v78, s1, 0
	v_wmma_f32_16x16x16_f16 v[70:77], v[41:44], v[65:68], v[70:77]
	v_wmma_f32_16x16x16_f16 v[94:101], v[41:44], v[61:64], v[94:101]
	v_fma_mixhi_f16 v93, v93, s1, 0
	v_fma_mixlo_f16 v93, v92, s1, 0
	v_fma_mixhi_f16 v92, v91, s1, 0
	v_fma_mixlo_f16 v92, v90, s1, 0
	v_fma_mixhi_f16 v91, v89, s1, 0
	v_fma_mixlo_f16 v91, v88, s1, 0
	v_fma_mixhi_f16 v90, v87, s1, 0
	v_fma_mixlo_f16 v90, v86, s1, 0
	v_wmma_f32_16x16x16_f16 v[70:77], v[41:44], v[106:109], v[70:77]
	v_fma_mixhi_f16 v89, v101, s1, 0
	v_fma_mixlo_f16 v89, v100, s1, 0
	v_fma_mixhi_f16 v88, v99, s1, 0
	v_fma_mixlo_f16 v88, v98, s1, 0
	v_fma_mixhi_f16 v87, v97, s1, 0
	v_fma_mixlo_f16 v87, v96, s1, 0
	v_fma_mixhi_f16 v86, v95, s1, 0
	v_fma_mixlo_f16 v86, v94, s1, 0
	v_wmma_f32_16x16x16_f16 v[70:77], v[41:44], v[90:93], v[70:77]
	v_mul_f32_e32 v85, 0x40400000, v40
	v_dual_mul_f32 v83, 0x40400000, v38 :: v_dual_mul_f32 v84, 0x40400000, v39
	v_mul_f32_e32 v82, 0x40400000, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[70:77], v[41:44], v[86:89], v[70:77]
	v_dual_mul_f32 v81, 0x40400000, v36 :: v_dual_mul_f32 v80, 0x40400000, v35
	v_dual_mul_f32 v79, 0x40400000, v34 :: v_dual_mul_f32 v78, 0x40400000, v33
	v_dual_add_f32 v70, 0, v70 :: v_dual_add_f32 v101, v36, v36
	v_dual_add_f32 v100, v35, v35 :: v_dual_add_f32 v99, v34, v34
	v_add_f32_e32 v98, v33, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_add_f32 v70, v70, v71 :: v_dual_mul_f32 v37, 4.0, v37
	v_mul_f32_e32 v35, 4.0, v35
	v_wmma_f32_16x16x16_f16 v[78:85], v[45:48], v[49:52], v[78:85]
	v_wmma_f32_16x16x16_f16 v[98:105], v[45:48], v[49:52], v[98:105]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_f32 v70, v70, v72 :: v_dual_mul_f32 v39, 4.0, v39
	v_mul_f32_e32 v33, 4.0, v33
	v_wmma_f32_16x16x16_f16 v[78:85], v[45:48], v[53:56], v[78:85]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[98:105], v[45:48], v[53:56], v[98:105]
	v_add_f32_e32 v70, v70, v73
	v_mul_f32_e32 v40, 4.0, v40
	v_mul_f32_e32 v36, 4.0, v36
	v_wmma_f32_16x16x16_f16 v[78:85], v[45:48], v[57:60], v[78:85]
	v_wmma_f32_16x16x16_f16 v[98:105], v[45:48], v[57:60], v[98:105]
	v_add_f32_e32 v70, v70, v74
	v_mul_f32_e32 v34, 4.0, v34
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[78:85], v[45:48], v[61:64], v[78:85]
	v_wmma_f32_16x16x16_f16 v[98:105], v[45:48], v[61:64], v[98:105]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v70, v70, v75
	v_wmma_f32_16x16x16_f16 v[78:85], v[41:44], v[65:68], v[78:85]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[98:105], v[41:44], v[65:68], v[98:105]
	v_wmma_f32_16x16x16_f16 v[78:85], v[41:44], v[106:109], v[78:85]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[98:105], v[41:44], v[106:109], v[98:105]
	v_wmma_f32_16x16x16_f16 v[78:85], v[41:44], v[90:93], v[78:85]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[98:105], v[41:44], v[90:93], v[98:105]
	v_wmma_f32_16x16x16_f16 v[78:85], v[41:44], v[86:89], v[78:85]
	v_add_f32_e32 v70, v70, v76
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[98:105], v[41:44], v[86:89], v[98:105]
	v_mul_f32_e32 v38, 4.0, v38
	v_add_f32_e32 v70, v70, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[33:40], v[45:48], v[49:52], v[33:40]
	v_add_f32_e32 v70, v70, v98
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[33:40], v[45:48], v[53:56], v[33:40]
	v_add_f32_e32 v70, v70, v99
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[33:40], v[45:48], v[57:60], v[33:40]
	v_add_f32_e32 v70, v70, v100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[33:40], v[45:48], v[61:64], v[33:40]
	v_add_f32_e32 v70, v70, v101
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[65:68], v[33:40]
	v_add_f32_e32 v70, v70, v102
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[106:109], v[33:40]
	v_add_f32_e32 v70, v70, v103
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[90:93], v[33:40]
	v_add_f32_e32 v70, v70, v104
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[86:89], v[33:40]
	v_add_f32_e32 v70, v70, v105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v70, v78
	v_add_f32_e32 v49, v49, v79
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v80
	v_add_f32_e32 v49, v49, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v82
	v_add_f32_e32 v49, v49, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v49, v49, v84
	v_add_f32_e32 v49, v49, v85
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v33, v49, v33 :: v_dual_mul_f32 v20, 0x3f7d70a4, v20
	v_dual_add_f32 v33, v33, v34 :: v_dual_mul_f32 v32, 0x3f7d70a4, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[65:68], v[17:24]
	v_dual_add_f32 v33, v33, v35 :: v_dual_mul_f32 v26, 0x3f7d70a4, v26
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[106:109], v[17:24]
	v_add_f32_e32 v33, v33, v36
	v_mul_f32_e32 v12, 0x3f7d70a4, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[41:44], v[65:68], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[90:93], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v33, v33, v37 :: v_dual_mul_f32 v4, 0x3f7d70a4, v4
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[65:68], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[41:44], v[106:109], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[86:89], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_f32_e32 v33, v33, v38
	v_mul_f32_e32 v6, 0x3f7d70a4, v6
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[106:109], v[9:16]
	v_wmma_f32_16x16x16_f16 v[25:32], v[41:44], v[90:93], v[25:32]
	v_dual_add_f32 v33, v33, v39 :: v_dual_mul_f32 v2, 0x3f7d70a4, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[90:93], v[9:16]
	v_wmma_f32_16x16x16_f16 v[25:32], v[41:44], v[86:89], v[25:32]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v33, v33, v40
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[65:68], v[1:8]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[86:89], v[9:16]
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[106:109], v[1:8]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v33, 0x3a83126f, v69, vcc_lo
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[90:93], v[1:8]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v45.l, v33
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[86:89], v[1:8]
	s_cbranch_scc1 .LBB2_2
	s_branch .LBB2_4
.LBB2_3:
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v7, v16 :: v_dual_mov_b32 v6, v32
	v_dual_mov_b32 v5, v24 :: v_dual_mov_b32 v4, v23
	v_dual_mov_b32 v3, v22 :: v_dual_mov_b32 v2, v21
	v_dual_mov_b32 v1, v20 :: v_dual_mov_b32 v14, v24
	v_dual_mov_b32 v15, v32 :: v_dual_mov_b32 v12, v22
	v_dual_mov_b32 v13, v23 :: v_dual_mov_b32 v10, v20
	v_dual_mov_b32 v11, v21 :: v_dual_mov_b32 v30, v23
	v_dual_mov_b32 v9, v19 :: v_dual_mov_b32 v28, v21
	v_dual_mov_b32 v31, v24 :: v_dual_mov_b32 v26, v19
	v_mov_b32_e32 v29, v22
	v_mov_b32_e32 v27, v20
	v_mov_b32_e32 v25, v18
.LBB2_4:
	v_add_f32_e32 v17, 0, v17
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v17, v17, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v26
	v_add_f32_e32 v17, v17, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v28
	v_add_f32_e32 v17, v17, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v30
	v_add_f32_e32 v17, v17, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v32
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v1, v6 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_f32_e32 v2, v2, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end2:
	.size	_Z5probeILi4ELb0ELb0EEvPfPKfi, .Lfunc_end2-_Z5probeILi4ELb0ELb0EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi4ELb0ELb0EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 110
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_Z5probeILi4ELb0ELb0EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi4ELb0ELb0EEvPfPKfi,"axG",@progbits,_Z5probeILi4ELb0ELb0EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.num_vgpr, 110
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.numbered_sgpr, 8
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi4ELb0ELb0EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3304
; TotalNumSgprs: 10
; NumVgprs: 110
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 110
; Occupancy: 12
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi4ELb1ELb0EEvPfPKfi,"axG",@progbits,_Z5probeILi4ELb1ELb0EEvPfPKfi,comdat
	.protected	_Z5probeILi4ELb1ELb0EEvPfPKfi ; -- Begin function _Z5probeILi4ELb1ELb0EEvPfPKfi
	.globl	_Z5probeILi4ELb1ELb0EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi4ELb1ELb0EEvPfPKfi,@function
_Z5probeILi4ELb1ELb0EEvPfPKfi:          ; @_Z5probeILi4ELb1ELb0EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	s_load_b32 s0, s[0:1], 0x10
	s_mov_b32 s1, 0x3c23d70a
	s_mov_b32 s2, 0x38d1b717
	s_mov_b32 s3, 0x3951b717
	s_mov_b32 s8, 0x399d4951
	s_mov_b32 s9, 0x39d1b717
	s_mov_b32 s10, 0x3a03126e
	s_mov_b32 s11, 0x3a1d4951
	s_mov_b32 s12, 0x3a378034
	s_mov_b32 s13, 0x3a51b717
	s_mov_b32 s14, 0x3a6bedfa
	s_mov_b32 s15, 0x3a83126e
	s_wait_kmcnt 0x0
	global_load_b32 v2, v1, s[6:7]
	s_cmp_lt_i32 s0, 1
	s_wait_loadcnt 0x0
	v_fma_mixhi_f16 v21, v2, s1, s2
	v_fma_mixhi_f16 v17, v2, s1, s3
	v_fma_mixlo_f16 v18, v2, s1, s8
	v_fma_mixhi_f16 v18, v2, s1, s9
	v_fma_mixlo_f16 v19, v2, s1, s10
	v_fma_mixhi_f16 v19, v2, s1, s11
	v_fma_mixlo_f16 v20, v2, s1, s12
	v_fma_mixhi_f16 v20, v2, s1, s13
	v_fma_mixhi_f16 v16, v2, s1, s14
	v_fma_mixlo_f16 v21, v2, s1, 0
	v_fma_mixhi_f16 v12, v2, s1, s15
	v_mov_b16_e32 v17.l, v21.h
	v_mov_b16_e32 v22.l, v17.h
	v_mov_b16_e32 v22.h, v18.l
	v_mov_b16_e32 v23.l, v18.h
	v_mov_b16_e32 v23.h, v19.l
	v_mov_b16_e32 v24.l, v19.h
	v_mov_b16_e32 v24.h, v20.l
	v_mov_b16_e32 v16.l, v20.h
	v_mov_b16_e32 v12.l, v16.h
	s_cbranch_scc1 .LBB3_3
; %bb.1:
	v_add_nc_u32_e32 v9, 9, v0
	v_add_nc_u32_e32 v10, 10, v0
	v_add_nc_u32_e32 v11, 11, v0
	v_add_nc_u32_e32 v13, 12, v0
	v_add_nc_u32_e32 v14, 13, v0
	v_and_b32_e32 v9, 0xff, v9
	v_add_nc_u32_e32 v15, 14, v0
	v_add_nc_u32_e32 v25, 15, v0
	v_add_nc_u32_e32 v26, 16, v0
	v_and_b32_e32 v10, 0xff, v10
	v_and_b32_e32 v11, 0xff, v11
	v_and_b32_e32 v13, 0xff, v13
	v_and_b32_e32 v14, 0xff, v14
	v_lshlrev_b32_e32 v9, 2, v9
	v_and_b32_e32 v15, 0xff, v15
	v_and_b32_e32 v25, 0xff, v25
	v_and_b32_e32 v26, 0xff, v26
	v_lshlrev_b32_e32 v10, 2, v10
	v_lshlrev_b32_e32 v11, 2, v11
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v15, 2, v15
	v_lshlrev_b32_e32 v25, 2, v25
	v_lshlrev_b32_e32 v26, 2, v26
	s_clause 0x7
	global_load_b32 v29, v9, s[6:7]
	global_load_b32 v30, v10, s[6:7]
	global_load_b32 v11, v11, s[6:7]
	global_load_b32 v31, v13, s[6:7]
	global_load_b32 v32, v14, s[6:7]
	global_load_b32 v34, v15, s[6:7]
	global_load_b32 v35, v25, s[6:7]
	global_load_b32 v36, v26, s[6:7]
	v_dual_mov_b32 v33, 0x3b03126f :: v_dual_add_nc_u32 v2, 1, v0
	v_dual_mov_b32 v14, v23 :: v_dual_add_nc_u32 v3, 2, v0
	v_dual_mov_b32 v13, v22 :: v_dual_add_nc_u32 v4, 3, v0
	v_dual_mov_b32 v10, v19 :: v_dual_add_nc_u32 v5, 4, v0
	v_dual_mov_b32 v9, v18 :: v_dual_add_nc_u32 v6, 5, v0
	v_add_nc_u32_e32 v7, 6, v0
	v_add_nc_u32_e32 v8, 7, v0
	v_dual_mov_b32 v15, v24 :: v_dual_and_b32 v2, 0xff, v2
	v_and_b32_e32 v3, 0xff, v3
	v_and_b32_e32 v4, 0xff, v4
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v6, 0xff, v6
	v_and_b32_e32 v7, 0xff, v7
	v_and_b32_e32 v8, 0xff, v8
	v_lshlrev_b32_e32 v2, 2, v2
	s_mov_b32 s1, 0x3a83126f
	v_lshlrev_b32_e32 v3, 2, v3
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v6, 2, v6
	v_lshlrev_b32_e32 v7, 2, v7
	v_lshlrev_b32_e32 v8, 2, v8
	s_mov_b32 s2, 0x3f7d70a4
	s_wait_loadcnt 0x7
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v29, v29, s1, 0
	s_wait_loadcnt 0x6
	v_fma_mixhi_f16 v29, v30, s1, 0
	s_wait_loadcnt 0x5
	v_fma_mixlo_f16 v30, v11, s1, 0
	v_dual_mov_b32 v11, v20 :: v_dual_lshlrev_b32 v4, 2, v4
	s_clause 0x7
	global_load_b32 v1, v1, s[6:7]
	global_load_b32 v2, v2, s[6:7]
	global_load_b32 v3, v3, s[6:7]
	global_load_b32 v4, v4, s[6:7]
	global_load_b32 v5, v5, s[6:7]
	global_load_b32 v6, v6, s[6:7]
	global_load_b32 v7, v7, s[6:7]
	global_load_b32 v8, v8, s[6:7]
	s_wait_loadcnt 0xc
	v_fma_mixhi_f16 v30, v31, s1, 0
	s_wait_loadcnt 0xb
	v_fma_mixlo_f16 v31, v32, s1, 0
	s_wait_loadcnt 0xa
	v_fma_mixhi_f16 v31, v34, s1, 0
	s_wait_loadcnt 0x9
	v_fma_mixlo_f16 v32, v35, s1, 0
	s_wait_loadcnt 0x8
	v_fma_mixhi_f16 v32, v36, s1, 0
	s_wait_loadcnt 0x7
	v_fma_mixlo_f16 v25, v1, s1, 0
	s_wait_loadcnt 0x6
	v_fma_mixhi_f16 v25, v2, s1, 0
	s_wait_loadcnt 0x5
	v_fma_mixlo_f16 v26, v3, s1, 0
	s_wait_loadcnt 0x4
	v_fma_mixhi_f16 v26, v4, s1, 0
	s_wait_loadcnt 0x3
	v_fma_mixlo_f16 v27, v5, s1, 0
	s_wait_loadcnt 0x2
	v_fma_mixhi_f16 v27, v6, s1, 0
	s_wait_loadcnt 0x1
	v_fma_mixlo_f16 v28, v7, s1, 0
	s_wait_loadcnt 0x0
	v_fma_mixhi_f16 v28, v8, s1, 0
	s_mov_b32 s1, 0x3727c5ac
.LBB3_2:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], 0
	s_add_co_i32 s0, s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s0, 0
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[29:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v41, 0x3727c5ac, v8 :: v_dual_add_f32 v40, 0x3727c5ac, v7
	v_dual_add_f32 v39, 0x3727c5ac, v6 :: v_dual_add_f32 v38, 0x3727c5ac, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v37, 0x3727c5ac, v4 :: v_dual_add_f32 v36, 0x3727c5ac, v3
	v_dual_add_f32 v35, 0x3727c5ac, v2 :: v_dual_add_f32 v34, 0x3727c5ac, v1
	v_dual_add_f32 v49, 0x37a7c5ac, v8 :: v_dual_add_f32 v48, 0x37a7c5ac, v7
	v_dual_add_f32 v47, 0x37a7c5ac, v6 :: v_dual_add_f32 v46, 0x37a7c5ac, v5
	v_dual_add_f32 v45, 0x37a7c5ac, v4 :: v_dual_add_f32 v44, 0x37a7c5ac, v3
	v_dual_add_f32 v43, 0x37a7c5ac, v2 :: v_dual_add_f32 v42, 0x37a7c5ac, v1
	v_wmma_f32_16x16x16_f16 v[34:41], v[25:28], v[21:24], v[34:41]
	v_dual_add_f32 v57, 0x37fba882, v8 :: v_dual_add_f32 v56, 0x37fba882, v7
	v_dual_add_f32 v55, 0x37fba882, v6 :: v_dual_add_f32 v54, 0x37fba882, v5
	v_dual_add_f32 v53, 0x37fba882, v4 :: v_dual_add_f32 v52, 0x37fba882, v3
	v_dual_add_f32 v51, 0x37fba882, v2 :: v_dual_add_f32 v50, 0x37fba882, v1
	v_wmma_f32_16x16x16_f16 v[66:73], v[29:32], v[21:24], v[1:8]
	v_wmma_f32_16x16x16_f16 v[42:49], v[25:28], v[21:24], v[42:49]
	v_wmma_f32_16x16x16_f16 v[34:41], v[25:28], v[17:20], v[34:41]
	v_dual_add_f32 v65, 0x3827c5ac, v8 :: v_dual_add_f32 v64, 0x3827c5ac, v7
	v_dual_add_f32 v63, 0x3827c5ac, v6 :: v_dual_add_f32 v62, 0x3827c5ac, v5
	v_dual_add_f32 v61, 0x3827c5ac, v4 :: v_dual_add_f32 v60, 0x3827c5ac, v3
	v_dual_add_f32 v59, 0x3827c5ac, v2 :: v_dual_add_f32 v58, 0x3827c5ac, v1
	v_wmma_f32_16x16x16_f16 v[50:57], v[25:28], v[21:24], v[50:57]
	v_wmma_f32_16x16x16_f16 v[66:73], v[29:32], v[17:20], v[66:73]
	v_wmma_f32_16x16x16_f16 v[42:49], v[25:28], v[17:20], v[42:49]
	v_wmma_f32_16x16x16_f16 v[34:41], v[25:28], v[13:16], v[34:41]
	v_wmma_f32_16x16x16_f16 v[58:65], v[25:28], v[21:24], v[58:65]
	v_wmma_f32_16x16x16_f16 v[50:57], v[25:28], v[17:20], v[50:57]
	v_wmma_f32_16x16x16_f16 v[66:73], v[29:32], v[13:16], v[66:73]
	v_wmma_f32_16x16x16_f16 v[42:49], v[25:28], v[13:16], v[42:49]
	v_wmma_f32_16x16x16_f16 v[34:41], v[25:28], v[9:12], v[34:41]
	v_wmma_f32_16x16x16_f16 v[58:65], v[25:28], v[17:20], v[58:65]
	v_wmma_f32_16x16x16_f16 v[50:57], v[25:28], v[13:16], v[50:57]
	v_wmma_f32_16x16x16_f16 v[66:73], v[29:32], v[9:12], v[66:73]
	v_wmma_f32_16x16x16_f16 v[42:49], v[25:28], v[9:12], v[42:49]
	v_fma_mixhi_f16 v85, v41, s1, 0
	v_fma_mixlo_f16 v85, v40, s1, 0
	v_fma_mixhi_f16 v84, v39, s1, 0
	v_fma_mixlo_f16 v84, v38, s1, 0
	v_fma_mixhi_f16 v83, v37, s1, 0
	v_fma_mixlo_f16 v83, v36, s1, 0
	v_fma_mixhi_f16 v82, v35, s1, 0
	v_fma_mixlo_f16 v82, v34, s1, 0
	v_wmma_f32_16x16x16_f16 v[58:65], v[25:28], v[13:16], v[58:65]
	v_wmma_f32_16x16x16_f16 v[50:57], v[25:28], v[9:12], v[50:57]
	v_fma_mixhi_f16 v89, v49, s1, 0
	v_fma_mixlo_f16 v89, v48, s1, 0
	v_fma_mixhi_f16 v88, v47, s1, 0
	v_fma_mixlo_f16 v88, v46, s1, 0
	v_fma_mixhi_f16 v87, v45, s1, 0
	v_fma_mixlo_f16 v87, v44, s1, 0
	v_fma_mixhi_f16 v86, v43, s1, 0
	v_fma_mixlo_f16 v86, v42, s1, 0
	v_wmma_f32_16x16x16_f16 v[66:73], v[25:28], v[82:85], v[66:73]
	v_wmma_f32_16x16x16_f16 v[58:65], v[25:28], v[9:12], v[58:65]
	v_fma_mixhi_f16 v93, v57, s1, 0
	v_fma_mixlo_f16 v93, v56, s1, 0
	v_fma_mixhi_f16 v92, v55, s1, 0
	v_fma_mixlo_f16 v92, v54, s1, 0
	v_fma_mixhi_f16 v91, v53, s1, 0
	v_fma_mixlo_f16 v91, v52, s1, 0
	v_fma_mixhi_f16 v90, v51, s1, 0
	v_fma_mixlo_f16 v90, v50, s1, 0
	v_wmma_f32_16x16x16_f16 v[66:73], v[25:28], v[86:89], v[66:73]
	v_fma_mixhi_f16 v97, v65, s1, 0
	v_fma_mixlo_f16 v97, v64, s1, 0
	v_fma_mixhi_f16 v96, v63, s1, 0
	v_fma_mixlo_f16 v96, v62, s1, 0
	v_fma_mixhi_f16 v95, v61, s1, 0
	v_fma_mixlo_f16 v95, v60, s1, 0
	v_fma_mixhi_f16 v94, v59, s1, 0
	v_fma_mixlo_f16 v94, v58, s1, 0
	v_wmma_f32_16x16x16_f16 v[66:73], v[25:28], v[90:93], v[66:73]
	v_dual_add_f32 v81, v8, v8 :: v_dual_add_f32 v80, v7, v7
	v_dual_add_f32 v79, v6, v6 :: v_dual_add_f32 v78, v5, v5
	v_dual_add_f32 v77, v4, v4 :: v_dual_add_f32 v76, v3, v3
	v_dual_add_f32 v75, v2, v2 :: v_dual_add_f32 v74, v1, v1
	v_wmma_f32_16x16x16_f16 v[66:73], v[25:28], v[94:97], v[66:73]
	v_mul_f32_e32 v41, 0x40400000, v8
	v_mul_f32_e32 v39, 0x40400000, v6
	v_mul_f32_e32 v35, 0x40400000, v2
	v_wmma_f32_16x16x16_f16 v[74:81], v[29:32], v[21:24], v[74:81]
	v_dual_add_f32 v34, 0, v66 :: v_dual_mul_f32 v37, 0x40400000, v4
	v_dual_mul_f32 v40, 0x40400000, v7 :: v_dual_mul_f32 v7, 4.0, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[74:81], v[29:32], v[17:20], v[74:81]
	v_add_f32_e32 v34, v34, v67
	v_dual_mul_f32 v38, 0x40400000, v5 :: v_dual_mul_f32 v5, 4.0, v5
	v_mul_f32_e32 v6, 4.0, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[74:81], v[29:32], v[13:16], v[74:81]
	v_add_f32_e32 v34, v34, v68
	v_mul_f32_e32 v4, 4.0, v4
	v_mul_f32_e32 v2, 4.0, v2
	v_fma_mix_f32 v43, v21, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[74:81], v[29:32], v[9:12], v[74:81]
	v_add_f32_e32 v34, v34, v69
	v_fma_mix_f32 v51, v17, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v59, v13, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v69, v10, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[74:81], v[25:28], v[82:85], v[74:81]
	v_add_f32_e32 v34, v34, v70
	v_fma_mix_f32 v70, v11, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v68, v10, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v67, v9, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[74:81], v[25:28], v[86:89], v[74:81]
	v_add_f32_e32 v34, v34, v71
	v_fma_mix_f32 v71, v11, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v66, v9, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v49, v24, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[74:81], v[25:28], v[90:93], v[74:81]
	v_add_f32_e32 v42, v34, v72
	v_dual_mul_f32 v34, 0x40400000, v1 :: v_dual_mul_f32 v1, 4.0, v1
	v_fma_mix_f32 v48, v24, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[74:81], v[25:28], v[94:97], v[74:81]
	v_add_f32_e32 v42, v42, v73
	v_fma_mix_f32 v47, v23, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v46, v23, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v45, v22, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v44, v22, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v42, v42, v74
	v_fma_mix_f32 v57, v20, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v56, v20, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v55, v19, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v54, v19, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v50, v42, v75
	v_fma_mix_f32 v42, v21, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v53, v18, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v52, v18, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v65, v16, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v50, v50, v76
	v_fma_mix_f32 v64, v16, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v63, v15, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v62, v15, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v61, v14, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v50, v50, v77
	v_fma_mix_f32 v60, v14, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v73, v12, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v12, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[42:49], v[25:28], v[82:85], v[42:49]
	v_add_f32_e32 v58, v50, v78
	v_fma_mix_f32 v50, v17, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[66:73], v[25:28], v[82:85], v[66:73]
	v_wmma_f32_16x16x16_f16 v[42:49], v[25:28], v[86:89], v[42:49]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v58, v58, v79
	v_wmma_f32_16x16x16_f16 v[50:57], v[25:28], v[82:85], v[50:57]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[66:73], v[25:28], v[86:89], v[66:73]
	v_wmma_f32_16x16x16_f16 v[42:49], v[25:28], v[90:93], v[42:49]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v58, v58, v80
	v_wmma_f32_16x16x16_f16 v[50:57], v[25:28], v[86:89], v[50:57]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[66:73], v[25:28], v[90:93], v[66:73]
	v_wmma_f32_16x16x16_f16 v[42:49], v[25:28], v[94:97], v[42:49]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v58, v58, v81
	v_dual_mul_f32 v36, 0x40400000, v3 :: v_dual_mul_f32 v3, 4.0, v3
	v_wmma_f32_16x16x16_f16 v[50:57], v[25:28], v[90:93], v[50:57]
	v_wmma_f32_16x16x16_f16 v[66:73], v[25:28], v[94:97], v[66:73]
	v_wmma_f32_16x16x16_f16 v[34:41], v[29:32], v[21:24], v[34:41]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[50:57], v[25:28], v[94:97], v[50:57]
	v_wmma_f32_16x16x16_f16 v[34:41], v[29:32], v[17:20], v[34:41]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[34:41], v[29:32], v[13:16], v[34:41]
	v_wmma_f32_16x16x16_f16 v[34:41], v[29:32], v[9:12], v[34:41]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[34:41], v[25:28], v[82:85], v[34:41]
	v_wmma_f32_16x16x16_f16 v[34:41], v[25:28], v[86:89], v[34:41]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[34:41], v[25:28], v[90:93], v[34:41]
	v_wmma_f32_16x16x16_f16 v[34:41], v[25:28], v[94:97], v[34:41]
	v_mul_f32_e32 v8, 4.0, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[1:8], v[29:32], v[21:24], v[1:8]
	v_add_f32_e32 v21, v58, v34
	v_fma_mix_f32 v58, v13, s2, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f16_f32_e32 v22.h, v45
	v_cvt_f16_f32_e32 v22.l, v44
	v_wmma_f32_16x16x16_f16 v[1:8], v[29:32], v[17:20], v[1:8]
	v_add_f32_e32 v17, v21, v35
	v_wmma_f32_16x16x16_f16 v[58:65], v[25:28], v[82:85], v[58:65]
	v_cvt_f16_f32_e32 v21.h, v43
	v_cvt_f16_f32_e32 v21.l, v42
	v_wmma_f32_16x16x16_f16 v[1:8], v[29:32], v[13:16], v[1:8]
	v_add_f32_e32 v13, v17, v36
	v_wmma_f32_16x16x16_f16 v[58:65], v[25:28], v[86:89], v[58:65]
	v_cvt_f16_f32_e32 v23.h, v47
	v_cvt_f16_f32_e32 v23.l, v46
	v_wmma_f32_16x16x16_f16 v[1:8], v[29:32], v[9:12], v[1:8]
	v_add_f32_e32 v11, v13, v37
	v_wmma_f32_16x16x16_f16 v[58:65], v[25:28], v[90:93], v[58:65]
	v_cvt_f16_f32_e32 v24.h, v49
	v_cvt_f16_f32_e32 v24.l, v48
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[82:85], v[1:8]
	v_add_f32_e32 v10, v11, v38
	v_wmma_f32_16x16x16_f16 v[58:65], v[25:28], v[94:97], v[58:65]
	v_cvt_f16_f32_e32 v17.h, v51
	v_cvt_f16_f32_e32 v17.l, v50
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[86:89], v[1:8]
	v_add_f32_e32 v9, v10, v39
	v_cvt_f16_f32_e32 v18.h, v53
	v_cvt_f16_f32_e32 v18.l, v52
	v_cvt_f16_f32_e32 v19.h, v55
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[90:93], v[1:8]
	v_add_f32_e32 v9, v9, v40
	v_cvt_f16_f32_e32 v19.l, v54
	v_cvt_f16_f32_e32 v20.h, v57
	v_cvt_f16_f32_e32 v20.l, v56
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:28], v[94:97], v[1:8]
	v_add_f32_e32 v9, v9, v41
	v_cvt_f16_f32_e32 v13.h, v59
	v_cvt_f16_f32_e32 v13.l, v58
	v_cvt_f16_f32_e32 v14.h, v61
	v_cvt_f16_f32_e32 v14.l, v60
	v_add_f32_e32 v1, v9, v1
	v_cvt_f16_f32_e32 v15.h, v63
	v_cvt_f16_f32_e32 v15.l, v62
	v_cvt_f16_f32_e32 v16.h, v65
	v_cvt_f16_f32_e32 v16.l, v64
	v_add_f32_e32 v1, v1, v2
	v_cvt_f16_f32_e32 v10.h, v69
	v_cvt_f16_f32_e32 v10.l, v68
	v_cvt_f16_f32_e32 v11.h, v71
	v_cvt_f16_f32_e32 v11.l, v70
	v_add_f32_e32 v1, v1, v3
	v_cvt_f16_f32_e32 v12.h, v73
	v_cvt_f16_f32_e32 v12.l, v72
	v_cvt_f16_f32_e32 v9.h, v67
	v_cvt_f16_f32_e32 v9.l, v66
	v_add_f32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v1, v1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x3a83126f, v33, vcc_lo
	v_cvt_f16_f32_e32 v29.l, v1
	s_cbranch_scc1 .LBB3_2
	s_branch .LBB3_4
.LBB3_3:
	v_dual_mov_b32 v11, v20 :: v_dual_mov_b32 v10, v19
	v_dual_mov_b32 v9, v18 :: v_dual_mov_b32 v14, v23
	v_mov_b32_e32 v15, v24
	v_mov_b32_e32 v13, v22
.LBB3_4:
	v_fma_mix_f32 v1, v21, 1.0, 0 op_sel_hi:[1,1,0]
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v21, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v22, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v22, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v23, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v23, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v24, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v24, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v17, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v17, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v18, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v18, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v19, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v19, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v20, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v20, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v13, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v13, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v14, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v14, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v15, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v15, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v16, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v16, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v9, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v9, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v10, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v10, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v11, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v2, v11, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mov_b32_e32 v1, 0
	v_fma_mix_f32 v2, v12, 1.0, v2 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fma_mix_f32 v2, v12, 1.0, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end3:
	.size	_Z5probeILi4ELb1ELb0EEvPfPKfi, .Lfunc_end3-_Z5probeILi4ELb1ELb0EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi4ELb1ELb0EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 98
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_Z5probeILi4ELb1ELb0EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi4ELb1ELb0EEvPfPKfi,"axG",@progbits,_Z5probeILi4ELb1ELb0EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.num_vgpr, 98
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.numbered_sgpr, 16
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi4ELb1ELb0EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3400
; TotalNumSgprs: 18
; NumVgprs: 98
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 98
; Occupancy: 12
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi8ELb0ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi8ELb0ELb1EEvPfPKfi,comdat
	.protected	_Z5probeILi8ELb0ELb1EEvPfPKfi ; -- Begin function _Z5probeILi8ELb0ELb1EEvPfPKfi
	.globl	_Z5probeILi8ELb0ELb1EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi8ELb0ELb1EEvPfPKfi,@function
_Z5probeILi8ELb0ELb1EEvPfPKfi:          ; @_Z5probeILi8ELb0ELb1EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_add_nc_u32_e32 v2, 9, v0
	v_add_nc_u32_e32 v3, 1, v0
	v_add_nc_u32_e32 v4, 10, v0
	v_add_nc_u32_e32 v5, 2, v0
	v_add_nc_u32_e32 v6, 11, v0
	v_add_nc_u32_e32 v7, 3, v0
	v_add_nc_u32_e32 v8, 12, v0
	v_add_nc_u32_e32 v9, 4, v0
	v_and_b32_e32 v2, 0xff, v2
	v_and_b32_e32 v3, 0xff, v3
	v_and_b32_e32 v4, 0xff, v4
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v6, 0xff, v6
	v_and_b32_e32 v7, 0xff, v7
	v_and_b32_e32 v8, 0xff, v8
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v12, 0xff, v9
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v3, 2, v3
	v_lshlrev_b32_e32 v4, 2, v4
	v_add_nc_u32_e32 v10, 13, v0
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v6, 2, v6
	v_add_nc_u32_e32 v11, 5, v0
	v_lshlrev_b32_e32 v7, 2, v7
	v_lshlrev_b32_e32 v8, 2, v8
	s_wait_kmcnt 0x0
	s_clause 0x7
	global_load_b32 v9, v1, s[6:7]
	global_load_b32 v1, v2, s[6:7]
	global_load_b32 v2, v3, s[6:7]
	global_load_b32 v3, v4, s[6:7]
	global_load_b32 v4, v5, s[6:7]
	global_load_b32 v5, v6, s[6:7]
	global_load_b32 v6, v7, s[6:7]
	global_load_b32 v7, v8, s[6:7]
	v_lshlrev_b32_e32 v8, 2, v12
	v_add_nc_u32_e32 v12, 14, v0
	v_add_nc_u32_e32 v13, 6, v0
	v_add_nc_u32_e32 v14, 15, v0
	v_add_nc_u32_e32 v15, 7, v0
	v_add_nc_u32_e32 v16, 16, v0
	v_and_b32_e32 v10, 0xff, v10
	v_and_b32_e32 v11, 0xff, v11
	v_and_b32_e32 v12, 0xff, v12
	v_and_b32_e32 v13, 0xff, v13
	v_and_b32_e32 v14, 0xff, v14
	v_and_b32_e32 v15, 0xff, v15
	v_and_b32_e32 v16, 0xff, v16
	v_lshlrev_b32_e32 v10, 2, v10
	v_lshlrev_b32_e32 v11, 2, v11
	v_lshlrev_b32_e32 v12, 2, v12
	v_lshlrev_b32_e32 v19, 2, v13
	v_lshlrev_b32_e32 v20, 2, v14
	v_lshlrev_b32_e32 v21, 2, v15
	v_lshlrev_b32_e32 v16, 2, v16
	s_clause 0x7
	global_load_b32 v17, v8, s[6:7]
	global_load_b32 v18, v10, s[6:7]
	global_load_b32 v13, v11, s[6:7]
	global_load_b32 v14, v12, s[6:7]
	global_load_b32 v15, v19, s[6:7]
	global_load_b32 v11, v20, s[6:7]
	global_load_b32 v12, v21, s[6:7]
	global_load_b32 v10, v16, s[6:7]
	v_and_b32_e32 v8, 63, v0
	s_mov_b32 s3, 0
	s_wait_loadcnt 0xf
	v_mul_f32_e32 v16, 0x38d1b717, v9
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB4_4
; %bb.1:
	v_mov_b32_e32 v19, 0
	v_lshl_add_u32 v20, v0, 2, 0
	v_lshrrev_b32_e32 v21, 6, v0
	s_mov_b32 s6, 0
.LBB4_2:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cmp_gt_u32_e32 vcc_lo, v8, v21
	v_add_nc_u32_e32 v19, -1, v19
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v22, 0, v16 :: v_dual_add_nc_u32 v21, 4, v21
	v_cmp_eq_u32_e32 vcc_lo, 0, v19
	ds_store_b32 v20, v22
	v_add_nc_u32_e32 v20, 0x400, v20
	s_or_b32 s6, vcc_lo, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB4_2
; %bb.3:
	s_or_b32 exec_lo, exec_lo, s6
.LBB4_4:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s2
	v_mov_b32_e32 v19, v0
.LBB4_5:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshrrev_b32_e32 v20, 6, v19
	v_lshl_add_u32 v21, v19, 2, 0
	v_add_nc_u32_e32 v22, 0x800, v19
	v_cmp_lt_u32_e32 vcc_lo, 0x7ff, v19
	v_add_nc_u32_e32 v19, 4, v20
	v_add_nc_u32_e32 v25, 12, v20
	v_add_nc_u32_e32 v26, 16, v20
	s_or_b32 s3, vcc_lo, s3
	v_add_nc_u32_e32 v27, 20, v20
	v_cmp_gt_u32_e32 vcc_lo, v8, v19
	v_add_nc_u32_e32 v28, 24, v20
	v_cmp_gt_u32_e64 s2, v8, v20
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v29, 0, v16, vcc_lo
	v_dual_mov_b32 v19, v22 :: v_dual_add_nc_u32 v24, 8, v20
	v_add_nc_u32_e32 v20, 28, v20
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v23, 0, v16, s2
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, v8, v24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v24, 0, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, v8, v25
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v25, 0, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, v8, v26
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v26, 0, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, v8, v27
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v27, 0, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, v8, v28
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v28, 0, v16, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, v8, v20
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v20, 0, v16, vcc_lo
	ds_store_2addr_stride64_b32 v21, v23, v29 offset1:4
	ds_store_2addr_stride64_b32 v21, v24, v25 offset0:8 offset1:12
	ds_store_2addr_stride64_b32 v21, v26, v27 offset0:16 offset1:20
	ds_store_2addr_stride64_b32 v21, v28, v20 offset0:24 offset1:28
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execnz .LBB4_5
; %bb.6:
	s_or_b32 exec_lo, exec_lo, s3
	s_load_b32 s0, s[0:1], 0x10
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s1, 0x3c23d70a
	v_fma_f32 v33, 0x3c23d70a, v9, 0
	v_fmaak_f32 v34, s1, v9, 0x38d1b717
	v_fmaak_f32 v35, s1, v9, 0x3951b717
	v_fmaak_f32 v36, s1, v9, 0x399d4951
	v_fmaak_f32 v37, s1, v9, 0x39d1b717
	v_fmaak_f32 v38, s1, v9, 0x3a03126e
	v_fmaak_f32 v39, s1, v9, 0x3a1d4951
	v_fmaak_f32 v40, s1, v9, 0x3a378034
	v_fmaak_f32 v64, s1, v9, 0x3a51b717
	v_fmaak_f32 v56, s1, v9, 0x3a6bedfa
	v_fmaak_f32 v48, s1, v9, 0x3a83126e
	v_fmaak_f32 v32, s1, v9, 0x3a902de0
	v_fmaak_f32 v24, s1, v9, 0x3a9d4951
	v_fmaak_f32 v16, s1, v9, 0x3aaa64c3
	v_fmaak_f32 v8, s1, v9, 0x3ab78034
	s_barrier_wait -1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB4_11
; %bb.7:
	s_mov_b32 s1, 0x3a83126f
	v_dual_mov_b32 v58, v35 :: v_dual_and_b32 v19, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v109, v1, s1, 0
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_fma_mixhi_f16 v109, v3, s1, 0
	v_fma_mixlo_f16 v106, v4, s1, 0
	v_fma_mixhi_f16 v105, v2, s1, 0
	v_lshrrev_b32_e32 v2, 1, v0
	v_xor_b32_e32 v3, 16, v1
	v_bfi_b32 v4, v1, 0, 32
	v_fma_mixhi_f16 v111, v14, s1, 0
	v_fma_mixlo_f16 v105, v9, s1, 0
	v_dual_mov_b32 v60, v37 :: v_dual_and_b32 v145, 8, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_lt_u32_e32 vcc_lo, v3, v4
	v_fma_mixlo_f16 v110, v5, s1, 0
	v_lshlrev_b32_e32 v14, 7, v0
	v_fma_mixhi_f16 v106, v6, s1, 0
	v_fma_mixhi_f16 v110, v7, s1, 0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo
	v_fma_mixlo_f16 v107, v17, s1, 0
	v_fma_mixlo_f16 v111, v18, s1, 0
	v_fma_mixhi_f16 v107, v13, s1, 0
	v_fma_mixlo_f16 v108, v15, s1, 0
	v_fma_mixlo_f16 v112, v11, s1, 0
	v_fma_mixhi_f16 v108, v12, s1, 0
	v_fma_mixhi_f16 v112, v10, s1, 0
	v_lshl_add_u32 v146, v19, 2, 0
	v_or_b32_e32 v2, 1, v145
	v_or_b32_e32 v3, 2, v145
	v_or_b32_e32 v4, 3, v145
	v_or_b32_e32 v5, 4, v145
	v_or_b32_e32 v6, 5, v145
	v_or_b32_e32 v7, 6, v145
	v_or_b32_e32 v9, 7, v145
	v_or_b32_e32 v10, 16, v145
	v_or_b32_e32 v11, 17, v145
	v_or_b32_e32 v12, 18, v145
	v_or_b32_e32 v13, 19, v145
	v_or_b32_e32 v15, 20, v145
	v_or_b32_e32 v17, 21, v145
	v_or_b32_e32 v18, 22, v145
	v_or_b32_e32 v19, 23, v145
	v_dual_mov_b32 v62, v39 :: v_dual_and_b32 v147, 0x800, v14
	v_dual_mov_b32 v57, v34 :: v_dual_lshlrev_b32 v148, 2, v1
	v_dual_mov_b32 v50, v36 :: v_dual_lshlrev_b32 v149, 2, v2
	v_dual_mov_b32 v59, v36 :: v_dual_lshlrev_b32 v150, 2, v3
	v_dual_mov_b32 v52, v38 :: v_dual_lshlrev_b32 v151, 2, v4
	v_dual_mov_b32 v61, v38 :: v_dual_lshlrev_b32 v152, 2, v5
	v_dual_mov_b32 v54, v40 :: v_dual_lshlrev_b32 v153, 2, v6
	v_dual_mov_b32 v63, v40 :: v_dual_lshlrev_b32 v154, 2, v7
	v_dual_mov_b32 v42, v37 :: v_dual_lshlrev_b32 v155, 2, v9
	v_dual_mov_b32 v49, v35 :: v_dual_lshlrev_b32 v156, 2, v10
	v_dual_mov_b32 v44, v39 :: v_dual_lshlrev_b32 v157, 2, v11
	v_dual_mov_b32 v51, v37 :: v_dual_lshlrev_b32 v158, 2, v12
	v_dual_mov_b32 v46, v64 :: v_dual_lshlrev_b32 v159, 2, v13
	v_dual_mov_b32 v53, v39 :: v_dual_lshlrev_b32 v160, 2, v15
	v_dual_mov_b32 v26, v38 :: v_dual_lshlrev_b32 v161, 2, v17
	v_dual_mov_b32 v55, v64 :: v_dual_lshlrev_b32 v162, 2, v18
	v_dual_mov_b32 v28, v40 :: v_dual_lshlrev_b32 v163, 2, v19
	v_dual_mov_b32 v41, v36 :: v_dual_mov_b32 v30, v56
	v_dual_mov_b32 v43, v38 :: v_dual_mov_b32 v18, v39
	v_dual_mov_b32 v45, v40 :: v_dual_mov_b32 v20, v64
	v_dual_mov_b32 v47, v56 :: v_dual_mov_b32 v22, v48
	v_dual_mov_b32 v25, v37 :: v_dual_mov_b32 v10, v40
	v_dual_mov_b32 v27, v39 :: v_dual_mov_b32 v12, v56
	v_dual_mov_b32 v29, v64 :: v_dual_mov_b32 v14, v32
	v_dual_mov_b32 v31, v48 :: v_dual_mov_b32 v2, v64
	v_dual_mov_b32 v17, v38 :: v_dual_mov_b32 v4, v48
	v_dual_mov_b32 v19, v40 :: v_dual_mov_b32 v6, v24
	v_mov_b32_e32 v21, v56
	v_mov_b32_e32 v23, v32
	v_mov_b32_e32 v9, v39
	v_mov_b32_e32 v11, v64
	v_mov_b32_e32 v13, v48
	v_mov_b32_e32 v15, v24
	v_mov_b32_e32 v1, v40
	v_mov_b32_e32 v3, v56
	v_mov_b32_e32 v5, v32
	v_mov_b32_e32 v7, v16
	v_or_b32_e32 v164, 0x2300, v147
	v_or_b32_e32 v165, 0x2200, v147
	v_or_b32_e32 v166, 0x2100, v147
	v_or_b32_e32 v167, 0x2000, v147
	v_or_b32_e32 v168, 0x1700, v147
	v_or_b32_e32 v169, 0x1600, v147
	v_or_b32_e32 v170, 0x1500, v147
	v_or_b32_e32 v171, 0x1400, v147
	v_or_b32_e32 v172, 0x1300, v147
	v_or_b32_e32 v173, 0x1200, v147
	v_or_b32_e32 v174, 0x1100, v147
	v_or_b32_e32 v175, 0x1000, v147
	v_or_b32_e32 v176, 0x700, v147
	v_or_b32_e32 v177, 0x600, v147
	v_or_b32_e32 v178, 0x500, v147
	v_or_b32_e32 v179, 0x400, v147
	v_or_b32_e32 v180, 0x300, v147
	v_or_b32_e32 v181, 0x200, v147
	v_or_b32_e32 v182, 0x100, v147
	s_mov_b32 s1, 0
.LBB4_8:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_9 Depth 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], 0
	v_cvt_f16_f32_e32 v116.h, v40
	v_cvt_f16_f32_e32 v116.l, v39
	v_cvt_f16_f32_e32 v115.h, v38
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e32 v115.l, v37
	v_cvt_f16_f32_e32 v114.h, v36
	v_cvt_f16_f32_e32 v114.l, v35
	v_cvt_f16_f32_e32 v113.h, v34
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e32 v113.l, v33
	v_cvt_f16_f32_e32 v120.h, v64
	v_cvt_f16_f32_e32 v120.l, v63
	v_cvt_f16_f32_e32 v119.h, v62
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e32 v119.l, v61
	v_cvt_f16_f32_e32 v118.h, v60
	v_cvt_f16_f32_e32 v118.l, v59
	v_cvt_f16_f32_e32 v117.h, v58
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e32 v117.l, v57
	v_cvt_f16_f32_e32 v124.h, v56
	v_cvt_f16_f32_e32 v124.l, v55
	v_cvt_f16_f32_e32 v123.h, v54
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e32 v123.l, v53
	v_cvt_f16_f32_e32 v122.h, v52
	v_cvt_f16_f32_e32 v122.l, v51
	v_cvt_f16_f32_e32 v121.h, v50
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e32 v121.l, v49
	v_cvt_f16_f32_e64 v128.h, v48
	v_cvt_f16_f32_e64 v128.l, v47
	v_cvt_f16_f32_e32 v127.h, v46
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e32 v127.l, v45
	v_cvt_f16_f32_e32 v126.h, v44
	v_cvt_f16_f32_e32 v126.l, v43
	v_cvt_f16_f32_e32 v125.h, v42
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e32 v125.l, v41
	v_cvt_f16_f32_e64 v132.h, v32
	v_cvt_f16_f32_e64 v132.l, v31
	v_cvt_f16_f32_e64 v131.h, v30
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e64 v131.l, v29
	v_cvt_f16_f32_e64 v130.h, v28
	v_cvt_f16_f32_e64 v130.l, v27
	v_cvt_f16_f32_e64 v129.h, v26
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e64 v129.l, v25
	v_cvt_f16_f32_e64 v136.h, v24
	v_cvt_f16_f32_e64 v136.l, v23
	v_cvt_f16_f32_e64 v135.h, v22
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e64 v135.l, v21
	v_cvt_f16_f32_e64 v134.h, v20
	v_cvt_f16_f32_e64 v134.l, v19
	v_cvt_f16_f32_e64 v133.h, v18
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e64 v133.l, v17
	v_cvt_f16_f32_e64 v140.h, v16
	v_cvt_f16_f32_e64 v140.l, v15
	v_cvt_f16_f32_e64 v139.h, v14
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e64 v139.l, v13
	v_cvt_f16_f32_e64 v138.h, v12
	v_cvt_f16_f32_e64 v138.l, v11
	v_cvt_f16_f32_e64 v137.h, v10
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e64 v137.l, v9
	v_cvt_f16_f32_e64 v144.h, v8
	v_cvt_f16_f32_e64 v144.l, v7
	v_cvt_f16_f32_e64 v143.h, v6
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e64 v143.l, v5
	v_cvt_f16_f32_e64 v142.h, v4
	v_cvt_f16_f32_e64 v142.l, v3
	v_cvt_f16_f32_e64 v141.h, v2
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_cvt_f16_f32_e64 v141.l, v1
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[109:112], v[65:72]
	v_dual_add_f32 v80, 0x3727c5ac, v72 :: v_dual_add_f32 v79, 0x3727c5ac, v71
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v78, 0x3727c5ac, v70 :: v_dual_add_f32 v77, 0x3727c5ac, v69
	v_dual_add_f32 v76, 0x3727c5ac, v68 :: v_dual_add_f32 v75, 0x3727c5ac, v67
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v74, 0x3727c5ac, v66 :: v_dual_add_f32 v73, 0x3727c5ac, v65
	v_dual_add_f32 v88, 0x37a7c5ac, v72 :: v_dual_add_f32 v87, 0x37a7c5ac, v71
	v_dual_add_f32 v86, 0x37a7c5ac, v70 :: v_dual_add_f32 v85, 0x37a7c5ac, v69
	v_dual_add_f32 v84, 0x37a7c5ac, v68 :: v_dual_add_f32 v83, 0x37a7c5ac, v67
	v_dual_add_f32 v82, 0x37a7c5ac, v66 :: v_dual_add_f32 v81, 0x37a7c5ac, v65
	v_dual_add_f32 v190, 0x37fba882, v72 :: v_dual_add_f32 v189, 0x37fba882, v71
	v_dual_add_f32 v188, 0x37fba882, v70 :: v_dual_add_f32 v187, 0x37fba882, v69
	v_dual_add_f32 v186, 0x37fba882, v68 :: v_dual_add_f32 v185, 0x37fba882, v67
	v_dual_add_f32 v184, 0x37fba882, v66 :: v_dual_add_f32 v183, 0x37fba882, v65
	v_dual_add_f32 v198, 0x3827c5ac, v72 :: v_dual_add_f32 v197, 0x3827c5ac, v71
	v_dual_add_f32 v196, 0x3827c5ac, v70 :: v_dual_add_f32 v195, 0x3827c5ac, v69
	v_dual_add_f32 v194, 0x3827c5ac, v68 :: v_dual_add_f32 v193, 0x3827c5ac, v67
	v_dual_add_f32 v192, 0x3827c5ac, v66 :: v_dual_add_f32 v191, 0x3827c5ac, v65
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[113:116], v[73:80]
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[113:116], v[81:88]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[113:116], v[183:190]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[113:116], v[191:198]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[117:120], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[117:120], v[81:88]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[117:120], v[183:190]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[117:120], v[191:198]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[121:124], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[121:124], v[81:88]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[121:124], v[183:190]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[121:124], v[191:198]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[125:128], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[125:128], v[81:88]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[125:128], v[183:190]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[125:128], v[191:198]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[129:132], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[129:132], v[81:88]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[129:132], v[183:190]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[129:132], v[191:198]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[133:136], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[133:136], v[81:88]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[133:136], v[183:190]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[133:136], v[191:198]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[137:140], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[137:140], v[81:88]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[137:140], v[183:190]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[137:140], v[191:198]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[141:144], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[141:144], v[81:88]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[141:144], v[183:190]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[141:144], v[191:198]
	v_dual_mul_f32 v97, 0x3727c5ac, v73 :: v_dual_mul_f32 v98, 0x3727c5ac, v74
	v_dual_mul_f32 v99, 0x3727c5ac, v75 :: v_dual_mul_f32 v100, 0x3727c5ac, v76
	v_dual_mul_f32 v101, 0x3727c5ac, v77 :: v_dual_mul_f32 v102, 0x3727c5ac, v78
	v_dual_mul_f32 v103, 0x3727c5ac, v79 :: v_dual_mul_f32 v104, 0x3727c5ac, v80
	v_dual_mul_f32 v89, 0x3727c5ac, v81 :: v_dual_mul_f32 v90, 0x3727c5ac, v82
	v_dual_mul_f32 v91, 0x3727c5ac, v83 :: v_dual_mul_f32 v92, 0x3727c5ac, v84
	v_dual_mul_f32 v93, 0x3727c5ac, v85 :: v_dual_mul_f32 v94, 0x3727c5ac, v86
	v_dual_mul_f32 v95, 0x3727c5ac, v87 :: v_dual_mul_f32 v96, 0x3727c5ac, v88
	v_dual_mul_f32 v81, 0x3727c5ac, v183 :: v_dual_mul_f32 v82, 0x3727c5ac, v184
	v_dual_mul_f32 v83, 0x3727c5ac, v185 :: v_dual_mul_f32 v84, 0x3727c5ac, v186
	v_dual_mul_f32 v85, 0x3727c5ac, v187 :: v_dual_mul_f32 v86, 0x3727c5ac, v188
	v_dual_mul_f32 v87, 0x3727c5ac, v189 :: v_dual_mul_f32 v88, 0x3727c5ac, v190
	v_dual_mul_f32 v73, 0x3727c5ac, v191 :: v_dual_mul_f32 v74, 0x3727c5ac, v192
	v_dual_mul_f32 v75, 0x3727c5ac, v193 :: v_dual_mul_f32 v76, 0x3727c5ac, v194
	v_dual_mul_f32 v77, 0x3727c5ac, v195 :: v_dual_mul_f32 v78, 0x3727c5ac, v196
	v_dual_mul_f32 v79, 0x3727c5ac, v197 :: v_dual_mul_f32 v80, 0x3727c5ac, v198
.LBB4_9:                                ;   Parent Loop BB4_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e32 vcc_lo, s2, v147
	v_or_b32_e32 v184, 0x2400, v147
	v_or_b32_e32 v185, 0x2500, v147
	v_or_b32_e32 v186, 0x3200, v147
	v_or_b32_e32 v187, 0x3300, v147
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, 0, v97, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v182
	v_or_b32_e32 v188, 0x3600, v147
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v98, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v181
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v99, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v180
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v100, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v179
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v101, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v178
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v102, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v177
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v103, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v176
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v104, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v175
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v89, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v174
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v90, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v173
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v91, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v172
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v92, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v171
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v93, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v170
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v94, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v169
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v95, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v168
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v96, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v167
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v81, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v166
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v82, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v165
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v83, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v164
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v183, v183, v84, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v184
	v_or_b32_e32 v184, 0x2600, v147
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v85, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v185
	v_or_b32_e32 v185, 0x2700, v147
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v183, v183, v86, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v184
	v_or_b32_e32 v184, 0x3000, v147
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v183, v183, v87, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v185
	v_or_b32_e32 v185, 0x3100, v147
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v183, v183, v88, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v184
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v183, v183, v73 :: v_dual_add_nc_u32 v184, s2, v146
	v_cmp_eq_u32_e32 vcc_lo, s2, v185
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v185, v183, v74, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v186
	v_or_b32_e32 v186, 0x3400, v147
	ds_load_2addr_b32 v[183:184], v184 offset1:32
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v185, v185, v75, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v187
	v_or_b32_e32 v187, 0x3500, v147
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v185, v185, v76, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v186
	v_lshlrev_b32_e32 v186, 2, v145
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v185, v185, v77, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v187
	v_or_b32_e32 v187, 0x3700, v147
	s_wait_dscnt 0x0
	ds_bpermute_b32 v189, v186, v183
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v185, v185, v78, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v188
	ds_bpermute_b32 v190, v149, v183
	ds_bpermute_b32 v188, v150, v183
	ds_bpermute_b32 v191, v151, v183
	ds_bpermute_b32 v192, v152, v183
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v185, v185, v79, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v187
	ds_bpermute_b32 v187, v153, v183
	ds_bpermute_b32 v193, v154, v183
	ds_bpermute_b32 v194, v155, v183
	ds_bpermute_b32 v195, v156, v183
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v185, v185, v80, vcc_lo
	ds_bpermute_b32 v197, v157, v183
	ds_bpermute_b32 v198, v158, v183
	ds_bpermute_b32 v199, v159, v183
	ds_bpermute_b32 v200, v160, v183
	ds_bpermute_b32 v196, v148, v185
	ds_bpermute_b32 v201, v161, v183
	ds_bpermute_b32 v202, v162, v183
	ds_bpermute_b32 v183, v163, v183
	ds_bpermute_b32 v186, v186, v184
	ds_bpermute_b32 v203, v149, v184
	ds_bpermute_b32 v204, v150, v184
	ds_bpermute_b32 v205, v151, v184
	ds_bpermute_b32 v206, v152, v184
	ds_bpermute_b32 v207, v153, v184
	ds_bpermute_b32 v208, v154, v184
	ds_bpermute_b32 v209, v155, v184
	ds_bpermute_b32 v210, v156, v184
	ds_bpermute_b32 v211, v157, v184
	ds_bpermute_b32 v212, v158, v184
	ds_bpermute_b32 v213, v159, v184
	ds_bpermute_b32 v214, v161, v184
	ds_bpermute_b32 v215, v162, v184
	s_addk_co_i32 s2, 0x100
	s_wait_dscnt 0x11
	v_add_f32_e32 v185, v185, v196
	ds_bpermute_b32 v196, v160, v184
	ds_bpermute_b32 v184, v163, v184
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, 0x4000
	v_fma_f32 v97, -v185, v189, v97
	v_fma_f32 v98, -v185, v190, v98
	v_fma_f32 v99, -v185, v188, v99
	v_fma_f32 v100, -v185, v191, v100
	v_fma_f32 v101, -v185, v192, v101
	v_fma_f32 v102, -v185, v187, v102
	v_fma_f32 v103, -v185, v193, v103
	v_fma_f32 v104, -v185, v194, v104
	v_fma_f32 v89, -v185, v195, v89
	v_fma_f32 v90, -v185, v197, v90
	v_fma_f32 v91, -v185, v198, v91
	v_fma_f32 v92, -v185, v199, v92
	v_fma_f32 v93, -v185, v200, v93
	s_wait_dscnt 0x12
	v_fma_f32 v94, -v185, v201, v94
	s_wait_dscnt 0x11
	v_fma_f32 v95, -v185, v202, v95
	s_wait_dscnt 0x10
	v_fma_f32 v96, -v185, v183, v96
	s_wait_dscnt 0xf
	v_fma_f32 v81, -v185, v186, v81
	s_wait_dscnt 0xe
	v_fma_f32 v82, -v185, v203, v82
	s_wait_dscnt 0xd
	v_fma_f32 v83, -v185, v204, v83
	s_wait_dscnt 0xc
	v_fma_f32 v84, -v185, v205, v84
	s_wait_dscnt 0xb
	v_fma_f32 v85, -v185, v206, v85
	s_wait_dscnt 0xa
	v_fma_f32 v86, -v185, v207, v86
	s_wait_dscnt 0x9
	v_fma_f32 v87, -v185, v208, v87
	s_wait_dscnt 0x8
	v_fma_f32 v88, -v185, v209, v88
	s_wait_dscnt 0x7
	v_fma_f32 v73, -v185, v210, v73
	s_wait_dscnt 0x6
	v_fma_f32 v74, -v185, v211, v74
	s_wait_dscnt 0x5
	v_fma_f32 v75, -v185, v212, v75
	s_wait_dscnt 0x4
	v_fma_f32 v76, -v185, v213, v76
	s_wait_dscnt 0x1
	v_fma_f32 v77, -v185, v196, v77
	v_fma_f32 v78, -v185, v214, v78
	v_fma_f32 v79, -v185, v215, v79
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v185, v184, v80
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB4_9
; %bb.10:                               ;   in Loop: Header=BB4_8 Depth=1
	v_wmma_f32_16x16x16_f16 v[183:190], v[109:112], v[113:116], v[65:72]
	v_cvt_f16_f32_e32 v104.h, v104
	v_cvt_f16_f32_e32 v104.l, v103
	v_cvt_f16_f32_e32 v103.h, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[183:190], v[109:112], v[117:120], v[183:190]
	v_cvt_f16_f32_e32 v103.l, v101
	v_cvt_f16_f32_e32 v102.h, v100
	v_cvt_f16_f32_e32 v102.l, v99
	v_cvt_f16_f32_e32 v101.h, v98
	v_wmma_f32_16x16x16_f16 v[183:190], v[109:112], v[121:124], v[183:190]
	v_cvt_f16_f32_e32 v101.l, v97
	v_dual_add_f32 v198, v72, v72 :: v_dual_add_f32 v197, v71, v71
	v_dual_add_f32 v196, v70, v70 :: v_dual_add_f32 v195, v69, v69
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[183:190], v[109:112], v[125:128], v[183:190]
	v_dual_add_f32 v194, v68, v68 :: v_dual_add_f32 v193, v67, v67
	v_dual_add_f32 v192, v66, v66 :: v_dual_add_f32 v191, v65, v65
	v_wmma_f32_16x16x16_f16 v[183:190], v[109:112], v[129:132], v[183:190]
	v_cvt_f16_f32_e32 v96.h, v96
	v_cvt_f16_f32_e32 v96.l, v95
	v_cvt_f16_f32_e32 v95.h, v94
	v_cvt_f16_f32_e32 v95.l, v93
	v_wmma_f32_16x16x16_f16 v[183:190], v[109:112], v[133:136], v[183:190]
	v_cvt_f16_f32_e32 v94.h, v92
	v_cvt_f16_f32_e32 v94.l, v91
	v_cvt_f16_f32_e32 v93.h, v90
	v_cvt_f16_f32_e32 v93.l, v89
	v_wmma_f32_16x16x16_f16 v[183:190], v[109:112], v[137:140], v[183:190]
	v_wmma_f32_16x16x16_f16 v[191:198], v[109:112], v[113:116], v[191:198]
	v_cvt_f16_f32_e32 v88.h, v88
	v_cvt_f16_f32_e32 v88.l, v87
	v_cvt_f16_f32_e32 v87.h, v86
	v_wmma_f32_16x16x16_f16 v[183:190], v[109:112], v[141:144], v[183:190]
	v_cvt_f16_f32_e32 v87.l, v85
	v_cvt_f16_f32_e32 v86.h, v84
	v_cvt_f16_f32_e32 v86.l, v83
	v_cvt_f16_f32_e32 v85.h, v82
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[101:104], v[183:190]
	v_cvt_f16_f32_e32 v85.l, v81
	v_wmma_f32_16x16x16_f16 v[191:198], v[109:112], v[117:120], v[191:198]
	v_cvt_f16_f32_e32 v84.h, v80
	v_cvt_f16_f32_e32 v84.l, v79
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[93:96], v[183:190]
	v_cvt_f16_f32_e32 v83.h, v78
	v_cvt_f16_f32_e32 v83.l, v77
	v_cvt_f16_f32_e32 v82.h, v76
	v_cvt_f16_f32_e32 v82.l, v75
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[85:88], v[183:190]
	v_cvt_f16_f32_e32 v81.h, v74
	v_wmma_f32_16x16x16_f16 v[191:198], v[109:112], v[121:124], v[191:198]
	v_cvt_f16_f32_e32 v81.l, v73
	v_dual_mul_f32 v80, 0x40400000, v72 :: v_dual_mul_f32 v79, 0x40400000, v71
	v_dual_mul_f32 v77, 0x40400000, v69 :: v_dual_mul_f32 v76, 0x40400000, v68
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[109:112], v[125:128], v[191:198]
	v_wmma_f32_16x16x16_f16 v[183:190], v[105:108], v[81:84], v[183:190]
	v_dual_mul_f32 v78, 0x40400000, v70 :: v_dual_mul_f32 v75, 0x40400000, v67
	v_mul_f32_e32 v72, 4.0, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[109:112], v[129:132], v[191:198]
	v_dual_add_f32 v73, 0, v183 :: v_dual_mul_f32 v74, 0x40400000, v66
	v_dual_mul_f32 v70, 4.0, v70 :: v_dual_mul_f32 v71, 4.0, v71
	v_mul_f32_e32 v66, 4.0, v66
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[109:112], v[133:136], v[191:198]
	v_add_f32_e32 v89, v73, v184
	v_dual_mul_f32 v73, 0x40400000, v65 :: v_dual_mul_f32 v68, 4.0, v68
	v_dual_mul_f32 v69, 4.0, v69 :: v_dual_mul_f32 v38, 0x3f7d70a4, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[109:112], v[137:140], v[191:198]
	v_add_f32_e32 v89, v89, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[113:116], v[73:80]
	v_mul_f32_e32 v40, 0x3f7d70a4, v40
	v_mul_f32_e32 v36, 0x3f7d70a4, v36
	v_wmma_f32_16x16x16_f16 v[191:198], v[109:112], v[141:144], v[191:198]
	v_add_f32_e32 v89, v89, v186
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[117:120], v[73:80]
	v_dual_mul_f32 v67, 4.0, v67 :: v_dual_mul_f32 v34, 0x3f7d70a4, v34
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[101:104], v[191:198]
	v_add_f32_e32 v89, v89, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[121:124], v[73:80]
	v_mul_f32_e32 v62, 0x3f7d70a4, v62
	v_mul_f32_e32 v60, 0x3f7d70a4, v60
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[93:96], v[191:198]
	v_add_f32_e32 v89, v89, v188
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[125:128], v[73:80]
	v_dual_mul_f32 v65, 4.0, v65 :: v_dual_mul_f32 v64, 0x3f7d70a4, v64
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[85:88], v[191:198]
	v_add_f32_e32 v89, v89, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[129:132], v[73:80]
	v_mul_f32_e32 v56, 0x3f7d70a4, v56
	v_mul_f32_e32 v50, 0x3f7d70a4, v50
	v_wmma_f32_16x16x16_f16 v[191:198], v[105:108], v[81:84], v[191:198]
	v_add_f32_e32 v89, v89, v190
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[133:136], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[113:116], v[65:72]
	v_dual_mul_f32 v46, 0x3f7d70a4, v46 :: v_dual_mul_f32 v39, 0x3f7d70a4, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v58, 0x3f7d70a4, v58 :: v_dual_add_f32 v89, v89, v191
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[137:140], v[73:80]
	v_mul_f32_e32 v32, 0x3f7d70a4, v32
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[117:120], v[65:72]
	v_dual_mul_f32 v37, 0x3f7d70a4, v37 :: v_dual_mul_f32 v54, 0x3f7d70a4, v54
	v_add_f32_e32 v89, v89, v192
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[141:144], v[73:80]
	v_mul_f32_e32 v28, 0x3f7d70a4, v28
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[121:124], v[65:72]
	v_dual_mul_f32 v35, 0x3f7d70a4, v35 :: v_dual_mul_f32 v52, 0x3f7d70a4, v52
	v_add_f32_e32 v89, v89, v193
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[101:104], v[73:80]
	v_mul_f32_e32 v24, 0x3f7d70a4, v24
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[125:128], v[65:72]
	v_dual_mul_f32 v33, 0x3f7d70a4, v33 :: v_dual_mul_f32 v48, 0x3f7d70a4, v48
	v_add_f32_e32 v89, v89, v194
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[93:96], v[73:80]
	v_mul_f32_e32 v18, 0x3f7d70a4, v18
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[129:132], v[65:72]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v16, 0x3f7d70a4, v16 :: v_dual_add_f32 v89, v89, v195
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[85:88], v[73:80]
	v_dual_mul_f32 v63, 0x3f7d70a4, v63 :: v_dual_mul_f32 v44, 0x3f7d70a4, v44
	v_mul_f32_e32 v12, 0x3f7d70a4, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v89, v89, v196
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[81:84], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[133:136], v[65:72]
	v_dual_mul_f32 v61, 0x3f7d70a4, v61 :: v_dual_mul_f32 v42, 0x3f7d70a4, v42
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v89, v89, v197 :: v_dual_mul_f32 v6, 0x3f7d70a4, v6
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[137:140], v[65:72]
	v_dual_mul_f32 v59, 0x3f7d70a4, v59 :: v_dual_mul_f32 v30, 0x3f7d70a4, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v89, v89, v198
	v_dual_mul_f32 v57, 0x3f7d70a4, v57 :: v_dual_mul_f32 v26, 0x3f7d70a4, v26
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[141:144], v[65:72]
	v_dual_mul_f32 v55, 0x3f7d70a4, v55 :: v_dual_mul_f32 v22, 0x3f7d70a4, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v73, v89, v73
	v_dual_mul_f32 v53, 0x3f7d70a4, v53 :: v_dual_mul_f32 v20, 0x3f7d70a4, v20
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[101:104], v[65:72]
	v_dual_mul_f32 v51, 0x3f7d70a4, v51 :: v_dual_mul_f32 v14, 0x3f7d70a4, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v73, v73, v74
	v_dual_mul_f32 v49, 0x3f7d70a4, v49 :: v_dual_mul_f32 v10, 0x3f7d70a4, v10
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[93:96], v[65:72]
	v_dual_mul_f32 v47, 0x3f7d70a4, v47 :: v_dual_mul_f32 v8, 0x3f7d70a4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v73, v73, v75
	v_dual_mul_f32 v45, 0x3f7d70a4, v45 :: v_dual_mul_f32 v4, 0x3f7d70a4, v4
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[85:88], v[65:72]
	v_dual_mul_f32 v43, 0x3f7d70a4, v43 :: v_dual_mul_f32 v2, 0x3f7d70a4, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v73, v73, v76
	v_mul_f32_e32 v41, 0x3f7d70a4, v41
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[81:84], v[65:72]
	v_mul_f32_e32 v31, 0x3f7d70a4, v31
	v_mul_f32_e32 v29, 0x3f7d70a4, v29
	v_add_f32_e32 v73, v73, v77
	v_mul_f32_e32 v27, 0x3f7d70a4, v27
	v_mul_f32_e32 v25, 0x3f7d70a4, v25
	v_mul_f32_e32 v23, 0x3f7d70a4, v23
	v_mul_f32_e32 v21, 0x3f7d70a4, v21
	v_add_f32_e32 v73, v73, v78
	v_mul_f32_e32 v19, 0x3f7d70a4, v19
	v_mul_f32_e32 v17, 0x3f7d70a4, v17
	v_mul_f32_e32 v15, 0x3f7d70a4, v15
	v_mul_f32_e32 v13, 0x3f7d70a4, v13
	v_add_f32_e32 v73, v73, v79
	v_mul_f32_e32 v11, 0x3f7d70a4, v11
	v_mul_f32_e32 v9, 0x3f7d70a4, v9
	v_mul_f32_e32 v7, 0x3f7d70a4, v7
	v_mul_f32_e32 v5, 0x3f7d70a4, v5
	v_add_f32_e32 v73, v73, v80
	v_mul_f32_e32 v3, 0x3f7d70a4, v3
	v_mul_f32_e32 v1, 0x3f7d70a4, v1
	v_wmma_f32_16x16x16_f16 v[33:40], v[105:108], v[101:104], v[33:40]
	v_wmma_f32_16x16x16_f16 v[57:64], v[105:108], v[101:104], v[57:64]
	v_add_f32_e32 v65, v73, v65
	v_wmma_f32_16x16x16_f16 v[49:56], v[105:108], v[101:104], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[105:108], v[101:104], v[41:48]
	v_wmma_f32_16x16x16_f16 v[25:32], v[105:108], v[101:104], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[105:108], v[101:104], v[17:24]
	v_add_f32_e32 v65, v65, v66
	v_wmma_f32_16x16x16_f16 v[9:16], v[105:108], v[101:104], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[105:108], v[101:104], v[1:8]
	v_wmma_f32_16x16x16_f16 v[33:40], v[105:108], v[93:96], v[33:40]
	v_wmma_f32_16x16x16_f16 v[57:64], v[105:108], v[93:96], v[57:64]
	v_add_f32_e32 v65, v65, v67
	v_wmma_f32_16x16x16_f16 v[49:56], v[105:108], v[93:96], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[105:108], v[93:96], v[41:48]
	v_wmma_f32_16x16x16_f16 v[25:32], v[105:108], v[93:96], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[105:108], v[93:96], v[17:24]
	v_add_f32_e32 v65, v65, v68
	v_wmma_f32_16x16x16_f16 v[9:16], v[105:108], v[93:96], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[105:108], v[93:96], v[1:8]
	v_mov_b32_e32 v66, 0x3b03126f
	v_wmma_f32_16x16x16_f16 v[33:40], v[105:108], v[85:88], v[33:40]
	v_add_f32_e32 v65, v65, v69
	v_wmma_f32_16x16x16_f16 v[57:64], v[105:108], v[85:88], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[105:108], v[85:88], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[105:108], v[85:88], v[41:48]
	v_wmma_f32_16x16x16_f16 v[25:32], v[105:108], v[85:88], v[25:32]
	v_add_f32_e32 v65, v65, v70
	v_wmma_f32_16x16x16_f16 v[17:24], v[105:108], v[85:88], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[105:108], v[85:88], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[105:108], v[85:88], v[1:8]
	v_wmma_f32_16x16x16_f16 v[33:40], v[105:108], v[81:84], v[33:40]
	v_add_f32_e32 v65, v65, v71
	v_wmma_f32_16x16x16_f16 v[57:64], v[105:108], v[81:84], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[105:108], v[81:84], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[105:108], v[81:84], v[41:48]
	v_wmma_f32_16x16x16_f16 v[25:32], v[105:108], v[81:84], v[25:32]
	v_add_f32_e32 v65, v65, v72
	v_wmma_f32_16x16x16_f16 v[17:24], v[105:108], v[81:84], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[105:108], v[81:84], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[105:108], v[81:84], v[1:8]
	s_add_co_i32 s1, s1, 1
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v65
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v65, 0x3a83126f, v66, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v109.l, v65
	s_cbranch_scc0 .LBB4_8
	s_branch .LBB4_12
.LBB4_11:
	v_dual_mov_b32 v7, v16 :: v_dual_mov_b32 v6, v24
	v_dual_mov_b32 v5, v32 :: v_dual_mov_b32 v4, v48
	v_dual_mov_b32 v3, v56 :: v_dual_mov_b32 v2, v64
	v_dual_mov_b32 v1, v40 :: v_dual_mov_b32 v14, v32
	v_dual_mov_b32 v15, v24 :: v_dual_mov_b32 v12, v56
	v_dual_mov_b32 v13, v48 :: v_dual_mov_b32 v10, v40
	v_dual_mov_b32 v11, v64 :: v_dual_mov_b32 v22, v48
	v_dual_mov_b32 v9, v39 :: v_dual_mov_b32 v20, v64
	v_dual_mov_b32 v23, v32 :: v_dual_mov_b32 v18, v39
	v_dual_mov_b32 v21, v56 :: v_dual_mov_b32 v30, v56
	v_dual_mov_b32 v19, v40 :: v_dual_mov_b32 v28, v40
	v_dual_mov_b32 v17, v38 :: v_dual_mov_b32 v26, v38
	v_dual_mov_b32 v31, v48 :: v_dual_mov_b32 v46, v64
	v_dual_mov_b32 v29, v64 :: v_dual_mov_b32 v44, v39
	v_dual_mov_b32 v27, v39 :: v_dual_mov_b32 v42, v37
	v_dual_mov_b32 v25, v37 :: v_dual_mov_b32 v54, v40
	v_dual_mov_b32 v47, v56 :: v_dual_mov_b32 v52, v38
	v_dual_mov_b32 v45, v40 :: v_dual_mov_b32 v50, v36
	v_dual_mov_b32 v43, v38 :: v_dual_mov_b32 v62, v39
	v_dual_mov_b32 v41, v36 :: v_dual_mov_b32 v60, v37
	v_dual_mov_b32 v55, v64 :: v_dual_mov_b32 v58, v35
	v_mov_b32_e32 v53, v39
	v_mov_b32_e32 v51, v37
	v_mov_b32_e32 v49, v35
	v_mov_b32_e32 v63, v40
	v_mov_b32_e32 v61, v38
	v_mov_b32_e32 v59, v36
	v_mov_b32_e32 v57, v34
.LBB4_12:
	v_add_f32_e32 v33, 0, v33
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v34
	v_add_f32_e32 v33, v33, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v36
	v_add_f32_e32 v33, v33, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v38
	v_add_f32_e32 v33, v33, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v40
	v_add_f32_e32 v33, v33, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v58
	v_add_f32_e32 v33, v33, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v60
	v_add_f32_e32 v33, v33, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v62
	v_add_f32_e32 v33, v33, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v64
	v_add_f32_e32 v33, v33, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v50
	v_add_f32_e32 v33, v33, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v52
	v_add_f32_e32 v33, v33, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v54
	v_add_f32_e32 v33, v33, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v56
	v_add_f32_e32 v33, v33, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v42
	v_add_f32_e32 v33, v33, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v44
	v_add_f32_e32 v33, v33, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v46
	v_add_f32_e32 v33, v33, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v48
	v_add_f32_e32 v25, v33, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v26
	v_add_f32_e32 v25, v25, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v28
	v_add_f32_e32 v25, v25, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v30
	v_add_f32_e32 v25, v25, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v32
	v_add_f32_e32 v17, v25, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v1, v6 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_f32_e32 v2, v2, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end4:
	.size	_Z5probeILi8ELb0ELb1EEvPfPKfi, .Lfunc_end4-_Z5probeILi8ELb0ELb1EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi8ELb0ELb1EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 216
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_Z5probeILi8ELb0ELb1EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi8ELb0ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi8ELb0ELb1EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.num_vgpr, 216
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.numbered_sgpr, 8
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6480
; TotalNumSgprs: 10
; NumVgprs: 216
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 26
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 216
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi8ELb1ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi8ELb1ELb1EEvPfPKfi,comdat
	.protected	_Z5probeILi8ELb1ELb1EEvPfPKfi ; -- Begin function _Z5probeILi8ELb1ELb1EEvPfPKfi
	.globl	_Z5probeILi8ELb1ELb1EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi8ELb1ELb1EEvPfPKfi,@function
_Z5probeILi8ELb1ELb1EEvPfPKfi:          ; @_Z5probeILi8ELb1ELb1EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_add_nc_u32_e32 v3, 1, v0
	s_load_b128 s[4:7], s[0:1], 0x0
	v_add_nc_u32_e32 v4, 10, v0
	v_add_nc_u32_e32 v5, 2, v0
	v_add_nc_u32_e32 v6, 11, v0
	v_and_b32_e32 v3, 0xff, v3
	v_add_nc_u32_e32 v2, 9, v0
	v_and_b32_e32 v4, 0xff, v4
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v6, 0xff, v6
	v_add_nc_u32_e32 v8, 12, v0
	v_lshlrev_b32_e32 v7, 2, v3
	v_add_nc_u32_e32 v3, 3, v0
	v_and_b32_e32 v2, 0xff, v2
	v_lshlrev_b32_e32 v9, 2, v4
	v_lshlrev_b32_e32 v11, 2, v5
	v_lshlrev_b32_e32 v12, 2, v6
	v_and_b32_e32 v4, 0xff, v8
	v_add_nc_u32_e32 v5, 4, v0
	v_add_nc_u32_e32 v6, 13, v0
	v_add_nc_u32_e32 v8, 5, v0
	v_and_b32_e32 v3, 0xff, v3
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_and_b32_e32 v15, 0xff, v5
	v_and_b32_e32 v16, 0xff, v6
	v_and_b32_e32 v17, 0xff, v8
	v_lshlrev_b32_e32 v13, 2, v3
	v_lshlrev_b32_e32 v14, 2, v4
	s_wait_kmcnt 0x0
	s_clause 0x7
	global_load_b32 v10, v1, s[6:7]
	global_load_b32 v3, v2, s[6:7]
	global_load_b32 v4, v7, s[6:7]
	global_load_b32 v5, v9, s[6:7]
	global_load_b32 v6, v11, s[6:7]
	global_load_b32 v7, v12, s[6:7]
	global_load_b32 v8, v13, s[6:7]
	global_load_b32 v9, v14, s[6:7]
	v_add_nc_u32_e32 v13, 14, v0
	v_add_nc_u32_e32 v14, 6, v0
	v_lshlrev_b32_e32 v2, 2, v15
	v_lshlrev_b32_e32 v11, 2, v16
	v_lshlrev_b32_e32 v12, 2, v17
	v_add_nc_u32_e32 v15, 15, v0
	v_add_nc_u32_e32 v16, 7, v0
	v_add_nc_u32_e32 v17, 16, v0
	v_and_b32_e32 v13, 0xff, v13
	v_and_b32_e32 v14, 0xff, v14
	v_and_b32_e32 v15, 0xff, v15
	v_and_b32_e32 v16, 0xff, v16
	v_and_b32_e32 v17, 0xff, v17
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v19, 2, v14
	v_lshlrev_b32_e32 v20, 2, v15
	v_lshlrev_b32_e32 v21, 2, v16
	v_lshlrev_b32_e32 v22, 2, v17
	s_clause 0x7
	global_load_b32 v17, v2, s[6:7]
	global_load_b32 v18, v11, s[6:7]
	global_load_b32 v14, v12, s[6:7]
	global_load_b32 v15, v13, s[6:7]
	global_load_b32 v16, v19, s[6:7]
	global_load_b32 v12, v20, s[6:7]
	global_load_b32 v13, v21, s[6:7]
	global_load_b32 v11, v22, s[6:7]
	v_mov_b32_e32 v20, 0
	v_add_co_u32 v1, s2, s6, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s7, 0, s2
	v_and_b32_e32 v19, 63, v0
	s_mov_b32 s2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB5_6
; %bb.1:
	v_lshl_add_u32 v21, v0, 2, 0
	v_lshrrev_b32_e32 v22, 6, v0
	s_mov_b32 s6, 0
	s_branch .LBB5_3
.LBB5_2:                                ;   in Loop: Header=BB5_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v20, -1, v20
	ds_store_b32 v21, v23
	v_add_nc_u32_e32 v21, 0x400, v21
	v_add_nc_u32_e32 v22, 4, v22
	v_cmp_eq_u32_e32 vcc_lo, 0, v20
	s_or_b32 s6, vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB5_5
.LBB5_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v23, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB5_2
; %bb.4:                                ;   in Loop: Header=BB5_3 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
	s_branch .LBB5_2
.LBB5_5:
	s_or_b32 exec_lo, exec_lo, s6
.LBB5_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v20, v0
	s_branch .LBB5_8
.LBB5_7:                                ;   in Loop: Header=BB5_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v22, 0x800, v20
	v_cmp_lt_u32_e32 vcc_lo, 0x7ff, v20
	ds_store_b32 v21, v23 offset:7168
	v_mov_b32_e32 v20, v22
	s_or_b32 s2, vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB5_24
.LBB5_8:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v22, 6, v20
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB5_10
; %bb.9:                                ;   in Loop: Header=BB5_8 Depth=1
	global_load_b32 v21, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v21
.LBB5_10:                               ;   in Loop: Header=BB5_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v25, 4, v22
	v_lshl_add_u32 v21, v20, 2, 0
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v24
	v_cmpx_gt_u32_e64 v19, v25
	s_cbranch_execz .LBB5_12
; %bb.11:                               ;   in Loop: Header=BB5_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
.LBB5_12:                               ;   in Loop: Header=BB5_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v24, 0 :: v_dual_add_nc_u32 v25, 8, v22
	ds_store_b32 v21, v23 offset:1024
	v_cmp_gt_u32_e32 vcc_lo, v19, v25
	v_mov_b32_e32 v25, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB5_14
; %bb.13:                               ;   in Loop: Header=BB5_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v25, 0x38d1b717, v23
.LBB5_14:                               ;   in Loop: Header=BB5_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v23, 12, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v25 offset:2048
	v_cmpx_gt_u32_e64 v19, v23
	s_cbranch_execz .LBB5_16
; %bb.15:                               ;   in Loop: Header=BB5_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v23
.LBB5_16:                               ;   in Loop: Header=BB5_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v23, 16, v22
	v_mov_b32_e32 v25, 0
	ds_store_b32 v21, v24 offset:3072
	v_cmp_gt_u32_e32 vcc_lo, v19, v23
	v_mov_b32_e32 v23, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB5_18
; %bb.17:                               ;   in Loop: Header=BB5_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
.LBB5_18:                               ;   in Loop: Header=BB5_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v24, 20, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v23 offset:4096
	v_cmpx_gt_u32_e64 v19, v24
	s_cbranch_execz .LBB5_20
; %bb.19:                               ;   in Loop: Header=BB5_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v25, 0x38d1b717, v23
.LBB5_20:                               ;   in Loop: Header=BB5_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v23, 0 :: v_dual_add_nc_u32 v24, 24, v22
	ds_store_b32 v21, v25 offset:5120
	v_cmp_gt_u32_e32 vcc_lo, v19, v24
	v_mov_b32_e32 v24, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB5_22
; %bb.21:                               ;   in Loop: Header=BB5_8 Depth=1
	global_load_b32 v24, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v24
.LBB5_22:                               ;   in Loop: Header=BB5_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v22, 28, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v24 offset:6144
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB5_7
; %bb.23:                               ;   in Loop: Header=BB5_8 Depth=1
	global_load_b32 v22, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v22
	s_branch .LBB5_7
.LBB5_24:
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	s_mov_b32 s1, 0x3c23d70a
	s_mov_b32 s2, 0x38d1b717
	s_mov_b32 s3, 0x399d4951
	s_wait_loadcnt 0xf
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v57, v10, s1, s2
	s_mov_b32 s2, 0x3951b717
	v_fma_mixlo_f16 v62, v10, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v61, v10, s1, s2
	s_mov_b32 s2, 0x39d1b717
	s_mov_b32 s3, 0x3a03126e
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v62, v10, s1, s2
	s_mov_b32 s2, 0x3a1d4951
	s_wait_loadcnt_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v63, v10, s1, s2
	s_mov_b32 s2, 0x3a51b717
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v64, v10, s1, s2
	s_mov_b32 s2, 0x3a6bedfa
	v_fma_mixlo_f16 v63, v10, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v48, v10, s1, s2
	s_mov_b32 s2, 0x3a83126e
	s_mov_b32 s3, 0x3a378034
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v52, v10, s1, s2
	s_mov_b32 s2, 0x3a9d4951
	v_fma_mixlo_f16 v64, v10, s1, s3
	s_mov_b32 s3, 0x3a902de0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v56, v10, s1, s2
	s_mov_b32 s2, 0x3aaa64c3
	v_fma_mixhi_f16 v52, v10, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v68, v10, s1, s2
	s_mov_b32 s2, 0x3ab78034
	v_fma_mixlo_f16 v57, v10, s1, 0
	v_mov_b16_e32 v61.l, v57.h
	v_mov_b16_e32 v58.l, v61.h
	v_mov_b16_e32 v58.h, v62.l
	v_mov_b16_e32 v59.l, v62.h
	v_mov_b16_e32 v59.h, v63.l
	v_mov_b16_e32 v48.h, v52.l
	v_mov_b16_e32 v56.l, v52.h
	v_mov_b16_e32 v60.l, v63.h
	v_mov_b16_e32 v60.h, v64.l
	v_mov_b16_e32 v72.l, v64.h
	v_mov_b16_e32 v72.h, v48.l
	v_mov_b16_e32 v68.l, v56.h
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v44, v10, s1, s2
	v_mov_b16_e32 v44.l, v68.h
	s_mov_b32 s1, 0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB5_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s2, 0x3a83126f
	v_dual_mov_b32 v70, v59 :: v_dual_and_b32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v77, v3, s2, 0
	v_fma_mixhi_f16 v73, v4, s2, 0
	v_fma_mixhi_f16 v77, v5, s2, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixhi_f16 v78, v9, s2, 0
	v_dual_mov_b32 v46, v63 :: v_dual_lshlrev_b32 v9, 7, v0
	v_fma_mixlo_f16 v73, v10, s2, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_and_b32_e32 v81, 8, v3
	v_fma_mixlo_f16 v74, v6, s2, 0
	v_fma_mixlo_f16 v78, v7, s2, 0
	v_fma_mixhi_f16 v74, v8, s2, 0
	v_fma_mixhi_f16 v75, v14, s2, 0
	v_fma_mixhi_f16 v79, v15, s2, 0
	v_fma_mixlo_f16 v76, v16, s2, 0
	v_fma_mixlo_f16 v80, v12, s2, 0
	v_fma_mixhi_f16 v76, v13, s2, 0
	v_fma_mixhi_f16 v80, v11, s2, 0
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	v_lshl_add_u32 v82, v1, 2, 0
	v_or_b32_e32 v1, 1, v81
	v_or_b32_e32 v3, 2, v81
	v_or_b32_e32 v4, 3, v81
	v_or_b32_e32 v5, 4, v81
	v_or_b32_e32 v6, 5, v81
	v_or_b32_e32 v7, 6, v81
	v_or_b32_e32 v8, 7, v81
	v_or_b32_e32 v10, 16, v81
	v_or_b32_e32 v11, 17, v81
	v_or_b32_e32 v12, 18, v81
	v_or_b32_e32 v13, 19, v81
	v_dual_mov_b32 v50, v60 :: v_dual_and_b32 v83, 0x800, v9
	v_or_b32_e32 v9, 20, v81
	v_or_b32_e32 v14, 21, v81
	v_or_b32_e32 v15, 22, v81
	v_or_b32_e32 v16, 23, v81
	v_fma_mixlo_f16 v75, v17, s2, 0
	v_fma_mixlo_f16 v79, v18, s2, 0
	v_or_b32_e32 v84, 0x1300, v83
	v_or_b32_e32 v85, 0x1200, v83
	v_or_b32_e32 v86, 0x1100, v83
	v_or_b32_e32 v87, 0x1000, v83
	v_or_b32_e32 v88, 0x700, v83
	v_or_b32_e32 v89, 0x600, v83
	v_or_b32_e32 v90, 0x500, v83
	v_or_b32_e32 v91, 0x400, v83
	v_or_b32_e32 v92, 0x300, v83
	v_or_b32_e32 v93, 0x200, v83
	v_or_b32_e32 v94, 0x100, v83
	v_dual_mov_b32 v54, v64 :: v_dual_lshlrev_b32 v95, 2, v2
	v_dual_mov_b32 v69, v58 :: v_dual_lshlrev_b32 v96, 2, v1
	v_dual_mov_b32 v66, v72 :: v_dual_lshlrev_b32 v97, 2, v3
	v_dual_mov_b32 v71, v60 :: v_dual_lshlrev_b32 v98, 2, v4
	v_dual_mov_b32 v42, v48 :: v_dual_lshlrev_b32 v99, 2, v5
	v_dual_mov_b32 v45, v62 :: v_dual_lshlrev_b32 v100, 2, v6
	v_lshlrev_b32_e32 v101, 2, v7
	v_dual_mov_b32 v47, v64 :: v_dual_lshlrev_b32 v102, 2, v8
	v_lshlrev_b32_e32 v103, 2, v10
	v_dual_mov_b32 v49, v59 :: v_dual_lshlrev_b32 v104, 2, v11
	v_lshlrev_b32_e32 v105, 2, v12
	v_dual_mov_b32 v51, v72 :: v_dual_lshlrev_b32 v106, 2, v13
	v_lshlrev_b32_e32 v107, 2, v9
	v_dual_mov_b32 v53, v63 :: v_dual_lshlrev_b32 v108, 2, v14
	v_lshlrev_b32_e32 v109, 2, v15
	v_dual_mov_b32 v55, v48 :: v_dual_lshlrev_b32 v110, 2, v16
	v_mov_b32_e32 v65, v60
	v_mov_b32_e32 v67, v52
	v_mov_b32_e32 v41, v64
	v_mov_b32_e32 v43, v56
	s_mov_b32 s2, 0x3f7d70a4
.LBB5_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_27 Depth 2
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], 0
	s_mov_b32 s3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], v[1:8]
	v_dual_add_f32 v16, 0x3727c5ac, v8 :: v_dual_add_f32 v15, 0x3727c5ac, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v14, 0x3727c5ac, v6 :: v_dual_add_f32 v13, 0x3727c5ac, v5
	v_dual_add_f32 v12, 0x3727c5ac, v4 :: v_dual_add_f32 v11, 0x3727c5ac, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v10, 0x3727c5ac, v2 :: v_dual_add_f32 v9, 0x3727c5ac, v1
	v_dual_add_f32 v24, 0x37a7c5ac, v8 :: v_dual_add_f32 v23, 0x37a7c5ac, v7
	v_dual_add_f32 v22, 0x37a7c5ac, v6 :: v_dual_add_f32 v21, 0x37a7c5ac, v5
	v_dual_add_f32 v20, 0x37a7c5ac, v4 :: v_dual_add_f32 v19, 0x37a7c5ac, v3
	v_dual_add_f32 v18, 0x37a7c5ac, v2 :: v_dual_add_f32 v17, 0x37a7c5ac, v1
	v_dual_add_f32 v118, 0x37fba882, v8 :: v_dual_add_f32 v117, 0x37fba882, v7
	v_dual_add_f32 v116, 0x37fba882, v6 :: v_dual_add_f32 v115, 0x37fba882, v5
	v_dual_add_f32 v114, 0x37fba882, v4 :: v_dual_add_f32 v113, 0x37fba882, v3
	v_dual_add_f32 v112, 0x37fba882, v2 :: v_dual_add_f32 v111, 0x37fba882, v1
	v_dual_add_f32 v126, 0x3827c5ac, v8 :: v_dual_add_f32 v125, 0x3827c5ac, v7
	v_dual_add_f32 v124, 0x3827c5ac, v6 :: v_dual_add_f32 v123, 0x3827c5ac, v5
	v_dual_add_f32 v122, 0x3827c5ac, v4 :: v_dual_add_f32 v121, 0x3827c5ac, v3
	v_dual_add_f32 v120, 0x3827c5ac, v2 :: v_dual_add_f32 v119, 0x3827c5ac, v1
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[57:60], v[9:16]
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[57:60], v[17:24]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[57:60], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[57:60], v[119:126]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[61:64], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[61:64], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[61:64], v[119:126]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[69:72], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[69:72], v[17:24]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[69:72], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[69:72], v[119:126]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[45:48], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[45:48], v[17:24]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[45:48], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[45:48], v[119:126]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[49:52], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[49:52], v[17:24]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[49:52], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[49:52], v[119:126]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[53:56], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[53:56], v[17:24]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[53:56], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[53:56], v[119:126]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[65:68], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[65:68], v[17:24]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[65:68], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[65:68], v[119:126]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[41:44], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[41:44], v[17:24]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[41:44], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[41:44], v[119:126]
	v_dual_mul_f32 v33, 0x3727c5ac, v9 :: v_dual_mul_f32 v34, 0x3727c5ac, v10
	v_dual_mul_f32 v35, 0x3727c5ac, v11 :: v_dual_mul_f32 v36, 0x3727c5ac, v12
	v_dual_mul_f32 v37, 0x3727c5ac, v13 :: v_dual_mul_f32 v38, 0x3727c5ac, v14
	v_dual_mul_f32 v39, 0x3727c5ac, v15 :: v_dual_mul_f32 v40, 0x3727c5ac, v16
	v_dual_mul_f32 v25, 0x3727c5ac, v17 :: v_dual_mul_f32 v26, 0x3727c5ac, v18
	v_dual_mul_f32 v27, 0x3727c5ac, v19 :: v_dual_mul_f32 v28, 0x3727c5ac, v20
	v_dual_mul_f32 v29, 0x3727c5ac, v21 :: v_dual_mul_f32 v30, 0x3727c5ac, v22
	v_dual_mul_f32 v31, 0x3727c5ac, v23 :: v_dual_mul_f32 v32, 0x3727c5ac, v24
	v_dual_mul_f32 v17, 0x3727c5ac, v111 :: v_dual_mul_f32 v18, 0x3727c5ac, v112
	v_dual_mul_f32 v19, 0x3727c5ac, v113 :: v_dual_mul_f32 v20, 0x3727c5ac, v114
	v_dual_mul_f32 v21, 0x3727c5ac, v115 :: v_dual_mul_f32 v22, 0x3727c5ac, v116
	v_dual_mul_f32 v23, 0x3727c5ac, v117 :: v_dual_mul_f32 v24, 0x3727c5ac, v118
	v_dual_mul_f32 v9, 0x3727c5ac, v119 :: v_dual_mul_f32 v10, 0x3727c5ac, v120
	v_dual_mul_f32 v11, 0x3727c5ac, v121 :: v_dual_mul_f32 v12, 0x3727c5ac, v122
	v_dual_mul_f32 v13, 0x3727c5ac, v123 :: v_dual_mul_f32 v14, 0x3727c5ac, v124
	v_dual_mul_f32 v15, 0x3727c5ac, v125 :: v_dual_mul_f32 v16, 0x3727c5ac, v126
.LBB5_27:                               ;   Parent Loop BB5_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e32 vcc_lo, s3, v83
	v_or_b32_e32 v112, 0x1400, v83
	v_or_b32_e32 v113, 0x1500, v83
	v_or_b32_e32 v114, 0x3200, v83
	v_or_b32_e32 v115, 0x3300, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, 0, v33, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v94
	v_or_b32_e32 v116, 0x3600, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v34, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v93
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v35, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v92
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v36, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v91
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v37, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v38, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v39, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v40, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v87
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v25, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v86
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v26, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v85
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v27, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v84
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v111, v111, v28, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v112
	v_or_b32_e32 v112, 0x1600, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v29, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v113
	v_or_b32_e32 v113, 0x1700, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v111, v111, v30, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v112
	v_or_b32_e32 v112, 0x2000, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v31, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v113
	v_or_b32_e32 v113, 0x2100, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v111, v111, v32, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v112
	v_or_b32_e32 v112, 0x2200, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v17, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v113
	v_or_b32_e32 v113, 0x2300, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v111, v111, v18, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v112
	v_or_b32_e32 v112, 0x2400, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v19, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v113
	v_or_b32_e32 v113, 0x2500, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v111, v111, v20, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v112
	v_or_b32_e32 v112, 0x2600, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v21, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v113
	v_or_b32_e32 v113, 0x2700, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v111, v111, v22, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v112
	v_or_b32_e32 v112, 0x3000, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v23, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v113
	v_or_b32_e32 v113, 0x3100, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v24, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v112
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v111, v111, v9 :: v_dual_add_nc_u32 v112, s3, v82
	v_cmp_eq_u32_e32 vcc_lo, s3, v113
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v113, v111, v10, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v114
	v_or_b32_e32 v114, 0x3400, v83
	ds_load_2addr_b32 v[111:112], v112 offset1:32
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v11, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v115
	v_or_b32_e32 v115, 0x3500, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v113, v113, v12, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v114
	v_lshlrev_b32_e32 v114, 2, v81
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v13, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v115
	v_or_b32_e32 v115, 0x3700, v83
	s_wait_dscnt 0x0
	ds_bpermute_b32 v117, v114, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v14, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v116
	ds_bpermute_b32 v118, v96, v111
	ds_bpermute_b32 v116, v97, v111
	ds_bpermute_b32 v119, v98, v111
	ds_bpermute_b32 v120, v99, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v15, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v115
	ds_bpermute_b32 v115, v100, v111
	ds_bpermute_b32 v121, v101, v111
	ds_bpermute_b32 v122, v102, v111
	ds_bpermute_b32 v123, v103, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v16, vcc_lo
	ds_bpermute_b32 v125, v104, v111
	ds_bpermute_b32 v126, v105, v111
	ds_bpermute_b32 v127, v106, v111
	ds_bpermute_b32 v128, v107, v111
	ds_bpermute_b32 v124, v95, v113
	ds_bpermute_b32 v129, v108, v111
	ds_bpermute_b32 v130, v109, v111
	ds_bpermute_b32 v111, v110, v111
	ds_bpermute_b32 v114, v114, v112
	ds_bpermute_b32 v131, v96, v112
	ds_bpermute_b32 v132, v97, v112
	ds_bpermute_b32 v133, v98, v112
	ds_bpermute_b32 v134, v99, v112
	ds_bpermute_b32 v135, v100, v112
	ds_bpermute_b32 v136, v101, v112
	ds_bpermute_b32 v137, v102, v112
	ds_bpermute_b32 v138, v103, v112
	ds_bpermute_b32 v139, v104, v112
	ds_bpermute_b32 v140, v105, v112
	ds_bpermute_b32 v141, v106, v112
	ds_bpermute_b32 v142, v108, v112
	ds_bpermute_b32 v143, v109, v112
	s_addk_co_i32 s3, 0x100
	s_wait_dscnt 0x11
	v_add_f32_e32 v113, v113, v124
	ds_bpermute_b32 v124, v107, v112
	ds_bpermute_b32 v112, v110, v112
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s3, 0x4000
	v_fma_f32 v33, -v113, v117, v33
	v_fma_f32 v34, -v113, v118, v34
	v_fma_f32 v35, -v113, v116, v35
	v_fma_f32 v36, -v113, v119, v36
	v_fma_f32 v37, -v113, v120, v37
	v_fma_f32 v38, -v113, v115, v38
	v_fma_f32 v39, -v113, v121, v39
	v_fma_f32 v40, -v113, v122, v40
	v_fma_f32 v25, -v113, v123, v25
	v_fma_f32 v26, -v113, v125, v26
	v_fma_f32 v27, -v113, v126, v27
	v_fma_f32 v28, -v113, v127, v28
	v_fma_f32 v29, -v113, v128, v29
	s_wait_dscnt 0x12
	v_fma_f32 v30, -v113, v129, v30
	s_wait_dscnt 0x11
	v_fma_f32 v31, -v113, v130, v31
	s_wait_dscnt 0x10
	v_fma_f32 v32, -v113, v111, v32
	s_wait_dscnt 0xf
	v_fma_f32 v17, -v113, v114, v17
	s_wait_dscnt 0xe
	v_fma_f32 v18, -v113, v131, v18
	s_wait_dscnt 0xd
	v_fma_f32 v19, -v113, v132, v19
	s_wait_dscnt 0xc
	v_fma_f32 v20, -v113, v133, v20
	s_wait_dscnt 0xb
	v_fma_f32 v21, -v113, v134, v21
	s_wait_dscnt 0xa
	v_fma_f32 v22, -v113, v135, v22
	s_wait_dscnt 0x9
	v_fma_f32 v23, -v113, v136, v23
	s_wait_dscnt 0x8
	v_fma_f32 v24, -v113, v137, v24
	s_wait_dscnt 0x7
	v_fma_f32 v9, -v113, v138, v9
	s_wait_dscnt 0x6
	v_fma_f32 v10, -v113, v139, v10
	s_wait_dscnt 0x5
	v_fma_f32 v11, -v113, v140, v11
	s_wait_dscnt 0x4
	v_fma_f32 v12, -v113, v141, v12
	s_wait_dscnt 0x1
	v_fma_f32 v13, -v113, v124, v13
	v_fma_f32 v14, -v113, v142, v14
	v_fma_f32 v15, -v113, v143, v15
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v113, v112, v16
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB5_27
; %bb.28:                               ;   in Loop: Header=BB5_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[111:118], v[77:80], v[57:60], v[1:8]
	v_cvt_f16_f32_e32 v40.h, v40
	v_cvt_f16_f32_e32 v40.l, v39
	v_cvt_f16_f32_e32 v39.h, v38
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[111:118], v[77:80], v[61:64], v[111:118]
	v_cvt_f16_f32_e32 v39.l, v37
	v_cvt_f16_f32_e32 v38.h, v36
	v_cvt_f16_f32_e32 v38.l, v35
	v_cvt_f16_f32_e32 v37.h, v34
	v_wmma_f32_16x16x16_f16 v[111:118], v[77:80], v[69:72], v[111:118]
	v_cvt_f16_f32_e32 v37.l, v33
	v_dual_add_f32 v126, v8, v8 :: v_dual_add_f32 v125, v7, v7
	v_dual_add_f32 v124, v6, v6 :: v_dual_add_f32 v123, v5, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[111:118], v[77:80], v[45:48], v[111:118]
	v_dual_add_f32 v122, v4, v4 :: v_dual_add_f32 v121, v3, v3
	v_dual_add_f32 v120, v2, v2 :: v_dual_add_f32 v119, v1, v1
	v_wmma_f32_16x16x16_f16 v[111:118], v[77:80], v[49:52], v[111:118]
	v_cvt_f16_f32_e32 v32.h, v32
	v_cvt_f16_f32_e32 v32.l, v31
	v_cvt_f16_f32_e32 v31.h, v30
	v_cvt_f16_f32_e32 v31.l, v29
	v_wmma_f32_16x16x16_f16 v[111:118], v[77:80], v[53:56], v[111:118]
	v_cvt_f16_f32_e32 v30.h, v28
	v_cvt_f16_f32_e32 v30.l, v27
	v_cvt_f16_f32_e32 v29.h, v26
	v_cvt_f16_f32_e32 v29.l, v25
	v_wmma_f32_16x16x16_f16 v[111:118], v[77:80], v[65:68], v[111:118]
	v_wmma_f32_16x16x16_f16 v[119:126], v[77:80], v[57:60], v[119:126]
	v_cvt_f16_f32_e32 v24.h, v24
	v_cvt_f16_f32_e32 v24.l, v23
	v_cvt_f16_f32_e32 v23.h, v22
	v_wmma_f32_16x16x16_f16 v[111:118], v[77:80], v[41:44], v[111:118]
	v_cvt_f16_f32_e32 v23.l, v21
	v_cvt_f16_f32_e32 v22.h, v20
	v_cvt_f16_f32_e32 v22.l, v19
	v_cvt_f16_f32_e32 v21.h, v18
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[37:40], v[111:118]
	v_cvt_f16_f32_e32 v21.l, v17
	v_wmma_f32_16x16x16_f16 v[119:126], v[77:80], v[61:64], v[119:126]
	v_cvt_f16_f32_e32 v16.h, v16
	v_cvt_f16_f32_e32 v16.l, v15
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[29:32], v[111:118]
	v_cvt_f16_f32_e32 v15.h, v14
	v_cvt_f16_f32_e32 v15.l, v13
	v_cvt_f16_f32_e32 v14.h, v12
	v_cvt_f16_f32_e32 v14.l, v11
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[21:24], v[111:118]
	v_cvt_f16_f32_e32 v13.h, v10
	v_wmma_f32_16x16x16_f16 v[119:126], v[77:80], v[69:72], v[119:126]
	v_cvt_f16_f32_e32 v13.l, v9
	v_dual_mul_f32 v134, 0x40400000, v8 :: v_dual_mul_f32 v133, 0x40400000, v7
	v_dual_mul_f32 v131, 0x40400000, v5 :: v_dual_mul_f32 v130, 0x40400000, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[77:80], v[45:48], v[119:126]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[13:16], v[111:118]
	v_dual_mul_f32 v132, 0x40400000, v6 :: v_dual_mul_f32 v129, 0x40400000, v3
	v_mul_f32_e32 v8, 4.0, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[77:80], v[49:52], v[119:126]
	v_dual_add_f32 v9, 0, v111 :: v_dual_mul_f32 v128, 0x40400000, v2
	v_dual_mul_f32 v127, 0x40400000, v1 :: v_dual_mul_f32 v6, 4.0, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[119:126], v[77:80], v[53:56], v[119:126]
	v_add_f32_e32 v9, v9, v112
	v_mul_f32_e32 v4, 4.0, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[127:134], v[77:80], v[57:60], v[127:134]
	v_dual_mul_f32 v7, 4.0, v7 :: v_dual_mul_f32 v2, 4.0, v2
	v_wmma_f32_16x16x16_f16 v[119:126], v[77:80], v[65:68], v[119:126]
	v_add_f32_e32 v9, v9, v113
	v_wmma_f32_16x16x16_f16 v[127:134], v[77:80], v[61:64], v[127:134]
	v_mul_f32_e32 v5, 4.0, v5
	v_mul_f32_e32 v3, 4.0, v3
	v_wmma_f32_16x16x16_f16 v[119:126], v[77:80], v[41:44], v[119:126]
	v_add_f32_e32 v9, v9, v114
	v_wmma_f32_16x16x16_f16 v[127:134], v[77:80], v[69:72], v[127:134]
	v_mul_f32_e32 v1, 4.0, v1
	v_fma_mix_f32 v114, v62, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[37:40], v[119:126]
	v_add_f32_e32 v9, v9, v115
	v_wmma_f32_16x16x16_f16 v[127:134], v[77:80], v[45:48], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[57:60], v[1:8]
	v_fma_mix_f32 v115, v63, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[29:32], v[119:126]
	v_add_f32_e32 v9, v9, v116
	v_wmma_f32_16x16x16_f16 v[127:134], v[77:80], v[49:52], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[61:64], v[1:8]
	v_fma_mix_f32 v116, v63, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[21:24], v[119:126]
	v_add_f32_e32 v9, v9, v117
	v_wmma_f32_16x16x16_f16 v[127:134], v[77:80], v[53:56], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[69:72], v[1:8]
	v_fma_mix_f32 v117, v64, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[13:16], v[119:126]
	v_add_f32_e32 v9, v9, v118
	v_wmma_f32_16x16x16_f16 v[127:134], v[77:80], v[65:68], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[45:48], v[1:8]
	v_fma_mix_f32 v118, v64, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v113, v62, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v119
	v_wmma_f32_16x16x16_f16 v[127:134], v[77:80], v[41:44], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[49:52], v[1:8]
	v_fma_mix_f32 v112, v61, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v111, v61, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v120
	v_wmma_f32_16x16x16_f16 v[127:134], v[73:76], v[37:40], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[53:56], v[1:8]
	v_fma_mix_f32 v120, v45, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[37:40], v[111:118]
	v_add_f32_e32 v9, v9, v121
	v_wmma_f32_16x16x16_f16 v[127:134], v[73:76], v[29:32], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[65:68], v[1:8]
	v_fma_mix_f32 v121, v46, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[29:32], v[111:118]
	v_add_f32_e32 v9, v9, v122
	v_wmma_f32_16x16x16_f16 v[127:134], v[73:76], v[21:24], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[41:44], v[1:8]
	v_fma_mix_f32 v122, v46, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[21:24], v[111:118]
	v_add_f32_e32 v9, v9, v123
	v_wmma_f32_16x16x16_f16 v[127:134], v[73:76], v[13:16], v[127:134]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[37:40], v[1:8]
	v_fma_mix_f32 v123, v47, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[13:16], v[111:118]
	v_add_f32_e32 v9, v9, v124
	v_fma_mix_f32 v124, v47, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[29:32], v[1:8]
	v_fma_mix_f32 v119, v45, s2, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f16_f32_e32 v61.h, v112
	v_add_f32_e32 v9, v9, v125
	v_fma_mix_f32 v125, v48, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[21:24], v[1:8]
	v_cvt_f16_f32_e32 v61.l, v111
	v_cvt_f16_f32_e32 v62.h, v114
	v_add_f32_e32 v9, v9, v126
	v_fma_mix_f32 v126, v48, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[13:16], v[1:8]
	v_cvt_f16_f32_e32 v62.l, v113
	v_cvt_f16_f32_e32 v63.h, v116
	v_add_f32_e32 v9, v9, v127
	v_cvt_f16_f32_e32 v63.l, v115
	v_cvt_f16_f32_e32 v64.h, v118
	v_cvt_f16_f32_e32 v64.l, v117
	v_fma_mix_f32 v118, v52, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v128
	v_fma_mix_f32 v117, v52, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v116, v51, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v115, v51, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v114, v50, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v129
	v_fma_mix_f32 v113, v50, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v112, v49, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v111, v49, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[37:40], v[119:126]
	v_add_f32_e32 v9, v9, v130
	v_fma_mix_f32 v142, v60, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v141, v60, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[37:40], v[111:118]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[29:32], v[119:126]
	v_add_f32_e32 v9, v9, v131
	v_fma_mix_f32 v140, v59, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v139, v59, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[29:32], v[111:118]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[21:24], v[119:126]
	v_add_f32_e32 v9, v9, v132
	v_fma_mix_f32 v138, v58, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v137, v58, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[21:24], v[111:118]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[13:16], v[119:126]
	v_add_f32_e32 v9, v9, v133
	v_fma_mix_f32 v136, v57, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v135, v57, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[13:16], v[111:118]
	v_fma_mix_f32 v150, v72, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v134
	v_fma_mix_f32 v149, v72, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v148, v71, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v147, v71, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v146, v70, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v9, v1
	v_fma_mix_f32 v145, v70, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v144, v69, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v143, v69, s2, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f16_f32_e32 v46.h, v122
	v_add_f32_e32 v1, v1, v2
	v_cvt_f16_f32_e32 v46.l, v121
	v_fma_mix_f32 v134, v56, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v133, v56, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v132, v55, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v1, v3
	v_fma_mix_f32 v131, v55, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v130, v54, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v129, v54, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v128, v53, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v1, v4
	v_fma_mix_f32 v127, v53, s2, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f16_f32_e32 v47.h, v124
	v_cvt_f16_f32_e32 v47.l, v123
	v_cvt_f16_f32_e32 v48.h, v126
	v_add_f32_e32 v1, v1, v5
	v_cvt_f16_f32_e32 v48.l, v125
	v_cvt_f16_f32_e32 v45.h, v120
	v_cvt_f16_f32_e32 v45.l, v119
	v_cvt_f16_f32_e32 v49.h, v112
	v_cvt_f16_f32_e32 v49.l, v111
	v_cvt_f16_f32_e32 v50.h, v114
	v_cvt_f16_f32_e32 v50.l, v113
	v_fma_mix_f32 v126, v68, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v125, v68, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v124, v67, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v123, v67, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v122, v66, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v121, v66, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v120, v65, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v119, v65, s2, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f16_f32_e32 v51.h, v116
	v_cvt_f16_f32_e32 v51.l, v115
	v_cvt_f16_f32_e32 v52.h, v118
	v_cvt_f16_f32_e32 v52.l, v117
	v_fma_mix_f32 v118, v44, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v117, v44, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v116, v43, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v115, v43, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v114, v42, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v113, v42, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v112, v41, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v111, v41, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v1, v6
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[37:40], v[135:142]
	v_wmma_f32_16x16x16_f16 v[143:150], v[73:76], v[37:40], v[143:150]
	v_wmma_f32_16x16x16_f16 v[127:134], v[73:76], v[37:40], v[127:134]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[37:40], v[119:126]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[37:40], v[111:118]
	v_add_f32_e32 v1, v1, v7
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[29:32], v[135:142]
	v_wmma_f32_16x16x16_f16 v[143:150], v[73:76], v[29:32], v[143:150]
	v_wmma_f32_16x16x16_f16 v[127:134], v[73:76], v[29:32], v[127:134]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[29:32], v[119:126]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[29:32], v[111:118]
	v_add_f32_e32 v1, v1, v8
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[21:24], v[135:142]
	v_wmma_f32_16x16x16_f16 v[143:150], v[73:76], v[21:24], v[143:150]
	v_wmma_f32_16x16x16_f16 v[127:134], v[73:76], v[21:24], v[127:134]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[21:24], v[119:126]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[21:24], v[111:118]
	v_mov_b32_e32 v2, 0x3b03126f
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v1
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[13:16], v[135:142]
	v_wmma_f32_16x16x16_f16 v[143:150], v[73:76], v[13:16], v[143:150]
	v_wmma_f32_16x16x16_f16 v[127:134], v[73:76], v[13:16], v[127:134]
	v_wmma_f32_16x16x16_f16 v[119:126], v[73:76], v[13:16], v[119:126]
	v_wmma_f32_16x16x16_f16 v[111:118], v[73:76], v[13:16], v[111:118]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x3a83126f, v2, vcc_lo
	v_cvt_f16_f32_e64 v57.h, v136
	v_cvt_f16_f32_e64 v57.l, v135
	v_cvt_f16_f32_e64 v58.h, v138
	v_cvt_f16_f32_e64 v58.l, v137
	v_cvt_f16_f32_e64 v59.h, v140
	v_cvt_f16_f32_e64 v59.l, v139
	v_cvt_f16_f32_e64 v60.h, v142
	v_cvt_f16_f32_e64 v60.l, v141
	v_cvt_f16_f32_e64 v69.h, v144
	v_cvt_f16_f32_e64 v69.l, v143
	v_cvt_f16_f32_e64 v70.h, v146
	v_cvt_f16_f32_e64 v70.l, v145
	v_cvt_f16_f32_e64 v71.h, v148
	v_cvt_f16_f32_e64 v71.l, v147
	v_cvt_f16_f32_e64 v72.h, v150
	v_cvt_f16_f32_e64 v72.l, v149
	v_cvt_f16_f32_e64 v53.h, v128
	v_cvt_f16_f32_e32 v53.l, v127
	v_cvt_f16_f32_e64 v54.h, v130
	v_cvt_f16_f32_e64 v54.l, v129
	v_cvt_f16_f32_e64 v55.h, v132
	v_cvt_f16_f32_e64 v55.l, v131
	v_cvt_f16_f32_e64 v56.h, v134
	v_cvt_f16_f32_e64 v56.l, v133
	v_cvt_f16_f32_e32 v65.h, v120
	v_cvt_f16_f32_e32 v65.l, v119
	v_cvt_f16_f32_e32 v66.h, v122
	v_cvt_f16_f32_e32 v66.l, v121
	v_cvt_f16_f32_e32 v67.h, v124
	v_cvt_f16_f32_e32 v67.l, v123
	v_cvt_f16_f32_e32 v68.h, v126
	v_cvt_f16_f32_e32 v68.l, v125
	v_cvt_f16_f32_e32 v42.h, v114
	v_cvt_f16_f32_e32 v42.l, v113
	v_cvt_f16_f32_e32 v43.h, v116
	v_cvt_f16_f32_e32 v43.l, v115
	v_cvt_f16_f32_e32 v44.h, v118
	v_cvt_f16_f32_e32 v44.l, v117
	v_cvt_f16_f32_e32 v41.h, v112
	v_cvt_f16_f32_e32 v41.l, v111
	v_cvt_f16_f32_e32 v77.l, v1
	s_add_co_i32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s0
	s_cbranch_scc0 .LBB5_26
	s_branch .LBB5_30
.LBB5_29:
	v_dual_mov_b32 v43, v56 :: v_dual_mov_b32 v42, v48
	v_dual_mov_b32 v41, v64 :: v_dual_mov_b32 v66, v72
	v_dual_mov_b32 v67, v52 :: v_dual_mov_b32 v54, v64
	v_dual_mov_b32 v65, v60 :: v_dual_mov_b32 v50, v60
	v_dual_mov_b32 v55, v48 :: v_dual_mov_b32 v46, v63
	v_dual_mov_b32 v53, v63 :: v_dual_mov_b32 v70, v59
	v_mov_b32_e32 v51, v72
	v_mov_b32_e32 v49, v59
	v_mov_b32_e32 v47, v64
	v_mov_b32_e32 v45, v62
	v_mov_b32_e32 v71, v60
	v_mov_b32_e32 v69, v58
.LBB5_30:
	v_fma_mix_f32 v1, v57, 1.0, 0 op_sel_hi:[1,1,0]
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v57, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v58, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v58, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v59, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v59, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v60, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v60, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v61, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v61, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v62, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v62, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v63, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v63, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v64, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v64, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v69, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v69, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v70, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v70, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v71, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v71, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v72, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v72, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v45, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v45, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v46, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v46, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v47, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v47, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v48, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v48, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v49, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v49, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v50, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v50, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v51, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v51, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v52, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v52, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v53, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v53, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v54, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v54, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v55, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v55, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v56, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v56, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v65, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v65, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v66, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v66, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v67, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v67, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v68, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v68, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v41, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v41, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v42, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v42, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v43, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v2, v43, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mov_b32_e32 v1, 0
	v_fma_mix_f32 v2, v44, 1.0, v2 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fma_mix_f32 v2, v44, 1.0, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end5:
	.size	_Z5probeILi8ELb1ELb1EEvPfPKfi, .Lfunc_end5-_Z5probeILi8ELb1ELb1EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi8ELb1ELb1EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 151
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-_Z5probeILi8ELb1ELb1EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi8ELb1ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi8ELb1ELb1EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.num_vgpr, 151
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.numbered_sgpr, 8
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7072
; TotalNumSgprs: 10
; NumVgprs: 151
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 18
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 151
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi4ELb0ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi4ELb0ELb1EEvPfPKfi,comdat
	.protected	_Z5probeILi4ELb0ELb1EEvPfPKfi ; -- Begin function _Z5probeILi4ELb0ELb1EEvPfPKfi
	.globl	_Z5probeILi4ELb0ELb1EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi4ELb0ELb1EEvPfPKfi,@function
_Z5probeILi4ELb0ELb1EEvPfPKfi:          ; @_Z5probeILi4ELb0ELb1EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_add_nc_u32_e32 v3, 1, v0
	s_load_b128 s[4:7], s[0:1], 0x0
	v_add_nc_u32_e32 v5, 2, v0
	v_add_nc_u32_e32 v6, 11, v0
	v_add_nc_u32_e32 v4, 10, v0
	v_and_b32_e32 v3, 0xff, v3
	v_add_nc_u32_e32 v2, 9, v0
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v6, 0xff, v6
	v_and_b32_e32 v4, 0xff, v4
	v_lshlrev_b32_e32 v7, 2, v3
	v_add_nc_u32_e32 v3, 3, v0
	v_add_nc_u32_e32 v8, 12, v0
	v_and_b32_e32 v2, 0xff, v2
	v_lshlrev_b32_e32 v10, 2, v5
	v_lshlrev_b32_e32 v12, 2, v6
	v_add_nc_u32_e32 v5, 4, v0
	v_add_nc_u32_e32 v6, 13, v0
	v_and_b32_e32 v3, 0xff, v3
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v9, 2, v4
	v_and_b32_e32 v4, 0xff, v8
	v_lshlrev_b32_e32 v2, 2, v2
	v_and_b32_e32 v15, 0xff, v5
	v_and_b32_e32 v16, 0xff, v6
	v_add_nc_u32_e32 v8, 5, v0
	v_lshlrev_b32_e32 v13, 2, v3
	v_lshlrev_b32_e32 v14, 2, v4
	s_wait_kmcnt 0x0
	s_clause 0x7
	global_load_b32 v11, v1, s[6:7]
	global_load_b32 v3, v2, s[6:7]
	global_load_b32 v4, v7, s[6:7]
	global_load_b32 v5, v9, s[6:7]
	global_load_b32 v6, v10, s[6:7]
	global_load_b32 v7, v12, s[6:7]
	global_load_b32 v9, v13, s[6:7]
	global_load_b32 v10, v14, s[6:7]
	v_add_nc_u32_e32 v13, 14, v0
	v_add_nc_u32_e32 v14, 6, v0
	v_lshlrev_b32_e32 v2, 2, v15
	v_lshlrev_b32_e32 v12, 2, v16
	v_add_nc_u32_e32 v15, 15, v0
	v_add_nc_u32_e32 v16, 7, v0
	v_add_nc_u32_e32 v17, 16, v0
	v_and_b32_e32 v8, 0xff, v8
	v_and_b32_e32 v13, 0xff, v13
	v_and_b32_e32 v14, 0xff, v14
	v_and_b32_e32 v15, 0xff, v15
	v_and_b32_e32 v16, 0xff, v16
	v_and_b32_e32 v17, 0xff, v17
	v_lshlrev_b32_e32 v8, 2, v8
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v18, 2, v15
	v_lshlrev_b32_e32 v16, 2, v16
	v_lshlrev_b32_e32 v17, 2, v17
	s_clause 0x7
	global_load_b32 v27, v2, s[6:7]
	global_load_b32 v28, v12, s[6:7]
	global_load_b32 v15, v8, s[6:7]
	global_load_b32 v25, v13, s[6:7]
	global_load_b32 v26, v14, s[6:7]
	global_load_b32 v13, v18, s[6:7]
	global_load_b32 v14, v16, s[6:7]
	global_load_b32 v12, v17, s[6:7]
	v_mov_b32_e32 v16, 0
	v_add_co_u32 v1, s2, s6, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s7, 0, s2
	v_and_b32_e32 v8, 63, v0
	s_mov_b32 s2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_6
; %bb.1:
	v_lshl_add_u32 v17, v0, 2, 0
	v_lshrrev_b32_e32 v18, 6, v0
	s_mov_b32 s6, 0
	s_branch .LBB6_3
.LBB6_2:                                ;   in Loop: Header=BB6_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v16, -1, v16
	ds_store_b32 v17, v19
	v_add_nc_u32_e32 v17, 0x400, v17
	v_add_nc_u32_e32 v18, 4, v18
	v_cmp_eq_u32_e32 vcc_lo, 0, v16
	s_or_b32 s6, vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB6_5
.LBB6_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v19, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_gt_u32_e64 v8, v18
	s_cbranch_execz .LBB6_2
; %bb.4:                                ;   in Loop: Header=BB6_3 Depth=1
	global_load_b32 v19, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v19, 0x38d1b717, v19
	s_branch .LBB6_2
.LBB6_5:
	s_or_b32 exec_lo, exec_lo, s6
.LBB6_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v16, v0
	s_branch .LBB6_8
.LBB6_7:                                ;   in Loop: Header=BB6_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v18, 0x800, v16
	v_cmp_lt_u32_e32 vcc_lo, 0x7ff, v16
	ds_store_b32 v17, v19 offset:7168
	v_mov_b32_e32 v16, v18
	s_or_b32 s2, vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB6_24
.LBB6_8:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v18, 6, v16
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e64 v8, v18
	s_cbranch_execz .LBB6_10
; %bb.9:                                ;   in Loop: Header=BB6_8 Depth=1
	global_load_b32 v17, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v20, 0x38d1b717, v17
.LBB6_10:                               ;   in Loop: Header=BB6_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v21, 4, v18
	v_lshl_add_u32 v17, v16, 2, 0
	s_mov_b32 s3, exec_lo
	ds_store_b32 v17, v20
	v_cmpx_gt_u32_e64 v8, v21
	s_cbranch_execz .LBB6_12
; %bb.11:                               ;   in Loop: Header=BB6_8 Depth=1
	global_load_b32 v19, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v19, 0x38d1b717, v19
.LBB6_12:                               ;   in Loop: Header=BB6_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v20, 0 :: v_dual_add_nc_u32 v21, 8, v18
	ds_store_b32 v17, v19 offset:1024
	v_cmp_gt_u32_e32 vcc_lo, v8, v21
	v_mov_b32_e32 v21, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB6_14
; %bb.13:                               ;   in Loop: Header=BB6_8 Depth=1
	global_load_b32 v19, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v21, 0x38d1b717, v19
.LBB6_14:                               ;   in Loop: Header=BB6_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v19, 12, v18
	s_mov_b32 s3, exec_lo
	ds_store_b32 v17, v21 offset:2048
	v_cmpx_gt_u32_e64 v8, v19
	s_cbranch_execz .LBB6_16
; %bb.15:                               ;   in Loop: Header=BB6_8 Depth=1
	global_load_b32 v19, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v20, 0x38d1b717, v19
.LBB6_16:                               ;   in Loop: Header=BB6_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v19, 16, v18
	v_mov_b32_e32 v21, 0
	ds_store_b32 v17, v20 offset:3072
	v_cmp_gt_u32_e32 vcc_lo, v8, v19
	v_mov_b32_e32 v19, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB6_18
; %bb.17:                               ;   in Loop: Header=BB6_8 Depth=1
	global_load_b32 v19, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v19, 0x38d1b717, v19
.LBB6_18:                               ;   in Loop: Header=BB6_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v20, 20, v18
	s_mov_b32 s3, exec_lo
	ds_store_b32 v17, v19 offset:4096
	v_cmpx_gt_u32_e64 v8, v20
	s_cbranch_execz .LBB6_20
; %bb.19:                               ;   in Loop: Header=BB6_8 Depth=1
	global_load_b32 v19, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v21, 0x38d1b717, v19
.LBB6_20:                               ;   in Loop: Header=BB6_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v20, 24, v18
	ds_store_b32 v17, v21 offset:5120
	v_cmp_gt_u32_e32 vcc_lo, v8, v20
	v_mov_b32_e32 v20, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB6_22
; %bb.21:                               ;   in Loop: Header=BB6_8 Depth=1
	global_load_b32 v20, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v20, 0x38d1b717, v20
.LBB6_22:                               ;   in Loop: Header=BB6_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v18, 28, v18
	s_mov_b32 s3, exec_lo
	ds_store_b32 v17, v20 offset:6144
	v_cmpx_gt_u32_e64 v8, v18
	s_cbranch_execz .LBB6_7
; %bb.23:                               ;   in Loop: Header=BB6_8 Depth=1
	global_load_b32 v18, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v19, 0x38d1b717, v18
	s_branch .LBB6_7
.LBB6_24:
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s1, 0x3c23d70a
	v_fma_f32 v17, 0x3c23d70a, v11, 0
	v_fmaak_f32 v18, s1, v11, 0x38d1b717
	v_fmaak_f32 v19, s1, v11, 0x3951b717
	v_fmaak_f32 v20, s1, v11, 0x399d4951
	v_fmaak_f32 v21, s1, v11, 0x39d1b717
	v_fmaak_f32 v22, s1, v11, 0x3a03126e
	v_fmaak_f32 v23, s1, v11, 0x3a1d4951
	v_fmaak_f32 v24, s1, v11, 0x3a378034
	v_fmaak_f32 v32, s1, v11, 0x3a51b717
	v_fmaak_f32 v16, s1, v11, 0x3a6bedfa
	v_fmaak_f32 v8, s1, v11, 0x3a83126e
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB6_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s1, 0x3a83126f
	v_and_b32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v77, v3, s1, 0
	v_fma_mixhi_f16 v73, v4, s1, 0
	v_fma_mixhi_f16 v77, v5, s1, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixhi_f16 v78, v10, s1, 0
	v_fma_mixhi_f16 v79, v25, s1, 0
	v_fma_mixlo_f16 v76, v26, s1, 0
	v_lshlrev_b32_e32 v10, 7, v0
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_and_b32_e32 v97, 8, v3
	v_fma_mixlo_f16 v73, v11, s1, 0
	v_fma_mixlo_f16 v74, v6, s1, 0
	v_fma_mixlo_f16 v78, v7, s1, 0
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	v_or_b32_e32 v25, 22, v97
	v_or_b32_e32 v26, 23, v97
	v_fma_mixhi_f16 v74, v9, s1, 0
	v_fma_mixhi_f16 v75, v15, s1, 0
	v_fma_mixlo_f16 v80, v13, s1, 0
	v_fma_mixhi_f16 v76, v14, s1, 0
	v_fma_mixhi_f16 v80, v12, s1, 0
	v_or_b32_e32 v3, 2, v97
	v_or_b32_e32 v4, 3, v97
	v_or_b32_e32 v5, 4, v97
	v_or_b32_e32 v6, 5, v97
	v_or_b32_e32 v7, 6, v97
	v_or_b32_e32 v9, 7, v97
	v_or_b32_e32 v11, 16, v97
	v_or_b32_e32 v12, 17, v97
	v_or_b32_e32 v13, 18, v97
	v_or_b32_e32 v14, 19, v97
	v_or_b32_e32 v15, 21, v97
	v_lshlrev_b32_e32 v133, 2, v25
	v_dual_mov_b32 v25, v18 :: v_dual_lshlrev_b32 v134, 2, v26
	v_and_b32_e32 v99, 0x800, v10
	v_or_b32_e32 v10, 20, v97
	v_mov_b32_e32 v26, v19
	v_lshl_add_u32 v98, v1, 2, 0
	v_or_b32_e32 v1, 1, v97
	v_fma_mixlo_f16 v75, v27, s1, 0
	v_fma_mixlo_f16 v79, v28, s1, 0
	v_dual_mov_b32 v30, v23 :: v_dual_lshlrev_b32 v119, 2, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v27, v20 :: v_dual_lshlrev_b32 v120, 2, v1
	v_lshlrev_b32_e32 v121, 2, v3
	v_dual_mov_b32 v29, v22 :: v_dual_lshlrev_b32 v122, 2, v4
	v_lshlrev_b32_e32 v123, 2, v5
	v_dual_mov_b32 v31, v24 :: v_dual_lshlrev_b32 v124, 2, v6
	v_lshlrev_b32_e32 v125, 2, v7
	v_dual_mov_b32 v9, v19 :: v_dual_lshlrev_b32 v126, 2, v9
	v_dual_mov_b32 v2, v21 :: v_dual_lshlrev_b32 v127, 2, v11
	v_dual_mov_b32 v11, v21 :: v_dual_lshlrev_b32 v128, 2, v12
	v_dual_mov_b32 v4, v23 :: v_dual_lshlrev_b32 v129, 2, v13
	v_dual_mov_b32 v13, v23 :: v_dual_lshlrev_b32 v130, 2, v14
	v_dual_mov_b32 v6, v32 :: v_dual_lshlrev_b32 v131, 2, v10
	v_dual_mov_b32 v15, v32 :: v_dual_lshlrev_b32 v132, 2, v15
	v_mov_b32_e32 v28, v21
	v_or_b32_e32 v100, 0x2300, v99
	v_or_b32_e32 v101, 0x2200, v99
	v_or_b32_e32 v102, 0x2100, v99
	v_or_b32_e32 v103, 0x2000, v99
	v_or_b32_e32 v104, 0x1700, v99
	v_or_b32_e32 v105, 0x1600, v99
	v_or_b32_e32 v106, 0x1500, v99
	v_or_b32_e32 v107, 0x1400, v99
	v_or_b32_e32 v108, 0x1300, v99
	v_or_b32_e32 v109, 0x1200, v99
	v_or_b32_e32 v110, 0x1100, v99
	v_or_b32_e32 v111, 0x1000, v99
	v_or_b32_e32 v112, 0x700, v99
	v_or_b32_e32 v113, 0x600, v99
	v_or_b32_e32 v114, 0x500, v99
	v_or_b32_e32 v115, 0x400, v99
	v_or_b32_e32 v116, 0x300, v99
	v_or_b32_e32 v117, 0x200, v99
	v_or_b32_e32 v118, 0x100, v99
	v_mov_b32_e32 v10, v20
	v_mov_b32_e32 v12, v22
	v_dual_mov_b32 v14, v24 :: v_dual_mov_b32 v1, v20
	v_mov_b32_e32 v3, v22
	v_mov_b32_e32 v5, v24
	v_mov_b32_e32 v7, v16
	s_mov_b32 s1, 0
.LBB6_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_27 Depth 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], 0
	v_cvt_f16_f32_e32 v84.h, v24
	v_cvt_f16_f32_e32 v84.l, v23
	v_cvt_f16_f32_e32 v83.h, v22
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_cvt_f16_f32_e32 v83.l, v21
	v_cvt_f16_f32_e32 v82.h, v20
	v_cvt_f16_f32_e32 v82.l, v19
	v_cvt_f16_f32_e32 v81.h, v18
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_cvt_f16_f32_e32 v81.l, v17
	v_cvt_f16_f32_e32 v88.h, v32
	v_cvt_f16_f32_e32 v88.l, v31
	v_cvt_f16_f32_e32 v87.h, v30
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_cvt_f16_f32_e32 v87.l, v29
	v_cvt_f16_f32_e32 v86.h, v28
	v_cvt_f16_f32_e32 v86.l, v27
	v_cvt_f16_f32_e32 v85.h, v26
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_cvt_f16_f32_e32 v85.l, v25
	v_cvt_f16_f32_e32 v92.h, v16
	v_cvt_f16_f32_e32 v92.l, v15
	v_cvt_f16_f32_e32 v91.h, v14
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_cvt_f16_f32_e32 v91.l, v13
	v_cvt_f16_f32_e32 v90.h, v12
	v_cvt_f16_f32_e32 v90.l, v11
	v_cvt_f16_f32_e32 v89.h, v10
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_cvt_f16_f32_e32 v89.l, v9
	v_cvt_f16_f32_e32 v96.h, v8
	v_cvt_f16_f32_e32 v96.l, v7
	v_cvt_f16_f32_e32 v95.h, v6
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_cvt_f16_f32_e32 v95.l, v5
	v_cvt_f16_f32_e32 v94.h, v4
	v_cvt_f16_f32_e32 v94.l, v3
	v_cvt_f16_f32_e32 v93.h, v2
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_cvt_f16_f32_e32 v93.l, v1
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[77:80], v[33:40]
	v_dual_add_f32 v48, 0x3727c5ac, v40 :: v_dual_add_f32 v47, 0x3727c5ac, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v46, 0x3727c5ac, v38 :: v_dual_add_f32 v45, 0x3727c5ac, v37
	v_dual_add_f32 v44, 0x3727c5ac, v36 :: v_dual_add_f32 v43, 0x3727c5ac, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v42, 0x3727c5ac, v34 :: v_dual_add_f32 v41, 0x3727c5ac, v33
	v_dual_add_f32 v56, 0x37a7c5ac, v40 :: v_dual_add_f32 v55, 0x37a7c5ac, v39
	v_dual_add_f32 v54, 0x37a7c5ac, v38 :: v_dual_add_f32 v53, 0x37a7c5ac, v37
	v_dual_add_f32 v52, 0x37a7c5ac, v36 :: v_dual_add_f32 v51, 0x37a7c5ac, v35
	v_dual_add_f32 v50, 0x37a7c5ac, v34 :: v_dual_add_f32 v49, 0x37a7c5ac, v33
	v_dual_add_f32 v142, 0x37fba882, v40 :: v_dual_add_f32 v141, 0x37fba882, v39
	v_dual_add_f32 v140, 0x37fba882, v38 :: v_dual_add_f32 v139, 0x37fba882, v37
	v_dual_add_f32 v138, 0x37fba882, v36 :: v_dual_add_f32 v137, 0x37fba882, v35
	v_dual_add_f32 v136, 0x37fba882, v34 :: v_dual_add_f32 v135, 0x37fba882, v33
	v_dual_add_f32 v150, 0x3827c5ac, v40 :: v_dual_add_f32 v149, 0x3827c5ac, v39
	v_dual_add_f32 v148, 0x3827c5ac, v38 :: v_dual_add_f32 v147, 0x3827c5ac, v37
	v_dual_add_f32 v146, 0x3827c5ac, v36 :: v_dual_add_f32 v145, 0x3827c5ac, v35
	v_dual_add_f32 v144, 0x3827c5ac, v34 :: v_dual_add_f32 v143, 0x3827c5ac, v33
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[81:84], v[41:48]
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[81:84], v[49:56]
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[81:84], v[135:142]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[143:150], v[73:76], v[81:84], v[143:150]
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[85:88], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[85:88], v[49:56]
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[85:88], v[135:142]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[143:150], v[73:76], v[85:88], v[143:150]
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[89:92], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[89:92], v[49:56]
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[89:92], v[135:142]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[143:150], v[73:76], v[89:92], v[143:150]
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[93:96], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[93:96], v[49:56]
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[93:96], v[135:142]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[143:150], v[73:76], v[93:96], v[143:150]
	v_dual_mul_f32 v65, 0x3727c5ac, v41 :: v_dual_mul_f32 v66, 0x3727c5ac, v42
	v_dual_mul_f32 v67, 0x3727c5ac, v43 :: v_dual_mul_f32 v68, 0x3727c5ac, v44
	v_dual_mul_f32 v69, 0x3727c5ac, v45 :: v_dual_mul_f32 v70, 0x3727c5ac, v46
	v_dual_mul_f32 v71, 0x3727c5ac, v47 :: v_dual_mul_f32 v72, 0x3727c5ac, v48
	v_dual_mul_f32 v57, 0x3727c5ac, v49 :: v_dual_mul_f32 v58, 0x3727c5ac, v50
	v_dual_mul_f32 v59, 0x3727c5ac, v51 :: v_dual_mul_f32 v60, 0x3727c5ac, v52
	v_dual_mul_f32 v61, 0x3727c5ac, v53 :: v_dual_mul_f32 v62, 0x3727c5ac, v54
	v_dual_mul_f32 v63, 0x3727c5ac, v55 :: v_dual_mul_f32 v64, 0x3727c5ac, v56
	v_dual_mul_f32 v49, 0x3727c5ac, v135 :: v_dual_mul_f32 v50, 0x3727c5ac, v136
	v_dual_mul_f32 v51, 0x3727c5ac, v137 :: v_dual_mul_f32 v52, 0x3727c5ac, v138
	v_dual_mul_f32 v53, 0x3727c5ac, v139 :: v_dual_mul_f32 v54, 0x3727c5ac, v140
	v_dual_mul_f32 v55, 0x3727c5ac, v141 :: v_dual_mul_f32 v56, 0x3727c5ac, v142
	v_dual_mul_f32 v41, 0x3727c5ac, v143 :: v_dual_mul_f32 v42, 0x3727c5ac, v144
	v_dual_mul_f32 v43, 0x3727c5ac, v145 :: v_dual_mul_f32 v44, 0x3727c5ac, v146
	v_dual_mul_f32 v45, 0x3727c5ac, v147 :: v_dual_mul_f32 v46, 0x3727c5ac, v148
	v_dual_mul_f32 v47, 0x3727c5ac, v149 :: v_dual_mul_f32 v48, 0x3727c5ac, v150
.LBB6_27:                               ;   Parent Loop BB6_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e32 vcc_lo, s2, v99
	v_or_b32_e32 v136, 0x2400, v99
	v_or_b32_e32 v137, 0x2500, v99
	v_or_b32_e32 v138, 0x3200, v99
	v_or_b32_e32 v139, 0x3300, v99
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, 0, v65, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v118
	v_or_b32_e32 v140, 0x3600, v99
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v66, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v117
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v67, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v116
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v68, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v115
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v69, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v114
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v70, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v113
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v71, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v112
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v72, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v57, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v110
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v58, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v109
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v59, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v108
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v60, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v107
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v61, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v106
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v62, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v105
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v63, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v104
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v64, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v103
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v49, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v102
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v50, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v101
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v51, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v100
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v135, v135, v52, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v136
	v_or_b32_e32 v136, 0x2600, v99
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v53, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v137
	v_or_b32_e32 v137, 0x2700, v99
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v135, v135, v54, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v136
	v_or_b32_e32 v136, 0x3000, v99
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v135, v135, v55, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v137
	v_or_b32_e32 v137, 0x3100, v99
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v135, v135, v56, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v136
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v135, v135, v41 :: v_dual_add_nc_u32 v136, s2, v98
	v_cmp_eq_u32_e32 vcc_lo, s2, v137
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v137, v135, v42, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v138
	v_or_b32_e32 v138, 0x3400, v99
	ds_load_2addr_b32 v[135:136], v136 offset1:32
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v137, v137, v43, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v139
	v_or_b32_e32 v139, 0x3500, v99
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v137, v137, v44, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v138
	v_lshlrev_b32_e32 v138, 2, v97
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v137, v137, v45, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v139
	v_or_b32_e32 v139, 0x3700, v99
	s_wait_dscnt 0x0
	ds_bpermute_b32 v141, v138, v135
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v137, v137, v46, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v140
	ds_bpermute_b32 v142, v120, v135
	ds_bpermute_b32 v140, v121, v135
	ds_bpermute_b32 v143, v122, v135
	ds_bpermute_b32 v144, v123, v135
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v137, v137, v47, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v139
	ds_bpermute_b32 v139, v124, v135
	ds_bpermute_b32 v145, v125, v135
	ds_bpermute_b32 v146, v126, v135
	ds_bpermute_b32 v147, v127, v135
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v137, v137, v48, vcc_lo
	ds_bpermute_b32 v149, v128, v135
	ds_bpermute_b32 v150, v129, v135
	ds_bpermute_b32 v151, v130, v135
	ds_bpermute_b32 v152, v131, v135
	ds_bpermute_b32 v148, v119, v137
	ds_bpermute_b32 v153, v132, v135
	ds_bpermute_b32 v154, v133, v135
	ds_bpermute_b32 v135, v134, v135
	ds_bpermute_b32 v138, v138, v136
	ds_bpermute_b32 v155, v120, v136
	ds_bpermute_b32 v156, v121, v136
	ds_bpermute_b32 v157, v122, v136
	ds_bpermute_b32 v158, v123, v136
	ds_bpermute_b32 v159, v124, v136
	ds_bpermute_b32 v160, v125, v136
	ds_bpermute_b32 v161, v126, v136
	ds_bpermute_b32 v162, v127, v136
	ds_bpermute_b32 v163, v128, v136
	ds_bpermute_b32 v164, v129, v136
	ds_bpermute_b32 v165, v130, v136
	ds_bpermute_b32 v166, v132, v136
	ds_bpermute_b32 v167, v133, v136
	s_addk_co_i32 s2, 0x100
	s_wait_dscnt 0x11
	v_add_f32_e32 v137, v137, v148
	ds_bpermute_b32 v148, v131, v136
	ds_bpermute_b32 v136, v134, v136
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, 0x4000
	v_fma_f32 v65, -v137, v141, v65
	v_fma_f32 v66, -v137, v142, v66
	v_fma_f32 v67, -v137, v140, v67
	v_fma_f32 v68, -v137, v143, v68
	v_fma_f32 v69, -v137, v144, v69
	v_fma_f32 v70, -v137, v139, v70
	v_fma_f32 v71, -v137, v145, v71
	v_fma_f32 v72, -v137, v146, v72
	v_fma_f32 v57, -v137, v147, v57
	v_fma_f32 v58, -v137, v149, v58
	v_fma_f32 v59, -v137, v150, v59
	v_fma_f32 v60, -v137, v151, v60
	v_fma_f32 v61, -v137, v152, v61
	s_wait_dscnt 0x12
	v_fma_f32 v62, -v137, v153, v62
	s_wait_dscnt 0x11
	v_fma_f32 v63, -v137, v154, v63
	s_wait_dscnt 0x10
	v_fma_f32 v64, -v137, v135, v64
	s_wait_dscnt 0xf
	v_fma_f32 v49, -v137, v138, v49
	s_wait_dscnt 0xe
	v_fma_f32 v50, -v137, v155, v50
	s_wait_dscnt 0xd
	v_fma_f32 v51, -v137, v156, v51
	s_wait_dscnt 0xc
	v_fma_f32 v52, -v137, v157, v52
	s_wait_dscnt 0xb
	v_fma_f32 v53, -v137, v158, v53
	s_wait_dscnt 0xa
	v_fma_f32 v54, -v137, v159, v54
	s_wait_dscnt 0x9
	v_fma_f32 v55, -v137, v160, v55
	s_wait_dscnt 0x8
	v_fma_f32 v56, -v137, v161, v56
	s_wait_dscnt 0x7
	v_fma_f32 v41, -v137, v162, v41
	s_wait_dscnt 0x6
	v_fma_f32 v42, -v137, v163, v42
	s_wait_dscnt 0x5
	v_fma_f32 v43, -v137, v164, v43
	s_wait_dscnt 0x4
	v_fma_f32 v44, -v137, v165, v44
	s_wait_dscnt 0x1
	v_fma_f32 v45, -v137, v148, v45
	v_fma_f32 v46, -v137, v166, v46
	v_fma_f32 v47, -v137, v167, v47
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v137, v136, v48
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB6_27
; %bb.28:                               ;   in Loop: Header=BB6_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[135:142], v[77:80], v[81:84], v[33:40]
	v_cvt_f16_f32_e32 v72.h, v72
	v_cvt_f16_f32_e32 v72.l, v71
	v_cvt_f16_f32_e32 v71.h, v70
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[135:142], v[77:80], v[85:88], v[135:142]
	v_cvt_f16_f32_e32 v71.l, v69
	v_cvt_f16_f32_e32 v70.h, v68
	v_cvt_f16_f32_e32 v70.l, v67
	v_cvt_f16_f32_e32 v69.h, v66
	v_wmma_f32_16x16x16_f16 v[135:142], v[77:80], v[89:92], v[135:142]
	v_cvt_f16_f32_e32 v69.l, v65
	v_cvt_f16_f32_e32 v64.h, v64
	v_cvt_f16_f32_e32 v64.l, v63
	v_cvt_f16_f32_e32 v63.h, v62
	v_wmma_f32_16x16x16_f16 v[135:142], v[77:80], v[93:96], v[135:142]
	v_cvt_f16_f32_e32 v63.l, v61
	v_cvt_f16_f32_e32 v62.h, v60
	v_cvt_f16_f32_e32 v62.l, v59
	v_cvt_f16_f32_e32 v61.h, v58
	v_cvt_f16_f32_e32 v61.l, v57
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[69:72], v[135:142]
	v_cvt_f16_f32_e32 v60.h, v56
	v_cvt_f16_f32_e32 v60.l, v55
	v_cvt_f16_f32_e32 v59.h, v54
	v_cvt_f16_f32_e32 v59.l, v53
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[61:64], v[135:142]
	v_cvt_f16_f32_e32 v58.h, v52
	v_cvt_f16_f32_e32 v58.l, v51
	v_cvt_f16_f32_e32 v57.h, v50
	v_cvt_f16_f32_e32 v57.l, v49
	v_cvt_f16_f32_e32 v68.h, v48
	v_cvt_f16_f32_e32 v68.l, v47
	v_cvt_f16_f32_e32 v67.h, v46
	v_cvt_f16_f32_e32 v67.l, v45
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[57:60], v[135:142]
	v_cvt_f16_f32_e32 v66.h, v44
	v_cvt_f16_f32_e32 v66.l, v43
	v_cvt_f16_f32_e32 v65.h, v42
	v_cvt_f16_f32_e32 v65.l, v41
	v_dual_add_f32 v48, v40, v40 :: v_dual_add_f32 v47, v39, v39
	v_dual_add_f32 v46, v38, v38 :: v_dual_add_f32 v45, v37, v37
	v_dual_add_f32 v44, v36, v36 :: v_dual_add_f32 v43, v35, v35
	v_dual_add_f32 v42, v34, v34 :: v_dual_add_f32 v41, v33, v33
	v_wmma_f32_16x16x16_f16 v[135:142], v[73:76], v[65:68], v[135:142]
	v_mul_f32_e32 v56, 0x40400000, v40
	v_mul_f32_e32 v54, 0x40400000, v38
	v_mul_f32_e32 v52, 0x40400000, v36
	v_wmma_f32_16x16x16_f16 v[41:48], v[77:80], v[81:84], v[41:48]
	v_add_f32_e32 v49, 0, v135
	v_dual_mul_f32 v55, 0x40400000, v39 :: v_dual_mul_f32 v38, 4.0, v38
	v_mul_f32_e32 v40, 4.0, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[77:80], v[85:88], v[41:48]
	v_add_f32_e32 v49, v49, v136
	v_dual_mul_f32 v53, 0x40400000, v37 :: v_dual_mul_f32 v36, 4.0, v36
	v_mul_f32_e32 v50, 0x40400000, v34
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[77:80], v[89:92], v[41:48]
	v_add_f32_e32 v49, v49, v137
	v_dual_mul_f32 v51, 0x40400000, v35 :: v_dual_mul_f32 v34, 4.0, v34
	v_dual_mul_f32 v39, 4.0, v39 :: v_dual_mul_f32 v26, 0x3f7d70a4, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[77:80], v[93:96], v[41:48]
	v_dual_add_f32 v49, v49, v138 :: v_dual_mul_f32 v24, 0x3f7d70a4, v24
	v_dual_mul_f32 v37, 4.0, v37 :: v_dual_mul_f32 v14, 0x3f7d70a4, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[69:72], v[41:48]
	v_dual_add_f32 v49, v49, v139 :: v_dual_mul_f32 v20, 0x3f7d70a4, v20
	v_dual_mul_f32 v35, 4.0, v35 :: v_dual_mul_f32 v10, 0x3f7d70a4, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[61:64], v[41:48]
	v_dual_add_f32 v49, v49, v140 :: v_dual_mul_f32 v22, 0x3f7d70a4, v22
	v_dual_mul_f32 v8, 0x3f7d70a4, v8 :: v_dual_mul_f32 v23, 0x3f7d70a4, v23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[57:60], v[41:48]
	v_add_f32_e32 v135, v49, v141
	v_mul_f32_e32 v49, 0x40400000, v33
	v_dual_mul_f32 v32, 0x3f7d70a4, v32 :: v_dual_mul_f32 v33, 4.0, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[65:68], v[41:48]
	v_add_f32_e32 v135, v135, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[49:56], v[77:80], v[81:84], v[49:56]
	v_mul_f32_e32 v18, 0x3f7d70a4, v18
	v_wmma_f32_16x16x16_f16 v[33:40], v[77:80], v[81:84], v[33:40]
	v_mul_f32_e32 v21, 0x3f7d70a4, v21
	v_add_f32_e32 v41, v135, v41
	v_wmma_f32_16x16x16_f16 v[49:56], v[77:80], v[85:88], v[49:56]
	v_mul_f32_e32 v30, 0x3f7d70a4, v30
	v_wmma_f32_16x16x16_f16 v[33:40], v[77:80], v[85:88], v[33:40]
	v_mul_f32_e32 v19, 0x3f7d70a4, v19
	v_add_f32_e32 v41, v41, v42
	v_wmma_f32_16x16x16_f16 v[49:56], v[77:80], v[89:92], v[49:56]
	v_mul_f32_e32 v28, 0x3f7d70a4, v28
	v_wmma_f32_16x16x16_f16 v[33:40], v[77:80], v[89:92], v[33:40]
	v_mul_f32_e32 v17, 0x3f7d70a4, v17
	v_add_f32_e32 v41, v41, v43
	v_wmma_f32_16x16x16_f16 v[49:56], v[77:80], v[93:96], v[49:56]
	v_mul_f32_e32 v16, 0x3f7d70a4, v16
	v_wmma_f32_16x16x16_f16 v[33:40], v[77:80], v[93:96], v[33:40]
	v_mul_f32_e32 v31, 0x3f7d70a4, v31
	v_add_f32_e32 v41, v41, v44
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[69:72], v[49:56]
	v_mul_f32_e32 v12, 0x3f7d70a4, v12
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[69:72], v[33:40]
	v_mul_f32_e32 v29, 0x3f7d70a4, v29
	v_add_f32_e32 v41, v41, v45
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[61:64], v[49:56]
	v_mul_f32_e32 v6, 0x3f7d70a4, v6
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[61:64], v[33:40]
	v_mul_f32_e32 v27, 0x3f7d70a4, v27
	v_add_f32_e32 v41, v41, v46
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[57:60], v[49:56]
	v_mul_f32_e32 v4, 0x3f7d70a4, v4
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[57:60], v[33:40]
	v_mul_f32_e32 v25, 0x3f7d70a4, v25
	v_add_f32_e32 v41, v41, v47
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[65:68], v[49:56]
	v_mul_f32_e32 v15, 0x3f7d70a4, v15
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[65:68], v[33:40]
	v_mul_f32_e32 v13, 0x3f7d70a4, v13
	v_dual_add_f32 v41, v41, v48 :: v_dual_mul_f32 v2, 0x3f7d70a4, v2
	v_mul_f32_e32 v11, 0x3f7d70a4, v11
	v_mul_f32_e32 v9, 0x3f7d70a4, v9
	v_mul_f32_e32 v7, 0x3f7d70a4, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v41, v41, v49
	v_mul_f32_e32 v5, 0x3f7d70a4, v5
	v_mul_f32_e32 v3, 0x3f7d70a4, v3
	v_mul_f32_e32 v1, 0x3f7d70a4, v1
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[69:72], v[17:24]
	v_add_f32_e32 v41, v41, v50
	v_wmma_f32_16x16x16_f16 v[25:32], v[73:76], v[69:72], v[25:32]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[69:72], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[69:72], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[61:64], v[17:24]
	v_add_f32_e32 v41, v41, v51
	v_wmma_f32_16x16x16_f16 v[25:32], v[73:76], v[61:64], v[25:32]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[61:64], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[61:64], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[57:60], v[17:24]
	v_add_f32_e32 v41, v41, v52
	v_wmma_f32_16x16x16_f16 v[25:32], v[73:76], v[57:60], v[25:32]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[57:60], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[57:60], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[65:68], v[17:24]
	v_add_f32_e32 v41, v41, v53
	v_wmma_f32_16x16x16_f16 v[25:32], v[73:76], v[65:68], v[25:32]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[65:68], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[65:68], v[1:8]
	s_add_co_i32 s1, s1, 1
	v_add_f32_e32 v41, v41, v54
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v55
	v_add_f32_e32 v41, v41, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v41, v33
	v_dual_add_f32 v33, v33, v34 :: v_dual_mov_b32 v34, 0x3b03126f
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v35
	v_add_f32_e32 v33, v33, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v37
	v_add_f32_e32 v33, v33, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v39
	v_add_f32_e32 v33, v33, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v33
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v33, 0x3a83126f, v34, vcc_lo
	v_cvt_f16_f32_e32 v77.l, v33
	s_cbranch_scc0 .LBB6_26
	s_branch .LBB6_30
.LBB6_29:
	v_dual_mov_b32 v7, v16 :: v_dual_mov_b32 v6, v32
	v_dual_mov_b32 v5, v24 :: v_dual_mov_b32 v4, v23
	v_dual_mov_b32 v3, v22 :: v_dual_mov_b32 v2, v21
	v_dual_mov_b32 v1, v20 :: v_dual_mov_b32 v14, v24
	v_dual_mov_b32 v15, v32 :: v_dual_mov_b32 v12, v22
	v_dual_mov_b32 v13, v23 :: v_dual_mov_b32 v10, v20
	v_dual_mov_b32 v11, v21 :: v_dual_mov_b32 v30, v23
	v_dual_mov_b32 v9, v19 :: v_dual_mov_b32 v28, v21
	v_dual_mov_b32 v31, v24 :: v_dual_mov_b32 v26, v19
	v_mov_b32_e32 v29, v22
	v_mov_b32_e32 v27, v20
	v_mov_b32_e32 v25, v18
.LBB6_30:
	v_add_f32_e32 v17, 0, v17
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v18
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_add_f32_e32 v17, v17, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v26
	v_add_f32_e32 v17, v17, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v28
	v_add_f32_e32 v17, v17, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v30
	v_add_f32_e32 v17, v17, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v32
	v_add_f32_e32 v9, v17, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v1, v6 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_f32_e32 v2, v2, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end6:
	.size	_Z5probeILi4ELb0ELb1EEvPfPKfi, .Lfunc_end6-_Z5probeILi4ELb0ELb1EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi4ELb0ELb1EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 168
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-_Z5probeILi4ELb0ELb1EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi4ELb0ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi4ELb0ELb1EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.num_vgpr, 168
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.numbered_sgpr, 8
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5548
; TotalNumSgprs: 10
; NumVgprs: 168
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 168
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi4ELb1ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi4ELb1ELb1EEvPfPKfi,comdat
	.protected	_Z5probeILi4ELb1ELb1EEvPfPKfi ; -- Begin function _Z5probeILi4ELb1ELb1EEvPfPKfi
	.globl	_Z5probeILi4ELb1ELb1EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi4ELb1ELb1EEvPfPKfi,@function
_Z5probeILi4ELb1ELb1EEvPfPKfi:          ; @_Z5probeILi4ELb1ELb1EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_add_nc_u32_e32 v3, 1, v0
	s_load_b128 s[4:7], s[0:1], 0x0
	v_add_nc_u32_e32 v4, 10, v0
	v_add_nc_u32_e32 v5, 2, v0
	v_add_nc_u32_e32 v6, 11, v0
	v_and_b32_e32 v3, 0xff, v3
	v_add_nc_u32_e32 v2, 9, v0
	v_and_b32_e32 v4, 0xff, v4
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v6, 0xff, v6
	v_add_nc_u32_e32 v8, 12, v0
	v_lshlrev_b32_e32 v7, 2, v3
	v_add_nc_u32_e32 v3, 3, v0
	v_and_b32_e32 v2, 0xff, v2
	v_lshlrev_b32_e32 v9, 2, v4
	v_lshlrev_b32_e32 v11, 2, v5
	v_lshlrev_b32_e32 v12, 2, v6
	v_and_b32_e32 v4, 0xff, v8
	v_add_nc_u32_e32 v5, 4, v0
	v_add_nc_u32_e32 v6, 13, v0
	v_add_nc_u32_e32 v8, 5, v0
	v_and_b32_e32 v3, 0xff, v3
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_and_b32_e32 v15, 0xff, v5
	v_and_b32_e32 v16, 0xff, v6
	v_and_b32_e32 v17, 0xff, v8
	v_lshlrev_b32_e32 v13, 2, v3
	v_lshlrev_b32_e32 v14, 2, v4
	s_wait_kmcnt 0x0
	s_clause 0x7
	global_load_b32 v10, v1, s[6:7]
	global_load_b32 v3, v2, s[6:7]
	global_load_b32 v4, v7, s[6:7]
	global_load_b32 v5, v9, s[6:7]
	global_load_b32 v6, v11, s[6:7]
	global_load_b32 v7, v12, s[6:7]
	global_load_b32 v8, v13, s[6:7]
	global_load_b32 v9, v14, s[6:7]
	v_add_nc_u32_e32 v13, 14, v0
	v_add_nc_u32_e32 v14, 6, v0
	v_lshlrev_b32_e32 v2, 2, v15
	v_lshlrev_b32_e32 v11, 2, v16
	v_lshlrev_b32_e32 v12, 2, v17
	v_add_nc_u32_e32 v15, 15, v0
	v_add_nc_u32_e32 v16, 7, v0
	v_add_nc_u32_e32 v17, 16, v0
	v_and_b32_e32 v13, 0xff, v13
	v_and_b32_e32 v14, 0xff, v14
	v_and_b32_e32 v15, 0xff, v15
	v_and_b32_e32 v16, 0xff, v16
	v_and_b32_e32 v17, 0xff, v17
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v19, 2, v14
	v_lshlrev_b32_e32 v20, 2, v15
	v_lshlrev_b32_e32 v21, 2, v16
	v_lshlrev_b32_e32 v22, 2, v17
	s_clause 0x7
	global_load_b32 v17, v2, s[6:7]
	global_load_b32 v18, v11, s[6:7]
	global_load_b32 v14, v12, s[6:7]
	global_load_b32 v15, v13, s[6:7]
	global_load_b32 v16, v19, s[6:7]
	global_load_b32 v12, v20, s[6:7]
	global_load_b32 v13, v21, s[6:7]
	global_load_b32 v11, v22, s[6:7]
	v_mov_b32_e32 v20, 0
	v_add_co_u32 v1, s2, s6, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s7, 0, s2
	v_and_b32_e32 v19, 63, v0
	s_mov_b32 s2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB7_6
; %bb.1:
	v_lshl_add_u32 v21, v0, 2, 0
	v_lshrrev_b32_e32 v22, 6, v0
	s_mov_b32 s6, 0
	s_branch .LBB7_3
.LBB7_2:                                ;   in Loop: Header=BB7_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v20, -1, v20
	ds_store_b32 v21, v23
	v_add_nc_u32_e32 v21, 0x400, v21
	v_add_nc_u32_e32 v22, 4, v22
	v_cmp_eq_u32_e32 vcc_lo, 0, v20
	s_or_b32 s6, vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB7_5
.LBB7_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v23, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB7_2
; %bb.4:                                ;   in Loop: Header=BB7_3 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
	s_branch .LBB7_2
.LBB7_5:
	s_or_b32 exec_lo, exec_lo, s6
.LBB7_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v20, v0
	s_branch .LBB7_8
.LBB7_7:                                ;   in Loop: Header=BB7_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v22, 0x800, v20
	v_cmp_lt_u32_e32 vcc_lo, 0x7ff, v20
	ds_store_b32 v21, v23 offset:7168
	v_mov_b32_e32 v20, v22
	s_or_b32 s2, vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB7_24
.LBB7_8:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v22, 6, v20
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB7_10
; %bb.9:                                ;   in Loop: Header=BB7_8 Depth=1
	global_load_b32 v21, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v21
.LBB7_10:                               ;   in Loop: Header=BB7_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v25, 4, v22
	v_lshl_add_u32 v21, v20, 2, 0
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v24
	v_cmpx_gt_u32_e64 v19, v25
	s_cbranch_execz .LBB7_12
; %bb.11:                               ;   in Loop: Header=BB7_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
.LBB7_12:                               ;   in Loop: Header=BB7_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v24, 0 :: v_dual_add_nc_u32 v25, 8, v22
	ds_store_b32 v21, v23 offset:1024
	v_cmp_gt_u32_e32 vcc_lo, v19, v25
	v_mov_b32_e32 v25, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB7_14
; %bb.13:                               ;   in Loop: Header=BB7_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v25, 0x38d1b717, v23
.LBB7_14:                               ;   in Loop: Header=BB7_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v23, 12, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v25 offset:2048
	v_cmpx_gt_u32_e64 v19, v23
	s_cbranch_execz .LBB7_16
; %bb.15:                               ;   in Loop: Header=BB7_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v23
.LBB7_16:                               ;   in Loop: Header=BB7_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v23, 16, v22
	v_mov_b32_e32 v25, 0
	ds_store_b32 v21, v24 offset:3072
	v_cmp_gt_u32_e32 vcc_lo, v19, v23
	v_mov_b32_e32 v23, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB7_18
; %bb.17:                               ;   in Loop: Header=BB7_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
.LBB7_18:                               ;   in Loop: Header=BB7_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v24, 20, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v23 offset:4096
	v_cmpx_gt_u32_e64 v19, v24
	s_cbranch_execz .LBB7_20
; %bb.19:                               ;   in Loop: Header=BB7_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v25, 0x38d1b717, v23
.LBB7_20:                               ;   in Loop: Header=BB7_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v23, 0 :: v_dual_add_nc_u32 v24, 24, v22
	ds_store_b32 v21, v25 offset:5120
	v_cmp_gt_u32_e32 vcc_lo, v19, v24
	v_mov_b32_e32 v24, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB7_22
; %bb.21:                               ;   in Loop: Header=BB7_8 Depth=1
	global_load_b32 v24, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v24
.LBB7_22:                               ;   in Loop: Header=BB7_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v22, 28, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v24 offset:6144
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB7_7
; %bb.23:                               ;   in Loop: Header=BB7_8 Depth=1
	global_load_b32 v22, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v22
	s_branch .LBB7_7
.LBB7_24:
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	s_mov_b32 s1, 0x3c23d70a
	s_mov_b32 s2, 0x38d1b717
	s_mov_b32 s3, 0x399d4951
	s_wait_loadcnt 0xf
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v53, v10, s1, s2
	s_mov_b32 s2, 0x3951b717
	s_wait_loadcnt_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v57, v10, s1, s2
	s_mov_b32 s2, 0x39d1b717
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v58, v10, s1, s2
	s_mov_b32 s2, 0x3a1d4951
	v_fma_mixlo_f16 v58, v10, s1, s3
	s_mov_b32 s3, 0x3a03126e
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v59, v10, s1, s2
	s_mov_b32 s2, 0x3a51b717
	v_fma_mixlo_f16 v59, v10, s1, s3
	s_mov_b32 s3, 0x3a378034
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v60, v10, s1, s2
	s_mov_b32 s2, 0x3a6bedfa
	v_fma_mixlo_f16 v60, v10, s1, s3
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v64, v10, s1, s2
	s_mov_b32 s2, 0x3a83126e
	v_fma_mixlo_f16 v53, v10, s1, 0
	v_mov_b16_e32 v57.l, v53.h
	v_mov_b16_e32 v54.l, v57.h
	v_mov_b16_e32 v54.h, v58.l
	v_mov_b16_e32 v55.l, v58.h
	v_mov_b16_e32 v55.h, v59.l
	v_mov_b16_e32 v56.l, v59.h
	v_mov_b16_e32 v56.h, v60.l
	v_mov_b16_e32 v64.l, v60.h
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v52, v10, s1, s2
	v_mov_b16_e32 v52.l, v64.h
	s_mov_b32 s1, 0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB7_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s2, 0x3a83126f
	v_dual_mov_b32 v62, v55 :: v_dual_and_b32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v45, v3, s2, 0
	v_fma_mixhi_f16 v41, v4, s2, 0
	v_fma_mixhi_f16 v45, v5, s2, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixhi_f16 v46, v9, s2, 0
	v_dual_mov_b32 v50, v59 :: v_dual_lshlrev_b32 v9, 7, v0
	v_fma_mixlo_f16 v41, v10, s2, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_and_b32_e32 v65, 8, v3
	v_fma_mixlo_f16 v42, v6, s2, 0
	v_fma_mixlo_f16 v46, v7, s2, 0
	v_fma_mixhi_f16 v42, v8, s2, 0
	v_fma_mixhi_f16 v43, v14, s2, 0
	v_fma_mixhi_f16 v47, v15, s2, 0
	v_fma_mixlo_f16 v44, v16, s2, 0
	v_fma_mixlo_f16 v48, v12, s2, 0
	v_fma_mixhi_f16 v44, v13, s2, 0
	v_fma_mixhi_f16 v48, v11, s2, 0
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	v_lshl_add_u32 v66, v1, 2, 0
	v_or_b32_e32 v1, 1, v65
	v_or_b32_e32 v3, 2, v65
	v_or_b32_e32 v4, 3, v65
	v_or_b32_e32 v5, 4, v65
	v_or_b32_e32 v6, 5, v65
	v_or_b32_e32 v7, 6, v65
	v_or_b32_e32 v8, 7, v65
	v_or_b32_e32 v10, 16, v65
	v_or_b32_e32 v11, 17, v65
	v_or_b32_e32 v12, 18, v65
	v_or_b32_e32 v13, 19, v65
	v_and_b32_e32 v67, 0x800, v9
	v_or_b32_e32 v9, 20, v65
	v_or_b32_e32 v14, 21, v65
	v_or_b32_e32 v15, 22, v65
	v_or_b32_e32 v16, 23, v65
	v_fma_mixlo_f16 v43, v17, s2, 0
	v_fma_mixlo_f16 v47, v18, s2, 0
	v_or_b32_e32 v68, 0x3000, v67
	v_or_b32_e32 v69, 0x2700, v67
	v_or_b32_e32 v70, 0x2600, v67
	v_or_b32_e32 v71, 0x2500, v67
	v_or_b32_e32 v72, 0x2400, v67
	v_or_b32_e32 v73, 0x2300, v67
	v_or_b32_e32 v74, 0x2200, v67
	v_or_b32_e32 v75, 0x2100, v67
	v_or_b32_e32 v76, 0x2000, v67
	v_or_b32_e32 v77, 0x1700, v67
	v_or_b32_e32 v78, 0x1600, v67
	v_or_b32_e32 v79, 0x1500, v67
	v_or_b32_e32 v80, 0x1400, v67
	v_or_b32_e32 v81, 0x1300, v67
	v_or_b32_e32 v82, 0x1200, v67
	v_or_b32_e32 v83, 0x1100, v67
	v_or_b32_e32 v84, 0x1000, v67
	v_or_b32_e32 v85, 0x700, v67
	v_or_b32_e32 v86, 0x600, v67
	v_or_b32_e32 v87, 0x500, v67
	v_or_b32_e32 v88, 0x400, v67
	v_or_b32_e32 v89, 0x300, v67
	v_or_b32_e32 v90, 0x200, v67
	v_or_b32_e32 v91, 0x100, v67
	v_dual_mov_b32 v61, v54 :: v_dual_lshlrev_b32 v92, 2, v2
	v_lshlrev_b32_e32 v93, 2, v1
	v_dual_mov_b32 v63, v56 :: v_dual_lshlrev_b32 v94, 2, v3
	v_lshlrev_b32_e32 v95, 2, v4
	v_dual_mov_b32 v49, v58 :: v_dual_lshlrev_b32 v96, 2, v5
	v_lshlrev_b32_e32 v97, 2, v6
	v_dual_mov_b32 v51, v60 :: v_dual_lshlrev_b32 v98, 2, v7
	v_lshlrev_b32_e32 v99, 2, v8
	v_lshlrev_b32_e32 v100, 2, v10
	v_lshlrev_b32_e32 v101, 2, v11
	v_lshlrev_b32_e32 v102, 2, v12
	v_lshlrev_b32_e32 v103, 2, v13
	v_lshlrev_b32_e32 v104, 2, v9
	v_lshlrev_b32_e32 v105, 2, v14
	v_lshlrev_b32_e32 v106, 2, v15
	v_lshlrev_b32_e32 v107, 2, v16
	s_mov_b32 s2, 0x3f7d70a4
.LBB7_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_27 Depth 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], 0
	s_mov_b32 s3, 0
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[45:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v16, 0x3727c5ac, v8 :: v_dual_add_f32 v15, 0x3727c5ac, v7
	v_dual_add_f32 v14, 0x3727c5ac, v6 :: v_dual_add_f32 v13, 0x3727c5ac, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v12, 0x3727c5ac, v4 :: v_dual_add_f32 v11, 0x3727c5ac, v3
	v_dual_add_f32 v10, 0x3727c5ac, v2 :: v_dual_add_f32 v9, 0x3727c5ac, v1
	v_dual_add_f32 v24, 0x37a7c5ac, v8 :: v_dual_add_f32 v23, 0x37a7c5ac, v7
	v_dual_add_f32 v22, 0x37a7c5ac, v6 :: v_dual_add_f32 v21, 0x37a7c5ac, v5
	v_dual_add_f32 v20, 0x37a7c5ac, v4 :: v_dual_add_f32 v19, 0x37a7c5ac, v3
	v_dual_add_f32 v18, 0x37a7c5ac, v2 :: v_dual_add_f32 v17, 0x37a7c5ac, v1
	v_dual_add_f32 v115, 0x37fba882, v8 :: v_dual_add_f32 v114, 0x37fba882, v7
	v_dual_add_f32 v113, 0x37fba882, v6 :: v_dual_add_f32 v112, 0x37fba882, v5
	v_dual_add_f32 v111, 0x37fba882, v4 :: v_dual_add_f32 v110, 0x37fba882, v3
	v_dual_add_f32 v109, 0x37fba882, v2 :: v_dual_add_f32 v108, 0x37fba882, v1
	v_dual_add_f32 v123, 0x3827c5ac, v8 :: v_dual_add_f32 v122, 0x3827c5ac, v7
	v_dual_add_f32 v121, 0x3827c5ac, v6 :: v_dual_add_f32 v120, 0x3827c5ac, v5
	v_dual_add_f32 v119, 0x3827c5ac, v4 :: v_dual_add_f32 v118, 0x3827c5ac, v3
	v_dual_add_f32 v117, 0x3827c5ac, v2 :: v_dual_add_f32 v116, 0x3827c5ac, v1
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[53:56], v[9:16]
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[53:56], v[17:24]
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[53:56], v[108:115]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[116:123], v[41:44], v[53:56], v[116:123]
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[57:60], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[57:60], v[17:24]
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[57:60], v[108:115]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[116:123], v[41:44], v[57:60], v[116:123]
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[61:64], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[61:64], v[108:115]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[116:123], v[41:44], v[61:64], v[116:123]
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[49:52], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[49:52], v[17:24]
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[49:52], v[108:115]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[116:123], v[41:44], v[49:52], v[116:123]
	v_dual_mul_f32 v33, 0x3727c5ac, v9 :: v_dual_mul_f32 v34, 0x3727c5ac, v10
	v_dual_mul_f32 v35, 0x3727c5ac, v11 :: v_dual_mul_f32 v36, 0x3727c5ac, v12
	v_dual_mul_f32 v37, 0x3727c5ac, v13 :: v_dual_mul_f32 v38, 0x3727c5ac, v14
	v_dual_mul_f32 v39, 0x3727c5ac, v15 :: v_dual_mul_f32 v40, 0x3727c5ac, v16
	v_dual_mul_f32 v25, 0x3727c5ac, v17 :: v_dual_mul_f32 v26, 0x3727c5ac, v18
	v_dual_mul_f32 v27, 0x3727c5ac, v19 :: v_dual_mul_f32 v28, 0x3727c5ac, v20
	v_dual_mul_f32 v29, 0x3727c5ac, v21 :: v_dual_mul_f32 v30, 0x3727c5ac, v22
	v_dual_mul_f32 v31, 0x3727c5ac, v23 :: v_dual_mul_f32 v32, 0x3727c5ac, v24
	v_dual_mul_f32 v17, 0x3727c5ac, v108 :: v_dual_mul_f32 v18, 0x3727c5ac, v109
	v_dual_mul_f32 v19, 0x3727c5ac, v110 :: v_dual_mul_f32 v20, 0x3727c5ac, v111
	v_dual_mul_f32 v21, 0x3727c5ac, v112 :: v_dual_mul_f32 v22, 0x3727c5ac, v113
	v_dual_mul_f32 v23, 0x3727c5ac, v114 :: v_dual_mul_f32 v24, 0x3727c5ac, v115
	v_dual_mul_f32 v9, 0x3727c5ac, v116 :: v_dual_mul_f32 v10, 0x3727c5ac, v117
	v_dual_mul_f32 v11, 0x3727c5ac, v118 :: v_dual_mul_f32 v12, 0x3727c5ac, v119
	v_dual_mul_f32 v13, 0x3727c5ac, v120 :: v_dual_mul_f32 v14, 0x3727c5ac, v121
	v_dual_mul_f32 v15, 0x3727c5ac, v122 :: v_dual_mul_f32 v16, 0x3727c5ac, v123
.LBB7_27:                               ;   Parent Loop BB7_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e32 vcc_lo, s3, v67
	v_or_b32_e32 v109, 0x3100, v67
	v_add_nc_u32_e32 v110, s3, v66
	v_or_b32_e32 v111, 0x3200, v67
	v_or_b32_e32 v113, 0x3600, v67
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, 0, v33, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v91
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v34, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v90
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v35, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v36, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v37, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v87
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v38, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v86
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v39, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v85
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v40, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v84
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v25, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v83
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v26, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v82
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v27, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v81
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v28, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v80
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v29, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v79
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v30, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v78
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v31, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v77
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v32, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v76
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v17, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v75
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v18, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v74
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v19, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v73
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v20, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v72
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v21, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v22, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v70
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v23, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v69
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v108, v108, v24, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v68
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v108, v9, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v109
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v112, v108, v10, vcc_lo
	ds_load_2addr_b32 v[108:109], v110 offset1:32
	v_or_b32_e32 v110, 0x3300, v67
	v_cmp_eq_u32_e32 vcc_lo, s3, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v112, v11, vcc_lo
	v_or_b32_e32 v112, 0x3400, v67
	v_cmp_eq_u32_e32 vcc_lo, s3, v110
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v110, v111, v12, vcc_lo
	v_or_b32_e32 v111, 0x3500, v67
	v_cmp_eq_u32_e32 vcc_lo, s3, v112
	v_lshlrev_b32_e32 v112, 2, v65
	s_wait_dscnt 0x0
	ds_bpermute_b32 v115, v93, v108
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v110, v110, v13, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v111
	v_or_b32_e32 v111, 0x3700, v67
	ds_bpermute_b32 v114, v112, v108
	ds_bpermute_b32 v116, v95, v108
	ds_bpermute_b32 v117, v96, v108
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v110, v110, v14, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v113
	ds_bpermute_b32 v113, v94, v108
	ds_bpermute_b32 v118, v98, v108
	ds_bpermute_b32 v119, v99, v108
	ds_bpermute_b32 v120, v100, v108
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v110, v110, v15, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v111
	ds_bpermute_b32 v111, v97, v108
	ds_bpermute_b32 v122, v101, v108
	ds_bpermute_b32 v123, v102, v108
	ds_bpermute_b32 v124, v103, v108
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v110, v110, v16, vcc_lo
	ds_bpermute_b32 v125, v104, v108
	ds_bpermute_b32 v126, v105, v108
	ds_bpermute_b32 v127, v106, v108
	ds_bpermute_b32 v108, v107, v108
	ds_bpermute_b32 v121, v92, v110
	ds_bpermute_b32 v112, v112, v109
	ds_bpermute_b32 v128, v93, v109
	ds_bpermute_b32 v129, v94, v109
	ds_bpermute_b32 v130, v95, v109
	ds_bpermute_b32 v131, v96, v109
	ds_bpermute_b32 v132, v97, v109
	ds_bpermute_b32 v133, v98, v109
	ds_bpermute_b32 v134, v99, v109
	ds_bpermute_b32 v135, v100, v109
	ds_bpermute_b32 v136, v101, v109
	ds_bpermute_b32 v137, v102, v109
	ds_bpermute_b32 v138, v103, v109
	ds_bpermute_b32 v139, v105, v109
	ds_bpermute_b32 v140, v106, v109
	s_addk_co_i32 s3, 0x100
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s3, 0x4000
	s_wait_dscnt 0xe
	v_add_f32_e32 v110, v110, v121
	ds_bpermute_b32 v121, v104, v109
	ds_bpermute_b32 v109, v107, v109
	v_fma_f32 v33, -v110, v114, v33
	v_fma_f32 v34, -v110, v115, v34
	v_fma_f32 v35, -v110, v113, v35
	v_fma_f32 v36, -v110, v116, v36
	v_fma_f32 v37, -v110, v117, v37
	v_fma_f32 v38, -v110, v111, v38
	v_fma_f32 v39, -v110, v118, v39
	v_fma_f32 v40, -v110, v119, v40
	v_fma_f32 v25, -v110, v120, v25
	v_fma_f32 v26, -v110, v122, v26
	v_fma_f32 v27, -v110, v123, v27
	v_fma_f32 v28, -v110, v124, v28
	v_fma_f32 v29, -v110, v125, v29
	v_fma_f32 v30, -v110, v126, v30
	v_fma_f32 v31, -v110, v127, v31
	v_fma_f32 v32, -v110, v108, v32
	s_wait_dscnt 0xf
	v_fma_f32 v17, -v110, v112, v17
	s_wait_dscnt 0xe
	v_fma_f32 v18, -v110, v128, v18
	s_wait_dscnt 0xd
	v_fma_f32 v19, -v110, v129, v19
	s_wait_dscnt 0xc
	v_fma_f32 v20, -v110, v130, v20
	s_wait_dscnt 0xb
	v_fma_f32 v21, -v110, v131, v21
	s_wait_dscnt 0xa
	v_fma_f32 v22, -v110, v132, v22
	s_wait_dscnt 0x9
	v_fma_f32 v23, -v110, v133, v23
	s_wait_dscnt 0x8
	v_fma_f32 v24, -v110, v134, v24
	s_wait_dscnt 0x7
	v_fma_f32 v9, -v110, v135, v9
	s_wait_dscnt 0x6
	v_fma_f32 v10, -v110, v136, v10
	s_wait_dscnt 0x5
	v_fma_f32 v11, -v110, v137, v11
	s_wait_dscnt 0x4
	v_fma_f32 v12, -v110, v138, v12
	s_wait_dscnt 0x1
	v_fma_f32 v13, -v110, v121, v13
	v_fma_f32 v14, -v110, v139, v14
	v_fma_f32 v15, -v110, v140, v15
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v110, v109, v16
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB7_27
; %bb.28:                               ;   in Loop: Header=BB7_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[108:115], v[45:48], v[53:56], v[1:8]
	v_cvt_f16_f32_e32 v119.h, v40
	v_cvt_f16_f32_e32 v119.l, v39
	v_cvt_f16_f32_e32 v118.h, v38
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[108:115], v[45:48], v[57:60], v[108:115]
	v_cvt_f16_f32_e32 v118.l, v37
	v_cvt_f16_f32_e32 v117.h, v36
	v_cvt_f16_f32_e32 v117.l, v35
	v_cvt_f16_f32_e32 v116.h, v34
	v_wmma_f32_16x16x16_f16 v[108:115], v[45:48], v[61:64], v[108:115]
	v_cvt_f16_f32_e32 v116.l, v33
	v_cvt_f16_f32_e32 v123.h, v32
	v_cvt_f16_f32_e32 v123.l, v31
	v_cvt_f16_f32_e32 v122.h, v30
	v_wmma_f32_16x16x16_f16 v[108:115], v[45:48], v[49:52], v[108:115]
	v_cvt_f16_f32_e32 v122.l, v29
	v_cvt_f16_f32_e32 v121.h, v28
	v_cvt_f16_f32_e32 v121.l, v27
	v_cvt_f16_f32_e32 v120.h, v26
	v_cvt_f16_f32_e32 v120.l, v25
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[116:119], v[108:115]
	v_cvt_f16_f32_e32 v127.h, v24
	v_cvt_f16_f32_e32 v127.l, v23
	v_cvt_f16_f32_e32 v126.h, v22
	v_cvt_f16_f32_e32 v126.l, v21
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[120:123], v[108:115]
	v_cvt_f16_f32_e32 v125.h, v20
	v_cvt_f16_f32_e32 v125.l, v19
	v_cvt_f16_f32_e32 v124.h, v18
	v_cvt_f16_f32_e32 v124.l, v17
	v_cvt_f16_f32_e64 v131.h, v16
	v_cvt_f16_f32_e64 v131.l, v15
	v_cvt_f16_f32_e64 v130.h, v14
	v_cvt_f16_f32_e64 v130.l, v13
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[124:127], v[108:115]
	v_cvt_f16_f32_e64 v129.h, v12
	v_cvt_f16_f32_e64 v129.l, v11
	v_cvt_f16_f32_e64 v128.h, v10
	v_cvt_f16_f32_e64 v128.l, v9
	v_dual_add_f32 v16, v8, v8 :: v_dual_add_f32 v15, v7, v7
	v_dual_add_f32 v14, v6, v6 :: v_dual_add_f32 v13, v5, v5
	v_dual_add_f32 v12, v4, v4 :: v_dual_add_f32 v11, v3, v3
	v_dual_add_f32 v10, v2, v2 :: v_dual_add_f32 v9, v1, v1
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[128:131], v[108:115]
	v_mul_f32_e32 v22, 0x40400000, v6
	v_mul_f32_e32 v24, 0x40400000, v8
	v_mul_f32_e32 v20, 0x40400000, v4
	v_wmma_f32_16x16x16_f16 v[9:16], v[45:48], v[53:56], v[9:16]
	v_add_f32_e32 v17, 0, v108
	v_dual_mul_f32 v23, 0x40400000, v7 :: v_dual_mul_f32 v8, 4.0, v8
	v_mul_f32_e32 v18, 0x40400000, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[45:48], v[57:60], v[9:16]
	v_add_f32_e32 v17, v17, v109
	v_dual_mul_f32 v21, 0x40400000, v5 :: v_dual_mul_f32 v2, 4.0, v2
	v_mul_f32_e32 v6, 4.0, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[45:48], v[61:64], v[9:16]
	v_add_f32_e32 v17, v17, v110
	v_dual_mul_f32 v19, 0x40400000, v3 :: v_dual_mul_f32 v4, 4.0, v4
	v_mul_f32_e32 v7, 4.0, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[45:48], v[49:52], v[9:16]
	v_add_f32_e32 v17, v17, v111
	v_mul_f32_e32 v5, 4.0, v5
	v_mul_f32_e32 v3, 4.0, v3
	v_fma_mix_f32 v32, v56, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[116:119], v[9:16]
	v_add_f32_e32 v17, v17, v112
	v_fma_mix_f32 v31, v56, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v30, v55, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v29, v55, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[120:123], v[9:16]
	v_add_f32_e32 v17, v17, v113
	v_fma_mix_f32 v28, v54, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v27, v54, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v26, v53, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[124:127], v[9:16]
	v_add_f32_e32 v25, v17, v114
	v_mul_f32_e32 v17, 0x40400000, v1
	v_mul_f32_e32 v1, 4.0, v1
	v_fma_mix_f32 v40, v60, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[128:131], v[9:16]
	v_add_f32_e32 v25, v25, v115
	v_wmma_f32_16x16x16_f16 v[17:24], v[45:48], v[53:56], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[53:56], v[1:8]
	v_fma_mix_f32 v39, v60, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v38, v59, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v25, v9
	v_wmma_f32_16x16x16_f16 v[17:24], v[45:48], v[57:60], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[57:60], v[1:8]
	v_fma_mix_f32 v25, v53, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v37, v59, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v10
	v_wmma_f32_16x16x16_f16 v[17:24], v[45:48], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[61:64], v[1:8]
	v_fma_mix_f32 v36, v58, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v35, v58, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v11
	v_wmma_f32_16x16x16_f16 v[17:24], v[45:48], v[49:52], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[45:48], v[49:52], v[1:8]
	v_fma_mix_f32 v34, v57, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v33, v57, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v12
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[116:119], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[116:119], v[1:8]
	v_fma_mix_f32 v115, v64, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v114, v64, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v13
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[120:123], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[120:123], v[1:8]
	v_fma_mix_f32 v113, v63, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v112, v63, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v14
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[124:127], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[124:127], v[1:8]
	v_fma_mix_f32 v111, v62, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v110, v62, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v15
	v_wmma_f32_16x16x16_f16 v[17:24], v[41:44], v[128:131], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[41:44], v[128:131], v[1:8]
	v_fma_mix_f32 v109, v61, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v108, v61, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v16
	v_fma_mix_f32 v16, v52, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v15, v52, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v14, v51, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v13, v51, s2, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v17
	v_fma_mix_f32 v12, v50, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v11, v50, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v10, v49, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[25:32], v[41:44], v[116:119], v[25:32]
	v_add_f32_e32 v9, v9, v18
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[116:119], v[33:40]
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[116:119], v[108:115]
	v_mov_b32_e32 v18, 0x3b03126f
	v_wmma_f32_16x16x16_f16 v[25:32], v[41:44], v[120:123], v[25:32]
	v_add_f32_e32 v9, v9, v19
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[120:123], v[33:40]
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[120:123], v[108:115]
	s_add_co_i32 s1, s1, 1
	v_wmma_f32_16x16x16_f16 v[25:32], v[41:44], v[124:127], v[25:32]
	v_add_f32_e32 v17, v9, v20
	v_fma_mix_f32 v9, v49, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[124:127], v[33:40]
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[124:127], v[108:115]
	v_wmma_f32_16x16x16_f16 v[25:32], v[41:44], v[128:131], v[25:32]
	v_add_f32_e32 v17, v17, v21
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[116:119], v[9:16]
	v_wmma_f32_16x16x16_f16 v[33:40], v[41:44], v[128:131], v[33:40]
	v_wmma_f32_16x16x16_f16 v[108:115], v[41:44], v[128:131], v[108:115]
	v_cvt_f16_f32_e32 v53.h, v26
	v_add_f32_e32 v17, v17, v22
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[120:123], v[9:16]
	v_cvt_f16_f32_e32 v53.l, v25
	v_cvt_f16_f32_e32 v54.h, v28
	v_cvt_f16_f32_e32 v54.l, v27
	v_add_f32_e32 v17, v17, v23
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[124:127], v[9:16]
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v56.h, v32
	v_add_f32_e32 v17, v17, v24
	v_wmma_f32_16x16x16_f16 v[9:16], v[41:44], v[128:131], v[9:16]
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v57.h, v34
	v_cvt_f16_f32_e32 v57.l, v33
	v_add_f32_e32 v1, v17, v1
	v_cvt_f16_f32_e32 v58.h, v36
	v_cvt_f16_f32_e32 v58.l, v35
	v_cvt_f16_f32_e32 v59.h, v38
	v_cvt_f16_f32_e32 v59.l, v37
	v_add_f32_e32 v1, v1, v2
	v_cvt_f16_f32_e32 v60.h, v40
	v_cvt_f16_f32_e32 v60.l, v39
	v_cvt_f16_f32_e32 v61.h, v109
	v_cvt_f16_f32_e32 v61.l, v108
	v_add_f32_e32 v1, v1, v3
	v_cvt_f16_f32_e32 v62.h, v111
	v_cvt_f16_f32_e32 v62.l, v110
	v_cvt_f16_f32_e32 v63.h, v113
	v_cvt_f16_f32_e32 v63.l, v112
	v_add_f32_e32 v1, v1, v4
	v_cvt_f16_f32_e32 v64.h, v115
	v_cvt_f16_f32_e32 v64.l, v114
	v_cvt_f16_f32_e32 v50.h, v12
	v_cvt_f16_f32_e32 v50.l, v11
	v_add_f32_e32 v1, v1, v5
	v_cvt_f16_f32_e32 v51.h, v14
	v_cvt_f16_f32_e32 v51.l, v13
	v_cvt_f16_f32_e32 v52.h, v16
	v_cvt_f16_f32_e32 v52.l, v15
	v_add_f32_e32 v1, v1, v6
	v_cvt_f16_f32_e32 v49.h, v10
	v_cvt_f16_f32_e32 v49.l, v9
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s0
	v_add_f32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v8
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x3a83126f, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v45.l, v1
	s_cbranch_scc0 .LBB7_26
	s_branch .LBB7_30
.LBB7_29:
	v_dual_mov_b32 v51, v60 :: v_dual_mov_b32 v50, v59
	v_dual_mov_b32 v49, v58 :: v_dual_mov_b32 v62, v55
	v_mov_b32_e32 v63, v56
	v_mov_b32_e32 v61, v54
.LBB7_30:
	v_fma_mix_f32 v1, v53, 1.0, 0 op_sel_hi:[1,1,0]
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v53, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v54, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v54, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v55, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v55, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v56, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v56, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v57, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v57, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v58, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v58, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v59, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v59, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v60, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v60, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v61, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v61, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v62, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v62, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v63, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v63, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v64, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v64, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v49, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v49, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v50, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v50, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v51, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v2, v51, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mov_b32_e32 v1, 0
	v_fma_mix_f32 v2, v52, 1.0, v2 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fma_mix_f32 v2, v52, 1.0, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end7:
	.size	_Z5probeILi4ELb1ELb1EEvPfPKfi, .Lfunc_end7-_Z5probeILi4ELb1ELb1EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi4ELb1ELb1EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 141
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-_Z5probeILi4ELb1ELb1EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi4ELb1ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi4ELb1ELb1EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.num_vgpr, 141
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.numbered_sgpr, 8
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5704
; TotalNumSgprs: 10
; NumVgprs: 141
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 17
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 141
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi2ELb0ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi2ELb0ELb1EEvPfPKfi,comdat
	.protected	_Z5probeILi2ELb0ELb1EEvPfPKfi ; -- Begin function _Z5probeILi2ELb0ELb1EEvPfPKfi
	.globl	_Z5probeILi2ELb0ELb1EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi2ELb0ELb1EEvPfPKfi,@function
_Z5probeILi2ELb0ELb1EEvPfPKfi:          ; @_Z5probeILi2ELb0ELb1EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0
	v_add_nc_u32_e32 v3, 1, v0
	v_add_nc_u32_e32 v5, 2, v0
	v_add_nc_u32_e32 v4, 10, v0
	v_add_nc_u32_e32 v6, 11, v0
	v_add_nc_u32_e32 v2, 9, v0
	v_and_b32_e32 v3, 0xff, v3
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v4, 0xff, v4
	v_and_b32_e32 v6, 0xff, v6
	v_add_nc_u32_e32 v8, 12, v0
	v_lshlrev_b32_e32 v7, 2, v3
	v_add_nc_u32_e32 v3, 3, v0
	v_and_b32_e32 v2, 0xff, v2
	v_lshlrev_b32_e32 v10, 2, v5
	v_add_nc_u32_e32 v5, 4, v0
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v9, 2, v4
	v_lshlrev_b32_e32 v11, 2, v6
	v_and_b32_e32 v3, 0xff, v3
	v_and_b32_e32 v4, 0xff, v8
	v_add_nc_u32_e32 v6, 13, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_and_b32_e32 v14, 0xff, v5
	v_add_nc_u32_e32 v8, 5, v0
	v_lshlrev_b32_e32 v12, 2, v3
	v_lshlrev_b32_e32 v13, 2, v4
	v_and_b32_e32 v15, 0xff, v6
	s_wait_kmcnt 0x0
	s_clause 0x7
	global_load_b32 v19, v1, s[6:7]
	global_load_b32 v3, v2, s[6:7]
	global_load_b32 v4, v7, s[6:7]
	global_load_b32 v5, v9, s[6:7]
	global_load_b32 v6, v10, s[6:7]
	global_load_b32 v7, v11, s[6:7]
	global_load_b32 v17, v12, s[6:7]
	global_load_b32 v18, v13, s[6:7]
	v_add_nc_u32_e32 v10, 14, v0
	v_add_nc_u32_e32 v11, 6, v0
	v_lshlrev_b32_e32 v2, 2, v14
	v_add_nc_u32_e32 v12, 15, v0
	v_add_nc_u32_e32 v13, 7, v0
	v_add_nc_u32_e32 v14, 16, v0
	v_and_b32_e32 v8, 0xff, v8
	v_and_b32_e32 v10, 0xff, v10
	v_and_b32_e32 v11, 0xff, v11
	v_and_b32_e32 v12, 0xff, v12
	v_and_b32_e32 v13, 0xff, v13
	v_and_b32_e32 v14, 0xff, v14
	v_lshlrev_b32_e32 v9, 2, v15
	v_lshlrev_b32_e32 v8, 2, v8
	v_lshlrev_b32_e32 v10, 2, v10
	v_lshlrev_b32_e32 v11, 2, v11
	v_lshlrev_b32_e32 v12, 2, v12
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v14, 2, v14
	s_clause 0x7
	global_load_b32 v26, v2, s[6:7]
	global_load_b32 v27, v9, s[6:7]
	global_load_b32 v23, v8, s[6:7]
	global_load_b32 v24, v10, s[6:7]
	global_load_b32 v25, v11, s[6:7]
	global_load_b32 v21, v12, s[6:7]
	global_load_b32 v22, v13, s[6:7]
	global_load_b32 v20, v14, s[6:7]
	v_add_co_u32 v1, s2, s6, v1
	v_mov_b32_e32 v9, 0
	v_add_co_ci_u32_e64 v2, null, s7, 0, s2
	v_and_b32_e32 v8, 63, v0
	s_mov_b32 s2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB8_6
; %bb.1:
	v_lshl_add_u32 v10, v0, 2, 0
	v_lshrrev_b32_e32 v11, 6, v0
	s_mov_b32 s6, 0
	s_branch .LBB8_3
.LBB8_2:                                ;   in Loop: Header=BB8_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v9, -1, v9
	ds_store_b32 v10, v12
	v_add_nc_u32_e32 v10, 0x400, v10
	v_add_nc_u32_e32 v11, 4, v11
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_or_b32 s6, vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB8_5
.LBB8_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v12, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_gt_u32_e64 v8, v11
	s_cbranch_execz .LBB8_2
; %bb.4:                                ;   in Loop: Header=BB8_3 Depth=1
	global_load_b32 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v12, 0x38d1b717, v12
	s_branch .LBB8_2
.LBB8_5:
	s_or_b32 exec_lo, exec_lo, s6
.LBB8_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v9, v0
	s_branch .LBB8_8
.LBB8_7:                                ;   in Loop: Header=BB8_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v11, 0x800, v9
	v_cmp_lt_u32_e32 vcc_lo, 0x7ff, v9
	ds_store_b32 v10, v12 offset:7168
	v_mov_b32_e32 v9, v11
	s_or_b32 s2, vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB8_24
.LBB8_8:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v11, 6, v9
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e64 v8, v11
	s_cbranch_execz .LBB8_10
; %bb.9:                                ;   in Loop: Header=BB8_8 Depth=1
	global_load_b32 v10, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v13, 0x38d1b717, v10
.LBB8_10:                               ;   in Loop: Header=BB8_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v14, 4, v11
	v_lshl_add_u32 v10, v9, 2, 0
	s_mov_b32 s3, exec_lo
	ds_store_b32 v10, v13
	v_cmpx_gt_u32_e64 v8, v14
	s_cbranch_execz .LBB8_12
; %bb.11:                               ;   in Loop: Header=BB8_8 Depth=1
	global_load_b32 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v12, 0x38d1b717, v12
.LBB8_12:                               ;   in Loop: Header=BB8_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v13, 0 :: v_dual_add_nc_u32 v14, 8, v11
	ds_store_b32 v10, v12 offset:1024
	v_cmp_gt_u32_e32 vcc_lo, v8, v14
	v_mov_b32_e32 v14, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB8_14
; %bb.13:                               ;   in Loop: Header=BB8_8 Depth=1
	global_load_b32 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v14, 0x38d1b717, v12
.LBB8_14:                               ;   in Loop: Header=BB8_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v12, 12, v11
	s_mov_b32 s3, exec_lo
	ds_store_b32 v10, v14 offset:2048
	v_cmpx_gt_u32_e64 v8, v12
	s_cbranch_execz .LBB8_16
; %bb.15:                               ;   in Loop: Header=BB8_8 Depth=1
	global_load_b32 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v13, 0x38d1b717, v12
.LBB8_16:                               ;   in Loop: Header=BB8_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v12, 16, v11
	v_mov_b32_e32 v14, 0
	ds_store_b32 v10, v13 offset:3072
	v_cmp_gt_u32_e32 vcc_lo, v8, v12
	v_mov_b32_e32 v12, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB8_18
; %bb.17:                               ;   in Loop: Header=BB8_8 Depth=1
	global_load_b32 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v12, 0x38d1b717, v12
.LBB8_18:                               ;   in Loop: Header=BB8_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v13, 20, v11
	s_mov_b32 s3, exec_lo
	ds_store_b32 v10, v12 offset:4096
	v_cmpx_gt_u32_e64 v8, v13
	s_cbranch_execz .LBB8_20
; %bb.19:                               ;   in Loop: Header=BB8_8 Depth=1
	global_load_b32 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v14, 0x38d1b717, v12
.LBB8_20:                               ;   in Loop: Header=BB8_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v12, 0 :: v_dual_add_nc_u32 v13, 24, v11
	ds_store_b32 v10, v14 offset:5120
	v_cmp_gt_u32_e32 vcc_lo, v8, v13
	v_mov_b32_e32 v13, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB8_22
; %bb.21:                               ;   in Loop: Header=BB8_8 Depth=1
	global_load_b32 v13, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v13, 0x38d1b717, v13
.LBB8_22:                               ;   in Loop: Header=BB8_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v11, 28, v11
	s_mov_b32 s3, exec_lo
	ds_store_b32 v10, v13 offset:6144
	v_cmpx_gt_u32_e64 v8, v11
	s_cbranch_execz .LBB8_7
; %bb.23:                               ;   in Loop: Header=BB8_8 Depth=1
	global_load_b32 v11, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v12, 0x38d1b717, v11
	s_branch .LBB8_7
.LBB8_24:
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s1, 0x3c23d70a
	v_fma_f32 v9, 0x3c23d70a, v19, 0
	v_fmaak_f32 v10, s1, v19, 0x38d1b717
	v_fmaak_f32 v11, s1, v19, 0x3951b717
	v_fmaak_f32 v12, s1, v19, 0x399d4951
	v_fmaak_f32 v13, s1, v19, 0x39d1b717
	v_fmaak_f32 v14, s1, v19, 0x3a03126e
	v_fmaak_f32 v15, s1, v19, 0x3a1d4951
	v_fmaak_f32 v16, s1, v19, 0x3a378034
	v_fmaak_f32 v8, s1, v19, 0x3a51b717
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_cbranch_scc1 .LBB8_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s1, 0x3a83126f
	v_and_b32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v57, v4, s1, 0
	v_fma_mixhi_f16 v61, v5, s1, 0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixlo_f16 v61, v3, s1, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_fma_mixhi_f16 v62, v18, s1, 0
	v_lshlrev_b32_e32 v18, 7, v0
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_fma_mixlo_f16 v57, v19, s1, 0
	v_and_b32_e32 v73, 8, v3
	v_fma_mixlo_f16 v58, v6, s1, 0
	v_fma_mixlo_f16 v62, v7, s1, 0
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	v_fma_mixhi_f16 v58, v17, s1, 0
	v_fma_mixhi_f16 v59, v23, s1, 0
	v_fma_mixhi_f16 v63, v24, s1, 0
	v_fma_mixlo_f16 v60, v25, s1, 0
	v_dual_mov_b32 v2, v11 :: v_dual_lshlrev_b32 v95, 2, v2
	v_lshl_add_u32 v74, v1, 2, 0
	v_or_b32_e32 v1, 1, v73
	v_fma_mixlo_f16 v64, v21, s1, 0
	v_fma_mixhi_f16 v60, v22, s1, 0
	v_fma_mixhi_f16 v64, v20, s1, 0
	v_or_b32_e32 v3, 2, v73
	v_or_b32_e32 v4, 3, v73
	v_or_b32_e32 v5, 4, v73
	v_or_b32_e32 v6, 5, v73
	v_or_b32_e32 v7, 6, v73
	v_or_b32_e32 v17, 7, v73
	v_or_b32_e32 v19, 16, v73
	v_or_b32_e32 v20, 17, v73
	v_or_b32_e32 v21, 18, v73
	v_or_b32_e32 v22, 19, v73
	v_or_b32_e32 v23, 21, v73
	v_or_b32_e32 v24, 22, v73
	v_or_b32_e32 v25, 23, v73
	v_dual_mov_b32 v1, v10 :: v_dual_lshlrev_b32 v96, 2, v1
	v_and_b32_e32 v75, 0x800, v18
	v_or_b32_e32 v18, 20, v73
	v_fma_mixlo_f16 v59, v26, s1, 0
	v_fma_mixlo_f16 v63, v27, s1, 0
	v_lshlrev_b32_e32 v97, 2, v3
	v_lshlrev_b32_e32 v98, 2, v4
	v_lshlrev_b32_e32 v99, 2, v5
	v_lshlrev_b32_e32 v100, 2, v6
	v_lshlrev_b32_e32 v101, 2, v7
	v_lshlrev_b32_e32 v102, 2, v17
	v_lshlrev_b32_e32 v103, 2, v19
	v_lshlrev_b32_e32 v104, 2, v20
	v_lshlrev_b32_e32 v105, 2, v21
	v_lshlrev_b32_e32 v106, 2, v22
	v_lshlrev_b32_e32 v107, 2, v18
	v_lshlrev_b32_e32 v108, 2, v23
	v_lshlrev_b32_e32 v109, 2, v24
	v_dual_mov_b32 v3, v12 :: v_dual_lshlrev_b32 v110, 2, v25
	v_mov_b32_e32 v4, v13
	v_or_b32_e32 v76, 0x2300, v75
	v_or_b32_e32 v77, 0x2200, v75
	v_or_b32_e32 v78, 0x2100, v75
	v_or_b32_e32 v79, 0x2000, v75
	v_or_b32_e32 v80, 0x1700, v75
	v_or_b32_e32 v81, 0x1600, v75
	v_or_b32_e32 v82, 0x1500, v75
	v_or_b32_e32 v83, 0x1400, v75
	v_or_b32_e32 v84, 0x1300, v75
	v_or_b32_e32 v85, 0x1200, v75
	v_or_b32_e32 v86, 0x1100, v75
	v_or_b32_e32 v87, 0x1000, v75
	v_or_b32_e32 v88, 0x700, v75
	v_or_b32_e32 v89, 0x600, v75
	v_or_b32_e32 v90, 0x500, v75
	v_or_b32_e32 v91, 0x400, v75
	v_or_b32_e32 v92, 0x300, v75
	v_or_b32_e32 v93, 0x200, v75
	v_or_b32_e32 v94, 0x100, v75
	v_dual_mov_b32 v5, v14 :: v_dual_mov_b32 v6, v15
	v_mov_b32_e32 v7, v16
	s_mov_b32 s1, 0
.LBB8_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB8_27 Depth 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], 0
	v_cvt_f16_f32_e32 v68.h, v16
	v_cvt_f16_f32_e32 v68.l, v15
	v_cvt_f16_f32_e32 v67.h, v14
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_cvt_f16_f32_e32 v67.l, v13
	v_cvt_f16_f32_e32 v66.h, v12
	v_cvt_f16_f32_e32 v66.l, v11
	v_cvt_f16_f32_e32 v65.h, v10
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_cvt_f16_f32_e32 v65.l, v9
	v_cvt_f16_f32_e32 v72.h, v8
	v_cvt_f16_f32_e32 v72.l, v7
	v_cvt_f16_f32_e32 v71.h, v6
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_cvt_f16_f32_e32 v71.l, v5
	v_cvt_f16_f32_e32 v70.h, v4
	v_cvt_f16_f32_e32 v70.l, v3
	v_cvt_f16_f32_e32 v69.h, v2
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_cvt_f16_f32_e32 v69.l, v1
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[61:64], v[17:24]
	v_dual_add_f32 v32, 0x3727c5ac, v24 :: v_dual_add_f32 v31, 0x3727c5ac, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v30, 0x3727c5ac, v22 :: v_dual_add_f32 v29, 0x3727c5ac, v21
	v_dual_add_f32 v28, 0x3727c5ac, v20 :: v_dual_add_f32 v27, 0x3727c5ac, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v26, 0x3727c5ac, v18 :: v_dual_add_f32 v25, 0x3727c5ac, v17
	v_dual_add_f32 v40, 0x37a7c5ac, v24 :: v_dual_add_f32 v39, 0x37a7c5ac, v23
	v_dual_add_f32 v38, 0x37a7c5ac, v22 :: v_dual_add_f32 v37, 0x37a7c5ac, v21
	v_dual_add_f32 v36, 0x37a7c5ac, v20 :: v_dual_add_f32 v35, 0x37a7c5ac, v19
	v_dual_add_f32 v34, 0x37a7c5ac, v18 :: v_dual_add_f32 v33, 0x37a7c5ac, v17
	v_dual_add_f32 v118, 0x37fba882, v24 :: v_dual_add_f32 v117, 0x37fba882, v23
	v_dual_add_f32 v116, 0x37fba882, v22 :: v_dual_add_f32 v115, 0x37fba882, v21
	v_dual_add_f32 v114, 0x37fba882, v20 :: v_dual_add_f32 v113, 0x37fba882, v19
	v_dual_add_f32 v112, 0x37fba882, v18 :: v_dual_add_f32 v111, 0x37fba882, v17
	v_dual_add_f32 v126, 0x3827c5ac, v24 :: v_dual_add_f32 v125, 0x3827c5ac, v23
	v_dual_add_f32 v124, 0x3827c5ac, v22 :: v_dual_add_f32 v123, 0x3827c5ac, v21
	v_dual_add_f32 v122, 0x3827c5ac, v20 :: v_dual_add_f32 v121, 0x3827c5ac, v19
	v_dual_add_f32 v120, 0x3827c5ac, v18 :: v_dual_add_f32 v119, 0x3827c5ac, v17
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[65:68], v[25:32]
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[65:68], v[33:40]
	v_wmma_f32_16x16x16_f16 v[111:118], v[57:60], v[65:68], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[57:60], v[65:68], v[119:126]
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[69:72], v[25:32]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[69:72], v[33:40]
	v_wmma_f32_16x16x16_f16 v[111:118], v[57:60], v[69:72], v[111:118]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[119:126], v[57:60], v[69:72], v[119:126]
	v_dual_mul_f32 v49, 0x3727c5ac, v25 :: v_dual_mul_f32 v50, 0x3727c5ac, v26
	v_dual_mul_f32 v51, 0x3727c5ac, v27 :: v_dual_mul_f32 v52, 0x3727c5ac, v28
	v_dual_mul_f32 v53, 0x3727c5ac, v29 :: v_dual_mul_f32 v54, 0x3727c5ac, v30
	v_dual_mul_f32 v55, 0x3727c5ac, v31 :: v_dual_mul_f32 v56, 0x3727c5ac, v32
	v_dual_mul_f32 v41, 0x3727c5ac, v33 :: v_dual_mul_f32 v42, 0x3727c5ac, v34
	v_dual_mul_f32 v43, 0x3727c5ac, v35 :: v_dual_mul_f32 v44, 0x3727c5ac, v36
	v_dual_mul_f32 v45, 0x3727c5ac, v37 :: v_dual_mul_f32 v46, 0x3727c5ac, v38
	v_dual_mul_f32 v47, 0x3727c5ac, v39 :: v_dual_mul_f32 v48, 0x3727c5ac, v40
	v_dual_mul_f32 v33, 0x3727c5ac, v111 :: v_dual_mul_f32 v34, 0x3727c5ac, v112
	v_dual_mul_f32 v35, 0x3727c5ac, v113 :: v_dual_mul_f32 v36, 0x3727c5ac, v114
	v_dual_mul_f32 v37, 0x3727c5ac, v115 :: v_dual_mul_f32 v38, 0x3727c5ac, v116
	v_dual_mul_f32 v39, 0x3727c5ac, v117 :: v_dual_mul_f32 v40, 0x3727c5ac, v118
	v_dual_mul_f32 v25, 0x3727c5ac, v119 :: v_dual_mul_f32 v26, 0x3727c5ac, v120
	v_dual_mul_f32 v27, 0x3727c5ac, v121 :: v_dual_mul_f32 v28, 0x3727c5ac, v122
	v_dual_mul_f32 v29, 0x3727c5ac, v123 :: v_dual_mul_f32 v30, 0x3727c5ac, v124
	v_dual_mul_f32 v31, 0x3727c5ac, v125 :: v_dual_mul_f32 v32, 0x3727c5ac, v126
.LBB8_27:                               ;   Parent Loop BB8_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e32 vcc_lo, s2, v75
	v_or_b32_e32 v112, 0x2400, v75
	v_or_b32_e32 v113, 0x2500, v75
	v_or_b32_e32 v114, 0x3200, v75
	v_or_b32_e32 v115, 0x3300, v75
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, 0, v49, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v94
	v_or_b32_e32 v116, 0x3600, v75
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v50, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v93
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v51, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v92
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v52, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v91
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v53, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v90
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v54, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v89
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v55, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v88
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v56, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v87
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v41, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v86
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v42, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v85
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v43, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v84
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v44, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v83
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v45, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v82
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v46, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v81
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v47, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v80
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v48, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v79
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v33, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v78
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v34, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v77
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v35, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v76
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v111, v111, v36, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v112
	v_or_b32_e32 v112, 0x2600, v75
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v37, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v113
	v_or_b32_e32 v113, 0x2700, v75
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v111, v111, v38, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v112
	v_or_b32_e32 v112, 0x3000, v75
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v111, v39, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v113
	v_or_b32_e32 v113, 0x3100, v75
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v111, v111, v40, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v112
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v111, v111, v25 :: v_dual_add_nc_u32 v112, s2, v74
	v_cmp_eq_u32_e32 vcc_lo, s2, v113
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v113, v111, v26, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v114
	v_or_b32_e32 v114, 0x3400, v75
	ds_load_2addr_b32 v[111:112], v112 offset1:32
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v27, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v115
	v_or_b32_e32 v115, 0x3500, v75
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v113, v113, v28, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v114
	v_lshlrev_b32_e32 v114, 2, v73
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v29, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v115
	v_or_b32_e32 v115, 0x3700, v75
	s_wait_dscnt 0x0
	ds_bpermute_b32 v117, v114, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v30, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v116
	ds_bpermute_b32 v118, v96, v111
	ds_bpermute_b32 v116, v97, v111
	ds_bpermute_b32 v119, v98, v111
	ds_bpermute_b32 v120, v99, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v31, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s2, v115
	ds_bpermute_b32 v115, v100, v111
	ds_bpermute_b32 v121, v101, v111
	ds_bpermute_b32 v122, v102, v111
	ds_bpermute_b32 v123, v103, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v113, v32, vcc_lo
	ds_bpermute_b32 v125, v104, v111
	ds_bpermute_b32 v126, v105, v111
	ds_bpermute_b32 v127, v106, v111
	ds_bpermute_b32 v128, v107, v111
	ds_bpermute_b32 v124, v95, v113
	ds_bpermute_b32 v129, v108, v111
	ds_bpermute_b32 v130, v109, v111
	ds_bpermute_b32 v111, v110, v111
	ds_bpermute_b32 v114, v114, v112
	ds_bpermute_b32 v131, v96, v112
	ds_bpermute_b32 v132, v97, v112
	ds_bpermute_b32 v133, v98, v112
	ds_bpermute_b32 v134, v99, v112
	ds_bpermute_b32 v135, v100, v112
	ds_bpermute_b32 v136, v101, v112
	ds_bpermute_b32 v137, v102, v112
	ds_bpermute_b32 v138, v103, v112
	ds_bpermute_b32 v139, v104, v112
	ds_bpermute_b32 v140, v105, v112
	ds_bpermute_b32 v141, v106, v112
	ds_bpermute_b32 v142, v108, v112
	ds_bpermute_b32 v143, v109, v112
	s_addk_co_i32 s2, 0x100
	s_wait_dscnt 0x11
	v_add_f32_e32 v113, v113, v124
	ds_bpermute_b32 v124, v107, v112
	ds_bpermute_b32 v112, v110, v112
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s2, 0x4000
	v_fma_f32 v49, -v113, v117, v49
	v_fma_f32 v50, -v113, v118, v50
	v_fma_f32 v51, -v113, v116, v51
	v_fma_f32 v52, -v113, v119, v52
	v_fma_f32 v53, -v113, v120, v53
	v_fma_f32 v54, -v113, v115, v54
	v_fma_f32 v55, -v113, v121, v55
	v_fma_f32 v56, -v113, v122, v56
	v_fma_f32 v41, -v113, v123, v41
	v_fma_f32 v42, -v113, v125, v42
	v_fma_f32 v43, -v113, v126, v43
	v_fma_f32 v44, -v113, v127, v44
	v_fma_f32 v45, -v113, v128, v45
	s_wait_dscnt 0x12
	v_fma_f32 v46, -v113, v129, v46
	s_wait_dscnt 0x11
	v_fma_f32 v47, -v113, v130, v47
	s_wait_dscnt 0x10
	v_fma_f32 v48, -v113, v111, v48
	s_wait_dscnt 0xf
	v_fma_f32 v33, -v113, v114, v33
	s_wait_dscnt 0xe
	v_fma_f32 v34, -v113, v131, v34
	s_wait_dscnt 0xd
	v_fma_f32 v35, -v113, v132, v35
	s_wait_dscnt 0xc
	v_fma_f32 v36, -v113, v133, v36
	s_wait_dscnt 0xb
	v_fma_f32 v37, -v113, v134, v37
	s_wait_dscnt 0xa
	v_fma_f32 v38, -v113, v135, v38
	s_wait_dscnt 0x9
	v_fma_f32 v39, -v113, v136, v39
	s_wait_dscnt 0x8
	v_fma_f32 v40, -v113, v137, v40
	s_wait_dscnt 0x7
	v_fma_f32 v25, -v113, v138, v25
	s_wait_dscnt 0x6
	v_fma_f32 v26, -v113, v139, v26
	s_wait_dscnt 0x5
	v_fma_f32 v27, -v113, v140, v27
	s_wait_dscnt 0x4
	v_fma_f32 v28, -v113, v141, v28
	s_wait_dscnt 0x1
	v_fma_f32 v29, -v113, v124, v29
	v_fma_f32 v30, -v113, v142, v30
	v_fma_f32 v31, -v113, v143, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v113, v112, v32
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB8_27
; %bb.28:                               ;   in Loop: Header=BB8_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[111:118], v[61:64], v[65:68], v[17:24]
	v_cvt_f16_f32_e32 v56.h, v56
	v_cvt_f16_f32_e32 v56.l, v55
	v_cvt_f16_f32_e32 v55.h, v54
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[111:118], v[61:64], v[69:72], v[111:118]
	v_cvt_f16_f32_e32 v55.l, v53
	v_cvt_f16_f32_e32 v54.h, v52
	v_cvt_f16_f32_e32 v54.l, v51
	v_cvt_f16_f32_e32 v53.h, v50
	v_cvt_f16_f32_e32 v53.l, v49
	v_cvt_f16_f32_e32 v48.h, v48
	v_cvt_f16_f32_e32 v48.l, v47
	v_cvt_f16_f32_e32 v47.h, v46
	v_cvt_f16_f32_e32 v47.l, v45
	v_wmma_f32_16x16x16_f16 v[111:118], v[57:60], v[53:56], v[111:118]
	v_cvt_f16_f32_e32 v46.h, v44
	v_cvt_f16_f32_e32 v46.l, v43
	v_cvt_f16_f32_e32 v45.h, v42
	v_cvt_f16_f32_e32 v45.l, v41
	v_cvt_f16_f32_e32 v44.h, v40
	v_cvt_f16_f32_e32 v44.l, v39
	v_cvt_f16_f32_e32 v43.h, v38
	v_cvt_f16_f32_e32 v43.l, v37
	v_wmma_f32_16x16x16_f16 v[111:118], v[57:60], v[45:48], v[111:118]
	v_cvt_f16_f32_e32 v42.h, v36
	v_cvt_f16_f32_e32 v42.l, v35
	v_cvt_f16_f32_e32 v41.h, v34
	v_cvt_f16_f32_e32 v41.l, v33
	v_cvt_f16_f32_e32 v52.h, v32
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v51.l, v29
	v_wmma_f32_16x16x16_f16 v[111:118], v[57:60], v[41:44], v[111:118]
	v_cvt_f16_f32_e32 v50.h, v28
	v_cvt_f16_f32_e32 v50.l, v27
	v_cvt_f16_f32_e32 v49.h, v26
	v_cvt_f16_f32_e32 v49.l, v25
	v_dual_add_f32 v32, v24, v24 :: v_dual_add_f32 v31, v23, v23
	v_dual_add_f32 v30, v22, v22 :: v_dual_add_f32 v29, v21, v21
	v_add_f32_e32 v28, v20, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[111:118], v[57:60], v[49:52], v[111:118]
	v_dual_add_f32 v27, v19, v19 :: v_dual_add_f32 v26, v18, v18
	v_dual_add_f32 v25, v17, v17 :: v_dual_mul_f32 v40, 0x40400000, v24
	v_dual_add_f32 v33, 0, v111 :: v_dual_mul_f32 v38, 0x40400000, v22
	v_mul_f32_e32 v36, 0x40400000, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[61:64], v[65:68], v[25:32]
	v_mul_f32_e32 v24, 4.0, v24
	v_dual_add_f32 v33, v33, v112 :: v_dual_mul_f32 v34, 0x40400000, v18
	v_mul_f32_e32 v22, 4.0, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[61:64], v[69:72], v[25:32]
	v_dual_mul_f32 v39, 0x40400000, v23 :: v_dual_mul_f32 v18, 4.0, v18
	v_dual_add_f32 v33, v33, v113 :: v_dual_mul_f32 v20, 4.0, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[53:56], v[25:32]
	v_mul_f32_e32 v37, 0x40400000, v21
	v_mul_f32_e32 v35, 0x40400000, v19
	v_dual_add_f32 v33, v33, v114 :: v_dual_mul_f32 v16, 0x3f7d70a4, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[45:48], v[25:32]
	v_mul_f32_e32 v23, 4.0, v23
	v_mul_f32_e32 v21, 4.0, v21
	v_dual_add_f32 v33, v33, v115 :: v_dual_mul_f32 v14, 0x3f7d70a4, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[41:44], v[25:32]
	v_mul_f32_e32 v19, 4.0, v19
	v_mul_f32_e32 v15, 0x3f7d70a4, v15
	v_add_f32_e32 v33, v33, v116
	v_mul_f32_e32 v12, 0x3f7d70a4, v12
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[49:52], v[25:32]
	v_mul_f32_e32 v13, 0x3f7d70a4, v13
	v_mul_f32_e32 v11, 0x3f7d70a4, v11
	v_dual_add_f32 v33, v33, v117 :: v_dual_mul_f32 v10, 0x3f7d70a4, v10
	v_mul_f32_e32 v9, 0x3f7d70a4, v9
	v_mul_f32_e32 v7, 0x3f7d70a4, v7
	v_mul_f32_e32 v5, 0x3f7d70a4, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_f32_e32 v33, v33, v118
	v_dual_mul_f32 v6, 0x3f7d70a4, v6 :: v_dual_mul_f32 v3, 0x3f7d70a4, v3
	v_mul_f32_e32 v1, 0x3f7d70a4, v1
	v_wmma_f32_16x16x16_f16 v[9:16], v[57:60], v[53:56], v[9:16]
	v_add_f32_e32 v25, v33, v25
	v_mul_f32_e32 v33, 0x40400000, v17
	v_dual_mul_f32 v8, 0x3f7d70a4, v8 :: v_dual_mul_f32 v17, 4.0, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[57:60], v[45:48], v[9:16]
	v_add_f32_e32 v25, v25, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[33:40], v[61:64], v[65:68], v[33:40]
	v_mul_f32_e32 v4, 0x3f7d70a4, v4
	v_wmma_f32_16x16x16_f16 v[17:24], v[61:64], v[65:68], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[57:60], v[41:44], v[9:16]
	v_add_f32_e32 v25, v25, v27
	v_wmma_f32_16x16x16_f16 v[33:40], v[61:64], v[69:72], v[33:40]
	v_mul_f32_e32 v2, 0x3f7d70a4, v2
	v_wmma_f32_16x16x16_f16 v[17:24], v[61:64], v[69:72], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[57:60], v[49:52], v[9:16]
	v_add_f32_e32 v25, v25, v28
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[53:56], v[33:40]
	v_wmma_f32_16x16x16_f16 v[1:8], v[57:60], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[53:56], v[17:24]
	s_add_co_i32 s1, s1, 1
	v_add_f32_e32 v25, v25, v29
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[1:8], v[57:60], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[45:48], v[17:24]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s0
	v_add_f32_e32 v25, v25, v30
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[41:44], v[33:40]
	v_wmma_f32_16x16x16_f16 v[1:8], v[57:60], v[41:44], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[41:44], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v25, v25, v31
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[49:52], v[33:40]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[57:60], v[49:52], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[49:52], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v32
	v_add_f32_e32 v25, v25, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v34
	v_add_f32_e32 v25, v25, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v36
	v_add_f32_e32 v25, v25, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v38
	v_add_f32_e32 v25, v25, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v40
	v_add_f32_e32 v17, v25, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v17, v17, v18 :: v_dual_mov_b32 v18, 0x3b03126f
	v_add_f32_e32 v17, v17, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v20
	v_add_f32_e32 v17, v17, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v22
	v_add_f32_e32 v17, v17, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v24
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, 0x3a83126f, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v61.l, v17
	s_cbranch_scc0 .LBB8_26
	s_branch .LBB8_30
.LBB8_29:
	v_dual_mov_b32 v7, v16 :: v_dual_mov_b32 v6, v15
	v_dual_mov_b32 v5, v14 :: v_dual_mov_b32 v4, v13
	v_dual_mov_b32 v3, v12 :: v_dual_mov_b32 v2, v11
	v_mov_b32_e32 v1, v10
.LBB8_30:
	v_add_f32_e32 v9, 0, v9
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v10
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v12
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v14
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v16
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v1, v6 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v2, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_f32_e32 v2, v2, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end8:
	.size	_Z5probeILi2ELb0ELb1EEvPfPKfi, .Lfunc_end8-_Z5probeILi2ELb0ELb1EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi2ELb0ELb1EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 144
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end8-_Z5probeILi2ELb0ELb1EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi2ELb0ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi2ELb0ELb1EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.num_vgpr, 144
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.numbered_sgpr, 8
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4948
; TotalNumSgprs: 10
; NumVgprs: 144
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 17
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 144
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z5probeILi2ELb1ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi2ELb1ELb1EEvPfPKfi,comdat
	.protected	_Z5probeILi2ELb1ELb1EEvPfPKfi ; -- Begin function _Z5probeILi2ELb1ELb1EEvPfPKfi
	.globl	_Z5probeILi2ELb1ELb1EEvPfPKfi
	.p2align	8
	.type	_Z5probeILi2ELb1ELb1EEvPfPKfi,@function
_Z5probeILi2ELb1ELb1EEvPfPKfi:          ; @_Z5probeILi2ELb1ELb1EEvPfPKfi
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_add_nc_u32_e32 v3, 1, v0
	s_load_b128 s[4:7], s[0:1], 0x0
	v_add_nc_u32_e32 v4, 10, v0
	v_add_nc_u32_e32 v5, 2, v0
	v_add_nc_u32_e32 v6, 11, v0
	v_and_b32_e32 v3, 0xff, v3
	v_add_nc_u32_e32 v2, 9, v0
	v_and_b32_e32 v4, 0xff, v4
	v_and_b32_e32 v5, 0xff, v5
	v_and_b32_e32 v6, 0xff, v6
	v_add_nc_u32_e32 v8, 12, v0
	v_lshlrev_b32_e32 v7, 2, v3
	v_add_nc_u32_e32 v3, 3, v0
	v_and_b32_e32 v2, 0xff, v2
	v_lshlrev_b32_e32 v9, 2, v4
	v_lshlrev_b32_e32 v11, 2, v5
	v_lshlrev_b32_e32 v12, 2, v6
	v_and_b32_e32 v4, 0xff, v8
	v_add_nc_u32_e32 v5, 4, v0
	v_add_nc_u32_e32 v6, 13, v0
	v_add_nc_u32_e32 v8, 5, v0
	v_and_b32_e32 v3, 0xff, v3
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v2, 2, v2
	v_and_b32_e32 v15, 0xff, v5
	v_and_b32_e32 v16, 0xff, v6
	v_and_b32_e32 v17, 0xff, v8
	v_lshlrev_b32_e32 v13, 2, v3
	v_lshlrev_b32_e32 v14, 2, v4
	s_wait_kmcnt 0x0
	s_clause 0x7
	global_load_b32 v10, v1, s[6:7]
	global_load_b32 v3, v2, s[6:7]
	global_load_b32 v4, v7, s[6:7]
	global_load_b32 v5, v9, s[6:7]
	global_load_b32 v6, v11, s[6:7]
	global_load_b32 v7, v12, s[6:7]
	global_load_b32 v8, v13, s[6:7]
	global_load_b32 v9, v14, s[6:7]
	v_add_nc_u32_e32 v13, 14, v0
	v_add_nc_u32_e32 v14, 6, v0
	v_lshlrev_b32_e32 v2, 2, v15
	v_lshlrev_b32_e32 v11, 2, v16
	v_lshlrev_b32_e32 v12, 2, v17
	v_add_nc_u32_e32 v15, 15, v0
	v_add_nc_u32_e32 v16, 7, v0
	v_add_nc_u32_e32 v17, 16, v0
	v_and_b32_e32 v13, 0xff, v13
	v_and_b32_e32 v14, 0xff, v14
	v_and_b32_e32 v15, 0xff, v15
	v_and_b32_e32 v16, 0xff, v16
	v_and_b32_e32 v17, 0xff, v17
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v19, 2, v14
	v_lshlrev_b32_e32 v20, 2, v15
	v_lshlrev_b32_e32 v21, 2, v16
	v_lshlrev_b32_e32 v22, 2, v17
	s_clause 0x7
	global_load_b32 v17, v2, s[6:7]
	global_load_b32 v18, v11, s[6:7]
	global_load_b32 v14, v12, s[6:7]
	global_load_b32 v15, v13, s[6:7]
	global_load_b32 v16, v19, s[6:7]
	global_load_b32 v12, v20, s[6:7]
	global_load_b32 v13, v21, s[6:7]
	global_load_b32 v11, v22, s[6:7]
	v_mov_b32_e32 v20, 0
	v_add_co_u32 v1, s2, s6, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s7, 0, s2
	v_and_b32_e32 v19, 63, v0
	s_mov_b32 s2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB9_6
; %bb.1:
	v_lshl_add_u32 v21, v0, 2, 0
	v_lshrrev_b32_e32 v22, 6, v0
	s_mov_b32 s6, 0
	s_branch .LBB9_3
.LBB9_2:                                ;   in Loop: Header=BB9_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v20, -1, v20
	ds_store_b32 v21, v23
	v_add_nc_u32_e32 v21, 0x400, v21
	v_add_nc_u32_e32 v22, 4, v22
	v_cmp_eq_u32_e32 vcc_lo, 0, v20
	s_or_b32 s6, vcc_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB9_5
.LBB9_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v23, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB9_2
; %bb.4:                                ;   in Loop: Header=BB9_3 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
	s_branch .LBB9_2
.LBB9_5:
	s_or_b32 exec_lo, exec_lo, s6
.LBB9_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v20, v0
	s_branch .LBB9_8
.LBB9_7:                                ;   in Loop: Header=BB9_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v22, 0x800, v20
	v_cmp_lt_u32_e32 vcc_lo, 0x7ff, v20
	ds_store_b32 v21, v23 offset:7168
	v_mov_b32_e32 v20, v22
	s_or_b32 s2, vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB9_24
.LBB9_8:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v22, 6, v20
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB9_10
; %bb.9:                                ;   in Loop: Header=BB9_8 Depth=1
	global_load_b32 v21, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v21
.LBB9_10:                               ;   in Loop: Header=BB9_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v25, 4, v22
	v_lshl_add_u32 v21, v20, 2, 0
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v24
	v_cmpx_gt_u32_e64 v19, v25
	s_cbranch_execz .LBB9_12
; %bb.11:                               ;   in Loop: Header=BB9_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
.LBB9_12:                               ;   in Loop: Header=BB9_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v24, 0 :: v_dual_add_nc_u32 v25, 8, v22
	ds_store_b32 v21, v23 offset:1024
	v_cmp_gt_u32_e32 vcc_lo, v19, v25
	v_mov_b32_e32 v25, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB9_14
; %bb.13:                               ;   in Loop: Header=BB9_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v25, 0x38d1b717, v23
.LBB9_14:                               ;   in Loop: Header=BB9_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v23, 12, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v25 offset:2048
	v_cmpx_gt_u32_e64 v19, v23
	s_cbranch_execz .LBB9_16
; %bb.15:                               ;   in Loop: Header=BB9_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v23
.LBB9_16:                               ;   in Loop: Header=BB9_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v23, 16, v22
	v_mov_b32_e32 v25, 0
	ds_store_b32 v21, v24 offset:3072
	v_cmp_gt_u32_e32 vcc_lo, v19, v23
	v_mov_b32_e32 v23, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB9_18
; %bb.17:                               ;   in Loop: Header=BB9_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v23
.LBB9_18:                               ;   in Loop: Header=BB9_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v24, 20, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v23 offset:4096
	v_cmpx_gt_u32_e64 v19, v24
	s_cbranch_execz .LBB9_20
; %bb.19:                               ;   in Loop: Header=BB9_8 Depth=1
	global_load_b32 v23, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v25, 0x38d1b717, v23
.LBB9_20:                               ;   in Loop: Header=BB9_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v23, 0 :: v_dual_add_nc_u32 v24, 24, v22
	ds_store_b32 v21, v25 offset:5120
	v_cmp_gt_u32_e32 vcc_lo, v19, v24
	v_mov_b32_e32 v24, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB9_22
; %bb.21:                               ;   in Loop: Header=BB9_8 Depth=1
	global_load_b32 v24, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v24, 0x38d1b717, v24
.LBB9_22:                               ;   in Loop: Header=BB9_8 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_add_nc_u32_e32 v22, 28, v22
	s_mov_b32 s3, exec_lo
	ds_store_b32 v21, v24 offset:6144
	v_cmpx_gt_u32_e64 v19, v22
	s_cbranch_execz .LBB9_7
; %bb.23:                               ;   in Loop: Header=BB9_8 Depth=1
	global_load_b32 v22, v[1:2], off
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v23, 0x38d1b717, v22
	s_branch .LBB9_7
.LBB9_24:
	s_or_b32 exec_lo, exec_lo, s2
	s_load_b32 s0, s[0:1], 0x10
	s_mov_b32 s1, 0x3c23d70a
	s_mov_b32 s2, 0x38d1b717
	s_mov_b32 s3, 0x399d4951
	s_wait_loadcnt 0xf
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v45, v10, s1, s2
	s_mov_b32 s2, 0x3951b717
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v46, v10, s1, s2
	v_fma_mixhi_f16 v46, v10, s1, s3
	s_mov_b32 s2, 0x39d1b717
	s_mov_b32 s3, 0x3a03126e
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v47, v10, s1, s2
	v_fma_mixhi_f16 v47, v10, s1, s3
	s_mov_b32 s2, 0x3a1d4951
	s_mov_b32 s3, 0x3a378034
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v48, v10, s1, s2
	v_fma_mixhi_f16 v48, v10, s1, s3
	s_mov_b32 s2, 0x3a51b717
	v_fma_mixlo_f16 v45, v10, s1, 0
	v_mov_b16_e32 v41.l, v45.h
	v_mov_b16_e32 v41.h, v46.l
	v_mov_b16_e32 v42.l, v46.h
	v_mov_b16_e32 v42.h, v47.l
	v_mov_b16_e32 v43.l, v47.h
	v_mov_b16_e32 v43.h, v48.l
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v44, v10, s1, s2
	v_mov_b16_e32 v44.l, v48.h
	s_mov_b32 s1, 0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s0, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB9_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s2, 0x3a83126f
	v_and_b32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v53, v3, s2, 0
	v_fma_mixhi_f16 v49, v4, s2, 0
	v_fma_mixhi_f16 v53, v5, s2, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixhi_f16 v54, v9, s2, 0
	v_lshlrev_b32_e32 v9, 7, v0
	v_fma_mixlo_f16 v49, v10, s2, 0
	v_fma_mixlo_f16 v50, v6, s2, 0
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_and_b32_e32 v57, 8, v3
	v_fma_mixlo_f16 v54, v7, s2, 0
	v_fma_mixhi_f16 v50, v8, s2, 0
	v_fma_mixhi_f16 v51, v14, s2, 0
	v_fma_mixhi_f16 v55, v15, s2, 0
	v_fma_mixlo_f16 v52, v16, s2, 0
	v_fma_mixlo_f16 v56, v12, s2, 0
	v_fma_mixhi_f16 v52, v13, s2, 0
	v_fma_mixhi_f16 v56, v11, s2, 0
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	v_lshl_add_u32 v58, v1, 2, 0
	v_or_b32_e32 v1, 1, v57
	v_or_b32_e32 v3, 2, v57
	v_or_b32_e32 v4, 3, v57
	v_or_b32_e32 v5, 4, v57
	v_or_b32_e32 v6, 5, v57
	v_or_b32_e32 v7, 6, v57
	v_or_b32_e32 v8, 7, v57
	v_or_b32_e32 v10, 16, v57
	v_or_b32_e32 v11, 17, v57
	v_or_b32_e32 v12, 18, v57
	v_or_b32_e32 v13, 19, v57
	v_and_b32_e32 v59, 0x800, v9
	v_or_b32_e32 v9, 20, v57
	v_or_b32_e32 v14, 21, v57
	v_or_b32_e32 v15, 22, v57
	v_or_b32_e32 v16, 23, v57
	v_fma_mixlo_f16 v51, v17, s2, 0
	v_fma_mixlo_f16 v55, v18, s2, 0
	v_or_b32_e32 v60, 0x1300, v59
	v_or_b32_e32 v61, 0x1200, v59
	v_or_b32_e32 v62, 0x1100, v59
	v_or_b32_e32 v63, 0x1000, v59
	v_or_b32_e32 v64, 0x700, v59
	v_or_b32_e32 v65, 0x600, v59
	v_or_b32_e32 v66, 0x500, v59
	v_or_b32_e32 v67, 0x400, v59
	v_or_b32_e32 v68, 0x300, v59
	v_or_b32_e32 v69, 0x200, v59
	v_or_b32_e32 v70, 0x100, v59
	v_lshlrev_b32_e32 v71, 2, v2
	v_lshlrev_b32_e32 v72, 2, v1
	v_lshlrev_b32_e32 v73, 2, v3
	v_lshlrev_b32_e32 v74, 2, v4
	v_lshlrev_b32_e32 v75, 2, v5
	v_lshlrev_b32_e32 v76, 2, v6
	v_lshlrev_b32_e32 v77, 2, v7
	v_lshlrev_b32_e32 v78, 2, v8
	v_lshlrev_b32_e32 v79, 2, v10
	v_lshlrev_b32_e32 v80, 2, v11
	v_lshlrev_b32_e32 v81, 2, v12
	v_lshlrev_b32_e32 v82, 2, v13
	v_lshlrev_b32_e32 v83, 2, v9
	v_lshlrev_b32_e32 v84, 2, v14
	v_lshlrev_b32_e32 v85, 2, v15
	v_lshlrev_b32_e32 v86, 2, v16
	s_mov_b32 s2, 0x3f7d70a4
.LBB9_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB9_27 Depth 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], 0
	s_mov_b32 s3, 0
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v16, 0x3727c5ac, v8 :: v_dual_add_f32 v15, 0x3727c5ac, v7
	v_dual_add_f32 v14, 0x3727c5ac, v6 :: v_dual_add_f32 v13, 0x3727c5ac, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v12, 0x3727c5ac, v4 :: v_dual_add_f32 v11, 0x3727c5ac, v3
	v_dual_add_f32 v10, 0x3727c5ac, v2 :: v_dual_add_f32 v9, 0x3727c5ac, v1
	v_dual_add_f32 v24, 0x37a7c5ac, v8 :: v_dual_add_f32 v23, 0x37a7c5ac, v7
	v_dual_add_f32 v22, 0x37a7c5ac, v6 :: v_dual_add_f32 v21, 0x37a7c5ac, v5
	v_dual_add_f32 v20, 0x37a7c5ac, v4 :: v_dual_add_f32 v19, 0x37a7c5ac, v3
	v_dual_add_f32 v18, 0x37a7c5ac, v2 :: v_dual_add_f32 v17, 0x37a7c5ac, v1
	v_dual_add_f32 v94, 0x37fba882, v8 :: v_dual_add_f32 v93, 0x37fba882, v7
	v_dual_add_f32 v92, 0x37fba882, v6 :: v_dual_add_f32 v91, 0x37fba882, v5
	v_dual_add_f32 v90, 0x37fba882, v4 :: v_dual_add_f32 v89, 0x37fba882, v3
	v_dual_add_f32 v88, 0x37fba882, v2 :: v_dual_add_f32 v87, 0x37fba882, v1
	v_dual_add_f32 v102, 0x3827c5ac, v8 :: v_dual_add_f32 v101, 0x3827c5ac, v7
	v_dual_add_f32 v100, 0x3827c5ac, v6 :: v_dual_add_f32 v99, 0x3827c5ac, v5
	v_dual_add_f32 v98, 0x3827c5ac, v4 :: v_dual_add_f32 v97, 0x3827c5ac, v3
	v_dual_add_f32 v96, 0x3827c5ac, v2 :: v_dual_add_f32 v95, 0x3827c5ac, v1
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[45:48], v[9:16]
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[45:48], v[17:24]
	v_wmma_f32_16x16x16_f16 v[87:94], v[49:52], v[45:48], v[87:94]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[95:102], v[49:52], v[45:48], v[95:102]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[41:44], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[41:44], v[17:24]
	v_wmma_f32_16x16x16_f16 v[87:94], v[49:52], v[41:44], v[87:94]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[95:102], v[49:52], v[41:44], v[95:102]
	v_dual_mul_f32 v33, 0x3727c5ac, v9 :: v_dual_mul_f32 v34, 0x3727c5ac, v10
	v_dual_mul_f32 v35, 0x3727c5ac, v11 :: v_dual_mul_f32 v36, 0x3727c5ac, v12
	v_dual_mul_f32 v37, 0x3727c5ac, v13 :: v_dual_mul_f32 v38, 0x3727c5ac, v14
	v_dual_mul_f32 v39, 0x3727c5ac, v15 :: v_dual_mul_f32 v40, 0x3727c5ac, v16
	v_dual_mul_f32 v25, 0x3727c5ac, v17 :: v_dual_mul_f32 v26, 0x3727c5ac, v18
	v_dual_mul_f32 v27, 0x3727c5ac, v19 :: v_dual_mul_f32 v28, 0x3727c5ac, v20
	v_dual_mul_f32 v29, 0x3727c5ac, v21 :: v_dual_mul_f32 v30, 0x3727c5ac, v22
	v_dual_mul_f32 v31, 0x3727c5ac, v23 :: v_dual_mul_f32 v32, 0x3727c5ac, v24
	v_dual_mul_f32 v17, 0x3727c5ac, v87 :: v_dual_mul_f32 v18, 0x3727c5ac, v88
	v_dual_mul_f32 v19, 0x3727c5ac, v89 :: v_dual_mul_f32 v20, 0x3727c5ac, v90
	v_dual_mul_f32 v21, 0x3727c5ac, v91 :: v_dual_mul_f32 v22, 0x3727c5ac, v92
	v_dual_mul_f32 v23, 0x3727c5ac, v93 :: v_dual_mul_f32 v24, 0x3727c5ac, v94
	v_dual_mul_f32 v9, 0x3727c5ac, v95 :: v_dual_mul_f32 v10, 0x3727c5ac, v96
	v_dual_mul_f32 v11, 0x3727c5ac, v97 :: v_dual_mul_f32 v12, 0x3727c5ac, v98
	v_dual_mul_f32 v13, 0x3727c5ac, v99 :: v_dual_mul_f32 v14, 0x3727c5ac, v100
	v_dual_mul_f32 v15, 0x3727c5ac, v101 :: v_dual_mul_f32 v16, 0x3727c5ac, v102
.LBB9_27:                               ;   Parent Loop BB9_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e32 vcc_lo, s3, v59
	v_or_b32_e32 v88, 0x1400, v59
	v_or_b32_e32 v89, 0x1500, v59
	v_or_b32_e32 v90, 0x3300, v59
	v_or_b32_e32 v92, 0x3700, v59
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, 0, v33, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v70
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v87, v87, v34, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v69
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v35, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v68
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v87, v87, v36, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v67
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v37, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v66
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v87, v87, v38, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v65
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v39, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v64
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v87, v87, v40, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v63
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v25, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v62
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v87, v87, v26, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v61
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v27, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v60
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v87, v87, v28, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	v_or_b32_e32 v88, 0x1600, v59
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v29, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	v_or_b32_e32 v89, 0x1700, v59
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v87, v87, v30, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	v_or_b32_e32 v88, 0x2000, v59
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v31, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	v_or_b32_e32 v89, 0x2100, v59
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v87, v87, v32, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	v_or_b32_e32 v88, 0x2200, v59
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v17, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	v_or_b32_e32 v89, 0x2300, v59
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v87, v87, v18, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	v_or_b32_e32 v88, 0x2400, v59
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v19, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	v_or_b32_e32 v89, 0x2500, v59
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v87, v87, v20, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	v_or_b32_e32 v88, 0x2600, v59
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v21, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	v_or_b32_e32 v89, 0x2700, v59
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v87, v87, v22, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	v_or_b32_e32 v88, 0x3000, v59
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v23, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	v_or_b32_e32 v89, 0x3100, v59
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v87, v87, v24, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	v_or_b32_e32 v88, 0x3200, v59
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v9, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	v_add_nc_u32_e32 v89, s3, v58
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v87, v87, v10, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v88
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v91, v87, v11, vcc_lo
	ds_load_2addr_b32 v[87:88], v89 offset1:32
	v_or_b32_e32 v89, 0x3400, v59
	v_cmp_eq_u32_e32 vcc_lo, s3, v90
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v90, v91, v12, vcc_lo
	v_or_b32_e32 v91, 0x3500, v59
	v_cmp_eq_u32_e32 vcc_lo, s3, v89
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v89, v90, v13, vcc_lo
	v_or_b32_e32 v90, 0x3600, v59
	v_cmp_eq_u32_e32 vcc_lo, s3, v91
	v_lshlrev_b32_e32 v91, 2, v57
	s_wait_dscnt 0x0
	ds_bpermute_b32 v93, v72, v87
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v89, v89, v14, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v90
	ds_bpermute_b32 v90, v91, v87
	ds_bpermute_b32 v94, v73, v87
	ds_bpermute_b32 v95, v75, v87
	ds_bpermute_b32 v96, v76, v87
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v89, v89, v15, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, s3, v92
	ds_bpermute_b32 v92, v74, v87
	ds_bpermute_b32 v97, v77, v87
	ds_bpermute_b32 v99, v78, v87
	ds_bpermute_b32 v100, v79, v87
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v89, v89, v16, vcc_lo
	ds_bpermute_b32 v101, v80, v87
	ds_bpermute_b32 v102, v81, v87
	ds_bpermute_b32 v103, v82, v87
	ds_bpermute_b32 v104, v83, v87
	ds_bpermute_b32 v98, v71, v89
	ds_bpermute_b32 v105, v84, v87
	ds_bpermute_b32 v106, v85, v87
	ds_bpermute_b32 v87, v86, v87
	ds_bpermute_b32 v91, v91, v88
	ds_bpermute_b32 v107, v72, v88
	ds_bpermute_b32 v108, v73, v88
	ds_bpermute_b32 v109, v74, v88
	ds_bpermute_b32 v110, v75, v88
	ds_bpermute_b32 v111, v76, v88
	ds_bpermute_b32 v112, v77, v88
	ds_bpermute_b32 v113, v78, v88
	ds_bpermute_b32 v114, v79, v88
	ds_bpermute_b32 v115, v80, v88
	ds_bpermute_b32 v116, v82, v88
	ds_bpermute_b32 v117, v83, v88
	ds_bpermute_b32 v118, v84, v88
	ds_bpermute_b32 v119, v85, v88
	s_addk_co_i32 s3, 0x100
	s_wait_dscnt 0x11
	v_add_f32_e32 v89, v89, v98
	ds_bpermute_b32 v98, v81, v88
	ds_bpermute_b32 v88, v86, v88
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s3, 0x4000
	v_fma_f32 v33, -v89, v90, v33
	v_fma_f32 v34, -v89, v93, v34
	v_fma_f32 v35, -v89, v94, v35
	v_fma_f32 v36, -v89, v92, v36
	v_fma_f32 v37, -v89, v95, v37
	v_fma_f32 v38, -v89, v96, v38
	v_fma_f32 v39, -v89, v97, v39
	v_fma_f32 v40, -v89, v99, v40
	v_fma_f32 v25, -v89, v100, v25
	v_fma_f32 v26, -v89, v101, v26
	v_fma_f32 v27, -v89, v102, v27
	v_fma_f32 v28, -v89, v103, v28
	v_fma_f32 v29, -v89, v104, v29
	s_wait_dscnt 0x12
	v_fma_f32 v30, -v89, v105, v30
	s_wait_dscnt 0x11
	v_fma_f32 v31, -v89, v106, v31
	s_wait_dscnt 0x10
	v_fma_f32 v32, -v89, v87, v32
	s_wait_dscnt 0xf
	v_fma_f32 v17, -v89, v91, v17
	s_wait_dscnt 0xe
	v_fma_f32 v18, -v89, v107, v18
	s_wait_dscnt 0xd
	v_fma_f32 v19, -v89, v108, v19
	s_wait_dscnt 0xc
	v_fma_f32 v20, -v89, v109, v20
	s_wait_dscnt 0xb
	v_fma_f32 v21, -v89, v110, v21
	s_wait_dscnt 0xa
	v_fma_f32 v22, -v89, v111, v22
	s_wait_dscnt 0x9
	v_fma_f32 v23, -v89, v112, v23
	s_wait_dscnt 0x8
	v_fma_f32 v24, -v89, v113, v24
	s_wait_dscnt 0x7
	v_fma_f32 v9, -v89, v114, v9
	s_wait_dscnt 0x6
	v_fma_f32 v10, -v89, v115, v10
	s_wait_dscnt 0x1
	v_fma_f32 v11, -v89, v98, v11
	v_fma_f32 v12, -v89, v116, v12
	v_fma_f32 v13, -v89, v117, v13
	v_fma_f32 v14, -v89, v118, v14
	v_fma_f32 v15, -v89, v119, v15
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v89, v88, v16
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB9_27
; %bb.28:                               ;   in Loop: Header=BB9_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[87:94], v[53:56], v[45:48], v[1:8]
	v_cvt_f16_f32_e32 v40.h, v40
	v_cvt_f16_f32_e32 v40.l, v39
	v_cvt_f16_f32_e32 v39.h, v38
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[87:94], v[53:56], v[41:44], v[87:94]
	v_cvt_f16_f32_e32 v39.l, v37
	v_cvt_f16_f32_e32 v38.h, v36
	v_cvt_f16_f32_e32 v38.l, v35
	v_cvt_f16_f32_e32 v37.h, v34
	v_cvt_f16_f32_e32 v37.l, v33
	v_cvt_f16_f32_e32 v32.h, v32
	v_cvt_f16_f32_e32 v32.l, v31
	v_cvt_f16_f32_e32 v31.h, v30
	v_cvt_f16_f32_e32 v31.l, v29
	v_wmma_f32_16x16x16_f16 v[87:94], v[49:52], v[37:40], v[87:94]
	v_cvt_f16_f32_e32 v30.h, v28
	v_cvt_f16_f32_e32 v30.l, v27
	v_cvt_f16_f32_e32 v29.h, v26
	v_cvt_f16_f32_e32 v29.l, v25
	v_cvt_f16_f32_e32 v28.h, v24
	v_cvt_f16_f32_e32 v28.l, v23
	v_cvt_f16_f32_e32 v27.h, v22
	v_cvt_f16_f32_e32 v27.l, v21
	v_wmma_f32_16x16x16_f16 v[87:94], v[49:52], v[29:32], v[87:94]
	v_cvt_f16_f32_e32 v26.h, v20
	v_cvt_f16_f32_e32 v26.l, v19
	v_cvt_f16_f32_e32 v25.h, v18
	v_cvt_f16_f32_e32 v25.l, v17
	v_cvt_f16_f32_e32 v36.h, v16
	v_cvt_f16_f32_e32 v36.l, v15
	v_cvt_f16_f32_e32 v35.h, v14
	v_cvt_f16_f32_e32 v35.l, v13
	v_wmma_f32_16x16x16_f16 v[87:94], v[49:52], v[25:28], v[87:94]
	v_cvt_f16_f32_e32 v34.h, v12
	v_cvt_f16_f32_e32 v34.l, v11
	v_cvt_f16_f32_e32 v33.h, v10
	v_cvt_f16_f32_e32 v33.l, v9
	v_dual_add_f32 v16, v8, v8 :: v_dual_add_f32 v15, v7, v7
	v_dual_add_f32 v14, v6, v6 :: v_dual_add_f32 v13, v5, v5
	v_add_f32_e32 v12, v4, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[87:94], v[49:52], v[33:36], v[87:94]
	v_dual_add_f32 v11, v3, v3 :: v_dual_add_f32 v10, v2, v2
	v_dual_add_f32 v9, v1, v1 :: v_dual_mul_f32 v24, 0x40400000, v8
	v_dual_add_f32 v17, 0, v87 :: v_dual_mul_f32 v22, 0x40400000, v6
	v_mul_f32_e32 v20, 0x40400000, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[53:56], v[45:48], v[9:16]
	v_mul_f32_e32 v8, 4.0, v8
	v_dual_add_f32 v17, v17, v88 :: v_dual_mul_f32 v18, 0x40400000, v2
	v_mul_f32_e32 v6, 4.0, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[53:56], v[41:44], v[9:16]
	v_dual_mul_f32 v23, 0x40400000, v7 :: v_dual_mul_f32 v2, 4.0, v2
	v_dual_add_f32 v17, v17, v89 :: v_dual_mul_f32 v4, 4.0, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[37:40], v[9:16]
	v_mul_f32_e32 v21, 0x40400000, v5
	v_mul_f32_e32 v19, 0x40400000, v3
	v_add_f32_e32 v17, v17, v90
	v_mul_f32_e32 v7, 4.0, v7
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[29:32], v[9:16]
	v_mul_f32_e32 v5, 4.0, v5
	v_mul_f32_e32 v3, 4.0, v3
	v_add_f32_e32 v17, v17, v91
	s_add_co_i32 s1, s1, 1
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[25:28], v[9:16]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s1, s0
	v_add_f32_e32 v17, v17, v92
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[33:36], v[9:16]
	v_add_f32_e32 v17, v17, v93
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v94
	v_add_f32_e32 v9, v17, v9
	v_mul_f32_e32 v17, 0x40400000, v1
	v_mul_f32_e32 v1, 4.0, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v9, v9, v10
	v_wmma_f32_16x16x16_f16 v[17:24], v[53:56], v[45:48], v[17:24]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[53:56], v[45:48], v[1:8]
	v_fma_mix_f32 v10, v45, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[53:56], v[41:44], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[53:56], v[41:44], v[1:8]
	v_fma_mix_f32 v11, v46, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v9, v9, v12
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[37:40], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[37:40], v[1:8]
	v_fma_mix_f32 v12, v46, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[29:32], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[29:32], v[1:8]
	v_fma_mix_f32 v13, v47, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v9, v9, v14
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[25:28], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[25:28], v[1:8]
	v_fma_mix_f32 v14, v47, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[33:36], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[33:36], v[1:8]
	v_fma_mix_f32 v15, v48, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v9, v16
	v_fma_mix_f32 v16, v48, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v9, v18
	v_fma_mix_f32 v18, v41, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v19
	v_fma_mix_f32 v19, v42, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v9, v20
	v_fma_mix_f32 v20, v42, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v21
	v_fma_mix_f32 v21, v43, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v9, v22
	v_fma_mix_f32 v22, v43, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v17, v9, v23
	v_fma_mix_f32 v9, v45, s2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v23, v44, s2, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v17, v17, v24
	v_fma_mix_f32 v24, v44, s2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[37:40], v[9:16]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v17, v1
	v_fma_mix_f32 v17, v41, s2, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[29:32], v[9:16]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v1, v2
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[37:40], v[17:24]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[25:28], v[9:16]
	v_dual_mov_b32 v2, 0x3b03126f :: v_dual_add_f32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[29:32], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[33:36], v[9:16]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v1, v4
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[25:28], v[17:24]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v48.h, v16
	v_cvt_f16_f32_e32 v48.l, v15
	v_cvt_f16_f32_e32 v47.h, v14
	v_add_f32_e32 v1, v1, v5
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[33:36], v[17:24]
	v_cvt_f16_f32_e32 v47.l, v13
	v_cvt_f16_f32_e32 v46.h, v12
	v_cvt_f16_f32_e32 v46.l, v11
	v_add_f32_e32 v1, v1, v6
	v_cvt_f16_f32_e32 v45.h, v10
	v_cvt_f16_f32_e32 v45.l, v9
	v_cvt_f16_f32_e32 v44.h, v24
	v_cvt_f16_f32_e32 v44.l, v23
	v_add_f32_e32 v1, v1, v7
	v_cvt_f16_f32_e32 v43.h, v22
	v_cvt_f16_f32_e32 v43.l, v21
	v_cvt_f16_f32_e32 v42.h, v20
	v_cvt_f16_f32_e32 v42.l, v19
	v_add_f32_e32 v1, v1, v8
	v_cvt_f16_f32_e32 v41.h, v18
	v_cvt_f16_f32_e32 v41.l, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x3a83126f, v2, vcc_lo
	v_cvt_f16_f32_e32 v53.l, v1
	s_cbranch_scc0 .LBB9_26
.LBB9_29:
	v_fma_mix_f32 v1, v45, 1.0, 0 op_sel_hi:[1,1,0]
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v45, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v46, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v46, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v47, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v47, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v48, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v48, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v41, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v41, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v42, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v42, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v43, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v2, v43, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mov_b32_e32 v1, 0
	v_fma_mix_f32 v2, v44, 1.0, v2 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fma_mix_f32 v2, v44, 1.0, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end9:
	.size	_Z5probeILi2ELb1ELb1EEvPfPKfi, .Lfunc_end9-_Z5probeILi2ELb1ELb1EEvPfPKfi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5probeILi2ELb1ELb1EEvPfPKfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 20
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
		.amdhsa_next_free_vgpr 120
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end9-_Z5probeILi2ELb1ELb1EEvPfPKfi)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z5probeILi2ELb1ELb1EEvPfPKfi,"axG",@progbits,_Z5probeILi2ELb1ELb1EEvPfPKfi,comdat
                                        ; -- End function
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.num_vgpr, 120
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.numbered_sgpr, 8
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5100
; TotalNumSgprs: 10
; NumVgprs: 120
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 120
; Occupancy: 12
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
	.type	__hip_cuid_da9ac63c7cf084e4,@object ; @__hip_cuid_da9ac63c7cf084e4
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_da9ac63c7cf084e4
__hip_cuid_da9ac63c7cf084e4:
	.byte	0                               ; 0x0
	.size	__hip_cuid_da9ac63c7cf084e4, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_da9ac63c7cf084e4
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi8ELb0ELb0EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi8ELb0ELb0EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     170
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi8ELb1ELb0EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi8ELb1ELb0EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     98
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi4ELb0ELb0EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi4ELb0ELb0EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     110
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi4ELb1ELb0EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi4ELb1ELb0EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     98
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi8ELb0ELb1EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi8ELb0ELb1EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     216
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi8ELb1ELb1EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi8ELb1ELb1EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     151
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi4ELb0ELb1EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi4ELb0ELb1EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     168
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi4ELb1ELb1EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi4ELb1ELb1EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     141
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi2ELb0ELb1EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi2ELb0ELb1EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     144
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 20
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z5probeILi2ELb1ELb1EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi2ELb1ELb1EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     120
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
